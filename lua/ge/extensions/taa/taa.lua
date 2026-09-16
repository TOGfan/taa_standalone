local M = {}

local MOD_VERSION = "1.2"
local settingsPath = "settings/taa_standalone.json"
local active = true
local pfx = nil

local initDone = false
local retryTimer = 0
local historyWarmupFrames = 0

local jitterIndex = 0
local jitterQuat = quat(0, 0, 0, 1)
local tmpQuat = quat(0, 0, 0, 1)
local prevYaw, prevPitch = 0, 0

local hookedCam = nil
local hookStamp, seenStamp = 0, 0
local rehookTimer = 0

local savedAA = nil

local function clearShaderCache()
    local cacheDirs = { "/temp/shaders", "temp/shaders", "/shaders/cache", "shaders/cache" }
    for _, dir in ipairs(cacheDirs) do
        if FS:directoryExists(dir) then
            local files = FS:findFiles(dir, "*", -1, true, false)
            if files then
                for _, file in ipairs(files) do
                    FS:removeFile(file)
                end
            end
            FS:directoryRemove(dir)
        end
    end
    log("I", "TAA", "Shader cache cleared for mod version " .. MOD_VERSION)
end

local function halton(index, base)
    local f, r, i = 1, 0, index
    while i > 0 do f = f / base; r = r + f * (i % base); i = math.floor(i / base) end
    return r
end

local function r2_sequence(n)
    local g = 1.32471795724474602596
    local a1 = 1.0 / g; local a2 = 1.0 / (g * g)
    return (0.5 + a1 * n) % 1.0, (0.5 + a2 * n) % 1.0
end

local function getGameengineCam()
    if not core_camera or not core_camera.getGlobalCameras then return nil end
    local ok, cams = pcall(core_camera.getGlobalCameras)
    if not ok or type(cams) ~= "table" then return nil end
    local cam = cams.gameengine
    if type(cam) ~= "table" or type(cam.update) ~= "function" then return nil end
    return cam
end

local function frameSize()
    local canvas = scenetree.Canvas
    if canvas then
        if canvas.getWindowClientSizeXY then
            local ok, w, h = pcall(canvas.getWindowClientSizeXY, canvas)
            if ok and w > 0 and h > 0 then return w, h end
        end
        if canvas.getExtent then
            local ok, extent = pcall(canvas.getExtent, canvas)
            if ok and extent and type(extent) ~= "string" and extent.x and extent.y then
                return math.max(1, tonumber(extent.x)), math.max(1, tonumber(extent.y))
            end
        end
    end
    return 1920, 1080
end

local function publishNoJitter()
    if pfx then pfx.setFrameState(1.0, 1.0, 0.0, 0.0, 0.0, 0.0) end
    prevYaw, prevPitch = 0, 0
end

local function applyJitter(data)
    if data.renderView and data.renderView ~= "main" then return end
    hookStamp = hookStamp + 1

    local res = data.res
    local simPaused = (data.dtSim or 1) < 1e-5
    
    local wantJitter = active and pfx and pfx.settings.useJitter and res and res.rot
        and not res.ortho
        and not data.openxrSessionRunning
        and not simPaused
        and not (freeroam_bigMapMode and freeroam_bigMapMode.bigMapActive and freeroam_bigMapMode.bigMapActive())

    if not wantJitter then
        publishNoJitter()
        return
    end

    local w, h = frameSize()
    local fovRad = math.rad(res.fov or 65)
    local tanHalfFovY = math.tan(fovRad * 0.5)
    local tanHalfFovX = tanHalfFovY * (w / h)

    local period = pfx.settings.useR2Jitter and 32 or 16
    jitterIndex = (jitterIndex + 1) % period

    local hx, hy
    if pfx.settings.useR2Jitter then 
        hx, hy = r2_sequence(jitterIndex)
        hx, hy = hx - 0.5, hy - 0.5
    else 
        hx = halton(jitterIndex + 1, 2) - 0.5
        hy = halton(jitterIndex + 1, 3) - 0.5 
    end

    local pxX = hx * pfx.settings.jitterScale
    local pxY = hy * pfx.settings.jitterScale

    local pitch = -math.atan((pxY * 2.0 / h) * tanHalfFovY)
    local yaw   = -math.atan((pxX * 2.0 / w) * tanHalfFovX)

    jitterQuat:setFromEuler(pitch, 0, yaw)
    tmpQuat:set(res.rot)
    res.rot:setMul2(jitterQuat, tmpQuat)

    pfx.setFrameState(tanHalfFovX, tanHalfFovY, yaw, pitch, prevYaw, prevPitch)
    prevYaw, prevPitch = yaw, pitch
end

local function unhookCamera()
    if hookedCam then
        rawset(hookedCam, "update", rawget(hookedCam, "taa_orig_update"))
        rawset(hookedCam, "taa_orig_update", nil)
        hookedCam = nil
    end
end

local function hookCamera()
    local cam = getGameengineCam()
    if not cam then return false end
    if cam == hookedCam and rawget(cam, "taa_orig_update") then return true end
    
    unhookCamera()
    
    local orig = cam.update
    rawset(cam, "taa_orig_update", orig)
    rawset(cam, "update", function(self, ...)
        local data = ...
        if type(data) == "table" then
            pcall(applyJitter, data)
        end
        return orig(self, ...)
    end)
    
    hookedCam = cam
    return true
end

local function hookScreenshot()
    if render_renderViews and not render_renderViews.taa_orig_takeScreenshot then
        render_renderViews.taa_orig_takeScreenshot = render_renderViews.takeScreenshot
        render_renderViews.takeScreenshot = function(options, callback)
            local function newCallback()
                if callback then callback() end
                if active and pfx then 
                    historyWarmupFrames = 0 
                    pfx.setupHistory("reset")
                    pfx.setEnabled(true) 
                end
            end
            if pfx then pfx.setEnabled(false) end
            render_renderViews.taa_orig_takeScreenshot(options, newCallback)
        end
    end
end

local function unhookScreenshot()
    if render_renderViews and render_renderViews.taa_orig_takeScreenshot then
        render_renderViews.takeScreenshot = render_renderViews.taa_orig_takeScreenshot
        render_renderViews.taa_orig_takeScreenshot = nil
    end
end

local function suppressGameAA()
    if savedAA then return end
    local fxaa = scenetree.findObject("FXAA_PostEffect")
    local smaa = scenetree.findObject("SMAA_PostEffect")
    if not fxaa and not smaa then return end
    
    savedAA = {
        fxaa = fxaa and fxaa:isEnabled() or false, 
        smaa = smaa and smaa:isEnabled() or false
    }
    if fxaa then fxaa:disable() end
    if smaa then smaa:disable() end
end

local function restoreGameAA()
    if not savedAA then return end
    local fxaa = scenetree.findObject("FXAA_PostEffect")
    local smaa = scenetree.findObject("SMAA_PostEffect")
    if fxaa and savedAA.fxaa then fxaa:enable() end
    if smaa and savedAA.smaa then smaa:enable() end
    savedAA = nil
end

local function saveState()
    local currentSettings = {}
    if pfx then 
        currentSettings = pfx.settings
    else
        local savedData = jsonReadFile(settingsPath)
        if savedData and savedData.settings then
            currentSettings = savedData.settings
        end
    end
    jsonWriteFile(settingsPath, { version = MOD_VERSION, active = active, settings = currentSettings }, true)
end

local function loadState()
    local savedData = jsonReadFile(settingsPath)
    local needsUpdate = false

    if savedData and type(savedData) == "table" then
        if savedData.version ~= MOD_VERSION then
            clearShaderCache()
            needsUpdate = true
        end
        if savedData.active ~= nil then active = savedData.active end
    else
        clearShaderCache()
        needsUpdate = true
        active = true
    end

    if needsUpdate then
        saveState()
    end
end

local function ensureChain()
    if pfx and pfx.exists() then return true end
    
    local ok, mod = pcall(require, "client/postFx/taa")
    if not ok or type(mod) ~= "table" then
        pfx = nil
        return false
    end

    if mod.build then mod.build() end
    if not mod.exists() then
        pfx = nil
        return false
    end
    pfx = mod
    
    local savedData = jsonReadFile(settingsPath)
    if savedData and type(savedData) == "table" and savedData.settings then
        pfx.applySettings(savedData.settings)
    end
    return true
end

local function start()
    if not ensureChain() then return false end
    
    publishNoJitter()
    jitterIndex = 0
    historyWarmupFrames = 0
    
    if pfx then 
        pfx.setupHistory("reset")
        pfx.setEnabled(true) 
    end
    
    suppressGameAA()
    active = true
    hookCamera()
    hookScreenshot()
    return true
end

local function stop()
    active = false 
    if pfx then 
        pfx.setEnabled(false)
        pfx.setupHistory("reset") 
    end
    
    publishNoJitter()
    unhookCamera()
    unhookScreenshot()
    restoreGameAA()
end

local function init()
    initDone = false
    retryTimer = 0
    loadState()
end

M.onClientPostStartMission = init
M.onExtensionLoaded = init
M.onModActivated = init

M.onModDeactivated = function()
    if not (FS:fileExists("/lua/ge/extensions/taa.lua") or FS:fileExists("/lua/ge/extensions/taa/taa.lua")) then
        stop()
        if pfx and pfx.destroy then pfx.destroy() end
        pfx = nil
        initDone = false
    end
end

M.onExtensionUnloaded = function()
    stop()
    if pfx and pfx.destroy then pfx.destroy() end
    pfx = nil
    initDone = false
end

M.onPreRender = function(dt)
    if not initDone then
        if worldReadyState < 1 then return end
        initDone = true
        
        if active then start() else ensureChain(); stop() end
        return
    end

    if not active then
        retryTimer = retryTimer + dt
        if retryTimer > 5.0 then
            retryTimer = 0
            ensureChain() 
        end
        return
    end

    if pfx and pfx.exists() then
        if historyWarmupFrames < 2 then
            historyWarmupFrames = historyWarmupFrames + 1
            if historyWarmupFrames == 2 then
                pfx.setupHistory("history")
            end
        end
    end
    
    local filterRan = (hookStamp ~= seenStamp)
    seenStamp = hookStamp
    
    if not filterRan then
        publishNoJitter()
        hookCamera() 
        return
    end
    
    rehookTimer = rehookTimer + dt
    if rehookTimer >= 2.0 then
        if hookedCam ~= getGameengineCam() then hookCamera() end
        rehookTimer = 0
    end
end

M.requestUIState = function()
    if not pfx then
        local ok, mod = pcall(require, "client/postFx/taa")
        if ok and type(mod) == "table" then pfx = mod end
    end

    return {
        active = active,
        settings = pfx and pfx.settings or {}
    }
end

M.uiSetEnabled = function(enabled)
    if enabled then start() else stop() end
    saveState()
end

M.uiSetSetting = function(key, value)
    if pfx then
        pfx.applySettings({[key] = value})
        saveState()
    end
end

return M