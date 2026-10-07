local M = {}

local MOD_VERSION = "1.14"
local settingsPath = "settings/taa_standalone.json"
local active = true
local pfx = nil

local initDone = false
local retryTimer = 0
local historyWarmupFrames = 0

local jitterIndex = 0
local jitterQuat = quat(0, 0, 0, 1)
local tmpQuat = quat(0, 0, 0, 1)
-- Jitter angle history: prev = frame t-1, prev2 = frame t-2.
-- The shader needs t-2 to cancel the jitter in both the velocity comparison
-- (second difference) and the pursuit transport (first-order combination).
local prevYaw, prevPitch = 0, 0
local prev2Yaw, prev2Pitch = 0, 0

-- Last known projection parameter, so no-jitter frames keep the shader's
-- tangent-space math consistent instead of falling back to a hardcoded FOV.
local lastFov = 65.0

-- Canvas size, tracked so resolution changes reset the history warmup (the
-- history buffer's contents do not survive a resize).
local lastCanvasW, lastCanvasH = 0, 0

-- Settings accepted while the effect chain is unavailable (require failed or
-- mid-rebuild): persisted immediately, applied on the next ensureChain(), and
-- surfaced by requestUIState().
local pendingSettings = nil

-- Set when a settings-file version change reset the settings to defaults;
-- start() persists the full defaults back to the file once the chain is live.
local settingsResetPending = false

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
    -- With zero jitter all screen maps are identities regardless of tan, but
    -- the disocclusion tangent-space math still consumes taaTanHalfFov, so
    -- keep it consistent with the last valid projection / current canvas size.
    local w, h = frameSize()
    local tanY = math.tan(math.rad(lastFov) * 0.5)
    local tanX = tanY * (w / h)
    if pfx then pfx.setFrameState(tanX, tanY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) end
    prevYaw, prevPitch = 0, 0
    prev2Yaw, prev2Pitch = 0, 0
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
    local fovDeg = res.fov or lastFov
    local fovRad = math.rad(fovDeg)
    local tanHalfFovY = math.tan(fovRad * 0.5)
    local tanHalfFovX = tanHalfFovY * (w / h)
    lastFov = fovDeg

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

    pfx.setFrameState(tanHalfFovX, tanHalfFovY, yaw, pitch, prevYaw, prevPitch, prev2Yaw, prev2Pitch)
    prev2Yaw, prev2Pitch = prevYaw, prevPitch
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
            local ok, err = pcall(applyJitter, data)
            if not ok then
                log("E", "TAA", "applyJitter failed: " .. tostring(err))
            end
        end
        return orig(self, ...)
    end)

    hookedCam = cam
    return true
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
    -- Settings staged while the effect chain was unavailable travel with the
    -- save so they survive into the next ensureChain().
    if pendingSettings then
        local merged = {}
        tableMerge(merged, currentSettings)
        tableMerge(merged, pendingSettings)
        currentSettings = merged
    end
    jsonWriteFile(settingsPath, { version = MOD_VERSION, active = active, settings = currentSettings }, true)
end

-- Debounced persistence: settings APPLY immediately but the JSON write
-- trails by 0.4s (a slider drag used to do one disk write per tick).
-- Flushed in onPreRender / stop().
local savePending, saveTimer = false, 0.0
local function saveStateSoon()
    savePending = true
    saveTimer = 0.4
end

local function loadState()
    local savedData = jsonReadFile(settingsPath)

    if savedData and type(savedData) == "table" then
        if savedData.active ~= nil then active = savedData.active end

        if savedData.version ~= MOD_VERSION then
            -- Settings saved by an older version may reference constants or
            -- semantics that no longer exist: reset to the current defaults
            -- instead of migrating. The enabled/disabled state is kept.
            -- The settings key is dropped from the file here (so the next
            -- ensureChain() loads pure defaults); start() persists the full
            -- defaults once the chain is live. A chain already running
            -- mid-session is reset through pfx.resetSettings().
            clearShaderCache()
            pendingSettings = nil
            settingsResetPending = true
            if pfx and pfx.resetSettings then pfx.resetSettings() end
            jsonWriteFile(settingsPath, { version = MOD_VERSION, active = active }, true)
            log("I", "TAA", "Settings reset to defaults (file version " .. tostring(savedData.version) .. " -> " .. MOD_VERSION .. ")")
        end
    else
        clearShaderCache()
        active = true
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
    -- Anything staged while the chain was down was already merged into the
    -- settings file by saveState(); it is live now.
    pendingSettings = nil
    return true
end

local function start()
    if not ensureChain() then return false end

    publishNoJitter()
    jitterIndex = 0
    historyWarmupFrames = 0
    lastCanvasW, lastCanvasH = frameSize()

    if pfx then
        pfx.setupHistory("reset")
        pfx.setEnabled(true)
    end

    suppressGameAA()
    active = true
    hookCamera()

    -- After a version-change reset the settings file has no settings key;
    -- persist the live (default) settings so the file is complete again.
    if settingsResetPending then
        settingsResetPending = false
        saveState()
    end
    return true
end

local function stop()
    active = false
    if savePending then savePending = false; saveState() end
    if pfx then
        pfx.setEnabled(false)
        pfx.setupHistory("reset")
    end

    publishNoJitter()
    unhookCamera()
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
    -- Debounced settings persistence (see saveStateSoon): flushed here so it
    -- runs regardless of the active/init branches below.
    if savePending then
        saveTimer = saveTimer - dt
        if saveTimer <= 0.0 then
            savePending = false
            saveState()
        end
    end

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
        -- Resolution change: the history buffer no longer corresponds to the
        -- new canvas; restart the warmup so accumulation rebuilds cleanly.
        local w, h = frameSize()
        if w ~= lastCanvasW or h ~= lastCanvasH then
            lastCanvasW, lastCanvasH = w, h
            historyWarmupFrames = 0
            pfx.setupHistory("reset")
        end

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

    local settings = pfx and pfx.settings or {}
    if pendingSettings then
        local merged = {}
        tableMerge(merged, settings)
        tableMerge(merged, pendingSettings)
        settings = merged
    end

    return {
        active = active,
        settings = settings
    }
end

-- Bulk settings application (UI presets): one apply pass and one settings-file
-- write instead of a round-trip per key.
M.uiSetSettings = function(settings)
    if type(settings) ~= "table" then return end
    if pfx then
        pfx.applySettings(settings)
        pendingSettings = nil
    else
        -- Effect chain currently unavailable: stage the values so they are
        -- persisted now and applied on the next ensureChain().
        pendingSettings = pendingSettings or {}
        tableMerge(pendingSettings, settings)
    end
    saveStateSoon()
end

M.uiSetSetting = function(key, value)
    M.uiSetSettings({ [key] = value })
end

M.uiSetEnabled = function(enabled)
    if enabled then start() else stop() end
    saveState()
end

return M