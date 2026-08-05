-- lua/ge/extensions/taa.lua
local M = {}
local active = true

local jitterIndex, frameIndex = 0, 0
local currentJitterQuat = quat(0, 0, 0, 1)
local prevJitterUV, currJitterUV = {0, 0}, {0, 0}
local prevYaw, prevPitch = 0, 0

local allHookedCameras = {}
local hookedCameraRef = nil
local lastActiveCamName = nil
local lastVehId = nil
local hookVerifyTimer = 0
local renderViewsLoaded = false

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

local function hookCamera(cam)
    if type(cam) ~= "table" or type(cam.update) ~= "function" then return end
    if cam.taa_hooked then return end

    cam.taa_orig_update = cam.update
    cam.taa_hooked = true
    allHookedCameras[cam] = true

    cam.update = function(self, data, ...)
        if self.taa_lastJitter and data and data.res and data.res.rot then
            local temp = quat(data.res.rot)
            data.res.rot:setMul2(self.taa_lastJitter:inversed(), temp)
        end
        
        local ret = self.taa_orig_update(self, data, ...)
        
        -- Apply our active states
        if active and _G.TAA_ChainPostFX and _G.TAA_ChainPostFX.useJitter and data and data.res and data.res.rot then
            self.taa_lastJitter = quat(currentJitterQuat)
            local temp = quat(data.res.rot)
            data.res.rot:setMul2(currentJitterQuat, temp)
        else
            self.taa_lastJitter = nil
        end
        return ret
    end
end

local function unhookCamera(cam)
    if type(cam) ~= "table" or not cam.taa_hooked then return end
    if type(cam.taa_orig_update) == "function" then cam.update = cam.taa_orig_update end
    cam.taa_orig_update = nil; cam.taa_hooked = false; cam.taa_lastJitter = nil
    allHookedCameras[cam] = nil
end

local function unhookAllCameras()
    for cam, _ in pairs(allHookedCameras) do unhookCamera(cam) end
    hookedCameraRef = nil
end

local function findAndHookActiveCamera(forceRehook)
    if not core_camera then return end
    local activeCamName = ""
    if core_camera.getActiveCamName then
        local s, name = pcall(core_camera.getActiveCamName)
        if s then activeCamName = name or "" end
    end

    local veh = getPlayerVehicle(0)
    if veh and veh.getJBeamFilename then
        local s, jbeam = pcall(veh.getJBeamFilename, veh)
        if s and jbeam == "unicycle" then activeCamName = "unicycle" end
    end

    if not forceRehook and activeCamName == lastActiveCamName and hookedCameraRef and hookedCameraRef.taa_hooked then return end
    if hookedCameraRef then unhookCamera(hookedCameraRef); hookedCameraRef = nil end

    if activeCamName == "bigMap" then
        lastActiveCamName = activeCamName
        return
    end

    local foundCam = nil
    if veh then
        local vehId = veh:getID()
        local s1, camData = pcall(core_camera.getCameraDataById, vehId)
        if s1 and type(camData) == "table" then
            if activeCamName ~= "" and camData[activeCamName] and type(camData[activeCamName].update) == "function" then foundCam = camData[activeCamName]
            elseif not foundCam then
                for name, c in pairs(camData) do if type(c) == "table" and type(c.update) == "function" then foundCam = c; break end end
            end
            if not foundCam and type(camData.update) == "function" then foundCam = camData end
        end

        if not foundCam then
            local s2, driverData = pcall(core_camera.getDriverDataById, vehId)
            if s2 and type(driverData) == "table" then
                if activeCamName ~= "" and driverData[activeCamName] and type(driverData[activeCamName].update) == "function" then foundCam = driverData[activeCamName]
                elseif not foundCam then
                    for name, c in pairs(driverData) do if type(c) == "table" and type(c.update) == "function" then foundCam = c; break end end
                end
                if not foundCam and type(driverData.update) == "function" then foundCam = driverData end
            end
        end
    end

    if not foundCam then
        local s3, globalCams = pcall(core_camera.getGlobalCameras)
        if s3 and type(globalCams) == "table" then
            if activeCamName ~= "" and globalCams[activeCamName] and type(globalCams[activeCamName].update) == "function" then foundCam = globalCams[activeCamName]
            elseif not foundCam then
                for name, c in pairs(globalCams) do if type(c) == "table" and type(c.update) == "function" then foundCam = c; break end end
            end
            if not foundCam and type(globalCams.update) == "function" then foundCam = globalCams end
        end
    end

    if foundCam then hookCamera(foundCam); hookedCameraRef = foundCam end
    lastActiveCamName = activeCamName
end

local function init()
    active = true -- FIX 1: Ensure it's active again if hot-loaded

    -- FIX 2: Check both string cases to bypass FS Case-Sensitivity issues in loaded zips
    local taaModule = rerequire("/lua/ge/client/postFx/taa")
    
    local pfx = type(taaModule) == "table" and taaModule or _G.TAA_ChainPostFX
    if pfx then
        pfx.setEnabled(true)
        pfx.setShaderConsts(pfx.DEFAULTS)
    end
    
    findAndHookActiveCamera(true)
end

local function disable()
    active = false -- FIX 1: Kills the preRender loop from re-hooking the camera
    if _G.TAA_ChainPostFX then _G.TAA_ChainPostFX.setEnabled(false) end
    unhookAllCameras()
end

M.onVehicleSwitched = function(oldId, newId, player)
    if active then findAndHookActiveCamera(true) end
end
M.onClientPostStartMission = init
M.onExtensionLoaded = init
M.onModActivated = init

M.onExtensionUnloaded = disable
M.onModDeactivated = disable

M.onUpdate = function()
    -- Hook renderViews to temporarily disable TAA for standard screenshot tools without ghosting artifacts
    if render_renderViews and not renderViewsLoaded then
        renderViewsLoaded = true

        if scenetree.TAA_PreFx then
            local origFunc = render_renderViews.takeScreenshot
            rawset(render_renderViews, "takeScreenshot", function(options, callback)
                local newCallback = function()
                    if callback then callback() end
                    if active and scenetree.TAA_PreFx then scenetree.TAA_PreFx:enable() end
                end
                if scenetree.TAA_PreFx then scenetree.TAA_PreFx:disable() end
                origFunc(options, newCallback)
            end)
        end
    elseif not render_renderViews and renderViewsLoaded then
        renderViewsLoaded = false
    end
end

M.onPreRender = function(dt)
    if not active then return end
    
    local currentCamName = ""
    if core_camera and core_camera.getActiveCamName then
        local s, name = pcall(core_camera.getActiveCamName)
        if s then currentCamName = name or "" end
    end
    
    local currentVeh = getPlayerVehicle(0)
    local currentVehId = currentVeh and currentVeh:getID() or nil
    
    if currentVeh and currentVeh.getJBeamFilename then
        local s, jbeam = pcall(currentVeh.getJBeamFilename, currentVeh)
        if s and jbeam == "unicycle" then currentCamName = "unicycle" end
    end

    local isBigMap = (currentCamName == "bigMap")
    
    if not isBigMap and _G.TAA_ChainPostFX and _G.TAA_ChainPostFX.useJitter and core_camera then
        local jitterPeriod = _G.TAA_ChainPostFX.useR2Jitter and 32 or 16
        jitterIndex = (jitterIndex + 1) % jitterPeriod
        frameIndex = (frameIndex + 1) % 1000
        
        local hx, hy
        if _G.TAA_ChainPostFX.useR2Jitter then 
            hx, hy = r2_sequence(jitterIndex); hx, hy = hx - 0.5, hy - 0.5
        else 
            hx = halton(jitterIndex + 1, 2) - 0.5; hy = halton(jitterIndex + 1, 3) - 0.5 
        end
        
        local width, height = 1920, 1080
        if scenetree.Canvas then 
            local extent = scenetree.Canvas:getExtent()
            if extent then
                local str = type(extent) == "string" and extent or tostring(extent)
                local w, h = str:match("(%d+) (%d+)")
                if w and h then width = math.max(1, tonumber(w)); height = math.max(1, tonumber(h))
                elseif type(extent) == "cdata" or type(extent) == "table" or type(extent) == "userdata" then
                    if extent.x and extent.y then width = math.max(1, tonumber(extent.x)); height = math.max(1, tonumber(extent.y)) end
                end
            end
        end

        local scale = _G.TAA_ChainPostFX.jitterScale
        local pixelJitterX, pixelJitterY = hx * scale, hy * scale
        
        currJitterUV[1] = pixelJitterX / width
        currJitterUV[2] = pixelJitterY / height

        local fovRad = core_camera.getFovRad and core_camera.getFovRad() or math.rad(65)
        if fovRad > 3.14159 then fovRad = math.rad(fovRad) end 
        
        local tanHalfFovY = math.tan(fovRad / 2)
        local tanHalfFovX = tanHalfFovY * (width / height)

        local pitch = -math.atan((pixelJitterY * 2.0 / height) * tanHalfFovY)
        local yaw   = -math.atan((pixelJitterX * 2.0 / width) * tanHalfFovX)

        currentJitterQuat:setFromEuler(pitch, 0, yaw)

        if scenetree.TAA_PreFx then
            scenetree.TAA_PreFx:setShaderConst("$taaTanHalfFovX", tanHalfFovX); scenetree.TAA_PreFx:setShaderConst("$taaTanHalfFovY", tanHalfFovY)
            scenetree.TAA_PreFx:setShaderConst("$taaJitterYaw", yaw); scenetree.TAA_PreFx:setShaderConst("$taaJitterPitch", pitch)
            scenetree.TAA_PreFx:setShaderConst("$taaPrevJitterYaw", prevYaw); scenetree.TAA_PreFx:setShaderConst("$taaPrevJitterPitch", prevPitch)
            scenetree.TAA_PreFx:setShaderConst("$taaRenderSizeX", width); scenetree.TAA_PreFx:setShaderConst("$taaRenderSizeY", height)

            if scenetree.TAA_FinalFx then
                scenetree.TAA_FinalFx:setShaderConst("$taaTanHalfFovX", tanHalfFovX); scenetree.TAA_FinalFx:setShaderConst("$taaTanHalfFovY", tanHalfFovY)
                scenetree.TAA_FinalFx:setShaderConst("$taaJitterYaw", yaw); scenetree.TAA_FinalFx:setShaderConst("$taaJitterPitch", pitch)
                scenetree.TAA_FinalFx:setShaderConst("$taaPrevJitterYaw", prevYaw); scenetree.TAA_FinalFx:setShaderConst("$taaPrevJitterPitch", prevPitch)
            end
        end
        prevYaw, prevPitch = yaw, pitch
    else
        currJitterUV[1], currJitterUV[2] = 0, 0; currentJitterQuat:set(0, 0, 0, 1)
        prevYaw, prevPitch = 0, 0
        if scenetree.TAA_FinalFx then
            scenetree.TAA_FinalFx:setShaderConst("$taaTanHalfFovX", 1.0); scenetree.TAA_FinalFx:setShaderConst("$taaTanHalfFovY", 1.0)
            scenetree.TAA_FinalFx:setShaderConst("$taaJitterYaw", 0.0); scenetree.TAA_FinalFx:setShaderConst("$taaJitterPitch", 0.0)
            scenetree.TAA_FinalFx:setShaderConst("$taaPrevJitterYaw", 0.0); scenetree.TAA_FinalFx:setShaderConst("$taaPrevJitterPitch", 0.0)
        end
        if scenetree.TAA_PreFx then
            scenetree.TAA_PreFx:setShaderConst("$taaTanHalfFovX", 1.0); scenetree.TAA_PreFx:setShaderConst("$taaTanHalfFovY", 1.0)
            scenetree.TAA_PreFx:setShaderConst("$taaJitterYaw", 0.0); scenetree.TAA_PreFx:setShaderConst("$taaJitterPitch", 0.0)
            scenetree.TAA_PreFx:setShaderConst("$taaPrevJitterYaw", 0.0); scenetree.TAA_PreFx:setShaderConst("$taaPrevJitterPitch", 0.0)
        end
    end
    prevJitterUV[1], prevJitterUV[2] = currJitterUV[1], currJitterUV[2]

    if currentCamName ~= lastActiveCamName or currentVehId ~= lastVehId then
        findAndHookActiveCamera(true); lastVehId = currentVehId
    else
        hookVerifyTimer = hookVerifyTimer + dt
        if hookVerifyTimer >= 2.0 then
            local actualCamNeedsHook = false
            if currentVehId and core_camera and core_camera.getCameraDataById then
               local s, camData = pcall(core_camera.getCameraDataById, currentVehId)
               if s and type(camData) == "table" and camData[currentCamName] then
                   if not camData[currentCamName].taa_hooked then actualCamNeedsHook = true end
               end
            end
            if actualCamNeedsHook or not hookedCameraRef or not hookedCameraRef.taa_hooked then findAndHookActiveCamera(true) end
            hookVerifyTimer = 0
        end
    end
end

return M