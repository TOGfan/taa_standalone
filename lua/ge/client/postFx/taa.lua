-- lua/ge/client/postFx/taa.lua
local M = {}

-- Helper to reduce boilerplate when creating StateBlocks
local function getOrCreateStateBlock(objName, setupSamplers)
    local obj = scenetree.findObject(objName)
    if not obj then
        obj = createObject("GFXStateBlockData")
        obj.zDefined = true; obj.zEnable = false; obj.zWriteEnable = false
        obj.blendDefined = true; obj:setField("blendSrc", 0, "GFXBlendOne"); obj:setField("blendDest", 0, "GFXBlendZero")
        obj.cullDefined = true; obj:setField("cullMode", 0, "GFXCullNone")
        obj.samplersDefined = true
        setupSamplers(obj)
        obj:registerObject(objName)
    end
    return obj
end

-- Helper to reduce boilerplate when creating Shaders
local function getOrCreateShader(objName, path)
    local obj = scenetree.findObject(objName)
    if not obj then
        obj = createObject("ShaderData")
        obj.DXVertexShaderFile = path
        obj.DXPixelShaderFile = path
        obj.pixVersion = 5.0
        obj:registerObject(objName)
    end
    return obj
end

-- Wrapped object creation so it can be re-triggered if the mod is reloaded
function M.build()
    getOrCreateStateBlock("TAA_StateBlock", function(sb)
        sb:setField("samplerStates", 0, "SamplerClampPoint")
        sb:setField("samplerStates", 1, "SamplerClampPoint")
        sb:setField("samplerStates", 2, "SamplerClampLinear")
        sb:setField("samplerStates", 3, "SamplerClampPoint")
    end)

    getOrCreateStateBlock("TAA_Copy_StateBlock", function(sb)
        sb:setField("samplerStates", 0, "SamplerClampLinear")
    end)

    getOrCreateShader("TAA_Resolve_ShaderData", "shaders/common/postFx/taa/taa.fx.hlsl")
    getOrCreateShader("TAA_Copy_ShaderData", "shaders/common/postFx/taa/taaCopy.fx.hlsl")
    getOrCreateShader("TAA_Final_ShaderData", "shaders/common/postFx/taa/taaFinal.fx.hlsl")

    local taaPreFx = scenetree.findObject("TAA_PreFx")
    if not taaPreFx then
        taaPreFx = createObject("PostEffect")
        taaPreFx.isEnabled = false; taaPreFx.allowReflectPass = false
        taaPreFx:setField("renderTime", 0, "PFXBeforeBin"); taaPreFx:setField("renderBin", 0, "EditorBin"); taaPreFx.renderPriority = 0.1
        taaPreFx:setField("shader", 0, "TAA_Resolve_ShaderData"); taaPreFx:setField("stateBlock", 0, "TAA_StateBlock")
        taaPreFx:setField("targetScale", 0, "1.0 1.0")

        -- Start with $backBuffer as history to prevent the Vulkan missing resource error on the first frame!
        taaPreFx:setField("texture", 0, "$backBuffer"); taaPreFx:setField("texture", 1, "#prepass[Depth]")
        taaPreFx:setField("texture", 2, "$backBuffer"); taaPreFx:setField("texture", 3, "#velocitybuffer")
        taaPreFx:setField("target", 0, "#TAA_Result"); taaPreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaPreFx:setField("targetClear", 0, "PFXTargetClear_OnDraw")

        local taaFinalFx = createObject("PostEffect")
        taaFinalFx:setField("shader", 0, "TAA_Final_ShaderData"); taaFinalFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
        taaFinalFx:setField("texture", 0, "#TAA_Result"); taaFinalFx:setField("texture", 1, "#velocitybuffer")
        taaFinalFx:setField("texture", 2, "$backBuffer"); taaFinalFx:setField("texture", 3, "#prepass[Depth]")
        taaFinalFx:setField("target", 0, "$backBuffer")
        taaFinalFx:registerObject("TAA_FinalFx"); taaPreFx:add(taaFinalFx)

        local taaStoreFx = createObject("PostEffect")
        taaStoreFx:setField("shader", 0, "TAA_Copy_ShaderData"); taaStoreFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
        taaStoreFx:setField("targetScale", 0, "1.0 1.0")
        taaStoreFx:setField("texture", 0, "#TAA_Result"); taaStoreFx:setField("target", 0, "#TAA_History")
        taaStoreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaStoreFx:setField("targetClear", 0, "PFXTargetClear_None")
        taaStoreFx:registerObject("TAA_StoreFx"); taaPreFx:add(taaStoreFx)

        taaPreFx:registerObject("TAA_PreFx")
    end
end

M.settings = {
    useJitter                 = true,
    useR2Jitter               = true,
    jitterScale               = 1.0,
    feedbackMin               = 0.99,
    feedbackMax               = 0.99,
    shadowMitigation          = 0.0,
    shadowDarknessThreshold   = 0.25,
    shadowBlendStrength       = 0.95,
    varianceGamma             = 1.25,
    softClip                  = 0.0,
    chromaVarianceMod         = 1.0,
    jitterFlickerPadding      = 0.0,
    directionalVariance       = 1.0,
    jitterFlickerFade         = 0.0,
    depthRejection            = 0.0,
    velDisocclusion           = 0.3,
    sharpness                 = 0.25,
    adaptiveSharp             = 0.0,
    debugMode                 = 0.0,
    useDepthDilation          = 1.0,
    adaptiveVariance          = 0.0,
    lumaVariance              = 0.0,
    useKDopClipping           = 1.0,
    kdopVarianceClipping      = 1.0,
    useCovarianceClipping     = 1.0,
    colorSpaceOklab           = 1.0,
    jitterAwareVariance       = 1.0,
    velocityAlignedVariance   = 0.0,
    alignmentFeedbackDrop     = 0.9,
    alignmentRCASBoost        = 0.0,
    motionBlendDropSpeed      = 1.0,
    bilinearHistoryVel        = 1.0,
    roundedAABB               = 0.0,
    useLanczos3               = 1.0,
    fireflyClamp              = 4.0,
    adaptiveVarStart          = 0.5,
    adaptiveVarEnd            = 2.0,
    shadowTemporalMult        = 10.0,
    shadowSpatialMult         = 5.0,
    clipDistanceRejectionEnabled  = 0.0,
    clipDistanceRejectionAmount   = 0.0,
    clipDistanceRejectionMinError = 0.05,
    depthRejRelStatic         = 0.1,
    depthRejRelMoving         = 0.02,
    depthRejAbs               = 0.01,
    velRejBaseStatic          = 0.5,
    velRejBaseMoving          = 0.05,
    velRejMotionScale         = 0.5,
    motionBlendStart          = 1.0,
    shadowVarianceBase        = 0.2,
    collapseRatioMin          = 0.05,
    collapseRatioMax          = 0.35,
    fallbackFXAA              = 1.0
}

function M.applySettings(inputs)
    if inputs and type(inputs) == "table" then tableMerge(M.settings, inputs) end

    local pre = scenetree.TAA_PreFx
    local fin = scenetree.TAA_FinalFx
    local s = M.settings

    if pre then
        pre:setShaderConst("$taaFeedbackMin",             s.feedbackMin)
        pre:setShaderConst("$taaFeedbackMax",             s.feedbackMax)
        pre:setShaderConst("$taaShadowMitigation",        s.shadowMitigation)
        pre:setShaderConst("$taaShadowDarknessThreshold", s.shadowDarknessThreshold)
        pre:setShaderConst("$taaShadowBlendStrength",     s.shadowBlendStrength)
        pre:setShaderConst("$taaVarianceGamma",           s.varianceGamma)
        pre:setShaderConst("$taaSoftClip",                s.softClip)
        pre:setShaderConst("$taaChromaVarianceMod",       s.chromaVarianceMod)
        pre:setShaderConst("$taaJitterFlickerPadding",    s.jitterFlickerPadding)
        pre:setShaderConst("$taaDirectionalVariance",     s.directionalVariance)
        pre:setShaderConst("$taaJitterFlickerFade",       s.jitterFlickerFade)
        pre:setShaderConst("$taaDepthRejection",          s.depthRejection)
        pre:setShaderConst("$taaVelDisocclusion",         s.velDisocclusion)
        pre:setShaderConst("$taaDebugMode",               s.debugMode)
        pre:setShaderConst("$taaUseDepthDilation",        s.useDepthDilation)
        pre:setShaderConst("$taaAdaptiveVariance",        s.adaptiveVariance)
        pre:setShaderConst("$taaLumaVariance",            s.lumaVariance)
        pre:setShaderConst("$taaUseKDopClipping",         s.useKDopClipping)
        pre:setShaderConst("$taaKDopVariance",            s.kdopVarianceClipping)
        pre:setShaderConst("$taaUseCovarianceClipping",   s.useCovarianceClipping)
        pre:setShaderConst("$taaColorSpaceOklab",         s.colorSpaceOklab)
        pre:setShaderConst("$taaJitterAwareVariance",     s.jitterAwareVariance)
        pre:setShaderConst("$taaVelocityAlignedVariance", s.velocityAlignedVariance)
        pre:setShaderConst("$taaAlignmentFeedbackDrop",   s.alignmentFeedbackDrop)
        pre:setShaderConst("$taaMotionBlendDropSpeed",    s.motionBlendDropSpeed)
        pre:setShaderConst("$taaBilinearHistoryVel",      s.bilinearHistoryVel)
        pre:setShaderConst("$taaRoundedAABB",             s.roundedAABB)
        pre:setShaderConst("$taaUseLanczos3",             s.useLanczos3)
        pre:setShaderConst("$taaFireflyClamp",            s.fireflyClamp)
        pre:setShaderConst("$taaAdaptiveVarStart",        s.adaptiveVarStart)
        pre:setShaderConst("$taaAdaptiveVarEnd",          s.adaptiveVarEnd)
        
        pre:setShaderConst("$taaShadowTemporalMult",      s.shadowTemporalMult)
        pre:setShaderConst("$taaShadowSpatialMult",       s.shadowSpatialMult)
        pre:setShaderConst("$taaClipDistanceRejectionEnabled",  s.clipDistanceRejectionEnabled)
        pre:setShaderConst("$taaClipDistanceRejectionAmount",   s.clipDistanceRejectionAmount)
        pre:setShaderConst("$taaClipDistanceRejectionMinError", s.clipDistanceRejectionMinError)
        pre:setShaderConst("$taaDepthRejRelStatic",       s.depthRejRelStatic)
        pre:setShaderConst("$taaDepthRejRelMoving",       s.depthRejRelMoving)
        pre:setShaderConst("$taaDepthRejAbs",             s.depthRejAbs)
        pre:setShaderConst("$taaVelRejBaseStatic",        s.velRejBaseStatic)
        pre:setShaderConst("$taaVelRejBaseMoving",        s.velRejBaseMoving)
        pre:setShaderConst("$taaVelRejMotionScale",       s.velRejMotionScale)
        pre:setShaderConst("$taaMotionBlendStart",        s.motionBlendStart)
        pre:setShaderConst("$taaShadowVarianceBase",      s.shadowVarianceBase)
        pre:setShaderConst("$taaCollapseRatioMin",        s.collapseRatioMin)
        pre:setShaderConst("$taaCollapseRatioMax",        s.collapseRatioMax)
        pre:setShaderConst("$taaFallbackFXAA",            s.fallbackFXAA)
    end

    if fin then
        fin:setShaderConst("$taaSharpness",             s.sharpness)
        fin:setShaderConst("$taaAdaptiveSharp",         s.adaptiveSharp)
        fin:setShaderConst("$taaDebugMode",             s.debugMode)
        fin:setShaderConst("$taaUseDepthDilation",      s.useDepthDilation)
        fin:setShaderConst("$taaShadowDarknessThreshold", s.shadowDarknessThreshold)
        fin:setShaderConst("$taaDepthRejection",        s.depthRejection)
        fin:setShaderConst("$taaVelDisocclusion",       s.velDisocclusion)
        fin:setShaderConst("$taaAlignmentRCASBoost",    s.alignmentRCASBoost)
        fin:setShaderConst("$taaShadowTemporalMult",    s.shadowTemporalMult)
        fin:setShaderConst("$taaShadowSpatialMult",     s.shadowSpatialMult)
    end
end

function M.setFrameState(tanX, tanY, yaw, pitch, prevYaw, prevPitch)
    local pre = scenetree.TAA_PreFx
    local fin = scenetree.TAA_FinalFx

    if pre then
        pre:setShaderConst("$taaTanHalfFovX", tanX)
        pre:setShaderConst("$taaTanHalfFovY", tanY)
        pre:setShaderConst("$taaJitterYaw", yaw)
        pre:setShaderConst("$taaJitterPitch", pitch)
        pre:setShaderConst("$taaPrevJitterYaw", prevYaw)
        pre:setShaderConst("$taaPrevJitterPitch", prevPitch)
    end
    if fin then
        fin:setShaderConst("$taaTanHalfFovX", tanX)
        fin:setShaderConst("$taaTanHalfFovY", tanY)
        fin:setShaderConst("$taaJitterYaw", yaw)
        fin:setShaderConst("$taaJitterPitch", pitch)
        fin:setShaderConst("$taaPrevJitterYaw", prevYaw)
        fin:setShaderConst("$taaPrevJitterPitch", prevPitch)
    end
end

function M.setEnabled(enabled)
    local pre = scenetree.TAA_PreFx
    if pre then
        if enabled then 
            pre:enable() 
        else 
            pre:disable()
        end
    end
end

function M.setPriority(priority)
    local pre = scenetree.TAA_PreFx
    if pre then
        local prevEnabled = pre:isEnabled()
        pre:disable()
        pre.renderPriority = priority
        if prevEnabled then pre:enable() end
    end
end

function M.exists()
    return scenetree.TAA_PreFx ~= nil
end

function M.destroy()
    if scenetree.TAA_StoreFx then scenetree.TAA_StoreFx:delete() end
    if scenetree.TAA_FinalFx then scenetree.TAA_FinalFx:delete() end
    if scenetree.TAA_PreFx then scenetree.TAA_PreFx:delete() end
end

-- Safely switches out the history target at runtime
function M.setupHistory(state)
    local pre = scenetree.TAA_PreFx
    local fin = scenetree.TAA_FinalFx
    if pre and fin then
        if state == "history" then
            pre:setField("texture", 2, "#TAA_History")
            fin:setField("texture", 2, "#TAA_History")
        else
            pre:setField("texture", 2, "$backBuffer")
            fin:setField("texture", 2, "$backBuffer")
        end
    end
end

-- Apply defaults on initial load
M.build()
M.applySettings()

return M