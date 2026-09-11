local M = {}

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

        taaPreFx:setField("texture", 0, "$backBuffer"); taaPreFx:setField("texture", 1, "#prepass[Depth]")
        taaPreFx:setField("texture", 2, "$backBuffer"); taaPreFx:setField("texture", 3, "#velocitybuffer")
        taaPreFx:setField("target", 0, "#TAA_Result"); taaPreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaPreFx:setField("targetClear", 0, "PFXTargetClear_OnDraw")

        local taaFinalFx = createObject("PostEffect")
        taaFinalFx:setField("shader", 0, "TAA_Final_ShaderData"); taaFinalFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
        taaFinalFx:setField("texture", 0, "#TAA_Result")
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
    useJitter                     = true,
    useR2Jitter                   = true,
    jitterScale                   = 1.0,
    feedbackMin                   = 0.97,
    feedbackMax                   = 0.97,
    shadowMitigation              = 0.0,
    shadowDarknessThreshold       = 0.25,
    shadowBlendStrength           = 0.95,
    varianceGamma                 = 1.25,
    softClip                      = 0.0,
    chromaVarianceMod             = 1.0,
    jitterFlickerPadding          = 0.0,
    directionalVariance           = 1.0,
    jitterFlickerFade             = 0.0,
    depthRejection                = 1.0,
    sharpness                     = 0.25,
    debugMode                     = 0.0,
    useDepthDilation              = 1.0,
    adaptiveVariance              = 0.0,
    lumaVariance                  = 0.0,
    useKDopClipping               = 1.0,
    kdopVarianceClipping          = 1.0,
    useCovarianceClipping         = 1.0,
    colorSpaceOklab               = 1.0,
    jitterAwareVariance           = 1.0,
    velocityAlignedVariance       = 0.0,
    alignmentFeedbackDrop         = 0.80,
    motionBlendDropSpeed          = 1.0,
    roundedAABB                   = 0.0,
    useLanczos3                   = 1.0,
    fireflyClamp                  = 4.0,
    adaptiveVarStart              = 0.5,
    adaptiveVarEnd                = 2.0,
    shadowTemporalMult            = 10.0,
    shadowSpatialMult             = 5.0,
    clipDistanceRejectionEnabled  = 0.0,
    clipDistanceRejectionAmount   = 0.0,
    clipDistanceRejectionMinError = 0.05,
    motionBlendStart              = 1.0,
    shadowVarianceBase            = 0.2,
    collapseRatioMin              = 0.05,
    collapseRatioMax              = 0.35,
    fallbackFXAA                  = 1.0
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
        pre:setShaderConst("$taaMotionBlendStart",        s.motionBlendStart)
        pre:setShaderConst("$taaShadowVarianceBase",      s.shadowShadowVarianceBase and s.shadowShadowVarianceBase or s.shadowVarianceBase)
        pre:setShaderConst("$taaCollapseRatioMin",        s.collapseRatioMin)
        pre:setShaderConst("$taaCollapseRatioMax",        s.collapseRatioMax)
        pre:setShaderConst("$taaFallbackFXAA",            s.fallbackFXAA)
    end

    if fin then
        fin:setShaderConst("$taaSharpness", s.sharpness)
        fin:setShaderConst("$taaDebugMode", s.debugMode)
    end
end

function M.setFrameState(tanX, tanY, yaw, pitch, prevYaw, prevPitch)
    local pre = scenetree.TAA_PreFx
    if not pre then return end

    pre:setShaderConst("$taaTanHalfFovX", tanX)
    pre:setShaderConst("$taaTanHalfFovY", tanY)

    -- PERF: reprojection bases, precomputed here so the pixel shader no longer
    -- runs sincos + builds rotation matrices per pixel. Matches the old shader
    -- math exactly (the shader previously built these from the raw angles):
    --   rot = float3x3( cy, -sy*cp,  sy*sp,
    --                   sy,  cy*cp, -cy*sp,
    --                   0.0,     sp,      cp )
    --   ray(uv) = uv.x*(2tanX,0,0) + (-tanX,1,tanY) - uv.y*(0,0,2tanY)
    --   P = 2tanX*col0, Q = -tanX*col0 + col1 + tanY*col2, R = 2tanY*col2
    -- so that:  rot * ray(uv) = uv.x*P + Q - uv.y*R
    -- and:      rot * ray(uv + vel) = that + vel.x*P - vel.y*R
    local function basis(yawA, pitchA)
        local sy, cy = math.sin(yawA), math.cos(yawA)
        local sp, cp = math.sin(pitchA), math.cos(pitchA)
        local c0x, c0y, c0z = cy, sy, 0.0
        local c1x, c1y, c1z = -sy * cp, cy * cp, sp
        local c2x, c2y, c2z = sy * sp, -cy * sp, cp
        return {
            2 * tanX * c0x, 2 * tanX * c0y, 2 * tanX * c0z,               -- P
            -tanX * c0x + c1x + tanY * c2x,                                -- Q
            -tanX * c0y + c1y + tanY * c2y,
            -tanX * c0z + c1z + tanY * c2z,
            2 * tanY * c2x, 2 * tanY * c2y, 2 * tanY * c2z,               -- R
        }
    end

    local cur = basis(yaw, pitch)
    -- the shader previously negated the prev-jitter angles internally
    local prv = basis(-prevYaw, -prevPitch)

    local function setBasis(tag, b)
        pre:setShaderConst("$taa" .. tag .. "PX", b[1])
        pre:setShaderConst("$taa" .. tag .. "PY", b[2])
        pre:setShaderConst("$taa" .. tag .. "PZ", b[3])
        pre:setShaderConst("$taa" .. tag .. "QX", b[4])
        pre:setShaderConst("$taa" .. tag .. "QY", b[5])
        pre:setShaderConst("$taa" .. tag .. "QZ", b[6])
        pre:setShaderConst("$taa" .. tag .. "RX", b[7])
        pre:setShaderConst("$taa" .. tag .. "RY", b[8])
        pre:setShaderConst("$taa" .. tag .. "RZ", b[9])
    end

    setBasis("Cur", cur)
    setBasis("Prev", prv)
end

function M.setEnabled(enabled)
    local pre = scenetree.TAA_PreFx
    if pre then
        if enabled then pre:enable() else pre:disable() end
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

function M.setupHistory(state)
    local pre = scenetree.TAA_PreFx
    if pre then
        pre:setField("texture", 2, (state == "history") and "#TAA_History" or "$backBuffer")
    end
end

M.build()
M.applySettings()

return M