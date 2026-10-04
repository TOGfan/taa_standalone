local M = {}

-- Every scenetree object this module owns. M.destroy() removes them ALL so
-- M.build() always constructs the chain fresh from the CURRENT files. A
-- persistent ShaderData object left behind by an older mod version can keep
-- serving a stale compiled shader (setShaderConst on constants that no longer
-- exist in the running binary silently no-ops).
local ownedObjects = {
    -- PostEffects first (they reference the shaders/stateblocks below)
    "TAA_StoreMotionFx",
    "TAA_StoreFx",
    "TAA_FinalFx",
    "TAA_PreFx",
    "TAA_Resolve_ShaderData",
    "TAA_Copy_ShaderData",
    "TAA_Motion_ShaderData",
    "TAA_Final_ShaderData",
    "TAA_StateBlock",
    "TAA_Copy_StateBlock",
    "TAA_Motion_StateBlock",
}

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
    else
        -- Force re-setup of samplers on existing stateblock
        setupSamplers(obj)
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


-- BASIS DIRECTION CONTRACT (must match the shaders):
--   This engine applies the camera jitter OPPOSITE to basis()'s rotation
--   convention: basis(-angles) models the ACTUAL rendered frame, basis(+angles)
--   models the mirrored frame. Verified empirically: the original build
--   (current basis +angles / previous basis -angles) produced an exact
--   reprojection -- the -angles previous basis was the TRUE previous frame,
--   and the snap hid the current basis' mirroring for sampling. Both bases
--   are therefore built with NEGATED angles so the shaders receive the true
--   per-frame maps (the shaders compute inverses themselves and are
--   convention-agnostic otherwise).
--
-- prev2Yaw/prev2Pitch are likewise passed NEGATED: RotationFlowUV uses the
--   same (mirrored) convention as basis(), so -angles yields the true t-2
--   content shift, matching the sense of the two bases above.
function M.setFrameState(tanX, tanY, yaw, pitch, prevYaw, prevPitch, prev2Yaw, prev2Pitch)
    local pre = scenetree.TAA_PreFx
    if not pre then return end

    tanX = math.max(tanX or 1.0, 1e-4)
    tanY = math.max(tanY or 1.0, 1e-4)

    pre:setShaderConst("$taaTanHalfFovX", tanX)
    pre:setShaderConst("$taaTanHalfFovY", tanY)
    pre:setShaderConst("$taaJitPrev2Yaw",   -(prev2Yaw or 0.0))
    pre:setShaderConst("$taaJitPrev2Pitch", -(prev2Pitch or 0.0))

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

    -- FIX: both bases NEGATED = the TRUE frames in this engine.
    local cur = basis(-(yaw or 0.0), -(pitch or 0.0))
    local prv = basis(-(prevYaw or 0.0), -(prevPitch or 0.0))

    local function setBasis(effect, tag, b)
        effect:setShaderConst("$taa" .. tag .. "PX", b[1])
        effect:setShaderConst("$taa" .. tag .. "PY", b[2])
        effect:setShaderConst("$taa" .. tag .. "PZ", b[3])
        effect:setShaderConst("$taa" .. tag .. "QX", b[4])
        effect:setShaderConst("$taa" .. tag .. "QY", b[5])
        effect:setShaderConst("$taa" .. tag .. "QZ", b[6])
        effect:setShaderConst("$taa" .. tag .. "RX", b[7])
        effect:setShaderConst("$taa" .. tag .. "RY", b[8])
        effect:setShaderConst("$taa" .. tag .. "RZ", b[9])
    end

    setBasis(pre, "Cur", cur)
    setBasis(pre, "Prev", prv)

    -- The motion-field writer needs the current basis + tan to locate the
    -- jittered sample position (it classifies the current frame exactly like
    -- the resolve does -- same true map, same snap).
    local mot = scenetree.TAA_StoreMotionFx
    if mot then
        mot:setShaderConst("$taaTanHalfFovX", tanX)
        mot:setShaderConst("$taaTanHalfFovY", tanY)
        setBasis(mot, "Cur", cur)
    end
end

function M.destroy()
    for _, name in ipairs(ownedObjects) do
        local obj = scenetree.findObject(name)
        if obj then obj:delete() end
    end
end

function M.build()
    -- Ensure clean recreation so changes to slots, children or shader files
    -- never leave a stale object behind.
    M.destroy()

    getOrCreateStateBlock("TAA_StateBlock", function(sb)
        -- Linear for sceneTex: every main-path fetch sits on an exact texel
        -- center (bilinear returns the exact texel there), while the FXAA
        -- fallback samples at sub-texel positions and needs real filtering to
        -- function as FXAA.
        sb:setField("samplerStates", 0, "SamplerClampLinear")   -- 0: sceneTex
        sb:setField("samplerStates", 1, "SamplerClampPoint")    -- 1: depthTex
        sb:setField("samplerStates", 2, "SamplerClampLinear")   -- 2: historyTex (fused Slepian tap needs it)
        sb:setField("samplerStates", 3, "SamplerClampPoint")    -- 3: velocityTex
        sb:setField("samplerStates", 4, "SamplerClampPoint")    -- 4: historyMotionTex (stored field, exact texel fetches)
    end)

    getOrCreateStateBlock("TAA_Copy_StateBlock", function(sb)
        -- Point sampling doubles as the RCAS integer-load callback in the
        -- final pass (FsrRcasLoadF fetches exact texel centers; edge offsets
        -- clamp to the border texel, matching integer-load semantics).
        sb:setField("samplerStates", 0, "SamplerClampPoint")
    end)

    getOrCreateStateBlock("TAA_Motion_StateBlock", function(sb)
        sb:setField("samplerStates", 0, "SamplerClampPoint")    -- 0: depthTex
        sb:setField("samplerStates", 1, "SamplerClampPoint")    -- 1: velocityTex
        sb:setField("samplerStates", 2, "SamplerClampPoint")    -- 2: taaResultTex (revocation bit in alpha sign)
    end)

    getOrCreateShader("TAA_Resolve_ShaderData", "shaders/common/postFx/taa/taa.fx.hlsl")
    getOrCreateShader("TAA_Copy_ShaderData", "shaders/common/postFx/taa/taaCopy.fx.hlsl")
    getOrCreateShader("TAA_Motion_ShaderData", "shaders/common/postFx/taa/taaMotion.fx.hlsl")
    getOrCreateShader("TAA_Final_ShaderData", "shaders/common/postFx/taa/taaFinal.fx.hlsl")

    -- ------------------------------------------------------------------
    -- Resolve pass (root effect): scene + depth + history + velocity +
    -- stored motion field -> #TAA_Result.
    -- ------------------------------------------------------------------
    local taaPreFx = createObject("PostEffect")
    taaPreFx.isEnabled = false; taaPreFx.allowReflectPass = false
    taaPreFx:setField("renderTime", 0, "PFXBeforeBin"); taaPreFx:setField("renderBin", 0, "EditorBin"); taaPreFx.renderPriority = 0.1
    taaPreFx:setField("shader", 0, "TAA_Resolve_ShaderData"); taaPreFx:setField("stateBlock", 0, "TAA_StateBlock")
    taaPreFx:setField("targetScale", 0, "1.0 1.0")

    taaPreFx:setField("texture", 0, "$backBuffer")
    taaPreFx:setField("texture", 1, "#prepass[Depth]")
    taaPreFx:setField("texture", 2, "$backBuffer")
    taaPreFx:setField("texture", 3, "#velocitybuffer")
    -- Always bound: last frame's layer-resolved motion+depth field. The
    -- writer's inputs (depth, velocity, #TAA_Result) all exist
    -- unconditionally from frame 1, so it always writes this target and the
    -- chain can never deadlock.
    taaPreFx:setField("texture", 4, "#TAA_HistMotion")
    taaPreFx:setField("target", 0, "#TAA_Result")
    taaPreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F")
    taaPreFx:setField("targetClear", 0, "PFXTargetClear_OnDraw")

    -- ------------------------------------------------------------------
    -- Final pass (auto-parity sharpening): #TAA_Result -> $backBuffer. MUST
    -- be a CHILD of the resolve pass: children execute immediately after
    -- their parent, inside the parent's pass, i.e. strictly after
    -- #TAA_Result is written and before anything else touches $backBuffer.
    -- Field set matches the last known-good version exactly: no own
    -- renderTime / renderBin / priority / targetScale -- a child inherits
    -- the parent's scheduling.
    -- ------------------------------------------------------------------
    local taaFinalFx = createObject("PostEffect")
    taaFinalFx:setField("shader", 0, "TAA_Final_ShaderData"); taaFinalFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
    taaFinalFx:setField("texture", 0, "#TAA_Result")
    taaFinalFx:setField("target", 0, "$backBuffer")
    taaFinalFx:registerObject("TAA_FinalFx"); taaPreFx:add(taaFinalFx)

    -- ------------------------------------------------------------------
    -- History copy (child; reads the UN-sharpened resolve output).
    -- ------------------------------------------------------------------
    local taaStoreFx = createObject("PostEffect")
    taaStoreFx:setField("shader", 0, "TAA_Copy_ShaderData"); taaStoreFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
    taaStoreFx:setField("targetScale", 0, "1.0 1.0")
    taaStoreFx:setField("texture", 0, "#TAA_Result"); taaStoreFx:setField("target", 0, "#TAA_History")
    taaStoreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaStoreFx:setField("targetClear", 0, "PFXTargetClear_None")
    taaStoreFx:registerObject("TAA_StoreFx"); taaPreFx:add(taaStoreFx)

    -- ------------------------------------------------------------------
    -- Motion-field writer (child): stores the POST-VALIDATION layer state
    -- (effective velocity/depth + flag) to #TAA_HistMotion. The resolve's
    -- revocation decision travels through the sign of #TAA_Result.a, which
    -- this pass decodes and persists. Must run AFTER the resolve (a child).
    -- ------------------------------------------------------------------
    local taaStoreMotionFx = createObject("PostEffect")
    taaStoreMotionFx:setField("shader", 0, "TAA_Motion_ShaderData"); taaStoreMotionFx:setField("stateBlock", 0, "TAA_Motion_StateBlock")
    taaStoreMotionFx:setField("targetScale", 0, "1.0 1.0")
    taaStoreMotionFx:setField("texture", 0, "#prepass[Depth]")
    taaStoreMotionFx:setField("texture", 1, "#velocitybuffer")
    taaStoreMotionFx:setField("texture", 2, "#TAA_Result")
    taaStoreMotionFx:setField("target", 0, "#TAA_HistMotion")
    taaStoreMotionFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaStoreMotionFx:setField("targetClear", 0, "PFXTargetClear_None")
    taaStoreMotionFx:registerObject("TAA_StoreMotionFx"); taaPreFx:add(taaStoreMotionFx)

    taaPreFx:registerObject("TAA_PreFx")
    M.setFrameState(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

M.settings = {
    useJitter                     = true,
    useR2Jitter                   = true,
    jitterScale                   = 1.0,
    feedbackMin                   = 0.97,
    feedbackMax                   = 0.97,
    lumaDriftStrength             = 0.0,
    lumaDriftChromaTol            = 0.1,
    shadowMitigation              = 0.0,
    shadowDarknessThreshold       = 0.25,
    shadowBlendStrength           = 0.95,
    varianceGamma                 = 1.50,
    softClip                      = 0.0,
    chromaVarianceMod             = 1.0,
    jitterFlickerPadding          = 0.0,
    directionalVariance           = 1.0,
    jitterFlickerFade             = 0.0,
    depthRejection                = 0.05,
    -- The de-jitter of the velocity comparison is verified exact, and the
    -- history-side velocity is the exact stored effective velocity of the
    -- history pixel (not a re-derived approximation), so 1.5 is pure safety
    -- margin. Tighten toward ~0.5 while watching DEBUG MODE 2 (static scene:
    -- R dark, B only wobbles; G only on true reveals) and 7.
    velRejection                  = 5.0,
    velGradientScale              = 1.0,
    crossTestStrength             = 0.35,
    -- Auto-parity sharpening target: 1.0 restores the raw frame's local
    -- sharpness (per-pixel lobe derived in closed form from the measured
    -- acutance ratio), 0 disables. The old manual RCAS strength semantics
    -- are gone; 0.5 roughly matches the previous default look.
    sharpness                     = 1.0,
    debugMode                     = 0.0,
    useDepthDilation              = 1.0,
    lumaVariance                  = 0.0,
    useKDopClipping               = 1.0,
    kdopVarianceClipping          = 0.0,
    useCovarianceClipping         = 0.0,
    colorSpaceOklab               = 1.0,
    jitterAwareVariance           = 1.0,
    velocityAlignedVariance       = 0.0,
    alignmentFeedbackDrop         = 0.25,
    motionBlendDropSpeed          = 1.0,
    useSlepian3                   = 1.0,
    historyOvershoot              = 1.0,
    clipOvershoot                 = 0.0,
    fireflyClamp                  = 4.0,
    shadowTemporalMult            = 10.0,
    shadowSpatialMult             = 5.0,
    clipDistanceRejectionEnabled  = 1.0,
    clipDistanceRejectionAmount   = 0.0,
    clipDistanceRejectionMinError = 0.15,
    motionBlendStart              = 1.0,
    shadowVarianceBase            = 0.2,
    fallbackFXAA                  = 1.0
}

function M.applySettings(inputs)
    if inputs and type(inputs) == "table" then tableMerge(M.settings, inputs) end

    local pre = scenetree.TAA_PreFx
    local mot = scenetree.TAA_StoreMotionFx
    local fin = scenetree.TAA_FinalFx
    local s = M.settings

    -- Keep the shader's feedback clamp well-defined: with lo > hi, HLSL
    -- clamp() silently collapses to hi, quietly disabling the motion /
    -- alignment / shadow feedback reductions.
    if s.feedbackMin and s.feedbackMax and s.feedbackMin > s.feedbackMax then
        s.feedbackMin = s.feedbackMax
    end

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
        pre:setShaderConst("$taaVelRejection",            s.velRejection)
        pre:setShaderConst("$taaVelGradientScale",        s.velGradientScale)
        pre:setShaderConst("$taaCrossTestStrength",       s.crossTestStrength)
        pre:setShaderConst("$taaDebugMode",               s.debugMode)
        pre:setShaderConst("$taaUseDepthDilation",        s.useDepthDilation)
        pre:setShaderConst("$taaLumaVariance",            s.lumaVariance)
        pre:setShaderConst("$taaUseKDopClipping",         s.useKDopClipping)
        pre:setShaderConst("$taaKDopVariance",            s.kdopVarianceClipping)
        pre:setShaderConst("$taaUseCovarianceClipping",   s.useCovarianceClipping)
        pre:setShaderConst("$taaColorSpaceOklab",         s.colorSpaceOklab)
        pre:setShaderConst("$taaJitterAwareVariance",     s.jitterAwareVariance)
        pre:setShaderConst("$taaVelocityAlignedVariance", s.velocityAlignedVariance)
        pre:setShaderConst("$taaAlignmentFeedbackDrop",   s.alignmentFeedbackDrop)
        pre:setShaderConst("$taaMotionBlendDropSpeed",    s.motionBlendDropSpeed)
        pre:setShaderConst("$taaUseSlepian3",             s.useSlepian3)
        pre:setShaderConst("$taaHistoryOvershoot",        s.historyOvershoot)
        pre:setShaderConst("$taaClipOvershoot",           s.clipOvershoot)
        pre:setShaderConst("$taaLumaDriftStrength",       s.lumaDriftStrength)
        pre:setShaderConst("$taaLumaDriftChromaTol",      s.lumaDriftChromaTol)
        pre:setShaderConst("$taaFireflyClamp",            s.fireflyClamp)
        pre:setShaderConst("$taaShadowTemporalMult",      s.shadowTemporalMult)
        pre:setShaderConst("$taaShadowSpatialMult",       s.shadowSpatialMult)
        pre:setShaderConst("$taaClipDistanceRejectionEnabled",  s.clipDistanceRejectionEnabled)
        pre:setShaderConst("$taaClipDistanceRejectionAmount",   s.clipDistanceRejectionAmount)
        pre:setShaderConst("$taaClipDistanceRejectionMinError", s.clipDistanceRejectionMinError)
        pre:setShaderConst("$taaMotionBlendStart",        s.motionBlendStart)
        pre:setShaderConst("$taaShadowVarianceBase",      s.shadowVarianceBase)
        pre:setShaderConst("$taaFallbackFXAA",            s.fallbackFXAA)
    end

    -- The motion-field writer must classify with the SAME parameters as the
    -- resolve, or the stored field disagrees with the resolve's own
    -- classification.
    if mot then
        mot:setShaderConst("$taaUseDepthDilation", s.useDepthDilation)
        mot:setShaderConst("$taaDepthRejection",   s.depthRejection)
    end

    if fin then
        fin:setShaderConst("$taaSharpness", s.sharpness)
        fin:setShaderConst("$taaDebugMode", s.debugMode)
    end
end

function M.setEnabled(enabled)
    -- Parent only: children toggle with their parent.
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
       and scenetree.TAA_StoreMotionFx ~= nil
       and scenetree.TAA_FinalFx ~= nil
end

function M.setupHistory(state)
    local pre = scenetree.TAA_PreFx
    if pre then
        local isHistory = (state == "history")
        pre:setField("texture", 2, isHistory and "#TAA_History" or "$backBuffer")
        -- texture 4 (#TAA_HistMotion) is permanently bound: the stored
        -- motion field is always last frame's data, including warmup frames
        -- (stale data there is safely rejected by the disocclusion tests).
    end
end

M.build()
M.applySettings()

return M