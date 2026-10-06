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
--   models the mirrored frame. Both bases are therefore built with NEGATED
--   angles so the shaders receive the true per-frame maps (the shaders compute
--   inverses themselves and are convention-agnostic otherwise).
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
        sb:setField("samplerStates", 2, "SamplerClampLinear")   -- 2: historyTex (fused Kaiser tap needs it)
        sb:setField("samplerStates", 3, "SamplerClampPoint")    -- 3: velocityTex
        sb:setField("samplerStates", 4, "SamplerClampPoint")    -- 4: historyMotionTex (stored field, exact texel fetches)
        -- Slot 5: #TAA_History bound a SECOND time, through a POINT sampler:
        -- the bit-packed clip state in its alpha must never see bilinear
        -- weights. A linear sampler is not bit-exact even at a snapped texel
        -- center (the center (k+0.5)/N is not float-representable for
        -- non-power-of-two sizes; the residual weight corrupts the packed
        -- low bits, the bias field first). Slot 2 stays linear for the
        -- fused Kaiser color taps -- one texture, two filterings, two
        -- bindings.
        sb:setField("samplerStates", 5, "SamplerClampPoint")    -- 5: historyStateTex
    end)

    getOrCreateStateBlock("TAA_Copy_StateBlock", function(sb)
        -- Point sampling doubles as the RCAS integer-load callback in the
        -- final pass (FsrRcasLoadF fetches exact texel centers; edge offsets
        -- clamp to the border texel, matching integer-load semantics).
        sb:setField("samplerStates", 0, "SamplerClampPoint")
        -- Slot 1: the mode-5 history view fetch (exact texel read; the
        -- packed debug alpha must never be filtered).
        sb:setField("samplerStates", 1, "SamplerClampPoint")
    end)

    getOrCreateStateBlock("TAA_Motion_StateBlock", function(sb)
        sb:setField("samplerStates", 0, "SamplerClampPoint")    -- 0: depthTex
        sb:setField("samplerStates", 1, "SamplerClampPoint")    -- 1: velocityTex
        sb:setField("samplerStates", 2, "SamplerClampPoint")    -- 2: resolveOutputTex
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
    -- chain can never deadlock. With useMotionField disabled the resolve
    -- stops reading it (the writer pass itself keeps running -- see the
    -- motion-writer note below).
    taaPreFx:setField("texture", 4, "#TAA_HistMotion")
    -- The clip-state read: #TAA_History bound a second time, point-sampled
    -- (slot 2 is the linear Kaiser color binding; the packed alpha needs a
    -- filter-free fetch).
    taaPreFx:setField("texture", 5, "#TAA_History")
    taaPreFx:setField("target", 0, "#TAA_Result")
    taaPreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F")
    taaPreFx:setField("targetClear", 0, "PFXTargetClear_OnDraw")

    -- ------------------------------------------------------------------
    -- Final pass (debug view rendering / auto-parity / manual sharpening):
    -- #TAA_Result -> $backBuffer. MUST be a CHILD of the resolve pass:
    -- children execute immediately after their parent, inside the parent's
    -- pass, i.e. strictly after #TAA_Result is written and before anything
    -- else touches $backBuffer. No own renderTime / renderBin / priority /
    -- targetScale: a child inherits the parent's scheduling.
    -- ------------------------------------------------------------------
    local taaFinalFx = createObject("PostEffect")
    taaFinalFx:setField("shader", 0, "TAA_Final_ShaderData"); taaFinalFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
    taaFinalFx:setField("texture", 0, "#TAA_Result")
    -- Read-only view of the stored history for debug mode 5. This child runs
    -- BEFORE the history-copy child in the pass chain, so it sees exactly the
    -- buffer contents the resolve consumed this frame (never races the copy).
    taaFinalFx:setField("texture", 1, "#TAA_History")
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
    --
    -- This pass is NEVER touched after creation -- no enable()/disable(), no
    -- isEnabled writes, no own scheduling fields. Any lifecycle manipulation
    -- of it re-registers the effect outside the resolve's pass chain on the
    -- Vulkan backend and its named-target bindings then fail ("missing
    -- vulkan resource" spam). useMotionField therefore gates the resolve's
    -- stored-field ANALYSIS only; this pass always runs (a single cheap
    -- fullscreen write, and its output is simply ignored when the resolve
    -- does not read the field).
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


-- ============================================================================
-- STUDENTIZATION FIT (exact; consumed by the resolve's statistic gate)
-- ----------------------------------------------------------------------------
-- The gate's sampling law is F(3, nu), nu = the record's effective dof
-- (Satterthwaite; taaClip.h.hlsl): nu = 4 at age 0, 12.33 converged. The
-- variance inflation restoring the NOMINAL chi-coverage is
--     S(nu) = 3 * F^-1_{3,nu}(p) / chi^2,   p = CDF_chi2_3(chi^2),
-- a function of the slider: S(4) is 1.18 at chi = 1.5 and 2.52 at 2.8 --
-- a fixed constant cannot serve both. The host fits
--     S(nu) = 1 + A/nu + B/nu^2
-- exactly at the trajectory's two extremes: exact at birth and at
-- convergence, <= ~1% across the lived range, at every slider value.
--
-- MUST MATCH taaClip.h.hlsl: kClipSigmaEmaRate (rho) and
-- kClipStudentPriorDof (nu0). nuInf = (2 - rho) / rho.
-- ============================================================================
local STUDENT_RHO   = 0.15                              -- kClipSigmaEmaRate
local STUDENT_NU0   = 4.0                               -- kClipStudentPriorDof
local STUDENT_NUINF = (2.0 - STUDENT_RHO) / STUDENT_RHO -- 12.333

local function erfApprox(x)  -- Abramowitz & Stegun 7.1.26 (|eps| <= 1.5e-7)
    local sign = 1.0
    if x < 0.0 then sign = -1.0; x = -x end
    local t = 1.0 / (1.0 + 0.3275911 * x)
    local poly = 0.254829592 + t * (-0.284496736
                + t * ( 1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    return sign * (1.0 - poly * t * math.exp(-x * x))
end

local function chiSq3Cdf(x)
    if x <= 0.0 then return 0.0 end
    return erfApprox(math.sqrt(x * 0.5))
         - math.sqrt(2.0 * x / math.pi) * math.exp(-x * 0.5)
end

local function beta15Cdf(u, b)
    -- P(Beta(1.5, b) <= u): Simpson on t = s^2 (kills the sqrt
    -- singularity; n is a power of two so the [0,1] step is exact).
    if u <= 0.0 then return 0.0 end
    if u >= 1.0 then return 1.0 end
    local n = 512
    local function integral(hi)
        local h = hi / n
        local sum = 0.0
        for i = 0, n do
            local s = i * h
            local w = (i == 0 or i == n) and 1.0 or (((i % 2) == 1) and 4.0 or 2.0)
            sum = sum + w * 2.0 * s * s * (1.0 - s * s) ^ (b - 1.0)
        end
        return sum * h / 3.0
    end
    return integral(math.sqrt(u)) / integral(1.0)
end

local function f3nuQuantile(p, nu)
    -- q with P(F(3, nu) <= q) = p, via P(F <= q) = I_u(1.5, nu/2),
    -- u = 3q / (3q + nu).
    local b = nu * 0.5
    local lo, hi = 0.0, 1.0
    for _ = 1, 48 do
        local mid = 0.5 * (lo + hi)
        if beta15Cdf(mid, b) < p then lo = mid else hi = mid end
    end
    local u = 0.5 * (lo + hi)
    return nu * u / (3.0 * (1.0 - u))
end

local studentFitCacheChi, studentFitCacheA, studentFitCacheB

local function studentFitAB(chi)
    chi = tonumber(chi) or 2.8
    if chi < 0.25 then chi = 0.25 end
    if studentFitCacheChi == chi then return studentFitCacheA, studentFitCacheB end

    local c2 = chi * chi
    local p = math.min(math.max(chiSq3Cdf(c2), 1e-6), 1.0 - 1e-6)

    local s0 = 3.0 * f3nuQuantile(p, STUDENT_NU0) / c2    -- age 0
    local s1 = 3.0 * f3nuQuantile(p, STUDENT_NUINF) / c2  -- converged

    -- Solve 1 + A/nu + B/nu^2 = S at both anchors (2x2 in 1/nu).
    local i0, i1 = 1.0 / STUDENT_NU0, 1.0 / STUDENT_NUINF
    local r0, r1 = s0 - 1.0, s1 - 1.0
    local det = i0 * i1 * (i1 - i0)
    local A = (r0 * i1 * i1 - r1 * i0 * i0) / det
    local B = (i0 * r1 - i1 * r0) / det

    studentFitCacheChi, studentFitCacheA, studentFitCacheB = chi, A, B
    return A, B
end



-- Defaults: the single source of truth for every setting. M.settings is a
-- working copy that applySettings/user input mutate. When the settings
-- file's version changes, ge/extensions/taa.lua calls M.resetSettings() to
-- rebuild this copy -- merging would keep stale keys from settings that were
-- renamed or removed in newer versions.
M.defaultSettings = {
    useJitter                     = true,
    useR2Jitter                   = true,
    jitterScale                   = 1.0,
    feedbackMin                   = 0.97,
    feedbackMax                   = 0.97,
    lumaDriftStrength             = 0.0,
    lumaDriftChromaTol            = 0.1,
    -- chi: the gate's coverage radius (3-dof). With the gate scale now
    -- per-texel and bias-subtracted, values below ~2.2 under-cover LEGIT
    -- sub-texel content for no ghost benefit (the ghost sits 3-10x outside
    -- the radius regardless). 2.2 = 86% coverage.
    varianceGamma                 = 2.8,
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
    autoSharpen                   = 1.0,
    -- Auto mode: parity target fraction (1.0 = restore the raw frame's local
    -- sharpness). Manual mode (autoSharpen = 0): fixed RCAS strength.
    sharpness                     = 1.0,
    debugMode                     = 0.0,
    useDepthDilation              = 1.0,
    -- Gates the resolve's stored-field analysis (dilation validation, the
    -- depth/velocity disocclusion tests, landing/pursuit). 0 = the resolve
    -- ignores the stored field and disocclusion relies on color clipping
    -- only. The writer pass itself always runs (it cannot be toggled without
    -- re-registering it outside the resolve's pass chain).
    useMotionField                = 1.0,
    lumaVariance                  = 0.0,
    useHullClipping               = 1.0,
    kdopVarianceClipping          = 0.0,
    colorSpaceOklab               = 1.0,
    jitterAwareVariance           = 1.0,
    velocityAlignedVariance       = 0.0,
    alignmentFeedbackDrop         = 0.25,
    motionBlendDropSpeed          = 1.0,
    useKaiser6                    = 1.0,
    historyOvershoot              = 1.0,
    clipOvershoot                 = 0.0,
    fireflyClamp                  = 4.0,
    clipDistanceRejectionEnabled  = 1.0,
    clipDistanceRejectionAmount   = 0.0,
    clipDistanceRejectionMinError = 0.15,
    motionBlendStart               = 1.0,
    fallbackFXAA                  = 1.0
}

local function freshSettingsCopy()
    local copy = {}
    for key, value in pairs(M.defaultSettings) do copy[key] = value end
    return copy
end

M.settings = freshSettingsCopy()

-- Rebuild the working copy from the defaults and push it to the shaders.
-- A tableMerge cannot be used here: it would keep stale keys from whatever
-- the copy previously held.
function M.resetSettings()
    M.settings = freshSettingsCopy()
    M.applySettings()
end

function M.applySettings(inputs)
    if inputs and type(inputs) == "table" then tableMerge(M.settings, inputs) end

    local pre = scenetree.TAA_PreFx
    local mot = scenetree.TAA_StoreMotionFx
    local fin = scenetree.TAA_FinalFx
    local s = M.settings

    -- Keep the shader's feedback clamp well-defined: with lo > hi, HLSL
    -- clamp() silently collapses to hi, quietly disabling the motion /
    -- alignment feedback reductions.
    if s.feedbackMin and s.feedbackMax and s.feedbackMin > s.feedbackMax then
        s.feedbackMin = s.feedbackMax
    end

    if pre then
        pre:setShaderConst("$taaFeedbackMin",             s.feedbackMin)
        pre:setShaderConst("$taaFeedbackMax",             s.feedbackMax)
        pre:setShaderConst("$taaVarianceGamma",           s.varianceGamma)
        pre:setShaderConst("$taaVarianceGamma",           s.varianceGamma)
        -- The Studentization EXACTLY tracks the slider: the fit consumes
        -- the EFFECTIVE radius (chi * (1 + clipOvershoot)) -- the same
        -- threshold the gate tests.
        local chiEff = (tonumber(s.varianceGamma) or 2.8)
                     * (1.0 + math.max(tonumber(s.clipOvershoot) or 0.0, 0.0))
        local studentA, studentB = studentFitAB(chiEff)
        pre:setShaderConst("$taaStudentA",               studentA)
        pre:setShaderConst("$taaStudentB",               studentB)
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
        pre:setShaderConst("$taaUseMotionField",          s.useMotionField)
        pre:setShaderConst("$taaLumaVariance",            s.lumaVariance)
        pre:setShaderConst("$taaUseHullClipping",         s.useHullClipping)
        pre:setShaderConst("$taaColorSpaceOklab",         s.colorSpaceOklab)
        pre:setShaderConst("$taaJitterAwareVariance",     s.jitterAwareVariance)
        pre:setShaderConst("$taaVelocityAlignedVariance", s.velocityAlignedVariance)
        pre:setShaderConst("$taaAlignmentFeedbackDrop",   s.alignmentFeedbackDrop)
        pre:setShaderConst("$taaMotionBlendDropSpeed",    s.motionBlendDropSpeed)
        pre:setShaderConst("$taaUseKaiser6",              s.useKaiser6)
        pre:setShaderConst("$taaHistoryOvershoot",        s.historyOvershoot)
        pre:setShaderConst("$taaClipOvershoot",           s.clipOvershoot)
        pre:setShaderConst("$taaLumaDriftStrength",       s.lumaDriftStrength)
        pre:setShaderConst("$taaLumaDriftChromaTol",      s.lumaDriftChromaTol)
        pre:setShaderConst("$taaFireflyClamp",            s.fireflyClamp)
        pre:setShaderConst("$taaClipDistanceRejectionEnabled",  s.clipDistanceRejectionEnabled)
        pre:setShaderConst("$taaClipDistanceRejectionAmount",   s.clipDistanceRejectionAmount)
        pre:setShaderConst("$taaClipDistanceRejectionMinError", s.clipDistanceRejectionMinError)
        pre:setShaderConst("$taaMotionBlendStart",        s.motionBlendStart)
        pre:setShaderConst("$taaFallbackFXAA",            s.fallbackFXAA)
        -- Debug modes no longer rebind the history source: the resolve's RGB
        -- is always the real blend (views ride the alpha, rendered by
        -- TAA_FinalFx), so the accumulation simply keeps running while a
        -- debug mode is active.
    end

    -- The motion-field writer must classify with the SAME parameters as the
    -- resolve, or the stored field disagrees with the resolve's own
    -- classification. Setting its constants is the ONLY interaction with
    -- this pass -- its lifecycle is never touched (see build()).
    if mot then
        mot:setShaderConst("$taaUseDepthDilation", s.useDepthDilation)
    end

    if fin then
        fin:setShaderConst("$taaAutoSharpen", (tonumber(s.autoSharpen) or 1.0))
        fin:setShaderConst("$taaSharpness", s.sharpness)
        fin:setShaderConst("$taaDebugMode", s.debugMode)
    end
end

function M.setEnabled(enabled)
    -- Parent only: children toggle with their parent. The motion writer is a
    -- child and is never individually toggled (see build()).
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
        -- Warmup bookkeeping ONLY. On a reset the raw current frame stands in
        -- as the history (a fresh / resized canvas can contain anything);
        -- after the warmup the stored buffer takes over. Debug modes never
        -- touch this: they are pure views rendered by TAA_FinalFx from the
        -- alpha payload, and the history stays the live accumulation.
        pre:setField("texture", 2, (state == "history") and "#TAA_History" or "$backBuffer")
        -- texture 4 (#TAA_HistMotion) is permanently bound: the stored
        -- motion field is always last frame's data.
    end
end

M.build()
M.applySettings()

return M