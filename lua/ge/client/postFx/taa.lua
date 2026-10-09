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
--   prev2Yaw/prev2Pitch are likewise passed NEGATED: RotationFlowUV uses the
--   same (mirrored) convention as basis(), so -angles yields the true t-2
--   content shift, matching the sense of the two bases above.
-- setFrameState no-op detection: when the camera state is bit-identical
-- frame to frame (paused, menus, orbit-static, jitter disabled) the ~30
-- named setShaderConst calls are skipped entirely; under active jitter every
-- value changes and the check costs 8 compares. M.build() resets this so
-- freshly created objects always receive their constants.
local fsLast = {}

function M.setFrameState(tanX, tanY, yaw, pitch, prevYaw, prevPitch, prev2Yaw, prev2Pitch)
    local pre = scenetree.TAA_PreFx
    if not pre then return end

    tanX = math.max(tanX or 1.0, 1e-4)
    tanY = math.max(tanY or 1.0, 1e-4)

    if  tanX == fsLast.tanX and tanY == fsLast.tanY
    and (yaw  or 0.0) == (fsLast.yaw  or 0.0) and (pitch  or 0.0) == (fsLast.pitch  or 0.0)
    and (prevYaw  or 0.0) == (fsLast.prevYaw  or 0.0) and (prevPitch  or 0.0) == (fsLast.prevPitch  or 0.0)
    and (prev2Yaw or 0.0) == (fsLast.prev2Yaw or 0.0) and (prev2Pitch or 0.0) == (fsLast.prev2Pitch or 0.0) then
        return
    end
    fsLast.tanX, fsLast.tanY = tanX, tanY
    fsLast.yaw,   fsLast.pitch   = yaw or 0.0,   pitch or 0.0
    fsLast.prevYaw,  fsLast.prevPitch  = prevYaw or 0.0,  prevPitch or 0.0
    fsLast.prev2Yaw, fsLast.prev2Pitch = prev2Yaw or 0.0, prev2Pitch or 0.0

    pre:setShaderConst("$taaTanHalfFovX", tanX)
    pre:setShaderConst("$taaTanHalfFovY", tanY)
    -- The t-2 jitter rotation pre-evaluated on the CPU: the angles are
    -- per-draw uniforms; the shader's per-pixel sin/cos of them was 4 wasted
    -- transcendentals per pixel. The negation convention is preserved
    -- exactly (sin/cos OF the already-negated angle).
    local p2y = -(prev2Yaw or 0.0)
    local p2p = -(prev2Pitch or 0.0)
    pre:setShaderConst("$taaJitPrev2YawSin",   math.sin(p2y))
    pre:setShaderConst("$taaJitPrev2YawCos",   math.cos(p2y))
    pre:setShaderConst("$taaJitPrev2PitchSin", math.sin(p2p))
    pre:setShaderConst("$taaJitPrev2PitchCos", math.cos(p2p))

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
    -- The resolve writes EVERY texel every frame (fullscreen quad at scale
    -- 1.0, all branches return a color, no discard) -- an OnDraw clear was a
    -- redundant fullscreen 32F clear per frame (~33 MB at 1080p, ~2 GB/s at
    -- 60fps) immediately overwritten. First-frame and post-resize contents
    -- are fully overwritten too.
    taaPreFx:setField("targetClear", 0, "PFXTargetClear_None")

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
    -- Fullscreen write of every pixel (inherited targetScale 1.0, all paths
    -- return a color): an engine-default OnDraw clear would be pure waste.
    -- Harmless no-op if the engine ignores clears on $backBuffer.
    taaFinalFx:setField("targetClear", 0, "PFXTargetClear_None")
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
    fsLast = {}   -- fresh objects: force the next setFrameState to send
    M.setFrameState(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    -- Every setShaderConst this module ever made lived on objects that
    -- destroy() just deleted; a rebuilt chain must re-push the current
    -- settings or it renders on engine-default constants.
    M.applySettings()
end


-- ============================================================================
-- STUDENTIZATION FIT (exact; consumed by the resolve's statistic gate)
-- ----------------------------------------------------------------------------
-- The gate's sampling law is F(3, nu), nu = the record's effective dof
-- (Satterthwaite; taaClip.h.hlsl): nu = 1 / [D/nu0 + varFrac(r)(1-D)] with
-- D = (1-rho)^{2t} and varFrac(r) = rho / (r (2-rho)), r = the LIVE
-- per-draw dof (the spatial covariance's effective rank, measured per pixel
-- by the shader; 3 for diverse RGB content, 1 for gray). The variance
-- inflation restoring the NOMINAL chi-coverage is
--     S(nu) = 3 * F^-1_{3,nu}(p) / chi^2,   p = CDF_chi2_3(chi^2),
-- a function of the slider: S(4) is 1.18 at chi = 1.5 and 2.52 at 2.8 --
-- a fixed constant cannot serve both. The host fits
--     S(nu) = 1 + A/nu + B/nu^2
-- exactly at the trajectory's two extremes. The pin uses r = 4 (the typical
-- recordSampleDof of the 5-tap spatial estimator: nu_inf = 4 * (2-rho)/rho
-- = 49.3). Gray content (r -> 1, true nu_inf = 12.33) reads the fit with
-- ~1% mid-range error -- immaterial next to the F-model's own accuracy, and
-- the previous pin (12.333) remains available as the conservative choice if
-- monochrome ghost trails ever matter more than post-reset tightness.
--
-- v3.9: the gate now consumes the fit at its OWN per-pixel estimation dofs
-- (Satterthwaite over the record + spatial components -- the flats' mu share
-- at nu ~ 6 from the full-set effective count, engaged steps at
-- cusumStepDof ~ 4-6, the transported record at StudentEffectiveDof). All
-- of these sit inside the anchor range [nu0, nu_inf]; the max(nu, 2) guards
-- in the shader can read the fit slightly below nu0 = 4, where the rational
-- form under-inflates relative to the exact S(2) -- acceptable for a guard
-- regime that only engages on degenerate 2-tap subsets.
--
-- MUST MATCH taaClip.h.hlsl: kClipSigmaEmaRate (rho) and
-- kClipStudentPriorDof (nu0).
-- ============================================================================
local STUDENT_RHO   = 0.15                              -- kClipSigmaEmaRate
local STUDENT_NU0   = 4.0                               -- kClipStudentPriorDof
-- v3.8.4: the estimator consumes the FULL 3x3 (9 taps). The typical
-- effective count of the kernel-weighted 3x3 is ~7 -> dof ~6, which this
-- anchor must track (the shader computes the actual per-pixel value from
-- the same weights). If kStdWeights differs materially from the
-- [1 2 1]-outer-product family, remeasure as 1/sum(w^2) - 1 and set this
-- to that value.
local STUDENT_NUINF = 6.0 * (2.0 - STUDENT_RHO) / STUDENT_RHO -- ~74 (r = 6)

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

local function beta15PdfSimpson(hi, n, bm1)
    -- Simpson on t = s^2 over [0, hi] (kills the sqrt singularity; n is a
    -- power of two so the [0,1] step is exact). bm1 = b - 1.
    local h = hi / n
    local sum = 0.0
    for i = 0, n do
        local s = i * h
        local w = (i == 0 or i == n) and 1.0 or (((i % 2) == 1) and 4.0 or 2.0)
        sum = sum + w * 2.0 * s * s * (1.0 - s * s) ^ bm1
    end
    return sum * h / 3.0
end

local function f3nuQuantile(p, nu)
    -- q with P(F(3, nu) <= q) = p, via P(F <= q) = I_u(1.5, nu/2),
    -- u = 3q / (3q + nu). The Simpson NORMALIZER is constant per b -- hoisted
    -- out of the bisection (it was recomputed on every evaluation: ~2x the
    -- work per fit, and every varianceGamma/clipOvershoot tick refits).
    -- Accurate deep into the tail (the coverage fit consumes p = 0.95-ish
    -- and the cold-gate side p = 1e-2): the bisection resolves u to double
    -- precision and the 512-interval Simpson carries ~1e-8 absolute CDF
    -- error, far beneath the tail rates that matter here. Do NOT replace
    -- these quantiles with asymptotic power-law estimates -- they disagree
    -- with the exact values by multiples in the F(3, small-nu) tail.
    local b = nu * 0.5
    local invNorm = 1.0 / beta15PdfSimpson(1.0, 512, b - 1.0)
    local lo, hi = 0.0, 1.0
    for _ = 1, 48 do
        local mid = 0.5 * (lo + hi)
        local cdf = beta15PdfSimpson(math.sqrt(mid), 512, b - 1.0) * invNorm
        if cdf < p then lo = mid else hi = mid end
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

-- ============================================================================
-- JITTER PHASE ENERGY (the exact record seed's phase coefficient)
-- ----------------------------------------------------------------------------
-- E[u_axis^2] of the ACTUAL jitter sequence, in texels^2. The seed models
-- the temporal variance of the innovation's phase bit (-beta * u) as
-- E[u^2] * |beta|^2, and the snap offset u IS the applied jitter in texels
-- (sub-texel jitter: |u| < 0.5, no wrap). Enumerating the exact sequence
-- (the same halton/r2 code, periods, and centering as the camera hook in
-- ge/extensions/taa.lua) makes the coefficient exact for the configured
-- sequence and scale -- for the default R2/32 sequence it lands within a
-- few percent of the uniform 1/12.
-- The shader applies ONE constant to both axes' gradients; the per-axis
-- energies differ only slightly over full periods, so the axis AVERAGE is
-- the least-squares constant (switch to math.max(ex, ey) for a strictly
-- anti-flicker-biased seed).
-- With jitter disabled the phase energy is exactly ZERO -- return a tiny
-- positive sentinel so the shader's "constant provided" gate engages the
-- exact (noise-only) seed instead of falling back to the legacy spatial
-- prior, which would over-seed a jitterless configuration.
-- ============================================================================
local function halton(index, base)
    local f, r, i = 1.0, 0.0, index
    while i > 0 do f = f / base; r = r + f * (i % base); i = math.floor(i / base) end
    return r
end

local function r2_sequence(n)
    local g = 1.32471795724474602596
    local a1 = 1.0 / g; local a2 = 1.0 / (g * g)
    return (0.5 + a1 * n) % 1.0, (0.5 + a2 * n) % 1.0
end

-- ============================================================================
-- JITTER PHASE COVARIANCE (v3.8.4: the FULL 2x2, per-axis and cross)
-- ----------------------------------------------------------------------------
-- E[u_x^2], E[u_y^2], E[u_x u_y] of the ACTUAL jitter sequence, in texels^2.
-- The shader consumes the quadratic form Q(g) = Ex*gx^2 + Ey*gy^2 +
-- 2*Exy*gx*gy -- the variance of the jitter's projection on ANY direction
-- (the sequence covariance is PSD by construction, so Q >= 0 everywhere).
-- Enumerating the exact sequence (the same halton/r2 code, periods and
-- centering as the camera hook) makes the coefficients exact for the
-- configured sequence and scale. The previous single-constant form dropped
-- the cross term and averaged the axes.
-- With jitter disabled the energies are exactly ZERO -- tiny positive
-- sentinels so the shader's "constant provided" gate engages the exact
-- (noise-only) seed.
-- ============================================================================
local function jitterPhaseCovariance()
    local s = M.settings
    local scale = tonumber(s.jitterScale) or 1.0
    if not s.useJitter or scale <= 0.0 then
        return 1e-6, 1e-6, 0.0
    end
    local period = s.useR2Jitter and 32 or 16
    local sumX, sumY, sumXY = 0.0, 0.0, 0.0
    for i = 0, period - 1 do
        local hx, hy
        if s.useR2Jitter then
            hx, hy = r2_sequence(i)
            hx, hy = hx - 0.5, hy - 0.5
        else
            hx = halton(i + 1, 2) - 0.5
            hy = halton(i + 1, 3) - 0.5
        end
        sumX  = sumX  + hx * hx
        sumY  = sumY  + hy * hy
        sumXY = sumXY + hx * hy
    end
    local sc = scale * scale
    return sc * sumX / period, sc * sumY / period, sc * sumXY / period
end

-- ============================================================================
-- THE EXACT SEQUENCE / KERNEL CONSTANTS (v3.2; consumed by the accumulator's
-- variance models and the VarH transport; recomputed on every apply --
-- they track the jitter sequence, its scale, and the feedback rate)
-- ----------------------------------------------------------------------------
--   * flipAcc: the accumulator's response power to the flip pattern, per
--     unit c(1-c) -- exact: the DFT of the actual sequence's bit pattern
--     (per coverage bin) through the accumulator filter
--     H(w) = a/(1-(1-a)e^{-iw}), taken at the WORST coverage bin (the
--     conservative end for the variance models: V >= the exact everywhere,
--     so the models can only be conservative, never hot). The
--     low-discrepancy sequences' bounded discrepancy suppresses the
--     low-frequency lines, so this lands ~2 orders below the marginal
--     Bernoulli charge a/(2-a).
--   * noiseAcc: the EXACT steady-state accumulated-NOISE share, varH/Var(x)
--     at the accumulation's spectral fixed point (v3.9 -- see
--     kernelChainConstants; the old a^2/(1-(1-a)^2 gbar) carried the
--     one-step gain and assumed a white-input read gain of 1).
--   * varhLoss: 1 - D, D = the fixed-point READ gain (kernelChainConstants)
--     -- varH is transported in READ semantics (the Kaiser-resampled
--     history the gate and the CUSUM test), and the history term of the
--     recursion re-reads the smooth accumulated field at the fixed-point
--     gain. Encoded as a LOSS so an unset constant (0) degrades to the
--     legacy undamped transport.
--   * whiteLoss: 1 - w, w = the white-input read gain (E_f[sum k^2])^2 --
--     the fresh current-frame content enters the accumulator white across
--     texels (its spatial correlation is the mu share's business, not the
--     noise chain's), so the fresh-x term of the recursion carries w, not 1.
--     Encoded as a LOSS for the same degradation reason.
--
-- KERNEL CONTRACT (v3.8: the ACTUAL kernels, no longer a proxy): the
-- resolve's history resample is a SEPARABLE per-axis windowed sinc --
-- logical taps n = -2..3, radius 3, Kaiser beta 5.2 (kKaiserBeta6Tap) for
-- the 6-tap tier, n = -1..2, radius 2, beta 3.2 (kKaiserBeta4Tap) for the
-- 4-tap tier -- selected by useKaiser6. MUST MATCH taaResample.h.hlsl; the
-- resulting constants are logged at startup for cross-checking. (The old
-- model -- a radial beta-2.5 disc over |o|^2 <= 5 -- violated this contract
-- by form, beta and radius, and never switched tiers.) The 21-tap fetch
-- pattern's corner skip is ~0.08% of the kernel mass and is not modeled.
-- ============================================================================
local KAISER6_BETA   = 5.2    -- kKaiserBeta6Tap
local KAISER6_RADIUS = 3.0
local KAISER4_BETA   = 3.2    -- kKaiserBeta4Tap
local KAISER4_RADIUS = 2.0

local function besselI0(x)
    local sum, term = 1.0, 1.0
    for i = 1, 40 do
        term = term * (x * x * 0.25) / (i * i)
        sum = sum + term
        if term < 1e-12 * sum then break end
    end
    return sum
end

-- The per-axis kernel evaluated at logical tap n for sub-texel phase f:
-- k(n, f) = sinc(n - f) * KaiserWindow(|n - f| / R), zero outside |n-f| < R.
local function kaiser1DTap(n, phase, beta, radius)
    local d = n - phase
    if math.abs(d) >= radius then return 0.0 end
    local sinc
    if math.abs(d) < 1e-6 then sinc = 1.0
    else sinc = math.sin(math.pi * d) / (math.pi * d) end
    local t = math.min(math.abs(d) / radius, 1.0)
    local win = besselI0(beta * math.sqrt(math.max(1.0 - t * t, 0.0))) / besselI0(beta)
    return sinc * win
end

-- ============================================================================
-- THE RESAMPLE KERNEL'S ACCUMULATION CHAIN (v3.9: the spectral fixed point,
-- not the one-step proxy)
-- ----------------------------------------------------------------------------
-- varH means Var(READ) -- the Kaiser-resampled history that the gate and the
-- CUSUM actually test. Three constants, all measured on the ACTUAL per-tier
-- kernel:
--   * w (whiteGain)  = (E_f[sum_n k(n,f)^2])^2: the read gain against WHITE
--     input. The fresh current-frame content enters the accumulator white
--     across texels, so the recursion's fresh-x term carries w. Parseval:
--     the per-axis w1D is also the frequency-mean of Kbar^2 -- a built-in
--     consistency check on the grid below.
--   * D (readGain)   = Vr/Vs at the accumulation's fixed point. With random
--     per-frame phases (the R2/Halton jitter), the stored SPECTRUM obeys
--         S'(w) = (1-a)^2 Kbar^2(w) S(w) + a^2 Var(x),
--     Kbar^2 = the DFT of the phase-averaged kernel autocorrelation
--     (Wiener-Khinchin), so the fixed point is CLOSED FORM:
--         S* = a^2 / (1 - (1-a)^2 Kbar^2).
--     The 2D fixed point does not separate, but it expands as
--         a^2 sum_k b^k Kx^{2k} Ky^{2k}      (b = (1-a)^2),
--     so the 2D integrals reduce to the 1D moments m_k = mean(Kbar^{2k}):
--         Vs (stored) = a^2 sum_k b^k m_k^2
--         Vr (read)   = a^2 sum_k b^k m_{k+1}^2
--     and D = Vr/Vs. (The old one-step gbar measured a SINGLE resample of
--     white noise; the fixed-point stored field is far more correlated, and
--     the sign of that error is not guessable -- hence the exact spectrum.)
--   * noiseAcc      = Vr / Var(x): the exact steady-state accumulated-noise
--     share.
-- The scalar TRANSPORT recursion in the shader (dampGain*(1-a)^2*varH +
-- freshGain*a^2*Var(x)) is the best scalar approximation of this spectral
-- truth; its own fixed point differs from Vr only by the scalar-collapse
-- error, well beneath the 6-bit transport quantization.
-- ============================================================================
local kernelChainCache = {}
local kernelChainCacheCount = 0

local function kernelChainConstants(beta, radius, a)
    local key = string.format("%.4f:%.2f:%.6f", beta, radius, a)
    local hit = kernelChainCache[key]
    if hit then return hit[1], hit[2], hit[3] end

    local lo = -(math.floor(radius) - 1)
    local hi = math.floor(radius)
    local nPhase = 32

    -- Per-phase kernels, DC-normalized (matches the shader's invSum).
    local ktab = {}
    for p = 1, nPhase do
        local f = (p - 0.5) / nPhase
        ktab[p] = {}
        local sum = 0.0
        for n = lo, hi do
            local k = kaiser1DTap(n, f, beta, radius)
            ktab[p][n] = k
            sum = sum + k
        end
        for n = lo, hi do ktab[p][n] = ktab[p][n] / math.max(sum, 1e-6) end
    end

    -- The phase-averaged kernel autocorrelation (Wiener-Khinchin's source).
    -- Exact truncation: |d| > hi - lo has no overlapping taps, so ac = 0
    -- there by construction.
    local maxD = hi - lo
    local ac = {}
    for d = -maxD, maxD do
        local acc = 0.0
        for p = 1, nPhase do
            local s = 0.0
            for n = lo, hi do
                local n2 = n + d
                if n2 >= lo and n2 <= hi then
                    s = s + ktab[p][n] * ktab[p][n2]
                end
            end
            acc = acc + s
        end
        ac[d] = acc / nPhase
    end

    -- Kbar^2 on a frequency grid (the DFT of ac; real and even).
    local N = 512
    local kbar2 = {}
    for j = 0, N - 1 do
        local omega = 2.0 * math.pi * j / N
        local v = ac[0]
        for d = 1, maxD do v = v + 2.0 * ac[d] * math.cos(omega * d) end
        -- DC-normalized kernels give Kbar^2(0) = 1 exactly and |Khat| <= 1
        -- up to float error; the clamp only guards the fixed-point's
        -- denominator against float noise.
        kbar2[j + 1] = math.min(v, 1.0)
    end

    -- The moment series: m_k = mean(Kbar^{2k}) (non-increasing in k), Vs =
    -- a^2 sum_k b^k m_k^2, Vr = a^2 sum_k b^k m_{k+1}^2. Iterated with a
    -- frozen-moment geometric tail (an over-estimate bounded by m_last^2,
    -- since m continues to decay past the loop).
    local b = (1.0 - a) * (1.0 - a)
    local pow = {}
    for j = 1, N do pow[j] = 1.0 end

    local vsSum, vrSum = 0.0, 0.0
    local bPow = 1.0
    local mLast = 1.0
    for k = 0, 383 do
        local mk = 0.0
        for j = 1, N do mk = mk + pow[j] end
        mk = mk / N
        vsSum = vsSum + bPow * mk * mk

        for j = 1, N do pow[j] = pow[j] * kbar2[j] end
        local mk1 = 0.0
        for j = 1, N do mk1 = mk1 + pow[j] end
        mk1 = mk1 / N
        vrSum = vrSum + bPow * mk1 * mk1

        mLast = mk1
        bPow = bPow * b
        if bPow * mLast * mLast < 1e-10 * math.max(vsSum, 1e-12) then break end
    end
    local tail = bPow * mLast * mLast / math.max(1.0 - b, 1e-6)
    vsSum = vsSum + tail
    vrSum = vrSum + tail

    local vs = a * a * vsSum
    local vr = a * a * vrSum
    -- Parseval: the per-axis white gain IS ac[0] (the frequency-mean of
    -- Kbar^2); the 2D white gain is its square (independent axes/phases).
    local whiteGain = ac[0] * ac[0]
    local readGain  = vr / math.max(vs, 1e-12)
    local noiseAcc  = vr        -- Var(x) = 1 in the series: Vr IS varH/Var(x)

    if kernelChainCacheCount > 32 then
        kernelChainCache = {}
        kernelChainCacheCount = 0
    end
    kernelChainCache[key] = { whiteGain, readGain, noiseAcc }
    kernelChainCacheCount = kernelChainCacheCount + 1
    return whiteGain, readGain, noiseAcc
end

local function accumulatorFilterPower(bits, a)
    -- The accumulator h = sum_k a(1-a)^k x_{t-k}: the mean-square of the
    -- steady-state response to a periodic bit pattern's deviation, exact
    -- (Parseval: mean square = (1/N^2) sum_omega |B(omega)|^2 |H(omega)|^2).
    local n = #bits
    local mean = 0.0
    for i = 1, n do mean = mean + bits[i] end
    mean = mean / n
    local power = 0.0
    for k = 0, n - 1 do
        local omega = 2.0 * math.pi * k / n
        local hMag = a / math.sqrt(1.0 - 2.0 * (1.0 - a) * math.cos(omega) + (1.0 - a) * (1.0 - a))
        local reB, imB = 0.0, 0.0
        for t = 0, n - 1 do
            local ang = -omega * t
            reB = reB + (bits[t + 1] - mean) * math.cos(ang)
            imB = imB + (bits[t + 1] - mean) * math.sin(ang)
        end
        power = power + (reB * reB + imB * imB) * hMag * hMag
    end
    return power / (n * n)
end

local function cusumHostConstants()
    local s = M.settings
    -- v3.3: the accumulator rate varies per pixel (motion feedback). All
    -- rate-dependent constants are computed at the FASTEST configured rate
    -- so the charged responses are conservative upper bounds everywhere: a
    -- wider whitening / flip term can only slow detection, never false-alarm.
    local a = 1.0 - math.min(tonumber(s.feedbackMin) or 0.97,
                              tonumber(s.feedbackMax) or 0.97)
    a = math.min(math.max(a, 0.001), 0.9)

    -- flipAcc: the worst coverage bin of the accumulator's flip-response
    -- gain. The 1D projection is the isotropic stand-in for the per-pixel
    -- edge normal (the R2/Halton projections are all bounded-discrepancy
    -- Kronecker sequences, so the spectra -- and hence the gains -- agree
    -- to the precision that matters here).
    local flipAcc = 0.0
    local scale = tonumber(s.jitterScale) or 1.0
    if s.useJitter and scale > 0.0 then
        local period = s.useR2Jitter and 32 or 16
        local proj = {}
        for i = 0, period - 1 do
            local hx
            if s.useR2Jitter then
                hx = r2_sequence(i)
            else
                hx = halton(i + 1, 2)
            end
            proj[i + 1] = (hx - 0.5) * scale
        end
        table.sort(proj)
        for k = 1, period - 1 do
            local cov = k / period
            if cov >= 0.1 and cov <= 0.9 then
                local thr = 0.5 * (proj[k] + proj[k + 1])
                local bits = {}
                for i = 1, period do bits[i] = (proj[i] > thr) and 1 or 0 end
                local p = accumulatorFilterPower(bits, a)
                flipAcc = math.max(flipAcc, p / (cov * (1.0 - cov)))
            end
        end
    end

    -- v3.9: the VarH chain's gains from the accumulation's spectral fixed
    -- point, per resample tier (kernelChainConstants).
    local use6 = ((tonumber(s.useKaiser6) or 1.0) > 0.5)
    local whiteGain, readGain, noiseAcc = kernelChainConstants(
        use6 and KAISER6_BETA  or KAISER4_BETA,
        use6 and KAISER6_RADIUS or KAISER4_RADIUS,
        a)
    local varhLoss  = math.min(math.max(1.0 - readGain, 0.0), 0.9)
    local whiteLoss = math.min(math.max(1.0 - whiteGain, 0.0), 0.95)

    return flipAcc, noiseAcc, varhLoss, whiteLoss
end

local cusumConstantsLogged = false


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
    -- v3.8: the drift predict step's enable (the lighting machine: the
    -- spatial estimator's common mode applied to the history with
    -- noise-aware shrinkage, capped at the content scale). Requires the
    -- same-layer mask (Depth-Dilated Motion Search).
    driftCompensation             = 1.0,
    -- v3.8.4: the drift tracker's gain cap -- THE speed/noise dial. A
    -- correction applied at gain g injects ~g*Var/2 of noise power into the
    -- history vs the accumulator's own (a/2)Var -- so g = 0.2 tracks a
    -- confirmed shift in ~5 frames with a bounded, brief noise bump.
    driftMaxGain                   = 0.2,
    lumaDriftStrength             = 0.0,
    lumaDriftChromaTol            = 0.1,
    -- chi: the gate's coverage radius (3-dof, PER-CHANNEL normalized -- the
    -- coverages below are actually true now). Below ~2.2 legit sub-texel
    -- content is under-covered. 2.2 = 82%, 2.8 = 95%.
    varianceGamma                 = 2.8,
    -- The posterior-mean soft clip's ENABLE. ON by default since v3.2: with
    -- the empirical spatial-marginal alternative and the sequential prior
    -- the action is the exact posterior mean -- the graded pull is where
    -- the remaining flicker:ghost headroom lives (the hard ellipsoid is the
    -- minimax action and overpays on marginal pixels).
    -- 0 = the pure hard ellipsoid (the A/B baseline).
    softClip                      = 1.0,
    chromaVarianceMod             = 1.0,
    jitterFlickerPadding          = 0.0,
    directionalVariance           = 1.0,
    jitterFlickerFade             = 0.0,
    depthRejection                = 0.05,
    -- v3.8: 5.0 was pure safety margin -- the comparison is exactly
    -- de-jittered and the landing's velocity is the exact stored effective
    -- velocity, so 1.5 is the documented conservative point. Tighten toward
    -- ~0.5 while watching DEBUG MODE 2 (static scene: R dark, B only
    -- wobbles; G only on true reveals) and 7.
    velRejection                  = 1.5,
    velGradientScale              = 1.0,
    crossTestStrength             = 0.35,
    autoSharpen                   = 1.0,
    -- Auto mode: the parity target fraction. 0.85 default: the raw frame's
    -- acutance includes above-Nyquist foldover energy at hard edges, so
    -- strict parity (1.0) over-sharpens them; ~0.85 discounts it. Note this
    -- is an ENERGY fraction (0.5 ~= 71% amplitude restoration).
    sharpness                     = 0.85,
    debugMode                     = 0.0,
    useDepthDilation              = 1.0,
    -- Gates the resolve's stored-field analysis (dilation validation, the
    -- depth/velocity disocclusion tests, landing/pursuit). 0 = the resolve
    -- ignores the stored field and disocclusion relies on color clipping
    -- only. The writer pass itself always runs (it cannot be toggled without
    -- re-registering it outside the resolve's pass chain).
    useMotionField                = 1.0,
    lumaVariance                  = 0.0,
    -- C3 (see taaClip.h.hlsl): the gate's mu-share scoping. 1.0 =
    -- residual-scoped -- the statistically exact mu variance; edge ghosts
    -- clip at ~0.2x local contrast instead of ~0.5x. 0.0 = the conservative
    -- full-E payment. Back off toward 0 (or raise the coverage radius) if
    -- debug mode 11 shows static-content shrinkage (blue) on fine detail.
    clipScopedMu                  = 1.0,
    useHullClipping               = 1.0,
    colorSpaceOklab               = 0.0,
    jitterAwareVariance           = 1.0,
    velocityAlignedVariance       = 0.0,
    alignmentFeedbackDrop         = 0.25,
    motionBlendDropSpeed          = 1.0,
    useKaiser6                    = 1.0,
    historyOvershoot              = 1.0,
    clipOvershoot                 = 0.0,
    clipDistanceRejectionEnabled  = 0.0,
    clipDistanceRejectionAmount   = 0.0,
    clipDistanceRejectionMinError = 0.15,
    motionBlendStart               = 1.0,
    fallbackFXAA                  = 1.0,
    -- The GHOST TRAIL DETECTOR (v3.10: the persistence statistic -- the
    -- EMA-normalized spatial t of the drift estimator's luma common mode;
    -- replaced the matched LLR pair, whose fallback half was statistically
    -- inverted in the sub-gate range). Consumers: the drift-corrector UNLOCK
    -- (confirmed trails track below the per-frame noise gate), the confirmed
    -- blend floor, and the alarm. 0 = off, 1 = soft (default), 2 = telemetry
    -- only (debug mode 11), 3 = soft + hard alarm.
    clipGhostReset                = 1.0
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
        pre:setShaderConst("$taaClipScopedMu",            s.clipScopedMu)
        -- Gates the resolve's acutance metric + transport EWMA (and, with
        -- the clip memory off, the entire historyStateTex fetch) on the
        -- final pass actually consuming the transport.
        pre:setShaderConst("$taaAcutanceActive",
            ((tonumber(s.autoSharpen) or 0) > 0.5 and (tonumber(s.sharpness) or 0) > 0.001) and 1 or 0)
        -- The camera's forward displacement this frame (units of 1/rawDepth);
        -- 0 = the shader measures T_y locally. The host could provide this
        -- from the camera hook (res.pos delta dotted with the previous
        -- forward axis) once the depth convention is confirmed.
        pre:setShaderConst("$taaDepthParallaxStep",       0.0)
        -- The Studentization EXACTLY tracks the slider: the fit consumes
        -- the EFFECTIVE radius (chi * (1 + clipOvershoot)) -- the same
        -- threshold the gate tests. Pin points: nu = 4 and nu_inf(r=4)=49.3.
        local chiEff = (tonumber(s.varianceGamma) or 2.8)
                     * (1.0 + math.max(tonumber(s.clipOvershoot) or 0.0, 0.0))
        local studentA, studentB = studentFitAB(chiEff)
        pre:setShaderConst("$taaStudentA",               studentA)
        pre:setShaderConst("$taaStudentB",               studentB)
        -- v3.8: the winsor tail factors are gone -- the v3.3 content-scale
        -- cap replaced the W(nu) factor and WinsorTailFactor had no call
        -- site (the host fit was dead weight). The drift predict step's
        -- enable takes the slot.
        pre:setShaderConst("$taaDriftCompensation",      s.driftCompensation)
        pre:setShaderConst("$taaDriftMaxGain",          s.driftMaxGain)
        -- The exact-seed phase covariance (v3.8.4: the full quadratic form,
        -- tracking useJitter / useR2Jitter / jitterScale).
        local phEx, phEy, phExy = jitterPhaseCovariance()
        pre:setShaderConst("$taaJitterPhaseEx",  phEx)
        pre:setShaderConst("$taaJitterPhaseEy",  phEy)
        pre:setShaderConst("$taaJitterPhaseExy", phExy)
        -- THE ACCUMULATOR'S HOST CONSTANTS (see cusumHostConstants): the
        -- accumulator's exact flip-response gain (the variance models' step
        -- charge), the exact accumulated-noise share, and the VarH
        -- transport's two read gains (the fixed-point gain for the history
        -- term, the white-input gain for the fresh-x term). They track
        -- useJitter / useR2Jitter / jitterScale / feedbackMax / useKaiser6,
        -- so every apply recomputes them. Both gains are encoded as LOSSes
        -- so an unset constant (0) degrades to the legacy undamped
        -- transport.
        local flipAcc, noiseAcc, varhLoss, whiteLoss = cusumHostConstants()
        pre:setShaderConst("$taaCusumFlipAcc",           flipAcc)
        pre:setShaderConst("$taaCusumNoiseAcc",          noiseAcc)
        pre:setShaderConst("$taaVarhResampleLoss",       varhLoss)
        pre:setShaderConst("$taaVarhWhiteLoss",          whiteLoss)
        if not cusumConstantsLogged then
            cusumConstantsLogged = true
            if log then
                log("I", "TAA", string.format(
                    "Accumulator host constants: flipAcc=%.2e noiseAcc=%.5f varhLoss=%.4f whiteLoss=%.4f (kernel: separable per-axis Kaiser, beta %.1f (6-tap) / %.1f (4-tap) by useKaiser6; VarH gains from the spectral fixed point -- MUST MATCH taaResample.h.hlsl)",
                    flipAcc, noiseAcc, varhLoss, whiteLoss, KAISER6_BETA, KAISER4_BETA))
            end
        end
        -- The persistence detector's mode: 0 off / 1 armed / 2 telemetry /
        -- 3 armed + hard alarm.
        pre:setShaderConst("$taaClipGhostReset",         s.clipGhostReset)
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
        -- Gates the writer's fast path (a 2-fetch write) -- see taaMotion.
        mot:setShaderConst("$taaUseMotionField", s.useMotionField)
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

return M