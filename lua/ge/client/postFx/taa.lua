local M = {}

-- Every scenetree object this module owns. M.destroy() removes them ALL so
-- M.build() always constructs the chain fresh from the CURRENT files. A
-- persistent ShaderData object left behind by an older mod version can keep
-- serving a stale compiled shader (setShaderConst on constants that no
-- longer exist in the running binary silently no-ops).
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

-- ALWAYS recreate, and stamp the rev as a DEFINE. The define is the
-- load-bearing piece: every compiled-shader cache in this engine family
-- (the in-memory manager, the disk caches, the DX12 PSO cache) keys on
-- path + MACRO SET, so a changed define forces a fresh compile of the
-- current file even when every path and object name is unchanged. The
-- delete-then-create covers the SimObject's own compiled-binary holder.
-- An unreferenced macro changes no cbuffer and no codegen -- key only.
-- (One-time verification that the `defines` field is live on this engine
-- build: point a probe ShaderData at the resolve with defines =
-- "kEpsilon=### invalid ###" -- a compile error in the log proves it
-- reaches the preprocessor.)
local function getOrCreateShader(objName, path, revDefine)
    local obj = scenetree.findObject(objName)
    if obj then obj:delete() end
    obj = createObject("ShaderData")
    obj.DXVertexShaderFile = path
    obj.DXPixelShaderFile = path
    obj.pixVersion = 5.0
    if revDefine then obj.defines = revDefine end
    obj:registerObject(objName)
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
--
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
    -- per-draw uniforms; the shader would otherwise pay 4 transcendentals
    -- per pixel. The negation convention is preserved exactly (sin/cos OF
    -- the already-negated angle).
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

function M.build(shaderKey)
    -- Clean recreation so changes to slots, children or shader files never
    -- leave a stale object behind.
    M.destroy()

    getOrCreateStateBlock("TAA_StateBlock", function(sb)
        -- Linear for sceneTex: every main-path fetch sits on an exact texel
        -- center (bilinear returns the exact texel there), while the FXAA
        -- fallback samples at sub-texel positions and needs real filtering.
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

    local revDefine = "TAA_REV_" .. tostring(shaderKey or "unknown"):gsub("%.", "_")
    getOrCreateShader("TAA_Resolve_ShaderData", "shaders/common/postFx/taa/taa.fx.hlsl", revDefine)
    getOrCreateShader("TAA_Copy_ShaderData", "shaders/common/postFx/taa/taaCopy.fx.hlsl", revDefine)
    getOrCreateShader("TAA_Motion_ShaderData", "shaders/common/postFx/taa/taaMotion.fx.hlsl", revDefine)
    getOrCreateShader("TAA_Final_ShaderData", "shaders/common/postFx/taa/taaFinal.fx.hlsl", revDefine)

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
    -- filter-free fetch). Frame 0 / post-resize reads whatever the engine
    -- left in the target (typically zero): alpha 0 decodes as tag 0 = a
    -- COLD record, so those pixels run the cold gate for one frame and the
    -- state self-corrects from the next store on.
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
    -- Final pass (debug view rendering / manual sharpening):
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
-- D = (1-rho)^{2t} and varFrac(r) = rho / (r (2-rho)), r = the ingest's
-- chi^2 dof (= 1.0: the record ingests the pixel's own squared innovation,
-- a single squared Gaussian). The converged limit is nu_inf = (2-rho)/rho
-- = 12.33. The variance inflation restoring the NOMINAL chi^2 coverage is
--     S(nu) = 3 * F^-1_{3,nu}(p) / chi^2,   p = CDF_chi2_3(chi^2),
-- a function of the coverage slider (S(4) is 1.18 at chi = 1.5 and 2.52 at
-- 2.8 -- a fixed constant cannot serve both). The host fits a PADE (2,1)
-- RATIONAL in x = 1/nu,
--     S(nu) = (1 + A*x + B*x^2) / (1 + C*x),
-- by weighted least squares over a dense anchor set -- the relation is
-- LINEAR in (A, B, C) after multiplying through (A*x + B*x^2 - S*C*x =
-- S - 1), so the fit is a closed 3x3 normal-equations solve. The anchor
-- weights concentrate on the operative points (the Satterthwaite clamp
-- floor nu = 2, the young record nu0, the converged nu_inf) and densify
-- the steep low-nu end, where S rises ~3x between nu = 4 and nu = 2.
-- Eight quantiles per fit; one cached fit per distinct chi (a
-- coverage-slider drag stays interactive).
--
-- The F(3,nu) law formally assumes one shared chi^2 denominator across the
-- three channels; the per-channel estimates share the same taps, so the
-- law is exact for gray/near-gray content and mildly approximate for
-- decorrelated RGB (second-order next to the dof modeling).
--
-- MUST MATCH taaClip.h.hlsl: kClipSigmaEmaRate (rho), kClipStudentPriorDof
-- (nu0), kRecordIngestDof, and the CONSUMPTION FORM (the Pade (2,1)
-- rational in 1/nu: A, B the numerator's x / x^2 coefficients, C the
-- denominator's x coefficient).
-- ============================================================================
local STUDENT_RHO   = 0.15                              -- kClipSigmaEmaRate
local STUDENT_NU0   = 4.0                               -- kClipStudentPriorDof
local STUDENT_NUINF = 1.0 * (2.0 - STUDENT_RHO) / STUDENT_RHO -- 12.33 (r = kRecordIngestDof = 1)

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
    -- out of the bisection. 128 intervals carry ~2e-7 absolute CDF error and
    -- 32 bisection steps resolve u to 2^-32: both far beneath the tail rates
    -- that matter here. Do NOT replace these quantiles with asymptotic
    -- power-law estimates -- they disagree with the exact values by
    -- multiples in the F(3, small-nu) tail.
    local b = nu * 0.5
    local invNorm = 1.0 / beta15PdfSimpson(1.0, 128, b - 1.0)
    local lo, hi = 0.0, 1.0
    for _ = 1, 32 do
        local mid = 0.5 * (lo + hi)
        local cdf = beta15PdfSimpson(math.sqrt(mid), 128, b - 1.0) * invNorm
        if cdf < p then lo = mid else hi = mid end
    end
    local u = 0.5 * (lo + hi)
    return nu * u / (3.0 * (1.0 - u))
end

-- Small exact-key table cache: a coverage-slider drag emits a stream of
-- distinct chis (one cheap refit per tick), and back-and-forth drags,
-- preset applications and rebuild() re-hit instead of refitting.
local studentFitCache = {}
local studentFitCacheCount = 0

local function studentFitABC(chi)
    chi = tonumber(chi) or 2.8
    if chi < 0.25 then chi = 0.25 end
    local hit = studentFitCache[chi]
    if hit then return hit[1], hit[2], hit[3] end

    local c2 = chi * chi
    local p = math.min(math.max(chiSq3Cdf(c2), 1e-6), 1.0 - 1e-6)

    -- Weighted least squares of S(x) = (1 + A*x + B*x^2) / (1 + C*x) over
    -- the anchor set. Linear in (A, B, C) after multiplying through
    -- (A*x + B*x^2 - S*C*x = S - 1): a closed 3x3 normal-equations solve
    -- (Gaussian elimination + back-substitution below).
    local anchors = {
        { 2.0,           8.0 },   -- the Satterthwaite clamp floor (pile-up)
        { 2.5,           2.0 },
        { 3.25,          2.0 },
        { STUDENT_NU0,   5.0 },   -- the young record (age 0)
        { 5.5,           1.0 },
        { 8.0,           1.0 },
        { STUDENT_NUINF, 5.0 },   -- the converged record
        { 18.0,          1.0 },
    }

    local sVals = {}
    local n = { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } }
    for idx, an in ipairs(anchors) do
        local nu, w = an[1], an[2]
        local x = 1.0 / nu
        local S = 3.0 * f3nuQuantile(p, nu) / c2
        sVals[idx] = S
        local c1, c2c, c3c = x, x * x, -S * x
        local rhs = S - 1.0
        n[1][1] = n[1][1] + w * c1 * c1; n[1][2] = n[1][2] + w * c1 * c2c; n[1][3] = n[1][3] + w * c1 * c3c
        n[2][2] = n[2][2] + w * c2c * c2c; n[2][3] = n[2][3] + w * c2c * c3c
        n[3][3] = n[3][3] + w * c3c * c3c
        n[1][4] = n[1][4] + w * c1 * rhs
        n[2][4] = n[2][4] + w * c2c * rhs
        n[3][4] = n[3][4] + w * c3c * rhs
    end
    n[2][1] = n[1][2]; n[3][1] = n[1][3]; n[3][2] = n[2][3]

    for col = 1, 3 do
        for row = col + 1, 3 do
            local f = n[row][col] / n[col][col]
            for k = col, 4 do n[row][k] = n[row][k] - f * n[col][k] end
        end
    end
    local C = n[3][4] / n[3][3]
    local B = (n[2][4] - n[2][3] * C) / n[2][2]
    local A = (n[1][4] - n[1][3] * C - n[1][2] * B) / n[1][1]

    -- Denominator safety: 1 + C*x must stay comfortably positive over the
    -- shader's whole nu range (x = 1/nu in [0, 0.5]; nu >= 2 everywhere).
    -- A sane fit never approaches the pole; the clamp plus the fixed-C 2x2
    -- refit guard the pathological case only.
    if C > 1.5 or C < -6.0 then
        C = math.min(math.max(C, -6.0), 1.5)
        local m11, m12, m22 = 0.0, 0.0, 0.0
        local r1, r2 = 0.0, 0.0
        for idx, an in ipairs(anchors) do
            local w = an[2]
            local x = 1.0 / an[1]
            local S = sVals[idx]
            local rhs = (S - 1.0) + S * C * x
            m11 = m11 + w * x * x
            m12 = m12 + w * x * x * x
            m22 = m22 + w * x * x * x * x
            r1 = r1 + w * x * rhs
            r2 = r2 + w * x * x * rhs
        end
        local det = m11 * m22 - m12 * m12
        if math.abs(det) > 1e-14 then
            B = (m11 * r2 - m12 * r1) / det
            A = (m22 * r1 - m12 * r2) / det
        end
    end

    if studentFitCacheCount >= 32 then
        studentFitCache = {}
        studentFitCacheCount = 0
    end
    studentFitCache[chi] = { A, B, C }
    studentFitCacheCount = studentFitCacheCount + 1
    return A, B, C
end

-- ============================================================================
-- THE ARMING CAP FIT (exact, nu-aware)
-- ----------------------------------------------------------------------------
-- The record-vs-prior arming cap is the honest one-sided bound for a nu-dof
-- variance estimate against its prior:
--     cap(nu) = chi^2_nu(0.999) / nu
-- fit as 1 + A/sqrt(nu) + B/nu, solved exactly at the record's dof
-- trajectory endpoints (nu0 = 4 at age 0, nu_inf = 12.33 converged):
--     chi^2_4(0.999)      = 18.467 (closed-form CDF, bisection) -> 4.6168
--     chi^2_12.33(0.999) ~ 33.4   (Wilson-Hilferty)            -> 2.71
-- The rational tracks the exact quantiles to better than 1% across
-- [4, 12.33]. The cap anchors at CONTENT scale by design: it is the
-- stale-high guard, and a record-anchored bound is vacuous.
-- MUST MATCH taaClip.h.hlsl's consumption (taaArmingA / taaArmingB).
-- ============================================================================
local function normQuantile(p)
    -- Acklam's rational approximation to the inverse standard normal CDF
    -- (|eps| < 1.15e-9).
    local a = { -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
                 1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 }
    local b = { -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
                 6.680131188771972e+01, -1.328068155288572e+01 }
    local c = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 }
    local d = {  7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                3.754408661907416e+00 }
    local pLow, pHigh = 0.02425, 1.0 - 0.02425
    if p < pLow then
        local q = math.sqrt(-2.0 * math.log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])
             / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= pHigh then
        local q = p - 0.5
        local r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q
             / (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        local q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])
              / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

local function chiSq4Quantile(p)
    -- exact: the nu=4 CDF is 1 - e^{-x/2}(1 + x/2) -- closed-form bisection.
    local lo, hi = 0.0, 64.0
    for _ = 1, 48 do
        local mid = 0.5 * (lo + hi)
        local t = 0.5 * mid
        if (1.0 - math.exp(-t) * (1.0 + t)) < p then lo = mid else hi = mid end
    end
    return 0.5 * (lo + hi)
end

local function studentArmingFit()
    local nu0, nuInf = STUDENT_NU0, STUDENT_NUINF
    local c0 = chiSq4Quantile(0.999) / nu0
    local z = normQuantile(0.999)
    local c1 = (nuInf + z * math.sqrt(2.0 * nuInf) + (2.0 / 3.0) * (z * z - 1.0)) / nuInf
    -- solve 1 + A/sqrt(nu) + B/nu = cap at both anchors
    local i0, i1 = 1.0 / math.sqrt(nu0), 1.0 / math.sqrt(nuInf)
    local r0, r1 = c0 - 1.0, c1 - 1.0
    local det = i0 * i1 * i1 - i1 * i0 * i0
    local A = (r0 * i1 * i1 - r1 * i0 * i0) / det
    local B = (i0 * r1 - i1 * r0) / det
    return A, B
end

local ARMING_A, ARMING_B = studentArmingFit()

-- ============================================================================
-- JITTER PHASE COVARIANCE (the full 2x2, per-axis and cross)
-- ----------------------------------------------------------------------------
-- Var[u_x], Var[u_y], Cov[u_x, u_y] of the ACTUAL jitter sequence, in
-- texels^2 -- CENTERED on the sequence mean. The innovation's phase bit is
-- -g * (u_t - ubar): the current jitter measured against the ACCUMULATED
-- phase average, and the accumulator's ~1/a-frame memory (~33 frames at
-- default feedback ~ the sequence period) makes ubar the sequence mean.
-- The shader consumes the quadratic form Q(g) = Ex*gx^2 + Ey*gy^2 +
-- 2*Exy*gx*gy -- the variance of the jitter's projection on ANY direction
-- (a centered covariance is PSD by construction, so Q >= 0 everywhere).
-- Enumerating the exact sequence (the same halton/r2 code, periods and
-- centering as the camera hook) makes the coefficients exact for the
-- configured sequence and scale. With jitter disabled the energies are
-- tiny positive sentinels so the shader's "constant provided" gate
-- engages the exact (noise-only) seed.
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

local function jitterPhaseCovariance()
    local s = M.settings
    local scale = tonumber(s.jitterScale) or 1.0
    if not s.useJitter or scale <= 0.0 then
        return 1e-6, 1e-6, 0.0
    end
    local period = s.useR2Jitter and 32 or 16
    local sumX, sumY = 0.0, 0.0
    local sumXX, sumYY, sumXY = 0.0, 0.0, 0.0
    for i = 0, period - 1 do
        local hx, hy
        if s.useR2Jitter then
            hx, hy = r2_sequence(i)
            hx, hy = hx - 0.5, hy - 0.5
        else
            hx = halton(i + 1, 2) - 0.5
            hy = halton(i + 1, 3) - 0.5
        end
        sumX  = sumX  + hx
        sumY  = sumY  + hy
        sumXX = sumXX + hx * hx
        sumYY = sumYY + hy * hy
        sumXY = sumXY + hx * hy
    end
    local n  = period
    local sc = scale * scale
    -- The centered 2x2 (per-axis variances + cross covariance).
    return sc * (sumXX - sumX * sumX / n) / n,
           sc * (sumYY - sumY * sumY / n) / n,
           sc * (sumXY - sumX * sumY / n) / n
end

-- ============================================================================
-- THE VARH TRANSPORT'S HOST CONSTANTS (recomputed on every apply -- they
-- track the feedback rate and the resample tier)
-- ----------------------------------------------------------------------------
--   * varhLoss: 1 - D, D = the fixed-point READ gain (kernelChainConstants)
--     -- varH is transported in READ semantics (the Kaiser-resampled
--     history the gate tests), and the history term of the recursion
--     re-reads the smooth accumulated field at the fixed-point gain.
--     Encoded as a LOSS so an unset constant (0) degrades to the legacy
--     undamped transport.
--   * whiteLoss: 1 - w, w = the white-input read gain (E_f[sum k^2])^2 --
--     the fresh current-frame content enters the accumulator white across
--     texels, so the fresh-x term of the recursion carries w, not 1.
--   * accFactor: (Dbar + 1 - 2*kappa)/Dbar, the accumulated components'
--     noise structure (see kernelChainConstants).
--
-- KERNEL CONTRACT: the resolve's history resample is a SEPARABLE per-axis
-- windowed sinc -- logical taps n = -2..3, radius 3, Kaiser beta 5.2
-- (kKaiserBeta6Tap) for the 6-tap tier, n = -1..2, radius 2, beta 3.2
-- (kKaiserBeta4Tap) for the 4-tap tier -- selected by useKaiser6. MUST
-- MATCH taaResample.h.hlsl; the constants are logged at startup for
-- cross-checking. The 21-tap fetch pattern's corner skip is ~0.08% of the
-- kernel mass and is not modeled in the spectral chain.
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
-- THE RESAMPLE KERNEL'S ACCUMULATION CHAIN (the spectral fixed point)
-- ----------------------------------------------------------------------------
-- varH means Var(READ) -- the Kaiser-resampled history that the gate and the
-- detector actually test. Three constants, all measured on the ACTUAL per-tier
-- kernel:
--   * w (whiteGain)  = (E_f[sum_n k(n,f)^2])^2: the read gain against WHITE
--     input. The fresh current-frame content enters the accumulator white
--     across texels, so the recursion's fresh-x term carries w. Parseval:
--     the per-axis w1D is also the frequency-mean of Kbar^2.
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
--     and D = Vr/Vs.
--
-- THE ACCUMULATED-NOISE STRUCTURE: the soft clip's accumulated components
-- price Var(Kaiser-read of the stored field - center texel) at the spectral
-- fixed point. The 2D noise spectrum is S2D = a^2/(1 - b*Kx2*Ky2); expanding
-- the geometric series, its autocovariance is R2D(x,y) = sum_k b^k * cx_k(x)
-- * cx_k(y) with cx_k = IDFT(Kbar^{2k}) (the a^2 is a common factor and
-- cancels in the normalized r2D). From r2D = R2D/R2D(0,0):
--   kappa = E_phase[ sum_nm c_nm r2D(n, m) ]       (read-vs-texel cov)
--   Dbar  = E_phase[ sum c c' r2D(n-n', m-m') ]    (read var/texel var)
-- both from the ACTUAL per-phase logical 2D coefficients (the 6-tap tier's
-- corner skip included), so Cauchy-Schwarz holds between them and the
-- factor (Dbar + 1 - 2*kappa)/Dbar is >= 0 by construction. The shader
-- consumes it as taaAccNoiseFactor (clamped [0.02, 4]). Built-in
-- consistency: R2D(0,0) = sum_k b^k m_k^2 = the moment series' vsSum
-- (different summation order -- a small float drift is expected, a large
-- one is a bug and warns).
-- ============================================================================
local kernelChainCache = {}
local kernelChainCacheCount = 0

local function kernelChainConstants(beta, radius, a)
    local key = string.format("%.4f:%.2f:%.6f", beta, radius, a)
    local hit = kernelChainCache[key]
    if hit then return hit[1], hit[2], hit[3], hit[4] end

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
        kbar2[j + 1] = math.min(v, 1.0)
    end

    -- The moment series: m_k = mean(Kbar^{2k}), Vs = a^2 sum_k b^k m_k^2,
    -- Vr = a^2 sum_k b^k m_{k+1}^2. Iterated with a frozen-moment geometric
    -- tail (an over-estimate bounded by m_last^2).
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
    local whiteGain = ac[0] * ac[0]
    local readGain  = vr / math.max(vs, 1e-12)
    local noiseAcc  = vr

    -- THE ACCUMULATED-NOISE STRUCTURE (r2D / kappa / Dbar -- see the block
    -- header above). The r2d grid spans +-maxOff in integer texel offsets.
    local maxOff = hi - lo
    local nOff = 2 * maxOff + 1
    local skipCorners = (radius >= 3.0)   -- the 21-tap's four skipped corners
    local r2d = {}
    for xi = 1, nOff do
        r2d[xi] = {}
        for yi = 1, nOff do r2d[xi][yi] = 0.0 end
    end
    do
        local cosTab = {}
        for oi = 1, nOff do
            local off = (oi - 1) - maxOff
            local ct = {}
            for j = 0, N - 1 do ct[j + 1] = math.cos(2.0 * math.pi * j * off / N) end
            cosTab[oi] = ct
        end
        local pow2 = {}
        for j = 1, N do pow2[j] = 1.0 end
        local bPow2, k2 = 1.0, 0
        while true do
            local row = {}
            for oi = 1, nOff do
                local ct = cosTab[oi]
                local s = 0.0
                for j = 1, N do s = s + pow2[j] * ct[j] end
                row[oi] = s / N
            end
            for xi = 1, nOff do
                local rx = r2d[xi]
                for yi = 1, nOff do
                    rx[yi] = rx[yi] + bPow2 * row[xi] * row[yi]
                end
            end
            local mRow = row[maxOff + 1]
            for j = 1, N do pow2[j] = pow2[j] * kbar2[j] end
            bPow2 = bPow2 * b
            k2 = k2 + 1
            if k2 > 384 or bPow2 * mRow * mRow < 1e-12 * math.max(r2d[maxOff + 1][maxOff + 1], 1e-12) then break end
        end
    end
    if math.abs(r2d[maxOff + 1][maxOff + 1] - vsSum) > 1e-3 * math.max(vsSum, 1e-12) then
        if log then
            log("W", "TAA", string.format("accumNoise: r2d(0,0)=%.6e vs vsSum=%.6e (mismatch -- chain bug)",
                r2d[maxOff + 1][maxOff + 1], vsSum))
        end
    end
    local vsNorm = math.max(r2d[maxOff + 1][maxOff + 1], 1e-12)
    local kapSum, dSum = 0.0, 0.0
    for p = 1, nPhase do
        local kp = ktab[p]
        local kapP, dP = 0.0, 0.0
        for n = lo, hi do
            local cn = kp[n]
            for m = lo, hi do
                local cnm = cn * kp[m]
                if skipCorners and (n == lo or n == hi) and (m == lo or m == hi) then
                    cnm = 0.0
                end
                if cnm ~= 0.0 then
                    kapP = kapP + cnm * r2d[n + maxOff + 1][m + maxOff + 1]
                    for n2 = lo, hi do
                        local dn = n - n2 + maxOff + 1
                        local cn2 = kp[n2]
                        for m2 = lo, hi do
                            dP = dP + cnm * cn2 * kp[m2] * r2d[dn][m - m2 + maxOff + 1]
                        end
                    end
                end
            end
        end
        kapSum = kapSum + kapP
        dSum = dSum + dP
    end
    local kappa = kapSum / nPhase / vsNorm
    local dBar  = dSum / nPhase / vsNorm
    local accFactor = (dBar + 1.0 - 2.0 * kappa) / math.max(dBar, 1e-6)

    if kernelChainCacheCount > 32 then
        kernelChainCache = {}
        kernelChainCacheCount = 0
    end
    kernelChainCache[key] = { whiteGain, readGain, noiseAcc, accFactor }
    kernelChainCacheCount = kernelChainCacheCount + 1
    return whiteGain, readGain, noiseAcc, accFactor
end

local function varhHostConstants()
    local s = M.settings
    -- The accumulator rate varies per pixel (motion feedback). The
    -- rate-dependent gains are computed at the FASTEST configured rate so
    -- the charged responses are conservative everywhere.
    local a = 1.0 - math.min(tonumber(s.feedbackMin) or 0.97,
                              tonumber(s.feedbackMax) or 0.97)
    a = math.min(math.max(a, 0.001), 0.9)

    -- The VarH chain's gains from the accumulation's spectral fixed point,
    -- per resample tier (kernelChainConstants).
    local use6 = ((tonumber(s.useKaiser6) or 1.0) > 0.5)
    local whiteGain, readGain, _, accFactorIn = kernelChainConstants(
        use6 and KAISER6_BETA  or KAISER4_BETA,
        use6 and KAISER6_RADIUS or KAISER4_RADIUS,
        a)
    local varhLoss  = math.min(math.max(1.0 - readGain, 0.0), 0.9)
    local whiteLoss = math.min(math.max(1.0 - whiteGain, 0.0), 0.95)
    local accFactor = math.min(math.max(accFactorIn, 0.02), 4.0)

    return varhLoss, whiteLoss, accFactor
end

local varhConstantsLogged = false


-- Defaults: the single source of truth for every setting. M.settings is a
-- working copy that applySettings/user input mutate. When the settings
-- file's version changes, ge/extensions/taa.lua calls M.resetSettings() to
-- rebuild this copy -- merging would keep stale keys from settings that
-- were renamed or removed.
-- RETIRED KEYS: lumaDrift*, clipDistanceRejection*, the CUSUM-era and
-- drift-era keys (clipGhostReset, driftCompensation, driftMaxGain,
-- jitterFlickerPadding, jitterFlickerFade, directionalVariance,
-- clipScopedMu) no longer exist anywhere in the chain. setShaderConst for a
-- constant absent from the cbuffer silently no-ops, so old settings files
-- still carrying these keys are inert; the UI strips unknown keys on load.
M.defaultSettings = {
    useJitter                     = true,
    useR2Jitter                   = true,
    jitterScale                   = 1.0,
    feedbackMin                   = 0.97,
    feedbackMax                   = 0.97,
    -- chi: the gate's coverage radius (3-dof, PER-CHANNEL normalized -- the
    -- coverages below are actually true now). Below ~2.2 legit sub-texel
    -- content is under-covered. 2.2 = 82%, 2.8 = 95%.
    varianceGamma                 = 2.8,
    -- The posterior-mean soft clip's ENABLE. ON by default: with the
    -- empirical spatial-marginal alternative and the sequential prior the
    -- action is the exact posterior mean -- the graded pull is where the
    -- remaining flicker:ghost headroom lives (the hard ellipsoid is the
    -- minimax action and overpays on marginal pixels).
    -- 0 = the pure hard ellipsoid (the A/B baseline).
    softClip                      = 0.0,
    chromaVarianceMod             = 1.0,
    depthRejection                = 0.05,
    -- The comparison is exactly de-jittered and the landing's velocity is
    -- the exact stored effective velocity, so 1.5 is the documented
    -- conservative point. Tighten toward ~0.5 while watching DEBUG MODE 2
    -- (static scene: R dark, B only wobbles; G only on true reveals) and 7.
    velRejection                  = 1.5,
    velGradientScale              = 1.0,
    crossTestStrength             = 0.35,
    -- Manual RCAS strength (0 disables; 1 is the maximum lobe RCAS permits
    -- -- very strong; ~0.3-0.5 is the typical range).
    sharpness                     = 0.50,
    debugMode                     = 0.0,
    useDepthDilation              = 1.0,
    -- Gates the resolve's stored-field analysis (dilation validation, the
    -- depth/velocity disocclusion tests, landing/pursuit). 0 = the resolve
    -- ignores the stored field and disocclusion relies on color clipping
    -- only. The writer pass itself always runs (it cannot be toggled
    -- without re-registering it outside the resolve's pass chain).
    useMotionField                = 1.0,
    lumaVariance                  = 0.0,
    useHullClipping               = 1.0,
    colorSpaceOklab               = 0.0,
    jitterAwareVariance           = 1.0,
    velocityAlignedVariance       = 0.0,
    alignmentFeedbackDrop         = 0.15,
    motionBlendDropSpeed          = 1.0,
    useKaiser6                    = 1.0,
    historyOvershoot              = 1.0,
    clipOvershoot                 = 0.0,
    -- The clip-excess blend reduction: start/width are in GATE RADII of
    -- excess past the boundary.
    clipBlendDrop                 = 0.0,
    clipBlendDropStart            = 0.5,
    clipBlendDropWidth            = 1.0,
    motionBlendStart               = 1.0,
    fallbackFXAA                  = 1.0,
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

        -- The camera's forward displacement this frame (units of 1/rawDepth);
        -- 0 = the shader measures T_y locally. The host could provide this
        -- from the camera hook (res.pos delta dotted with the previous
        -- forward axis) once the depth convention is confirmed.
        pre:setShaderConst("$taaDepthParallaxStep",       0.0)
        -- The Studentization EXACTLY tracks the slider: the fit consumes
        -- the EFFECTIVE radius (chi * (1 + clipOvershoot)) -- the same
        -- threshold the gate tests. The Pade (2,1) rational is fit over
        -- nu in [2 (the Satterthwaite clamp floor), 18], with the anchor
        -- weights concentrated at the operative points (2, nu0, nu_inf).
        local chiEff = (tonumber(s.varianceGamma) or 2.8)
                     * (1.0 + math.max(tonumber(s.clipOvershoot) or 0.0, 0.0))
        local studentA, studentB, studentC = studentFitABC(chiEff)
        pre:setShaderConst("$taaStudentA",               studentA)
        pre:setShaderConst("$taaStudentB",               studentB)
        pre:setShaderConst("$taaStudentC",               studentC)
        -- The arming cap's nu-aware fit: the exact chi^2_nu(0.999)/nu bound
        -- at the record's dof trajectory endpoints, consumed as
        -- 1 + A/sqrt(nu) + B/nu at the live per-pixel nu.
        pre:setShaderConst("$taaArmingA",                ARMING_A)
        pre:setShaderConst("$taaArmingB",                ARMING_B)
        -- The exact-seed phase covariance (the full quadratic form,
        -- tracking useJitter / useR2Jitter / jitterScale; CENTERED on the
        -- sequence mean -- the accumulator's phase average).
        local phEx, phEy, phExy = jitterPhaseCovariance()
        pre:setShaderConst("$taaJitterPhaseEx",  phEx)
        pre:setShaderConst("$taaJitterPhaseEy",  phEy)
        pre:setShaderConst("$taaJitterPhaseExy", phExy)
        -- The VarH transport's two read gains + the accumulated components'
        -- noise factor: they track feedbackMax / useKaiser6, so every apply
        -- recomputes them.
        local varhLoss, whiteLoss, accFactor = varhHostConstants()
        pre:setShaderConst("$taaVarhResampleLoss",       varhLoss)
        pre:setShaderConst("$taaVarhWhiteLoss",          whiteLoss)
        pre:setShaderConst("$taaAccNoiseFactor",         accFactor)
        if not varhConstantsLogged then
            varhConstantsLogged = true
            if log then
                log("I", "TAA", string.format(
                    "VarH host constants: varhLoss=%.4f whiteLoss=%.4f accNoiseFactor=%.4f armingA=%.4f armingB=%.4f studentA=%.4f studentB=%.4f studentC=%.4f (kernel: separable per-axis Kaiser, beta %.1f (6-tap) / %.1f (4-tap) by useKaiser6; VarH gains and the accumulated-noise factor from the spectral fixed point -- MUST MATCH taaResample.h.hlsl / taaClip.h.hlsl; the arming pair and the Student Pade triple MUST MATCH taaClip.h.hlsl's consumption (1 + A*x + B*x^2)/(1 + C*x), x = 1/nu; ingest dof = 1.0, nu_inf = %.2f)",
                    varhLoss, whiteLoss, accFactor, ARMING_A, ARMING_B, studentA, studentB, studentC, KAISER6_BETA, KAISER4_BETA, STUDENT_NUINF))
            end
        end
        pre:setShaderConst("$taaSoftClip",                s.softClip)
        pre:setShaderConst("$taaChromaVarianceMod",       s.chromaVarianceMod)
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
        pre:setShaderConst("$taaClipBlendDrop",           s.clipBlendDrop)
        pre:setShaderConst("$taaClipBlendDropStart",      s.clipBlendDropStart)
        pre:setShaderConst("$taaClipBlendDropWidth",      s.clipBlendDropWidth)
        pre:setShaderConst("$taaMotionBlendStart",        s.motionBlendStart)
        pre:setShaderConst("$taaFallbackFXAA",            s.fallbackFXAA)
        -- Debug modes never rebind the history source: the resolve's RGB is
        -- always the real blend (views ride the alpha, rendered by
        -- TAA_FinalFx), so the accumulation simply keeps running while a
        -- debug mode is active.
    end

    -- The motion-field writer must classify with the SAME parameters as the
    -- resolve, or the stored field disagrees with the resolve's own
    -- classification. Setting its constants is the ONLY interaction with
    -- this pass -- its lifecycle is never touched (see build()).
    if mot then
        mot:setShaderConst("$taaUseDepthDilation", s.useDepthDilation)
        -- Gates the writer's fast path (a 2-fetch write).
        mot:setShaderConst("$taaUseMotionField", s.useMotionField)
    end

    if fin then
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