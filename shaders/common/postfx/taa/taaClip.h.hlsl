// ============================================================================
// TAA history clipping: THE FLAT MODEL -- exact least-squares bookkeeping
// ----------------------------------------------------------------------------
// THE PLANE: the robust coupled 2x2 fit (one t-IRLS step, drop-one trimmed
// scale) with the exact hat-matrix bookkeeping:
//   * residSq's dof: E[Sum w r^2] = sigma^2 * Sum w(1-h), h_i =
//     (w_i/W)(1 + mahal_i), Sum h_i = 3 exactly (the trace identity).
//   * the IRLS studentization: q_i = r_i^2/((1-h_i)s^2) has E = 1 per tap
//     (the classical internally-studentized residual), so clean taps keep
//     weight exactly 1 and a contaminated CENTER tap -- the highest-hat --
//     is downweighted at its true deviation.
//   * the evaluation variance: vFit = sPer * Sum c_k^2 with c_k the EXACT
//     evaluation functional coefficients; Sum c^2 reproduces the classical
//     (1/n)(1+mahal) on uniform weights.
//   * the correlation dof from the W2 Kish count (the moments' own
//     effective size); correlations shrunk by the MSE-optimal linear
//     shrink and shared by every consumer through ONE metric (one hoisted
//     R^-1 at unit scale -- the |R| log-dets cancel exactly between the
//     hypotheses' log-dets).
//   * shared chi^2_3 levels: the corroboration reset (5%), the winsor
//     (0.1%), the spike's record branch (~1e-6) -- one law, three tests,
//     each through the /3 of the 3-channel sum.
//
// THE RECORD: the transported sigma^2 MEANS sPer, ingested H-free (noise
// lives in dev = x0c - plane0; the de-bias is the exact identity E[dev^2]
// = sPer + vPrior - 2*cPT with the exact lever). The winsor censors at the
// ingest's own Satterthwaite/Pearson-III dof, at RECORD scale with the
// mean shift. The spike's record branch is a whitened chi^2_3 test at
// ~1e-6; the range branch (16x the AABB range) is the distribution-free
// garbage backstop. The change-class eviction job belongs to the hard
// clamp (clamp-rate eviction ~3 frames at 3 sigma); the reveal job to
// geoCorroboration.
//
// THE GATE: the history's accumulation target is the phase-swept CENTER
// TEXEL; the gate tests the history against the posterior of that target
// given the current frame. PRIOR: plane0, variance tauB2 + sPer*evalVar.
// LIKELIHOOD: the phase-corrected center tap, variance sPer (the record),
// t-robustified at the record's dof with the exact Var(dev) quadratic
// form, the shared-sample covariance charged. At edges the plane residual
// carries the step -- the flip REQUIRES that width.
//
// THE SOFT CLIP (the measured miss model): past the hard boundary the
// action is the exact posterior mean. H0: the hierarchical null at its
// honest un-inflated predictive variance. H1: the LANDING-MISS hypothesis
// -- a proper mixture of the 8 NEIGHBOR TEXELS' accumulated content plus
// one temporal tail; the CENTER TAP is not a component (it IS the null's
// likelihood). The whole alternative is priced by ONE miss law: the
// ELLIPTICAL STUDENT at the measured scale dof (the scale is a point
// estimate; the tail and the cell masses marginalize its uncertainty --
// Log2RadialTail2 / the mass core), and the prior odds additionally carry
// a measured CONDITIONAL RATE multiplier for the two high-risk landing
// classes (similarity-extrapolated motion, velocity-straddled quads --
// computed in taa.fx, missModel.w). On clean static content the prior
// still collapses (large dof: the floors are conventions, not estimates)
// and the soft clip switches itself off; on noisy-velocity content it
// engages more decisively than any fixed prior could. The base-2 log
// domain throughout (log2/exp2 are the native opcodes).
//
// THE H1 COMPONENTS: the accumulated history texels at the landing's
// neighbors. The component variance is the accumulated read-vs-texel
// structure (varH * taaAccNoiseFactor) PLUS the within-cell gradient term
// -- the miss model's own TRUNCATED-cell second moment (per axis, per
// cell geometry; TruncatedCellMoments), NOT the uniform cell's 1/12. NO
// sPer (the accumulated proxy carries no current-frame noise) and NO
// transport term (the miss distribution -- the masses -- already charges
// the landing scatter). The cold path keeps the no-record limit of the
// same model (gateVarBase is the honest stand-in).
//
// THE CLIP-EXCESS BLEND REDUCTION: excess = sqrt(mdd)/chi, in gate radii,
// linearly drops the history blend to zero past a start margin. Purely
// blend-side; the mode-13 null audit is its license to exist. Null
// behavior at the defaults (start 0.5, width 1.0, 95% coverage): ~0.3% of
// null pixels touch the bottom of the ramp past 1.5 radii, full drop at
// 2.5 radii (~2e-8) -- true reveals sit at 3-10+ radii.
//
// THE TRANSPORT (tag 101): [31] sign, [30:28] tag, [27:22] sigma (6-bit,
// = sPer), [21:10] VarH ratio (12-bit), [9:6] age (4-bit), [5:0] split
// fraction (6-bit, 1/32 steps over [-1, 0.97]). Every packer is TOTAL:
// each uint() operand is scrubbed to its field's range first (f2u of NaN
// is undefined on every backend, and NaN in any transported field would
// persist through the history forever).
//
// AUDITED, DO NOT RE-ATTEMPT:
//   * a Sum-c^2 (or any kernel-quadratic-form) discount on the gate's
//     varH charge, INCLUDING at the decode-blend: varH is transported in
//     READ semantics -- it already IS the Kaiser read's variance, and
//     each neighbor's transported value already contains the full
//     kernel-weights-vs-field-autocovariance quadratic form. Discounting
//     double-counts the read's smoothing.
//   * the -3 in the correlation dof's Fisher-z form (already exact).
//   * a directional soft-clip pull toward the matched neighbor (the
//     current-frame posterior IS the correct store under H1; pulling
//     toward the neighbor would store NEIGHBOR CONTENT at this texel and
//     perpetuate the miss).
//   * a dynamic ingest dof (would break the STUDENT_NUINF host sync; the
//     dev IS a single Gaussian linear functional -- chi^2_1 is exact).
//
// MUST MATCH client/postFx/taa.lua: kClipSigmaEmaRate (rho),
// kClipStudentPriorDof (nu0), kRecordIngestDof, the Pade (2,1)
// Studentization triple (taaStudentA/B/C), the arming pair
// (taaArmingA/B), and the kernel betas/radii (taaResample.h.hlsl).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host
// context: cbuffer perDraw (taaVarianceGamma, taaClipOvershoot,
// taaChromaVarianceMod, taaSoftClip, taaStudentA/B/C, taaArmingA/B,
// taaJitterPhaseEx/Ey/Exy, taaJitterAwareVariance, taaLumaVariance,
// taaVelocityAlignedVariance, taaHistoryOvershoot, the feedback
// constants, taaAccNoiseFactor, taaClipBlendDrop/Start/Width). Requires
// taaShared.h.hlsl, taaConstants.h.hlsl included before.
// ============================================================================
#ifndef TAA_CLIP_H_HLSL
#define TAA_CLIP_H_HLSL

// ============================================================================
// THE COLOR-FIELD STATISTICS (moments -> the robust coupled 2x2 plane)
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 sigmaClean;          // the post-plane residual scale, per channel
                                // (the soft clip's temporal tail)
    float  invNeff;             // the SANDWICH information count (dof side:
                                // Sum w*psi' mass, information-positive only)
    float  invNeffW2;           // the W2 Kish count -- the effective sample
                                // size of the moments the correlations are
                                // computed FROM (the correlations' dof base)
    float  evalVar;             // Sum c_k^2: the EXACT evaluation variance
                                // (the vFit multiplier -- the plain w2
                                // concentration plus the exact lever terms)
    float  residDof;            // Sum w(1-h) on the robust weights: residSq's
                                // own dof (the Satterthwaite denominator)
    float3 phaseShift;          // -(gradLS * j): the innovation's phase
                                // correction, the tap's, the soft clip's
    float3 gradLSx;             // the robust LS plane gradient
    float3 gradLSy;
    float3 plane0;              // the plane at the target (the prior center)
    float3 covCross;            // the taps' cross-channel covariances
                                // (the metric's shape)
    float3 residSq;             // the post-plane residual (exact dof)
    float  rangeSq;             // full-set squared AABB range (the spike
                                // reference's garbage branch)
    float3 priorTapCov;         // F1: Cov(fitted value, center tap)
    float3 transportVar;        // the gated landing error projected on the
                                // working gradient (MOTION-GATED)
};

// ============================================================================
// CONSTANTS
// ============================================================================
static const float kClipSigmaEmaRate         = 0.15;
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);
// ---- the shared chi^2_3 levels (one law, three tests) ----
// All three innovation-scale tests run through the SAME channel model: the
// corroboration reset and the spike's record branch are WHITENED through
// the shrunk full metric (exactly chi^2_3 under the null), and the winsor
// censors at the ingest's own Satterthwaite dof. A scalar-total form would
// assume exchangeable uncorrelated channels: on luminance-dominated YCoCg
// deviations the 3-channel sum's effective dof is ~1, so the priced 1e-6
// spike level would run at ~1e-3 and the 0.1% winsor point at ~2%.
static const float kChiSq3_95   = 7.815;     // the corroboration reset: 5%
static const float kChiSq3_1e6  = 31.0;      // the spike's record branch
                                             // (exact tail 8.5e-7 via the
                                             // erfc form)
// The degenerate-spread guard for the exact evaluation variance (the
// un-clamped value is unbounded along an unobserved direction; the
// decoupled-fit branch drops that direction and this bounds the charge).
static const float kMaxEvalVar  = 4.0;
// ---- the record ----
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
// The record ingest's effective chi^2 dof: the de-biased squared dev of a
// single tap, chi^2_1. MUST MATCH client/postFx/taa.lua's STUDENT_NUINF.
static const float kRecordIngestDof = 1.0;
// ---- the soft clip's measured miss model ----
// The miss law is the elliptical Student at the measured scale dof (see
// the header). r0 = the "holds another texel" radius (the prior odds'
// threshold -- the principled dial if the hot regime over-pulls: raise to
// 1.25 to cool the prior at every sigma). r1 = beyond the neighbor ring
// (the tail). kTailChangeFloor: optional mass for the change-class ghost
// (old content absent for reasons other than a miss, e.g. lighting steps)
// -- 0 = the pure miss model; raise to ~0.02-0.1 if static-scene lighting
// transitions evict too slowly.
static const float kMinMissSigma    = 0.125;  // texels: the floor (underflow
                                              // safety; below it "no miss")
static const float kMissRadiusOdds  = 1.0;
static const float kMissRadiusTail  = 1.5;
static const float kTailChangeFloor = 0.0;
// The cell-mass Euler-Maclaurin correction's clamp floor: below sigma ~
// 0.29 texel the two-term expansion's zero-z factor would go negative; the
// prior odds have already collapsed there, so the clamp only keeps the log
// finite.
static const float kMinMassCorr     = 0.0625;
// The tail component's content-scale prior (the unknown old content's
// spread around the neighborhood mean, in content sigmas).
static const float kGhostPriorSigma = 1.0;
// ---- the miss scale's dof and the conditional miss rate (consumed in
//      taa.fx's miss-model block; see the header). The dof constants
//      price the velocity-noise component's estimation uncertainty -- the
//      tail law and the cell masses marginalize it as an Inv-chi^2 scale
//      mixture. The rate constants are log2 multipliers on the prior odds
//      for the two measured high-risk landing classes: the SCALE prices
//      how far a miss lands, these price how OFTEN one happens at all.
static const float kMissNuVelFloor        = 32.0;  // the fixed 1/16-px floor and the uniform-field classification: conventions, not measurements (numerically Gaussian by nu ~ 256)
static const float kMissNuVelMeasured     = 6.0;   // the 36-pair quant-step scan's effective dof (nine heavily correlated taps)
static const float kMissRateExtrapolated2 = 3.0;   // log2(8): the landing's motion came from the two-tap similarity extrapolation
static const float kMissRateStraddled2    = 2.0;   // log2(4): the velocity phase quad straddles a layer step
static const float kMissRateBoostMax2     = 5.0;   // log2(32): the combined cap
// ---- the plane's one-step t-IRLS ----
// The fit's robustification dof. The drop-one trimmed scale makes the
// step self-discriminating: a surface or color step through the center
// texel leaves >= 3 coherent far-side taps in the scale (mild
// reweighting -- the gate stays step-wide), while an isolated sparkle is
// 1-2 texels (excluded from the scale, strong downweighting).
static const float kPlaneIrlsDof = 4.5;
// ---- the structure split's transport ----
// The EMA rate for the split fraction. Steady-state noise ~0.31; ~5-frame
// convergence after a reset.
static const float kSplitEmaRate = 0.2;
// ---- the transport scatter's motion gate ----
static const float kTransportGatePx = 2.0;

// ============================================================================
// THE MOMENT ACCUMULATORS (lean for pass 1, full for pass 2)
// ============================================================================
// Pass 1 (the plain weighted LS plane) consumes only the first moments and
// the plane system; the second-moment / w^2-squared fields of the full
// accumulator are dead until pass 2. The lean variant keeps ~15 floats
// live instead of the full ~27 -- source-level translation paths that do
// not DCE struct fields through out-params would otherwise carry all of
// them through pass 1 (~108 dead MADs per pixel).
struct ColorMomentAccumLean
{
    float3 sum3;
    float3 covXSum;
    float3 covYSum;
    float  totalW;
    float2 offSum;
    float  offXSq;
    float  offYSq;
    float  offXY;
};

void AccumulateColorMomentsLean(float3 taps[9], float w[9], out ColorMomentAccumLean a)
{
    a.sum3    = float3(0.0, 0.0, 0.0);
    a.covXSum = float3(0.0, 0.0, 0.0);
    a.covYSum = float3(0.0, 0.0, 0.0);
    a.totalW  = 0.0;
    a.offSum  = float2(0.0, 0.0);
    a.offXSq  = 0.0;
    a.offYSq  = 0.0;
    a.offXY   = 0.0;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float3 c  = taps[i];
        float2 o  = kOffsets3x3[i];
        float  wi = w[i];

        a.sum3    += c * wi;
        a.covXSum += o.x * c * wi;
        a.covYSum += o.y * c * wi;
        a.totalW  += wi;
        a.offSum  += o * wi;
        a.offXSq  += o.x * o.x * wi;
        a.offYSq  += o.y * o.y * wi;
        a.offXY   += o.x * o.y * wi;
    }
}

struct ColorMomentAccum
{
    float3 sum3;
    float3 sumSq3;
    float  totalW;
    float  totalWSq;
    float2 offSum;
    float  offXSq;
    float  offYSq;
    float  offXY;
    float3 covXSum;
    float3 covYSum;
    float  wXY;
    float  wXZ;
    float  wYZ;
    float2 offSumW2;   // Sum w^2 o       (the exact evaluation variance)
    float  offXSqW2;   // Sum w^2 ox^2
    float  offYYSqW2;  // Sum w^2 oy^2
    float  offXYW2;    // Sum w^2 ox oy
};

void AccumulateColorMoments(float3 taps[9], float w[9], out ColorMomentAccum a)
{
    a.sum3     = float3(0.0, 0.0, 0.0);
    a.sumSq3   = float3(0.0, 0.0, 0.0);
    a.totalW   = 0.0;
    a.totalWSq = 0.0;
    a.offSum   = float2(0.0, 0.0);
    a.offXSq   = 0.0;
    a.offYSq   = 0.0;
    a.offXY    = 0.0;
    a.covXSum  = float3(0.0, 0.0, 0.0);
    a.covYSum  = float3(0.0, 0.0, 0.0);
    a.wXY = 0.0;
    a.wXZ = 0.0;
    a.wYZ = 0.0;
    a.offSumW2 = float2(0.0, 0.0);
    a.offXSqW2 = 0.0;
    a.offYYSqW2 = 0.0;
    a.offXYW2  = 0.0;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float3 c   = taps[i];
        float2 o   = kOffsets3x3[i];
        float  wi  = w[i];
        float  wi2 = wi * wi;

        a.sum3     += c * wi;
        a.sumSq3   += c * c * wi;
        a.totalW   += wi;
        a.totalWSq += wi2;
        a.offSum   += o * wi;
        a.offXSq   += o.x * o.x * wi;
        a.offYSq   += o.y * o.y * wi;
        a.offXY    += o.x * o.y * wi;
        a.covXSum  += o.x * c * wi;
        a.covYSum  += o.y * c * wi;
        a.wXY += wi * c.x * c.y;
        a.wXZ += wi * c.x * c.z;
        a.wYZ += wi * c.y * c.z;
        a.offSumW2 += o * wi2;
        a.offXSqW2 += o.x * o.x * wi2;
        a.offYYSqW2 += o.y * o.y * wi2;
        a.offXYW2  += o.x * o.y * wi2;
    }
}

struct ColorPlaneFit
{
    float3 gradX;
    float3 gradY;
    float  vXX;
    float  vYY;
    float  vXY;
    float  detV;
    float3 covX;
    float3 covY;
};

ColorPlaneFit SolveColorPlaneCore(
    float offXSq, float offYSq, float offXY,
    float3 covXSum, float3 covYSum,
    float3 mean3, float2 cenPx, float invTotalWeight)
{
    ColorPlaneFit p;
    p.vXX  = max(offXSq * invTotalWeight - cenPx.x * cenPx.x, 1e-6);
    p.vYY  = max(offYSq * invTotalWeight - cenPx.y * cenPx.y, 1e-6);
    p.vXY  = offXY * invTotalWeight - cenPx.x * cenPx.y;
    p.covX = covXSum * invTotalWeight - cenPx.x * mean3;
    p.covY = covYSum * invTotalWeight - cenPx.y * mean3;
    p.detV = p.vXX * p.vYY - p.vXY * p.vXY;
    if (p.detV > 1e-6)
    {
        p.gradX = (p.vYY * p.covX - p.vXY * p.covY) / p.detV;
        p.gradY = (p.vXX * p.covY - p.vXY * p.covX) / p.detV;
    }
    else
    {
        // degenerate spread (1D): the decoupled fit, exact when vXY = 0
        p.gradX = p.covX / p.vXX;
        p.gradY = p.covY / p.vYY;
    }
    return p;
}

ColorPlaneFit SolveColorPlane(ColorMomentAccum a, float3 mean3, float2 cenPx, float invTotalWeight)
{
    return SolveColorPlaneCore(a.offXSq, a.offYSq, a.offXY, a.covXSum, a.covYSum, mean3, cenPx, invTotalWeight);
}

ColorPlaneFit SolveColorPlane(ColorMomentAccumLean a, float3 mean3, float2 cenPx, float invTotalWeight)
{
    return SolveColorPlaneCore(a.offXSq, a.offYSq, a.offXY, a.covXSum, a.covYSum, mean3, cenPx, invTotalWeight);
}

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx, float2 transportErrPx)
{
    ColorNeighborhoodStats stats;
    stats.priorTapCov = float3(0.0, 0.0, 0.0);
    stats.transportVar = float3(0.0, 0.0, 0.0);

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    // The jitter-centered Gaussian is separable: six exp2 for nine weights.
    float wAxisX[3] = { 0.0, 0.0, 0.0 };
    float wAxisY[3] = { 0.0, 0.0, 0.0 };
    if (jitterCenteredWeights)
    {
        [unroll]
        for (int ax = 0; ax < 3; ++ax)
        {
            float dx = float(ax - 1) - weightCenterPx.x;
            float dy = float(ax - 1) - weightCenterPx.y;
            wAxisX[ax] = exp2(-dx * dx * kLog2E);
            wAxisY[ax] = exp2(-dy * dy * kLog2E);
        }
    }

    // ---- the design weights: the plain kernel times the weighting
    //      modifiers -------------------------------------------------------
    float motionFactor = saturate(motionNormalized);
    float3 aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    float3 aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
    float w9[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        aabbMin = min(aabbMin, tapColorSpace);
        aabbMax = max(aabbMax, tapColorSpace);

        float w;
        if (jitterCenteredWeights)
            w = wAxisX[(int)kOffsets3x3[i].x + 1] * wAxisY[(int)kOffsets3x3[i].y + 1];
        else
            w = kStdWeights[i];

        if (taaLumaVariance > 0.5) { w *= (1.0 / (1.0 + max(tapColorSpace.x, 0.0))); }
        if (taaVelocityAlignedVariance > 0.5 && i > 0)
        {
            w *= lerp(1.0, saturate(dot(tapOffsetPx, motionDirUnit) * kInvLength[i] * 0.5 + 0.5), motionFactor);
        }
        w9[i] = w;
    }

    // The AABB's only consumer is the spike reference's garbage branch, and
    // it consumes just the squared range: collapse the extremes HERE so six
    // min/max registers do not stay live across both IRLS passes.
    stats.rangeSq = dot(aabbMax - aabbMin, aabbMax - aabbMin);

    // ---- PASS 1: the plain weighted LS plane ---------------------------
    ColorMomentAccumLean acc1;
    AccumulateColorMomentsLean(neighborhoodColorSpace, w9, acc1);
    float  invW1 = 1.0 / max(acc1.totalW, kEpsilon);
    float3 mean1 = acc1.sum3 * invW1;
    float2 cen1  = acc1.offSum * invW1;
    ColorPlaneFit fit1 = SolveColorPlane(acc1, mean1, cen1, invW1);

    // ---- the PASS-1 HAT DIAGONALS and the studentized residuals -------
    // h_i = (w_i/W)(1 + mahal_i);  Var(r_i) = sigma^2 (1 - h_i);  Sum h_i
    // = 3 exactly (the trace identity). The studentized residual has
    // E[q] = 1 per tap, so the IRLS sees a calibrated chi^2_1: clean taps
    // keep weight exactly 1 and a contaminated CENTER tap (the highest
    // hat) is downweighted at its true deviation.
    float2 vInv1_0 = float2(0.0, 0.0);
    float2 vInv1_1 = float2(0.0, 0.0);
    if (fit1.detV > 1e-6)
    {
        vInv1_0 = float2( fit1.vYY, -fit1.vXY) / fit1.detV;
        vInv1_1 = float2(-fit1.vXY,  fit1.vXX) / fit1.detV;
    }
    float  rStdSq[9];
    float  sumWrStd2 = 0.0;
    float  maxWrStd2 = 0.0;
    float  wAtMax = 0.0;
    [unroll]
    for (int r = 0; r < 9; ++r)
    {
        float2 d1 = kOffsets3x3[r] - cen1;
        float3 res = neighborhoodColorSpace[r]
                   - (mean1 + fit1.gradX * d1.x + fit1.gradY * d1.y);
        float mahal = (fit1.detV > 1e-6)
            ? (d1.x * dot(vInv1_0, d1) + d1.y * dot(vInv1_1, d1))
            : 0.0;
        float oneMinusH = max(1.0 - w9[r] * (1.0 + mahal) * invW1, 0.1);
        rStdSq[r] = dot(res, res) / oneMinusH;
        float wr = w9[r] * rStdSq[r];
        sumWrStd2 += wr;
        if (wr > maxWrStd2) { maxWrStd2 = wr; wAtMax = w9[r]; }
    }
    // The drop-one trimmed scale on the STUDENTIZED residuals (E = sigma^2
    // per tap; dropping the largest biases low -> q high -> conservative).
    // No dof correction: the studentization already makes the scale exact.
    float sigmaTrimSq = max((sumWrStd2 - maxWrStd2)
                            / max(acc1.totalW - wAtMax, 0.5),
                            kMinSigma * kMinSigma);

    // ---- the t-IRLS multipliers and the SANDWICH count -----------------
    // w2: the Lange refit weights (w_k * (nu+1)/(nu+q)). One step from
    //     the LS start: full efficiency at the Gaussian, bounded influence
    //     at the contamination.
    // w3: the INFORMATION-POSITIVE mass (the Godambe/sandwich count). The
    //     true information of an M-estimate is Sum w*psi'(u), and psi' for
    //     the Student weight goes NEGATIVE past q = nu: a tap pushed past
    //     the redescend threshold is anti-informative, not weakly
    //     informative. The count saturates its factor at zero; on clean
    //     content the factor is a near-constant that cancels in the
    //     concentration ratio (~6% dof inflation -- conservative), and at
    //     a firefly the tap leaves the count entirely, dropping the dof
    //     and widening the Student factor exactly when the plane was
    //     contaminated.
    float w2[9];
    float w3[9];
    float sumW3 = 0.0;
    float sumW3Sq = 0.0;
    [unroll]
    for (int t = 0; t < 9; ++t)
    {
        float q = rStdSq[t] / sigmaTrimSq;
        w2[t] = w9[t] * ((kPlaneIrlsDof + 1.0) / (kPlaneIrlsDof + q));
        w3[t] = w2[t] * saturate((kPlaneIrlsDof - q) / (kPlaneIrlsDof + q));
        sumW3   += w3[t];
        sumW3Sq += w3[t] * w3[t];
    }

    // ---- PASS 2: the reweighted moments -- everything downstream -------
    ColorMomentAccum acc;
    AccumulateColorMoments(neighborhoodColorSpace, w2, acc);

    float invTotalWeight = 1.0 / max(acc.totalW, kEpsilon);
    float3 mean3         = acc.sum3 * invTotalWeight;
    stats.invNeff        = saturate(sumW3Sq / max(sumW3 * sumW3, 1e-12));
    float2 cenPx         = acc.offSum * invTotalWeight;

    {
        ColorPlaneFit fit = SolveColorPlane(acc, mean3, cenPx, invTotalWeight);
        stats.gradLSx = fit.gradX;
        stats.gradLSy = fit.gradY;
        stats.phaseShift = -(fit.gradX * jitterPx.x + fit.gradY * jitterPx.y);

        float2 dEval = jitterPx - cenPx;
        float plainInvNeff = acc.totalWSq / max(acc.totalW * acc.totalW, 1e-6);
        // The W2 Kish count: the moments' own effective sample size,
        // exported for the correlations' dof (see ShrunkenCorrelations).
        // plainInvNeff IS it -- no recomputation.
        stats.invNeffW2 = saturate(plainInvNeff);

        // The inverse scatter matrix and its rank flag, built ONCE: the
        // exact evaluation variance, the exact residual dof and the F1
        // prior/tap covariance below each would otherwise rebuild identical
        // copies of this from the same pass-2 fit (three conditional
        // inversions collapsed into one).
        float2 vInv0 = float2(0.0, 0.0);
        float2 vInv1 = float2(0.0, 0.0);
        bool scatterFullRank = (fit.detV > 1e-6);
        if (scatterFullRank)
        {
            vInv0 = float2( fit.vYY, -fit.vXY) / fit.detV;
            vInv1 = float2(-fit.vXY,  fit.vXX) / fit.detV;
        }

        // THE EXACT EVALUATION VARIANCE: plane0 = Sum c_k tap_k with
        //   c_k = (w_k/W)(1 + dEval' S^-1 (o_k - cen)),  vFit = sPer*Sum c_k^2
        //   Sum c^2 = (Sum w^2 + 2*cross + quad) / W^2
        //     cross = dEval' S^-1 (Sum w^2 (o - cen))
        //     quad  = dEval' S^-1 (Sum w^2 (o-cen)(o-cen)') S^-1 dEval
        // Reproduces the classical (1/n)(1 + mahal) on uniform weights
        // exactly (cross vanishes by centering). The PLAIN w2
        // concentration -- NOT the sandwich count: w3 is the information
        // (a dof-side quantity); mixing it here would double-count the
        // psi' factor into the variance. Degenerate spread: the unobserved
        // direction is dropped by the decoupled fit; the charge is guarded.
        // Shared w^2 moments and S^-1 projections: the exact evaluation
        // variance and the F1 lever below both consume these -- hoisted
        // once (also guarantees the sharing on translation paths whose CSE
        // is weak across scopes).
        float2 sumW2offCen = acc.offSumW2 - cenPx * acc.totalWSq;
        float  sXX = acc.offXSqW2 - 2.0 * cenPx.x * acc.offSumW2.x + acc.totalWSq * cenPx.x * cenPx.x;
        float  sYY = acc.offYYSqW2 - 2.0 * cenPx.y * acc.offSumW2.y + acc.totalWSq * cenPx.y * cenPx.y;
        float  sXY = acc.offXYW2 - cenPx.x * acc.offSumW2.y
                   - cenPx.y * acc.offSumW2.x
                   + acc.totalWSq * cenPx.x * cenPx.y;
        float2 vC = float2(dot(vInv0, cenPx),  dot(vInv1, cenPx));   // S^-1 cen
        float2 vD = float2(dot(vInv0, dEval), dot(vInv1, dEval));   // S^-1 dEval

        float evalVar;
        if (scatterFullRank)
        {
            float cross = dot(vD, sumW2offCen);
            float quad  = vD.x * vD.x * sXX + vD.y * vD.y * sYY
                        + 2.0 * vD.x * vD.y * sXY;
            evalVar = (acc.totalWSq + 2.0 * cross + quad)
                    / max(acc.totalW * acc.totalW, 1e-6);
            evalVar = clamp(evalVar, 1.0 / 9.0, plainInvNeff + kMaxEvalVar);
        }
        else
        {
            evalVar = plainInvNeff + kMaxEvalVar;
        }
        stats.evalVar = evalVar;
        stats.plane0 = mean3 + fit.gradX * dEval.x + fit.gradY * dEval.y;

        // THE EXACT RESIDUAL DOF: E[Sum w r^2] = sigma^2 * Sum w(1-h).
        // This is the stability condition of the H-free ingest (with the
        // dof biased high, the de-bias over-subtracts and the record sags
        // without a fixed point).
        float sumW2OnePlusMahal = 0.0;
        if (scatterFullRank)
        {
            [unroll]
            for (int hh = 0; hh < 9; ++hh)
            {
                float2 d2 = kOffsets3x3[hh] - cenPx;
                sumW2OnePlusMahal += w2[hh] * w2[hh]
                    * (1.0 + d2.x * dot(vInv0, d2) + d2.y * dot(vInv1, d2));
            }
        }
        else
        {
            sumW2OnePlusMahal = acc.totalWSq;
        }
        float dofSSR = max(acc.totalW - sumW2OnePlusMahal * invTotalWeight, 0.25);
        stats.residDof = dofSSR;

        float3 scc = acc.sumSq3 * invTotalWeight - mean3 * mean3;
        float3 ssr = scc
                   - 2.0 * (fit.gradX * fit.covX + fit.gradY * fit.covY)
                   + (fit.gradX * fit.gradX * fit.vXX + fit.gradY * fit.gradY * fit.vYY + 2.0 * fit.gradX * fit.gradY * fit.vXY);
        stats.residSq = max(ssr * acc.totalW / dofSSR, 0.0);

        // F1: the prior/tap shared-sample covariance on the robust weights
        // -- THE EXACT LEVER. The load-bearing identity is the ingest's
        // de-bias, E[dev^2] = sPer + vPrior - 2*cPT with dev = x0c - plane0
        // and the FITTED phase correction. The GEOMETRY (k0 + L below) is
        // exact for any white per-tap perturbation; the SCALE is residSq,
        // NOT sPer: within one frame the fitted plane and the tap share
        // BOTH the temporal noise and the stable structure (the same tap
        // realization), so the shared-sample covariance is charged at the
        // full measured spatial residual. Intended consequence: the
        // ingest's fixed point is sPer + sigma_S^2*evalVar -- the record
        // tracks the tap's effective variance against the PHASE-SWEPT
        // target (the structure's phase-variance included), which is the
        // likelihood variance the gate consumes; the structure side is
        // bounded by the winsor and arming caps.
        //     k0 = (w2_0/W) * (1 - dEval' S^-1 cen)   -- exact when the
        //          phase gradient is treated as deterministic,
        //     L  = jitterPx' S^-1 c2
        //        + 0.5 * (dEval' S^-1 S2 S^-1 dEval - cen' S^-1 S2 S^-1 cen)
        //        + (w2_0/W) * jitterPx' S^-1 cen
        // with c2 = Sum (w2/W)^2 (o - cen) and S2 = Sum (w2/W)^2 (o-cen)
        // (o-cen)' -- the w2-SQUARED centroid offset and scatter (the same
        // accumulator class the exact evaluation variance uses).
        // DERIVATION (do not flip signs again): with plane0 = Sum a_k x_k
        // and x0c = x_0 + ghat*j = Sum b_k x_k (b_0 = 1 + beta_0, beta_k =
        // (w_k/W) jitterPx' S^-1 (o_k - cen)),
        //     E[dev^2]/sigma^2 = 1 + 2*beta_0 + Sum beta^2 + Sum a^2
        //                        - 2*Sum a*b,
        // and matching the transported identity pins
        //     cPT/sigma^2 = Sum a*b - beta_0 - Sum beta^2 / 2
        //                 = k0 + L  exactly as written. The last term IS
        // -beta_0 and enters with PLUS; the degenerate limits (j = 0,
        // uniform weights) both zero it and CANNOT discriminate its sign.
        // Degenerate spread: the plain shared-sample term stands (the
        // lever's S^-1 does not exist). jitterPx = 0 makes L vanish and
        // k0 = b0 identically; uniform w2 reduces L to
        // dEval' S^-1 dEval / (2N).
        float priorTapCoef = w2[0] * invTotalWeight;
        if (scatterFullRank)
        {
            float  invW2 = invTotalWeight * invTotalWeight;
            float2 uC2 = float2(dot(vInv0, sumW2offCen), dot(vInv1, sumW2offCen)); // S^-1 Sum w2^2 (o-cen)
            float quadD = (sXX * vD.x * vD.x + 2.0 * sXY * vD.x * vD.y + sYY * vD.y * vD.y) * invW2;
            float quadC = (sXX * vC.x * vC.x + 2.0 * sXY * vC.x * vC.y + sYY * vC.y * vC.y) * invW2;
            float k0 = (w2[0] * invTotalWeight) * (1.0 - dot(dEval, vC));
            float lever = dot(jitterPx, uC2) * invW2
                        + 0.5 * (quadD - quadC)
                        + (w2[0] * invTotalWeight) * dot(jitterPx, vC);
            priorTapCoef = k0 + lever;
        }
        stats.priorTapCov = max(stats.residSq * priorTapCoef, float3(0.0, 0.0, 0.0));

        // the transport scatter, projected on the robust gradient
        stats.transportVar = (transportErrPx.x * transportErrPx.x) * (fit.gradX * fit.gradX)
                           + (transportErrPx.y * transportErrPx.y) * (fit.gradY * fit.gradY);

        stats.sigmaClean = max(sqrt(stats.residSq), kMinSigma);
        stats.covCross = float3(acc.wXY * invTotalWeight - mean3.x * mean3.y,
                                acc.wXZ * invTotalWeight - mean3.x * mean3.z,
                                acc.wYZ * invTotalWeight - mean3.y * mean3.z);
    }

    return stats;
}

// ============================================================================
// THE RECORD'S DOF
// ============================================================================
float StudentEffectiveDof(float recordAge, float sampleDof)
{
    float decayPrior = exp2(recordAge * kClipStudentDecay);   // (1-rho)^{2t}
    float varFrac    = kClipSigmaEmaRate / (max(sampleDof, 1.0) * (2.0 - kClipSigmaEmaRate));
    return 1.0 / (decayPrior / kClipStudentPriorDof + varFrac * (1.0 - decayPrior));
}

// ============================================================================
// THE CORRELATION SHRINK (F8: the MSE-optimal linear shrink)
// ============================================================================
float ShrinkCorrelation(float r, float nu)
{
    float r2 = clamp(r * r, 0.0, 0.9025);
    float lambda = r2 / max(r2 + (1.0 - r2) * (1.0 - r2) / max(nu, 1.0), 1e-6);
    return r * lambda;
}

void ShrunkenCorrelations(ColorNeighborhoodStats stats, out float rXY, out float rXZ, out float rYZ)
{
    // The correlation dof from the W2 Kish count: the moments these
    // correlations are computed FROM are w2-weighted, so their own
    // effective size is the honest dof base -- the w3 sandwich count is
    // the information-side quantity and understates n_eff here. nu =
    // n_eff(w2) - 3 is the Fisher-z form EXACTLY: for a correlation from
    // n samples, Var(r) ~= (1-r^2)^2/(n-3) -- the -3 is already in the
    // call, not a large-n limit left un-corrected (audited: do not "fix"
    // it again).
    float nuCorr = max(1.0 / max(stats.invNeffW2, 1e-3) - 3.0, 1.0);
    rXY = clamp(ShrinkCorrelation(stats.covCross.x / max(stats.sigmaClean.x * stats.sigmaClean.y, kEpsilon), nuCorr), -0.95, 0.95);
    rXZ = clamp(ShrinkCorrelation(stats.covCross.y / max(stats.sigmaClean.x * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
    rYZ = clamp(ShrinkCorrelation(stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
}

// ============================================================================
// THE SHARED CORRELATION METRIC'S QUADRATIC FORM
// ============================================================================
// y' R^-1 y at unit scale, R = [[1, rXY, rXZ], [rXY, 1, rYZ], [rXZ, rYZ, 1]],
// with the inverse passed as its symmetric cofactors over detR
// (rC01 = (rXZ*rYZ - rXY)/detR etc). Standardizing a deviation by its
// per-channel scales and applying this is EXACTLY the full metric
// diag(v) R diag(v) -- one implementation serves the hard test, the soft
// clip's null, its 8 neighbor components and its temporal tail.
float QuadFormUnitR(float3 y, float rC00, float rC01, float rC02, float rC11, float rC12, float rC22)
{
    return rC00 * y.x * y.x + rC11 * y.y * y.y + rC22 * y.z * y.z
         + 2.0 * (rC01 * y.x * y.y + rC02 * y.x * y.z + rC12 * y.y * y.z);
}

// ============================================================================
// THE SHARED CORRELATION METRIC -- built ONCE per pixel (taa.fx), consumed
// by WhitenedInnovSq AND the gate. R^-1 at unit scale as symmetric
// cofactors over detR; degenerate detR (|R| ~ 0 under heavy shrink) keeps
// the diagonal fallback semantics of both former consumers.
// ============================================================================
struct SharedCorrelationMetric
{
    bool  full;
    float detR;
    float c00, c01, c02, c11, c12, c22;   // R^-1 cofactors / detR
};

SharedCorrelationMetric BuildSharedCorrelationMetric(float rXY, float rXZ, float rYZ)
{
    SharedCorrelationMetric m;
    float detR = 1.0 + 2.0 * rXY * rXZ * rYZ - rXY * rXY - rXZ * rXZ - rYZ * rYZ;
    m.full = (detR > 1e-6);
    m.detR = detR;
    float invDetR = m.full ? (1.0 / detR) : 0.0;
    m.c00 = (1.0 - rYZ * rYZ) * invDetR;
    m.c01 = (rXZ * rYZ - rXY) * invDetR;
    m.c02 = (rXY * rYZ - rXZ) * invDetR;
    m.c11 = (1.0 - rXZ * rXZ) * invDetR;
    m.c12 = (rXY * rXZ - rYZ) * invDetR;
    m.c22 = (1.0 - rXY * rXY) * invDetR;
    return m;
}

// ============================================================================
// THE ANISOTROPIC SCALE-MIXTURE (STUDENT) RADIAL TAIL:
// log2 P(|miss| > r) at unequal x/y scatter AND an uncertain scale.
// ----------------------------------------------------------------------------
// MODEL: the miss scale is a point estimate; the honest law marginalizes
// an Inv-chi^2(nuMiss) scale mixture, under which the quadratic form
// r^2 = sx^2 z1^2 + sy^2 z2^2 maps to
//     r^2  ~  delta + gammaMean * F(2*alpha, nuMiss),
// with the Pearson-III match carrying the ANISOTROPY:
//     kap1 = m = sx^2 + sy^2,  kap2 = 2(sx^4 + sy^4),  kap3 = 8(sx^6 + sy^6)
//     alpha = kap1^2 / kap2  in [1/2, 1],  gammaMean = 2 kap2^2 / kap3,
//     delta = kap1 - gammaMean >= 0
// (alpha by AM-GM, delta by Cauchy-Schwarz; both are equalities at sx = sy,
// and delta = 0 also at full anisotropy -- the chi^2_1 family is exactly
// gamma). u = (r^2 - delta)/theta, theta = gammaMean/alpha.
//
// BOTH ENDPOINT FAMILIES ARE EXACT:
//   * sx = sy, every nu: the F(2, nu) survival is exactly
//     (1 + 2u/nu)^{-nu/2} -- the isotropic bivariate Student law, verbatim.
//   * nu -> inf, every anisotropy: F(2 alpha, inf) = chi^2_{2 alpha}/(2
//     alpha) -- EXACTLY the Pearson-III model, and the leading
//     beta-integral term below degenerates to its uniform tail asymptotic
//     term-for-term (the cancellation is carried by rLog's log2(w)
//     divergence against term 2's).
// The evaluated form is the left-tail (small-z) leading term of
//     S = I_{1/(1+x')}(nu/2, alpha),   x' = 2u/nu:
//     log2 S ~ -(nu/2) log2(1+x') + (alpha-1) log2(x'/(1+x'))
//              - log2[ (nu/2) B(nu/2, alpha) ].
// The polynomial rate nu/2 is exact for every alpha -- the heavy tail
// comes from small denominator draws, which is the point: a point-Gaussian
// prior collapses artificially hard at tight sigma, while the dof
// bookkeeping (taa.fx) keeps clean STATIC content collapsing anyway (the
// floors are conventions -> large nu).
//
// Numerics: the gamma-ratio correction
//     R(w, alpha) = log2[Gamma(w+1)/Gamma(w+alpha)]
//                ~ (1-alpha) log2(w) + log2(1 + alpha(1-alpha)/(2w))
// (the Bernoulli first correction; error < 0.01 bit over w in [1, 128],
// alpha in [0.5, 1]). Precision class: exact at alpha = 1; the leading
// term's first correction is O(z (1-alpha)) -- the same ~+/-20% through
// the hot transition class as the underlying Pearson-III match. lnGamma
// on [1/2, 1] is the quadratic through (1/2, 0.5724), (3/4, 0.2033),
// (1, 0): max error ~5e-3 in the log. The Pearson-III shift delta is
// carried UNSCALED (the honest mixture scales it by tau as well; delta =
// 0 at both axis endpoints and <= 7% of kap1 in the mixed middle --
// inside the law's accuracy class).
// ============================================================================
float Log2RadialTail2(float rSq, float sx, float sy, float nuMiss)
{
    float a = sx * sx;
    float b = sy * sy;
    float m = a + b;
    float kap2 = 2.0 * (a * a + b * b);
    float kap3 = 8.0 * (a * a * a + b * b * b);
    float alpha = (m * m) / max(kap2, 1e-20);                      // [0.5, 1]
    float gammaMean = (2.0 * kap2 * kap2) / max(kap3, 1e-30);
    float delta = max(m - gammaMean, 0.0);
    float u = max((rSq - delta) * alpha / max(gammaMean, 1e-20), 0.0);  // (r^2 - delta)/theta

    float w = max(nuMiss, 2.0) * 0.5;                              // nu/2
    float xp = max(u / w, 1e-30);                                  // x' = 2u/nu
    float xp1 = 1.0 + xp;
    float xq = max(xp / xp1, 1e-10);                               // 1 - z

    float oneMinusA = 1.0 - alpha;
    float lnGamma = 1.3264 * oneMinusA * oneMinusA + 0.4815 * oneMinusA;   // lnGamma(alpha)
    float wSafe = max(w, 1.0);
    float rLog = oneMinusA * log2(wSafe)
               + log2(1.0 + alpha * oneMinusA / (2.0 * wSafe));    // log2[Gamma(w+1)/Gamma(w+alpha)]

    float lSurv = -w * log2(xp1) + (alpha - 1.0) * log2(xq) - rLog - lnGamma * kLog2E;
    // The min() pins S <= 0.9997: near u = 0 the leading term can drift a
    // hair above S = 1, and S > 1 would explode the prior odds through
    // the (1 - p) term.
    return min(lSurv, -0.0005);
}

// ============================================================================
// THE TRUNCATED CELL MOMENTS (the H1 components' within-cell gradient
// term, exact form)
// ----------------------------------------------------------------------------
// Per axis, the second moment of the miss about the cell center, GIVEN the
// miss landed in that cell -- the miss model's own truncated Gaussian, not
// the uniform cell the old (gx^2+gy^2)/12 assumed everywhere. Two cell
// geometries per axis (X ~ N(0, s^2), h = 0.5/s):
//   * the CENTER cell ([-0.5, 0.5], contains the mode):
//       P_c = 2*Phi(h) - 1 = erf(h/sqrt(2))
//       E_c = s^2 [1 - 2 h phi(h) / P_c]         (the mean is 0 by evenness)
//   * the EDGE cell ([0.5, 1.5]; equal to [-1.5,-0.5] by evenness):
//       P_e = Phi(3h) - Phi(h)
//       E_e = s^2 [1 - (3h phi(3h) - h phi(h)) / P_e]
//             - 2 s (phi(h) - phi(3h)) / P_e + 1.
// Limits: s -> 0 gives (s^2, 0.25 - 2 s^2 + 8 s^4) -- the center cell IS
// the scale, the edge cell hangs at the inner boundary; s -> inf gives
// (1/12, 1/12) -- the uniform. The erf is A&S 7.1.26 (the SAME 5-term
// rational the host's erfApprox uses, |eps| <= 1.5e-7 absolute); the two
// exp(-x^2) factors are shared with the phi evaluations (e^{-h^2/2} and
// its 9th power, the latter by repeated squaring). Float safety:
// h <= 4 (s >= kMinMissSigma = 0.125) keeps every term normal; the
// 1 - 2h phi/P_c cancellation at wide s resolves h^2/3 ~ 1e-4 above the
// float32 noise floor for any real miss scale (s <= ~10).
// ============================================================================
void TruncatedCellMoments(float s, out float mCenter, out float mEdge)
{
    float h  = 0.5 / max(s, 1e-4);
    float h2 = h * h;

    float e1 = exp2(-0.5 * kLog2E * h2);        // e^{-h^2/2}
    float e2 = e1 * e1;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e9 = e8 * e1;                          // e^{-(3h)^2/2}

    float cNorm = 0.3989422804014327;            // 1/sqrt(2 pi)
    float phi1 = cNorm * e1;                     // phi(h)
    float phi3 = cNorm * e9;                     // phi(3h)

    // erf(x) = 1 - t(a1 + t(a2 + t(a3 + t(a4 + t a5)))) e^{-x^2},
    // t = 1/(1 + p x); the e^{-x^2} factors are e1 / e9 (x = h/sqrt(2)).
    float x1 = h * 0.70710678;
    float x2 = 3.0 * x1;
    float t1 = 1.0 / (1.0 + 0.3275911 * x1);
    float t2 = 1.0 / (1.0 + 0.3275911 * x2);
    float q1 = t1 * (0.254829592 + t1 * (-0.284496736 + t1 * (1.421413741 + t1 * (-1.453152027 + t1 * 1.061405429))));
    float q2 = t2 * (0.254829592 + t2 * (-0.284496736 + t2 * (1.421413741 + t2 * (-1.453152027 + t2 * 1.061405429))));
    float erf1 = 1.0 - q1 * e1;                  // erf(h/sqrt(2))
    float erf2 = 1.0 - q2 * e9;                  // erf(3h/sqrt(2))

    float pC = max(erf1, 1e-12);                 // P(-0.5 < X < 0.5)
    float pE = max(0.5 * (erf2 - erf1), 1e-12);  // P(0.5 < X < 1.5)

    mCenter = s * s * max(1.0 - 2.0 * h * phi1 / pC, 0.0);

    float m1 = s * (phi1 - phi3) / pE;                                   // E[X | edge]
    float m2 = s * s * max(1.0 - (3.0 * h * phi3 - h * phi1) / pE, 0.0); // E[X^2 | edge]
    mEdge = max(m2 - 2.0 * m1 + 1.0, 0.0);
}

// ============================================================================
// CLIP-STATE TRANSPORT (tag 101)
// ============================================================================
static const float kVarHCodeLogMin  = -6.0;
static const float kVarHCodeLogSpan = 9.0;

float EncodeVarHRatio(float varH, float recordSq)
{
    // Total: NaN would reach uint() in the packer (f2u of NaN is undefined
    // on every backend). The scrubbed fallback (0) is the conservative
    // side -- a garbage input yields a TIGHT gate, not the code-range
    // ceiling.
    varH     = max(ScrubF(varH, 0.0), 0.0);
    recordSq = max(ScrubF(recordSq, kEpsilon), kEpsilon);
    float ratio = varH / recordSq;
    float code  = round((log2(max(ratio, 1e-8)) - kVarHCodeLogMin) * (4095.0 / kVarHCodeLogSpan));
    return clamp(code, 0.0, 4095.0);
}

float DecodeVarHRatio(float code, float recordSq)
{
    float ratio = exp2(code * (kVarHCodeLogSpan / 4095.0) + kVarHCodeLogMin);
    return ratio * recordSq;
}

float PackClipStateAlpha(bool revoked, float sigma, float varHRatioCode, float age,
                         float splitCode)
{
    // Total: every uint() operand is scrubbed to its field's range first
    // (f2u of NaN is undefined on every backend; NaN in any transported
    // field would persist through the history forever).
    varHRatioCode = min(ScrubF(max(varHRatioCode, 0.0), 0.0), 4095.0);
    age           = min(ScrubF(max(age, 0.0), 0.0), kClipMaxAge);
    splitCode     = min(ScrubF(max(splitCode, 0.0), 0.0), 63.0);
    uint u = 0x50000000u
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x3Fu)    << 22)
           | ((uint(varHRatioCode) & 0xFFFu)                    << 10)
           | ((uint(age) & 0xFu)                                << 6)
           |  (uint(splitCode) & 0x3Fu);
    return asfloat(revoked ? (u | 0x80000000u) : u);
}

void DecodeClipState(float alphaValue, out float sigmaSq, out float varHSq,
                     out float recordAge, out float splitFrac)
{
    sigmaSq = -1.0;
    varHSq = 0.0;
    recordAge = 0.0;
    splitFrac = 0.0;

    uint u   = asuint(alphaValue);
    uint tag = (u >> 28) & 0x7u;
    uint code;
    float varHCode = 1820.0;   // ratio ~0.25: the near-stationary default
    if      (tag == 0x5u)
    {
        code = (u >> 22) & 0x3Fu;
        varHCode = (float)((u >> 10) & 0xFFFu);
        recordAge = (float)((u >> 6) & 0xFu);
        splitFrac = ((float)(u & 0x3Fu) - 32.0) * 0.03125;
    }
    else if (tag == 0x1u)
    {
        // The debug payload carries the sigma only: the split reads neutral
        // while a debug mode is active and re-converges in ~5 frames.
        code = u & 0x3Fu;
    }
    else
    {
        return;
    }

    if (code == 0u) return;
    float s = ClipUnpackSigmaCode(code);
    sigmaSq = s * s;
    varHSq = DecodeVarHRatio(varHCode, sigmaSq);
}

// ============================================================================
// THE RECORD SEED (the model's per-channel E[i^2] -- the COLD estimate.
// The transported record means sPer; the seed models sPer + varH, and the
// varH share (~2%) is absorbed by the EMA within a few frames.)
// ============================================================================
float3 SeedRecordVarX(ColorNeighborhoodStats stats)
{
    float3 phase = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                 + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                 + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy);
    return phase + stats.residSq;
}

float SeedRecordTotalSq(ColorNeighborhoodStats stats)
{
    return max(dot(SeedRecordVarX(stats), float3(1.0, 1.0, 1.0)), kClipSigmaRecordFloorSq);
}

// ============================================================================
// THE RECORD'S PER-CHANNEL SPLIT (the transported record IS sPer)
// ============================================================================
struct RecordVarianceParts
{
    float3 ePer;           // per-channel PRE-update E[i_k^2] = sPer + varH
    float3 varHPer;        // per-channel accumulator variance (READ semantics)
    float3 sPer;           // the tap likelihood's variance (the record itself)
    float  spatialPriorSq; // the content-scale reference (the winsor anchor)
};

RecordVarianceParts SplitRecordVariance(
    ColorNeighborhoodStats stats,
    float sPerSqTotal, float varHTotal, float nuRecord)
{
    RecordVarianceParts p;

    // varX is PSD by construction (the jitter phase covariance is PSD and
    // residSq >= 0), so the max() clamps nothing on real input and tot
    // below is EXACTLY SeedRecordTotalSq's inner dot -- computed here once
    // (the cold paths in taa.fx still call SeedRecordTotalSq).
    float3 varX = max(SeedRecordVarX(stats), float3(0.0, 0.0, 0.0));
    float  tot  = dot(varX, float3(1.0, 1.0, 1.0));
    float3 share = (tot > 1e-10)
        ? varX / tot
        : stats.sigmaClean * stats.sigmaClean / max(dot(stats.sigmaClean, stats.sigmaClean), kEpsilon);

    float spatialPriorSq = max(tot, kClipSigmaRecordFloorSq);

    // THE ARMING CAP at the record's LIVE dof (the exact chi^2_nu(0.999)/nu
    // bound, host-fit) -- binds sPer AND the transported VarH share.
    float nuCap = max(nuRecord, 2.0);
    float armingCap = 1.0 + taaArmingA / sqrt(nuCap) + taaArmingB / nuCap;

    float sPerSq = min(max(sPerSqTotal, 0.0), armingCap * spatialPriorSq);
    float varHSq = min(max(varHTotal,  0.0), armingCap * spatialPriorSq);
    p.sPer    = max(share * sPerSq, float3(kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq));
    p.varHPer = share * varHSq;
    p.ePer    = p.sPer + p.varHPer;
    p.spatialPriorSq = spatialPriorSq;
    return p;
}

// The whitened innovation on the READ-side denominators + the shrunk full
// metric. y'(D R D)^-1 y == (y./s)' R^-1 (y./s) EXACTLY (D^-1 R^-1 D^-1),
// and det(D R D) = detR * (v.x*v.y*v.z) -- the original's own degenerate
// test on the scaled determinant, preserved in form. The diagonal fallback
// is bit-identical to the original's.
float WhitenedInnovSq(float3 innov, RecordVarianceParts p, float3 transportVar,
                      SharedCorrelationMetric m)
{
    float3 v = max(p.ePer + transportVar, float3(1e-10, 1e-10, 1e-10));
    float3 y = innov / sqrt(v);
    if (m.full && (m.detR * (v.x * v.y * v.z)) > 1e-12)
        return QuadFormUnitR(y, m.c00, m.c01, m.c02, m.c11, m.c12, m.c22);
    return dot(y * y, float3(1.0, 1.0, 1.0));
}

// ============================================================================
// THE WINSOR'S CENSORING POINT
// ----------------------------------------------------------------------------
// The ingest is a 3-channel SUM of correlated squared devs. For x ~ N(0, S),
// the sum of squares has Satterthwaite dof
//     nuStar = (tr S)^2 / tr(S^2)
//            = (sum v)^2 / (sum v^2 + 2*sum rho^2 v_j v_k)   in [1, 3]
// -- 1 on luminance-dominated YCoCg deviations (all power in Y), 3 only for
// exchangeable uncorrelated channels. NOTE the rho^2: the covariance of
// SQUARES carries the SQUARED correlation (Cov(x_j^2, x_k^2) = 2 S_jk^2) --
// not rho; do not "simplify" it. The per-channel null variances are the tap
// dev's own (vPrior + sPer - 2*cPT -- the same identity the ingest
// de-biases with, replicated bit-for-bit from the gate's qTapDen), so the
// censoring point is the honest 0.1% quantile of the actual null:
//     cap = Sum sPer + Q_0.999(null) - kap1
// with Q via the PEARSON-III (3-cumulant) match -- the sum's skewness,
// kap3 = 8*tr(Sigma^3), is invisible to the 2-cumulant Satterthwaite form,
// which runs the 0.1% point low on mixed variance/correlation structures
// (a low cap over-censors legitimate variance). The matched law is a
// SHIFTED gamma,
//     delta = kap1 - 2 kap2^2 / kap3 >= 0   (Cauchy-Schwarz on the
//              eigenvalues; zero exactly on the chi^2 family),
//     alpha = 4 kap2^3 / kap3^2  in [0.5, 1.5]  (skew <= 2*sqrt(2)
//              for any PSD Sigma),
// and the 0.999 quantile is delta + (2 kap2^2 / kap3) * WH(alpha)^3. The
// Wilson-Hilferty quantile of the gamma part runs ~1-3% high -- the
// conservative side for a censor.
//
// The cap anchors at RECORD scale WITH the mean shift: the ingest's null
// is the SHIFTED sum sIngest = Sum[dev^2] - (kap1 - Sum sPer) --
// E[sIngest] = Sum sPer exactly (the de-bias identity) -- so its honest
// 0.1% point is the sum-of-squares quantile minus the shift. Documented
// cost: a sagged record under rising content recovers through the EMA at
// ~10-20x per frame (the quantile excess) instead of one jump to content
// scale -- ~2-3 frames in the extreme, still bounded above by the ARMING
// cap (which keeps its content anchor: it is the stale-high GUARD, and a
// record-anchored bound is vacuous -- cap = f(record) always passes).
// ============================================================================
float WinsorCapNuStar(
    RecordVarianceParts parts,
    ColorNeighborhoodStats stats,
    float  splitFrac,
    float  rXY, float rXZ, float rYZ)
{
    float3 tauB2  = max(splitFrac, 0.0) * stats.residSq;
    float3 vPrior = tauB2 + parts.sPer * stats.evalVar;
    float3 qDen   = max(vPrior + parts.sPer - 2.0 * stats.priorTapCov,
                        0.25 * (vPrior + parts.sPer));
    float vx = qDen.x, vy = qDen.y, vz = qDen.z;
    float vSum   = vx + vy + vz;
    float vSqSum = vx * vx + vy * vy + vz * vz
                 + 2.0 * (rXY * rXY * vx * vy + rXZ * rXZ * vx * vz + rYZ * rYZ * vy * vz);
    float vXY3 = rXY * rXY * vx * vy * (vx + vy)
               + rXZ * rXZ * vx * vz * (vx + vz)
               + rYZ * rYZ * vy * vz * (vy + vz);
    float trSigma3 = vx * vx * vx + vy * vy * vy + vz * vz * vz
                   + 3.0 * vXY3
                   + 6.0 * (rXY * rXZ * rYZ) * vx * vy * vz;
    float kap1 = vSum;
    float kap2 = 2.0 * vSqSum;
    float kap3 = 8.0 * trSigma3;
    float gammaMean = 2.0 * kap2 * kap2 / max(kap3, 1e-30);      // alpha * theta
    float delta = max(kap1 - gammaMean, 0.0);
    float alphaW = 4.0 * (kap2 * kap2 * kap2) / max(kap3 * kap3, 1e-60);
    // Wilson-Hilferty at z_0.999 = 3.090232306167813
    float wh = 1.0 - 1.0 / (9.0 * alphaW) + 3.090232306167813 * sqrt(1.0 / (9.0 * alphaW));
    float chiQ = delta + gammaMean * wh * wh * wh;
    float muIng = parts.sPer.x + parts.sPer.y + parts.sPer.z;
    return max(muIng + chiQ - kap1, kClipSigmaRecordFloorSq);
}

// ============================================================================
// THE MAHALANOBIS GATE
// ============================================================================
struct ClipGateResult
{
    float3 clippedColorSpace;
    float3 center;          // the gate's content reference
    float  tGate;
    float3 rawGateVar;      // the UN-INFLATED gate variance (the varH seed)
    float  mdd;             // the realized Mahalanobis statistic (mode 13)
};

ClipGateResult ClipHistoryToStatisticGate(
    float3 historyColorSpace,
    float3 currentColorSpace,
    ColorNeighborhoodStats stats,
    RecordVarianceParts parts,
    bool   recordLive,
    float  recordAge,
    float  splitFrac,
    float4 missModel,                   // xy = the miss sigmas (texels),
                                        // z = the scale's dof, w = the
                                        // conditional rate boost (log2) --
                                        // see taa.fx's miss-model block
    float3 accumulatedNeighbors[8],
    SharedCorrelationMetric sharedMetric)   // R^-1 at unit scale, built ONCE
                                            // by the caller (taa.fx) -- the
                                            // whitened innovation shares it
{
    ClipGateResult r;
    r.center = stats.plane0;
    r.tGate = 1.0;
    r.rawGateVar = float3(kMinSigma * kMinSigma, kMinSigma * kMinSigma, kMinSigma * kMinSigma);
    r.mdd = 0.0;

    float  minGateVar = kMinSigma * kMinSigma;
    float3 gateVar;
    float3 gateVarBase = float3(minGateVar, minGateVar, minGateVar);
    float3 center = stats.plane0;
    float  nuPost = 4.0;

    if (recordLive)
    {
        // ============ THE HIERARCHICAL NULL ============
        float nuRecord = StudentEffectiveDof(recordAge, kRecordIngestDof);
        float nuRecordSafe = max(nuRecord, 2.0);

        float3 sPer    = parts.sPer;
        float3 varHPer = parts.varHPer;

        // THE STRUCTURE SPLIT: the transported, temporally stabilized
        // fraction times the live spatial residual. The clamp acts on the
        // stable side.
        float3 tauB2 = max(splitFrac, 0.0) * stats.residSq;

        // THE FIT'S NOISE TERM (exact): sPer * Sum c_k^2 -- the evaluation
        // functional's exact variance; the structure is charged once.
        float3 vFit   = sPer * stats.evalVar;
        float3 vPrior = tauB2 + vFit;

        float3 x0c = currentColorSpace - stats.phaseShift;
        float3 dev = x0c - stats.plane0;
        // The exact quadratic form of dev: Var(dev) = sPer + vPrior - 2*cPT
        // (the same identity the ingest de-biases with). The floor guards
        // the perfect-correlation limit (the CS bound guarantees only >= 0).
        float3 qTapDen = max(vPrior + sPer - 2.0 * stats.priorTapCov,
                             0.25 * (vPrior + sPer));
        float3 qTap = (dev * dev) / max(qTapDen, float3(1e-10, 1e-10, 1e-10));
        // F5: the exact one-step t weight (Lange-Little-Taylor).
        float3 sPerEff = sPer * (nuRecordSafe + qTap) / (nuRecordSafe + 1.0);

        float3 wTap = vPrior / max(vPrior + sPerEff, float3(1e-10, 1e-10, 1e-10));
        center = stats.plane0 + wTap * dev;

        // F1: the shared-sample covariance, charged into the prior-side share.
        float3 cPT = clamp(stats.priorTapCov * (sPerEff / max(sPer, float3(1e-10, 1e-10, 1e-10))),
                           float3(0.0, 0.0, 0.0),
                           sqrt(vPrior * sPerEff));
        float3 A = varHPer + wTap * wTap * sPerEff;
        float3 B = (1.0 - wTap) * (1.0 - wTap) * vPrior
                 + 2.0 * wTap * (1.0 - wTap) * cPT;
        gateVarBase = A + B;

        // ---- the Satterthwaite dof ----
        // tauB2's per-frame noise is the live residSq's (f^2-scaled through
        // the reconstruction), charged at residSq's OWN exact dof (Sum
        // w(1-h) on the robust weights). The EMA'd fraction's residual
        // noise is NOT charged: it manifests as the clamp's positive-part
        // mean, already carried in vPrior, and charging it again would
        // double-count. (If mode 13 shows over-engagement on static smooth
        // content, add tauB2^2 * kSplitEmaRate/(2-kSplitEmaRate) /
        // max(splitFrac,0.04)^2 -- the documented revert.)
        float3 denomTau = tauB2 * tauB2 / max(stats.residDof, 1.0);
        float3 nuPrior = vPrior * vPrior
                       / max(denomTau + vFit * vFit / nuRecordSafe, float3(1e-10, 1e-10, 1e-10));
        nuPrior = max(nuPrior, float3(2.0, 2.0, 2.0));

        float3 nuFlat = (A + B) * (A + B)
                      / max(A * A / nuRecordSafe + B * B / nuPrior, float3(1e-10, 1e-10, 1e-10));
        nuFlat = max(nuFlat, float3(2.0, 2.0, 2.0));

        // The host fit is a Pade (2,1) rational in x = 1/nu,
        //     S(nu) = (1 + A*x + B*x^2) / (1 + C*x),
        // weighted-least-squares fit to the exact F(3,nu) quantile relation
        // over a dense anchor set (taa.lua -- the same f3nuQuantile
        // machinery). The denominator is Lua-verified positive over
        // x in [0, 0.5] (nu >= 2 everywhere here); the max() is a
        // NaN-safety floor that never binds on a valid fit.
        float3 xNu = 1.0 / nuFlat;
        float3 studentS = (1.0 + taaStudentA * xNu + taaStudentB * xNu * xNu)
                        / max(1.0 + taaStudentC * xNu, 0.0625);
        gateVar = max(gateVarBase * studentS, float3(minGateVar, minGateVar, minGateVar));

        float gSum = gateVar.x + gateVar.y + gateVar.z;
        nuPost = max(dot(nuFlat, gateVar) / max(gSum, 1e-10), 2.0);
    }
    else
    {
        // ---- the cold path: the no-record LIMIT of the same model, not a
        // separate policy. With no record, stability cannot be certified:
        // all residual is per-sample noise (tauB2 = 0, sPer = the seed's
        // full E[i^2] model), so vPrior = seed * Sum c_k^2 with the phase
        // folded in -- the form below. At straddled pixels the step sits
        // in residSq, so the cold width is step-inflated there. ----
        float3 phase = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                     + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                     + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy);
        gateVarBase = stats.evalVar * (phase + stats.residSq);
        gateVar = gateVarBase;
        center = stats.plane0;
        nuPost = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
    }

    // The transport scatter (added post-Studentization by design: it is a
    // MEASURED component, not an estimated one; the record measures the
    // realized innovation within its EMA lag).
    float3 transportTerm = stats.transportVar;
    gateVar += transportTerm;
    r.rawGateVar = max(gateVarBase + transportTerm, float3(minGateVar, minGateVar, minGateVar));

    float3 dCorr = historyColorSpace - center;
    r.center = center;

    // taaChromaVarianceMod is the one DELIBERATELY uncalibrated knob: scaling
    // the metric post-Studentization turns the statistic into a weighted
    // chi^2, so the F(3,nu) coverage calibration holds exactly only at 1.0.
    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    // ---- THE SHARED METRIC ----
    // One correlation matrix R (the caller's shrunk correlations -- the
    // same values the spike, the winsor and the corroboration test
    // consume), one hoisted R^-1 at unit scale, built ONCE by the caller
    // (taa.fx's BuildSharedCorrelationMetric -- the whitened innovation
    // uses the same inverse). Every quadratic form in this gate -- the
    // hard test's mdd, the soft clip's null, its 8 neighbor components,
    // its temporal tail -- standardizes its deviation by the hypothesis'
    // per-channel scales and runs through this R^-1: exactly the full
    // metric diag(v) R diag(v), and the SAME channel coupling on both
    // sides of the soft clip's likelihood ratio.
    bool  rFull = sharedMetric.full;
    float rC00 = sharedMetric.c00;
    float rC01 = sharedMetric.c01;
    float rC02 = sharedMetric.c02;
    float rC11 = sharedMetric.c11;
    float rC12 = sharedMetric.c12;
    float rC22 = sharedMetric.c22;

    // The hard test's statistic through the shared metric.
    float3 vMet = max(gateVar * chromaScale * chromaScale, float3(minGateVar, minGateVar, minGateVar));
    float mdd = rFull
        ? QuadFormUnitR(dCorr / sqrt(vMet), rC00, rC01, rC02, rC11, rC12, rC22)
        : dot(dCorr * dCorr, float3(1.0, 1.0, 1.0) / vMet);
    r.mdd = mdd;

    float tHard = (mdd > 1e-20) ? min(chi / sqrt(mdd), 1.0) : 1.0;

    bool posteriorPath = (taaSoftClip > 0.001) && (mdd > chi * chi);
    if (posteriorPath)
    {
        // ============ THE GHOST POSTERIOR (the measured miss model) =====
        // H0: the hierarchical null at its honest UN-INFLATED predictive
        // variance, on the SAME shrunk full metric as the hard test with
        // the exact relative log-det normalizer. H1: the landing-miss
        // hypothesis -- a PROPER mixture of the 8 NEIGHBOR TEXELS'
        // accumulated content (what a landing-miss ghost holds) plus one
        // temporal tail (the old content absent from the 9-tap sample).
        // THE CENTER TAP IS NOT A COMPONENT -- it IS the null's likelihood.
        // Every weight is generated by ONE miss law: the elliptical
        // Student at the measured scale dof -- the prior odds from its
        // radial tail at r0 (plus the measured conditional rate), the
        // neighbor masses from its density at the texel offsets, the tail
        // mass from its tail at r1 -- all renormalized to the conditional
        // distribution of the miss GIVEN a ghost.
        //
        // THE WHOLE POSTERIOR RUNS IN BASE-2 LOGS: every term is
        // homogeneous in the log base, so scaling the entire log-odds by
        // log2(e) converts every ln to log2 with ALL COEFFICIENTS
        // UNCHANGED -- and log2/exp2 are the native opcodes.
        //
        // The neighbor masses carry the two-term Euler-Maclaurin cell
        // correction, and the components and the tail run through the
        // SHARED correlation metric above -- so the |R| log-det factors
        // cancel EXACTLY between the hypotheses (both sides keep only
        // their diagonal log-dets).
        float3 nullScale = max((gateVarBase + transportTerm) * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));

        // ---- the H1 component variance: the accumulated read vs the
        //      accumulated texel -- Var = Vr + Vs - 2*Cov, all from the
        //      transported varH (Vr, READ semantics) and the host's
        //      spectral chain constants folded into taaAccNoiseFactor =
        //      (D + 1 - 2*kappa)/D (Cauchy-Schwarz on the shared field
        //      keeps the factor >= 0, the clamp is numerical). PLUS the
        //      within-cell gradient term: given the miss landed in texel
        //      j, the read center sits somewhere in that texel, so the
        //      term is E[(g.(delta - o_j))^2] per channel -- the miss
        //      model's own TRUNCATED-cell second moment (per axis, per
        //      cell geometry; TruncatedCellMoments), evaluated in the
        //      Gaussian core (the mixture's within-cell moments differ
        //      only at O(x'/nu) -- second-order next to the mass and tail
        //      Studentization, accepted). At tight scatter the center
        //      column overprices ~5x (s^2 vs 1/12) and the edge cells
        //      underprice by the missing conditional-mean offset (-> 0.25
        //      as s -> 0) -- and the edge cells are exactly what the
        //      ghost-conditional mixture concentrates on. NO sPer and NO
        //      transport term (the masses charge the miss; adding either
        //      would double-count). The cold path keeps its stand-in:
        //      with no record there is no varH to price the structure, and
        //      the seed's gateVarBase is the honest substitute. ----
        float3 vCompCE, vCompEC, vCompEE;
        {
            float mCX, mEX, mCY, mEY;
            TruncatedCellMoments(max(missModel.x, kMinMissSigma), mCX, mEX);
            TruncatedCellMoments(max(missModel.y, kMinMissSigma), mCY, mEY);
            float3 gX2 = stats.gradLSx * stats.gradLSx;
            float3 gY2 = stats.gradLSy * stats.gradLSy;
            float3 vBase = recordLive
                ? max(parts.varHPer, float3(1e-10, 1e-10, 1e-10)) * max(taaAccNoiseFactor, 0.02)
                : (gateVarBase + transportTerm);
            vCompCE = vBase;
            vCompEC = vBase;
            vCompEE = vBase;
            if (recordLive)
            {
                vCompCE += gX2 * mCX + gY2 * mEY;   // cells (0,+-1): center column x, edge y
                vCompEC += gX2 * mEX + gY2 * mCY;   // cells (+-1,0): edge x, center row y
                vCompEE += gX2 * mEX + gY2 * mEY;   // corners: edge, edge
            }
            float3 cs2 = chromaScale * chromaScale;
            float3 vFloor = float3(minGateVar, minGateVar, minGateVar);
            vCompCE = max(vCompCE * cs2, vFloor);
            vCompEC = max(vCompEC * cs2, vFloor);
            vCompEE = max(vCompEE * cs2, vFloor);
        }

        float nu = max(nuPost, 2.0);
        float kT = 0.5 * (nu + 3.0);

        // The component log-dets: one per cell-geometry variant (the
        // within-cell term is per-cell), hoisted out of the unrolled
        // mixture loop. Diagonal only -- the |R| factor cancels against
        // the null's.
        float logDetCE2 = 0.5 * (log2(vCompCE.x) + log2(vCompCE.y) + log2(vCompCE.z));
        float logDetEC2 = 0.5 * (log2(vCompEC.x) + log2(vCompEC.y) + log2(vCompEC.z));
        float logDetEE2 = 0.5 * (log2(vCompEE.x) + log2(vCompEE.y) + log2(vCompEE.z));
        float3 invCE     = float3(1.0, 1.0, 1.0) / vCompCE;
        float3 invEC     = float3(1.0, 1.0, 1.0) / vCompEC;
        float3 invEE     = float3(1.0, 1.0, 1.0) / vCompEE;
        float3 invSqrtCE = 1.0 / sqrt(vCompCE);
        float3 invSqrtEC = 1.0 / sqrt(vCompEC);
        float3 invSqrtEE = 1.0 / sqrt(vCompEE);

        // ---- the null: q0 through the shared R^-1; the DIAGONAL log-det --
        // |Sigma_0| = |nullScale| * |R| and every component's
        // |Sigma_j| = |vComp| * |R|: the |R| factors cancel in the LLR, so
        // BOTH sides keep only their diagonal log-dets.
        float logDetNull2 = log2(nullScale.x) + log2(nullScale.y) + log2(nullScale.z);
        float q0Comp = rFull
            ? QuadFormUnitR(dCorr / sqrt(nullScale), rC00, rC01, rC02, rC11, rC12, rC22)
            : dot(dCorr * dCorr, float3(1.0, 1.0, 1.0) / nullScale);

        // ---- the miss model ----
        // The core is the ELLIPTICAL STUDENT DENSITY at nuMiss -- and its
        // normalizer is EXACTLY the Gaussian's: for the per-axis scaled
        // elliptical t, Gamma((nu+2)/2) / (Gamma(nu/2) nu pi) = 1/(2 pi)
        // identically (Gamma(w+1) = w Gamma(w), w = nu/2), so logNormC2
        // below is the Gaussian's, unchanged. The core degenerates to the
        // Gaussian's per-axis -z^2/2 terms term-for-term as nu -> inf
        // (kMassT/nu -> log2(e)/2).
        float sx = max(missModel.x, kMinMissSigma);
        float sy = max(missModel.y, kMinMissSigma);
        float nuMiss = max(missModel.z, 2.0);
        float logNormC2 = -log2(6.2831853 * sx * sy);
        float kMassT = 0.5 * (nuMiss + 2.0);
        float invNuM = 1.0 / nuMiss;
        float invSx2 = 1.0 / (sx * sx);
        float invSy2 = 1.0 / (sy * sy);

        // The prior odds: P(|miss| > r0) / P(|miss| <= r0), the radial tail
        // of the Student mixture at the anisotropic scales, plus the
        // measured conditional rate boost. Collapses on clean content (the
        // soft clip switches itself off); runs hot on large measured
        // scatter (the history really is miss-contaminated). The dial is
        // kMissRadiusOdds, never a cap.
        float logPH1_2 = Log2RadialTail2(kMissRadiusOdds * kMissRadiusOdds, sx, sy, nuMiss);
        float logPriorOdds2 = max(logPH1_2, -100.0)
                            - log2(max(1.0 - exp2(logPH1_2), 1e-6))
                            + missModel.w;   // the conditional miss rate

        // The component masses: the mass of texel j is, per axis, the exact
        // Phi-difference's two-term Euler-Maclaurin midpoint expansion
        //     P = 2*delta*phi(z) * (1 + delta^2*(z^2-1)/6 + O(delta^4)),
        // delta = 0.5/sigma (the first term IS the density-times-area
        // form; the correction takes it under ~0.5% for sigma >= 0.5),
        // with the Student core in place of the Gaussian's per-axis
        // exponent. logSumM renormalizes the 9 into the conditional
        // distribution GIVEN a ghost.
        float lMass2[8];
        float lMassTail2;
        float logSumM2;
        {
            float hx2 = 0.25 * invSx2;   // (0.5/sigma_x)^2
            float hy2 = 0.25 * invSy2;
            float lmMax = -1e30;
            [unroll]
            for (int j = 1; j < 9; ++j)
            {
                float2 o = kOffsets3x3[j];
                float zx2 = (o.x * o.x) * invSx2;
                float zy2 = (o.y * o.y) * invSy2;
                // Clamped: below sigma ~ 0.29 the expansion's zero-z factor
                // would go negative -- a regime where the prior odds have
                // already collapsed, so the clamp only keeps the log
                // finite.
                float corrX = max(1.0 + hx2 * (zx2 - 1.0) * (1.0 / 6.0), kMinMassCorr);
                float corrY = max(1.0 + hy2 * (zy2 - 1.0) * (1.0 / 6.0), kMinMassCorr);
                lMass2[j - 1] = logNormC2 - kMassT * log2(1.0 + (zx2 + zy2) * invNuM)
                              + log2(corrX) + log2(corrY);
                lmMax = max(lmMax, lMass2[j - 1]);
            }
            // kTailChangeFloor = 0: the branch collapses at compile time
            // and the default build pays neither the exp2 nor the log2.
            lMassTail2 = (kTailChangeFloor > 0.0)
                ? log2(exp2(Log2RadialTail2(kMissRadiusTail * kMissRadiusTail, sx, sy, nuMiss)) + kTailChangeFloor)
                : Log2RadialTail2(kMissRadiusTail * kMissRadiusTail, sx, sy, nuMiss);
            lmMax = max(lmMax, lMassTail2);
            float mSum = 0.0;
            [unroll]
            for (int j1 = 0; j1 < 8; ++j1)
                mSum += exp2(lMass2[j1] - lmMax);
            mSum += exp2(lMassTail2 - lmMax);
            logSumM2 = lmMax + log2(max(mSum, 1e-10));
        }

        float lComp2[8];
        float lMax2 = -1e30;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            // The component is the ACCUMULATED history at the landing's
            // neighbor texel -- what a landing-miss ghost actually holds.
            // Both sides are stable-space accumulated values: NO
            // frame-to-stable phase correction (the accumulated texel IS
            // the stable content), and no current-frame noise in vComp.
            float3 dd = historyColorSpace - accumulatedNeighbors[j - 1];
            // The per-cell variant: cells (0,+-1) -> CE (center column),
            // (+-1,0) -> EC (center row), corners -> EE. The offsets are
            // unroll constants: these selects fold at compile time.
            float3 invSel     = (kOffsets3x3[j].x == 0.0) ? invCE     : ((kOffsets3x3[j].y == 0.0) ? invEC     : invEE);
            float3 invSqrtSel = (kOffsets3x3[j].x == 0.0) ? invSqrtCE : ((kOffsets3x3[j].y == 0.0) ? invSqrtEC : invSqrtEE);
            float  logDetJ2   = (kOffsets3x3[j].x == 0.0) ? logDetCE2 : ((kOffsets3x3[j].y == 0.0) ? logDetEC2 : logDetEE2);
            // q_j through the SHARED R^-1 (standardized): the components'
            // channel coupling is the content's, same as the null's.
            float  qj = rFull
                ? QuadFormUnitR(dd * invSqrtSel, rC00, rC01, rC02, rC11, rC12, rC22)
                : dot(dd * dd, invSel);
            lComp2[j - 1] = lMass2[j - 1]
                         - kT * log2(1.0 + qj / nu)
                         - logDetJ2;
            lMax2 = max(lMax2, lComp2[j - 1]);
        }

        // The temporal tail: the old content absent from the 9-tap sample.
        float3 sigAC = stats.sigmaClean;
        float3 tailScale = nullScale + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float qTail = rFull
            ? QuadFormUnitR(dCorr / sqrt(tailScale), rC00, rC01, rC02, rC11, rC12, rC22)
            : dot(dCorr * dCorr, float3(1.0, 1.0, 1.0) / tailScale);

        float logDetTailHalf2 = 0.5 * (log2(tailScale.x) + log2(tailScale.y) + log2(tailScale.z));
        float lTail2 = lMassTail2
                     - kT * log2(1.0 + qTail / nu)
                     - logDetTailHalf2;
        lMax2 = max(lMax2, lTail2);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 8; ++j2)
            wSum += exp2(lComp2[j2] - lMax2);
        wSum += exp2(lTail2 - lMax2);

        // llr2 = log2 p1 - log2 p0: the conditional mixture's logsumexp
        // (renormalized through -logSumM), plus the null's full -log2 p0
        // (the t-core and its log-det normalizer). Every term carries the
        // uniform log2(e) scaling, so the final sigmoid consumes the sum
        // directly through exp2.
        float llrFrame2 = (lMax2 + log2(max(wSum, 1e-10))) - logSumM2
                        + kT * log2(1.0 + q0Comp / nu)
                        + 0.5 * logDetNull2;

        float logOdds2 = logPriorOdds2 + llrFrame2;
        float p1 = 1.0 / (1.0 + exp2(-logOdds2));

        // The scalar pull -- the posterior-mean action, floored by the
        // hard ray clamp. The pull IS the eviction mechanism for the miss
        // class; the change class is the hard clamp's. The pull target is
        // `center`, NOT the matched neighbor, and that is exact Bayes:
        // under H0 the optimal store keeps the history; under H1 this
        // texel's store should be the CURRENT-frame posterior -- which IS
        // `center` (the phase-corrected plane evaluation). Pulling toward
        // the matched neighbor component would store NEIGHBOR CONTENT at
        // this texel and perpetuate the miss. Do not "fix" this into a
        // directional pull.
        float wPull = saturate(p1);
        wPull = max(wPull, 1.0 - tHard);

        r.clippedColorSpace = historyColorSpace - wPull * dCorr;
        r.tGate = 1.0 - wPull;
    }
    else
    {
        r.clippedColorSpace = historyColorSpace - (1.0 - tHard) * dCorr;
        r.tGate = tHard;
    }
    return r;
}

// ============================================================================
// THE CLIP-EXCESS BLEND REDUCTION
// ----------------------------------------------------------------------------
// How far the history landed OUTSIDE the gate, in gate radii:
//     excess = sqrt(mdd) / chi    (1.0 = exactly on the boundary)
// -- the same realized statistic debug mode 13 displays (mdd/16 gray).
// The drop is LINEAR in excess from (1 + taaClipBlendDropStart) over
// taaClipBlendDropWidth, saturating at 1; the caller multiplies the
// history feedback by (1 - drop). The t-IRLS/F5 robustification keeps
// current-frame fireflies from tripping it (a firefly pulls the gate's
// CENTER, not the history's residual). Null-safety at any coverage is
// the START slider's job: the coverage band itself (5% at 95%) sits
// between 1.0 and ~1.2 radii -- the margin must clear it.
// MUST MATCH the gate's own chi (and taa.fx's chiDbg): the effective
// radius taaVarianceGamma * (1 + taaClipOvershoot).
// ============================================================================
float ComputeClipExcessBlendDrop(ClipGateResult gate)
{
    if (taaClipBlendDrop < 0.5) return 0.0;
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));
    float excess = (gate.mdd > 1e-20) ? (sqrt(gate.mdd) / chi) : 0.0;
    return saturate((excess - (1.0 + max(taaClipBlendDropStart, 0.0)))
                    / max(taaClipBlendDropWidth, 1e-3));
}

// ============================================================================
// FEEDBACK
// ============================================================================
float ComputeMotionFeedback(HistoryReprojection repro)
{
    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    return clamp(lerp(taaFeedbackMax, taaFeedbackMin, motionDrop), taaFeedbackMin, taaFeedbackMax);
}

float ComputeHistoryFeedback(HistoryReprojection repro, bool planarSurface)
{
    float feedback = ComputeMotionFeedback(repro);

    if (planarSurface)
    {
        feedback -= taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment);
        feedback = max(feedback, 0.0);
    }
    return feedback;
}

#endif // TAA_CLIP_H_HLSL