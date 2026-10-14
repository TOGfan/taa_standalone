// ============================================================================
// TAA history clipping v5.6: THE FLAT MODEL -- exact least-squares bookkeeping
// ----------------------------------------------------------------------------
// v5.3: exact prior variance (the normal-equation identity) + the
//       transported structure split.
// v5.4: the robust plane (one t-IRLS step, drop-one trimmed scale).
// v5.5: the H-free record (noise lives in dev, change in dCorr; the ingest
//       is the de-biased dev^2 identity) + the sandwich information count.
// v5.6: the HAT MATRIX. The weighted-least-squares bookkeeping no longer
//       pretends the weights aren't there:
//         * residSq's dof: E[Sum w r^2] = sigma^2 * Sum w(1-h), h_i =
//           (w_i/W)(1 + mahal_i), Sum h_i = 3 exactly (the trace identity
//           -- the built-in sanity check). The uniform-weight dofCorr was
//           biasing residSq ~1.3x high in the default config, which in
//           v5.5 was an UNSTABLE loop (the ingest's de-bias over-
//           subtracted, the record sagged without a fixed point); exact
//           now by construction.
//         * the IRLS studentization: q_i = r_i^2/((1-h_i)s^2) has E = 1
//           per tap (the classical internally-studentized residual), so
//           clean taps keep weight exactly 1 and a contaminated CENTER
//           tap -- the highest-hat, previously the most under-tested --
//           is downweighted at its true deviation.
//         * the evaluation variance: vFit = sPer * Sum c_k^2 with c_k the
//           EXACT evaluation functional coefficients; Sum c^2 reproduces
//           the classical (1/n)(1+mahal) on uniform weights. The v5.5
//           form overcharged the mahal term by ~W when jitter-aware
//           weighting was off.
//         * the shared chi^2_3 levels: the corroboration reset (5%), the
//           winsor (0.1%), the spike's record branch (~1e-6) -- one law,
//           three tests, each through the /3 of the 3-channel sum.
//       THE MEASURED MISS MODEL (the soft clip): H1 is now EXACTLY the
//       landing-miss hypothesis, priced by one Gaussian miss distribution
//       whose scatter the transport machinery already measures. It
//       generates the prior odds P(|miss| > r0), the neighbor masses
//       P(miss in texel j) and the tail P(|miss| > r1) coherently --
//       replacing four hand-set stand-ins (the 1/512 base, the 64x motion
//       odds, the 0.1 tail weight, the kernel-shaped mixture). On clean
//       static content the prior collapses (~e^-32) and the soft clip
//       switches itself OFF -- correctly: no miss is possible, and
//       change-class deviations are the hard clamp's business since the
//       v5.5 unsheltered record (clamp-rate eviction ~3 frames at 3 sigma
//       instead of the tail's ~1.5; raise kTailChangeFloor if field
//       testing misses the fast response). On noisy-velocity content it
//       runs HOTTER than the old fixed prior ever did -- also correctly:
//       a measured ~1-texel landing scatter means the history really is
//       miss-contaminated. The principled dial is kMissRadiusOdds, never
//       an odds cap.
// v5.6.1 (ALU pass -- no numerical policy change): the ghost posterior
//       now runs in base-2 logs (log2/exp2 are the native opcodes; HLSL
//       log and exp each carry an extra multiply for the ln(2)/log2(e)
//       factor, which the base-2 form folds into constants -- exact in
//       real arithmetic, floating-point differences are rounding-level
//       only); the pass-2 scatter-matrix inverse is built ONCE instead of
//       three times from the same fit; pass 1's moment accumulator drops
//       the fields only pass 2 consumes; the neighborhood AABB collapses
//       to its squared range at the scan (six fewer registers live across
//       both IRLS passes). The hoists and collapses are bit-identical:
//       same operations, same order.
//
// v5.6.3 (hypothesis-side exactness): the v5.6 hat-matrix work closed the
//       plane-side bookkeeping; the residual calibration error lived in
//       the TEST LEVELS and the soft clip's metric. Every channel-model
//       asymmetry between the tests is removed:
//         * ONE channel law for the innovation-scale tests: the spike's
//           record branch is WHITENED through the shrunk full metric
//           (exactly chi^2_3 under the null). The old scalar-total form
//           assumed exchangeable uncorrelated channels: on
//           luminance-dominated YCoCg content the 3-channel sum's
//           effective dof is ~1, the priced ~1e-6 spike level ran at
//           ~1e-3 (records resetting ~1000x the priced rate) and the 0.1%
//           winsor point at ~2% (the record systematically under-ingested
//           legitimate variance). The winsor now censors at the ingest's
//           OWN Satterthwaite dof (WinsorCapNuStar, Wilson-Hilferty
//           quantile); the transport scatter is charged per-channel
//           inside the metric instead of as a scalar threshold addend.
//         * THE SHARED METRIC: the soft clip's 8 neighbor components and
//           its temporal tail are standardized through the SAME shrunk
//           correlation matrix R as the null and the hard test (one
//           hoisted R^-1 at unit scale). The old diagonal-only component
//           metric left a spurious -0.5*log2|R| in the odds (x0.5-0.7 at
//           typical correlations) and mis-shaped q_j along the coupled
//           direction: the soft clip under-engaged ~1.4-2.2x on
//           correlated content. The |R| factors now cancel exactly
//           between the hypotheses' log-dets.
//         * The component cell masses at two-term Euler-Maclaurin (the
//           density-times-area form is ~4% low at sigma ~ 0.7 texel).
//         * The correlation dof from the W2 Kish count (the moments' own
//           effective size; the w3 sandwich count understated it and
//           over-shrank).
//         * The Studentization is a Pade (2,1) rational in 1/nu, host-fit
//           by weighted least squares over a dense anchor set (the
//           3-anchor cubic was exact at its anchors but wobbled a few
//           percent between them on the steep low-nu end).
//       Reviewed and deliberately NOT changed: the record's ingest dof
//       stays kRecordIngestDof (a dynamic nu* there would break the
//       STUDENT_NUINF host sync -- version it deliberately if ever
//       needed); the gate's varH charge stays the full transported
//       varHPer (varH is maintained in READ semantics -- it already IS
//       the Kaiser read's variance; a Sum-c^2 discount would
//       double-count); the nuPost dof blend stays variance-weighted (the
//       honest recalibration is a host-side simulation, not shader
//       algebra -- the documented open item).
//
// v5.6.4 (two local tail-law upgrades): the winsor's censoring point is a
//       PEARSON-III (3-cumulant) match -- the sum's skewness,
//       kap3 = 8*tr(Sigma^3), was invisible to the 2-cumulant
//       Satterthwaite form, which ran the 0.1% point low on mixed
//       variance/correlation structures (a low cap over-censors
//       legitimate variance). The soft clip's radial tails (the prior
//       odds and the temporal tail mass) are ANISOTROPY-CORRECT: a
//       3-cumulant gamma for r^2 replaces the Rayleigh-at-sEffSq form,
//       which was exact only at sx = sy and UNDERstated the tail on
//       directional misses (a one-axis scatter is a chi^2_1-type tail,
//       half the exponential's rate). Both degenerate to the old forms
//       exactly on their old validity sets.
//
// THE GATE: the history's accumulation target is the phase-swept CENTER
// TEXEL; the gate tests the history against the posterior of that target
// given the current frame. PRIOR: plane0, variance tauB2 + sPer*evalVar.
// LIKELIHOOD: the phase-corrected center tap, variance sPer (the record),
// t-robustified at the record's dof with the exact Var(dev) quadratic
// form. F1: the shared-sample covariance charged. At edges the plane
// residual carries the step -- the flip REQUIRES that width.
//
// THE RECORD: the transported sigma^2 MEANS sPer, ingested H-free
// (dev^2 - vPrior + 2*cPT). The winsor caps the ingest at its OWN
// Satterthwaite dof's 0.1% point (v5.6.3 -- WinsorCapNuStar; the null
// bias is ~0.6% and a center-tap firefly's record impact is half the old
// 9x cap's). The spike's record branch is a priced ~1e-6 chi^2_3 test,
// whitened through the shrunk full metric; the range branch (16x the
// observed AABB range) is the distribution-free garbage backstop --
// effectively never on the null by design, the level-change job having
// moved to the H-free ingest and the reveal job to geoCorroboration.
//
// TRANSPORT (tag 101): [31] sign, [30:28] tag, [27:22] sigma (6-bit,
// = sPer), [21:10] VarH ratio (12-bit), [9:6] age (4-bit), [5:0] split
// fraction (6-bit, 1/32 steps over [-1, 0.97]).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host
// context: cbuffer perDraw (taaVarianceGamma, taaClipOvershoot,
// taaChromaVarianceMod, taaSoftClip, taaStudentA/B/C, taaArmingA/B,
// taaJitterPhaseEx/Ey/Exy, taaJitterAwareVariance, taaLumaVariance,
// taaVelocityAlignedVariance, taaHistoryOvershoot, the feedback
// constants). Requires taaShared.h.hlsl, taaConstants.h.hlsl included
// before.
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
    float  invNeffW2;           // (v5.6.3) the W2 Kish count -- the effective
                                // sample size of the moments the correlations
                                // are computed FROM (the correlations' own
                                // dof base; the sandwich count is the
                                // predictive side's)
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
                                // reference's garbage branch), collapsed at
                                // the scan -- the min/max extremes are not
                                // transported through the IRLS passes
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
// (v5.6.3) All three innovation-scale tests now run through the SAME
// channel model: the corroboration reset and the spike's record branch are
// WHITENED through the shrunk full metric (exactly chi^2_3 under the
// null), and the winsor censors at the ingest's own Satterthwaite dof
// (Wilson-Hilferty quantile, computed in WinsorCapNuStar). The old
// scalar-total forms assumed exchangeable uncorrelated channels: on
// luminance-dominated YCoCg deviations the 3-channel sum's effective dof
// is ~1, so the priced 1e-6 spike level ran at ~1e-3 and the 0.1% winsor
// point at ~2%.
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
// The miss distribution is N(0, diag(sx^2, sy^2)) in texel units, sx/sy
// measured by the transport machinery. r0 = the "holds another texel"
// radius (the prior odds' threshold -- the principled dial if the hot
// regime over-pulls: raise to 1.25 to cool the prior at every sigma).
// r1 = beyond the neighbor ring (the tail). kTailChangeFloor: optional
// mass for the change-class ghost (old content absent for reasons other
// than a miss, e.g. lighting steps) -- 0 = the pure miss model; raise to
// ~0.02-0.1 if static-scene lighting transitions evict too slowly.
// (v5.6.3 note: with the shared component metric the posterior runs
// somewhat hotter on correlated content at a fixed dial -- the honest
// direction; re-baseline field tuning here if the soft clip over-engages.)
static const float kMinMissSigma    = 0.125;  // texels: the floor (underflow
                                              // safety; below it "no miss")
static const float kMissRadiusOdds  = 1.0;
static const float kMissRadiusTail  = 1.5;
static const float kTailChangeFloor = 0.0;
// (v5.6.3) The cell-mass Euler-Maclaurin correction's clamp floor: below
// sigma ~ 0.29 texel the two-term expansion's zero-z factor would go
// negative; the prior odds have already collapsed there, so the clamp
// only keeps the log finite.
static const float kMinMassCorr     = 0.0625;
// The tail component's content-scale prior (the unknown old content's
// spread around the neighborhood mean, in content sigmas).
static const float kGhostPriorSigma = 1.0;
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
// not DCE struct fields through out-params were carrying all of them
// through pass 1 (~108 dead MADs per pixel).
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
    // hat, previously the most under-tested) is downweighted at its true
    // deviation.
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
        // (v5.6.3) The W2 Kish count: the moments' own effective sample
        // size, exported for the correlations' dof (see
        // ShrunkenCorrelations). plainInvNeff IS it -- no recomputation.
        stats.invNeffW2 = saturate(plainInvNeff);

        // The inverse scatter matrix and its rank flag, built ONCE: the
        // exact evaluation variance, the exact residual dof and the F1
        // prior/tap covariance below each rebuilt identical copies of this
        // from the same pass-2 fit (three conditional inversions collapsed
        // into one).
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
        float evalVar;
        if (scatterFullRank)
        {
            float2 uEval = float2(dot(vInv0, dEval), dot(vInv1, dEval));
            float2 sumW2offCen = acc.offSumW2 - cenPx * acc.totalWSq;
            float  sXX = acc.offXSqW2 - 2.0 * cenPx.x * acc.offSumW2.x
                       + acc.totalWSq * cenPx.x * cenPx.x;
            float  sYY = acc.offYYSqW2 - 2.0 * cenPx.y * acc.offSumW2.y
                       + acc.totalWSq * cenPx.y * cenPx.y;
            float  sXY = acc.offXYW2 - cenPx.x * acc.offSumW2.y
                       - cenPx.y * acc.offSumW2.x
                       + acc.totalWSq * cenPx.x * cenPx.y;
            float cross = dot(uEval, sumW2offCen);
            float quad  = uEval.x * uEval.x * sXX + uEval.y * uEval.y * sYY
                        + 2.0 * uEval.x * uEval.y * sXY;
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
        // dof biased high, the split anchor R exceeded sPer, the de-bias
        // over-subtracted, and the record sagged without a fixed point).
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

        // F1: the prior/tap shared-sample covariance, on the robust
        // weights -- (v5.6.5) THE EXACT LEVER. The load-bearing identity
        // is the ingest's de-bias, E[dev^2] = sPer + vPrior - 2*cPT with
        // dev = x0c - plane0 and the FITTED phase correction, which pins
        //     cPT = sPer * (k0 + L):
        //     k0 = (w2_0/W) * (1 - dEval' S^-1 cen)   -- the plain
        //          shared-sample term (exact when the phase gradient is
        //          treated as deterministic),
        //     L  = jitterPx' S^-1 c2
        //        + 0.5 * (dEval' S^-1 S2 S^-1 dEval - cen' S^-1 S2 S^-1 cen)
        //        - (w2_0/W) * jitterPx' S^-1 cen
        // with c2 = Sum (w2/W)^2 (o - cen) and S2 = Sum (w2/W)^2 (o-cen)
        // (o-cen)' -- the w2-SQUARED centroid offset and scatter (the
        // same accumulator class the exact evaluation variance uses).
        // The old crossQuad form was the UNIFORM-WEIGHT lever of a
        // different convention (the full Cov(plane0, x0c)): exact only
        // at w2 = const, where the conventions meet. Under the actual
        // kernel/IRLS weights it over-charged cPT two ways at once --
        // the convention factor (2x even at uniform weights) and the
        // w2-for-w2^2 substitution (~3.3x at the default kernel shape)
        // -- biasing the ingest's de-bias low (the record's fixed point
        // sagged ~21% at extreme sub-pixel phase with jitter-aware
        // weighting off) and qTapDen tight (~29%). With jitter-aware
        // weighting ON (the default) dEval ~ 0 and neither form's lever
        // engages: the fix is invisible there by construction.
        // Degenerate spread: the plain shared-sample term stands (the
        // lever's S^-1 does not exist). jitterPx = 0 makes L vanish and
        // k0 = b0 identically; uniform w2 reduces L to
        // dEval' S^-1 dEval / (2N).
        float priorTapCoef = w2[0] * invTotalWeight;
        if (scatterFullRank)
        {
            float  invW2 = invTotalWeight * invTotalWeight;
            float2 sumW2offCen = acc.offSumW2 - cenPx * acc.totalWSq;
            float  sXX = acc.offXSqW2 - 2.0 * cenPx.x * acc.offSumW2.x + acc.totalWSq * cenPx.x * cenPx.x;
            float  sYY = acc.offYYSqW2 - 2.0 * cenPx.y * acc.offSumW2.y + acc.totalWSq * cenPx.y * cenPx.y;
            float  sXY = acc.offXYW2 - cenPx.x * acc.offSumW2.y - cenPx.y * acc.offSumW2.x + acc.totalWSq * cenPx.x * cenPx.y;
            float2 vC  = float2(dot(vInv0, cenPx),  dot(vInv1, cenPx));    // S^-1 cen
            float2 vD  = float2(dot(vInv0, dEval),  dot(vInv1, dEval));    // S^-1 dEval
            float2 uC2 = float2(dot(vInv0, sumW2offCen), dot(vInv1, sumW2offCen)); // S^-1 Sum w2^2 (o-cen)
            float quadD = (sXX * vD.x * vD.x + 2.0 * sXY * vD.x * vD.y + sYY * vD.y * vD.y) * invW2;
            float quadC = (sXX * vC.x * vC.x + 2.0 * sXY * vC.x * vC.y + sYY * vC.y * vC.y) * invW2;
            float k0 = (w2[0] * invTotalWeight) * (1.0 - dot(dEval, vC));
            float lever = dot(jitterPx, uC2) * invW2
                        + 0.5 * (quadD - quadC)
                        - (w2[0] * invTotalWeight) * dot(jitterPx, vC);
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
    // (v5.6.3) The correlation dof from the W2 Kish count: the moments
    // these correlations are computed FROM are w2-weighted, so their own
    // effective size is the honest dof base -- the w3 sandwich count is
    // the information-side quantity and understates n_eff here (the old
    // use over-shrank every correlation). nu = n_eff(w2) - 3 is the
    // Fisher-z form EXACTLY: for a correlation from n samples,
    // Var(r) ~= (1-r^2)^2/(n-3) -- the -3 is already in the call, not a
    // large-n limit left un-corrected (audited: do not "fix" it again).
    float nuCorr = max(1.0 / max(stats.invNeffW2, 1e-3) - 3.0, 1.0);
    rXY = clamp(ShrinkCorrelation(stats.covCross.x / max(stats.sigmaClean.x * stats.sigmaClean.y, kEpsilon), nuCorr), -0.95, 0.95);
    rXZ = clamp(ShrinkCorrelation(stats.covCross.y / max(stats.sigmaClean.x * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
    rYZ = clamp(ShrinkCorrelation(stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
}

// ============================================================================
// THE SHARED CORRELATION METRIC'S QUADRATIC FORM (v5.6.3)
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
// (v5.6.4) THE ANISOTROPIC RADIAL TAIL: log2 P(|miss| > r) at unequal x/y
// scatter. r^2 = sx^2 z1^2 + sy^2 z2^2 is a two-term generalized
// chi-square; its 3-cumulant match is a SHIFTED gamma,
//     kap1 = m = sx^2 + sy^2,  kap2 = 2(sx^4 + sy^4),  kap3 = 8(sx^6 + sy^6)
//     alpha = kap1^2 / kap2  in [1/2, 1],  delta = kap1 - 2 kap2^2 / kap3 >= 0
// (both bounds are identities: alpha by AM-GM on (a, b), delta by
// Cauchy-Schwarz on the eigenvalues; both are equalities at sx = sy, and
// delta = 0 also at full anisotropy -- the chi^2_1 family is exactly
// gamma). The survival is the gamma's uniform tail asymptotic,
//     S ~ u^(alpha-1) e^(-u) / Gamma(alpha),   u = (r^2 - delta) / theta,
// evaluated in log2 (the ln terms fold: ln(u) * kLog2E = log2(u)).
// EXACTLY the old Rayleigh form when sx = sy (alpha = 1, delta = 0:
// log2 S = -r^2 * kLog2E / m, at every r). Accuracy: a tail estimate --
// solid for p <~ 0.01, ~±20% through the hot transition, exact at both
// axis endpoints. lnGamma on [1/2, 1] is the quadratic through
// (1/2, 0.5724), (3/4, 0.2033), (1, 0): max error ~5e-3 in the log.
// ============================================================================
float Log2RadialTail2(float rSq, float sx, float sy)
{
    float a = sx * sx;
    float b = sy * sy;
    float m = a + b;
    float kap2 = 2.0 * (a * a + b * b);
    float kap3 = 8.0 * (a * a * a + b * b * b);
    float alpha = (m * m) / max(kap2, 1e-20);                      // [0.5, 1]
    float delta = max(m - (2.0 * kap2 * kap2) / max(kap3, 1e-30), 0.0);
    float u = max((rSq - delta) * alpha / max(m, 1e-20), 0.0);     // (r^2 - delta)/theta
    float uLog = max(u, 0.5);   // the log term's asymptotic validity floor
    float oneMinusA = 1.0 - alpha;
    float lnGamma = 1.3264 * oneMinusA * oneMinusA + 0.4815 * oneMinusA;
    // The min() pins S <= 0.9997: near u = 0 with alpha near 1 the
    // asymptotic can drift a hair above S = 1, and S > 1 would explode the
    // prior odds through the (1 - p) term.
    return min(-u * kLog2E + (alpha - 1.0) * log2(uLog) - lnGamma * kLog2E, -0.0005);
}

// ============================================================================
// CLIP-STATE TRANSPORT (tag 101)
// ============================================================================
static const float kVarHCodeLogMin  = -6.0;
static const float kVarHCodeLogSpan = 9.0;

float EncodeVarHRatio(float varH, float recordSq)
{
    float ratio = varH / max(recordSq, kEpsilon);
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
    uint u = 0x50000000u
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x3Fu)    << 22)
           | ((uint(varHRatioCode) & 0xFFFu)                    << 10)
           | ((uint(clamp(age, 0.0, kClipMaxAge)) & 0xFu)        << 6)
           |  (uint(clamp(splitCode, 0.0, 63.0)) & 0x3Fu);
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

    // v5.6.1 (perf, exact): varX is already PSD (the jitter phase covariance
    // is PSD by construction and residSq >= 0), so the max() clamps nothing
    // on real input and tot below is EXACTLY SeedRecordTotalSq's inner dot.
    // Computing it here avoids SeedRecordTotalSq's second evaluation of the
    // phase quadratic form -- the old code computed it twice (once for the
    // shares, once inside SeedRecordTotalSq for the winsor anchor), ~15 ops
    // on every live-record pixel. SeedRecordTotalSq remains for the cold
    // paths in taa.fx.
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
// metric (the geometric corroboration test's form -- the shared 95% level).
float WhitenedInnovSq(float3 innov, RecordVarianceParts p, float3 transportVar,
                      float rXY, float rXZ, float rYZ)
{
    float3 v = max(p.ePer + transportVar, float3(1e-10, 1e-10, 1e-10));
    float sX = sqrt(v.x), sY = sqrt(v.y), sZ = sqrt(v.z);
    float m01 = rXY * sX * sY;
    float m02 = rXZ * sX * sZ;
    float m12 = rYZ * sY * sZ;
    float c0 = v.y * v.z - m12 * m12;
    float c1 = m12 * m02 - m01 * v.z;
    float c2 = m01 * m12 - v.y * m02;
    float det = v.x * c0 + m01 * c1 + m02 * c2;
    if (det <= 1e-12)
        return dot(innov * innov, float3(1.0, 1.0, 1.0) / v);
    float invDet = 1.0 / det;
    float i00 = c0 * invDet;
    float i01 = c1 * invDet;
    float i02 = c2 * invDet;
    float i11 = (v.x * v.z - m02 * m02) * invDet;
    float i12 = (m02 * m01 - v.x * m12) * invDet;
    float i22 = (v.x * v.y - m01 * m01) * invDet;
    return i00 * innov.x * innov.x + i11 * innov.y * innov.y + i22 * innov.z * innov.z
         + 2.0 * (i01 * innov.x * innov.y + i02 * innov.x * innov.z + i12 * innov.y * innov.z);
}

// ============================================================================
// THE WINSOR'S CENSORING POINT (v5.6.3: the ingest's own Satterthwaite dof)
// ============================================================================
// The ingest is a 3-channel SUM of correlated squared devs. For
// x ~ N(0, S), the sum of squares has Satterthwaite dof
//     nuStar = (tr S)^2 / tr(S^2)
//            = (sum v)^2 / (sum v^2 + 2*sum rho^2 v_j v_k)   in [1, 3]
// -- 1 on luminance-dominated YCoCg deviations (all power in Y), 3 only
// for exchangeable uncorrelated channels. NOTE the rho^2: the covariance
// of SQUARES carries the SQUARED correlation (Cov(x_j^2, x_k^2) =
// 2 S_jk^2) -- not rho; do not "simplify" it. The per-channel null
// variances are the tap dev's own (vPrior + sPer - 2*cPT -- the same
// identity the ingest de-biases with, replicated here bit-for-bit from
// the gate's qTapDen), so the censoring point is the honest 0.1%
// quantile of the actual null:
//     cap = Q_0.999(null) / kap1 * spatialPriorSq
// with Q via the PEARSON-III match (v5.6.4: the 3-cumulant SHIFTED gamma;
// the v5.6.3 nuStar form was its 2-cumulant special case -- exact when the
// channel sum is chi^2-shaped, ~2-6% low otherwise, and a low cap
// over-censors legitimate variance). The Wilson-Hilferty quantile of the
// gamma part runs ~1-3% high -- the conservative side for a censor.
// The OLD form (chi^2_3/3 = 5.42x) ran the 0.1% point at ~2% on gray
// content: the record systematically under-ingested legitimate variance
// there, sagging the aged gate.
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
    // (v5.6.4) PEARSON-III (3-cumulant) instead of Satterthwaite: the sum's
    // skewness, kap3 = 8*tr(Sigma^3), is invisible to the 2-cumulant chi^2
    // form. The matched law is a SHIFTED gamma,
    //     delta = kap1 - 2 kap2^2 / kap3 >= 0   (Cauchy-Schwarz on the
    //              eigenvalues; zero exactly on the chi^2 family),
    //     alpha = 4 kap2^3 / kap3^2  in [0.5, 1.5]  (skew <= 2*sqrt(2)
    //              for any PSD Sigma),
    // and the 0.999 quantile is delta + (2 kap2^2 / kap3) * WH(alpha)^3.
    // On gamma-consistent channel sums this is form-identical to the old
    // nuStar path (delta = 0, alpha = nuStar/2, the same WH); on mixed
    // variance/correlation structures it roughly halves the 0.1%-quantile
    // error, with the residual on the high side (a slightly high cap
    // censors slightly less -- the safe direction for a censor).
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
    return (chiQ / max(kap1, 1e-20)) * max(parts.spatialPriorSq, kClipSigmaRecordFloorSq);
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
    float2 missSigmaPx,
    float3 neighborhoodTaps[9],
    float  rXY, float rXZ, float rYZ)   // (v5.6.3) the caller's shrunk
                                        // correlations -- ONE evaluation
                                        // shared by every consumer
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
        // (the same identity the ingest de-biases with). The prior form
        // (vPrior + sPer) understated q exactly at the phase-near taps,
        // where fireflies land. The floor guards the perfect-correlation
        // limit (the CS bound guarantees only >= 0).
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
        // the reconstruction), charged at residSq's OWN exact dof
        // (Sum w(1-h) on the robust weights -- the sandwich count's
        // variance-side twin). The EMA'd fraction's residual noise is NOT
        // charged: it manifests as the clamp's positive-part mean, already
        // carried in vPrior, and charging it again would double-count. (If
        // mode 13 shows over-engagement on static smooth content, add
        // tauB2^2 * kSplitEmaRate/(2-kSplitEmaRate) / max(splitFrac,0.04)^2
        // -- the documented revert.)
        float3 denomTau = tauB2 * tauB2 / max(stats.residDof, 1.0);
        float3 nuPrior = vPrior * vPrior
                       / max(denomTau + vFit * vFit / nuRecordSafe, float3(1e-10, 1e-10, 1e-10));
        nuPrior = max(nuPrior, float3(2.0, 2.0, 2.0));

        float3 nuFlat = (A + B) * (A + B)
                      / max(A * A / nuRecordSafe + B * B / nuPrior, float3(1e-10, 1e-10, 1e-10));
        nuFlat = max(nuFlat, float3(2.0, 2.0, 2.0));

        // (v5.6.3) The host fit is a Pade (2,1) rational in x = 1/nu,
        //     S(nu) = (1 + A*x + B*x^2) / (1 + C*x),
        // weighted-least-squares fit to the exact F(3,nu) quantile
        // relation over a dense anchor set (taa.lua -- the same
        // f3nuQuantile machinery the old cubic consumed). The cubic was
        // exact at its three anchors but wobbled a few percent between
        // them on the steep low-nu end (S rises ~3x between nu = 4 and
        // nu = 2); the rational tracks the exact curve roughly an order
        // of magnitude tighter. The denominator is Lua-verified positive
        // over x in [0, 0.5] (nu >= 2 everywhere here); the max() is a
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

    // ---- THE SHARED METRIC (v5.6.3) ----
    // One correlation matrix R (the caller's shrunk correlations -- the
    // same values the spike, the winsor and the corroboration test
    // consume), one hoisted R^-1 at unit scale. Every quadratic form in
    // this gate -- the hard test's mdd, the soft clip's null, its 8
    // neighbor components, its temporal tail -- standardizes its
    // deviation by the hypothesis' per-channel scales and runs through
    // this R^-1: exactly the full metric diag(v) R diag(v), and the SAME
    // channel coupling on both sides of the soft clip's likelihood ratio
    // (the old per-use inline inverses and the components' diagonal-only
    // metric are gone).
    float detR = 1.0 + 2.0 * rXY * rXZ * rYZ - rXY * rXY - rXZ * rXZ - rYZ * rYZ;
    bool  rFull = (detR > 1e-6);
    float invDetR = rFull ? (1.0 / detR) : 0.0;
    float rC00 = (1.0 - rYZ * rYZ) * invDetR;
    float rC01 = (rXZ * rYZ - rXY) * invDetR;
    float rC02 = (rXY * rYZ - rXZ) * invDetR;
    float rC11 = (1.0 - rXZ * rXZ) * invDetR;
    float rC12 = (rXY * rXZ - rYZ) * invDetR;
    float rC22 = (1.0 - rXY * rXY) * invDetR;

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
        // variance, on the SAME shrunk full metric as the hard test (F6)
        // with the exact relative log-det normalizer. H1: the landing-miss
        // hypothesis -- a PROPER mixture of the 8 NEIGHBOR TEXELS'
        // accumulated content (what a landing-miss ghost holds) plus one
        // temporal tail (the old content absent from the 9-tap sample).
        // THE CENTER TAP IS NOT A COMPONENT -- it IS the null's likelihood.
        // Every weight is generated by ONE Gaussian miss distribution
        // N(0, diag(sx^2, sy^2)) in texel units, measured by the transport
        // machinery (see the call site): the prior odds from its Rayleigh
        // tail at r0, the neighbor masses from its density at the texel
        // offsets, the tail mass from its tail at r1 -- all renormalized
        // to the conditional distribution of the miss GIVEN a ghost.
        //
        // THE WHOLE POSTERIOR RUNS IN BASE-2 LOGS (v5.6.1): every term is
        // homogeneous in the log base, so scaling the entire log-odds by
        // log2(e) converts every ln to log2 with ALL COEFFICIENTS
        // UNCHANGED -- and log2/exp2 are the native opcodes (HLSL log/exp
        // each pay an extra multiply against the ln(2)/log2(e) factor that
        // this form folds into constants instead).
        //
        // (v5.6.3) The neighbor masses carry the two-term Euler-Maclaurin
        // cell correction, and the components and the tail run through the
        // SHARED correlation metric above -- so the |R| log-det factors
        // cancel EXACTLY between the hypotheses (both sides keep only
        // their diagonal log-dets).
        float3 nullScale = max((gateVarBase + transportTerm) * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));
        float3 vCompRaw = recordLive
            ? parts.varHPer + max(parts.sPer, float3(1e-8, 1e-8, 1e-8)) + transportTerm
            : (gateVarBase + transportTerm);
        float3 vComp = max(vCompRaw * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));

        float nu = max(nuPost, 2.0);
        float kT = 0.5 * (nu + 3.0);

        float3 invComp = float3(1.0, 1.0, 1.0) / vComp;
        float3 invSqrtComp = 1.0 / sqrt(vComp);

        // ---- the null: q0 through the shared R^-1; the DIAGONAL log-det --
        // (v5.6.3) |Sigma_0| = |nullScale| * |R| and every component's
        // |Sigma_j| = |vComp| * |R|: the |R| factors cancel in the LLR, so
        // BOTH sides keep only their diagonal log-dets. The old null
        // log-det carried the |R| alone -- a spurious -0.5*log2|R| in the
        // odds (x0.5-0.7 at typical correlations) biasing the soft clip
        // cold on correlated content.
        float logDetNull2 = log2(nullScale.x) + log2(nullScale.y) + log2(nullScale.z);
        float q0Comp = rFull
            ? QuadFormUnitR(dCorr / sqrt(nullScale), rC00, rC01, rC02, rC11, rC12, rC22)
            : dot(dCorr * dCorr, float3(1.0, 1.0, 1.0) / nullScale);

        // ---- the miss model ----
        float sx = max(missSigmaPx.x, kMinMissSigma);
        float sy = max(missSigmaPx.y, kMinMissSigma);
        // (v5.6.4) sEffSq is gone: both radial-tail consumers now take the
        // anisotropic scales directly (Log2RadialTail2).
        // ln-exponents, base-2-scaled: the log2(e) factor folded in so the
        // Gaussian log-density terms below are straight log2.
        float inv2Sx2 = (0.5 * kLog2E) / (sx * sx);
        float inv2Sy2 = (0.5 * kLog2E) / (sy * sy);
        float logNormC2 = -log2(6.2831853 * sx * sy);

        // The prior odds: P(|miss| > r0) / P(|miss| <= r0), the radial
        // tail at the ANISOTROPIC scales (v5.6.4: Log2RadialTail2 -- the
        // old Rayleigh-at-sEffSq form was exact only at sx = sy and
        // UNDERstated the tail on directional misses; a one-axis scatter
        // is a chi^2_1-type tail, half the exponential's rate). Collapses
        // on clean content (the soft clip switches itself off); runs hot
        // on large measured scatter (the history really is
        // miss-contaminated). The dial is kMissRadiusOdds, never a cap.
        // (Also kills the old exp2 -> log2 round trip: the helper already
        // returns log2.)
        float logPH1_2 = Log2RadialTail2(kMissRadiusOdds * kMissRadiusOdds, sx, sy);
        float logPriorOdds2 = max(logPH1_2, -100.0)
                            - log2(max(1.0 - exp2(logPH1_2), 1e-6));

        // The component masses (v5.6.3: two-term Euler-Maclaurin): the
        // mass of texel j is, per axis, the exact Phi-difference's midpoint
        // expansion
        //     P = 2*delta*phi(z) * (1 + delta^2*(z^2-1)/6 + O(delta^4)),
        // delta = 0.5/sigma -- the first term IS the density-times-area
        // form the old code used (~4% low at sigma ~ 0.7 texel, worse
        // below); the correction takes it under ~0.5% for sigma >= 0.5.
        // logSumM renormalizes the 9 into the conditional distribution
        // GIVEN a ghost.
        float lMass2[8];
        float lMassTail2;
        float logSumM2;
        {
            float invSx2 = 1.0 / (sx * sx);
            float invSy2 = 1.0 / (sy * sy);
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
                lMass2[j - 1] = logNormC2 - (o.x * o.x) * inv2Sx2 - (o.y * o.y) * inv2Sy2
                              + log2(corrX) + log2(corrY);
                lmMax = max(lmMax, lMass2[j - 1]);
            }
            // kTailChangeFloor = 0: log(exp(x)) is x. fxc does not fold the
            // round trip (it is not NaN-safe algebraically) -- but the
            // floor is a compile-time constant, so the branch collapses and
            // the default build pays neither the exp2 nor the log2.
            lMassTail2 = (kTailChangeFloor > 0.0)
                ? log2(exp2(Log2RadialTail2(kMissRadiusTail * kMissRadiusTail, sx, sy)) + kTailChangeFloor)
                : Log2RadialTail2(kMissRadiusTail * kMissRadiusTail, sx, sy);
            lmMax = max(lmMax, lMassTail2);
            float mSum = 0.0;
            [unroll]
            for (int j1 = 0; j1 < 8; ++j1)
                mSum += exp2(lMass2[j1] - lmMax);
            mSum += exp2(lMassTail2 - lmMax);
            logSumM2 = lmMax + log2(max(mSum, 1e-10));
        }

        // The component log-dets are j-invariant: hoisted out of the
        // unrolled mixture loop (the compiler may CSE them; the hoist
        // guarantees it on every backend). Diagonal only -- the |R| factor
        // cancels against the null's (see above).
        float logDetCompHalf2 = 0.5 * (log2(vComp.x) + log2(vComp.y) + log2(vComp.z));

        float lComp2[8];
        float lMax2 = -1e30;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            // The phase-corrected component residual (the neighbor texel's
            // footprint-average estimate is tap_j + g*j). dCorr + center is
            // historyColorSpace exactly (dCorr was defined as history minus
            // center), so the center terms CANCEL; the simplified form saves
            // two float3 adds per component and one rounding, exact in real
            // arithmetic (compilers cannot reassociate float +/- to find it).
            float3 dd = historyColorSpace - neighborhoodTaps[j] + stats.phaseShift;
            // (v5.6.3) q_j through the SHARED R^-1 (standardized): the
            // components' channel coupling is the content's, same as the
            // null's -- the old diagonal-only form mis-shaped q_j along
            // the coupled direction and, with the log-det asymmetry above,
            // under-engaged the soft clip ~1.4-2.2x on correlated content.
            float  qj = rFull
                ? QuadFormUnitR(dd * invSqrtComp, rC00, rC01, rC02, rC11, rC12, rC22)
                : dot(dd * dd, invComp);
            lComp2[j - 1] = lMass2[j - 1]
                         - kT * log2(1.0 + qj / nu)
                         - logDetCompHalf2;
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
        // class; the change class is the hard clamp's (v5.5).
        // (v5.6.3 audit note) The pull target is `center`, NOT the matched
        // neighbor, and that is exact Bayes: under H0 the optimal store
        // keeps the history; under H1 this texel's store should be the
        // CURRENT-frame posterior -- which IS `center` (the phase-corrected
        // plane evaluation). Pulling toward the matched neighbor component
        // would store NEIGHBOR CONTENT at this texel and perpetuate the
        // miss. Do not "fix" this into a directional pull.
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