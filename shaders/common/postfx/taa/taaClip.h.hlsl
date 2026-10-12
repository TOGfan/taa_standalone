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
// (dev^2 - vPrior + 2*cPT). The winsor caps the chi^2_3-scaled ingest at
// 0.1%: the null bias is ~0.6% and a center-tap firefly's record impact
// is half the old 9x cap's. The spike's record branch is a priced ~1e-6
// chi^2_3 test; the range branch (16x the observed AABB range) is the
// distribution-free garbage backstop -- effectively never on the null by
// design, the level-change job having moved to the H-free ingest and the
// reveal job to geoCorroboration.
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
    float3 aabbMin;             // full set (the spike reference's observed range)
    float3 aabbMax;
    float3 sigmaClean;          // the post-plane residual scale, per channel
                                // (the soft clip's temporal tail)
    float  invNeff;             // the SANDWICH information count (dof side:
                                // Sum w*psi' mass, information-positive only)
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
// The 3-dof null's quantiles. The innovation-scale consumers (the winsor,
// the spike) go through the /3: a 3-channel SUM whose null mean is the
// total scale is (total/3) * chi^2_3.
static const float kChiSq3_95   = 7.815;     // the corroboration reset: 5%
static const float kChiSq3_999  = 16.266;    // the winsor: 0.1% censoring
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
static const float kMinMissSigma    = 0.125;  // texels: the floor (underflow
                                              // safety; below it "no miss")
static const float kMissRadiusOdds  = 1.0;
static const float kMissRadiusTail  = 1.5;
static const float kTailChangeFloor = 0.0;
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
// THE MOMENT ACCUMULATOR (both IRLS passes share it)
// ============================================================================
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
    a.offXYW2 = 0.0;

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

ColorPlaneFit SolveColorPlane(ColorMomentAccum a, float3 mean3, float2 cenPx, float invTotalWeight)
{
    ColorPlaneFit p;
    p.vXX  = max(a.offXSq * invTotalWeight - cenPx.x * cenPx.x, 1e-6);
    p.vYY  = max(a.offYSq * invTotalWeight - cenPx.y * cenPx.y, 1e-6);
    p.vXY  = a.offXY * invTotalWeight - cenPx.x * cenPx.y;
    p.covX = a.covXSum * invTotalWeight - cenPx.x * mean3;
    p.covY = a.covYSum * invTotalWeight - cenPx.y * mean3;
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

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx, float2 transportErrPx)
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
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
    float w9[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        stats.aabbMin = min(stats.aabbMin, tapColorSpace);
        stats.aabbMax = max(stats.aabbMax, tapColorSpace);

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

    // ---- PASS 1: the plain weighted LS plane ---------------------------
    ColorMomentAccum acc1;
    AccumulateColorMoments(neighborhoodColorSpace, w9, acc1);
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
        if (fit.detV > 1e-6)
        {
            float2 vInv0 = float2( fit.vYY, -fit.vXY) / fit.detV;
            float2 vInv1 = float2(-fit.vXY,  fit.vXX) / fit.detV;
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
        if (fit.detV > 1e-6)
        {
            float2 vInv0 = float2( fit.vYY, -fit.vXY) / fit.detV;
            float2 vInv1 = float2(-fit.vXY,  fit.vXX) / fit.detV;
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
        // weights. The residSq basis models the structure as sharing
        // through the kernel; the Cauchy-Schwarz clamp guards both limits.
        float crossQuad = 0.0;
        if (fit.detV > 1e-6)
        {
            float2 vInv0 = float2( fit.vYY, -fit.vXY) / fit.detV;
            float2 vInv1 = float2(-fit.vXY,  fit.vXX) / fit.detV;
            float2 cRhs  = jitterPx - w2[0] * cenPx;
            crossQuad = dEval.x * dot(vInv0, cRhs) + dEval.y * dot(vInv1, cRhs);
        }
        stats.priorTapCov = max(stats.residSq * invTotalWeight * (w2[0] + crossQuad), float3(0.0, 0.0, 0.0));

        // the transport scatter, projected on the robust gradient
        stats.transportVar = (transportErrPx.x * transportErrPx.x) * (fit.gradX * fit.gradX)
                           + (transportErrPx.y * transportErrPx.y) * (fit.gradY * fit.gradY);

        stats.sigmaClean = max(sqrt(stats.residSq), kMinSigma);
        stats.covCross = float3(acc.wXY * invTotalWeight - mean3.x * mean3.y,
                                acc.wXZ * invTotalWeight - mean3.x * mean3.z,
                                acc.wYZ * invTotalWeight - mean3.y * mean3.z);
    }

    stats.rangeSq = dot(stats.aabbMax - stats.aabbMin, stats.aabbMax - stats.aabbMin);
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
    float rDof = max(1.0 / max(stats.invNeff, 1e-3) - 1.0, 2.0);
    float nuCorr = max(rDof - 2.0, 1.0);   // n_eff - 3: the correlation's own dof
    rXY = clamp(ShrinkCorrelation(stats.covCross.x / max(stats.sigmaClean.x * stats.sigmaClean.y, kEpsilon), nuCorr), -0.95, 0.95);
    rXZ = clamp(ShrinkCorrelation(stats.covCross.y / max(stats.sigmaClean.x * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
    rYZ = clamp(ShrinkCorrelation(stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon), nuCorr), -0.95, 0.95);
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

    float3 varX = max(SeedRecordVarX(stats), float3(0.0, 0.0, 0.0));
    float  tot  = dot(varX, float3(1.0, 1.0, 1.0));
    float3 share = (tot > 1e-10)
        ? varX / tot
        : stats.sigmaClean * stats.sigmaClean / max(dot(stats.sigmaClean, stats.sigmaClean), kEpsilon);

    float spatialPriorSq = max(SeedRecordTotalSq(stats), kClipSigmaRecordFloorSq);

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
    float3 neighborhoodTaps[9])
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

        // The host fit is exact at THREE anchors (nu = 2, nu0, nu_inf).
        float3 studentS = 1.0 + taaStudentA / nuFlat
                       + taaStudentB / (nuFlat * nuFlat)
                       + taaStudentC / (nuFlat * nuFlat * nuFlat);
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

    // ---- THE FULL METRIC (the shrunk sample correlations, clamped) ----
    float rXY, rXZ, rYZ;
    ShrunkenCorrelations(stats, rXY, rXZ, rYZ);
    float vX = max(gateVar.x * chromaScale.x * chromaScale.x, minGateVar);
    float vY = max(gateVar.y * chromaScale.y * chromaScale.y, minGateVar);
    float vZ = max(gateVar.z * chromaScale.z * chromaScale.z, minGateVar);
    float sX = sqrt(vX), sY = sqrt(vY), sZ = sqrt(vZ);
    float m01 = rXY * sX * sY;
    float m02 = rXZ * sX * sZ;
    float m12 = rYZ * sY * sZ;

    float c0 = vY * vZ - m12 * m12;
    float c1 = m12 * m02 - m01 * vZ;
    float c2 = m01 * m12 - vY * m02;
    float det = vX * c0 + m01 * c1 + m02 * c2;
    float mdd;
    if (det > 1e-12)
    {
        float invDet = 1.0 / det;
        float i00 = c0 * invDet;
        float i01 = c1 * invDet;
        float i02 = c2 * invDet;
        float i11 = (vX * vZ - m02 * m02) * invDet;
        float i12 = (m02 * m01 - vX * m12) * invDet;
        float i22 = (vX * vY - m01 * m01) * invDet;
        mdd = i00 * dCorr.x * dCorr.x + i11 * dCorr.y * dCorr.y + i22 * dCorr.z * dCorr.z
            + 2.0 * (i01 * dCorr.x * dCorr.y + i02 * dCorr.x * dCorr.z + i12 * dCorr.y * dCorr.z);
    }
    else
    {
        mdd = dCorr.x * dCorr.x / vX + dCorr.y * dCorr.y / vY + dCorr.z * dCorr.z / vZ;
    }
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
        float3 nullScale = max((gateVarBase + transportTerm) * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));
        float3 vCompRaw = recordLive
            ? parts.varHPer + max(parts.sPer, float3(1e-8, 1e-8, 1e-8)) + transportTerm
            : (gateVarBase + transportTerm);
        float3 vComp = max(vCompRaw * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));

        float nu = max(nuPost, 2.0);
        float kT = 0.5 * (nu + 3.0);

        float3 invComp = float3(1.0, 1.0, 1.0) / vComp;

        float q0Comp;
        float logDetNull;
        {
            float nX = nullScale.x, nY = nullScale.y, nZ = nullScale.z;
            float nsXY = rXY * sqrt(nX * nY);
            float nsXZ = rXZ * sqrt(nX * nZ);
            float nsYZ = rYZ * sqrt(nY * nZ);
            float nc0 = nY * nZ - nsYZ * nsYZ;
            float nc1 = nsYZ * nsXZ - nsXY * nZ;
            float nc2 = nsXY * nsYZ - nY * nsXZ;
            float ndet = nX * nc0 + nsXY * nc1 + nsXZ * nc2;
            if (ndet > 1e-12)
            {
                float nInvDet = 1.0 / ndet;
                float ni00 = nc0 * nInvDet;
                float ni01 = nc1 * nInvDet;
                float ni02 = nc2 * nInvDet;
                float ni11 = (nX * nZ - nsXZ * nsXZ) * nInvDet;
                float ni12 = (nsXZ * nsXY - nX * nsYZ) * nInvDet;
                float ni22 = (nX * nY - nsXY * nsXY) * nInvDet;
                q0Comp = ni00 * dCorr.x * dCorr.x + ni11 * dCorr.y * dCorr.y + ni22 * dCorr.z * dCorr.z
                       + 2.0 * (ni01 * dCorr.x * dCorr.y + ni02 * dCorr.x * dCorr.z + ni12 * dCorr.y * dCorr.z);
                logDetNull = log(ndet);
            }
            else
            {
                q0Comp = dot(dCorr * dCorr, float3(1.0, 1.0, 1.0) / nullScale);
                logDetNull = log(nullScale.x) + log(nullScale.y) + log(nullScale.z);
            }
        }

        // ---- the miss model ----
        float sx = max(missSigmaPx.x, kMinMissSigma);
        float sy = max(missSigmaPx.y, kMinMissSigma);
        float sEffSq = 0.5 * (sx * sx + sy * sy);
        float inv2Sx = 0.5 / (sx * sx);
        float inv2Sy = 0.5 / (sy * sy);
        float logNormC = -log(6.2831853 * sx * sy);

        // The prior odds: P(|miss| > r0) / P(|miss| <= r0), the Rayleigh
        // tail. Collapses on clean content (the soft clip switches itself
        // off); runs hot on large measured scatter (the history really is
        // miss-contaminated). The dial is kMissRadiusOdds, never a cap.
        float pH1 = exp(-0.5 * kMissRadiusOdds * kMissRadiusOdds / sEffSq);
        float logPriorOdds = log(max(pH1, 1e-30)) - log(max(1.0 - pH1, 1e-6));

        // The component masses: P(miss in texel j) = the Gaussian density
        // at offset j (texel area 1 -- the density-times-area form is
        // within ~4% of the exact integral for all sigma); the tail:
        // P(|miss| > r1) + the optional change-class floor. logSumM
        // renormalizes the 9 into the conditional distribution GIVEN a
        // ghost.
        float lMass[8];
        float lMassTail;
        float logSumM;
        {
            float lmMax = -1e30;
            [unroll]
            for (int j = 1; j < 9; ++j)
            {
                float2 o = kOffsets3x3[j];
                lMass[j - 1] = logNormC - (o.x * o.x) * inv2Sx - (o.y * o.y) * inv2Sy;
                lmMax = max(lmMax, lMass[j - 1]);
            }
            lMassTail = log(exp(-0.5 * kMissRadiusTail * kMissRadiusTail / sEffSq)
                            + kTailChangeFloor);
            lmMax = max(lmMax, lMassTail);
            float mSum = 0.0;
            [unroll]
            for (int j1 = 0; j1 < 8; ++j1)
                mSum += exp(lMass[j1] - lmMax);
            mSum += exp(lMassTail - lmMax);
            logSumM = lmMax + log(max(mSum, 1e-10));
        }

        float lComp[8];
        float lMax = -1e30;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            // The phase-corrected component residual (the neighbor texel's
            // footprint-average estimate is tap_j + g*j).
            float3 dd = dCorr + (center - neighborhoodTaps[j]) + stats.phaseShift;
            float  qj = dot(dd * dd, invComp);
            lComp[j - 1] = lMass[j - 1]
                         - kT * log(1.0 + qj / nu)
                         - 0.5 * (log(vComp.x) + log(vComp.y) + log(vComp.z));
            lMax = max(lMax, lComp[j - 1]);
        }

        // The temporal tail: the old content absent from the 9-tap sample.
        float3 sigAC = stats.sigmaClean;
        float3 tailScale = nullScale + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invTail = float3(1.0, 1.0, 1.0) / tailScale;
        float qTail = dot(dCorr * dCorr, invTail);

        float lTail = lMassTail
                    - kT * log(1.0 + qTail / nu)
                    - 0.5 * (log(tailScale.x) + log(tailScale.y) + log(tailScale.z));
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 8; ++j2)
            wSum += exp(lComp[j2] - lMax);
        wSum += exp(lTail - lMax);

        // llr = log p1 - log p0: the conditional mixture's logsumexp
        // (renormalized through -logSumM), plus the null's full -log p0
        // (the t-core and its log-det normalizer).
        float llrFrame = (lMax + log(max(wSum, 1e-10))) - logSumM
                       + kT * log(1.0 + q0Comp / nu)
                       + 0.5 * logDetNull;

        float logOdds = logPriorOdds + llrFrame;
        float p1 = 1.0 / (1.0 + exp(-logOdds));

        // The scalar pull -- the posterior-mean action, floored by the
        // hard ray clamp. The pull IS the eviction mechanism for the miss
        // class; the change class is the hard clamp's (v5.5).
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