// ============================================================================
// TAA history clipping v5.3: THE FLAT MODEL -- exact prior, transported split
// ----------------------------------------------------------------------------
// The hierarchical gate (v3.11 lineage, F-series exactness fixes) + the plain
// single-sample record. No layer masks, no stored flags, no step model, no
// coverage machinery, no detector: every action is the posterior mean.
//
// THE GATE: the history's accumulation target is the phase-swept CENTER
// TEXEL; the gate tests the history against the posterior of that target
// given the current frame.
//   PRIOR (v5.3, EXACT): plane0, variance vPrior = tauB2 + sPer*(invNeff
//   + lever). The evaluation functional's residual projection is annihilated
//   by the LS normal equations (Sum w r = 0, Sum w(k-cen) r = 0), so
//       plane0 - T = Sum c_k eps_k - r(0)      (exact to linear order):
//   the noise enters at sPer scale through the leverage quadratic, the
//   structure ONCE as r(0). The v5.2 form residSq*(invNeff+lever) billed
//   the structure twice (tauB2 + (tauB2+sPer)*q vs tauB2 + sPer*q) --
//   ~q ~ 25-45% of the structural charge as ghost-side slack.
//   tauB2 = the STABLE structural share, from the TRANSPORTED split
//   fraction (below).
//   LIKELIHOOD: the phase-corrected center tap, variance sPer = the
//   record's implied per-sample variance (E[i^2] - varH). The identity
//   that makes the form self-calibrating: sPer absorbs the sweep delta
//   (the flip charge at edges) AND the phase correction's own error,
//   and wherever the neighborhood carries no certifiable structure
//   (tauB2 large) the posterior degenerates EXACTLY to the innovation
//   test, whose null the record measures directly.
//   The tap is t-robustified (one IRLS step at the record's dof; the
//   exact (nu+q)/(nu+1) weight).
//   F1: the prior/tap shared-sample covariance is charged (both are
//   linear functionals of the same nine taps).
//   At edges the plane residual carries the step, so the gate is
//   step-wide there -- the flip REQUIRES that width.
//
// THE STRUCTURE SPLIT (v5.3): tauB2 is certified TEMPORALLY. The per-frame
// split D = Sum(residSq) - sPer carries ~100% noise (a 3x3 variance
// estimate has dof ~2-4), so clamping it per frame charges E[max(D,0)] =
// sigma*phi(0) ~ 0.4*sigma_D of spurious structure on smooth content --
// roughly DOUBLING vPrior exactly where ghosting is most visible. The
// fix transports the split: EMA the signed normalized fraction
//     f = (R - s) / max(R, s)   in [-1, 1],  rate 0.2,  6 bits,
// and reconstruct tauB2 = max(f, 0) * residSq -- the clamp acts on the
// STABLE side (f +- 0.3 post-EMA), the same trick the record itself uses
// to beat the 9-tap noise. Regimes:
//   * smooth: E[f] <= 0 (Jensen-negative at R = s); residual spurious
//     charge ~0.12*residSq (3-4x down from ~0.4).
//   * texel-aligned patterns (chessboard): sPer ~ 0, f -> ~0.9, full
//     charge, magnitude from the live residSq. A per-frame threshold
//     CANNOT separate this from the smooth noise case (D/sigma ~ 1 in
//     both) -- only temporal stability can.
//   * sub-texel-rough texture: D = tauB2 - E[delta^2] < 0 -> reads as
//     noise -> the innovation test, which is CORRECT there (the history
//     holds the sweep average; the innovation measures exactly the
//     delta).
//   * edges: the step's f ~ 0.67, stable, full charge.
//   * cold: the seed >= Sum(residSq) by construction, so a fresh record
//     certifies nothing (f <= 0) -- the cold path below is the no-record
//     LIMIT of this model, not a separate policy.
//
// THE SOFT CLIP: beyond the gate radius the action is the exact posterior
// mean. H0: the hierarchical null at its honest un-inflated predictive
// variance, on the same shrunk full metric as the hard test (F6) with the
// exact relative log-det normalizer. H1: a PROPER mixture of the 8
// NEIGHBOR TEXELS' accumulated content plus one temporal tail -- every
// component a full log-density carrying its own scale normalizer inside
// the logsumexp; the null's +0.5*logDetNull is charged once, outside.
// THE CENTER TAP IS NOT A COMPONENT (it IS the null's likelihood). The
// prior odds are the per-pixel base plus the motion term (the one
// hand-set prior); the sequential detector is retired -- the soft clip's
// own pull (wPull per frame at the frame-LLR's posterior) is the
// eviction mechanism, and the empirical verdict was that it suffices.
//
// ---------------------------------------------------------------------------
// POST-MORTEM: THE COVERAGE MODEL (v4.0-v4.4), REMOVED
// ---------------------------------------------------------------------------
// C(p) = L(p) + c(p)*D, coverage measured from color, one 3x3 solve --
// mathematically sound, and removed on empirical verdict: with its
// actions disabled the system measured near-perfect AND the trail
// problem it was built for was already solved elsewhere.
//   1. A void initializer (the LS residual regressed on the design --
//      identically zero) left the whole model dead for two versions.
//   2. The collinear overfit: soft content makes u ~ linear in position;
//      the (plane, step) split goes ill-conditioned and LS finds a
//      CANCELING solution (huge D, zero residual) that fits the nine
//      taps and explodes off them -- conf measures explanation, not
//      identifiability. Through Untonemap's pole at luma 1: white pixels.
//   3. The tap-grid quantization: hard edges give BINARY point samples,
//      so the staircase's stable-position estimate sawtooths +-0.5
//      texel at the jitter frequency.
// THE LESSONS FOR ANY SUCCESSOR (do not relearn these):
//   (a) the fit's legitimate resolution is the TAP GRID; the sub-texel
//       structure lives in the ACCUMULATION and must be read from time;
//   (b) identifiability and explanation are different statistics;
//   (c) bound every fitted magnitude against a CANCELLATION-FREE
//       measurement of the same quantity.
//
// TRANSPORT (tag 101, v5.3): [31] sign, [30:28] tag, [27:22] sigma
// (6-bit), [21:10] VarH ratio (12-bit, 0.15% steps -- quantization
// retired as an error source), [9:6] age (4-bit), [5:0] structure split
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
// THE COLOR-FIELD STATISTICS (moments -> the coupled 2x2 plane)
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 aabbMin;             // full set (the spike reference's observed range)
    float3 aabbMax;
    float3 sigmaClean;          // the post-plane residual scale, per channel
                                // (the soft clip's temporal tail)
    float  invNeff;             // the fit's effective count inverse
    float3 phaseShift;          // -(gradLS * j): the innovation's phase
                                // correction, the tap's, the soft clip's
    float3 gradLSx;             // the LS plane gradient
    float3 gradLSy;
    float3 plane0;              // the plane at the target (the prior center)
    float  lever;               // the fitted-value lever at the target
    float3 covCross;            // the taps' cross-channel covariances
                                // (the metric's shape)
    float3 residSq;             // the post-plane residual (dof-corr)
    float  rangeSq;             // full-set squared AABB range (the spike
                                // reference)
    float3 priorTapCov;         // F1: Cov(fitted value, center tap)
    float3 transportVar;        // the gated landing error projected on the
                                // working gradient (MOTION-GATED)
    float  mixWeights[9];       // the plain kernel weights
};

// ============================================================================
// CONSTANTS
// ============================================================================
static const float kClipSigmaEmaRate         = 0.15;
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);
// The record's outlier guard: max(record, AABB range) + the measured
// transport scatter. r = 4 against a reference DOMINATED by the 3x3's
// observed range (~20 sigma^2 on Gaussian content): the spike is a
// "beyond everything the neighborhood saw" test -- the honest role for a
// hard reset.
static const float kClipSpikeRatio           = 4.0;
static const float kClipWinsorC              = 3.0;
static const float kGeoCorroborateChiSq      = 7.815;         // chi^2_3(0.95)
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
static const float kMaxLever                 = 4.0;
// ---- the posterior-mean soft clip ----
static const float kGhostLogOddsBase         = -6.2383;       // ln(1/512)
static const float kGhostMotionOdds          = 64.0;          // the one hand-set prior
static const float kGhostPriorSigma          = 1.0;
static const float kCusumTailWeight          = 0.1;
// ---- the structure split's transport ----
// The EMA rate for the split fraction. Steady-state noise = the per-frame
// fraction's std (~1, it is the split's own noise normalized by itself)
// times sqrt(rate/(2-rate)) ~ 0.31 at 0.2. Convergence ~5 frames after a
// reset -- the record's own timescale.
static const float kSplitEmaRate             = 0.2;
// The record ingest's effective chi^2 dof: the ingest is the pixel's own
// squared innovation, a single squared Gaussian = chi^2_1. (v5.2: v4.1's
// 0.5 was the relative-variance INVERSE -- half the dof.) MUST MATCH
// client/postFx/taa.lua's STUDENT_NUINF.
static const float kRecordIngestDof = 1.0;
// ---- the transport scatter's motion gate ----
static const float kTransportGatePx = 2.0;
// ---- the plane's one-step t-IRLS (v5.4) ----
// The fit's robustification dof. The drop-one trimmed scale makes the
// step self-discriminating: a surface or color step through the center
// texel leaves >= 3 coherent far-side taps in the scale (q ~ 2-3, ~15-20%
// reweighting -- the gate stays step-wide), while an isolated sparkle is
// 1-2 texels (excluded from the scale, q >> 1, strong downweighting).
// 4.5 keeps a coherent group at >= 80% weight and drops a 7-sigma
// isolated tap below 15%. Raise it if edges over-engage; lower it for
// harsher sparkle suppression.
static const float kPlaneIrlsDof = 4.5;

// ============================================================================
// THE STATISTICS
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

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float3 c  = taps[i];
        float2 o  = kOffsets3x3[i];
        float  wi = w[i];

        a.sum3     += c * wi;
        a.sumSq3   += c * c * wi;
        a.totalW   += wi;
        a.totalWSq += wi * wi;
        a.offSum   += o * wi;
        a.offXSq   += o.x * o.x * wi;
        a.offYSq   += o.y * o.y * wi;
        a.offXY    += o.x * o.y * wi;
        a.covXSum  += o.x * c * wi;
        a.covYSum  += o.y * c * wi;
        a.wXY += wi * c.x * c.y;
        a.wXZ += wi * c.x * c.z;
        a.wYZ += wi * c.y * c.z;
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

    // ---- the design weights: the plain kernel (stored unmodified as the
    //      soft clip's spatial prior) times the weighting modifiers -------
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
        stats.mixWeights[i] = w;

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

    // ---- the per-tap residuals and the DROP-ONE trimmed scale ----------
    // Dropping only the largest weighted residual makes the scale
    // group-aware: a coherent step keeps its group in the scale (the
    // far side's q stays moderate, the gate keeps its step width); an
    // isolated sparkle is excluded (its q explodes, its design weight
    // collapses). No classifier gating needed -- the geometry
    // self-discriminates.
    float3 res9[9];
    float  sumWr2 = 0.0;
    float  maxWr2 = 0.0;
    [unroll]
    for (int r = 0; r < 9; ++r)
    {
        float2 o = kOffsets3x3[r];
        res9[r] = neighborhoodColorSpace[r]
                - (mean1 + fit1.gradX * (o.x - cen1.x) + fit1.gradY * (o.y - cen1.y));
        float wr2 = w9[r] * dot(res9[r], res9[r]);
        sumWr2 += wr2;
        maxWr2  = max(maxWr2, wr2);
    }
    float dofCorr1    = 1.0 / max(1.0 - 3.0 * saturate(acc1.totalWSq * invW1 * invW1), 1.0 / 3.0);
    float sigmaTrimSq = max((sumWr2 - maxWr2) * invW1 * dofCorr1, kMinSigma * kMinSigma);

    // ---- the t-IRLS multipliers (the same Lange form as the tap's F5) --
    // w_k / v_k with v_k = sPer * (nu + q_k)/(nu + 1): the heteroscedastic
    // one-step refit. Clean taps are mildly up-weighted (q < 1), the
    // isolated outlier collapses. One step from the LS start is the
    // canonical point: full efficiency at the Gaussian, bounded influence
    // at the contamination.
    float w2[9];
    [unroll]
    for (int t = 0; t < 9; ++t)
    {
        float q = dot(res9[t], res9[t]) / sigmaTrimSq;
        w2[t] = w9[t] * ((kPlaneIrlsDof + 1.0) / (kPlaneIrlsDof + q));
    }

    // ---- PASS 2: the reweighted moments -- everything downstream -------
    ColorMomentAccum acc;
    AccumulateColorMoments(neighborhoodColorSpace, w2, acc);

    float invTotalWeight = 1.0 / max(acc.totalW, kEpsilon);
    float3 mean3         = acc.sum3 * invTotalWeight;
    stats.invNeff        = saturate(acc.totalWSq * invTotalWeight * invTotalWeight);
    float2 cenPx         = acc.offSum * invTotalWeight;

    {
        ColorPlaneFit fit = SolveColorPlane(acc, mean3, cenPx, invTotalWeight);
        stats.gradLSx = fit.gradX;
        stats.gradLSy = fit.gradY;
        stats.phaseShift = -(fit.gradX * jitterPx.x + fit.gradY * jitterPx.y);

        // The fitted value at the history's target (content position 0):
        // plane(0) = mean + g*(j - cen); variance sPer*(invNeff + lever)
        // (the v5.3 exact form), lever = the Mahalanobis quadratic of the
        // eval offset.
        float2 dEval = jitterPx - cenPx;
        float  leverFull = (fit.vYY * dEval.x * dEval.x - 2.0 * fit.vXY * dEval.x * dEval.y + fit.vXX * dEval.y * dEval.y)
                         / max(fit.detV, 1e-6);
        stats.lever  = min(max(leverFull, 0.0), kMaxLever);
        stats.plane0 = mean3 + fit.gradX * dEval.x + fit.gradY * dEval.y;

        float3 scc = acc.sumSq3 * invTotalWeight - mean3 * mean3;
        float3 ssr = scc
                   - 2.0 * (fit.gradX * fit.covX + fit.gradY * fit.covY)
                   + (fit.gradX * fit.gradX * fit.vXX + fit.gradY * fit.gradY * fit.vYY + 2.0 * fit.gradX * fit.gradY * fit.vXY);
        float dofCorr = 1.0 / max(1.0 - 3.0 * stats.invNeff, 1.0 / 3.0);
        stats.residSq = max(ssr * dofCorr, 0.0);

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
// CLIP-STATE TRANSPORT (tag 101, v5.3 layout)
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
        // (0.0) while a debug mode is active and re-converges in ~5 frames
        // after. The RECORD chain survives (sigma rides the payload).
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
// THE RECORD SEED (the model's per-channel E[i^2] -- ONE formula feeding
// the seed AND the split's share). The full quadratic form on the working
// LS gradient; with jitter off the phase term is naturally 0.
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
// THE RECORD'S PER-CHANNEL SPLIT
// ============================================================================
struct RecordVarianceParts
{
    float3 ePer;           // per-channel PRE-update E[i_k^2] (arming-capped)
    float3 varHPer;        // per-channel accumulator variance (READ semantics)
    float3 sPer;           // implied per-sample variance: the tap likelihood's
    float  spatialPriorSq; // the content-scale reference (the winsor anchor)
};

RecordVarianceParts SplitRecordVariance(
    ColorNeighborhoodStats stats,
    float recordSqTotal, float varHTotal, float nuRecord)
{
    RecordVarianceParts p;

    float3 varX = max(SeedRecordVarX(stats), float3(0.0, 0.0, 0.0));
    float  tot  = dot(varX, float3(1.0, 1.0, 1.0));
    float3 share = (tot > 1e-10)
        ? varX / tot
        : stats.sigmaClean * stats.sigmaClean / max(dot(stats.sigmaClean, stats.sigmaClean), kEpsilon);

    float spatialPriorSq = max(SeedRecordTotalSq(stats), kClipSigmaRecordFloorSq);

    // THE ARMING CAP at the record's LIVE dof (the exact chi^2_nu(0.999)/nu
    // bound, host-fit) -- binds the record AND the transported VarH share.
    float nuCap = max(nuRecord, 2.0);
    float armingCap = 1.0 + taaArmingA / sqrt(nuCap) + taaArmingB / nuCap;

    float recordSq = min(max(recordSqTotal, 0.0), armingCap * spatialPriorSq);
    float varHSq   = min(max(varHTotal, 0.0), armingCap * spatialPriorSq);
    p.ePer    = max(share * recordSq, float3(kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq));
    p.spatialPriorSq = spatialPriorSq;
    p.varHPer = share * varHSq;
    p.sPer    = max(p.ePer - p.varHPer, 0.0);
    return p;
}

// The whitened innovation on the READ-side denominators + the shrunk full
// metric (the geometric corroboration test's form).
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
    float  motionNormalized,
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

        // THE STRUCTURE SPLIT (v5.3): the transported, temporally
        // stabilized fraction times the live spatial residual. The clamp
        // acts on the stable side (see the file header).
        float3 tauB2 = max(splitFrac, 0.0) * stats.residSq;

        // THE FIT'S NOISE TERM (v5.3, exact): sPer-scaled through the
        // leverage quadratic; the structure is charged once, as tauB2.
        float3 vFit   = sPer * (stats.invNeff + stats.lever);
        float3 vPrior = tauB2 + vFit;

        float3 x0c = currentColorSpace - stats.phaseShift;
        float3 dev = x0c - stats.plane0;
        float3 qTap = (dev * dev) / max(vPrior + sPer, float3(1e-10, 1e-10, 1e-10));
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
        // the reconstruction), charged at nuResid. The EMA'd fraction's own
        // residual noise is NOT charged here: it manifests as the clamp's
        // positive-part mean, already carried in vPrior, and charging it
        // again would double-count. (If mode 13 shows over-engagement on
        // static smooth content, add tauB2^2 * kSplitEmaRate/(2-rate) /
        // max(splitFrac,0.04)^2 -- the documented revert.)
        float  nuResid = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
        float3 denomTau = tauB2 * tauB2 / nuResid;
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
        // full E[i^2] model), so vPrior = seed * (invNeff + lever) with the
        // phase folded in -- the form below. At straddled pixels the step
        // sits in residSq, so the cold width is step-inflated there (the
        // flat machinery's honest edge acceptance). ----
        float3 phase = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                     + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                     + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy);
        gateVarBase = stats.invNeff * (phase + stats.residSq);
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
        // ============ THE GHOST POSTERIOR ============
        // H0: the hierarchical null at its honest UN-INFLATED predictive
        // variance, on the SAME shrunk full metric as the hard test (F6)
        // with the exact relative log-det normalizer. H1: a PROPER mixture
        // -- the 8 NEIGHBOR TEXELS' accumulated content (what a
        // landing-miss ghost holds) plus one temporal tail. THE CENTER TAP
        // IS NOT A COMPONENT -- it IS the null's likelihood. Every
        // component is a full log-density: its own scale normalizer sits
        // INSIDE the logsumexp, and the null's +0.5*logDetNull is charged
        // once, outside.
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

        // The tap log-weights: the jitter-centered spatial prior
        // renormalized over the 8 neighbor components -- a PROPER mixture.
        float wSum8 = 0.0;
        [unroll]
        for (int w8 = 0; w8 < 9; ++w8)
            wSum8 += stats.mixWeights[w8];
        wSum8 = max(wSum8 - stats.mixWeights[0], 1e-4);
        float logTapW = log(1.0 - kCusumTailWeight) - log(wSum8);

        float lComp[8];
        float lMax = -1e30;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            // The phase-corrected component residual (the neighbor texel's
            // footprint-average estimate is tap_j + g*j).
            float3 dd = dCorr + (center - neighborhoodTaps[j]) + stats.phaseShift;
            float  qj = dot(dd * dd, invComp);
            lComp[j - 1] = log(max(stats.mixWeights[j], 1e-4)) + logTapW
                         - kT * log(1.0 + qj / nu)
                         - 0.5 * (log(vComp.x) + log(vComp.y) + log(vComp.z));
            lMax = max(lMax, lComp[j - 1]);
        }

        // The temporal tail: the old content absent from the 9-tap sample.
        float3 sigAC = stats.sigmaClean;
        float3 tailScale = nullScale + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invTail = float3(1.0, 1.0, 1.0) / tailScale;
        float qTail = dot(dCorr * dCorr, invTail);

        float lTail = log(kCusumTailWeight)
                    - kT * log(1.0 + qTail / nu)
                    - 0.5 * (log(tailScale.x) + log(tailScale.y) + log(tailScale.z));
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 8; ++j2)
            wSum += exp(lComp[j2] - lMax);
        wSum += exp(lTail - lMax);

        // llr = log p1 - log p0: the mixture's logsumexp, plus the null's
        // full -log p0 (the t-core and its log-det normalizer).
        float llrFrame = (lMax + log(max(wSum, 1e-10)))
                       + kT * log(1.0 + q0Comp / nu)
                       + 0.5 * logDetNull;

        // The prior odds: the per-pixel base plus the motion term (the one
        // hand-set prior -- a landing miss is likely at full motion).
        float logOdds = kGhostLogOddsBase
                      + log(1.0 + kGhostMotionOdds * saturate(motionNormalized))
                      + llrFrame;
        float p1 = 1.0 / (1.0 + exp(-logOdds));

        // The scalar pull -- the posterior-mean action, floored by the
        // hard ray clamp. The pull IS the eviction mechanism: the stored
        // history moves toward the best-explaining content at wPull per
        // frame, and the next frame reads the moved history.
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