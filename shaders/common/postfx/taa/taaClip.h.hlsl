// ============================================================================
// TAA history clipping v5.0: THE FLAT MODEL
// ----------------------------------------------------------------------------
// The hierarchical gate (v3.11 lineage, with the F-series exactness fixes)
// + the full-strength persistence detector (v4.1) + the plain single-sample
// record. No layer masks, no stored flags, no step model, no coverage
// machinery.
//
// THE GATE: the history's accumulation target is the phase-swept CENTER
// TEXEL; the gate tests the history against the posterior of that target
// given the current frame.
//   PRIOR: the LS plane at the target, variance vFit = residSq * (invNeff
//   + lever), widened by tauB2 = the temporally stable structural share
//   (max(residSq - sPer, 0)).
//   LIKELIHOOD: the phase-corrected center tap, variance sPer = the
//   record's implied per-sample variance (E[i^2] - varH) -- the identity
//   that makes the form self-calibrating: wherever the neighborhood has
//   no structural information the posterior degenerates EXACTLY to the
//   innovation test, whose null the record measures directly.
//   The tap is t-robustified (one IRLS step at the record's dof; the
//   exact (nu+q)/(nu+1) weight): a firefly widens the gate and stays out
//   of the center; structural deviations barely notice.
//   F1: the prior/tap shared-sample covariance is charged (both are
//   linear functionals of the same nine taps -- the center tap sits
//   inside the plane's fit; the independent form understates the tested
//   variance, flicker-side, ~19% radius at the white-texture limit).
//   At edges the plane residual carries the step, so the gate is
//   step-wide there -- the flat machinery's honest statement (the flip
//   REQUIRES that width; anything tighter flickers at the jitter
//   frequency).
//
// THE RECORD: the plain single-sample EMA of i^2. The innovation IS
// (c_t - cbar)*D + (eps - r) at edges -- its variance already contains
// the flip charge, so the raw second moment is the correct statistic with
// no step referencing (the coverage state analysis that motivated v3.x's
// MODEL-REFERENCED ingest; the plain form is safe because the DETECTOR,
// not the gate, owns edge-ghost eviction now). The ingest's noise is
// priced by kRecordIngestDof: the aged gate runs wider (retention is the
// gate's side; ghosting is the detector's side; they share no knob).
//
// THE PERSISTENCE DETECTOR: the pixel's OWN phase-corrected innovation,
// integrated over time (T: the luma EMA; Tc: the half-wave chroma vector
// magnitude against the measured null). THE FLIP IMMUNITY IS WHAT MAKES
// THE FLAT MODEL VIABLE AT EDGES: the phase flip is zero-mean over the
// low-discrepancy sequence (the partial-window mean is bounded by the
// discrepancy, |t| ~ 0.2 against a ~1.2 null), so the integral never
// fires on it -- while a persistent offset integrates at FULL strength
// (no spatial dilution: the v3.x common-mode estimator carried a
// single-texel trail at 0.11-0.33 in t-units depending on its landing
// alignment, leaving the visible 1.5-3.5 sigma class unconfirmed). The
// corrector is CONFIRMED-ONLY at full gain (confirmation substitutes for
// the sparse-prior shrinkage); the alarm force-replaces. ONE guard: kept
// dilation bands (the stored history legitimately holds the borrowed
// layer -- geometry, not persistence evidence).
//
// THE SOFT CLIP: beyond the gate radius the action is the exact
// posterior mean. H1 = the 8 NEIGHBOR TEXELS' accumulated content + one
// temporal tail; THE CENTER TAP IS NOT A COMPONENT (it IS the null's
// likelihood). The null's quadratic runs on the same shrunk full metric
// as the hard test (F6), with the exact relative log-det normalizer.
// p1 exports the CONFIRMED persistence evidence ONLY -- the frame-LLR
// posterior drives the clip action, never the blend (feeding it to the
// blend re-introduced the raw point sample at every moving edge).
//
// ---------------------------------------------------------------------------
// POST-MORTEM: THE COVERAGE MODEL (v4.0-v4.4), REMOVED
// ---------------------------------------------------------------------------
// C(p) = L(p) + c(p)*D, coverage measured from color, one 3x3 solve --
// mathematically sound, and removed on empirical verdict: with its
// actions disabled the system measured near-perfect AND the trail
// problem it was built for was already solved by the detector above.
// Three defect rounds, each real, each leaving a residue:
//   1. A void initializer (the LS residual regressed on the design --
//      identically zero) left the whole model dead for two versions.
//   2. The collinear overfit: soft content makes u ~ linear in position;
//      the (plane, step) split goes ill-conditioned and LS finds a
//      CANCELING solution (huge D, zero residual) that fits the nine
//      taps and explodes off them -- and conf measures explanation, not
//      identifiability, so the WORST fits reported the HIGHEST
//      confidence. Through Untonemap's pole at luma 1: white pixels.
//      Fixed by the uRes admission + cancellation-free magnitude belts.
//   3. The tap-grid quantization: hard edges give BINARY point samples,
//      so the staircase's stable-position estimate sawtooths +-0.5
//      texel at the jitter frequency. Every consumer that trusted it
//      beyond tap-grid resolution (the tight D-hat width, the clamp
//      margin, the geometric display pedestal) flickered and aliased.
//      Fixed by routing the display target through the measured sweep --
//      but the residue (oversharpening at partial commitment) closed the
//      case: the per-frame solve was still being asked sub-texel
//      questions.
// THE LESSONS FOR ANY SUCCESSOR (do not relearn these):
//   (a) the fit's legitimate resolution is the TAP GRID; the sub-texel
//       structure lives in the ACCUMULATION and must be read from time,
//       never from one frame's point samples;
//   (b) identifiability and explanation are different statistics -- gate
//       on the unexplained variance (uRes = det3/det2), never on conf;
//   (c) bound every fitted magnitude against a CANCELLATION-FREE
//       measurement of the same quantity (the outer-column contrast).
//
// TRANSPORT (tag 101, UNCHANGED): [31] sign, [30:28] tag, [27:22] sigma
// (6-bit), [21:16] Tc (0.25 quanta), [15:10] VarH ratio (6-bit), [9:6]
// age, [5:0] T (0.25 quanta).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host
// context: cbuffer perDraw (taaVarianceGamma, taaClipOvershoot,
// taaChromaVarianceMod, taaSoftClip, taaStudentA/B, taaArmingA/B,
// taaJitterPhaseEx/Ey/Exy, taaJitterAwareVariance, taaLumaVariance,
// taaVelocityAlignedVariance, taaHistoryOvershoot,
// taaClipDistanceRejection*, taaLumaDrift*, the feedback constants).
// Requires taaShared.h.hlsl, taaConstants.h.hlsl included before.
// ============================================================================
#ifndef TAA_CLIP_H_HLSL
#define TAA_CLIP_H_HLSL

// ============================================================================
// THE COLOR-FIELD STATISTICS (moments -> the coupled 2x2 plane)
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 aabbMin;             // full set (luma drift chroma gate)
    float3 aabbMax;
    float3 sigmaClean;          // the post-plane residual scale, per channel
                                // (drift caps, luma drift, chroma null)
    float  invNeff;             // the fit's effective count inverse
    float3 phaseShift;          // -(gradLS * j): the innovation's phase
                                // correction, the tap's, the soft clip's
    float3 gradLSx;             // the LS plane gradient
    float3 gradLSy;
    float3 plane0;              // the plane at the target (the prior center)
    float  lever;               // the fitted-value lever at the target
    float  phaseErrScale;       // j^T V^-1 j: the phase correction's own
                                // error scale (the color-side residual)
    float3 covCross;            // the taps' cross-channel covariances
                                // (the metric's shape)
    float3 residSq;             // the post-plane residual (dof-corr)
    float  rangeSq;             // full-set squared AABB range (the spike
                                // reference)
    float3 priorTapCov;         // F1: Cov(fitted value, center tap)
    float3 transportVar;        // the gated landing error projected on the
                                // working gradient (MOTION-GATED)
    float  mixWeights[9];       // the plain kernel weights
    float2 jitterPx;
};

// ============================================================================
// CONSTANTS
// ============================================================================
static const float kClipSigmaEmaRate         = 0.15;
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);
// The record's outlier guard: max(record, AABB range) + the measured
// transport scatter (READ-side vs WRITE-side). r = 4 gives ~0.74%/frame on
// balanced content, ~4.6% on rank-1. Retune ONLY with the mortality number.
static const float kClipSpikeRatio           = 4.0;
static const float kClipWinsorC              = 3.0;
static const float kGeoCorroborateChiSq      = 7.815;         // chi^2_3(0.95)
static const float kClipRejectionFeedbackFloor = 0.5;
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
static const float kResampleVarSq            = 0.0;
static const float kMaxLever                 = 4.0;
// ---- the posterior-mean soft clip ----
static const float kGhostLogOddsBase         = -6.2383;       // ln(1/512)
static const float kGhostMotionOdds          = 64.0;
static const float kGhostPriorSigma          = 1.0;
static const float kCusumTailWeight          = 0.1;
// ---- the persistence detector ----
static const float kGhostEmaRate   = 0.15;
static const float kGhostEmaNorm   = 3.5119; // sqrt((2-rate)/rate)
static const float kGhostConfirmT  = 4.0;
static const float kGhostAlarmT    = 6.5;
static const float kGhostNullStdSq = 1.5;    // PLACEHOLDER -- measure via mode 11
static const float kGhostEvCoef    = 0.25 / kGhostNullStdSq;
// ---- the drift corrector (CONFIRMED-ONLY; the full-gain ramp) ----
static const float kDriftCapSigmas  = 3.0;   // the magnitude BELT (content sigmas)
// ---- the transport scatter's motion gate ----
static const float kTransportGatePx = 2.0;
// The record ingest's effective per-frame sample count: the ingest is the
// pixel's own squared innovation, Var(i^2)/E[i^2]^2 = 2 (the squared
// Gaussian's kurtosis), and the varFrac convention (per-frame relative
// variance = 1/r) gives r = 1/2 -> nu_inf ~ 6.2. The HONEST dof of the
// single-sample EMA: the aged Student factor widens accordingly, and the
// widened gate's shelter is supervised by the full-strength detector.
static const float kRecordIngestDof = 0.5;

// ============================================================================
// THE STATISTICS
// ============================================================================
ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx, float2 transportErrPx)
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
    stats.jitterPx = jitterPx;
    stats.priorTapCov = float3(0.0, 0.0, 0.0);
    stats.transportVar = float3(0.0, 0.0, 0.0);

    // ---- the moments --------------------------------------------------------
    float3 weightedSum   = float3(0.0, 0.0, 0.0);
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);
    float  totalWeight   = 0.0;
    float  totalWeightSq = 0.0;
    float2 weightedOffsetSum = float2(0.0, 0.0);
    float  weightedOffXSqSum = 0.0, weightedOffYSqSum = 0.0, weightedOffXYmSum = 0.0;
    float3 weightedCovXSum   = float3(0.0, 0.0, 0.0);
    float3 weightedCovYSum   = float3(0.0, 0.0, 0.0);
    float  wSumXY = 0.0, wSumXZ = 0.0, wSumYZ = 0.0;
    float  motionFactor = saturate(motionNormalized);

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

        weightedSum       += tapColorSpace * w;
        weightedSumSq     += tapColorSpace * tapColorSpace * w;
        totalWeight       += w;
        totalWeightSq     += w * w;
        weightedOffsetSum += tapOffsetPx * w;
        weightedOffXSqSum += tapOffsetPx.x * tapOffsetPx.x * w;
        weightedOffYSqSum += tapOffsetPx.y * tapOffsetPx.y * w;
        weightedOffXYmSum += tapOffsetPx.x * tapOffsetPx.y * w;
        weightedCovXSum   += tapOffsetPx.x * tapColorSpace * w;
        weightedCovYSum   += tapOffsetPx.y * tapColorSpace * w;
        wSumXY += w * tapColorSpace.x * tapColorSpace.y;
        wSumXZ += w * tapColorSpace.x * tapColorSpace.z;
        wSumYZ += w * tapColorSpace.y * tapColorSpace.z;
        w9[i] = w;
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    float3 mean3        = weightedSum * invTotalWeight;
    stats.invNeff       = saturate(totalWeightSq * invTotalWeight * invTotalWeight);
    float2 cenPx        = weightedOffsetSum * invTotalWeight;

    // ---- the coupled 2x2 plane, solved on the CENTERED second-moment
    //      matrix (the exact LS plane; an uncentered solve is shrunk by
    //      (I + C^-1 c c^T)^-1 whenever the weighted centroid is nonzero,
    //      which the jitter-centered weights are) ----------------------------
    {
        float  vXX = max(weightedOffXSqSum * invTotalWeight - cenPx.x * cenPx.x, 1e-6);
        float  vYY = max(weightedOffYSqSum * invTotalWeight - cenPx.y * cenPx.y, 1e-6);
        float  vXY = weightedOffXYmSum * invTotalWeight - cenPx.x * cenPx.y;
        float3 covX = weightedCovXSum * invTotalWeight - cenPx.x * mean3;
        float3 covY = weightedCovYSum * invTotalWeight - cenPx.y * mean3;
        float  detV = vXX * vYY - vXY * vXY;
        float3 lsGX, lsGY;
        if (detV > 1e-6)
        {
            lsGX = (vYY * covX - vXY * covY) / detV;
            lsGY = (vXX * covY - vXY * covX) / detV;
        }
        else
        {
            // degenerate spread (1D): the decoupled fit, exact when vXY = 0
            lsGX = covX / vXX;
            lsGY = covY / vYY;
        }
        stats.gradLSx = lsGX;
        stats.gradLSy = lsGY;
        stats.phaseShift = -(lsGX * jitterPx.x + lsGY * jitterPx.y);

        // The fitted value at the history's target (content position 0 =
        // the texel center): plane(0) = mean + g*(j - cen). Its exact
        // estimation variance is sigma^2 * (invNeff + lever) with the
        // lever the Mahalanobis quadratic of the eval offset.
        float2 dEval = jitterPx - cenPx;
        float  leverFull = (vYY * dEval.x * dEval.x - 2.0 * vXY * dEval.x * dEval.y + vXX * dEval.y * dEval.y)
                         / max(detV, 1e-6);
        stats.lever  = min(max(leverFull, 0.0), kMaxLever);
        stats.plane0 = mean3 + lsGX * dEval.x + lsGY * dEval.y;

        float3 scc = weightedSumSq * invTotalWeight - mean3 * mean3;
        float3 ssr = scc
                   - 2.0 * (lsGX * covX + lsGY * covY)
                   + (lsGX * lsGX * vXX + lsGY * lsGY * vYY + 2.0 * lsGX * lsGY * vXY);
        float dofCorr = 1.0 / max(1.0 - 3.0 * stats.invNeff, 1.0 / 3.0);
        stats.residSq = max(ssr * dofCorr, 0.0);

        // F1: the prior/tap shared-sample covariance. plane0 - T = ebar +
        // (g_hat - g)*dEval and x0c - T = e0 + (g_hat - g)*j over the SAME
        // taps, so c = residSq * invW * (w0 + dEval^T V^-1 (j - w0*cen)).
        // Under the static kernel dEval = j: both functionals carry the
        // IDENTICAL gradient term and the second summand is its full
        // variance -- the charge matters in both weight configurations.
        float crossQuad = 0.0;
        if (detV > 1e-6)
        {
            float2 vInv0 = float2( vYY, -vXY) / detV;
            float2 vInv1 = float2(-vXY,  vXX) / detV;
            float2 cRhs  = jitterPx - w9[0] * cenPx;
            crossQuad = dEval.x * dot(vInv0, cRhs) + dEval.y * dot(vInv1, cRhs);
        }
        stats.priorTapCov = max(stats.residSq * invTotalWeight * (w9[0] + crossQuad), float3(0.0, 0.0, 0.0));

        // the phase correction's error scale: Var((g_hat - g)*j) =
        // residSq * j^T V^-1 j (the exact LS covariance of the gradient).
        stats.phaseErrScale = max((vYY * jitterPx.x * jitterPx.x
                                 - 2.0 * vXY * jitterPx.x * jitterPx.y
                                 + vXX * jitterPx.y * jitterPx.y) / max(detV, 1e-6), 0.0);

        // the transport scatter, projected on the LS gradient (the ML
        // estimate of the content slope)
        stats.transportVar = (transportErrPx.x * transportErrPx.x) * (lsGX * lsGX)
                           + (transportErrPx.y * transportErrPx.y) * (lsGY * lsGY);

        stats.sigmaClean = max(sqrt(stats.residSq), kMinSigma);
        stats.covCross = float3(wSumXY * invTotalWeight - mean3.x * mean3.y,
                                wSumXZ * invTotalWeight - mean3.x * mean3.z,
                                wSumYZ * invTotalWeight - mean3.y * mean3.z);
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
// THE CHROMA MAGNITUDE'S NULL MOMENTS (measured, at the local Y-Z correlation)
// ============================================================================
void GhostChromaNullMoments(ColorNeighborhoodStats stats, out float nullMean, out float nullVar)
{
    float meanRho0 = 1.4720;                        // iid-t numeric (+-1%)
    float meanRho1 = 1.2990;                        // 3*sqrt(3)/4, exact
    float varRho0  = 0.8332;                        // 3 - meanRho0^2 (the E[m^2] identity)
    float varRho1  = 1.3125;                        // 21/16, exact
    float rDof = max(1.0 / max(stats.invNeff, 1e-3) - 1.0, 2.0);
    float rho = clamp(ShrinkCorrelation(stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon),
                                        max(rDof - 2.0, 1.0)), -0.95, 0.95);
    float r2 = saturate(rho * rho);
    nullMean = lerp(meanRho0, meanRho1, r2);
    nullVar  = lerp(varRho0, varRho1, r2);
}

// ============================================================================
// CLIP-STATE TRANSPORT (tag 101 -- UNCHANGED)
// ============================================================================
static const float kVarHCodeLogMin  = -6.0;
static const float kVarHCodeLogSpan = 9.0;

float EncodeVarHRatio(float varH, float recordSq)
{
    float ratio = varH / max(recordSq, kEpsilon);
    float code  = round((log2(max(ratio, 1e-8)) - kVarHCodeLogMin) * (63.0 / kVarHCodeLogSpan));
    return clamp(code, 0.0, 63.0);
}

float DecodeVarHRatio(float code, float recordSq)
{
    float ratio = exp2(code * (kVarHCodeLogSpan / 63.0) + kVarHCodeLogMin);
    return ratio * recordSq;
}

float PackClipStateAlpha(bool revoked, float sigma, float varHRatioCode, float age,
                         float ghostCodeC, float ghostCode)
{
    uint u = 0x50000000u
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x3Fu)        << 22)
           | ((uint(clamp(ghostCodeC, 1.0, 63.0)) & 0x3Fu)           << 16)
           | ((uint(varHRatioCode) & 0x3Fu)                          << 10)
           | ((uint(clamp(age, 0.0, kClipMaxAge)) & 0xFu)            << 6)
           |  (uint(clamp(ghostCode, 1.0, 63.0)) & 0x3Fu);
    return asfloat(revoked ? (u | 0x80000000u) : u);
}

void DecodeClipState(float alphaValue, out float sigmaSq, out float varHSq,
                     out float recordAge, out float ghostT, out float ghostTc)
{
    sigmaSq = -1.0;
    varHSq = 0.0;
    recordAge = 0.0;
    ghostT = 0.0;
    ghostTc = 0.0;

    uint u   = asuint(alphaValue);
    uint tag = (u >> 28) & 0x7u;
    uint code;
    float varHCode = 28.0;   // ratio ~0.25: the near-stationary default
    if      (tag == 0x5u)
    {
        code = (u >> 22) & 0x3Fu;
        varHCode = (float)((u >> 10) & 0x3Fu);
        recordAge = (float)((u >> 6) & 0xFu);
        ghostT  = ((float)(u & 0x3Fu) - 32.0) * 0.25;
        ghostTc = ((float)((u >> 16) & 0x3Fu) - 32.0) * 0.25;
    }
    else if (tag == 0x1u)
    {
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
    return max(dot(SeedRecordVarX(stats), float3(1.0, 1.0, 1.0)) + 3.0 * kResampleVarSq, kClipSigmaRecordFloorSq);
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
    p.sPer    = max(p.ePer - p.varHPer - kResampleVarSq, 0.0);
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
    float  p1;              // the blend floor's input: the CONFIRMED
                            // persistence evidence ONLY
};

ClipGateResult ClipHistoryToStatisticGate(
    float3 historyColorSpace,
    float3 currentColorSpace,
    ColorNeighborhoodStats stats,
    RecordVarianceParts parts,
    bool   recordLive,
    float  recordAge,
    float  motionNormalized,
    float3 neighborhoodTaps[9],
    float  ghostEvNats)
{
    ClipGateResult r;
    r.center = stats.plane0;
    r.tGate = 1.0;
    r.rawGateVar = float3(kMinSigma * kMinSigma, kMinSigma * kMinSigma, kMinSigma * kMinSigma);
    r.mdd = 0.0;

    // The sequential evidence floor (the confirmed persistence only).
    float seqFloor = 0.0;
    if (taaSoftClip > 0.001 && ghostEvNats > 0.0)
    {
        float logOddsSeq = kGhostLogOddsBase + ghostEvNats;
        seqFloor = 1.0 / (1.0 + exp(-logOddsSeq));
    }
    r.p1 = seqFloor;

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

        float3 tauB2 = max(stats.residSq - sPer, float3(0.0, 0.0, 0.0));
        float3 vFit  = stats.residSq * (stats.invNeff + stats.lever);
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
        float  nuResid = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
        float3 denomTau = stats.residSq * stats.residSq / nuResid
                        + sPer * sPer / nuRecordSafe;
        float3 nuPrior = vPrior * vPrior
                       / max(vFit * vFit / nuResid + denomTau, float3(1e-10, 1e-10, 1e-10));
        nuPrior = max(nuPrior, float3(2.0, 2.0, 2.0));

        float3 nuFlat = (A + B) * (A + B)
                      / max(A * A / nuRecordSafe + B * B / nuPrior, float3(1e-10, 1e-10, 1e-10));
        nuFlat = max(nuFlat, float3(2.0, 2.0, 2.0));

        float3 studentS = 1.0 + taaStudentA / nuFlat + taaStudentB / (nuFlat * nuFlat);
        gateVar = max(gateVarBase * studentS, float3(minGateVar, minGateVar, minGateVar));

        float gSum = gateVar.x + gateVar.y + gateVar.z;
        nuPost = max(dot(nuFlat, gateVar) / max(gSum, 1e-10), 2.0);
    }
    else
    {
        // ---- the cold path: the plane prior, deliberately NOT a coverage
        // claim -- after a reset/spike the policy is to pull the history
        // hard toward the live neighborhood and re-establish trust. At
        // straddled pixels the step sits in residSq, so the cold width is
        // step-inflated there (the flat machinery's honest edge
        // acceptance). ----
        float3 phase = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                     + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                     + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy);
        gateVarBase = stats.invNeff * (phase + stats.residSq);
        gateVar = gateVarBase;
        center = stats.plane0;
        nuPost = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
    }

    // The transport scatter (added post-Studentization by design; the
    // record measures the realized innovation within its EMA lag).
    float3 transportTerm = stats.transportVar;
    gateVar += transportTerm;
    r.rawGateVar = max(gateVarBase + transportTerm, float3(minGateVar, minGateVar, minGateVar));

    float3 dCorr = historyColorSpace - center;
    r.center = center;

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
        // with the exact relative log-det normalizer. H1: the 8 NEIGHBOR
        // TEXELS' accumulated content (what a landing-miss ghost holds) +
        // one temporal tail. THE CENTER TAP IS NOT A COMPONENT -- it IS
        // the null's likelihood.
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

        bool jitW = (taaJitterAwareVariance > 0.5);
        float lj[8];
        float lMax = -1e30;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            // The phase-corrected component residual (the neighbor texel's
            // footprint-average estimate is tap_j + g*j).
            float3 dd = dCorr + (center - neighborhoodTaps[j]) + stats.phaseShift;
            float  qj = dot(dd * dd, invComp);
            float2 dj = kOffsets3x3[j] - stats.jitterPx;
            float  lw = jitW ? (-dot(dj, dj)) : log(max(stats.mixWeights[j], 1e-4));
            lj[j - 1] = -kT * log(1.0 + qj / nu) + lw + logTapW;
            lMax = max(lMax, lj[j - 1]);
        }

        // The temporal tail: the old content absent from the 9-tap sample.
        float3 sigAC = stats.sigmaClean;
        float3 tailScale = nullScale + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invTail = float3(1.0, 1.0, 1.0) / tailScale;
        float qTail = dot(dCorr * dCorr, invTail);

        float lTailNorm = 0.5 * (logDetNull - (log(vComp.x) + log(vComp.y) + log(vComp.z)));
        float lTail = log(kCusumTailWeight) - kT * log(1.0 + qTail / nu) - lTailNorm;
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 8; ++j2)
            wSum += exp(lj[j2] - lMax);
        wSum += exp(lTail - lMax);

        float llrFrame = (lMax + log(max(wSum, 1e-10))) + kT * log(1.0 + q0Comp / nu) + lTailNorm;

        float logOdds = kGhostLogOddsBase
                      + log(1.0 + kGhostMotionOdds * saturate(motionNormalized))
                      + ghostEvNats
                      + llrFrame;
        float p1 = 1.0 / (1.0 + exp(-logOdds));

        // The scalar pull -- the posterior-mean action, floored by the
        // hard ray clamp.
        float wPull = saturate(p1);
        wPull = max(wPull, 1.0 - tHard);

        r.clippedColorSpace = historyColorSpace - wPull * dCorr;
        r.tGate = 1.0 - wPull;
        // p1 stays seqFloor: the blend floor exports the CONFIRMED
        // persistence evidence ONLY.
        r.p1 = seqFloor;
    }
    else
    {
        r.clippedColorSpace = historyColorSpace - (1.0 - tHard) * dCorr;
        r.tGate = tHard;
    }
    return r;
}

// ============================================================================
// CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / stats.sigmaClean;
    float maxChannelDistance = max(clipDistance.x, max(clipDistance.y, clipDistance.z));
    return saturate((maxChannelDistance - taaClipDistanceRejectionMinError) * taaClipDistanceRejectionAmount);
}

void ApplyLumaDriftCorrection(inout float3 clippedHistorySpace, float3 currentColorSpace, ColorNeighborhoodStats stats)
{
    if (taaLumaDriftStrength <= 0.001)
        return;

    float lumaBias     = clippedHistorySpace.x - currentColorSpace.x;
    float relativeBias = lumaBias / max(currentColorSpace.x, kLumaDriftLumaFloor);

    float chromaSpread = max(stats.aabbMax.y - stats.aabbMin.y, stats.aabbMax.z - stats.aabbMin.z);
    float chromaGate   = 1.0 - saturate((chromaSpread - taaLumaDriftChromaTol) / kLumaDriftChromaFadeWidth);

    if (abs(relativeBias) > kLumaDriftRelThreshold && abs(lumaBias) > kLumaDriftAbsThreshold)
        clippedHistorySpace.x -= lumaBias * taaLumaDriftStrength * chromaGate;

    clippedHistorySpace.x = max(clippedHistorySpace.x, 0.0);
}

float ComputeMotionFeedback(HistoryReprojection repro)
{
    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    return clamp(lerp(taaFeedbackMax, taaFeedbackMin, motionDrop), taaFeedbackMin, taaFeedbackMax);
}

float ComputeHistoryFeedback(HistoryReprojection repro, float clipDistanceRejection, bool planarSurface)
{
    float feedback = ComputeMotionFeedback(repro);

    float rejected = min(feedback, kClipRejectionFeedbackFloor);
    feedback = lerp(feedback, rejected, clipDistanceRejection);

    if (planarSurface)
    {
        feedback -= taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment);
        feedback = max(feedback, 0.0);
    }
    return feedback;
}

#endif // TAA_CLIP_H_HLSL