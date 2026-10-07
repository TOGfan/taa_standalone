// ============================================================================
// TAA history clipping: color statistics, the temporal clip-state transport
// (sigma + age + fiction), the Mahalanobis statistic gate, and the feedback
// consumers
// ----------------------------------------------------------------------------
// THE CLIP (design): the accumulator is the ESTIMATOR, h_{t+1} = (1-a)h_t +
// a x_t, and the clip is a robust innovation gate on the STATISTIC -- a
// coverage statement about the only two random objects:
//     h   the accumulator:   Var(h)  = a/(2-a) Var(x)      (iid limit)
//     i_t = x_t - h_{t-1}:   E[i^2]  = 2/(2-a) Var(x)
//         =>  Var(h) = (a/2) E[i^2]                        -- EXACT
//     mu  the phase-mean estimate (the jitter-aware tap mean):
//         Var(mu) = invNeff Var(x), invNeff = SumW2/SumW^2 (measured live)
//
// THE PER-CHANNEL NORMALIZATION (fixed in this revision -- read first): the
// record transports the TOTAL E[|i|^2] (summed over the working-space
// channels); the Mahalanobis test needs the PER-CHANNEL variance
// cStat * E[i_k^2]. The previous form paid the full total to every channel:
// 3x the true per-channel variance under isotropy (chi = 2.8 actually gated
// at chi*sqrt(3) ~ 4.85 sigma) and unboundedly loose on the quiet channels
// of luma-dominant content. The gate now splits the total by the live
// spatial covariance's diagonal:
//     share_k = sigma_k^2 / tr(Cov),   gateVar_k = (...) * E * share_k
// exact per channel under the design's own stationarity model
// (E[i_k^2] = 2/(2-a) Var(x_k); sigma_k^2 estimates Var(x_k)). share is
// computed from the ANISO-CAPPED sigma (AnisoClampSigma): temporally-white,
// spatially-correlated noise on a quiet channel would otherwise be
// under-covered by its share (chroma shimmer); the cap bounds per-channel
// tightening at kAnisoSigmaCap^-2 of the loudest channel.
//
// THE CORRECTED TEST VECTOR: the gate (and the anti-alignment) test
//     d_corr = h - mu + beta * (centroid - u)
// with u = the snap residual and centroid = the ACTUAL weighted centroid of
// the tap kernel. The taps sample the stable field SHIFTED by -u, so for a
// linear field mu = f(S + centroid - u): mu's residual phase displacement is
// beta*(centroid - u), and it is ADDED to d to cancel it. The jitter-aware
// kernel's TRUNCATED centroid tracks only ~0.81 of the phase at corner
// phases, so the un-corrected d carried a ~(1-kappa)*beta*u wobble. Exact
// for linear fields, any kernel, any weight mode (non-jitter-aware:
// centroid ~ 0 -> the full -beta*u correction -- that mode's "center
// wanders with the phase" shimmer mechanism disappears with it). The gate's
// OUTPUT is anchored at the history:
//     clipped = h - (1 - tGate) * d_corr
// tGate = 1 is a bit-exact passthrough; tGate = 0 lands on the phase-free
// AA estimate (mu - gatePhaseShift) instead of the phase-carrying mu.
//
// THE INNOVATION, SPLIT BY CONSUMER (unchanged): i = PHASE (beta*u: kept in
// the raw second moment -- the record; subtracted for every mean-structure
// test: i_corr) + RESAMPLE (the Kaiser's reconstruction error: the record's
// footprint variance) + DISCREPANCY (the anti-alignment's subject).
// MIXTURES ARE CONTENT: the stats run on the FULL tap set, always.
// DILATION SANCTITY: every dilation zone is exempt from the anti-alignment;
// the fiction machinery owns traveling bands.
//
// THE SAMPLING LAW (Studentization -- now on the correctly normalized
// statistic): mdd/3 = [chi^2_3/3]/[chi^2_nu/nu] ~ F(3, nu) under the null;
// the exact threshold at coverage p is 3*F_{3,nu}(p). nu(t) Satterthwaite
// with the seed's declared prior dof (nu0 = 4), fit by the host as
// S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2, exact at nu = 4 and
// 12.33 (mid-range error ~1.4% at chi = 2.8, up to ~6% at chi >= 3.5 -- an
// immaterial comfort-gate regime). The COLD path is deliberately NOT
// Studentized: it is the change-point replacement, not a coverage statement.
//
// THE ACCUMULATOR'S TRANSIENT (exact, per channel): Var(h_t) =
// decayH*Var(h_0) + (1-decayH)*(a/2)*E_k with Var(h_0) = Var(mu_prev) =
// muVar, relaxing at (1-a)^{2t}. The gate variance is
//     gateVar_k = muVar_k*(1 + decayH) + (a/2)*E_k*(1 - decayH)
// (2*muVar at t = 0: exactly Var(mu_prev - mu_cur)).
//
// THE MU SHARE, conservative or residual-scoped (taaClipScopedMu, blended):
//   * conservative (0): muVar_k = invNeff*(1-a/2)*E_k -- mu modelled as an
//     invNeff-attenuated draw of the full x distribution. Exact for
//     noise-dominated content; over-pays the phase energy on edges (the
//     safe direction: a noiseless edge ran ~3x true, chi_eff ~ 5, ghosts
//     below ~0.5x local contrast riding).
//   * residual-scoped (1): muVar_k = invNeff * residSq_k, the weighted-LS
//     residual variance around the local linear model (noise + curvature,
//     dof-corrected for the 3 parameters the plane fit absorbs). Equals the
//     conservative value on noise; collapses toward the floor on clean
//     ramps and the gate recovers its nominal chi (ghost threshold ~0.2x
//     local contrast). SELF-COVERING on straddles: the central-difference
//     phase gradient underestimates the true slope there, so residSq
//     inflates by the missing-slope energy -- exactly the scale of the
//     phase correction's own error. Floors: (kScopedResidFrac*sigma)^2.
//   scoped = 0 reduces EXACTLY to the previous cStat*E*share form.
//
// THE SEED: the spatial prior tr(Cov_taps) -- the new regime's MEASURED
// spread, never the change-point's own squared residual (which re-armed the
// gate to the ghost's scale). Over-seeds edges ~5x for ~7 frames: benign
// (the reset itself already pulled h onto mu; the seed only governs
// post-reset coverage).
//
// THE GHOST SIGNATURE (anti-alignment; the test lives in the resolve, the
// constants here): mu between x and h, i_corr anti-aligned with d_corr.
// Guards (standardized per channel by the aniso-capped sigma -- the old
// trCov forms were luma-scaled and blocked pure-chroma ghosts):
//   * corroboration: Sum(i_corr_k^2/sigma_k^2) > chi^2_3(0.80) = 4.64
//   * distance:      Sum(d_corr_k^2/sigma_k^2)  > 0.75
//     (both identical to the old forms under isotropy)
//   * dilation exemption. Null fires: rare on edges (the distance guard),
//     cheap on flat noise (cold ~= live there).
//
// THE WINSOR CAP stays record-referenced (C^2 E_prev, the Student-t
// predictive bound): engages at chi^2_3 > 27 (~6e-6 under Gaussian) --
// negligible bias, outlier protection only.
//
// CHANGE-POINTS, in order of authority (all route to the cold gate, the
// spatial seed, age 0): (1) the anti-alignment reset; (2) THE SHOCK --
// innovation > kClipSpikeRatio (4x) times BOTH the record and the
// neighborhood's squared AABB range (a heuristic conjunction, conservative
// through the range term -- not a derived quantile); (3) the fiction
// sentinel; (4) geometry as evidence, not command (corroborated at
// chi^2_3(0.95)/3 = 2.6).
//
// TRANSPORT LAYOUT (the output alpha, non-debug): [31] revocation sign (the
// motion-field writer reads ONLY this); [30:27] tag 0110; [26:20] the state
// sigma (7-bit log2, TOTAL across channels, code 0 = cold); [19:8] the
// acutance target, 12 bits SQRT-COMPRESSED (packed linear [0,1] <->
// energy [0,9]; the old linear packing saturated at E = 1 -- every
// full-contrast LDR edge and all HDR content) and TEMPORALLY STABILIZED
// (EWMA at rate kSharpEwmaRate against the previous frame's decoded value
// at the landing -- the raw measurement swings ~2x across the jitter phase
// cycle on edges and would flicker the sharpener's boost); [7:1] the
// record's age; [0] the fiction flag. WHILE DEBUG IS ON the debug payload
// (tag 0111) carries the sigma in its low 7 bits. Read back bit-exactly
// through the DEDICATED POINT-SAMPLED history binding.
//
// SAFETY LAYERS: (1) the accumulator share floored at kStatAlphaFloor;
// (2) the record floored at kClipSigmaRecordFloorSq; (3) the winsor cap;
// (4) the Student + transient inflations (the whole live gateVar is
// inflated by S(nu), including the mu term -- a deliberate simplification
// in the safe direction); (5) the soft clip; (6) the aniso cap and the
// scoped-residual floor. The emitted value lies on the segment
// [phase-free AA estimate, history]; it may exceed the tap AABB BY DESIGN.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (clipping + feedback constants, taaStudentA/B,
// taaClipScopedMu). Requires fragments included before: taaShared.h.hlsl
// (kClipSigmaRef, ClipPackSigmaCode, ClipUnpackSigmaCode), taaConstants,
// taaFrame.h.hlsl (HistoryReprojection).
// ============================================================================
#ifndef TAA_CLIP_H_HLSL
#define TAA_CLIP_H_HLSL

// ============================================================================
// SMALL CLIPPING MATH HELPERS
// ============================================================================
// Soft-clip scale factor: 1.0 = hard clip; grows toward (1 + softClipAmount)
// as the requested overshoot (in gate-radius units) grows; motion restores
// hard clip.
float SoftClipUnitScale(float overshootUnits, float softClipAmount, float motionFactor)
{
    float softLimit = 1.0 + softClipAmount * (1.0 - exp2(-(overshootUnits - 1.0) * kLog2E));
    return lerp(softLimit, 1.0, motionFactor);
}

// ============================================================================
// COLOR NEIGHBORHOOD STATISTICS
// ============================================================================
// The FULL tap set, always: the AABB (spike test + the luma-drift chroma
// gate), the mean (gate center, NaN fallback), sigma (cold gate, smear
// rejection, spatial shape), invNeff (the mean's estimator share), the
// phase corrections and the LS residual (the scoped mu share).
struct ColorNeighborhoodStats
{
    float3 aabbMin;
    float3 aabbMax;
    float3 mean;
    float3 sigma;
    float  invNeff;         // SumW2 / SumW^2: Var(mu) = invNeff * Var(x)
    float3 phaseShift;      // beta * u: the INNOVATION's phase bit (u = -fracPx sense)
    float3 gatePhaseShift;  // beta * (centroid - u): mu's residual phase displacement
                            // (the GATE/anti-alignment test-vector correction)
    float3 residSq;         // per-channel weighted-LS residual variance around the
                            // local linear model, dof-corrected (noise + curvature)
};

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx)
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    bool jitterPaddingEnabled = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 weightedSum   = float3(0.0, 0.0, 0.0);
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);
    float  totalWeight   = 0.0;
    float  totalWeightSq = 0.0;
    float  motionFactor  = saturate(motionNormalized);

    // Weighted-offset accumulators: the kernel's ACTUAL centroid, the
    // per-axis offset variances (LS slopes) and the offset cross-moments.
    float2 weightedOffsetSum = float2(0.0, 0.0);
    float  weightedOffXSqSum = 0.0;
    float  weightedOffYSqSum = 0.0;
    float3 weightedCovXSum   = float3(0.0, 0.0, 0.0);   // Sum w * offx * c
    float3 weightedCovYSum   = float3(0.0, 0.0, 0.0);   // Sum w * offy * c

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    // The LS-residual machinery (the scoped mu share's input) is skipped
    // entirely when scoping is off -- a per-draw uniform branch. residSq is
    // then unused (the gate's lerp blend factor is 0).
    bool clipScopedOn = (taaClipScopedMu > 0.001);

    float w9[9];

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        stats.aabbMin = min(stats.aabbMin, tapColorSpace);
        stats.aabbMax = max(stats.aabbMax, tapColorSpace);

        float2 offsetFromCenterPx = tapOffsetPx - weightCenterPx;
        // BRANCH, not ternary: a ternary materializes the exp2 even when the
        // standard weights are selected (the condition is a per-draw uniform).
        float w;
        if (jitterCenteredWeights)
            w = exp2(-dot(offsetFromCenterPx, offsetFromCenterPx) * kLog2E);
        else
            w = kStdWeights[i];

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
        if (clipScopedOn)
        {
            w9[i] = w;   // stored only for the LS second pass (gated with it)
            weightedOffXSqSum += tapOffsetPx.x * tapOffsetPx.x * w;
            weightedOffYSqSum += tapOffsetPx.y * tapOffsetPx.y * w;
            weightedCovXSum   += tapOffsetPx.x * tapColorSpace * w;
            weightedCovYSum   += tapOffsetPx.y * tapColorSpace * w;
        }
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean    = weightedSum * invTotalWeight;
    stats.sigma   = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));
    stats.invNeff = saturate(totalWeightSq * invTotalWeight * invTotalWeight);

    // ---- kernel geometry: the ACTUAL weighted centroid ----------------------
    // NB: named kernelCentroidPx, NOT 'centroid' -- 'centroid' is a reserved
    // GLSL interpolation qualifier and the engine's HLSL->GLSL front end
    // lexes it as a keyword even inside .hlsl files (that was the compile
    // failure). weightCenterPx above is the INTENDED kernel center; this is
    // the actual one (grid truncation included -- it tracks only ~0.81x the
    // phase at corner phases).
    float2 kernelCentroidPx = weightedOffsetSum * invTotalWeight;

    // ---- gradient estimator (1): central differences ------------------------
    // The established phase-bit regressor. The taps sample the stable field
    // SHIFTED by -u, so for a linear field:
    //   x  = f(S - u),  mu = f(S + kernelCentroidPx - u)
    //     -> i = x - h ~= -beta*u                             (phaseShift cancels it)
    //     -> d = h - mu ~= -beta*(kernelCentroidPx - u)   (gatePhaseShift cancels it)
    // Auto-adapts to both weight modes (non-jitter-aware: centroid ~ 0 -> the
    // full -beta*u); exact for linear fields; its own error on straddles is
    // covered by residSq below.
    float3 gradCDx = 0.5 * (neighborhoodColorSpace[4] - neighborhoodColorSpace[3]);
    float3 gradCDy = 0.5 * (neighborhoodColorSpace[2] - neighborhoodColorSpace[1]);
    stats.phaseShift     = -(gradCDx * jitterPx.x + gradCDy * jitterPx.y);
    stats.gatePhaseShift = gradCDx * (kernelCentroidPx.x - jitterPx.x)
                         + gradCDy * (kernelCentroidPx.y - jitterPx.y);

    // ---- gradient estimator (2): weighted-LS slopes (decoupled normal
    // equations; the small offset cross-covariance is neglected, which
    // UNDER-explains -> OVER-estimates the residual -> the safe direction).
    // NOT used for the phase corrections. The whole estimator + residual
    // pass is gated on the scoping uniform (see clipScopedOn above).
    float3 lsGX = 0.0;
    float3 lsGY = 0.0;
    if (clipScopedOn)
    {
        float  vXX = max(weightedOffXSqSum * invTotalWeight - kernelCentroidPx.x * kernelCentroidPx.x, 1e-6);
        float  vYY = max(weightedOffYSqSum * invTotalWeight - kernelCentroidPx.y * kernelCentroidPx.y, 1e-6);
        float3 covX  = weightedCovXSum * invTotalWeight - kernelCentroidPx.x * stats.mean;
        float3 covY  = weightedCovYSum * invTotalWeight - kernelCentroidPx.y * stats.mean;
        lsGX = covX / vXX;
        lsGY = covY / vYY;
    }

    // ---- the LS residual: noise + curvature, dof-corrected -----------------
    // Second pass over the (in-register) taps. The mean squared residual of a
    // 3-parameter plane fit under-estimates the per-tap residual variance by
    // the dof the fit absorbed; N_eff = 1/invNeff corrects, clamped at 3x
    // (corner-phase kernels have N_eff ~ 3.7 -- their invNeff is
    // correspondingly larger, which partially self-corrects).
    if (clipScopedOn)
    {
        float3 ssr = float3(0.0, 0.0, 0.0);
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float2 dOff = kOffsets3x3[j] - kernelCentroidPx;
            float3 fit  = stats.mean + lsGX * dOff.x + lsGY * dOff.y;
            float3 r    = neighborhoodColorSpace[j] - fit;
            ssr += w9[j] * (r * r);
        }
        float dofCorr = 1.0 / max(1.0 - 3.0 * stats.invNeff, 1.0 / 3.0);
        stats.residSq = max(ssr * invTotalWeight * dofCorr, 0.0);
    }
    else
    {
        stats.residSq = 0.0;   // unused: the gate's lerp blend is 0
    }

    // Firefly clamp: pulls the AABB (used by the luma-drift chroma gate and
    // the spike test) into the mean +/- k*sigma band.
    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    // POPOVICIU: variance <= (max - min)^2 / 4 for anything supported on
    // [min, max]. A no-op by construction on unclamped neighborhoods.
    stats.sigma = min(stats.sigma, 0.5 * (stats.aabbMax - stats.aabbMin));

    float spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

    // Jitter flicker padding (a manual stand-in for the temporal memory on
    // the paths where the record is absent; zero it once trusted).
    if (jitterPaddingEnabled)
    {
        float3 padGradX = jitterPx.x > 0.0 ? (neighborhoodColorSpace[4] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[3] - neighborhoodColorSpace[0]);
        float3 padGradY = jitterPx.y > 0.0 ? (neighborhoodColorSpace[2] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[1] - neighborhoodColorSpace[0]);
        float3 expectedJitterShift = (padGradX * abs(jitterPx.x)) + (padGradY * abs(jitterPx.y));

        float paddingFade   = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - motionNormalized) : 1.0;
        float paddingAmount = taaJitterFlickerPadding * paddingFade;

        if (taaDirectionalVariance > 0.5) { stats.sigma += abs(expectedJitterShift * paddingAmount); }
        else                              { stats.sigma += (spatialContrast * length(jitterPx) * paddingAmount); }
    }

    stats.sigma = max(stats.sigma, kMinSigma);
    return stats;
}

// ============================================================================
// CLIP CONSTANTS (the sigma encoding lives in taaShared -- two transports)
// ============================================================================
static const float kClipSigmaEmaRate         = 0.15;          // rho: the record's EMA rate (~7-frame re-warm)
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);  // the record's minimum carried VARIANCE (TOTAL across channels). NOT the transport's resolution -- the deliberate floor the change-point tests and the winsor cap normalize against.
static const float kStatAlphaFloor           = 0.10;          // accumulator-share floor (blind-phase coverage)
static const float kClipSpikeRatio           = 4.0;           // innovation shock: >4x the record AND the AABB range = change-point. A heuristic conjunction (the range term tracks the population's tail, so heavy-tailed legit content never shocks) -- conservative, not a derived quantile.
static const float kClipWinsorC              = 3.0;           // E's winsorization cap, in sigma_hat of the PREVIOUS record (Student-t predictive outlier bound; record-referenced so a straddling texel's blind-phase room survives)
static const float kClipCorroborateRatio     = 2.6;           // geometric-reject corroboration, in units of the carried E[i^2]: chi^2_3(0.95)/3
static const float kAlignCos                 = 0.5;           // anti-alignment cone: dot(i_corr,d) < -0.5|i||d| (the slack absorbs ghost+phase cross terms)
// Standardized anti-alignment guards (chi^2_3 units, per channel by the
// aniso-capped spatial sigma; applied by the resolve). Both are the exact
// isotropic equivalents of the old tr(Cov) forms:
static const float kAlignDistChiSq           = 0.75;          // was |d|^2 > 0.25*tr(Cov)
static const float kAlignSpatialChiSq        = 4.642;         // chi^2_3(0.80); was i_corr^2 > 1.55*tr(Cov)
static const float kClipRejectionFeedbackFloor = 0.5;         // the clip-distance rejection's INDEPENDENT floor: a full-rejection event injects a half-weight current sample. Deliberately NOT taaFeedbackMin.
// Per-channel shape cap: the gate share / guard normalization floors each
// channel's sigma at this fraction of the loudest channel. Bounds the
// tightening on quiet channels (spatially-correlated temporal noise
// insurance) at kAnisoSigmaCap^-2 in variance.
static const float kAnisoSigmaCap            = 0.10;
// Scoped-mu residual floor, as a fraction of the (aniso-capped) channel
// sigma: leakage insurance for the LS residual estimate.
static const float kScopedResidFrac          = 0.10;
// Acutance transport: the EWMA rate of the stabilized target (the resolve
// blends the fresh measurement with the previous decoded value at the
// landing). ~4-frame stabilization.
static const float kSharpEwmaRate            = 0.25;
// Studentization: nu(t) = 1 / [D/nu0 + sigmaW2 (1-D)], D = (1-rho)^{2t},
// sigmaW2 = rho/(2-rho) = 0.0811, 2*log2(1-rho) = -0.46893 (rho = 0.15).
// S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2 is HOST-FIT to the exact
// F(3,nu) quantiles at nu = 4 and 12.33, consuming the EFFECTIVE radius
// chi*(1+clipOvershoot). The host's STUDENT_RHO / STUDENT_NU0 must match
// the two constants below.
static const float kClipStudentPriorDof      = 4.0;           // the SEED's declared prior dof (a 9-tap spread ~ 4 honest dof after correlation)
static const float kClipStudentVarFrac       = 0.0810811;     // rho/(2-rho)
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 127.0;         // the age transport's saturation

// The per-channel spatial sigma, capped against extreme anisotropy: every
// channel is floored at kAnisoSigmaCap of the loudest. Consumers: the
// gate's per-channel share and the standardized anti-alignment guards.
float3 AnisoClampSigma(float3 sigma)
{
    float sMax   = max(sigma.x, max(sigma.y, sigma.z));
    float sFloor = max(sMax * kAnisoSigmaCap, kMinSigma);
    return max(sigma, float3(sFloor, sFloor, sFloor));
}

// ============================================================================
// CLIP-STATE TRANSPORT (the output alpha; layout in the file header)
// ============================================================================
// acutanceLinear: the SQRT-COMPRESSED, TEMPORALLY STABILIZED acutance target
// in [0,1] (energy in [0,9]). The resolve computes it; this only packs it.
float PackClipStateAlpha(bool revoked, float sigma, float age, float acutanceLinear, bool fictionAdvected)
{
    uint u = 0x30000000u                                      // tag 0110
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x7Fu)    << 20)
           | ((uint(saturate(acutanceLinear) * 4095.0 + 0.5) & 0xFFFu) << 8)
           | ((uint(clamp(age, 0.0, kClipMaxAge)) & 0x7Fu)       << 1)
           |  (fictionAdvected ? 0x1u : 0x0u);
    return asfloat(revoked ? (u | 0x80000000u) : u);
}

// Full state decode. sigmaSq < 0 when no valid state is present. Accepts BOTH
// transports: the clip-state alpha (tag 0110: sigma at [26:20], age at
// [7:1], fiction at [0]) and the debug payload (tag 0111: sigma only; the
// age and the fiction flag re-warm after debug -- the age from 0, the
// maximum-inflation direction).
void DecodeClipState(float alphaValue, out float sigmaSq, out float recordAge, out bool fictionAdvected)
{
    sigmaSq = -1.0;
    recordAge = 0.0;
    fictionAdvected = false;

    uint u   = asuint(alphaValue);
    uint tag = (u >> 27) & 0xFu;
    uint code;
    if      (tag == 0x6u)                     // clip-state alpha
    {
        code = (u >> 20) & 0x7Fu;
        recordAge = (float)((u >> 1) & 0x7Fu);
        fictionAdvected = ((u & 0x1u) != 0u);
    }
    else if (tag == 0x7u)                     // debug payload: sigma only
    {
        code = u & 0x7Fu;
    }
    else
    {
        return;                               // foreign alpha: cold
    }

    if (code == 0u) return;                   // explicit cold marker
    float s = ClipUnpackSigmaCode(code);
    sigmaSq = s * s;
}

// ============================================================================
// THE MAHALANOBIS GATE
// ============================================================================
struct ClipGateResult
{
    float3 clippedColorSpace;
    float  tGate;             // 1 = untouched; < 1 = shrunk (soft clip included)
};

ClipGateResult ClipHistoryToStatisticGate(
    float3 historyColorSpace,
    ColorNeighborhoodStats stats,
    float  sigmaStatSq,        // the temporal record (TOTAL E[|i|^2]); < 0 = cold -> tap-based gate
    float  recordAge,          // frames since the record's seed (transported)
    float  statAlpha,          // the accumulator's motion-based blend weight (a)
    float  motionNormalized)   // soft-clip restore only -- the gate itself has no motion term
{
    ClipGateResult r;
    bool recordLive = (sigmaStatSq >= 0.0);

    float alphaStat = max(statAlpha, kStatAlphaFloor);

    // ---- per-channel shape: the record is a TOTAL; split it by the live
    // spatial covariance's diagonal (see the header). Aniso-capped.
    float3 sigA     = AnisoClampSigma(stats.sigma);
    float  traceSig = max(dot(sigA, sigA), kEpsilon);
    float3 share    = (sigA * sigA) / traceSig;
    float3 ePer     = share * max(sigmaStatSq, kClipSigmaRecordFloorSq);

    // ---- the accumulator transient (exact, per channel):
    // Var(h_t) = decayH*Var(h_0) + (1-decayH)*(a/2)*E_k, Var(h_0) = muVar.
    float decayH = exp2(recordAge * 2.0 * log2(max(1.0 - alphaStat, 0.5)));

    // ---- THE STUDENT FACTOR: the record's effective dof (Satterthwaite) and
    // the host-fit exact-threshold inflation. Unset constants read 0 -> S=1.
    float decayPrior = exp2(recordAge * kClipStudentDecay);   // (1-rho)^{2t}
    float nu         = 1.0 / (decayPrior / kClipStudentPriorDof
                            + kClipStudentVarFrac * (1.0 - decayPrior));
    float invNu      = 1.0 / nu;
    float studentS   = 1.0 + taaStudentA * invNu + taaStudentB * invNu * invNu;

    // ---- the mu share: conservative or residual-scoped (see the header).
    float3 muVarFull    = stats.invNeff * (1.0 - alphaStat * 0.5) * ePer;
    float3 residFloor   = sigA * kScopedResidFrac;
    float3 muVarScoped  = stats.invNeff * max(stats.residSq, residFloor * residFloor);
    float3 muVar        = lerp(muVarFull, muVarScoped, saturate(taaClipScopedMu));

    // chromaScale multiplies the TOTAL gate variance per channel (up to
    // sqrt(3) radius headroom on luma-dominant content for a moderate mod).
    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);

    // LIVE: [muVar*(1+decayH) + (a/2)*E_k*(1-decayH)] * S(nu) -- the Student
    // factor is applied to the whole live variance (a deliberate
    // simplification in the safe direction; only the record-derived parts
    // strictly carry nu's uncertainty). COLD: the mean's own estimator
    // uncertainty, per channel -- the change-point replacement, deliberately
    // tight (NOT Studentized).
    float3 gateVar = recordLive
        ? (muVar * (1.0 + decayH) + (alphaStat * 0.5) * ePer * (1.0 - decayH)) * studentS
        : (stats.invNeff * stats.sigma * stats.sigma);

    float3 invGateVar = 1.0 / max(gateVar * (chromaScale * chromaScale),
                                  (kMinSigma * kMinSigma).xxx);

    // chi = the NOMINAL coverage radius (taaVarianceGamma; chi_3(0.95) =
    // 2.7959 -- now actually true per channel), with taaClipOvershoot as
    // multiplicative radius slack. The Student inflation makes the nominal
    // TRUE.
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    // The PHASE-CORRECTED test vector: d_corr = h - mu + beta*(centroid - u).
    float3 dCorr = historyColorSpace - stats.mean + stats.gatePhaseShift;
    float  mdd   = dot(dCorr * dCorr, invGateVar);
    float  tGate = (mdd > 1e-20) ? min(chi / sqrt(mdd), 1.0) : 1.0;

    if (tGate < 1.0 && taaSoftClip > 0.0)
    {
        float overshootUnits = 1.0 / max(tGate, kEpsilon);       // Mahalanobis overshoot
        float softScale      = SoftClipUnitScale(overshootUnits, taaSoftClip, saturate(motionNormalized));
        tGate = min(softScale * tGate, 1.0);
    }
    r.tGate = tGate;

    // Shrink along the CORRECTED displacement, anchored at the history:
    // tGate = 1 is a bit-exact passthrough (an un-fired gate never touches
    // the accumulator); tGate = 0 lands on the phase-free AA estimate
    // (mean - gatePhaseShift). Deliberately NOT clamped to the tap AABB:
    // exceeding it is exactly the sub-texel room the record purchased.
    r.clippedColorSpace = historyColorSpace - (1.0 - tGate) * dCorr;
    return r;
}

// ============================================================================
// CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
// Normalized by the SPATIAL sigma (the neighborhood's own spread), NOT the
// gate's variance: normalizing by the record would disarm smear rejection
// precisely on ghosts. The blend's event-relative responsive channel.
float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / stats.sigma;
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

// Motion-only feedback: the accumulator's blend weight at clip time (the
// statistic-scale input for the gate).
float ComputeMotionFeedback(HistoryReprojection repro)
{
    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    return clamp(lerp(taaFeedbackMax, taaFeedbackMin, motionDrop), taaFeedbackMin, taaFeedbackMax);
}

// History feedback. The rejection can only LOWER feedback; the alignment
// drop is applied AFTER the clamp and ONLY on planar surfaces.
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