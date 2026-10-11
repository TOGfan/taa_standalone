// ============================================================================
// TAA history clipping: color statistics (mixture + same-depth-layer clean
// + the coverage step model), the temporal clip-state transport (sigma +
// VarH + age + the persistence statistics), the Mahalanobis gate,
// and the feedback consumers
// ----------------------------------------------------------------------------
// THE CLIP: the accumulator is the ESTIMATOR, h_{t+1} = (1-a)h_t + a x_t;
// the clip is a robust innovation gate on the STATISTIC:
//     Var(h) = (a/2) E[i^2],   E[i^2] = 2/(2-a) Var(x)   (iid limit)
// Two clocks: the record's INFORMATION clock (age -> Student dof) and the
// accumulator's STATE clock (VarH, transported, damped-resample recursion).
//
// ARCHITECTURE (top-down):
//   * The 9-tap record/drift estimator (taa.fx) feeds the record's spatial
//     seed, the drift predict step, and the persistence detector. Its tap
//     mask is the same-depth layer partition MINUS the stored motion-field
//     edge flags. Its per-channel spatial estimate is the MEASURED
//     composition the record split distributes.
//   * THE STATISTIC GATE (v3.11: THE HIERARCHICAL NULL): the history's
//     accumulation target is the phase-swept CENTER TEXEL, not the
//     neighborhood mean -- the blend input is always the center texel, so
//     the accumulator converges to that texel's footprint average. The gate
//     therefore tests the history against the POSTERIOR of that target
//     given the current frame:
//       - PRIOR: the same-layer plane at the target, with the exact LS
//         fitted-value variance vFit = residSq * (invNeff + lever) (the
//         lever is the Mahalanobis quadratic of the eval offset), widened
//         by the structural prior tauB2 = the temporally STABLE share of
//         the spatial residual, MEASURED as max(residSq - sPer, 0) --
//         exact for smooth structure (the full residual), for white
//         texture (zero) and for the independent-phase footprint; errs
//         low mid-range, priced through the dof) -- plus the straddle's
//         coverage-mixture floor.
//       - LIKELIHOOD: the phase-corrected center tap, variance sPer = the
//         record's own implied per-sample variance (E[i^2] - varH). This
//         identity is what makes the form self-calibrating: wherever the
//         neighborhood has no structural information (thin detail, white
//         texture, pure flip steps) the posterior degenerates EXACTLY to
//         the innovation test (h - x0c at variance varH + sPer), whose
//         null the record measures directly.
//       - the tap's likelihood is t-robustified (one IRLS step at the
//         record's dof, the deviation standardized against the TOTAL
//         predictive width): a firefly widens the gate and stays out of
//         the center; structural deviations (q ~ O(1)) barely notice.
//     This replaces the mean-estimation model. On static textured content
//     the old test charged invNeff*residSq for the texel-scale structural
//     term -- the kernel cannot interpolate structure at the correlation
//     length it varies at -- which was 2-4x tight (the fine-detail
//     shimmer). The prior/tap error cross-covariance (positive by
//     Cauchy-Schwarz on the footprint average, zero at both structure
//     limits, ~0.1-0.2 tau^2 mid-range) is deliberately unmodeled: the
//     independent form then slightly OVERSTATES the posterior variance
//     (history-survival-side), and the Satterthwaite dof pricing absorbs
//     the mid-range remainder.
//   * THE PERSISTENCE DETECTOR: two transported scalars (T: the EMA of the
//     drift estimator's luma t; Tc: the EMA of the half-wave chroma drift
//     vector magnitude). The chroma magnitude's null moments are MEASURED
//     from the local Y-Z correlation (the endpoints are the exact moments
//     of the established t marginal).
//   * THE STRADDLE MODEL: while off-layer mass is partitioned out, the
//     clip target is the pixel's OWN layer's value (the clean prior
//     package; the step fit's levels are the SAME LS planes), and the step
//     model blends the coverage target in by its adequacy. The acceptance
//     keeps the coverage-mixture uncertainty at EVERY adequacy
//     (covVarFloor, inside the prior width). The transport scatter is
//     projected on the OWN-LAYER LS gradient (the full-set CD charged the
//     landing error across the step).
//   * THE CARRYOVER CAPS: the record AND its transported VarH share are
//     arming-capped against the current spatial prior (the SEED -- the
//     spatial E[i^2] estimate, phase term included) at the record's LIVE
//     dof: the exact chi^2_nu(0.999)/nu bound, host-fit as the constant
//     pair taaArmingA/B.
//   * THE SOFT CLIP (v3.11.1: the belief runs on UN-INFLATED predictives):
//     beyond the gate radius the action is the exact posterior mean. The
//     Student factor is the hard test's coverage device and is NOT part of
//     the belief -- the t at the Satterthwaite dof prices the variance
//     estimation noise there instead, and the hard ray clamp still bounds
//     the action. H1 is the empirical alternative = the 8 NEIGHBOR TEXELS'
//     accumulated content (what a landing-miss ghost holds; variance
//     varH + sPer -- the per-sample law, on steps too) + one temporal
//     tail. THE CENTER TAP IS NOT A COMPONENT -- it IS the null's
//     likelihood. Component centers are phase-corrected (tap + g*j: the
//     neighbor texel's footprint-average estimate) and the log-density
//     bookkeeping carries the exact normalizers for the unequal
//     component/null scales.
//
// TRANSPORT (tag 101): [31] sign, [30:28] tag, [27:22] sigma (6-bit),
// [21:16] the chroma persistence statistic Tc (0.25 quanta, code =
// round(4Tc) + 32; nonnegative by the half-wave input, so codes 32-63),
// [15:10] VarH ratio (6-bit), [9:6] age, [5:0] the luma persistence
// statistic T (0.25 quanta, code = round(4T) + 32; the sign is the luma
// direction).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (the clipping + feedback constants, taaStudentA/B,
// taaArmingA/B, taaJitterPhaseEx/Ey/Exy, taaClipGhostReset,
// taaCusumFlipAcc, taaCusumNoiseAcc, taaVarhResampleLoss,
// taaVarhWhiteLoss, taaDriftCompensation, taaDriftMaxGain,
// taaJitterAwareVariance). Requires taaShared.h.hlsl, taaConstants.h.hlsl,
// taaFrame.h.hlsl included before.
// ============================================================================
#ifndef TAA_CLIP_H_HLSL
#define TAA_CLIP_H_HLSL

// ============================================================================
// COLOR NEIGHBORHOOD STATISTICS
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 aabbMin;             // full set (luma drift chroma gate)
    float3 aabbMax;
    float3 mean;                // the same-layer mean on straddles (the NaN
                                // fallback's reference); the step target
                                // rides stepTarget now
    float3 sigma;               // full set (mixture): smear rejection
    float3 sigmaClean;          // same-layer subsample (the B level, raw)
    float  invNeff;             // the WORKING prior package's effective
                                // count inverse (the same-layer count on
                                // healthy straddles; the full set when the
                                // C-subset is too small to carry its own
                                // plane)
    float  invNeffClean;        // the C-subset's count (raw)
    float  invNeffF;            // the F-side effective count inverse (the
                                // step dof input; 1.0 when no off-layer
                                // mass -- the safe divide)
    float3 phaseShift;          // the innovation's phase correction: the
                                // LS gradient form on flats, the step
                                // model's form on engaged steps (the
                                // ingest's step reference)
    float3 phaseShiftPlane;     // the PURE plane form -(gLS * j): never
                                // step-overloaded; the gate's tap
                                // correction and the soft clip's component
                                // reference
    float3 gradLSx;             // the working LS plane gradient (same-layer
    float3 gradLSy;             // on healthy straddles; the seed / cold
                                // gate / transport / fallback phase
                                // consumers)
    float3 gradCDx;             // central differences, full set (the step
    float3 gradCDy;             // fit's edge-normal direction input)
    float  effRank;             // r in [1,3]: the per-draw dof
    float3 covCross;            // the taps' cross-channel covariances
                                // (xy, xz, yz) -- the same-layer moments on
                                // straddles; the measured correlation SHAPE
                                // the gate's full metric consumes
    float3 residSq;             // the WORKING post-plane residual
                                // (dof-corr; same-layer on healthy
                                // straddles, the full set on degenerate
                                // ones -- see the rebase)
    float  rangeSq;             // full-set squared AABB range
    float  rangeSqClean;        // same-layer squared range
    bool   straddled;           // any off-layer tap (mask valid)
    // ---- the prior package (the hierarchical gate's neighborhood side) ----
    float3 plane0;               // the working LS plane evaluated at the
                                 // history's target (the texel center)
    float  lever;               // the fitted-value lever at the target
                                 // (the exact Mahalanobis quadratic of
                                 // the eval offset, capped)
    // ---- the coverage step fit ----
    float  stepW;               // the adequacy blend [0,1]
    float3 stepTarget;          // the engaged coverage target (B0 + cEff*dFB);
                                 // = mean where the step never engaged
    float3 stepTargetVar;       // the effective target variance (flip
                                // charge, varC, levels, and the
                                // mode-blend cross term included)
    float3 stepTargetStable;    // B0 + cStable*dFB (the ingest / phase-bit
                                // reference; the STABLE coverage)
    float3 stepDFB;             // the corrected step (F0 - B0)
    float  stepC;               // the fit's own target coverage (telemetry
                                // only; no decision consumes it)
    float  stepCStable;         // the widened bracket's mid coverage
    float3 stepXModel;          // B0 + proj0*dFB (the classified sample)
    float3 stepResidVarB;       // the within-layer residual variances
    float3 stepResidVarF;
    float  stepPhaseVarScale;   // the landing-phase residual's variance share
                                // (jrN^2)
    float3 transportVar;        // the measured transport scatter's
                                // per-channel color variance -- the gated
                                // landing-error vector projected on the
                                // WORKING LS gradient (same-layer on
                                // straddles), MOTION-GATED
    float  cusumStepDof;        // the levels' estimation dof, nB + nF - 4
    float  mixWeights[9];       // the plain kernel weights (the ghost
                                // mixture's spatial prior; the estimator's
                                // tap weights)
    float  recordSampleDof;     // the record estimator's per-frame dof: the
                                // same-layer count over the FULL 3x3 minus
                                // the stored-edge-flagged taps
    float3 covVarFloor;         // the coverage-mixture uncertainty's
                                // per-channel variance (straddles only; 0
                                // otherwise) -- the honest acceptance floor
    float  covSpanSq;           // the coverage span squared, total (the
                                // spike test's range reference on
                                // straddles)
    float2 jitterPx;            // the sub-texel jitter phase (the stats' own
                                // input; consumed by the posterior's
                                // closed-form log-weights)
};

// ============================================================================
// CLIP CONSTANTS
// ============================================================================
// THE record's bias-variance dial (named, not derived): a variance
// CHANGE-POINT is unmodelled, so the EMA trades record lag (ghost shelter
// when Var drops, flicker pressure when it rises) against estimate noise.
// rho = 0.15 is the chosen operating point.
static const float kClipSigmaEmaRate         = 0.15;
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);
// The record's outlier guard. DERIVATION (honest, both regimes): the test
// compares the corrected innovation against max(record, range). On ordinary
// content the AABB range dominates (range^2 ~ 25x E[i^2] per channel on a
// Gaussian flat) and the test fires only when the history lands OUTSIDE the
// current neighborhood. In the record-dominated regime the record converges
// to E[i^2] and the mortality is the chi^2_3 tail at 3*ratio: r = 4 ->
// P(chi^2_3 > 12) ~= 6.2%/frame (cheap: resets re-seed from the spatial
// model and keep the Student dof low -- anti-flicker there); a 0.1% target
// would need r ~= 7.4. 4.0 keeps the spike's content-change latency low;
// retune ONLY with the mortality number, not by eye.
static const float kClipSpikeRatio           = 4.0;
static const float kClipWinsorC              = 3.0;
static const float kGeoCorroborateChiSq      = 7.815;         // chi^2_3(0.95), whitened
static const float kClipRejectionFeedbackFloor = 0.5;
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
// The record-vs-prior arming cap is the host-fit chi^2_nu(0.999)/nu at the
// record's LIVE dof (taaArmingA/B, applied as 1 + A/sqrt(nu) + B/nu) --
// see the host module's ARMING CAP FIT block. Retires the fixed 5.0 (whose
// "EMA-lag headroom" was arbitrary: a RISING variance never binds the cap
// -- the record lags below the current prior -- and the falling case is
// exactly the band-carryover fiction the cap exists to bound, where the
// honest quantile is the right threshold).
// The resample's variance effect is carried by the multiplicative damping
// in the VarH recursion (taaVarhResampleLoss / taaVarhWhiteLoss); the
// additive slot stays 0.
static const float kResampleVarSq            = 0.0;
// The post-plane residual's collapse guard (the step fit's level variances
// on synthetic flats): 10% of sigma_clean.
static const float kScopedResidFrac          = 0.10;
// The adversarial lever cap: the exact fitted-value quadratic form is
// unbounded on rank-deficient designs; 4.0 matches the estimator's
// phase-error cap (the same class of bound).
static const float kMaxLever                 = 4.0;
// ---- the posterior-mean soft clip ----
static const float kGhostLogOddsBase         = -6.2383;       // ln(1/512)
// The motion term of the ghost prior, DERIVED as an order-of-magnitude
// population ratio (the one underived population number left --
// calibratable from mode-11 telemetry): at an ENGAGED swept step the
// stale-landing probability is the off-coverage ~1/8; on flats it is the
// base 1/512. The ratio 64 = (1/8)/(1/512) is the prior-odds boost at full
// motion, and it only matters where the frame evidence is ambiguous (the
// frame LLR overwhelms it on clean nulls and clean ghosts alike).
static const float kGhostMotionOdds          = 64.0;
// The temporal tail width, in units of the local CLEAN-layer sigma.
static const float kGhostPriorSigma          = 1.0;
// ---- the coverage step fit ----
// The step model's admission SNR (dFB^2 / the levels' error variance).
// The zero point is the levels' 2-sigma separation (sig = 4): below it the
// step is indistinguishable from the levels' own noise; the ramp gives
// half-credit at 6 and full admission above 8. (This is also the BF~1
// crossover of the levels-vs-flat model selection to first order.)
static const float kStepMinLevels = 4.0;
// The crossing bracket's maximum width in texels (the 3x3's diagonal
// reach) -- a geometric support bound.
static const float kStepMaxSpan   = 3.0;
// The direction-aware jitter span floor: the sequence's projected variance
// (JitterProjVar) can collapse toward zero on axis-aligned edges with
// symmetric sequences; the floor keeps the coverage inversion finite.
static const float kMinJitterSpan = 0.35;
// ---- THE PERSISTENCE DETECTOR ----
// Two transported scalars per pixel. T: the EMA of the drift estimator's
// per-frame LUMA spatial t, scaled so the H0 law is a pinned ~N(0,1)
// (self-normalized: a variance-model error cancels between numerator and
// denominator). Tc: the EMA of the half-wave CHROMA drift vector magnitude.
// Thresholds are set against the t_6 input's variance inflation
// (nu/(nu-2) = 1.5 -> the normalized null's effective std is ~1.22, not 1
// -- kGhostNullStdSq below carries this number for the evidence mapping):
//   |T| >= kGhostConfirmT (4.0): ~1e-3 engagement, EMA-smooth. The action
//        is the drift-corrector UNLOCK -- the ghost-trail eviction engine.
//   |T| >= kGhostAlarmT   (6.5): ~1e-7; the blend floor approaches full
//        replacement; mode 3 additionally hard-replaces.
static const float kGhostEmaRate   = 0.15;   // the detectors' EMA rate (memory ~13 frames)
static const float kGhostEmaNorm   = 3.5119; // sqrt((2-rate)/rate): the EMA's null std -> unit
static const float kGhostConfirmT  = 4.0;    // the confirmation threshold (|T| luma / Tc chroma)
static const float kGhostAlarmT    = 6.5;    // the deep-confirmation / alarm threshold
static const float kGhostUnlockThrSigmas = 0.75; // the CONFIRMED drift threshold, in estimator sigmas (vs 2.0 unconfirmed)
// The luma statistic's REALIZED null variance. The per-frame t input is
// t_dof at dof ~ 6 (9 same-layer taps - mean - 2 gradient dof), variance
// nu/(nu-2) = 1.5; kGhostEmaNorm normalizes iid UNIT-variance inputs, so
// T's null variance is 1.5 (std ~ 1.2247). The |T| thresholds above are
// set against it -- and the EVIDENCE mapping (kGhostEvCoef) prices against
// it too. The EMA's cross-frame correlation (the innovations are
// near-independent frame to frame; only the slowly-varying variance
// estimates correlate them, a bounded ~O(a) per-lag effect) is
// second-order against this and is not separately derived.
static const float kGhostNullStdSq = 1.5;
// ln BF = T^2 / (4 sigma0^2) - 0.5 ln 2  (tau = sigma0 Gaussian prior),
// priced against the detector's REALIZED null (sigma0^2 =
// kGhostNullStdSq).
static const float kGhostEvCoef = 0.25 / kGhostNullStdSq;   // 1/6
// ---- THE CHROMA PERSISTENCE CHANNEL ----
// A second transported scalar, Tc: the EMA of the half-wave chroma drift
// VECTOR magnitude, extending the detector's AXIS coverage to the trails
// the luma channel cannot see -- saturated chroma smears with sub-floor
// luma deltas (colored lights, emissives on neutral backgrounds). The
// per-frame input is
//     tC = max( length(driftMean.yz / sqrt(driftVar.yz))
//               - nullMean(rhoYZ), 0.0 ) * sqrt(kGhostNullStdSq / nullVar(rhoYZ))
// over the estimator's own per-channel scales. The VECTOR magnitude is
// DIRECTION-BLIND-FREE. The HALF-WAVE at the null mean makes the channel
// ONE-SIDED (quiet chroma is never evidence) and keeps gray content at
// Tc ~ 0 instead of a deeply negative baseline a new trail would have to
// climb out of. The CENTERING is essential: a magnitude's null mean is
// nonzero, and an uncentered input would confirm everywhere.
// The null moments are MEASURED per pixel from the local Y-Z correlation
// (GhostChromaNullMoments below). CALIBRATION (the mode-11 telemetry; the
// channel-separation swap is documented at the dbgB site). FLOOR: the
// channel's detection floor is a chroma vector magnitude of ~2.5 per-axis
// sigma_D (~2.2x the luma floor) -- the mission is saturated chroma
// trails, not depth parity with luma.
// The temporal-tail component's prior mass, shared by the gate's posterior
// alternative (8 neighbor texels + 1 tail).
static const float kCusumTailWeight   = 0.1;
// ---- the drift predict step ----
static const float kDriftCapSigmas  = 3.0;   // the magnitude BELT (content sigmas)
// The soft threshold (estimator sigmas). Its position relative to the
// Studentized gate radius sets the smooth-lighting band; 2.0 engages the
// tracker inside the gate with the noise cost bounded by the SNR gain and
// taaDriftMaxGain. This is the UNCONFIRMED gate; confirmation (either
// channel) drops it to kGhostUnlockThrSigmas.
static const float kDriftThreshSigmas = 2.0;
// The lighting-rate prior -- the per-frame drift scale as a fraction of
// the content scale. Prices H_drift in the event guard's posterior.
static const float kDriftPriorScaleSigmas = 0.5;
// ---- the transport scatter's motion gate ----
// The static floor is MEASURED per pixel (the velocity buffer's own
// quantization/noise scale, passed in by the resolve). kTransportGatePx is
// the motion scale over which the jitter-residual proxy ramps to full
// charge.
static const float kTransportGatePx      = 2.0;

// The jitter sequence's full 2x2 phase covariance as a quadratic form --
// the variance of the jitter's projection on a unit direction. The
// sequence covariance is PSD by construction, so Q >= 0 for every
// direction (saturate guards float error only).
float JitterProjVar(float2 gDir)
{
    return taaJitterPhaseEx  * gDir.x * gDir.x
         + taaJitterPhaseEy  * gDir.y * gDir.y
         + 2.0 * taaJitterPhaseExy * gDir.x * gDir.y;
}

// ============================================================================
// THE SQUARE / HALF-PLANE COVERAGE (exact, any angle)
// ============================================================================
float QuadClampIntegral(float s)
{
    float cl = clamp(s, 0.0, 1.0);
    return 0.5 * cl * cl + max(s - 1.0, 0.0);
}

float SquareCoverage(float2 gDir, float e)
{
    float a = max(abs(gDir.x), abs(gDir.y));
    float b = min(abs(gDir.x), abs(gDir.y));
    if (b < 1e-4)
        return saturate(0.5 - e / max(a, 1e-4));
    return saturate((a / b) * (QuadClampIntegral(0.5 - (e - b * 0.5) / a)
                             - QuadClampIntegral(0.5 - (e + b * 0.5) / a)));
}

// max of the concave c(1-c) over [lo, hi].
float MaxBinomial(float lo, float hi)
{
    float a = saturate(min(lo, hi));
    float b = saturate(max(lo, hi));
    if (a <= 0.5 && b >= 0.5) return 0.25;
    return max(a * (1.0 - a), b * (1.0 - b));
}

// ============================================================================
// THE COVERAGE STEP FIT
// ============================================================================
struct CoverageStepFit
{
    float  w;            // the adequacy blend
    float3 dFB;          // F0 - B0
    float  cTarget;      // the interval-mid coverage
    float  cLo, cHi;     // the WIDENED bracket's coverage range
    float3 residVarB;    // the within-layer residual variances
    float3 residVarF;
    float3 lvlVarB;      // the corrected levels' error variances
    float3 lvlVarF;
    float3 xModel;       // B0 + proj0*dFB
    float3 target;       // B0 + cTarget*dFB
    float2 gDir;         // the oriented edge normal (B -> F)
    float  cSlope;       // the crossing run's mass per coverage unit
};

CoverageStepFit FitCoverageStep(
    float3 taps[9], float2 u,
    float3 B0, float3 F0,
    float3 lvlVarB, float3 lvlVarF,
    float3 residVarB, float3 residVarF,
    float3 gradCDx, float3 gradCDy)
{
    CoverageStepFit f;
    f.w = 0.0;
    f.dFB = F0 - B0;
    f.cTarget = 0.0; f.cLo = 0.0; f.cHi = 1.0;
    f.residVarB = residVarB;
    f.residVarF = residVarF;
    f.lvlVarB = lvlVarB;
    f.lvlVarF = lvlVarF;
    f.xModel = B0;
    f.target = B0;
    f.cSlope = 1.0;

    float3 dFB   = F0 - B0;
    float  dFBsq = dot(dFB, dFB);

    float lvlErr = (lvlVarB.x + lvlVarB.y + lvlVarB.z)
                 + (lvlVarF.x + lvlVarF.y + lvlVarF.z);
    float sig = dFBsq / max(lvlErr, 1e-12);
    f.w = saturate((sig - kStepMinLevels) / kStepMinLevels);
    if (f.w <= 0.0) return f;

    float2 gRaw = float2(length(gradCDx), length(gradCDy));
    if (length(gRaw) < 1e-5) { f.w = 0.0; return f; }
    float2 gDir = normalize(gRaw);

    // The projected tap positions and the dFB projections are
    // loop-invariant -- hoisted once. uMid = dot(mid - B0, dFB) with
    // mid - B0 = dFB/2, i.e. exactly half the squared step.
    float qPrj[9];
    float uPrj[9];
    [unroll]
    for (int hj = 0; hj < 9; ++hj)
    {
        qPrj[hj] = dot(kOffsets3x3[hj], gDir);
        uPrj[hj] = dot(taps[hj] - B0, dFB);
    }
    float uMid = 0.5 * dFBsq;

    float  sep = 0.0;
    float  mObs = 0.0;
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        sep  += sign(uPrj[i] - uMid) * qPrj[i];
        mObs += kStdWeights[i] * clamp(uPrj[i] / max(dFBsq, 1e-10), 0.0, 1.0);
    }
    mObs = saturate(mObs);

    if (sep < 0.0)
    {
        gDir = -gDir;
        [unroll]
        for (int nq = 0; nq < 9; ++nq)
            qPrj[nq] = -qPrj[nq];
    }
    f.gDir = gDir;

    // THE TAIL-MASS STAIRCASE INVERSION (de-phased by u_g; interval-mid
    // target; the widened bracket charges the variance). Only the FIRST
    // crossing is used, so the scan stops there.
    float uG = dot(u, gDir);
    float qMax = -1e9;
    [unroll]
    for (int jq = 0; jq < 9; ++jq)
        qMax = max(qMax, qPrj[jq]);

    float qMid = qMax + 0.5, qA = qMax, qB2 = qMax + 1.0;
    bool  crossed = false;
    float runMassHit = 1.0;
    [unroll]
    for (int k = 0; k < 9; ++k)
    {
        if (crossed) break;
        float qk = qPrj[k];
        float massAbove = 0.0, massRun = 0.0;
        float qAbove = qk; bool anyAbove = false;
        float qBelow = qk; bool anyBelow = false;
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float qj = qPrj[j];
            if      (qj > qk + 1e-4) { massAbove += kStdWeights[j]; if (!anyAbove || qj > qAbove) { qAbove = qj; anyAbove = true; } }
            else if (qj > qk - 1e-4) { massRun   += kStdWeights[j]; }
            else                     {                                       if (!anyBelow || qj > qBelow) { qBelow = qj; anyBelow = true; } }
        }
        if (massAbove < mObs && mObs <= massAbove + massRun)
        {
            qMid = 0.5 * (qk + (anyBelow ? qBelow : qk - 1.0));
            qA = anyBelow ? qBelow : (qk - 1.0);
            qB2 = anyAbove ? qAbove : (qk + 1.0);
            runMassHit = massRun;
            crossed = true;
        }
    }
    if (!crossed) { qMid = qMax + 0.5; qA = qMax; qB2 = qMax + 1.0; }

    if (abs(qB2 - qA) > kStepMaxSpan) { f.w = 0.0; return f; }

    // The jitter span is DIRECTION-AWARE -- the sequence's variance along
    // the edge normal is the quadratic form Q(gDir), not the axis average.
    float jitterSpan = max(sqrt(saturate(12.0 * JitterProjVar(gDir))), kMinJitterSpan);
    float cT  = SquareCoverage(gDir, (qMid - uG) / jitterSpan);
    float cLo = SquareCoverage(gDir, (qA - uG) / jitterSpan);
    float cHi = SquareCoverage(gDir, (qB2 - uG) / jitterSpan);
    if (cLo < cHi) { float tmp = cLo; cLo = cHi; cHi = tmp; }

    f.cSlope = runMassHit / max(abs(cLo - cHi), 0.05);

    float proj0 = clamp(uPrj[0] / max(dFBsq, 1e-10), 0.0, 1.0);

    f.cTarget = cT; f.cLo = cLo; f.cHi = cHi;
    f.xModel  = B0 + proj0 * dFB;
    f.target  = B0 + cT * dFB;
    return f;
}

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx, float2 jitterResidualPx, float edgeSweepPx,
    float2 transportErrPx,
    bool sameLayerMask[9], bool layerMaskValid,
    bool historyTapIsBand[9])
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
    stats.jitterPx = jitterPx;

    bool exactSeedInputs = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0);
    bool jitterPaddingEnabled = (taaJitterFlickerPadding > kFlickerPadThreshold) && !exactSeedInputs;

    float3 weightedSum   = float3(0.0, 0.0, 0.0);
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);
    float  totalWeight   = 0.0;
    float  totalWeightSq = 0.0;
    float  motionFactor  = saturate(motionNormalized);

    float2 weightedOffsetSum = float2(0.0, 0.0);
    float  weightedOffXSqSum = 0.0;
    float  weightedOffYSqSum = 0.0;
    float  weightedOffXYmSum = 0.0;
    float3 weightedCovXSum   = float3(0.0, 0.0, 0.0);
    float3 weightedCovYSum   = float3(0.0, 0.0, 0.0);
    float  wSumXY = 0.0, wSumXZ = 0.0, wSumYZ = 0.0;
    float  recDofWSum = 0.0, recDofWSq = 0.0, recDofUsed = 0.0;

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    // The jitter-centered Gaussian is separable --
    // exp(-(dx^2+dy^2)) = exp(-dx^2)*exp(-dy^2) -- so the nine weights
    // cost six exp2 (one per axis per distinct offset) plus nine
    // multiplies.
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

        // The record estimator's per-frame dof accumulates in the same
        // pass over the estimator's OWN tap set (the same-layer mask minus
        // the stored edge flags); the finalize sits after effRank.
        if ((i == 0) || (sameLayerMask[i] && !historyTapIsBand[i]))
        {
            recDofWSum += w;
            recDofWSq  += w * w;
            recDofUsed += 1.0;
        }

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
        wSumXY += w * tapColorSpace.x * tapColorSpace.y;
        wSumXZ += w * tapColorSpace.x * tapColorSpace.z;
        wSumYZ += w * tapColorSpace.y * tapColorSpace.z;
        w9[i] = w;
        // The offset/color moments are UNCONDITIONAL: the hierarchical
        // gate's prior (plane0 / lever / residSq) consumes them on every
        // pixel.
        weightedOffXSqSum += tapOffsetPx.x * tapOffsetPx.x * w;
        weightedOffYSqSum += tapOffsetPx.y * tapOffsetPx.y * w;
        weightedOffXYmSum += tapOffsetPx.x * tapOffsetPx.y * w;
        weightedCovXSum   += tapOffsetPx.x * tapColorSpace * w;
        weightedCovYSum   += tapOffsetPx.y * tapColorSpace * w;
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean    = weightedSum * invTotalWeight;
    stats.sigma   = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));
    stats.invNeff = saturate(totalWeightSq * invTotalWeight * invTotalWeight);

    float2 kernelCentroidPx = weightedOffsetSum * invTotalWeight;

    float3 gradCDx = 0.5 * (neighborhoodColorSpace[4] - neighborhoodColorSpace[3]);
    float3 gradCDy = 0.5 * (neighborhoodColorSpace[2] - neighborhoodColorSpace[1]);
    stats.gradCDx = gradCDx;
    stats.gradCDy = gradCDy;

    {
        float trC = dot(stats.sigma, stats.sigma);
        float covXY = wSumXY * invTotalWeight - stats.mean.x * stats.mean.y;
        float covXZ = wSumXZ * invTotalWeight - stats.mean.x * stats.mean.z;
        float covYZ = wSumYZ * invTotalWeight - stats.mean.y * stats.mean.z;
        float trCsq = trC * trC + 2.0 * (covXY * covXY + covXZ * covXZ + covYZ * covYZ);
        stats.effRank = (trC > 1e-12) ? clamp(trC * trC / max(trCsq, 1e-12), 1.0, 3.0) : 1.0;
        stats.covCross = float3(covXY, covXZ, covYZ);
    }

    // The record estimator's per-frame dof over the FULL 3x3 (the
    // estimator consumes all nine taps; the typical effective count of the
    // kernel-weighted 3x3 is ~7 -> dof ~6, which the host's Student-fit
    // anchor tracks). The count mirrors the estimator's ACTUAL tap set --
    // the edge-flagged neighbors are excluded there, so they are excluded
    // here too (an unmasked count would overstate the dof and
    // under-inflate the Student factor on exactly the wake pixels).
    if (layerMaskValid && recDofUsed >= 3.0 && recDofWSum > 1e-6)
    {
        float invNeffSub = recDofWSq / (recDofWSum * recDofWSum);
        stats.recordSampleDof = max(1.0 / max(invNeffSub, 0.05) - 1.0, 1.0);
    }
    else
    {
        stats.recordSampleDof = stats.effRank;
    }

    // =====================================================================
    // THE FULL-SET COUPLED LS PLANE (unconditional: the hierarchical gate's
    // prior consumes plane0 / lever / residSq on every pixel; the phase
    // correction's fallback and the cold gate read the LS gradient).
    // Solved on the CENTERED second-moment matrix -- the exact LS plane
    // (an uncentered solve is shrunk by (I + C^-1 c c^T)^-1 whenever the
    // weighted centroid is nonzero, which the jitter-centered weights
    // are).
    // =====================================================================
    {
        float  vXX = max(weightedOffXSqSum * invTotalWeight - kernelCentroidPx.x * kernelCentroidPx.x, 1e-6);
        float  vYY = max(weightedOffYSqSum * invTotalWeight - kernelCentroidPx.y * kernelCentroidPx.y, 1e-6);
        float  vXY = weightedOffXYmSum * invTotalWeight - kernelCentroidPx.x * kernelCentroidPx.y;
        float3 covX  = weightedCovXSum * invTotalWeight - kernelCentroidPx.x * stats.mean;
        float3 covY  = weightedCovYSum * invTotalWeight - kernelCentroidPx.y * stats.mean;
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
        stats.phaseShift      = -(lsGX * jitterPx.x + lsGY * jitterPx.y);
        stats.phaseShiftPlane = stats.phaseShift;

        // The fitted value at the history's target (content position 0 =
        // the texel center): plane(0) = mean + g*(j - cen). Its exact
        // estimation variance is sigma^2 * (invNeff + lever) with the
        // lever the Mahalanobis quadratic of the eval offset d = j - cen
        // (the mean's factor and the gradient's factor are orthogonal by
        // LS construction -- both factors come from the SAME centered
        // moment matrix as the fit).
        float2 dEval = jitterPx - kernelCentroidPx;
        float  leverFull = (vYY * dEval.x * dEval.x - 2.0 * vXY * dEval.x * dEval.y + vXX * dEval.y * dEval.y)
                         / max(detV, 1e-6);
        stats.lever  = min(max(leverFull, 0.0), kMaxLever);
        stats.plane0 = stats.mean + lsGX * dEval.x + lsGY * dEval.y;

        float3 scc = weightedSumSq * invTotalWeight - stats.mean * stats.mean;
        float3 ssr = scc
                   - 2.0 * (lsGX * covX + lsGY * covY)
                   + (lsGX * lsGX * vXX + lsGY * lsGY * vYY + 2.0 * lsGX * lsGY * vXY);
        float dofCorr = 1.0 / max(1.0 - 3.0 * stats.invNeff, 1.0 / 3.0);
        stats.residSq = max(ssr * dofCorr, 0.0);

        // The transport scatter: the gated landing-error vector projected
        // on the WORKING LS gradient (the full-set plane here; the straddle
        // rebase below re-projects it on the same-layer gradient). The LS
        // gradient is the ML estimate of the content slope; the CD form
        // this replaces was both noisier and inconsistent with the plane
        // the prior evaluates against.
        stats.transportVar = (transportErrPx.x * transportErrPx.x) * (lsGX * lsGX)
                           + (transportErrPx.y * transportErrPx.y) * (lsGY * lsGY);
    }

    stats.sigma = min(stats.sigma, 0.5 * (stats.aabbMax - stats.aabbMin));

    float spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

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
    stats.rangeSq = dot(stats.aabbMax - stats.aabbMin, stats.aabbMax - stats.aabbMin);

    // ---- the same-depth-layer (B) + off-layer (F) accumulators ---------------
    stats.sigmaClean        = stats.sigma;
    stats.invNeffClean      = stats.invNeff;
    stats.invNeffF          = 1.0;
    stats.cusumStepDof      = 2.0;
    stats.rangeSqClean      = stats.rangeSq;
    stats.stepW             = 0.0;
    stats.stepTarget        = stats.mean;
    stats.stepTargetVar     = float3(0.0, 0.0, 0.0);
    stats.stepTargetStable  = stats.mean;
    stats.stepDFB           = float3(0.0, 0.0, 0.0);
    stats.stepC             = 0.0;
    stats.stepCStable       = 0.5;
    stats.stepXModel        = stats.mean;
    stats.stepResidVarB     = float3(0.0, 0.0, 0.0);
    stats.stepResidVarF     = float3(0.0, 0.0, 0.0);
    stats.stepPhaseVarScale = 0.0;
    stats.covVarFloor       = float3(0.0, 0.0, 0.0);
    stats.covSpanSq         = 0.0;

    if (layerMaskValid)
    {
        float3 sumC = float3(0.0, 0.0, 0.0);
        float3 ssqC = float3(0.0, 0.0, 0.0);
        float2 offC = float2(0.0, 0.0);
        float  wC = 0.0, wC2 = 0.0;
        float  offXSqC = 0.0, offYSqC = 0.0;
        float  offXYC = 0.0;
        float3 covXSumC = float3(0.0, 0.0, 0.0);
        float3 covYSumC = float3(0.0, 0.0, 0.0);
        float  momXYC = 0.0, momXZC = 0.0, momYZC = 0.0;
        float3 cMin = float3(kLargeValue, kLargeValue, kLargeValue);
        float3 cMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
        float3 sumF = float3(0.0, 0.0, 0.0);
        float3 ssqF = float3(0.0, 0.0, 0.0);
        float2 offF = float2(0.0, 0.0);
        float  wF = 0.0, wF2 = 0.0;
        float  offXSqF = 0.0, offYSqF = 0.0, offXYF = 0.0;
        float3 covXSumF = float3(0.0, 0.0, 0.0);
        float3 covYSumF = float3(0.0, 0.0, 0.0);
        bool   anyOff = false;

        [unroll]
        for (int k = 0; k < 9; ++k)
        {
            float  w = w9[k];
            float3 c = neighborhoodColorSpace[k];
            float2 o = kOffsets3x3[k];
            if (!sameLayerMask[k])
            {
                anyOff = true;
                sumF += c * w;
                ssqF += (c * c) * w;
                offF += o * w;
                offXSqF += o.x * o.x * w;
                offYSqF += o.y * o.y * w;
                offXYF   += o.x * o.y * w;
                covXSumF += o.x * c * w;
                covYSumF += o.y * c * w;
                wF   += w;
                wF2  += w * w;
                continue;
            }
            sumC += c * w;
            ssqC += (c * c) * w;
            offC += o * w;
            offXSqC += o.x * o.x * w;
            offYSqC += o.y * o.y * w;
            offXYC   += o.x * o.y * w;
            covXSumC += o.x * c * w;
            covYSumC += o.y * c * w;
            momXYC   += w * c.x * c.y;
            momXZC   += w * c.x * c.z;
            momYZC   += w * c.y * c.z;
            wC   += w;
            wC2  += w * w;
            cMin = min(cMin, c);
            cMax = max(cMax, c);
        }
        stats.straddled = anyOff;

        if (anyOff && wC > 0.2 * totalWeight)   // enough same-layer mass
        {
            float  invWC = 1.0 / max(wC, kEpsilon);
            float3 muC   = sumC * invWC;
            float2 cenC  = offC * invWC;
            stats.sigmaClean      = max(sqrt(max(ssqC * invWC - muC * muC, 0.0)), kMinSigma);
            stats.invNeffClean    = saturate(wC2 * invWC * invWC);
            stats.rangeSqClean    = dot(cMax - cMin, cMax - cMin);
            float3 cR = sameLayerMask[4] ? neighborhoodColorSpace[4] : muC;
            float3 cL = sameLayerMask[3] ? neighborhoodColorSpace[3] : muC;
            float3 cU = sameLayerMask[2] ? neighborhoodColorSpace[2] : muC;
            float3 cD = sameLayerMask[1] ? neighborhoodColorSpace[1] : muC;
            float3 gBx = 0.5 * (cR - cL);
            float3 gBy = 0.5 * (cU - cD);

            // THE CLEAN REBASING. While off-layer mass is partitioned out,
            // the non-engaged clip target is the pixel's OWN layer's value:
            // the off-layer taps are OTHER TEXELS' values -- in the target
            // they are a bias toward the ghost, not a variance.
            stats.mean = muC;
            stats.covCross = float3(momXYC * invWC - muC.x * muC.y,
                                    momXZC * invWC - muC.x * muC.z,
                                    momYZC * invWC - muC.y * muC.z);

            // The same-layer COUPLED plane, its residual, and the PRIOR
            // PACKAGE rebasing. (gXc/gYc/ssrC/dofCorrC are hoisted past the
            // package gate: the step fit's B-level residual and the
            // transport rebase below consume the same solve.)
            float3 gXc = float3(0.0, 0.0, 0.0);
            float3 gYc = float3(0.0, 0.0, 0.0);
            float3 ssrC = float3(0.0, 0.0, 0.0);
            float  dofCorrC = 1.0;
            bool   cHasDof = (1.0 / max(stats.invNeffClean, 1e-3) >= 3.5);
            {
                float  mXXc  = max(offXSqC * invWC - cenC.x * cenC.x, 1e-6);
                float  mYYc  = max(offYSqC * invWC - cenC.y * cenC.y, 1e-6);
                float  mXYc  = offXYC   * invWC - cenC.x * cenC.y;
                float3 gXNc  = covXSumC * invWC - cenC.x * muC;
                float3 gYNc  = covYSumC * invWC - cenC.y * muC;
                float  detNc = mXXc * mYYc - mXYc * mXYc;
                if (detNc > 1e-6)
                {
                    gXc = (mYYc * gXNc - mXYc * gYNc) / detNc;
                    gYc = (mXXc * gYNc - mXYc * gXNc) / detNc;
                }
                else
                {
                    gXc = gXNc / mXXc;
                    gYc = gYNc / mYYc;
                }
                float3 sccC = ssqC * invWC - muC * muC;
                ssrC = sccC
                     - 2.0 * (gXc * gXNc + gYc * gYNc)
                     + (gXc * gXc * mXXc + gYc * gYc * mYYc + 2.0 * gXc * gYc * mXYc);
                dofCorrC = 1.0 / max(1.0 - 3.0 * stats.invNeffClean, 1.0 / 3.0);

                // THE PRIOR PACKAGE REBASING: only when the C-subset has
                // dof for its own plane (~>= 4 effective taps). A 1-3 tap
                // C-subset (thin foreground detail over background) has
                // zero residual dof: its "plane" through one point has no
                // residual, and rebasing would collapse the prior width
                // toward zero -- the classic thin-detail shimmer. The
                // full-set package (step-contaminated, hence conservative-
                // wide) is the honest statement there.
                if (cHasDof)
                {
                    float2 dEvalC = jitterPx - cenC;
                    float  leverC = (mYYc * dEvalC.x * dEvalC.x - 2.0 * mXYc * dEvalC.x * dEvalC.y + mXXc * dEvalC.y * dEvalC.y)
                                  / max(detNc, 1e-6);
                    stats.residSq  = max(ssrC * dofCorrC, 0.0);
                    stats.invNeff  = stats.invNeffClean;
                    stats.plane0   = muC + gXc * dEvalC.x + gYc * dEvalC.y;
                    stats.lever    = min(max(leverC, 0.0), kMaxLever);
                    stats.gradLSx = gXc;
                    stats.gradLSy = gYc;
                    stats.phaseShift      = -(gXc * jitterPx.x + gYc * jitterPx.y);
                    stats.phaseShiftPlane = stats.phaseShift;
                }
            }

            // The transport scatter rebased onto the OWN-LAYER LS gradient:
            // the full-set projection charges the landing error across
            // the layer step on straddles -- a dFB-scaled overcharge on
            // exactly the pixels the step model owns. (A 1-tap C degenerates
            // to a zero gradient -- acceptable: the degenerate prior is
            // step-wide anyway and the transport term is a small correction
            // on it; 2-3 tap C-subsets get an honest measured projection.)
            stats.transportVar = (transportErrPx.x * transportErrPx.x) * (gXc * gXc)
                               + (transportErrPx.y * transportErrPx.y) * (gYc * gYc);

            if (wF > 1e-4)
            {
                float  invWF = 1.0 / wF;
                float3 muF   = sumF * invWF;
                float2 cenF  = offF * invWF;
                float3 sigmaF = max(sqrt(max(ssqF * invWF - muF * muF, 0.0)), kMinSigma);
                float invNeffF = saturate(wF2 * invWF * invWF);
                stats.invNeffF = invNeffF;

                // The F-subset's OWN coupled LS plane (the same solve the C
                // side runs; the moments were accumulated in the partition
                // pass). A 1-2 tap F degenerates to the tap value (the
                // gradient solve collapses to zero) -- the honest level
                // there.
                float  mXXf  = max(offXSqF * invWF - cenF.x * cenF.x, 1e-6);
                float  mYYf  = max(offYSqF * invWF - cenF.y * cenF.y, 1e-6);
                float  mXYf  = offXYF  * invWF - cenF.x * cenF.y;
                float3 gXNf  = covXSumF * invWF - cenF.x * muF;
                float3 gYNf  = covYSumF * invWF - cenF.y * muF;
                float  detNf = mXXf * mYYf - mXYf * mXYf;
                float3 gFx, gFy;
                if (detNf > 1e-6)
                {
                    gFx = (mYYf * gXNf - mXYf * gYNf) / detNf;
                    gFy = (mXXf * gYNf - mXYf * gXNf) / detNf;
                }
                else
                {
                    gFx = gXNf / mXXf;
                    gFy = gYNf / mYYf;
                }

                // THE CORRECTED LEVELS: pixel-center, phase-stable -- on the
                // SAME-SET LS planes. B0 IS the prior package's plane0 when
                // the C-subset has dof for its own plane (one estimator, one
                // exact variance form), with the robust CD form as the
                // thin-C fallback; F0 is the F-LS plane above. The LS levels
                // are tighter than the CD forms the previous version used AND
                // carry the exact estimation-variance structure the
                // (invNeff + lever) levers below encode.
                float3 B0 = cHasDof
                    ? stats.plane0
                    : muC + gBx * (jitterPx.x - cenC.x) + gBy * (jitterPx.y - cenC.y);
                float3 F0 = muF + gFx * (jitterPx.x - cenF.x) + gFy * (jitterPx.y - cenF.y);
                float3 dFB0 = F0 - B0;

                // The within-layer residuals: the LS planes' OWN residuals
                // (ssrC from the C solve; the F analog solved here),
                // dof-corrected and floored at the collapse guard. At a thin
                // C (1-3 taps) ssrC is ~0 by construction and the floor
                // carries it -- harmless in every consumer (the F-side
                // residual dominates the admission; flipAcc*dFB^2 dominates
                // the varH model).
                float  dofCorrF = 1.0 / max(1.0 - 3.0 * invNeffF, 1.0 / 3.0);
                float3 sccF = ssqF * invWF - muF * muF;
                float3 ssrF = sccF
                            - 2.0 * (gFx * gXNf + gFy * gYNf)
                            + (gFx * gFx * mXXf + gFy * gFy * mYYf + 2.0 * gFx * gFy * mXYf);
                float3 residVarB = max(ssrC * dofCorrC,
                                       stats.sigmaClean * stats.sigmaClean * (kScopedResidFrac * kScopedResidFrac));
                float3 residVarF = max(ssrF * dofCorrF,
                                       sigmaF * sigmaF * (kScopedResidFrac * kScopedResidFrac));

                // THE LEVELS' ESTIMATION VARIANCES -- the exact LS
                // fitted-value form residVar * (invNeff + lever), with the
                // lever the Mahalanobis quadratic of the eval offset
                // (retires kStepLeverK: |d|^2 * sigmaClean^2 * 0.5 charged
                // the layer's own SLOPE as estimation error and over-widened
                // every level on slanted surfaces). The B-side lever is the
                // prior package's own (the same eval point as B0); the
                // F-side is solved on its own moments above, with the
                // Euclidean |d|^2 as the degenerate-F stand-in.
                float2 dEvalF = jitterPx - cenF;
                float  leverF = (detNf > 1e-6)
                    ? (mYYf * dEvalF.x * dEvalF.x - 2.0 * mXYf * dEvalF.x * dEvalF.y + mXXf * dEvalF.y * dEvalF.y) / detNf
                    : dot(dEvalF, dEvalF);
                leverF = min(max(leverF, 0.0), kMaxLever);
                float3 lvlVarB = residVarB * (stats.invNeffClean + stats.lever);
                float3 lvlVarF = residVarF * (invNeffF + leverF);

                CoverageStepFit fit = FitCoverageStep(
                    neighborhoodColorSpace, jitterPx,
                    B0, F0, lvlVarB, lvlVarF, residVarB, residVarF,
                    gradCDx, gradCDy);

                stats.stepW         = fit.w;
                stats.stepDFB       = dFB0;
                stats.stepCStable   = saturate(0.5 * (fit.cLo + fit.cHi));
                stats.stepTargetStable = B0 + stats.stepCStable * dFB0;
                stats.stepXModel    = fit.xModel;
                stats.stepResidVarB = fit.residVarB;
                stats.stepResidVarF = fit.residVarF;

                stats.cusumStepDof = max(1.0 / max(stats.invNeffClean, 1e-4)
                                       + 1.0 / max(invNeffF, 1e-4) - 4.0, 2.0);

                // The coverage bracket's variance and span, computed at
                // ANY step adequacy -- they describe the pixel's
                // plausible-VALUE uncertainty; the adequacy gate only
                // decides whether the model may move the TARGET.
                float loEff = min(fit.cLo, fit.cHi);
                float hiEff = max(fit.cLo, fit.cHi);
                float flip  = MaxBinomial(loEff, hiEff);
                float varC  = (hiEff - loEff) * (hiEff - loEff) * (1.0 / 12.0);

                float  cSVar   = stats.stepCStable;
                float3 residMV = (1.0 - cSVar) * stats.stepResidVarB + cSVar * stats.stepResidVarF;
                float  residAV = dot(residMV, float3(1.0, 1.0, 1.0)) * (1.0 / 3.0);
                float  dFB0Sq  = dot(dFB0, dFB0);
                float  wSqMass = 0.0;
                [unroll]
                for (int wI = 0; wI < 9; ++wI)
                    wSqMass += kStdWeights[wI] * kStdWeights[wI];
                float  varCfresh = min(wSqMass * residAV
                                     / max(dFB0Sq * fit.cSlope * fit.cSlope, 1e-8),
                                     varC);
                float  varCeff = max(varCfresh, varC * saturate(edgeSweepPx));

                // THE HONEST STRADDLE FLOOR: the coverage-mixture variance,
                // per channel, blending the unmodelled uniform share
                // (1-w)/12 with the modelled bracket share w*varCeff.
                float  covVarC = (1.0 - fit.w) * (1.0 / 12.0) + fit.w * varCeff;
                stats.covVarFloor = covVarC * (dFB0 * dFB0);
                stats.covSpanSq   = ((1.0 - fit.w)
                                   + fit.w * (hiEff - loEff) * (hiEff - loEff))
                                  * dFB0Sq;

                if (fit.w > 0.0)
                {
                    // Nothing migrates the clip target: the persistence
                    // detector's consumers (the drift-corrector unlock,
                    // the blend floor, the alarm) all act on the
                    // history/evidence side -- a moving target is an
                    // edge-flicker source.
                    float cEff = fit.cTarget;
                    float jrN = dot(jitterResidualPx, fit.gDir)
                              / max(sqrt(saturate(12.0 * JitterProjVar(fit.gDir))), kMinJitterSpan);
                    stats.stepPhaseVarScale = jrN * jrN;

                    float3 tStep = B0 + cEff * dFB0;
                    float3 tVar  = (1.0 - cEff) * (1.0 - cEff) * lvlVarB
                                 + cEff * cEff * lvlVarF
                                 + (varCeff + taaCusumFlipAcc * flip + stats.stepPhaseVarScale) * (dFB0 * dFB0);

                    stats.stepC = cEff;
                    stats.stepTarget = tStep;

                    float3 tDiff = tStep - stats.mean;
                    stats.stepTargetVar = tVar + fit.w * (1.0 - fit.w) * (tDiff * tDiff);
                    stats.phaseShift = lerp(stats.phaseShift,
                                            fit.xModel - stats.stepTargetStable,
                                            fit.w);
                }
            }
        }
    }

    // Straddle policy in full: the non-engaged target is the own-layer
    // value (the hierarchical prior + tap posterior); the step model
    // blends the coverage target in by its adequacy; the acceptance keeps
    // the coverage-mixture floor at every adequacy; nothing else migrates
    // the clip target.
    return stats;
}

// ============================================================================
// STUDENT FACTOR
// ============================================================================
float StudentEffectiveDof(float recordAge, float sampleDof)
{
    float decayPrior = exp2(recordAge * kClipStudentDecay);   // (1-rho)^{2t}
    float varFrac    = kClipSigmaEmaRate / (max(sampleDof, 1.0) * (2.0 - kClipSigmaEmaRate));
    return 1.0 / (decayPrior / kClipStudentPriorDof + varFrac * (1.0 - decayPrior));
}

float StudentFactor(float nu)
{
    float invNu = 1.0 / max(nu, 1e-3);
    return 1.0 + taaStudentA * invNu + taaStudentB * invNu * invNu;
}

// ============================================================================
// THE CHROMA MAGNITUDE'S NULL MOMENTS (measured, at the local Y-Z correlation)
// ============================================================================
// The chroma persistence input is the vector magnitude of the two per-axis
// t-statistics; its null law depends on the chroma channels' correlation
// rho: the magnitude's mean falls (Rayleigh-like -> collinear half-normal)
// and its variance rises as rho -> 1. The endpoints are the exact moments
// of the ESTABLISHED per-axis marginal -- the t at the null variance
// kGhostNullStdSq (the same t-family law kGhostNullStdSq itself encodes):
//   rho = 1 (exact, closed form): m = sqrt(2)|t|, so with
//     E|t_6| = 9/(4 sqrt(6)) and Var|t_6| = 21/32:
//       E[m]   = 3*sqrt(3)/4 = 1.2990,  Var(m) = 21/16 = 1.3125
//   rho = 0 (iid t marginal, numeric): E[m] = 1.4720 (+-1%, a Simpson
//     integration of the 2D magnitude; the Gaussian-form 1.535 overstates
//     it by 4.5% -- the t's central concentration), and Var(m) =
//     3 - E[m]^2 = 0.8332 EXACTLY given the mean, via the
//     distribution-free identity E[m^2] = 2 * kGhostNullStdSq (the
//     per-axis second moments are pinned by the null model regardless of
//     the dependence). Both endpoint pairs satisfy the identity exactly
//     (27/16 + 21/16 = 3).
// The bivariate-t (shared-denominator) model brackets the rho = 0 pair
// from the other side (E[m] = 1.442, Var = 0.922) -- the true joint law
// of the two per-channel studentizations (shared taps, separate
// denominators) sits between; the scale below normalizes the std away, so
// only the MEAN is load-bearing, and the 2% bracket is the residual model
// uncertainty. The interpolation in rho^2 is exact at both ends (the
// moments are even in rho by the axes' exchange symmetry). The measured
// rho is the shrunk tap correlation (covCross.z over the same-layer
// sigmas), the same shrink the gate's metric applies.
void GhostChromaNullMoments(ColorNeighborhoodStats stats, out float nullMean, out float nullVar)
{
    float meanRho0 = 1.4720;                        // iid-t numeric (+-1%); see the block header
    float meanRho1 = 1.2990;                        // 3*sqrt(3)/4, exact
    float varRho0  = 0.8332;                        // 3 - meanRho0^2 (the E[m^2] identity)
    float varRho1  = 1.3125;                         // 21/16, exact
    float rDof = max(1.0 / max(stats.invNeff, 1e-3) - 1.0, 2.0);
    float corrShrink = max(0.0, 1.0 - 1.0 / rDof);
    float rho = clamp((stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon)) * corrShrink, -0.95, 0.95);
    float r2 = saturate(rho * rho);
    nullMean = lerp(meanRho0, meanRho1, r2);
    nullVar  = lerp(varRho0, varRho1, r2);
}

// ============================================================================
// CLIP-STATE TRANSPORT (tag 101)
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
    if      (tag == 0x5u)                     // clip-state alpha
    {
        code = (u >> 22) & 0x3Fu;
        varHCode = (float)((u >> 10) & 0x3Fu);
        recordAge = (float)((u >> 6) & 0xFu);
        ghostT  = ((float)(u & 0x3Fu) - 32.0) * 0.25;          // the luma persistence statistic, 0.25 quanta
        ghostTc = ((float)((u >> 16) & 0x3Fu) - 32.0) * 0.25;  // the chroma persistence statistic, 0.25 quanta
    }
    else if (tag == 0x1u)                     // debug payload: sigma only
    {
        code = u & 0x3Fu;
    }
    else
    {
        return;          // foreign alpha (any other tag): cold
    }

    if (code == 0u) return;                   // explicit cold marker
    float s = ClipUnpackSigmaCode(code);
    sigmaSq = s * s;
    varHSq = DecodeVarHRatio(varHCode, sigmaSq);
}

// ============================================================================
// THE STEP'S VARIANCE MODEL (the seed / the arming reference)
// ============================================================================
float3 StepVarianceModelX(ColorNeighborhoodStats stats)
{
    float cS = stats.stepCStable;
    float3 phase = (cS * (1.0 - cS)) * (stats.stepDFB * stats.stepDFB);
    float3 noise = (1.0 - cS) * stats.stepResidVarB + cS * stats.stepResidVarF;
    return phase + noise;
}

// ============================================================================
// THE EXACT RECORD SEED (also the arming cap's anchor)
// ============================================================================
float SeedRecordTotalSq(ColorNeighborhoodStats stats)
{
    if (stats.stepW > 0.5)
    {
        float3 varX = StepVarianceModelX(stats);
        float  sTotal = varX.x + varX.y + varX.z + 3.0 * kResampleVarSq;
        return max(sTotal, kClipSigmaRecordFloorSq);
    }

    // A straddled pixel's post-reset innovations include the coverage
    // jitter under the sweep -- the seed carries the coverage-mixture
    // variance, or the first frames after every reset on an edge
    // over-clip.
    float covFloorTotal = stats.covVarFloor.x + stats.covVarFloor.y + stats.covVarFloor.z;

    bool exactSeed = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0);
    if (!exactSeed)
        // The own-layer scale (sigmaClean degenerates to sigma on
        // unstraddled pixels).
        return max(dot(stats.sigmaClean, stats.sigmaClean) + covFloorTotal, kClipSigmaRecordFloorSq);

    // The full quadratic form, per channel, on the WORKING LS gradient
    // (the same-layer coupled solve; the tighter estimate -- the CD form
    // the previous version used was ~2x noisier and inflated the seed).
    float3 phase = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                 + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                 + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy);
    float3 noise = stats.residSq;
    float3 varX  = phase + noise + stats.covVarFloor;
    float  sTotal = varX.x + varX.y + varX.z + 3.0 * kResampleVarSq;
    return max(sTotal, kClipSigmaRecordFloorSq);
}

// ============================================================================
// THE RECORD'S PER-CHANNEL SPLIT
// ============================================================================
struct RecordVarianceParts
{
    float3 ePer;        // per-channel PRE-update E[i_k^2] (arming-capped)
    float3 varHPer;     // per-channel accumulator variance (READ semantics
                        // -- what the gate and the step model consume
                        // directly)
    float3 sPer;        // implied per-sample variance: ePer - varH - v.
                        // THE CENTER-TAP LIKELIHOOD VARIANCE of the
                        // hierarchical gate (E[i^2] - varH is exactly the
                        // variance of a fresh sample about the
                        // accumulation target).
    float  spatialPriorSq; // the content-scale reference (the winsor cap's
                            // anchor)
};

RecordVarianceParts SplitRecordVariance(
    ColorNeighborhoodStats stats, float recordSqTotal, float varHTotal,
    float nuRecord,
    bool recordSpatialValid, float3 recordSpatialPerCh)
{
    RecordVarianceParts p;

    float3 share;
    if (stats.stepW > 0.5)
    {
        float3 dSq = stats.stepDFB * stats.stepDFB;
        share = dSq / max(dSq.x + dSq.y + dSq.z, 1e-12);
    }
    else if (recordSpatialValid)
    {
        // The MEASURED per-channel composition of the record: the
        // estimator's per-channel spatial innovation (varSpatial +
        // phaseSq) IS the statistic the record transports, measured per
        // channel this frame -- exact at a reset (the seed is this
        // estimate) and the best single-frame estimate of the transported
        // proportions thereafter. The sigmaClean proxy conflates the
        // between-layer slope with the innovation and excludes the fresh
        // noise entirely.
        float3 recPerCh = max(recordSpatialPerCh, float3(0.0, 0.0, 0.0));
        float  recTotal = dot(recPerCh, float3(1.0, 1.0, 1.0));
        share = (recTotal > 1e-10)
            ? recPerCh / recTotal
            : stats.sigmaClean * stats.sigmaClean / max(dot(stats.sigmaClean, stats.sigmaClean), kEpsilon);
    }
    else
    {
        float  traceSig = max(dot(stats.sigmaClean, stats.sigmaClean), kEpsilon);
        share = (stats.sigmaClean * stats.sigmaClean) / traceSig;
    }

    // The content-scale reference: the SEED -- the spatial E[i^2] estimate
    // (phase + residual + the coverage floor) on flats, the step's own
    // variance model on steps. The seed is degenerate-aware (the working
    // package) and includes the PHASE term, which the previous
    // sigmaClean^2 proxy missed entirely -- a proxy that would have
    // clamped every record on slanted static surfaces to the floor.
    float spatialPriorSq;
    if (stats.stepW > 0.5)
    {
        float3 varX = StepVarianceModelX(stats);
        spatialPriorSq = max(varX.x + varX.y + varX.z, kClipSigmaRecordFloorSq);
    }
    else
    {
        spatialPriorSq = max(SeedRecordTotalSq(stats), kClipSigmaRecordFloorSq);
    }

    // THE ARMING CAP at the record's LIVE dof: the exact one-sided 99.9%
    // bound chi^2_nu(0.999)/nu, host-fit as 1 + A/sqrt(nu) + B/nu (exact
    // at the record's dof trajectory endpoints nu=4 and nu~74). The cap
    // binds the record AND the transported VarH share (the share is a
    // component of the record's total, so the prior bound applies a
    // fortiori) -- the anti-fiction role: the texel that just left a
    // dilation band behind a departing foreground enters capped while an
    // uncapped transported share would keep the band's edge-scale ratio
    // and self-sustain through the (1-a)^2 damp. Legit states never bind
    // (varH <= ~(a/2)E[i^2] << the cap; the record's estimation noise
    // beyond 99.9% is the winsor ingest's business).
    float nuCap = max(nuRecord, 2.0);
    float armingCap = 1.0 + taaArmingA / sqrt(nuCap) + taaArmingB / nuCap;

    float recordSq = min(max(recordSqTotal, 0.0), armingCap * spatialPriorSq);
    float varHSq   = min(max(varHTotal, 0.0), armingCap * spatialPriorSq);
    // The per-channel floor AT the quantization floor (not share-scaled):
    // every channel carries at least the buffer's own quantization noise,
    // so a measured-zero channel share cannot collapse that channel's
    // gate below it.
    p.ePer    = max(share * recordSq, float3(kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq));
    p.spatialPriorSq = spatialPriorSq;

    if (stats.stepW > 0.5)
    {
        // taaCusumNoiseAcc is the EXACT fixed-point accumulated-read noise
        // share (the host's kernelChainConstants series) -- the model
        // below is in READ semantics, consistent with the transported
        // varH and with what the gate tests.
        float  cS = stats.stepCStable;
        float3 residMix = (1.0 - cS) * stats.stepResidVarB + cS * stats.stepResidVarF;
        float3 modelVarH = taaCusumFlipAcc * (cS * (1.0 - cS)) * (stats.stepDFB * stats.stepDFB)
                         + taaCusumNoiseAcc * residMix;
        p.varHPer = share * max(modelVarH, float3(0.0, 0.0, 0.0));
    }
    else
    {
        p.varHPer = share * varHSq;
    }
    p.sPer = max(p.ePer - p.varHPer - kResampleVarSq, 0.0);
    return p;
}

float WhitenedInnovSq(float3 innovCorr, RecordVarianceParts p)
{
    float3 invE = float3(1.0, 1.0, 1.0)
                / max(p.ePer, float3(1e-10, 1e-10, 1e-10));
    return dot(innovCorr * innovCorr, invE);
}

// ============================================================================
// THE MAHALANOBIS GATE (the hierarchical null)
// ============================================================================
struct ClipGateResult
{
    float3 clippedColorSpace;
    float3 center;          // the gate's content reference (the posterior of
                            // the history's accumulation target): also the
                            // varH seed's dCorr reference and the soft
                            // clip's action target
    float  tGate;
    float3 rawGateVar;      // the UN-INFLATED per-channel gate variance
                            // (base + transport, pre-chroma-scale): the
                            // varH seed's input -- a physical state, not a
                            // test radius
    float  mdd;              // the realized Mahalanobis statistic
                            // (mode-13 telemetry: the realized-null-law
                            // histogram; the empirical Student factor
                            // is its 95th percentile / chi^2)
    float  p1;               // the ghost posterior (the blend floor's
                            // input), exported from the CONFIRMED
                            // persistence evidence.
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
    r.center = stats.mean;
    r.tGate = 1.0;
    r.rawGateVar = float3(kMinSigma * kMinSigma, kMinSigma * kMinSigma, kMinSigma * kMinSigma);
    r.mdd = 0.0;

    // The sequential evidence is the persistence detector's CONFIRMED
    // statistic (either channel), mapped through the exact tau = sigma
    // Gaussian-prior Bayes factor. ZERO below the confirmation threshold:
    // on ~99.9% of pixels the gate runs as the pure clip and the detector
    // contributes NOTHING to its null behavior.
    float seqFloor = 0.0;
    if (taaSoftClip > 0.001 && ghostEvNats > 0.0)
    {
        float logOddsSeq = kGhostLogOddsBase + ghostEvNats;
        seqFloor = 1.0 / (1.0 + exp(-logOddsSeq));
    }
    r.p1 = seqFloor;

    float  minGateVar = kMinSigma * kMinSigma;
    float3 gateVar;
    // The UN-inflated belief base, hoisted: the posterior and the varH
    // seed consume it after both branches close.
    float3 gateVarBase = float3(minGateVar, minGateVar, minGateVar);
    float3 center = stats.mean;
    float  nuPost = 4.0;
    float  stepW = saturate(stats.stepW);

    if (recordLive)
    {
        // ====================== THE HIERARCHICAL NULL ======================
        // See the architecture header. The prior's plane / lever / residual
        // are the WORKING package (same-layer on healthy straddles, the
        // degenerate-aware full set on thin detail); the tap correction
        // uses the SAME gradient the plane was evaluated with, so the
        // plane parts cancel exactly under static content.
        float nuRecord = StudentEffectiveDof(recordAge, stats.recordSampleDof);
        float nuRecordSafe = max(nuRecord, 2.0);

        float3 sPer    = parts.sPer;
        float3 varHPer = parts.varHPer;

        // tauB2: the temporally STABLE structural variance -- the spatial
        // residual minus the per-sample temporal variance.
        float3 tauB2 = max(stats.residSq - sPer, float3(0.0, 0.0, 0.0));
        float3 vFit  = stats.residSq * (stats.invNeff + stats.lever);
        float3 vPrior = tauB2 + vFit + stats.covVarFloor;

        // The t_nu tap likelihood, one IRLS step: the deviation
        // standardized against the TOTAL predictive width redescends
        // through the t's bounded influence -- a firefly widens the gate
        // and stays out of the center; structural deviations (q ~ O(1))
        // barely notice.
        float3 x0c = currentColorSpace - stats.phaseShiftPlane;
        float3 qTap = (x0c - stats.plane0) * (x0c - stats.plane0)
                    / max(vPrior + sPer, float3(1e-10, 1e-10, 1e-10));
        float3 sPerEff = sPer * (1.0 + qTap / nuRecordSafe);

        float3 wTap = vPrior / max(vPrior + sPerEff, float3(1e-10, 1e-10, 1e-10));
        float3 centerFlat = lerp(stats.plane0, x0c, wTap);
        float3 vPost = vPrior * sPerEff / max(vPrior + sPerEff, float3(1e-10, 1e-10, 1e-10));

        // ---- the Satterthwaite dof ----
        // A = varH + w^2 * sPerEff: record-derived (the record's clock;
        // varH and sPer derive from the same transported scalar).
        // B = (1-w)^2 * vPrior: vFit rides the resid dof (n_eff - 3);
        // tauB2 is a DIFFERENCE of two variance estimates, whose own dof is
        // the harmonic of its sources -- the difference-noise folds into
        // the prior's denominator directly (tau^2/nu_tau = residSq^2/nu_resid
        // + sPer^2/nu_record by construction), so no separate guard is
        // needed. The prior/tap cross-covariance (positive, zero at both
        // structure limits, ~0.1-0.2 tau^2 mid-range) is deliberately
        // unmodeled: this dof pricing is what absorbs it (the Student
        // inflation at these dofs exceeds the shortfall everywhere in the
        // range). CALIBRATION (the mode-13 procedure): if a static scene's
        // 95th percentile lands systematically dimmer than nominal, raise
        // the resid count floor below; brighter = the gate runs hot.
        float  nuResid = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
        float3 denomTau = stats.residSq * stats.residSq / nuResid
                        + sPer * sPer / nuRecordSafe;
        float3 nuPrior = vPrior * vPrior
                       / max(vFit * vFit / nuResid + denomTau, float3(1e-10, 1e-10, 1e-10));
        nuPrior = max(nuPrior, float3(2.0, 2.0, 2.0));

        float3 A = varHPer + wTap * wTap * sPerEff;
        float3 B = (1.0 - wTap) * (1.0 - wTap) * vPrior;
        float3 nuFlat = (A + B) * (A + B)
                      / max(A * A / nuRecordSafe + B * B / nuPrior, float3(1e-10, 1e-10, 1e-10));
        nuFlat = max(nuFlat, float3(2.0, 2.0, 2.0));

        // ---- the step blend (the coverage model's own target / variance) ----
        float3 gateVarStep = varHPer + stats.stepTargetVar;
        gateVarBase = lerp(A + B, gateVarStep, stepW);
        float  nuStepC = max(stats.cusumStepDof, 2.0);
        float3 nuEff = lerp(nuFlat, float3(nuStepC, nuStepC, nuStepC), stepW);
        float3 studentS = 1.0 + taaStudentA / nuEff + taaStudentB / (nuEff * nuEff);
        gateVar = max(gateVarBase * studentS, float3(minGateVar, minGateVar, minGateVar));
        center = lerp(centerFlat, stats.stepTarget, stepW);

        // the posterior's scalar dof (the variance-weighted channel mean)
        float gSum = gateVar.x + gateVar.y + gateVar.z;
        nuPost = max(dot(nuEff, gateVar) / max(gSum, 1e-10), 2.0);
    }
    else
    {
        if (stats.stepW > 0.5)
        {
            // The cold step path is Studentized at the levels' own
            // estimation dof (S() at that dof restores the coded coverage).
            float studentStep = StudentFactor(max(stats.cusumStepDof, 2.0));
            gateVarBase = stats.stepTargetVar;
            gateVar = max(studentStep * stats.stepTargetVar, float3(minGateVar, minGateVar, minGateVar));
            center = stats.stepTarget;
        }
        else
        {
            // Deliberately NOT a coverage claim -- the cold-start PRIOR:
            // after a reset/spike the policy is to pull the history hard
            // toward the live neighborhood and re-establish trust (the
            // blend floor / current-frame replacement carries the reveal
            // side); honest coverage lives on the record-live path above.
            // The phase term and the residual ride the WORKING LS package
            // (same-layer on healthy straddles; the degenerate-aware full
            // set on thin detail), and the coverage-mixture variance is
            // added unscaled by invNeff (it is the VALUE's own
            // uncertainty, not a mean-estimation error).
            float3 varX = taaJitterPhaseEx  * (stats.gradLSx * stats.gradLSx)
                        + taaJitterPhaseEy  * (stats.gradLSy * stats.gradLSy)
                        + 2.0 * taaJitterPhaseExy * (stats.gradLSx * stats.gradLSy)
                        + stats.residSq;
            gateVarBase = stats.invNeff * varX + stats.covVarFloor;
            gateVar = gateVarBase;
            center = stats.plane0;
        }
        // The posterior's dof mirrors the live path's structure: the cold
        // flat variance is a resid-dof estimate (n_eff - 3), the cold step
        // variance rides the levels' dof -- not the bare cusumStepDof
        // default (2) the flat path would otherwise fall through to.
        {
            float nuResidCold = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
            float nuStepCold  = max(stats.cusumStepDof, 2.0);
            nuPost = lerp(nuResidCold, nuStepCold, stepW);
        }
    }

    // Coherence note (deliberate): the transport scatter is added AFTER
    // the Studentization -- it is not S(nu)-inflated. Its static floor is
    // a stable min-statistic (the measured velocity-quantization scale),
    // but the motion-scaled share is a single noisy measurement; mode 13
    // UNDER MOTION is the view that would expose any under-coverage here.
    // The Kaiser read's own structural wander at sub-texel landings is
    // NOT separately charged: the record measures the realized innovation
    // (which includes it) within its EMA lag, and charging it here too
    // would double-count against the record.
    float3 transportTerm = (1.0 - stepW) * stats.transportVar;
    gateVar += transportTerm;
    // The varH seed's input: the UN-INFLATED base -- varH is a physical
    // state (the accumulated read variance), not a test radius; the
    // Student factor is the hard test's coverage device and does not
    // belong in the transported state (the host's kernel-chain fixed
    // point the recursion feeds is a physical model).
    r.rawGateVar = max(gateVarBase + transportTerm, float3(minGateVar, minGateVar, minGateVar));

    float3 dCorr = historyColorSpace - center;
    r.center = center;

    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    // FULL METRIC (the sample correlations, clamped). The correlations
    // are SHRUNK toward zero BEFORE the conditioning clamp: at the full
    // set's effective count (~7) a sampled correlation carries ~0.4
    // stderr -- an inflated one narrows the ellipsoid along its direction
    // (spurious clips -- flicker), a deflated one widens it (ghost
    // shelter). The shrink strips the leading-order estimation-noise
    // bias. covCross is the same-layer cross-covariance and the
    // denominators follow it (sigmaClean; = sigma unstraddled): with the
    // full-set sigma in the denominator the same-layer covariances would
    // be shrunk by the sigma ratio -- a spurious deflation toward zero,
    // i.e. ghost shelter along the luma direction on every straddle.
    float rDof = max(1.0 / max(stats.invNeff, 1e-3) - 1.0, 2.0);
    float corrShrink = max(0.0, 1.0 - 1.0 / rDof);
    float vX = max(gateVar.x * chromaScale.x * chromaScale.x, minGateVar);
    float vY = max(gateVar.y * chromaScale.y * chromaScale.y, minGateVar);
    float vZ = max(gateVar.z * chromaScale.z * chromaScale.z, minGateVar);
    float sX = sqrt(vX), sY = sqrt(vY), sZ = sqrt(vZ);
    float rXY = clamp((stats.covCross.x / max(stats.sigmaClean.x * stats.sigmaClean.y, kEpsilon)) * corrShrink, -0.95, 0.95);
    float rXZ = clamp((stats.covCross.y / max(stats.sigmaClean.x * stats.sigmaClean.z, kEpsilon)) * corrShrink, -0.95, 0.95);
    float rYZ = clamp((stats.covCross.z / max(stats.sigmaClean.y * stats.sigmaClean.z, kEpsilon)) * corrShrink, -0.95, 0.95);
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
    r.mdd = mdd;   // exported for the mode-13 null-law telemetry

    float tHard = (mdd > 1e-20) ? min(chi / sqrt(mdd), 1.0) : 1.0;

    bool posteriorPath = (taaSoftClip > 0.001) && (mdd > chi * chi);
    if (posteriorPath)
    {
        // ====================== THE GHOST POSTERIOR ======================
        // H0: the hierarchical null -- the posterior center at its honest
        // UN-INFLATED predictive variance (the Student factor is the hard
        // test's coverage device and is not part of the belief; the t at
        // the Satterthwaite dof prices the variance estimation noise
        // here instead, and the hard ray clamp below still bounds the
        // action -- the posterior can only pull FURTHER toward the
        // current frame than the hard floor requires). H1: the history
        // holds one of the 8 NEIGHBOR TEXELS' accumulated content (what a
        // landing-miss ghost is: this pixel's history read pulled a
        // neighbor's accumulation) or the temporal tail. THE CENTER TAP IS
        // NOT A COMPONENT -- it IS the null's likelihood; the old mixture
        // put it in H1 and generated positive ghost evidence for exactly
        // the valid fine-detail outliers the null now explains.
        //
        // The components' scale: varH + sPer -- the read noise plus the
        // tap's own sampling variance about an accumulated value (the
        // record's implied per-sample law, on steps too: the
        // step-referenced record makes sPer the flip-draw variance there).
        // The transport term is charged symmetrically to both hypotheses
        // (the landing error moves the READ regardless of whose content
        // it holds). On cold pixels (no record, no sPer) the null's own
        // scale carries the belief.
        float3 nullScale = max((gateVarBase + transportTerm) * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));
        float3 vCompRaw = recordLive
            ? parts.varHPer + max(parts.sPer, float3(1e-8, 1e-8, 1e-8)) + transportTerm
            : (gateVarBase + transportTerm);
        float3 vComp = max(vCompRaw * (chromaScale * chromaScale), float3(minGateVar, minGateVar, minGateVar));

        float nu = max(nuPost, 2.0);
        float kT = 0.5 * (nu + 3.0);

        // The null's NLL (the diagonal quadratic; the mixture is a proper
        // density either way and the per-frame evidence is a Gibbs step).
        float3 invNull = float3(1.0, 1.0, 1.0) / nullScale;
        float3 invComp = float3(1.0, 1.0, 1.0) / vComp;
        float q0Comp = dot(dCorr * dCorr, invNull);

        // The tap log-weights: the jitter-centered spatial prior
        // renormalized over the 8 neighbor components -- a PROPER mixture.
        // Closed form for the jitter weights: exp2(-d^2 log2 e) has
        // natural log exactly -d^2 (requires kLog2E == log2(e)); the
        // table-weight path's logs are compile-time-adjacent constants.
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
            // The phase-corrected component residual: the neighbor texel's
            // accumulated estimate is tap_j + g*j (the plane's phase
            // offset converts the jittered sample into an estimate of what
            // that texel's accumulation holds); on edges this term is up to
            // half the local contrast and phase-correlated -- the old raw
            // dd = h - tap_j was a jitter-flicker source inside the
            // posterior itself.
            float3 dd = dCorr + (center - neighborhoodTaps[j]) + stats.phaseShiftPlane;
            float  qj = dot(dd * dd, invComp);
            float2 dj = kOffsets3x3[j] - stats.jitterPx;
            float  lw = jitW ? (-dot(dj, dj)) : log(max(stats.mixWeights[j], 1e-4));
            lj[j - 1] = -kT * log(1.0 + qj / nu) + lw + logTapW;
            lMax = max(lMax, lj[j - 1]);
        }

        // The temporal tail: the old content absent from the 9-tap sample,
        // at the shared kGhostPriorSigma bandwidth around the null's
        // center (the one bandwidth, two constructions of the same prior).
        float3 sigAC = stats.sigmaClean;
        float3 tailScale = nullScale + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invTail = float3(1.0, 1.0, 1.0) / tailScale;
        float qTail = dot(dCorr * dCorr, invTail);

        // The exact log-density bookkeeping for UNEQUAL scales: each
        // component's full log-density carries the normalizer
        // -0.5*sum(ln(vComp_ch)); the shared part strips out of the
        // log-sum-exp and restores against the null's, and the tail
        // carries its own relative term:
        //   llr = logsumexp_j[ln w_j - kT ln(1+qj/nu)]
        //         (+ the tail's ln tailW - kT ln(1+qTail/nu)
        //            - 0.5*sum ln(tailScale/vComp))
        //         + kT ln(1+q0/nu) + 0.5*sum_ch ln(nullScale_ch / vComp_ch)
        // (the t's Gamma constants cancel -- one nu everywhere). At equal
        // scales this reduces exactly to the previous form (the shared
        // normalizers cancel; the tail's term is the old lTailNorm).
        float lTailNorm = 0.5 * (log(tailScale.x / vComp.x) + log(tailScale.y / vComp.y) + log(tailScale.z / vComp.z));
        float lTail = log(kCusumTailWeight) - kT * log(1.0 + qTail / nu) - lTailNorm;
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 8; ++j2)
            wSum += exp(lj[j2] - lMax);
        wSum += exp(lTail - lMax);

        float lNormNull = 0.5 * (log(nullScale.x / vComp.x) + log(nullScale.y / vComp.y) + log(nullScale.z / vComp.z));
        float llrFrame = (lMax + log(max(wSum, 1e-10))) + kT * log(1.0 + q0Comp / nu) + lNormNull;

        // The frame evidence always adds: the persistence statistics are
        // computed from the drift estimator's common modes, DIFFERENT
        // objects than this frame-LLR on dCorr; there is no double count.
        float logOdds = kGhostLogOddsBase
                      + log(1.0 + kGhostMotionOdds * saturate(motionNormalized))
                      + ghostEvNats
                      + llrFrame;
        float p1 = 1.0 / (1.0 + exp(-logOdds));
        r.p1 = max(p1, seqFloor);   // the blend floor export

        // The scalar pull -- the posterior-mean action: under H0 the valid
        // history IS the best estimate (keep it); under H1 the replacement
        // is the current content estimate (the hierarchical center, which
        // carries both the neighborhood plane and the tap). Floored by
        // the hard ray clamp (the containment backstop).
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
// CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    // Normalized by the OWN-LAYER spread (sigmaClean; = sigma unstraddled)
    // -- the distance metric reads "how far did the gate move the history,
    // in units of this texel's own content scale".
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / stats.sigmaClean;
    float maxChannelDistance = max(clipDistance.x, max(clipDistance.y, clipDistance.z));
    return saturate((maxChannelDistance - taaClipDistanceRejectionMinError) * taaClipDistanceRejectionAmount);
}

// The manual fallback for the drift predict step -- redundant while Drift
// Compensation is on and off by default.
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