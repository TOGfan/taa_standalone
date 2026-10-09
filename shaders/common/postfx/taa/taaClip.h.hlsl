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
//     edge flags: a same-depth neighbor flagged as foreground edge or kept
//     dilation band carries the edge transition's accumulation in its
//     history -- its mismatch is the transition's state, not this texel's.
//   * THE PERSISTENCE DETECTOR: two transported scalars -- T, the EMA of
//     the drift estimator's luma t-statistic, and Tc, the EMA of the
//     half-wave chroma drift vector magnitude (the chroma channel extends
//     the detector's AXIS coverage to saturated chroma trails with
//     sub-floor luma deltas) -- each self-normalized to a pinned null.
//     Their consumers are all smooth actions: the drift-corrector UNLOCK
//     (the ghost-trail eviction engine), the confirmed blend floor, and
//     the alarm. Nothing migrates the clip target -- a moving target is
//     an edge-flicker source.
//   * THE STATISTIC GATE: the mu share is Studentized at its own
//     estimation dof (Satterthwaite over the transported record and the
//     spatial mean/target estimate); the sampled correlations are shrunk
//     before the conditioning clamp; the full metric is the 3x3
//     Mahalanobis distance.
//   * THE STRADDLE MODEL: while off-layer mass is partitioned out, the
//     clip target is the pixel's OWN layer's value (the clean statistics),
//     and the step model blends the coverage target in by its adequacy.
//     The acceptance keeps the coverage-mixture uncertainty at EVERY
//     adequacy (covVarFloor): the unmodelled uniform share (1-w)/12 --
//     cheap, since admission failure bounds the step below twice the
//     levels' own noise -- and the measured bracket share w*varCeff.
//   * THE CARRYOVER CAPS: the record AND its transported VarH share are
//     arming-capped against the current spatial prior (the share is a
//     component of the record's total, E[i^2] = varH + Var(x) >= varH; an
//     uncapped share self-sustains through the (1-a)^2 damp on texels
//     leaving a dilation band -- the trailing-ghost state).
//   * THE SOFT CLIP: beyond the gate radius the action is the exact
//     posterior mean against the empirical 10-component alternative (the
//     9 taps sharp + one temporal tail), with the prior odds carrying the
//     confirmed persistence evidence.
//
// TRANSPORT (tag 101): [31] sign, [30:28] tag, [27:22] sigma (6-bit),
// [21:16] the chroma persistence statistic Tc (0.25 quanta, code =
// round(4Tc) + 32; nonnegative by the half-wave input, so codes 32-63),
// [15:10] VarH ratio (6-bit), [9:6] age, [5:0] the luma persistence
// statistic T (0.25 quanta, code = round(4T) + 32; the sign is the luma
// direction).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host
// context: cbuffer perDraw (the clipping + feedback constants,
// taaStudentA/B, taaJitterPhaseEx/Ey/Exy, taaClipGhostReset,
// taaClipScopedMu, taaCusumFlipAcc, taaCusumNoiseAcc,
// taaVarhResampleLoss, taaVarhWhiteLoss, taaDriftCompensation,
// taaDriftMaxGain, taaJitterAwareVariance). Requires taaShared.h.hlsl,
// taaConstants, taaFrame.h.hlsl included before.
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
    float3 mean;                // the step target on engaged straddles (the
                                // coverage fit), the same-layer mean
                                // otherwise
    float3 sigma;               // full set (mixture): smear rejection
    float3 sigmaClean;          // same-layer subsample (the B level, raw)
    float  invNeff;             // the target's effective count inverse
                                // (the same-layer count on straddles)
    float  invNeffClean;
    float  invNeffF;            // the F-side effective count inverse (the
                                // step dof input; 1.0 when no off-layer
                                // mass -- the safe divide)
    float3 phaseShift;          // FUSED: the step form on engaged steps,
                                // the same-layer CD form otherwise. The
                                // primary innovation phase correction is
                                // the estimator's LS gradient (taa.fx);
                                // this field is the fallback and the
                                // step-path model.
    float3 gatePhaseShift;      // FUSED: 0 on engaged steps
    float3 gradCDx;             // central differences: the same-layer
    float3 gradCDy;             // gradients (the seed / cold-gate phase
                                // energy consumers)
    float3 gradCDxClean;        // same-layer (off-layer sides -> flat)
    float3 gradCDyClean;
    float3 meanClean;
    float3 phaseShiftClean;
    float3 gatePhaseShiftClean;
    float  centerWeight;        // w0' (the same-layer center share on
                                // straddles)
    float  effRank;             // r in [1,3]: the per-draw dof
    float3 covCross;            // the taps' cross-channel covariances
                                // (xy, xz, yz) -- the same-layer moments on
                                // straddles; the measured correlation SHAPE
                                // the gate's full metric consumes
    float3 residSq;             // the weighted-LS plane residual of the
                                // clip target (dof-corr; the same-layer
                                // plane on straddles)
    float  rangeSq;             // full-set squared AABB range
    float  rangeSqClean;        // same-layer squared range
    bool   straddled;           // any off-layer tap (mask valid)
    // ---- the coverage step fit ----
    float  stepW;               // the adequacy blend [0,1]
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
                                // per-channel color variance -- the landing
                                // error (motion-scaled jitter residual /
                                // measured static floor) projected on the
                                // content gradient, MOTION-GATED
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
                                // for the mixtures a straddled pixel
                                // legitimately accumulates
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
// rho = 0.15 is the chosen operating point. The statistically complete
// upgrade would be a second CUSUM on innovSq feeding the record's rate --
// deliberately not built (the transported dof + VarH clocks already absorb
// most of the lag cost, and the arming caps bound the drop-lag window).
static const float kClipSigmaEmaRate         = 0.15;
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);
// The record's outlier guard. DERIVATION (honest, both regimes): the test
// compares the corrected innovation against max(record, range). On ordinary
// content the AABB range dominates (range^2 ~ 25x E[i^2] per channel on a
// Gaussian flat) and the test fires only when the history lands OUTSIDE the
// current neighborhood -- the intended semantics, mortality ~0. In the
// record-dominated regime (temporal noise above the spatial range --
// sparkles, fire) the record converges to E[i^2] and the mortality is the
// chi^2_3 tail at 3*ratio: r = 4 -> P(chi^2_3 > 12) ~= 6.2%/frame (cheap:
// resets re-seed from the spatial model and keep the Student dof low --
// anti-flicker there); a 0.1% target would need r ~= 7.4. 4.0 keeps the
// spike's content-change latency low; retune ONLY with the mortality
// number, not by eye.
static const float kClipSpikeRatio           = 4.0;
static const float kClipWinsorC              = 3.0;
static const float kGeoCorroborateChiSq      = 7.815;         // chi^2_3(0.95), whitened
static const float kClipRejectionFeedbackFloor = 0.5;
// The per-channel anisotropy floor: channels with genuinely less content
// still carry the neighborhood's noise floor (a floor at 10% of the max
// channel's sigma).
static const float kAnisoSigmaCap            = 0.10;
// The residual floor share: the same-layer subsample's residual variance
// can collapse on synthetic flats; 10% of sigma_clean^2 keeps the scoped mu
// variance positive.
static const float kScopedResidFrac          = 0.10;
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
// The record-vs-prior arming cap. The honest upper bound for a nu-dof
// variance estimate against its prior is chi^2_nu(0.999)/nu (~3.7 at the
// flats' nu ~ 6); 5.0 adds EMA-lag headroom for rising variance. Under
// honest dynamics it should rarely bind -- verify with mode 13 + the record
// telemetry; if it binds measurably the winsor ingest (9x prior) is the
// dial that actually needs attention. The cap binds the record AND the
// transported VarH share (SplitRecordVariance): the share is a component
// of the record's total (E[i^2] = varH + Var(x) >= varH), so the prior
// bound applies a fortiori.
static const float kArmingCap                = 5.0;
// The resample's variance effect is carried by the multiplicative damping
// in the VarH recursion (taaVarhResampleLoss / taaVarhWhiteLoss); the
// additive slot stays 0.
static const float kResampleVarSq            = 0.0;
// ---- the posterior-mean soft clip ----
static const float kGhostLogOddsBase         = -6.2383;       // ln(1/512)
// The motion term of the ghost prior, DERIVED as an order-of-magnitude
// population ratio (the one underived population number left -- calibratable
// from mode-11 telemetry): at an ENGAGED swept step the stale-landing
// probability is the off-coverage ~1/8; on flats it is the base 1/512. The
// ratio 64 = (1/8)/(1/512) is the prior-odds boost at full motion, and it
// only matters where the frame evidence is ambiguous (the frame LLR
// overwhelms it on clean nulls and clean ghosts alike).
static const float kGhostMotionOdds          = 64.0;
// The temporal tail width, in units of the local CLEAN-layer sigma: the
// old content that is ABSENT from the 9-tap sample (what a ghost IS, when
// it is not one of the taps) lives in the distribution's tail beyond it.
// Shared by BOTH consumers (the CUSUM's tail component and the gate
// posterior's tail component) -- the one bandwidth, two constructions of
// the same prior.
static const float kGhostPriorSigma          = 1.0;
// ---- the coverage step fit ----
// The step model's admission SNR (dFB^2 / the levels' error variance):
// half-credit at 8, full above -- a model-selection threshold, not a
// coverage device.
static const float kStepMinLevels = 4.0;
// The crossing bracket's maximum width in texels (the 3x3's diagonal
// reach) -- a geometric support bound.
static const float kStepMaxSpan   = 3.0;
// The direction-aware jitter span floor: the sequence's projected variance
// (JitterProjVar) can collapse toward zero on axis-aligned edges with
// symmetric sequences; the floor keeps the coverage inversion finite.
static const float kMinJitterSpan = 0.35;
// The levels' extrapolation lever share: the pixel-center extrapolation is
// capped at 1 texel^2 of lever, and half the residual variance is charged
// for it.
static const float kStepLeverK    = 0.5;
// ---- THE PERSISTENCE DETECTOR ----
// Two transported scalars per pixel. T: the EMA of the drift estimator's
// per-frame LUMA spatial t, scaled so the H0 law is a pinned ~N(0,1)
// (self-normalized: a variance-model error cancels between numerator and
// denominator). Tc: the EMA of the half-wave CHROMA drift vector magnitude
// (see the chroma channel block below). Thresholds are set against the
// t_6 input's variance inflation (nu/(nu-2) = 1.5 -> the normalized null's
// effective std is ~1.22, not 1 -- kGhostNullStdSq below carries this
// number for the evidence mapping, which must price against the same null
// the thresholds do):
//   |T| >= kGhostConfirmT (4.0): ~1e-3 engagement, EMA-smooth. The action
//        is the drift-corrector UNLOCK -- a smooth correction, not a blend
//        jump. This is the ghost-trail eviction engine.
//   |T| >= kGhostAlarmT   (6.5): ~1e-7; the blend floor approaches full
//        replacement; mode 3 additionally hard-replaces.
// Luma detection floor: a persistent luma offset >= ~0.6 sigma_content
// confirms in ~10 frames (per-frame t = mu/sigma_D with sigma_D ~ 0.5
// sigma_content; steady-state T = t * kGhostEmaNorm). Chroma-dominant
// trails are the chroma channel's business (below); the luma channel
// alone does not see them.
static const float kGhostEmaRate   = 0.15;   // the detectors' EMA rate (memory ~13 frames)
static const float kGhostEmaNorm   = 3.5119; // sqrt((2-rate)/rate): the EMA's null std -> unit
static const float kGhostConfirmT  = 4.0;    // the confirmation threshold (|T| luma / Tc chroma)
static const float kGhostAlarmT    = 6.5;    // the deep-confirmation / alarm threshold
static const float kGhostUnlockThrSigmas = 0.75; // the CONFIRMED drift threshold, in estimator sigmas (vs 2.0 unconfirmed)
// The luma statistic's REALIZED null variance. The per-frame t input is
// t_dof at dof ~ 6 (9 same-layer taps - mean - 2 gradient dof), variance
// nu/(nu-2) = 1.5; kGhostEmaNorm normalizes iid UNIT-variance inputs, so
// T's null variance is 1.5 (std ~ 1.2247). The |T| thresholds above are
// set against it -- and the EVIDENCE mapping (kGhostEvCoef, consumed by
// the resolve) prices against it too. Successive t frames are also
// positively correlated (overlapping 3x3 windows, the history carryover):
// if mode-11 telemetry on static NOISY content shows sustained |T| > ~1.5,
// raise this to the measured inflation rather than moving the thresholds.
static const float kGhostNullStdSq = 1.5;
// ln BF = T^2 / (4 sigma0^2) - 0.5 ln 2  (tau = sigma0 Gaussian prior),
// priced against the detector's REALIZED null (sigma0^2 =
// kGhostNullStdSq). A unit-sigma pricing would overstate the exponent by
// 1.5x: at T = 4 the blend floor would read 7% where the honest value is
// 2%; at T = 6 it would read 92% vs 36%.
static const float kGhostEvCoef = 0.25 / kGhostNullStdSq;   // 1/6
// ---- THE CHROMA PERSISTENCE CHANNEL ----
// A second transported scalar, Tc: the EMA of the half-wave chroma drift
// VECTOR magnitude, extending the detector's AXIS coverage to the trails
// the luma channel cannot see -- saturated chroma smears with sub-floor
// luma deltas (colored lights, emissives on neutral backgrounds). The
// per-frame input is
//     tC = max( length(driftMean.yz / sqrt(driftVar.yz))
//               - kGhostChromaNullMean, 0.0 ) * kGhostChromaScale
// over the estimator's own per-channel scales. The VECTOR magnitude is
// DIRECTION-BLIND-FREE (any chroma direction reads at full vector
// strength; a signed per-axis pick would cancel on the anti-diagonal half
// of the chroma plane). The HALF-WAVE at the null mean makes the channel
// ONE-SIDED (quiet chroma is never evidence) and keeps gray content at
// Tc ~ 0 instead of a deeply negative baseline a new trail would have to
// climb out of. The CENTERING is essential: a magnitude's null mean is
// nonzero, and an uncentered input would confirm everywhere.
// NULL LAW: under H0 each per-axis chroma t is the same t_6-family
// statistic as the luma channel; the vector magnitude's mean and variance
// then depend on the chroma channels' correlation rho (E[m]: ~1.5 at
// rho=0 falling to ~1.3 at rho=1; var(m): ~0.65 rising to ~1.25 -- both
// bounded, since E[m^2] = 2x the per-axis variance regardless of rho).
// The constants anchor the CONSERVATIVE end of each band (the mean at its
// maximum, the variance at its maximum), so the SHARED thresholds and the
// SHARED evidence coefficient carry a per-channel false rate <= the luma
// channel's everywhere in the band, one-sided. Both are calibration
// points, not tuning knobs -- verify on static content via the mode-11
// telemetry (the channel-separation swap is documented at the dbgB site).
// FLOOR: the channel's detection floor is a chroma vector magnitude of
// ~2.5 per-axis sigma_D (~2.2x the luma floor: the centering subtraction
// plus the conservative anchors) -- the mission is saturated chroma
// trails (multi-sigma vectors), not depth parity with the luma channel.
static const float kGhostChromaNullMean = 1.50;   // E[m] under H0, the rho=0 anchor
static const float kGhostChromaNullVar  = 1.25;   // var(m) under H0, the rho=1 anchor
static const float kGhostChromaScale    = sqrt(kGhostNullStdSq / kGhostChromaNullVar);  // ~1.10
// The temporal-tail component's prior mass, shared by the gate's posterior
// alternative (9 taps + 1 tail, 1/10 each).
static const float kCusumTailWeight   = 0.1;
// ---- the drift predict step ----
static const float kDriftCapSigmas  = 3.0;   // the magnitude BELT (content sigmas)
// The soft threshold (estimator sigmas). Its position relative to the
// Studentized gate radius sets the smooth-lighting band: the flat gate's
// mu share carries S(nu~6), putting the gate at ~1.4 sigma_content, and
// the drift estimator's own noise (mean + phase-correction error +
// transport scatter ~ 0.5 sigma_content/channel) puts 3 sigma at ~1.5 --
// ABOVE the gate, i.e. the tracker would be redundant exactly where it is
// meant to be the smooth alternative. 2.0 engages it at ~1.0
// sigma_content, inside the gate, with the noise cost still bounded by the
// SNR gain and taaDriftMaxGain. This is the UNCONFIRMED gate; confirmation
// (either channel) drops it to kGhostUnlockThrSigmas.
static const float kDriftThreshSigmas = 2.0;
// The lighting-rate prior -- the per-frame drift scale as a fraction of
// the content scale. Prices H_drift in the event guard's posterior (see
// the drift block in taa.fx): smooth lighting moves at most ~half the
// content scale per frame; larger common-mode offsets are step-like and
// the reveal paths own them. The posterior's crossover sits near
// sigma_clean regardless of this value (bounded by the wider hypothesis);
// the dial shapes how FAST the stand-down happens.
static const float kDriftPriorScaleSigmas = 0.5;
// ---- the transport scatter's motion gate ----
// The static floor is MEASURED per pixel (the velocity buffer's own
// quantization/noise scale, passed in by the resolve). kTransportGatePx is
// the motion scale over which the jitter-residual proxy ramps to full
// charge (the landing error grows with the reprojection error, a fraction
// of the velocity).
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

float3 AnisoClampSigma(float3 sigma)
{
    float sMax   = max(sigma.x, max(sigma.y, sigma.z));
    float sFloor = max(sMax * kAnisoSigmaCap, kMinSigma);
    return max(sigma, float3(sFloor, sFloor, sFloor));
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
    // loop-invariant -- hoisted once (the staircase alone would evaluate
    // the projection 81 times). uMid = dot(mid - B0, dFB) with
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
    bool sameLayerMask[9], bool layerMaskValid,
    float transportFloorPx,
    bool historyTapIsBand[9])
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
    stats.jitterPx = jitterPx;

    bool exactSeedInputs = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0) && (taaClipScopedMu > 0.001);
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
    float  centerWeightRaw  = 0.0;
    float  recDofWSum = 0.0, recDofWSq = 0.0, recDofUsed = 0.0;

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);
    bool clipScopedOn = (taaClipScopedMu > 0.001);

    // The jitter-centered Gaussian is separable --
    // exp(-(dx^2+dy^2)) = exp(-dx^2)*exp(-dy^2) -- so the nine weights
    // cost six exp2 (one per axis per distinct offset) plus nine
    // multiplies. Every consumer uses the weights only up to
    // normalization, so the <= 2 ulp product-vs-sum exponent difference
    // is immaterial.
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

        if (i == 0) centerWeightRaw = w;

        weightedSum       += tapColorSpace * w;
        weightedSumSq     += tapColorSpace * tapColorSpace * w;
        totalWeight       += w;
        totalWeightSq     += w * w;
        weightedOffsetSum += tapOffsetPx * w;
        wSumXY += w * tapColorSpace.x * tapColorSpace.y;
        wSumXZ += w * tapColorSpace.x * tapColorSpace.z;
        wSumYZ += w * tapColorSpace.y * tapColorSpace.z;
        w9[i] = w;
        if (clipScopedOn)
        {
            weightedOffXSqSum += tapOffsetPx.x * tapOffsetPx.x * w;
            weightedOffYSqSum += tapOffsetPx.y * tapOffsetPx.y * w;
            weightedOffXYmSum += tapOffsetPx.x * tapOffsetPx.y * w;
            weightedCovXSum   += tapOffsetPx.x * tapColorSpace * w;
            weightedCovYSum   += tapOffsetPx.y * tapColorSpace * w;
        }
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean    = weightedSum * invTotalWeight;
    stats.sigma   = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));
    stats.invNeff = saturate(totalWeightSq * invTotalWeight * invTotalWeight);
    stats.centerWeight = centerWeightRaw * invTotalWeight;

    float2 kernelCentroidPx = weightedOffsetSum * invTotalWeight;

    float3 gradCDx = 0.5 * (neighborhoodColorSpace[4] - neighborhoodColorSpace[3]);
    float3 gradCDy = 0.5 * (neighborhoodColorSpace[2] - neighborhoodColorSpace[1]);
    stats.gradCDx = gradCDx;
    stats.gradCDy = gradCDy;

    // The transport scatter, MOTION-GATED with a MEASURED static floor.
    // The raw proxy (jitterResidualPx^2 * grad^2) would charge the
    // jitter-phase scale even on static scenes, where the phase is already
    // removed from the tested statistic by the phase correction and
    // charged into the record by phaseSq -- a ~3x double count that widens
    // the static gate above the nominal coverage. The landing-error scale
    // the proxy honestly represents grows with motion (the reprojection
    // error is a fraction of the velocity); the STATIC floor is the
    // velocity buffer's own measured quantization/noise scale (mainP's
    // MeasureVelocityQuantStepPx). All consumers (the gate, the detector's
    // drift variance) read the gated value below.
    float transportMotionPx = saturate(motionNormalized) * kMotionFullStrengthPx;
    float motionShare        = saturate(transportMotionPx / kTransportGatePx);
    float2 landingErrPx;
    landingErrPx.x = max(abs(jitterResidualPx.x) * motionShare, transportFloorPx);
    landingErrPx.y = max(abs(jitterResidualPx.y) * motionShare, transportFloorPx);
    stats.transportVar = (landingErrPx.x * landingErrPx.x) * (gradCDx * gradCDx)
                       + (landingErrPx.y * landingErrPx.y) * (gradCDy * gradCDy);

    stats.phaseShift     = -(gradCDx * jitterPx.x + gradCDy * jitterPx.y);
    stats.gatePhaseShift = gradCDx * (kernelCentroidPx.x - jitterPx.x)
                         + gradCDy * (kernelCentroidPx.y - jitterPx.y);

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

    // The full-set weighted LS plane (decoupled per axis; the gradients
    // feed only the residual) and its residual in CLOSED FORM from the
    // moments the main pass already accumulated:
    //   SSR/W = Scc - 2*(gX*covX + gY*covY)
    //           + gX^2*vXX + gY^2*vYY + 2*gX*gY*vXY
    // (the mean-subtracted identity; all terms in mean units). The
    // cancellation caveat is honest: the residuals sit 3-6 digits below
    // the tap magnitudes, leaving >= 5 significant digits in fp32 -- the
    // mu share consumes residSq at percent scale.
    float3 lsGX = 0.0;
    float3 lsGY = 0.0;
    if (clipScopedOn)
    {
        float  vXX = max(weightedOffXSqSum * invTotalWeight - kernelCentroidPx.x * kernelCentroidPx.x, 1e-6);
        float  vYY = max(weightedOffYSqSum * invTotalWeight - kernelCentroidPx.y * kernelCentroidPx.y, 1e-6);
        float  vXY = weightedOffXYmSum * invTotalWeight - kernelCentroidPx.x * kernelCentroidPx.y;
        float3 covX  = weightedCovXSum * invTotalWeight - kernelCentroidPx.x * stats.mean;
        float3 covY  = weightedCovYSum * invTotalWeight - kernelCentroidPx.y * stats.mean;
        lsGX = covX / vXX;
        lsGY = covY / vYY;

        float3 scc = weightedSumSq * invTotalWeight - stats.mean * stats.mean;
        float3 ssr = scc
                   - 2.0 * (lsGX * covX + lsGY * covY)
                   + (lsGX * lsGX * vXX + lsGY * lsGY * vYY + 2.0 * lsGX * lsGY * vXY);
        float dofCorr = 1.0 / max(1.0 - 3.0 * stats.invNeff, 1.0 / 3.0);
        stats.residSq = max(ssr * dofCorr, 0.0);
    }
    else
    {
        stats.residSq = 0.0;
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
    stats.meanClean         = stats.mean;
    stats.phaseShiftClean   = stats.phaseShift;
    stats.gatePhaseShiftClean = stats.gatePhaseShift;
    stats.gradCDxClean      = gradCDx;
    stats.gradCDyClean      = gradCDy;
    stats.straddled         = false;
    stats.stepW             = 0.0;
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
        float  offXSqF = 0.0, offYSqF = 0.0;
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
            stats.meanClean       = muC;
            stats.sigmaClean      = max(sqrt(max(ssqC * invWC - muC * muC, 0.0)), kMinSigma);
            stats.invNeffClean    = saturate(wC2 * invWC * invWC);
            stats.rangeSqClean    = dot(cMax - cMin, cMax - cMin);
            float3 cR = sameLayerMask[4] ? neighborhoodColorSpace[4] : muC;
            float3 cL = sameLayerMask[3] ? neighborhoodColorSpace[3] : muC;
            float3 cU = sameLayerMask[2] ? neighborhoodColorSpace[2] : muC;
            float3 cD = sameLayerMask[1] ? neighborhoodColorSpace[1] : muC;
            float3 gBx = 0.5 * (cR - cL);
            float3 gBy = 0.5 * (cU - cD);
            stats.gradCDxClean = gBx;
            stats.gradCDyClean = gBy;
            stats.phaseShiftClean   = -(gBx * jitterPx.x + gBy * jitterPx.y);
            stats.gatePhaseShiftClean = gBx * (cenC.x - jitterPx.x)
                                      + gBy * (cenC.y - jitterPx.y);

            // THE CLEAN REBASING. While off-layer mass is partitioned out,
            // the non-engaged clip target is the pixel's OWN layer's value:
            // the off-layer taps are OTHER TEXELS' values -- in the target
            // they are a bias toward the ghost (a background texel's mean
            // pulled toward the foreground it neighbors), not a variance.
            // The step model owns the target where it engages (the lerp
            // below runs from the clean base); w = 0 is the exact own-layer
            // target and the blend is continuous in the adequacy. NOT
            // applied when wC <= 0.2*totalWeight (the mostly-foreground
            // corner): the coverage is genuinely ambiguous there and the
            // step fit usually engages anyway.
            stats.mean             = muC;
            stats.gatePhaseShift   = stats.gatePhaseShiftClean;
            stats.phaseShift       = stats.phaseShiftClean;
            stats.gradCDx          = gBx;   // the seed's / cold gate's phase
            stats.gradCDy          = gBy;   // gradients: the own-layer ones
            stats.invNeff          = stats.invNeffClean;   // the mu share's / nuMuFlat's count
            stats.centerWeight     = w9[0] * invWC;
            stats.covCross         = float3(momXYC * invWC - muC.x * muC.y,
                                            momXZC * invWC - muC.x * muC.z,
                                            momYZC * invWC - muC.y * muC.z);

            // The same-layer COUPLED plane + its residual in closed form
            // (the mean-subtracted identity on the C-subset moments the
            // partition pass accumulated):
            //   SSR/W = Scc - 2*(gX*gXN + gY*gYN)
            //           + gX^2*mXX + gY^2*mYY + 2*gX*gY*mXY
            // This is the honest estimation noise of the rebased target:
            // the full-set residual would charge the off-layer taps' step
            // residuals to this texel's mu share. The plane solve is
            // COUPLED (the diagonals make mXY nonzero; a decoupled
            // per-axis fit is biased there), with the decoupled form as
            // the degenerate-spread fallback (exact when mXY = 0).
            {
                float  mXXc  = max(offXSqC * invWC - cenC.x * cenC.x, 1e-6);
                float  mYYc  = max(offYSqC * invWC - cenC.y * cenC.y, 1e-6);
                float  mXYc  = offXYC   * invWC - cenC.x * cenC.y;
                float3 gXNc  = covXSumC * invWC - cenC.x * muC;
                float3 gYNc  = covYSumC * invWC - cenC.y * muC;
                float  detNc = mXXc * mYYc - mXYc * mXYc;
                float3 gXc, gYc;
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
                float3 ssrC = sccC
                            - 2.0 * (gXc * gXNc + gYc * gYNc)
                            + (gXc * gXc * mXXc + gYc * gYc * mYYc + 2.0 * gXc * gYc * mXYc);
                float dofCorrC = 1.0 / max(1.0 - 3.0 * stats.invNeffClean, 1.0 / 3.0);
                stats.residSq = max(ssrC * dofCorrC, 0.0);
            }

            if (wF > 1e-4)
            {
                float  invWF = 1.0 / wF;
                float3 muF   = sumF * invWF;
                float2 cenF  = offF * invWF;
                float3 sigmaF = max(sqrt(max(ssqF * invWF - muF * muF, 0.0)), kMinSigma);
                float3 fR = !sameLayerMask[4] ? neighborhoodColorSpace[4] : muF;
                float3 fL = !sameLayerMask[3] ? neighborhoodColorSpace[3] : muF;
                float3 fU = !sameLayerMask[2] ? neighborhoodColorSpace[2] : muF;
                float3 fD = !sameLayerMask[1] ? neighborhoodColorSpace[1] : muF;
                float3 gFx = 0.5 * (fR - fL);
                float3 gFy = 0.5 * (fU - fD);

                // THE CORRECTED LEVELS: pixel-center, phase-stable.
                float3 B0 = muC + gBx * (jitterPx.x - cenC.x) + gBy * (jitterPx.y - cenC.y);
                float3 F0 = muF + gFx * (jitterPx.x - cenF.x) + gFy * (jitterPx.y - cenF.y);
                float3 dFB0 = F0 - B0;

                float2 varOffB = float2(max(offXSqC * invWC - cenC.x * cenC.x, 0.0),
                                        max(offYSqC * invWC - cenC.y * cenC.y, 0.0));
                float2 varOffF = float2(max(offXSqF * invWF - cenF.x * cenF.x, 0.0),
                                        max(offYSqF * invWF - cenF.y * cenF.y, 0.0));
                float3 residVarB = max(stats.sigmaClean * stats.sigmaClean
                                       - (gBx * gBx * varOffB.x + gBy * gBy * varOffB.y),
                                       stats.sigmaClean * stats.sigmaClean * (kScopedResidFrac * kScopedResidFrac));
                float3 residVarF = max(sigmaF * sigmaF
                                       - (gFx * gFx * varOffF.x + gFy * gFy * varOffF.y),
                                       sigmaF * sigmaF * (kScopedResidFrac * kScopedResidFrac));

                float leverB = min(dot(jitterPx - cenC, jitterPx - cenC), 1.0);
                float leverF = min(dot(jitterPx - cenF, jitterPx - cenF), 1.0);
                float3 lvlVarB = residVarB * stats.invNeffClean
                               + stats.sigmaClean * stats.sigmaClean * leverB * kStepLeverK;
                float3 lvlVarF = residVarF * saturate(wF2 * invWF * invWF)
                               + sigmaF * sigmaF * leverF * kStepLeverK;

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

                stats.invNeffF = saturate(wF2 * invWF * invWF);
                stats.cusumStepDof = max(1.0 / max(stats.invNeffClean, 1e-4)
                                       + 1.0 / max(stats.invNeffF, 1e-4) - 4.0, 2.0);

                // The coverage bracket's variance and span, computed at
                // ANY step adequacy -- they describe the pixel's
                // plausible-VALUE uncertainty; the adequacy gate only
                // decides whether the model may move the TARGET. At an
                // unadmitted step (w = 0) the fit's bracket defaults to
                // the full [0, 1] coverage span (cLo/cHi = 0/1): the
                // honest statement when the step is real but unresolvable
                // -- and CHEAP ghosting-wise precisely because admission
                // failed (w = 0 means dFB0^2 <= 4*lvlErr: the step is
                // below twice the levels' own noise, so the mixture it
                // admits is sub-perceptual). Engaged steps keep their
                // measured bracket, and stepTargetVar already carries
                // varCeff there -- the max() at the consumers is a no-op
                // at w = 1.
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
                // (1-w)/12 with the modelled bracket share w*varCeff. A
                // straddled pixel's legitimate accumulated values are the
                // coverage mixtures; without this floor every
                // partially-admitted step (real textured edges live at
                // w in (0,1)) would clip its mixtures -- edge shimmer.
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
                    // edge-flicker source. The bracket stays at the fit's
                    // own support.
                    float cEff = fit.cTarget;
                    float jrN = dot(jitterResidualPx, fit.gDir)
                              / max(sqrt(saturate(12.0 * JitterProjVar(fit.gDir))), kMinJitterSpan);
                    stats.stepPhaseVarScale = jrN * jrN;

                    float3 tStep = B0 + cEff * dFB0;
                    float3 tVar  = (1.0 - cEff) * (1.0 - cEff) * lvlVarB
                                 + cEff * cEff * lvlVarF
                                 + (varCeff + taaCusumFlipAcc * flip + stats.stepPhaseVarScale) * (dFB0 * dFB0);

                    stats.stepC = cEff;

                    float3 tDiff = tStep - stats.mean;
                    stats.mean = lerp(stats.mean, tStep, fit.w);
                    stats.stepTargetVar = tVar + fit.w * (1.0 - fit.w) * (tDiff * tDiff);
                    stats.phaseShift = lerp(stats.phaseShift,
                                            fit.xModel - stats.stepTargetStable,
                                            fit.w);
                    stats.gatePhaseShift = lerp(stats.gatePhaseShift, float3(0.0, 0.0, 0.0), fit.w);
                }
            }
        }
    }

    // Straddle policy in full: the non-engaged target is the own-layer
    // value; the step model blends the coverage target in by its adequacy;
    // the acceptance keeps the coverage-mixture floor at every adequacy;
    // nothing else migrates the clip target.
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
// THE RECORD'S PER-CHANNEL SPLIT
// ============================================================================
struct RecordVarianceParts
{
    float3 ePer;        // per-channel PRE-update E[i_k^2] (arming-capped)
    float3 varHPer;     // per-channel accumulator variance (READ semantics
                        // -- what the gate and the step model consume
                        // directly)
    float3 sPer;        // implied Var(x_k): ePer - varH - v
    float3 muVarFull;   // the mu share (conservative form; the fallback)
    float3 muVarScoped; // the mu share (residual-scoped; the fallback)
    float3 muVar;       // = the effective step target variance when engaged
    float  centerW;     // w0'
    float  spatialPriorSq; // the content-scale reference (the winsor cap's anchor)
};

RecordVarianceParts SplitRecordVariance(
    ColorNeighborhoodStats stats, float recordSqTotal, float varHTotal,
    float alphaStat)
{
    RecordVarianceParts p;
    // The content scale for the residual floor is the OWN-LAYER sigma (the
    // kScopedResidFrac contract: 10% of sigma_clean^2; sigmaClean
    // degenerates to sigma on unstraddled pixels).
    float3 sigA  = AnisoClampSigma(stats.sigmaClean);
    float3 share;

    if (stats.stepW > 0.5)
    {
        float3 dSq = stats.stepDFB * stats.stepDFB;
        share = dSq / max(dSq.x + dSq.y + dSq.z, 1e-12);
    }
    else
    {
        float3 sigAC    = AnisoClampSigma(stats.sigmaClean);
        float  traceSig = max(dot(sigAC, sigAC), kEpsilon);
        share = (sigAC * sigAC) / traceSig;
    }

    float spatialPriorSq;
    if (stats.stepW > 0.5)
    {
        float3 varX = StepVarianceModelX(stats);
        spatialPriorSq = max(varX.x + varX.y + varX.z, kClipSigmaRecordFloorSq);
    }
    else
    {
        float3 sigAC = AnisoClampSigma(stats.sigmaClean);
        spatialPriorSq = max(dot(sigAC, sigAC), kClipSigmaRecordFloorSq);
    }

    float recordSq = min(max(recordSqTotal, 0.0), kArmingCap * spatialPriorSq);
    // The transported VarH is a COMPONENT of the record's total
    // (E[i^2] = varH + Var(x) >= varH), so the arming prior that binds the
    // record binds the share a fortiori. The cap is what closes the
    // band-carryover state -- the texel that just left the dilation band
    // behind a departing foreground: the record enters capped (<= 5x the
    // now-flat spatial prior) while an uncapped transported share would
    // keep the band's edge-scale ratio, sPer's max(...,0) clamp would
    // fire, the recursion's fresh-x term would zero, and the share would
    // decay only through the (1-a)^2 damp -- ~30 frames at default
    // feedback, i.e. a ~0.3 dFB acceptance radius on a flat background:
    // the trailing ghost. The cap binds the share at both ends (the
    // gate's varHTerm here and, via the resolve's recursion input, the
    // transport itself). Legit states never bind: varH <= ~(a/2)E[i^2] <<
    // the cap, and even a 5x-capped sparkly record leaves the share at
    // ~half the cap.
    float varHSq = min(max(varHTotal, 0.0), kArmingCap * spatialPriorSq);
    p.ePer    = share * max(recordSq, kClipSigmaRecordFloorSq);
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
    p.sPer    = max(p.ePer - p.varHPer - kResampleVarSq, 0.0);
    p.centerW = stats.centerWeight;

    if (stats.stepW > 0.0)
    {
        p.muVarFull   = stats.stepTargetVar;
        p.muVarScoped = stats.stepTargetVar;
        p.muVar       = stats.stepTargetVar;
        if (stats.stepW < 1.0)
        {
            float3 residFloor = sigA * kScopedResidFrac;
            float3 w0Floor    = stats.centerWeight * stats.centerWeight * p.sPer;
            float3 scopedBase = max(max(stats.residSq, residFloor * residFloor), w0Floor);
            float3 muVarFb    = lerp(stats.invNeff * (1.0 - alphaStat * 0.5) * p.ePer,
                                    stats.invNeff * scopedBase,
                                    saturate(taaClipScopedMu));
            p.muVar = lerp(muVarFb, stats.stepTargetVar, stats.stepW);
        }
    }
    else
    {
        float3 residFloor = sigA * kScopedResidFrac;
        float3 w0Floor    = stats.centerWeight * stats.centerWeight * p.sPer;
        float3 scopedBase = max(max(stats.residSq, residFloor * residFloor), w0Floor);
        p.muVarScoped = stats.invNeff * scopedBase;
        p.muVarFull   = stats.invNeff * p.sPer;
        p.muVar       = lerp(p.muVarFull, p.muVarScoped, saturate(taaClipScopedMu));
    }

    // THE STRADDLE COVERAGE FLOOR. The gate's acceptance must cover the
    // pixel's plausible ACCUMULATED values; on a straddled pixel those
    // are the coverage mixtures, not just the own-layer values the
    // rebased target centers on. The floor is the coverage-mixture
    // variance (ComputeColorNeighborhoodStats): the unmodelled uniform
    // share at low step adequacy (cheap: admission failure means
    // dFB0^2 <= 4*lvlErr -- the admitted mixture is sub-perceptual), the
    // measured bracket at high adequacy (where stepTargetVar already
    // carries it -- the max is then a no-op). No-op on unstraddled pixels
    // (the floor is 0): the wake's eviction path is untouched.
    p.muVar = max(p.muVar, stats.covVarFloor);
    return p;
}

float WhitenedInnovSq(float3 innovCorr, RecordVarianceParts p)
{
    float3 invE = float3(1.0, 1.0, 1.0)
                / max(p.ePer, float3(1e-10, 1e-10, 1e-10));
    return dot(innovCorr * innovCorr, invE);
}

// ============================================================================
// THE EXACT RECORD SEED
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

    bool exactSeed = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0) && (taaClipScopedMu > 0.001);
    if (!exactSeed)
        // The own-layer scale (sigmaClean degenerates to sigma on
        // unstraddled pixels); the full-set sigma would seed the record at
        // the BETWEEN-layer scale on straddles -- the post-reset gate
        // then rides the arming cap for its whole EMA memory.
        return max(dot(stats.sigmaClean, stats.sigmaClean) + covFloorTotal, kClipSigmaRecordFloorSq);

    // The full quadratic form, per channel. The gradients and residual
    // are the same-layer values whenever the straddle ran -- the phase
    // energy of the own-layer gradient, not the step's.
    float3 phase = taaJitterPhaseEx  * (stats.gradCDx * stats.gradCDx)
                 + taaJitterPhaseEy  * (stats.gradCDy * stats.gradCDy)
                 + 2.0 * taaJitterPhaseExy * (stats.gradCDx * stats.gradCDy);
    float3 noise = stats.residSq;
    float3 varX  = phase + noise + stats.covVarFloor;
    float  sTotal = varX.x + varX.y + varX.z + 3.0 * kResampleVarSq;
    return max(sTotal, kClipSigmaRecordFloorSq);
}

// ============================================================================
// THE MAHALANOBIS GATE
// ============================================================================
struct ClipGateResult
{
    float3 clippedColorSpace;
    float  tGate;
    float3 rawGateVar;
    float  mdd;                   // the realized Mahalanobis statistic
                                  // (mode-13 telemetry: the realized-null-law
                                  // histogram; the empirical Student factor
                                  // is its 95th percentile / chi^2)
    float  p1;                   // the ghost posterior (the blend floor's
                                 // input), exported from the CONFIRMED
                                 // persistence evidence.
};

ClipGateResult ClipHistoryToStatisticGate(
    float3 historyColorSpace,
    ColorNeighborhoodStats stats,
    RecordVarianceParts parts,
    bool   recordLive,
    float  recordAge,
    float  statAlpha,
    float  motionNormalized,
    float3 neighborhoodTaps[9],
    float  ghostEvNats,
    float  nuHonest)
{
    ClipGateResult r;

    // The sequential evidence is the persistence detector's CONFIRMED
    // statistic (either channel), mapped through the exact tau = sigma
    // Gaussian-prior Bayes factor (ln BF = T^2/4 - 0.5*ln 2). ZERO below
    // the confirmation threshold: on ~99.9% of pixels the gate runs as
    // the pure clip and the detector contributes NOTHING to its null
    // behavior. No age/multiplicity discount: the EMA's memory is bounded
    // (~13 frames), so the unknown-onset prior mass is bounded by
    // construction.
    float seqFloor = 0.0;
    if (taaSoftClip > 0.001 && ghostEvNats > 0.0)
    {
        float logOddsSeq = kGhostLogOddsBase + ghostEvNats;
        seqFloor = 1.0 / (1.0 + exp(-logOddsSeq));
    }
    r.p1 = seqFloor;

    float3 sigA = AnisoClampSigma(stats.sigma);
    float  minGateVar = kMinSigma * kMinSigma;

    float3 gateVar;
    if (recordLive)
    {
        // Studentize the DOMINANT share. The mu share is ~96% of the flat
        // gate's width (the neighborhood-mean estimation error);
        // inflating only the ~4% varH share would leave the realized null
        // law ~3.4x hotter than the coded coverage (mdd ~ 3*F(3, nu_s),
        // nu_s ~ 6 on flats, so the coded chi^2 = 7.84 reads as ~17%
        // instead of 5%). The form: Satterthwaite over the two INDEPENDENT
        // components -- the transported record (dof: the record's
        // information clock) and the spatial mean/target estimate (dof:
        // the resid-fit count n_eff - 3 on flats -- the scoped estimator's
        // own dof correction, see nuMuFlat below; the step levels'
        // estimation dof on engaged steps) -- then the host's exact
        // F(3,nu) fit at the blended dof. Both anchors of the host fit
        // (nu0 = 4, nu_inf ~ 74) bracket the typical range; the flat mu
        // path can land at ~2.6 (below nu0), where the rational form
        // UNDER-inflates relative to the exact S (fit(2.6) ~ 3.95 vs
        // exact ~4.7) -- tight-side only, never wide. stats.invNeff /
        // residSq / the grads are the same-layer values whenever the
        // straddle ran -- the count, the residual and the mu share
        // describe the SAME estimator that produced the target.
        float  muRecordShare = (1.0 - stats.stepW) * (1.0 - saturate(taaClipScopedMu));
        float3 varHTerm = parts.varHPer
                        + float3(kResampleVarSq, kResampleVarSq, kResampleVarSq)
                        + muRecordShare * parts.muVar;
        float3 muTerm   = (1.0 - muRecordShare) * parts.muVar;
        float  nuVarH   = StudentEffectiveDof(recordAge, stats.recordSampleDof);
        // The flat mu share's dof is the RESID-FIT count -- the scoped
        // base (residSq) removes an explicitly fitted plane (mean + 2
        // gradients; the 1/(1-3*invNeff) dofCorr in
        // ComputeColorNeighborhoodStats is its own unbiasing), so the
        // t-like ratio carries n_eff - 3 dof: ~2.6 at the default
        // jitter-centered weights (invNeff ~ 0.18), ~4.1 at plain table
        // weights. The unscoped end (clipScopedMu -> 0) moves muVar into
        // varHTerm via muRecordShare, where the record's clock already
        // prices it -- muTerm vanishes there, so this dof only scales the
        // scoped blend.
        // CALIBRATION (the mode-13 procedure): if a static scene's 95th
        // percentile lands DIMMER than the nominal chi^2/16, the resid
        // model over-discounts (the max() floorings and the record
        // carryover absorb more noise than the plane-fit dof models) --
        // the alternative count floor
        //     max(1.0 / max(stats.invNeff, 0.2) - 1.0, 2.0)
        // is the dial.
        float  nuMuFlat = max(1.0 / max(stats.invNeff, 1e-3) - 3.0, 2.0);
        float  nuMu     = lerp(nuMuFlat, max(stats.cusumStepDof, 2.0), saturate(stats.stepW));
        float3 total    = varHTerm + muTerm;
        float3 nuEff    = total * total
                        / max(varHTerm * varHTerm / nuVarH + muTerm * muTerm / nuMu,
                              float3(1e-12, 1e-12, 1e-12));
        nuEff = max(nuEff, float3(2.0, 2.0, 2.0));
        float3 studentS = float3(1.0, 1.0, 1.0)
                        + float3(taaStudentA, taaStudentA, taaStudentA) / nuEff
                        + float3(taaStudentB, taaStudentB, taaStudentB) / (nuEff * nuEff);
        gateVar = total * studentS;
    }
    else
    {
        if (stats.stepW > 0.5)
        {
            // The cold step path is Studentized at the levels' own
            // estimation dof (a ~cusumStepDof estimate; S() at that dof
            // restores the coded coverage).
            float studentStep = StudentFactor(max(stats.cusumStepDof, 2.0));
            gateVar = max(studentStep * stats.stepTargetVar,
                          float3(minGateVar, minGateVar, minGateVar));
        }
        else
        {
            // Deliberately NOT Studentized: this is the cold-start PRIOR,
            // not a coverage claim -- after a reset/spike the policy is to
            // pull the history hard toward the live neighborhood and
            // re-establish trust (the blend floor / current-frame
            // replacement carries the reveal side); honest coverage lives
            // on the record-live path above. The exactCold gradients /
            // residual and this fallback's sigma are the same-layer values
            // whenever the straddle ran -- the cold pull is toward the
            // pixel's own layer -- and the coverage-mixture variance is
            // added unscaled by invNeff (it is the VALUE's own
            // uncertainty, not a mean-estimation error): the acceptance
            // around the own-layer target must still cover the mixtures.
            bool exactCold = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0) && (taaClipScopedMu > 0.001);
            float3 varX = exactCold
                ? taaJitterPhaseEx  * (stats.gradCDx * stats.gradCDx)
                + taaJitterPhaseEy  * (stats.gradCDy * stats.gradCDy)
                + 2.0 * taaJitterPhaseExy * (stats.gradCDx * stats.gradCDy)
                + stats.residSq
                : stats.sigmaClean * stats.sigmaClean;
            gateVar = stats.invNeff * varX + stats.covVarFloor;
        }
    }
    // Coherence note (deliberate): the transport scatter is added AFTER
    // the Studentization -- it is not S(nu)-inflated. Its static floor is
    // a stable min-statistic (the measured velocity-quantization scale),
    // but the motion-scaled share is a single noisy measurement; mode 13
    // UNDER MOTION is the view that would expose any under-coverage here.
    gateVar += (1.0 - stats.stepW) * stats.transportVar;   // motion-gated
    r.rawGateVar = gateVar;

    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    float3 dCorr = historyColorSpace - stats.mean + stats.gatePhaseShift;

    // FULL METRIC (the sample correlations, clamped). The correlations
    // are SHRUNK toward zero BEFORE the conditioning clamp: at the full
    // set's effective count (~7) a sampled correlation carries ~0.4
    // stderr -- an inflated one narrows the ellipsoid along its direction
    // (spurious clips -- flicker), a deflated one widens it (ghost
    // shelter). The shrink strips the leading-order estimation-noise
    // bias; calibrate the factor from the mode-13 static-scene histogram
    // if it ever needs to be tighter. The shrink's count is the honest
    // n_eff - 1 (covCross are mean-centered raw moments with NO plane
    // removed). covCross is the same-layer cross-covariance and the
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
        // The evidence and the action are separate objects. The
        // posterior's alternative is the unified 10-component
        // construction -- the 9 taps SHARP at the gate's own variance,
        // plus one low-weight temporal tail with the exact normalizer
        // Jacobian. The form keeps a tap-matching ghost strong evidence,
        // bounds the tail's H0 footprint by its prior weight, and charges
        // the exact Jacobian.
        float3 gW = max(gateVar * (chromaScale * chromaScale),
                        float3(minGateVar, minGateVar, minGateVar));
        // Coherence note (deliberate): this posterior runs at nuHonest --
        // the RECORD's dof (up to ~66) -- while the hard gate's radius
        // above used the blended Satterthwaite dof (~2.6-4 on flats).
        // Both the null and the components here are t_nuHonest in this
        // S-inflated gW, so the pair is self-consistent as "Gaussian at
        // the record-inflated scale"; the asymmetry (the soft action
        // seeing finer tails than the hard floor beneath it) is intended
        // -- the hard floor carries the coverage guarantee, the posterior
        // only ever pulls FURTHER toward the current frame than that
        // floor requires (wPull is max'd against it below).
        float3 sigAC = AnisoClampSigma(stats.sigmaClean);
        float  tailW = kCusumTailWeight;
        float3 tailVar = gW + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invCompVar = float3(1.0, 1.0, 1.0) / gW;
        float3 invTailVar = float3(1.0, 1.0, 1.0) / tailVar;

        float nu = max(nuHonest, 2.0);
        float kT = 0.5 * (nu + 3.0);

        // The null's NLL in the component whitening (the diagonal
        // approximation of mdd -- the mixture is a proper density either
        // way and the per-frame Gibbs property is unaffected).
        float q0Comp = dot(dCorr * dCorr, invCompVar);
        float qTail  = dot(dCorr * dCorr, invTailVar);
        float lTailNorm = 0.5 * (log(tailVar.x / gW.x) + log(tailVar.y / gW.y) + log(tailVar.z / gW.z));

        // The tap log-weights in closed form: the jitter-centered weights
        // are exp2(-d^2 * log2(e)), so their natural log is exactly -d^2
        // (requires kLog2E == log2(e), taaConstants; the 1e-4 clamp never
        // binds: d^2 <= 4.5 -> w >= 0.011). The table-weight path's logs
        // are compile-time constants.
        bool jitW = (taaJitterAwareVariance > 0.5);
        float  lj[9];
        float  lMax = -1e30;
        float  logTapW = log(1.0 - tailW);
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float3 dd = dCorr - (neighborhoodTaps[j] - stats.mean + stats.gatePhaseShift);
            float  qj = dot(dd * dd, invCompVar);
            float2 dj = kOffsets3x3[j] - stats.jitterPx;
            float  lw = jitW ? (-dot(dj, dj)) : log(max(stats.mixWeights[j], 1e-4));
            lj[j] = -kT * log(1.0 + qj / nu) + lw + logTapW;
            lMax = max(lMax, lj[j]);
        }
        float lTail = log(tailW) - kT * log(1.0 + qTail / nu) - lTailNorm;
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 9; ++j2)
            wSum += exp(lj[j2] - lMax);
        wSum += exp(lTail - lMax);

        // ln[mixture] - ln[t_0]; the null's NLL from the diagonal quadratic.
        float llrFrame = (lMax + log(max(wSum, 1e-10))) + kT * log(1.0 + q0Comp / nu);

        // The frame evidence always adds: the persistence statistics are
        // computed from the drift estimator's common modes, DIFFERENT
        // objects than this frame-LLR on dCorr; there is no double count.
        float logOdds = kGhostLogOddsBase
                      + log(1.0 + kGhostMotionOdds * saturate(motionNormalized))
                      + ghostEvNats
                      + llrFrame;
        float p1 = 1.0 / (1.0 + exp(-logOdds));
        r.p1 = max(p1, seqFloor);   // the blend floor export

        // The scalar pull -- the metric-consistent posterior action,
        // floored by the hard ray clamp.
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
    // in units of this texel's own content scale", not the between-layer
    // scale a straddle would charge.
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