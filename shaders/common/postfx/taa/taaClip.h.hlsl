// ============================================================================
// TAA history clipping: color statistics (mixture + same-depth-layer clean
// + the coverage step model), the temporal clip-state transport (sigma +
// VarH + age + the matched LLR pair), the Mahalanobis gate,
// and the feedback consumers
// ----------------------------------------------------------------------------
// THE CLIP: the accumulator is the ESTIMATOR, h_{t+1} = (1-a)h_t + a x_t;
// the clip is a robust innovation gate on the STATISTIC:
//     Var(h) = (a/2) E[i^2],   E[i^2] = 2/(2-a) Var(x)   (iid limit)
// Two clocks: the record's INFORMATION clock (age -> Student dof) and the
// accumulator's STATE clock (VarH, transported, damped-resample recursion).
//
// THE MATCHED LLR PAIR (the sequential detector): the ghost's per-frame
// signature on an engaged step is a PERSISTENT offset of the BIT-CONDITIONED
// innovation; the discrimination is TEMPORAL. The increment is the exact
// per-side log-likelihood ratio, Student-t in the whitening V:
//     l_s = (nu+3)/2 [ ln(1 + q0^2/nu) - ln(1 + q_s^2/nu) ]
// The PAIR (each side its own max(0, .)) keeps E[l | H0] < 0 per side
// (Gibbs). v3.8 added INTERIOR COVERAGE: each side's alternative is a
// two-component mixture {the endpoint, the half-coverage midpoint} --
// a swept edge leaves the history stuck at intermediate coverage, which
// the two pure levels cannot see.
//
// THE FALLBACK (flats): the exact alternative is the EMPIRICAL SPATIAL
// MARGINAL -- the nine taps are literally draws from the ghost content's
// distribution -- plus (v3.8.2) ONE low-weight TEMPORAL TAIL component:
//     f1 = (1-w) sum_j w_j t(z; mean - x_j, V) + w t(z; 0, V + T^2)
// covering the ghost whose old content is ABSENT from the sample (the
// wide reveal). v3.8 fixed the SIGN of the mixture term (the pre-v3.8
// form was a magnitude detector that pinned the walk under H0).
//
// THE SOFT CLIP (the posterior-mean action): the alternative is the SAME
// 10-component construction (v3.8.4: unified with the CUSUM -- the 9 taps
// at the statistic's own variance, sharp, plus the one tail component
// with the exact normalizer Jacobian), and the prior odds are the base +
// motion + the DISCOUNTED transported LLR + this frame's evidence.
//
// v3.8.4 (FLAG 2): THE CHANGE-POINT MULTIPLICITY DISCOUNT. The transported
// walk is evidence for "a ghost began at SOME frame in its life"; the
// honest Bayes factor for an unknown onset time carries -ln(#candidate
// onsets). The record's age upper-bounds the walk's life (record resets
// drain the walk), so ln(age) is the conservative discount. It is applied
// at EVERY posterior consumer -- the stats' migration input, the gate's
// prior, the sequential floor, and the hard alarm threshold (in taa.fx)
// -- so the alarm's "50% posterior crossing" identity holds by
// construction: both sides of the crossing carry the same discount.
// kCusumLlrClamp rises to 10 so the worst-case age-15 alarm (6.2383 +
// ln 15 = 8.95) remains reachable.
//
// v3.9 (statistical audit): (1) the gate's mu share is Studentized at its
// OWN estimation dof -- Satterthwaite over the transported record and the
// spatial mean/target estimate (the old form inflated only the ~4% varH
// share while the ~96% mu share ran at nominal: the realized flat null law
// ran ~3.4x hotter than the coded coverage, P(F(3,6) > chi^2/3) ~ 17% at
// chi 2.8 instead of 5%); (2) the full metric's sampled correlations are
// SHRUNK toward zero before the conditioning clamp (a ~7-tap correlation
// carries ~0.4 stderr); (3) the VarH chain runs READ semantics with the
// implied Var(x) input and the two host-measured kernel gains
// (taaVarhResampleLoss = 1 - the fixed-point read gain, taaVarhWhiteLoss =
// 1 - the white-input read gain); (4) the transport scatter's static floor
// is the measured velocity-quantization scale (passed in by the resolve);
// (5) the drift event guard is a per-channel drift-vs-step posterior and
// the soft threshold sits at 2 estimator sigmas (inside the Studentized
// gate -- the smooth-lighting band).
//
// TRANSPORT (tag 101): [31] sign, [30:28] tag, [27:22] sigma (6-bit),
// [21:16] acutance (6-bit), [15:10] VarH ratio (6-bit), [9:6] age,
// [5:0] the SIGNED LLR code (0.5-nat quanta, code = round(2L) + 32).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host
// context: cbuffer perDraw (the clipping + feedback constants,
// taaStudentA/B, taaJitterPhaseEx/Ey/Exy, taaClipGhostReset,
// taaClipScopedMu, taaCusumFlipAcc, taaCusumNoiseAcc,
// taaVarhResampleLoss, taaVarhWhiteLoss, taaDriftCompensation,
// taaDriftMaxGain). Requires taaShared.h.hlsl, taaConstants, taaFrame.h.hlsl
// included before.
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
    float3 mean;                // FUSED: the (migrated) step target on
                                // engaged straddles; the drift fusion on
                                // the fallback; blended by stepW between
    float3 sigma;               // full set (mixture): smear rejection,
                                // clip-distance normalization
    float3 sigmaClean;          // same-layer subsample (the B level, raw)
    float  invNeff;             // FUSED (the fallback path)
    float  invNeffClean;
    float  invNeffF;            // the F-side effective count inverse (the
                                // step dof input; 1.0 when no off-layer
                                // mass -- the safe divide)
    float3 phaseShift;          // FUSED: the step form on engaged steps
                                // (the CD form otherwise -- v3.8.4: the
                                // innovation's phase correction is
                                // unified onto the estimator's LS
                                // gradient in taa.fx; this field remains
                                // the fallback and the step-path model)
    float3 gatePhaseShift;      // FUSED: 0 on engaged steps
    float3 gradCDx;             // full-set central differences
    float3 gradCDy;
    float3 gradCDxClean;        // same-layer (off-layer sides -> flat)
    float3 gradCDyClean;
    float3 meanClean;
    float3 phaseShiftClean;
    float3 gatePhaseShiftClean;
    float  centerWeight;        // w0'
    float  effRank;             // r in [1,3]: the per-draw dof
    float3 covCross;            // the taps' full-set cross-channel
                                // covariances (xy, xz, yz) -- the measured
                                // correlation SHAPE the gate's full metric
                                // consumes
    float3 residSq;             // full-set weighted-LS residual (dof-corr)
    float  rangeSq;             // full-set squared AABB range
    float  rangeSqClean;        // same-layer squared range
    bool   straddled;           // any off-layer tap (mask valid)
    // ---- the coverage step fit ----
    float  stepW;               // the adequacy blend [0,1]
    float3 stepTargetVar;       // the effective target variance (flip
                                // charge, varC, levels, migration, and the
                                // mode-blend cross term included)
    float3 stepTargetStable;    // B0 + cStable*dFB (the ingest / phase-bit
                                // reference; the STABLE coverage, never the
                                // migrated one)
    float3 stepDFB;             // the corrected step (F0 - B0)
    float  stepC;               // the effective (migrated) coverage
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
                                // content gradient, MOTION-GATED (v3.8.4 +
                                // v3.9: the static charge is the measured
                                // velocity-quantization scale)
    float  cusumStepDof;        // the levels' estimation dof, nB + nF - 4
    float  mixWeights[9];       // the plain kernel weights (the ghost
                                // mixture's spatial prior)
    float  recordSampleDof;     // the record estimator's per-frame dof: the
                                // same-layer count over the FULL 3x3
                                // (v3.8.4: 9-tap; feeds the Student law)
};

// ============================================================================
// CLIP CONSTANTS
// ============================================================================
// THE record's bias-variance dial (named, not derived): a variance
// CHANGE-POINT is unmodelled, so the EMA trades record lag (ghost shelter
// when Var drops, flicker pressure when it rises) against estimate noise.
// rho = 0.15 is the chosen operating point. The statistically complete
// upgrade would be a second CUSUM on innovSq feeding the rate --
// deliberately not built (the transported dof + VarH clocks already absorb
// most of the lag cost).
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
// anti-flicker there); a 0.1% target needs r ~= 7.4. 4.0 keeps the spike's
// content-change latency low; retune ONLY against the mortality number,
// never by eye.
static const float kClipSpikeRatio           = 4.0;
static const float kClipWinsorC              = 3.0;
static const float kGeoCorroborateChiSq      = 7.815;         // chi^2_3(0.95), whitened
static const float kClipRejectionFeedbackFloor = 0.5;
static const float kAnisoSigmaCap            = 0.10;
static const float kScopedResidFrac          = 0.10;
static const float kSharpEwmaRate            = 0.25;
static const float kClipStudentPriorDof      = 4.0;
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 15.0;          // 4-bit age
// The record-vs-prior arming cap. The honest upper bound for a nu-dof
// variance estimate against its prior is chi^2_nu(0.999)/nu (~3.7 at the
// flats' nu ~ 6); 5.0 adds EMA-lag headroom for rising variance. Under
// honest dynamics it should rarely bind -- verify with the mode-13 null-law
// view + the record telemetry; if it binds measurably, the winsor ingest
// (9x prior) is the dial that actually needs attention.
static const float kArmingCap                = 5.0;
// THE KERNEL AUDIT (v3.2): the resample's variance effect is carried by the
// multiplicative damping in the VarH recursion (taaVarhResampleLoss /
// taaVarhWhiteLoss, v3.9); the additive slot stays 0.
static const float kResampleVarSq            = 0.0;
// ---- the posterior-mean soft clip ----
static const float kGhostLogOddsBase         = -6.2383;       // ln(1/512)
// The motion term of the ghost prior, DERIVED as an order-of-magnitude
// population ratio (the one population number without a measurement --
// calibratable from mode-11 telemetry): at an ENGAGED swept step the
// stale-landing probability is the off-coverage ~1/8; on flats it is the
// base 1/512. The ratio 64 = (1/8)/(1/512) is the prior-odds boost at full
// motion, and it only matters where the frame evidence is ambiguous (the
// frame LLR overwhelms it on clean nulls and clean ghosts alike).
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
// ---- the matched LLR pair ----
// The alarm is the posterior crossing 50% at the 1/512 base prior
// (ln 512 = 6.2383, v3.9: the exact value) WITH the age discount carried
// on both sides (v3.8.4). The soft onset is ~posterior 1.4% (e^2/512).
// kCusumLlrClamp = 10 leaves headroom above the worst-case age-discounted
// alarm (6.2383 + ln 15 = 8.95).
static const float kCusumSoftLlr   = 2.0;    // nats: soft migration onset (~1.4% posterior)
static const float kCusumAlarmLlr  = 6.2383;  // nats: ln(512) -- the EXACT 50%-crossing
static const float kCusumLlrClamp  = 10.0;   // nats: the saturation
// The interior (half-coverage) components' admission gate, in whitened
// variance units: a component must be resolvable (offset^2 >= 2V) -- a
// model-selection threshold.
static const float kCusumInteriorMinSq = 2.0;
// The fallback CUSUM's temporal-tail component's prior mass: one
// component's worth of the mixture (9 taps + 1 tail, 1/10 each).
static const float kCusumTailWeight   = 0.1;
// ---- the drift predict step ----
static const float kDriftCapSigmas  = 3.0;   // the magnitude BELT (content sigmas)
// v3.9 (audit §4): 3.0 -> 2.0. The soft threshold's position relative to
// the (now Studentized) gate radius sets the smooth-lighting band: with the
// gate's mu share carrying S(nu ~ 6) the flat gate sits at ~1.4
// sigma_content, and the drift estimator's own noise (weighted mean +
// phase-correction error + transport scatter ~ 0.5 sigma_content/channel)
// put the 3-sigma onset at ~1.5 -- ABOVE the gate, i.e. the tracker was
// redundant exactly where it was meant to be the smooth alternative. 2.0
// engages it at ~1.0 sigma_content, inside the gate, with the noise cost
// still bounded by the SNR gain and taaDriftMaxGain.
static const float kDriftThreshSigmas = 2.0; // the soft threshold (estimator sigmas)
// v3.9 (audit §4): the lighting-rate prior -- the per-frame drift scale as
// a fraction of the content scale. Prices H_drift in the event guard's
// posterior (see the drift block in taa.fx): smooth lighting moves at most
// ~half the content scale per frame; larger common-mode offsets are
// step-like and the reveal paths own them. The posterior's crossover sits
// near sigma_clean regardless of this value (bounded by the wider
// hypothesis); the dial shapes how FAST the stand-down happens.
static const float kDriftPriorScaleSigmas = 0.5;
// ---- v3.8.4 (FLAG 1) + v3.9 (audit §5): the transport scatter's motion
// gate. The static floor is MEASURED per pixel (the velocity buffer's own
// quantization/noise scale, passed in by the resolve; was the 0.35
// multiplier constant). kTransportGatePx is the motion scale over which
// the jitter-residual proxy ramps to full charge (the landing error grows
// with the reprojection error, a fraction of the velocity).
static const float kTransportGatePx      = 2.0;

// v3.8.4 (FLAG 5): the jitter sequence's full 2x2 phase covariance as a
// quadratic form -- the variance of the jitter's projection on a unit
// direction. The sequence covariance is PSD by construction, so Q >= 0
// for every direction (saturate guards float error only).
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

    float3 mid = 0.5 * (B0 + F0);
    float  sep = 0.0;
    [unroll]
    for (int i = 0; i < 9; ++i)
        sep += sign(dot(taps[i] - mid, dFB)) * dot(kOffsets3x3[i], gDir);
    if (sep < 0.0) gDir = -gDir;
    f.gDir = gDir;

    float mObs = 0.0;
    [unroll]
    for (int i2 = 0; i2 < 9; ++i2)
        mObs += kStdWeights[i2] * clamp(dot(taps[i2] - B0, dFB) / max(dFBsq, 1e-10), 0.0, 1.0);
    mObs = saturate(mObs);

    // THE TAIL-MASS STAIRCASE INVERSION (de-phased by u_g; interval-mid
    // target; the widened bracket charges the variance).
    float uG = dot(u, gDir);
    float qMax = -1e9;
    [unroll]
    for (int jq = 0; jq < 9; ++jq)
        qMax = max(qMax, dot(kOffsets3x3[jq], gDir));

    float qMid = qMax + 0.5, qA = qMax, qB2 = qMax + 1.0;
    bool  crossed = false;
    float runMassHit = 1.0;
    [unroll]
    for (int k = 0; k < 9; ++k)
    {
        float qk = dot(kOffsets3x3[k], gDir);
        float massAbove = 0.0, massRun = 0.0;
        float qAbove = qk; bool anyAbove = false;
        float qBelow = qk; bool anyBelow = false;
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float qj = dot(kOffsets3x3[j], gDir);
            if      (qj > qk + 1e-4) { massAbove += kStdWeights[j]; if (!anyAbove || qj > qAbove) { qAbove = qj; anyAbove = true; } }
            else if (qj > qk - 1e-4) { massRun   += kStdWeights[j]; }
            else                     {                                       if (!anyBelow || qj > qBelow) { qBelow = qj; anyBelow = true; } }
        }
        if (!crossed && massAbove < mObs && mObs <= massAbove + massRun)
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

    // v3.8.4 (FLAG 5): the jitter span is DIRECTION-AWARE -- the sequence's
    // variance along the edge normal is the quadratic form Q(gDir), not the
    // axis average.
    float jitterSpan = max(sqrt(saturate(12.0 * JitterProjVar(gDir))), kMinJitterSpan);
    float cT  = SquareCoverage(gDir, (qMid - uG) / jitterSpan);
    float cLo = SquareCoverage(gDir, (qA - uG) / jitterSpan);
    float cHi = SquareCoverage(gDir, (qB2 - uG) / jitterSpan);
    if (cLo < cHi) { float tmp = cLo; cLo = cHi; cHi = tmp; }

    f.cSlope = runMassHit / max(abs(cLo - cHi), 0.05);

    float proj0 = clamp(dot(taps[0] - B0, dFB) / max(dFBsq, 1e-10), 0.0, 1.0);

    f.cTarget = cT; f.cLo = cLo; f.cHi = cHi;
    f.xModel  = B0 + proj0 * dFB;
    f.target  = B0 + cT * dFB;
    return f;
}

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized,
    float2 jitterPx, float2 jitterResidualPx, float edgeSweepPx,
    bool sameLayerMask[9], bool layerMaskValid,
    float driftLlr, bool driftSoftActive,
    float transportFloorPx)
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

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
    float3 weightedCovXSum   = float3(0.0, 0.0, 0.0);
    float3 weightedCovYSum   = float3(0.0, 0.0, 0.0);
    float  wSumXY = 0.0, wSumXZ = 0.0, wSumYZ = 0.0;
    float  centerWeightRaw  = 0.0;

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);
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
        float w;
        if (jitterCenteredWeights)
            w = exp2(-dot(offsetFromCenterPx, offsetFromCenterPx) * kLog2E);
        else
            w = kStdWeights[i];

        stats.mixWeights[i] = w;

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

    // v3.8.4 (FLAG 1) + v3.9 (audit §5): the transport scatter, MOTION-GATED
    // with a MEASURED static floor. The raw proxy (jitterResidualPx^2 * grad^2)
    // charges the jitter-phase scale even on static scenes, where the phase is
    // already removed from the tested statistic by the phase correction and
    // charged into the record by phaseSq -- a ~3x double count that widened the
    // static gate above the nominal coverage (anti-flicker, but a ghost shelter
    // on static content, and an inflated drift threshold). The landing-error
    // scale the proxy honestly represents grows with motion (the reprojection
    // error is a fraction of the velocity); the STATIC floor is the velocity
    // buffer's own measured quantization/noise scale (mainP's lazy
    // MeasureVelocityQuantStepPx -- was the 0.35 multiplier constant, which
    // charged a fraction of the jitter phase the tested statistic no longer
    // carries). All consumers (the gate, the CUSUM's V, the drift variance)
    // read the gated value below.
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

    // v3.8.4: the record estimator's per-frame dof over the FULL 3x3
    // (the estimator now consumes all nine taps; the typical effective
    // count of the kernel-weighted 3x3 is ~7 -> dof ~6, which the host's
    // Student-fit anchor tracks).
    {
        float dofWSum = 0.0, dofWSq = 0.0;
        float dofUsed = 0.0;
        [unroll]
        for (int tDof = 0; tDof < 9; ++tDof)
        {
            if (tDof == 0 || sameLayerMask[tDof])
            {
                dofWSum += stats.mixWeights[tDof];
                dofWSq  += stats.mixWeights[tDof] * stats.mixWeights[tDof];
                dofUsed += 1.0;
            }
        }
        if (layerMaskValid && dofUsed >= 3.0 && dofWSum > 1e-6)
        {
            float invNeffSub = dofWSq / (dofWSum * dofWSum);
            stats.recordSampleDof = max(1.0 / max(invNeffSub, 0.05) - 1.0, 1.0);
        }
        else
        {
            stats.recordSampleDof = stats.effRank;
        }
    }

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

    if (layerMaskValid)
    {
        float3 sumC = float3(0.0, 0.0, 0.0);
        float3 ssqC = float3(0.0, 0.0, 0.0);
        float2 offC = float2(0.0, 0.0);
        float  wC = 0.0, wC2 = 0.0;
        float  offXSqC = 0.0, offYSqC = 0.0;
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

                if (fit.w > 0.0)
                {
                    // THE DRIFT MIGRATION: the LLR's live side moves the
                    // coverage toward the sample side (the input is the
                    // age-discounted walk -- v3.8.4).
                    float wDrift = driftSoftActive
                        ? saturate((abs(driftLlr) - kCusumSoftLlr) / (kCusumAlarmLlr - kCusumSoftLlr))
                        : 0.0;
                    float cEff = (driftLlr < 0.0)
                        ? fit.cTarget * (1.0 - wDrift)
                        : fit.cTarget + wDrift * (1.0 - fit.cTarget);
                    float loEff = lerp(min(fit.cLo, fit.cHi), cEff, wDrift);
                    float hiEff = lerp(max(fit.cLo, fit.cHi), cEff, wDrift);
                    float flip  = MaxBinomial(loEff, hiEff);
                    float varC  = (hiEff - loEff) * (hiEff - loEff) * (1.0 / 12.0);

                    float jrN = dot(jitterResidualPx, fit.gDir)
                              / max(sqrt(saturate(12.0 * JitterProjVar(fit.gDir))), kMinJitterSpan);
                    stats.stepPhaseVarScale = jrN * jrN;

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

    // ---- THE FALLBACK fusion: mixture -> clean by the (discounted) drift
    //      weight ----------------------------------------------------------
    {
        float wMig = driftSoftActive
            ? saturate((abs(driftLlr) - kCusumSoftLlr) / (kCusumAlarmLlr - kCusumSoftLlr))
            : 0.0;
        if (stats.stepW < 1.0)
        {
            float wFb = (1.0 - stats.stepW) * wMig;
            stats.mean           = lerp(stats.mean, stats.meanClean, wFb);
            stats.phaseShift     = lerp(stats.phaseShift, stats.phaseShiftClean, wFb);
            stats.gatePhaseShift = lerp(stats.gatePhaseShift, stats.gatePhaseShiftClean, wFb);
            stats.invNeff        = lerp(stats.invNeff, stats.invNeffClean, wFb);
        }
    }
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
// CLIP-STATE TRANSPORT (format v3.8, tag 101)
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
                         float acutanceLinear, float llrCode)
{
    uint u = 0x50000000u
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x3Fu)        << 22)
           | ((uint(saturate(acutanceLinear) * 63.0 + 0.5) & 0x3Fu)  << 16)
           | ((uint(varHRatioCode) & 0x3Fu)                          << 10)
           | ((uint(clamp(age, 0.0, kClipMaxAge)) & 0xFu)            << 6)
           |  (uint(clamp(llrCode, 1.0, 63.0)) & 0x3Fu);
    return asfloat(revoked ? (u | 0x80000000u) : u);
}

void DecodeClipState(float alphaValue, out float sigmaSq, out float varHSq,
                     out float recordAge, out float llrNats)
{
    sigmaSq = -1.0;
    varHSq = 0.0;
    recordAge = 0.0;
    llrNats = 0.0;

    uint u   = asuint(alphaValue);
    uint tag = (u >> 28) & 0x7u;
    uint code;
    float varHCode = 28.0;   // ratio ~0.25: the near-stationary default
    if      (tag == 0x5u)                     // clip-state alpha (v3.8)
    {
        code = (u >> 22) & 0x3Fu;
        varHCode = (float)((u >> 10) & 0x3Fu);
        recordAge = (float)((u >> 6) & 0xFu);
        llrNats = ((float)(u & 0x3Fu) - 32.0) * 0.5;
    }
    else if (tag == 0x1u)                     // debug payload: sigma only
    {
        code = u & 0x3Fu;
    }
    else
    {
        return;          // foreign alpha (incl. v3.2 tag 100 and older): cold
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
    float3 varHPer;     // per-channel accumulator variance (READ semantics,
                        // v3.9 -- what the gate and the CUSUM's step model
                        // consume directly)
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
    float3 sigA  = AnisoClampSigma(stats.sigma);
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
    p.ePer    = share * max(recordSq, kClipSigmaRecordFloorSq);
    p.spatialPriorSq = spatialPriorSq;

    if (stats.stepW > 0.5)
    {
        // v3.9: taaCusumNoiseAcc is the EXACT fixed-point accumulated-read
        // noise share (the host's kernelChainConstants series) -- the model
        // below is in READ semantics, consistent with the transported varH
        // and with what the gate tests.
        float  cS = stats.stepCStable;
        float3 residMix = (1.0 - cS) * stats.stepResidVarB + cS * stats.stepResidVarF;
        float3 modelVarH = taaCusumFlipAcc * (cS * (1.0 - cS)) * (stats.stepDFB * stats.stepDFB)
                         + taaCusumNoiseAcc * residMix;
        p.varHPer = share * max(modelVarH, float3(0.0, 0.0, 0.0));
    }
    else
    {
        p.varHPer = share * max(varHTotal, 0.0);
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
    return p;
}

float WhitenedInnovSq(float3 innovCorr, RecordVarianceParts p)
{
    float3 invE = float3(1.0, 1.0, 1.0)
                / max(p.ePer, float3(1e-10, 1e-10, 1e-10));
    return dot(innovCorr * innovCorr, invE);
}

// ============================================================================
// THE MATCHED LLR PAIR
// ============================================================================
struct DriftCusumState
{
    float llrNext;     // the signed LLR summary (nats)
    bool  alarm;       // RAW threshold cross (the age-discounted alarm is
                       // computed in taa.fx where the age lives)
};

DriftCusumState UpdateDriftCusum(
    float llrPrev,
    float3 innovRaw,
    float3 innovCorr,
    ColorNeighborhoodStats stats,
    float3 taps[9],
    RecordVarianceParts parts,
    float  nuFallback,
    bool   recordLive,
    bool   frozen)
{
    DriftCusumState d;
    d.alarm = false;

    if (frozen)
    {
        d.llrNext = llrPrev;
        d.alarm = (abs(llrPrev) >= kCusumAlarmLlr);
        return d;
    }

    if (stats.stepW > 0.5)
    {
        // ---- THE MATCHED STEP PAIR (+ interior coverage), conditioned on
        //      the measured bit ---------------------------------------------
        float3 dFB   = stats.stepDFB;
        float  cS    = stats.stepCStable;
        float3 mCond = stats.stepXModel - stats.stepTargetStable;
        float3 z     = innovRaw - mCond;

        float dFBsq = dot(dFB, dFB);
        float proj0 = saturate(cS + dot(mCond, dFB) / max(dFBsq, 1e-10));

        float3 residMix = (1.0 - cS) * stats.stepResidVarB + cS * stats.stepResidVarF;
        float3 residBit = (1.0 - proj0) * stats.stepResidVarB + proj0 * stats.stepResidVarF;
        float3 V = residBit
                 + taaCusumFlipAcc * (cS * (1.0 - cS)) * (dFB * dFB)
                 + stats.stepPhaseVarScale * (dFB * dFB)
                 + taaCusumNoiseAcc * residMix;
        float3 invV = float3(1.0, 1.0, 1.0) / max(V, float3(1e-12, 1e-12, 1e-12));

        float nu = max(stats.cusumStepDof, 2.0);
        float kT = 0.5 * (nu + 3.0);
        float invNu = 1.0 / nu;

        float  q0Sq   = dot(z * z, invV);
        float  d2w    = dot(dFB * dFB, invV);
        float3 zPlus  = z - cS * dFB;                 // H+: stuck at c_g = 0
        float3 zMinus = z + (1.0 - cS) * dFB;          // H-: stuck at c_g = 1
        float  qPSq   = dot(zPlus * zPlus, invV);
        float  qMSq   = dot(zMinus * zMinus, invV);

        // v3.8: the INTERIOR coverage midpoints, admitted per side when
        // their whitened offset clears kCusumInteriorMinSq.
        float3 zPlusMid  = z - 0.5 * cS * dFB;          // stuck at c_g = cS/2
        float3 zMinusMid = z + 0.5 * (1.0 - cS) * dFB;  // stuck at c_g = (1+cS)/2
        float  qPMSq = dot(zPlusMid * zPlusMid, invV);
        float  qMMSq = dot(zMinusMid * zMinusMid, invV);
        bool   plusMidOk  = (0.25 * cS * cS * d2w)                 >= kCusumInteriorMinSq;
        bool   minusMidOk = (0.25 * (1.0 - cS) * (1.0 - cS) * d2w) >= kCusumInteriorMinSq;

        float nllNull = kT * log(1.0 + q0Sq * invNu);
        float tP = pow(1.0 + qPSq * invNu, -kT);
        float tM = pow(1.0 + qMSq * invNu, -kT);
        float mixP = plusMidOk  ? 0.5 * (tP + pow(1.0 + qPMSq * invNu, -kT)) : tP;
        float mixM = minusMidOk ? 0.5 * (tM + pow(1.0 + qMMSq * invNu, -kT)) : tM;
        float lPlus  = nllNull + log(mixP);
        float lMinus = nllNull + log(mixM);

        float sPlus  = max(llrPrev, 0.0);
        float sMinus = max(-llrPrev, 0.0);
        sPlus  = min(max(sPlus  + lPlus,  0.0), kCusumLlrClamp);
        sMinus = min(max(sMinus + lMinus, 0.0), kCusumLlrClamp);
        d.llrNext = sPlus - sMinus;
    }
    else
    {
        // ---- THE EMPIRICAL SPATIAL MARGINAL + THE TEMPORAL TAIL ----------
        float3 sigAC = AnisoClampSigma(stats.sigmaClean);
        float3 V;
        float  nu;
        if (recordLive)
        {
            V  = max(parts.ePer + stats.transportVar, float3(1e-10, 1e-10, 1e-10));
            nu = max(nuFallback, 2.0);
        }
        else
        {
            V  = max(sigAC * sigAC + stats.transportVar, float3(1e-10, 1e-10, 1e-10));
            nu = max(1.0 / max(stats.invNeffClean, 0.2) - 2.0, 2.0);
        }
        float3 invV = float3(1.0, 1.0, 1.0) / V;
        float kT = 0.5 * (nu + 3.0);

        float q0Sq = dot(innovCorr * innovCorr, invV);

        // THE TEMPORAL TAIL: the tenth component -- the ghost whose old
        // content is absent from the sample. Same construction and same
        // bandwidth as the gate's posterior (v3.8.4: unified).
        float  tailW = kCusumTailWeight;
        float3 VTail = V + (kGhostPriorSigma * kGhostPriorSigma) * (sigAC * sigAC);
        float3 invVT = float3(1.0, 1.0, 1.0) / VTail;
        float  qTailSq = dot(innovCorr * innovCorr, invVT);
        float  lTailNorm = 0.5 * (log(VTail.x / V.x) + log(VTail.y / V.y) + log(VTail.z / V.z));

        float3 compOff[9];
        float  lj[9];
        float  lMax = -1e30;
        float  logTapW = log(1.0 - tailW);
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            compOff[j] = stats.mean - taps[j];
            float3 dd = innovCorr - compOff[j];
            float  qj = dot(dd * dd, invV);
            lj[j] = -kT * log(1.0 + qj / nu) + log(max(stats.mixWeights[j], 1e-4)) + logTapW;
            lMax = max(lMax, lj[j]);
        }
        float lTail = log(tailW) - kT * log(1.0 + qTailSq / nu) - lTailNorm;
        lMax = max(lMax, lTail);

        float wSum = 0.0;
        [unroll]
        for (int j2 = 0; j2 < 9; ++j2)
            wSum += exp(lj[j2] - lMax);
        wSum += exp(lTail - lMax);

        // l = NLL_null + ln[mixture] (the v3.8 sign fix preserved).
        float llrMix = kT * log(1.0 + q0Sq / nu) + (lMax + log(max(wSum, 1e-10)));

        float3 zw = innovCorr * sqrt(invV);
        float azx = abs(zw.x), azy = abs(zw.y), azz = abs(zw.z);
        float zDom = (azx >= azy && azx >= azz) ? zw.x : ((azy >= azz) ? zw.y : zw.z);
        float zSign = (zDom >= 0.0) ? 1.0 : -1.0;

        float wPrev = abs(llrPrev);
        float wNext = min(max(wPrev + llrMix, 0.0), kCusumLlrClamp);
        d.llrNext = zSign * wNext;
    }

    d.alarm = (abs(d.llrNext) >= kCusumAlarmLlr);
    return d;
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

    bool exactSeed = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0) && (taaClipScopedMu > 0.001);
    if (!exactSeed)
        return max(dot(stats.sigma, stats.sigma), kClipSigmaRecordFloorSq);

    // v3.8.4 (FLAG 5): the full quadratic form, per channel.
    float3 phase = taaJitterPhaseEx  * (stats.gradCDx * stats.gradCDx)
                 + taaJitterPhaseEy  * (stats.gradCDy * stats.gradCDy)
                 + 2.0 * taaJitterPhaseExy * (stats.gradCDx * stats.gradCDy);
    float3 noise = stats.residSq;
    float3 varX  = phase + noise;
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
    float  mdd;                   // v3.9: the realized Mahalanobis statistic
                                  // (mode-13 telemetry: the realized-null-law
                                  // histogram; the empirical Student factor
                                  // is its 95th percentile / chi^2)
    float  p1;                   // the ghost posterior (the blend floor's
                                 // input), exported from the DISCOUNTED
                                 // sequential evidence whenever it exceeds
                                 // the soft onset.
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
    float  llrAccum,
    float  nuHonest,
    bool   llrAccumIncludesFrame)
{
    ClipGateResult r;

    // v3.8.4 (FLAG 2): the change-point multiplicity discount. The record's
    // age upper-bounds the walk's life (record resets drain the walk), so
    // ln(age) is the conservative discount; applied here, in the migration
    // input, and in the hard alarm threshold -- the alarm's 50%-crossing
    // identity holds at every age by construction.
    float seqEvidence = abs(llrAccum) - log(max(recordAge, 1.0));

    // THE SEQUENTIAL BLEND FLOOR (no motion term: below the gate the only
    // evidence is the walk; the motion odds belong to the posterior path's
    // population).
    float seqFloor = 0.0;
    if (taaSoftClip > 0.001 && seqEvidence > kCusumSoftLlr)
    {
        float logOddsSeq = kGhostLogOddsBase + seqEvidence;
        seqFloor = 1.0 / (1.0 + exp(-logOddsSeq));
    }
    r.p1 = seqFloor;

    float3 sigA = AnisoClampSigma(stats.sigma);
    float  minGateVar = kMinSigma * kMinSigma;

    float3 gateVar;
    if (recordLive)
    {
        // v3.9 (audit §1): Studentize the DOMINANT share. The old form
        // inflated only the varH share (~4% of the flat gate's width at
        // default feedback) while the mu share (~96% -- the neighborhood-mean
        // estimation error) ran at nominal: the realized null law ran ~3.4x
        // hotter than the coded coverage (mdd ~ 3*F(3, nu_s), nu_s ~ 6 on
        // flats, so the coded chi^2 = 7.84 read as ~17% instead of 5%). The
        // fix: Satterthwaite over the two INDEPENDENT components -- the
        // transported record (dof: the record's information clock) and the
        // spatial mean/target estimate (dof: the full-set effective count on
        // flats, the step levels' estimation dof on engaged steps) -- then
        // the host's exact F(3,nu) fit at the blended dof. Both anchors of
        // the host fit (nu0 = 4, nu_inf ~ 74) bracket every blended dof this
        // can produce; the max(nu, 2) guards below nu0 are degenerate-subset
        // regimes only.
        float  muRecordShare = (1.0 - stats.stepW) * (1.0 - saturate(taaClipScopedMu));
        float3 varHTerm = parts.varHPer
                        + float3(kResampleVarSq, kResampleVarSq, kResampleVarSq)
                        + muRecordShare * parts.muVar;
        float3 muTerm   = (1.0 - muRecordShare) * parts.muVar;
        float  nuVarH   = StudentEffectiveDof(recordAge, stats.recordSampleDof);
        float  nuMuFlat = max(1.0 / max(stats.invNeff, 0.2) - 1.0, 2.0);
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
            // v3.9 (audit §1): the cold step path was un-Studentized. The
            // step's target variance is a ~cusumStepDof estimate (the levels'
            // own error model); S() at that dof restores the coded coverage.
            float studentStep = StudentFactor(max(stats.cusumStepDof, 2.0));
            gateVar = max(studentStep * stats.stepTargetVar,
                          float3(minGateVar, minGateVar, minGateVar));
        }
        else
        {
            // Deliberately NOT Studentized (v3.9): this is the cold-start
            // PRIOR, not a coverage claim -- after a reset/spike the policy
            // is to pull the history hard toward the live neighborhood and
            // re-establish trust (the blend floor / current-frame replacement
            // carries the reveal side); honest coverage lives on the
            // record-live path above.
            bool exactCold = ((taaJitterPhaseEx + taaJitterPhaseEy) > 0.0) && (taaClipScopedMu > 0.001);
            float3 varX = exactCold
                ? taaJitterPhaseEx  * (stats.gradCDx * stats.gradCDx)
                + taaJitterPhaseEy  * (stats.gradCDy * stats.gradCDy)
                + 2.0 * taaJitterPhaseExy * (stats.gradCDx * stats.gradCDy)
                + stats.residSq
                : stats.sigma * stats.sigma;
            gateVar = stats.invNeff * varX;
        }
    }
    gateVar += (1.0 - stats.stepW) * stats.transportVar;   // motion-gated (v3.8.4 + v3.9)
    r.rawGateVar = gateVar;

    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    float3 dCorr = historyColorSpace - stats.mean + stats.gatePhaseShift;

    // v3.7 FULL METRIC (the sample correlations, clamped). v3.9 (audit §2):
    // the correlations are SHRUNK toward zero BEFORE the conditioning clamp.
    // At the full set's effective count (~7) a sampled correlation carries
    // ~0.4 stderr: an inflated one narrows the ellipsoid along its direction
    // (spurious clips -- flicker), a deflated one widens it (ghost shelter).
    // The shrink strips the leading-order estimation-noise bias; calibrate
    // the factor from the mode-13 static-scene histogram if it ever needs
    // to be tighter.
    float rDof = max(1.0 / max(stats.invNeff, 0.25) - 1.0, 2.0);
    float corrShrink = max(0.0, 1.0 - 1.0 / rDof);
    float vX = max(gateVar.x * chromaScale.x * chromaScale.x, minGateVar);
    float vY = max(gateVar.y * chromaScale.y * chromaScale.y, minGateVar);
    float vZ = max(gateVar.z * chromaScale.z * chromaScale.z, minGateVar);
    float sX = sqrt(vX), sY = sqrt(vY), sZ = sqrt(vZ);
    float rXY = clamp((stats.covCross.x / max(stats.sigma.x * stats.sigma.y, kEpsilon)) * corrShrink, -0.95, 0.95);
    float rXZ = clamp((stats.covCross.y / max(stats.sigma.x * stats.sigma.z, kEpsilon)) * corrShrink, -0.95, 0.95);
    float rYZ = clamp((stats.covCross.z / max(stats.sigma.y * stats.sigma.z, kEpsilon)) * corrShrink, -0.95, 0.95);
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
    r.mdd = mdd;   // v3.9: exported for the mode-13 null-law telemetry

    float tHard = (mdd > 1e-20) ? min(chi / sqrt(mdd), 1.0) : 1.0;

    bool posteriorPath = (taaSoftClip > 0.001) && (mdd > chi * chi);
    if (posteriorPath)
    {
        // v3.5: the evidence and the action are separate objects.
        //
        // v3.8.4 (FLAG 4): the posterior's alternative is the SAME
        // 10-component construction as the CUSUM's fallback -- the 9 taps
        // SHARP at the gate's own variance, plus one low-weight temporal
        // tail with the exact normalizer Jacobian. The previous form
        // broadened every component by the tail bandwidth (a KDE) and
        // ignored the normalizer of the broadened variance -- the unified
        // form keeps a tap-matching ghost strong evidence, bounds the
        // tail's H0 footprint by its prior weight, and charges the exact
        // Jacobian, identically in both consumers.
        float3 gW = max(gateVar * (chromaScale * chromaScale),
                        float3(minGateVar, minGateVar, minGateVar));
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

        float  lj[9];
        float  lMax = -1e30;
        float  logTapW = log(1.0 - tailW);
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float3 dd = dCorr - (neighborhoodTaps[j] - stats.mean + stats.gatePhaseShift);
            float  qj = dot(dd * dd, invCompVar);
            lj[j] = -kT * log(1.0 + qj / nu) + log(max(stats.mixWeights[j], 1e-4)) + logTapW;
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

        float logOdds = kGhostLogOddsBase
                      + log(1.0 + kGhostMotionOdds * saturate(motionNormalized))
                      + seqEvidence
                      + (llrAccumIncludesFrame ? 0.0 : llrFrame);
        float p1 = 1.0 / (1.0 + exp(-logOdds));
        r.p1 = max(p1, seqFloor);   // the blend floor export

        // v3.7: the scalar pull -- the metric-consistent posterior action,
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
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / stats.sigma;
    float maxChannelDistance = max(clipDistance.x, max(clipDistance.y, clipDistance.z));
    return saturate((maxChannelDistance - taaClipDistanceRejectionMinError) * taaClipDistanceRejectionAmount);
}

// (Retained as the documented legacy/manual fallback for the drift predict
// step -- redundant while Drift Compensation is on and off by default.)
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