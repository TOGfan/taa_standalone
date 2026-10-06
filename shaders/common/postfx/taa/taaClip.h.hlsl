// ============================================================================
// TAA history clipping: color statistics, the temporal clip-state transport
// (sigma + age + fiction), the Mahalanobis statistic gate, and the
// feedback consumers
// ----------------------------------------------------------------------------
// THE CLIP (design, stated honestly): the accumulator is the ESTIMATOR,
// h_{t+1} = (1-a) h_t + a x_t, and the clip is a robust innovation gate on
// the STATISTIC -- not a plausibility test of a fresh sample. The gate is a
// coverage statement about the only two random objects in the system:
//
//     h   the accumulator:   Var(h)  = a/(2-a) Var(x)      (iid limit)
//     i_t = x_t - h_{t-1}:  E[i^2]  = 2/(2-a) Var(x)
//         =>  Var(h) = (a/2) E[i^2]                        -- EXACT
//     mu  the phase-mean estimate (the jitter-aware tap mean):
//         Var(mu) = invNeff Var(x),  invNeff = SumW2/SumW^2
//         -- measured live from the ACTUAL weight kernel (0.18 standard
//            weights, up to ~0.27 at a corner-phase kernel).
//
//     gate_var = cStat(t) * S(nu(t)) * E[i^2]   (live; see below)
//     gate_var = invNeff * sigma_c^2            (cold: the change-point
//                                                replacement, tight)
//
// THE INNOVATION, SPLIT BY CONSUMER (landings between pixels are
// first-class): after translating out the (dejittered) content
// displacement, the innovation compares a POINT SAMPLE of the raw field,
// at sub-texel offset u = -fracPx from the stable point, with a KAISER
// RECONSTRUCTION of the accumulator field at the landing fraction phi_h.
// The accumulator field is PHASE-FREE (each stored texel is the
// jitter-centered mean at its position), and the innovation splits into
// exactly three bits:
//   * PHASE beta*u: the point sample's phase (the accumulator has none).
//     Jitter-correlated; energy = the sub-texel room. CONSUMER: subtracted
//     for every mean-structure test (the anti-alignment); KEPT in the raw
//     second moment (the record -- the accumulator carries the full phase
//     oscillation, so the variance path must measure it).
//   * RESAMPLE -R(phi_h): the Kaiser's error reconstructing the AA field
//     at the landing fraction. Zero at texel centers; second order on
//     coverage ramps (the stored history near an edge IS the ramp -- a
//     between-pixels landing resamples the RAMP, not the two layers);
//     first order only for sub-texel edges, bounded by the kernel's
//     across-edge leakage (~10-20% of the gap). CONSUMER: the record, as
//     the resampler's footprint variance.
//   * DISCREPANCY: everything persistent (content change, staleness,
//     wrong-layer reads, velocity error). CONSUMER: the anti-alignment's
//     actual subject.
// The corrected innovation (the anti-alignment's input) is
//     i_corr = i - beta*u
// with beta the local gradient (central differences -- the ramp model; the
// step mismatch at hard edges peaks at ~gap/4 on 50/50 straddles, is
// zero-mean over the sequence, and is covered by the record).
//
// MIXTURES ARE CONTENT (the anti-aliased texel): an AA texel's current
// sample is itself a coverage mixture of foreground and background -- the
// mixture IS the sub-texel signal. The statistics therefore run on the
// FULL tap set, always: mixtures enter through (i) the jitter-centered
// mean (the phase-correct target: the mixture at the CURRENT phase, so
// the shrinkage never fights the jitter), (ii) the record (the mixture's
// phase spread: the sub-texel room the gate must cover), (iii) the phase
// bit above. No depth-plane masking of the color stats anywhere: masking
// would treat a texel's own mixture as an outlier and de-antialias it.
//
// DILATION SANCTITY: the clip treats every dilation zone identically --
// exempt from the anti-alignment reset, full-mixture stats, the fiction
// machinery as designed. The band is the silhouette's AA feather; the
// clip's domain begins beyond it. Where bands exist at all is the
// own-history validation's decision, never the color side's.
//
// THE SAMPLING LAW OF THE GATE (Studentization -- the exact threshold):
// the gate standardizes d by the record's ESTIMATE of E, not by E itself.
// Under the null (isotropic Gaussian; the estimate independent of d -- the
// conservative side, the accumulator's coupling only tightens the true
// law):
//     mdd / 3 = [chi^2_3 / 3] / [chi^2_nu / nu]  ~  F(3, nu)
// The exact threshold at coverage p is 3 F_{3,nu}(p), NOT chi^2_3(p). The
// record's EWMA carries nu(t) effective dof (Satterthwaite):
//     nu = 1 / [D/nu0 + (rho/(2-rho)) (1-D)],   D = (1-rho)^{2t}
// (nu0 = 4: the seed's declared prior dof -- see THE SEED below) --
// nu = 4 at the seed, 12.33 at convergence for rho = 0.15. Implemented as
//     S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2
// fit BY THE HOST (studentFitAB, client/postFx/taa.lua) to the exact
// F(3,nu) quantiles at the trajectory's two extremes -- nu = 4 (age 0)
// and nu = 12.33 (converged) -- so the inflation is exact at birth and at
// convergence and within ~1% across the lived range, AT EVERY SLIDER
// VALUE: the required inflation is a function of the requested coverage
// (the exact first-order term is (chi^2 - 1)/(2 nu); the full S(4) is
// 1.18 at chi = 1.5 and 2.52 at chi = 2.8 -- a fixed constant cannot
// serve both). The fit consumes the EFFECTIVE radius
// chi * (1 + clipOvershoot) -- the same threshold the gate tests -- and
// the host's STUDENT_RHO / STUDENT_NU0 must match kClipSigmaEmaRate /
// kClipStudentPriorDof. Unset constants read 0: S = 1, the pre-Student
// gate (graceful). The COLD path is deliberately NOT Studentized: it is
// not a coverage statement but the change-point replacement.
//
// THE ACCUMULATOR'S TRANSIENT (the exact center statistic): Var(h) =
// (a/2)E is the STEADY state. After a reset h IS a fresh mu-estimate
// (variance muShare E, muShare = invNeff (1 - a/2)), relaxing at (1-a)^2
// per frame -- at feedback 0.97 a genuine ~10-frame transient:
//     cStat(t) = [a/2 + (muShare - a/2)(1-a)^{2t}] + muShare
// At t = 0 this is 2 muShare -- exactly Var(mu_prev - mu_cur), two
// independent 9-tap means.
//
// THE SEED (and the faint-ghost shelter it closes): a change-point's
// first sample is the best estimate of the new regime's OFFSET -- and the
// WORST available estimate of its VARIANCE: a single squared residual is
// one draw, drawn from the OLD error, not the new spread. Seeding E :=
// innovSq (the pre-fix form) armed the gate to the change-point's own
// magnitude: post-reset the remnant sat deep inside a gap-wide gate,
// tGate saturated to 1, and the only decay left was the blend's -- the
// faint, clip-shaped ghost. Worse, a ghost AT the gate boundary feeds its
// own residual back into E (innov^2 ~= r^2 while the boundary is
// chi*sqrt(cStat)*r ~= 0.98 r at the default chi): a self-sustaining
// neutral fixed point. The seed is now the SPATIAL prior tr(Cov_taps) --
// the new regime's MEASURED spread (9 taps, this frame) -- and the
// Student factor prices that seed's own uncertainty (nu = 4 at age 0: a
// coarse prior is exactly what low dof means). On a reveal the gate
// collapses to content scale on the first post-reset frame; the reset
// frame's hard clip feeds the clip-distance rejection (the endorsed
// refresh channel), and the accumulator re-averages from the honest
// scale.
//
// THE GHOST SIGNATURE (the anti-alignment reset -- no state, all
// channels): for the three points every texel already has -- the current
// sample x, the accumulator h, the neighborhood mean mu -- a ghost is
// unambiguous GEOMETRY: mu sits BETWEEN x and h:
//     d = h - mu;   i_corr = x - h - beta*u;
//     ghost  <=>  dot(i_corr, d) < -kAlignCos |i_corr| |d|
// Under the null i_corr ~ 0 (the phase bit cancelled exactly) and d ~ the
// phase spread -- no fire, for any landing phase or fraction. A ghost --
// LUMA OR CHROMA, armed record or not -- points anti-aligned. Guards:
//   * SPATIAL corroboration: innovCorr^2 > kAlignSpatial * tr(Cov_taps),
//     kAlignSpatial = chi^2_3(0.80)/3 = 1.55: the phase-clean corrected
//     innovation must exceed the 80% bound of the neighborhood's own
//     trace. (The pre-fix 3.0 ~= chi^2_3(0.97) existed to survive the
//     mis-signed phase regressor, which left a DOUBLED phase oscillation
//     in i_corr; with the corrected phase bit the null is single-
//     amplitude and the 80% bound holds -- the faint-ghost kill floor
//     drops from ~1.7 to ~1.2 sigma.)
//   * DISTANCE: |d|^2 > kAlignDist^2 * tr(Cov) -- blocks sub-texel fires
//     (their accumulator sits near the mean) and the test's own one-frame
//     lag (a kill rebuilds it over a full frame; 0.5 sits between).
//   * DILATION EXEMPTION: every dilation zone, both sides, direction-
//     blind (see DILATION SANCTITY).
// On fire: RESET -> the COLD gate this frame -> h pulled onto mu -> the
// ghost replaced by current content; the record re-seeds from the
// spatial prior and re-warms in ~7 frames.
//
// WHERE SIGMA COMES FROM (the information wall): the phase variance is
// unmeasurable in a single frame at blind phases (an all-background 3x3
// says nothing about the gap); it is sampled by the INNOVATION SEQUENCE,
// measured PRE-CLIP against the stored accumulator, transported
// bit-exactly through the output alpha, read back through a DEDICATED
// POINT-SAMPLED history binding, advected with the content, reset on
// disocclusion, re-warmed in ~7 frames. The record is the only carrier
// of blind-phase room -- which is why the WINSOR CAP stays
// record-referenced (C^2 E_prev, the Student-t predictive bound): the
// cap must not throttle a straddling texel's blind-phase re-learning (at
// revealing phases the edge is in the 3x3 and the ingest is uncapped; at
// blind phases E_prev carries the room).
//
// CHANGE-POINTS, in order of authority (all route to the cold gate, the
// spatial seed, and age 0):
//   (1) THE ANTI-ALIGNMENT RESET (above).
//   (2) THE SHOCK: innovation > kClipSpikeRatio times BOTH the record
//       and the neighborhood's squared AABB range (the range tracks the
//       population's tail, so heavy-tailed legit content never shocks).
//   (3) THE FICTION SENTINEL: a record written by a traveling dilation
//       band is foreign content; every non-band reader kills it.
//   (4) GEOMETRY AS EVIDENCE, NOT COMMAND: a geometric reject resets only
//       when the color side corroborates (chi^2_3 at 95%).
//
// NO STATIC/MOTION DIVISION: no motion fade, no ingest floor, no absolute
// pixel threshold in the gate. The record ingests the TOTAL innovation at
// full rate, winsorized; motion enters through what it does to the
// innovation. The blend's responsive channels are the clip-distance
// rejection (event-relative, independent floor) and the alignment drop
// (phase-relative); the motion drop is inert in the pinned configuration
// by design.
//
// THE FAINT-GHOST FLOOR, stated honestly: with geometric disocclusion
// off, the color side kills anything persistently anti-aligned above
// ~1.2 sigma_layer; below that the evidence of a single frame cannot
// distinguish a ghost from content noise, and the reset frame's
// clip-distance refresh plus re-averaging handles the remnant. That
// floor is the information limit of single-frame color evidence -- the
// geometric tests are the other half.
//
// SAFETY LAYERS: (1) the accumulator share floored at kStatAlphaFloor;
// (2) the record floored at kClipSigmaRecordFloorSq; (3) the E winsor cap
// (record-referenced -- see WHERE SIGMA COMES FROM); (4) the Student and
// transient inflations (the exact estimation-uncertainty accounting,
// slider-coupled); (5) soft clip. The emitted value always lies ON THE
// SEGMENT [mu, history]; it may exceed the tap AABB BY DESIGN (the
// sub-texel room).
//
// HONEST LIMITS: chi = taaVarianceGamma is the NOMINAL coverage radius
// of a 3-dof Gaussian (1.0 -> 20%, 1.5 -> 48%, 2.0 -> 74%, 2.2 -> 82%,
// 2.5 -> 90%, 2.7959 -> 95%, 3.0 -> 97%, 3.5 -> 99.3%); the Student
// inflation is what makes the nominal TRUE, and it TRACKS THE SLIDER.
// The F law assumes the isotropic Gaussian and estimate-independence
// (the conservative side). Var(mu) via invNeff assumes tap independence.
// THE MOTION FRINGE: under motion faster than the accumulator can track
// (the pinned-feedback regime), the anti-alignment re-fires on
// flat-near-edge texels as the one-frame lag rebuilds -- those texels
// degenerate to the SPATIAL estimator for the motion's duration.
// kAlignDist sizes the fringe; unpinning feedbackMin restores the
// blend-side response.
//
// TRANSPORT LAYOUT (the output alpha, non-debug): [31] revocation sign
// (the motion-field writer reads ONLY this); [30:27] tag 0110; [26:20]
// the state sigma, 7-bit log2 scale s = (1/255) 2^(code/8 - 8), code 0 =
// the explicit cold marker; [19:8] the acutance energy, 12 bits (the
// quantization step 2.4e-4 sits at the sharpener's own noise floor
// kSharpEnergyFloor = 1e-4, and pack-time saturation only lowers boost --
// the safe direction); [7:1] the RECORD'S AGE -- frames since the seed,
// 7 bits linear, saturating at 127 (the Student dof and the accumulator
// transient; re-warms from 0 after debug -- the conservative, maximum-
// inflation direction); [0] the FICTION FLAG (traveling-band writer).
// WHILE DEBUG IS ON the debug payload (tag 0111) carries the sigma in
// its low 7 bits; the age and the fiction flag re-warm after debug.
// Read back bit-exactly through the DEDICATED POINT-SAMPLED history
// binding (a linear sampler is NOT exact even at a snapped texel center:
// the center (k+0.5)/N is not float-representable for non-power-of-two
// sizes).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (clipping + feedback constants, taaStudentA/B). Requires
// fragments included before: taaShared.h.hlsl (kClipSigmaRef,
// ClipPackSigmaCode, ClipUnpackSigmaCode), taaConstants.h.hlsl,
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
// The FULL tap set, always (see MIXTURES ARE CONTENT): the AABB (spike test
// + the luma-drift chroma gate), the mean (gate center, NaN fallback),
// sigma (cold gate, smear rejection, spatial scales), invNeff (the mean's
// estimator share, measured live from the actual weight kernel) and
// phaseShift (the phase-correlated part of the innovation, from the EXACT
// sample snap offset).
struct ColorNeighborhoodStats
{
    float3 aabbMin;
    float3 aabbMax;
    float3 mean;
    float3 sigma;
    float  invNeff;         // SumW2 / SumW^2: Var(mu) = invNeff * Var(x)
    float3 phaseShift;      // beta * u: the phase-correlated part (u = -fracPx)
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

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        stats.aabbMin = min(stats.aabbMin, tapColorSpace);
        stats.aabbMax = max(stats.aabbMax, tapColorSpace);

        float2 offsetFromCenterPx = tapOffsetPx - weightCenterPx;
        float w = jitterCenteredWeights
            ? exp2(-dot(offsetFromCenterPx, offsetFromCenterPx) * kLog2E)
            : kStdWeights[i];

        if (taaLumaVariance > 0.5) { w *= (1.0 / (1.0 + max(tapColorSpace.x, 0.0))); }
        if (taaVelocityAlignedVariance > 0.5 && i > 0)
        {
            w *= lerp(1.0, saturate(dot(tapOffsetPx, motionDirUnit) * kInvLength[i] * 0.5 + 0.5), motionFactor);
        }

        weightedSum   += tapColorSpace * w;
        weightedSumSq += tapColorSpace * tapColorSpace * w;
        totalWeight   += w;
        totalWeightSq += w * w;
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean    = weightedSum * invTotalWeight;
    stats.sigma   = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));
    stats.invNeff = saturate(totalWeightSq * invTotalWeight * invTotalWeight);

    // THE PHASE BIT (the corrected sign + structure): the sample sits at
    // S + u with u = -jitterPx (jitterPx is pixel.fracPx, the snap residual
    // -- the sample's offset from the stable point, negated). The
    // accumulator field is PHASE-FREE (each stored texel is the
    // jitter-centered mean at its position), so the phase-correlated part
    // of the innovation is exactly beta * u = -beta * jitterPx. Subtracting
    // phaseShift from the innovation cancels it for every mean-structure
    // test; the RAW second moment (the variance path) keeps its energy
    // (the sub-texel room). The landing's fraction enters elsewhere, as
    // the resample bit R(phi_h) the record absorbs.
    float3 gradX = 0.5 * (neighborhoodColorSpace[4] - neighborhoodColorSpace[3]);
    float3 gradY = 0.5 * (neighborhoodColorSpace[2] - neighborhoodColorSpace[1]);
    stats.phaseShift = -(gradX * jitterPx.x + gradY * jitterPx.y);

    // Firefly clamp: pulls the AABB (used by the luma-drift chroma gate and
    // the spike test) into the mean +/- k*sigma band.
    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    // POPOVICIU: any distribution supported on [min, max] has variance <=
    // (max - min)^2 / 4, so the (possibly firefly-clamped) range bounds the
    // tap sigma. A no-op by construction on unclamped neighborhoods.
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
static const float kClipSigmaRecordFloorSq   = (1.0 / 255.0) * (1.0 / 255.0);  // the record's minimum carried VARIANCE (sigma >= kClipSigmaRef). NOT the transport's resolution -- the deliberate floor the change-point tests and the winsor cap normalize against.
static const float kStatAlphaFloor           = 0.10;          // accumulator-share floor (blind-phase coverage); the mu share does not depend on it
static const float kClipSpikeRatio           = 4.0;           // innovation shock: >4x the record AND the AABB range = change-point (t(11) @ ~99.8% ~= 3.9, derived not tuned)
static const float kClipWinsorC              = 3.0;           // E's winsorization cap, in sigma_hat of the PREVIOUS record (Student-t predictive outlier bound; record-referenced so a straddling texel's blind-phase room survives -- see WHERE SIGMA COMES FROM)
static const float kClipCorroborateRatio     = 2.6;           // geometric-reject corroboration, in units of the carried E[i^2]: chi^2_3(0.95)/3
static const float kAlignCos                 = 0.5;           // anti-alignment strength: dot(i_corr,d) < -0.5|i||d| (the slack absorbs ghost+phase cross terms)
static const float kAlignDist                = 0.5;           // |d| must exceed this fraction of the spatial trace: blocks sub-texel fires and the test's own one-frame lag
static const float kAlignSpatial             = 1.55;          // innovCorr^2 vs tr(Cov): chi^2_3(0.80)/3 -- the 80% bound of the neighborhood's trace. (The pre-fix 3.0 ~= chi^2_3(0.97) existed to survive the mis-signed phase regressor, which left a DOUBLED phase oscillation in i_corr; with the corrected phase bit the null is single-amplitude and the 80% bound holds -- the faint-ghost floor drops from ~1.7 to ~1.2 sigma.)
static const float kClipRejectionFeedbackFloor = 0.5;         // the clip-distance rejection's INDEPENDENT floor: a full-rejection event injects a half-weight current sample. Deliberately NOT taaFeedbackMin -- with a pinned feedback pair a floor tied to feedbackMin made the rejection a silent no-op.
// Studentization: nu(t) = 1 / [D/nu0 + sigmaW2 (1-D)], D = (1-rho)^{2t},
// sigmaW2 = rho/(2-rho) = 0.0811, 2*log2(1-rho) = -0.46893 (rho = 0.15).
// S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2 is HOST-FIT (studentFitAB
// in client/postFx/taa.lua) to the exact F(3,nu) quantiles at the lived
// range's extremes, consuming the EFFECTIVE radius chi*(1+clipOvershoot).
// The host's STUDENT_RHO / STUDENT_NU0 must match the two constants below.
static const float kClipStudentPriorDof      = 4.0;           // the SEED's declared prior dof: the spatial seed is a 9-tap spread, ~4 honest dof after correlation -- a coarse prior is exactly what low dof means, and S(4) prices it
static const float kClipStudentVarFrac       = 0.0810811;     // rho/(2-rho)
static const float kClipStudentDecay         = -0.4689303;    // 2*log2(1-rho)
static const float kClipMaxAge               = 127.0;         // the age transport's saturation (both decays are ~0 there)

// ============================================================================
// CLIP-STATE TRANSPORT (the output alpha; layout in the file header)
// ============================================================================
float PackClipStateAlpha(bool revoked, float sigma, float age, float acutance, bool fictionAdvected)
{
    uint u = 0x30000000u                                      // tag 0110
           | ((uint(ClipPackSigmaCode(sigma) + 0.5) & 0x7Fu)    << 20)
           | ((uint(saturate(acutance) * 4095.0 + 0.5) & 0xFFFu) << 8)
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
    float  sigmaStatSq,        // the temporal record (E[i^2]); < 0 = cold -> tap-based gate
    float  recordAge,          // frames since the record's seed (transported)
    float  statAlpha,          // the accumulator's motion-based blend weight (a)
    float  motionNormalized)   // soft-clip restore only -- the gate itself has no motion term
{
    ClipGateResult r;
    bool recordLive = (sigmaStatSq >= 0.0);

    // THE EXACT CENTER STATISTIC: muShare = invNeff (1 - a/2) (age-
    // independent); hShare(t) = a/2 + (muShare - a/2)(1-a)^{2t} (the
    // accumulator IS a fresh mu-estimate at t = 0, relaxing to the steady
    // (a/2)E); cStat(t) = hShare + muShare (2 muShare at t = 0: exactly
    // Var(mu_prev - mu_cur)).
    float alphaStat = max(statAlpha, kStatAlphaFloor);
    float muShare   = stats.invNeff * (1.0 - alphaStat * 0.5);
    float decayH    = exp2(recordAge * 2.0 * log2(max(1.0 - alphaStat, 0.5)));
    float hShare    = alphaStat * 0.5 + (muShare - alphaStat * 0.5) * decayH;
    float cStat     = hShare + muShare;

    // THE STUDENT FACTOR: the record's effective dof (Satterthwaite) and
    // the host-fit exact-threshold inflation S(nu) = 1 + A/nu + B/nu^2
    // (slider-coupled; unset constants read 0 -> S = 1, graceful).
    float decayPrior = exp2(recordAge * kClipStudentDecay);   // (1-rho)^{2t}
    float nu         = 1.0 / (decayPrior / kClipStudentPriorDof
                            + kClipStudentVarFrac * (1.0 - decayPrior));
    float invNu      = 1.0 / nu;
    float studentS   = 1.0 + taaStudentA * invNu + taaStudentB * invNu * invNu;

    // chromaScale multiplies the TOTAL gate (the live term applies the 3-D
    // energy per channel: up to sqrt(3) headroom on luma-dominant content,
    // which a moderate mod < 1 consumes before biting the measured chroma
    // variance).
    float3 chromaScale = max(float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod), 0.0);

    // LIVE: cStat * S(nu) * E. COLD: the mean's own estimator uncertainty,
    // per channel, on the full tap set -- the change-point replacement,
    // deliberately tight (NOT Studentized: not a coverage statement).
    float3 gateVar = recordLive
        ? (cStat * studentS * max(sigmaStatSq, kClipSigmaRecordFloorSq)).xxx
        : (stats.invNeff * stats.sigma * stats.sigma);

    float3 invGateVar = 1.0 / max(gateVar * (chromaScale * chromaScale),
                                  (kMinSigma * kMinSigma).xxx);

    // chi = the NOMINAL coverage radius (taaVarianceGamma; chi_3(0.95) =
    // 2.7959), with taaClipOvershoot as multiplicative radius slack. The
    // Student inflation above is what makes the nominal TRUE.
    float chi = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));

    // The gate on the ray from the phase-mean estimate: one dot, one sqrt.
    float3 d   = historyColorSpace - stats.mean;
    float  mdd = dot(d * d, invGateVar);
    float  tGate = (mdd > 1e-20) ? min(chi / sqrt(mdd), 1.0) : 1.0;

    if (tGate < 1.0 && taaSoftClip > 0.0)
    {
        float overshootUnits = 1.0 / max(tGate, kEpsilon);       // Mahalanobis overshoot
        float softScale      = SoftClipUnitScale(overshootUnits, taaSoftClip, saturate(motionNormalized));
        tGate = min(softScale * tGate, 1.0);
    }
    r.tGate = tGate;

    // Shrink toward the phase-mean estimate. Deliberately NOT clamped to
    // the tap AABB: exceeding it is exactly the sub-texel room the record
    // purchased.
    r.clippedColorSpace = stats.mean + d * tGate;
    return r;
}

// ============================================================================
// CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
// Normalized by the SPATIAL sigma (the neighborhood's own spread), NOT the
// gate's variance: normalizing by the record would disarm smear rejection
// precisely on ghosts. The blend's event-relative responsive channel: it
// fires when the clip fires, whatever the cause. Requires nothing of the
// feedback pair: the drop lands on an independent floor and acts at any
// feedbackMin/Max, including the pinned configuration.
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

// History feedback. The rejection can only LOWER feedback (never raise it
// back toward the floor when the alignment drop already went below); the
// alignment drop is applied AFTER the clamp and ONLY on planar surfaces.
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