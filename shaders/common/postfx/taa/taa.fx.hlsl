// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect
// ----------------------------------------------------------------------------
// CONTEXT: this mod produces sub-pixel jitter by PHYSICALLY ROTATING the
// in-game camera every frame (there is no projection-matrix jitter). Two
// consequences shape this whole file:
//
//   1. The per-frame camera bases (P/Q/R below) already contain the jitter, so
//      reprojecting a stable output pixel through the CURRENT camera yields
//      its jittered sample position, and through the PREVIOUS camera yields
//      the previous frame's jitter offset (used for jitter cancellation).
//   2. The velocity buffers therefore ALSO contain the jitter motion. Every
//      comparison that must be jitter-free subtracts the jitter component
//      (EstimateJitterCancelUV / EstimateJitterTransportUV).
//
// The history buffer stores the previous frame's STABLE (de-jittered) output.
//
// NAMING CONVENTIONS
//   *VelocityJitteredUV  motion in UV units (current -> previous frame) taken
//                        straight from a velocity buffer: INCLUDES camera
//                        jitter, is NOT dilated, NOT interpolated.
//   *Dilated*            taken from the closest-depth (foreground) tap rather
//                        than the exact / bilinear position.
//   *Effective*          the layer a pixel ACTS as: a dilation zone resolves
//                        to the foreground it dilates to (both frame sides).
//   *Nearest*            single point-sampled tap (velocity samplers are Point).
//   *Prev*               sampled from the previous frame (prevVelocityTex or
//                        the history buffer's depth channel).
//   Depth "Raw"          raw depth-buffer value (non-linear; LinearizeDepth()
//                        converts to linear view depth; reverse-Z: larger = closer).
//   *ColorSpace          color in the tonemapped Oklab/YCoCg working space.
//   *Px                  quantity expressed in pixels.
//
// DEBUG MODES (taaDebugMode):
//   0 off | 1 frame motion | 2 disocclusion breakdown (R=depth, G=velocity,
//   B=suppressed pursuit alert) | 3 center velocity | 4 linearized depth |
//   5 history color | 6 previous-frame velocity | 7 pursuit divergence |
//   8 depth edge state (red=dilation zone, cyan=foreground edge, magenta=
//   depth-flat but velocity-straddled) | 9 alignment-drop activity (red=applied
//   drop, green tint=planar/eligible, blue tint=edge/dilation exempt)
//
// CHANGELOG:
//   [FIX 1] Firefly clamp no longer collapses the AABB to a point (it used to
//           clamp aabbMax FROM aabbMin: zeroed spatialContrast, clipMargin,
//           and could invert the variance clip box).
//   [FIX 2] Velocity-rejection mask striping on depth-missed disocclusions:
//     a) prev-velocity candidate is no longer a per-pixel binary choice; the
//        dejittered error is the MIN over both candidates (dilated + nearest).
//     b) velocity-noise gradients no longer use depth-layer tests; they now
//        segment by VELOCITY coherence.
//     c) the pursuit landing uses the exact jitter transport term
//        (s_t - s_{t-1} + s_{t-2}) instead of the comparison junk.
//   [FIX 3] Striped rejections / striped ghosting behind FAST occluders: the
//           pursuit tolerance's distance-scaled gradient term grew as fast as
//           the disocclusion signal itself (with the exact transport, landing
//           distance ~= relative motion ~= divergence), so any saturated local
//           gradient made rejection impossible except in bands. The tolerance
//           is now the base plus the LANDING's own velocity-coherent spread
//           (the local noise floor of vLanding). Same-surface protection comes
//           from the exact landing (distance ~ 0) and the alert gate.
//   [FIX 4] False velocity disocclusions on graded surfaces (the ground):
//     a) the history landing and the pursuit landing now resolve their
//        velocity/depth with the SAME state-aware rules as the current frame
//        (dilation zone -> foreground tap, crest -> center, flat -> bilinear
//        at the exact sub-texel position). The old unconditional
//        closest-depth tap read the NEAREST texel of graded ground -- a
//        different point whose parallax velocity differs by grad * ~1 texel
//        -- and that bias, not a surface change, was the "divergence".
//     b) the tolerance gains a smooth-field advection allowance, granted ONLY
//        where the landing's velocity field is locally continuous (velocity-
//        curvature test; a min-over-pairs test misreads smooth ANISOTROPIC
//        gradients -- a strafed ground plane -- as layered, because the pair
//        perpendicular to the gradient always measures zero). Layer
//        boundaries keep the [FIX 3] noise-floor tolerance, and the allowance
//        never scales with the transport distance or the measured error
//        (both grow as fast as the signal itself; see [FIX 3]).
//   [FIX 5] False velocity disocclusions on the NEAR ground (bottom of the
//           image) under forward motion:
//     a) continuity ratio 0.30 -> 0.65: a curved parallax field has
//        curvature/pair-gradient ~ 1/L (<= ~0.5 even at the image bottom,
//        L = field scale length in texels >= ~2) while a true velocity step
//        has ~1..2; the old value classified the near ground as layered and
//        withheld the advection allowance exactly there.
//     b) the allowance includes the transport's second-order miss
//        (0.5 * measured dejittered error): under a dolly the linear
//        2*x_{t-1}-x_{t-2} transport lands short by ~half the per-frame
//        velocity change, and the resulting divergence scales as
//        gradient * that miss.
//     c) the allowance is gated by the DEPTH test's same-surface confidence
//        (kappa-corrected transport score, "depthGate"). On a genuine
//        disocclusion the depth transport mismatches, the gate closes, and
//        the velocity test keeps its tight [FIX 3] tolerance.
//   [FIX 6] False velocity disocclusions on the ground at HIGH speed
//           (150+ kph): the same-surface advection (per-frame velocity change
//           of one point advecting through its own parallax field) grows past
//           any fixed tolerance clamp. The advection is now PREDICTED and
//           SUBTRACTED (first-order Jacobian correction, the velocity
//           analogue of the depth test's kappa):
//             * step-1 alert: tolerance += |J_current * v_current| (exact
//               first-order prediction of the same-surface error);
//             * pursuit: divergence -= J_landing * missVec, where missVec is
//               the dejittered step-1 error vector (IDENTICALLY the linear
//               transport's landing miss) -- cancels the same-surface
//               divergence to ~1-3 px at 300 kph where the raw value is ~80.
//           Both terms are gated by field continuity, single-layer depth
//           structure and the depth test's same-surface confidence; a
//           residual budget covers only second-order terms. Layer boundaries
//           and depth mismatches keep the raw signal + tight tolerance.
//   [FIX 7] The sub-pixel alignment feedback drop was applied BEFORE the
//           feedbackMin/Max clamp, so with equal bounds (the default
//           0.97 / 0.97) the clamp swallowed the subtraction and the drop was
//           completely inert. It is now applied AFTER the clamp (and after
//           the shadow / clip-distance reductions, so they cannot pull a
//           lowered feedback back up), floored at 0, and ONLY on planar
//           surfaces (the bilinear-velocity path): on planes the current
//           frame is jittered supersampling, so trading a little accumulation
//           for resampling accuracy preserves texture sharpness; on edges and
//           dilation zones the current sample is aliased and dropping history
//           there reintroduces flicker. (Static scenes produce a FIXED
//           per-pixel pattern, not temporal pulsing: the de-jittered landing
//           phase is constant per pixel; in motion the phase is decorrelated.)
//   [REVERT] Dilation-zone handling in the velocity disocclusion test: two
//           attempts (a same-layer velocity-diameter noise allowance; then a
//           depth-gated same-layer stand-down) each suppressed the random
//           false rejections on fast-foreground dilation zones but also muted
//           genuine reveals ("dilation zone replaced by background"). Both
//           removed; the original comparison is restored. The occasional
//           random activation on fast-foreground dilation zones was ACCEPTED
//           as a precision limitation of the point-sampled velocity
//           comparison (a dilation zone resolves to a single foreground
//           texel, and the texels chosen on either frame side can sit a few
//           texels apart through the layer's own velocity gradient). Tuning
//           levers: taaVelGradientScale (noise allowance) and
//           taaCrossTestStrength (pursuit confirmation strictness).
//           (Root cause since addressed by [FIX 8]: the speckle was the
//           dilation zone's OWN advection -- unpredicted on the alert side
//           and uncorrected in the pursuit, because both corrections were
//           gated off exactly at silhouette landings.)
//   [SEMANTICS] Dilation zones act as the object they dilate to on BOTH sides
//        of every comparison. The history landing validates ONLY against its
//        effective layer (the rendered background texel of a dilation zone is
//        no longer a rejection candidate), so "dilation zone replaced by
//        background" rejects while "foreground replaced by its own dilation
//        zone" keeps. The depth test uses the landing's effective depth too.
//   [FIX 8] The velocity disocclusion test's layer semantics are now
//           VELOCITY-NATIVE instead of depth-derived. Two long-standing
//           symptoms share this root cause:
//             * the missed ghost under a lip angled down onto the ground (or
//               any fast object skimming a surface): the grazing geometry
//               collapses the depth gap below edgeEps at the boundary texels,
//               so BOTH frame sides classify FLAT and bilinear-blend a quad
//               that straddles the two layers. The two blend weight sets are
//               set by independent sub-texel phases, so for a fraction of
//               pixels the inter-layer velocity step -- the entire signal --
//               averaged to zero: no alert, ghost kept. The velocity test had
//               inherited depth's blindness because it asked DEPTH which
//               layer to compare.
//             * the random dilation-zone rejections: a dilation zone's
//               resolved velocity is one foreground texel's, so its
//               cross-frame comparison carries that point's own advection
//               (J*V-scale on a fast, strongly graded foreground) -- which
//               the alert never predicted (single-layer gate) and the pursuit
//               never corrected (single-layer landing gate), exactly at
//               silhouettes, where dilation zones live.
//           a) the flat-path bilinear velocity (ALL THREE landing sites:
//              current frame, history landing, pursuit landing) commits to
//              the rendered layer's center tap when the phase-selected quad
//              straddles a velocity step. The straddle test requires a
//              DISCONTINUOUS field (a locally linear field of ANY gradient is
//              exactly reproducible by the bilinear, so magnitude alone never
//              commits) AND a quad step beyond kQuadStepGradMul x the
//              velocity-COHERENT gradient -- never the step itself, which
//              would scale the threshold with the signal ([FIX 3] failure
//              mode). Steps below the coherence floor keep the bilinear:
//              their blending error is below the alert tolerance by
//              construction (SNR floor, degrades gracefully).
//           b) the Jacobian and the continuity/shape estimators take layer-
//              consistent differences (one-sided across layer boundaries).
//              Raw central differences at a straddled neighborhood are half
//              the inter-layer step, which both poisoned the [FIX 6]
//              prediction and made IsContinuousVelocityField misread
//              silhouettes as curvature. On all-coherent neighborhoods the
//              coherent estimators are bit-identical to the raw ones.
//           c) dilation-zone / crest pixels get their advection predicted on
//              the ALERT side too (landing-field max-norm bound, gated by the
//              landing's continuity and the depth gate).
//           d) the pursuit's [FIX 6] correction no longer requires a
//              single-layer landing: a dilation/crest landing resolves to its
//              effective (foreground) layer, so its divergence carries the
//              SAME same-surface advection the correction cancels. The
//              correction is self-discriminating: same surface ->
//              divergence ~= J*miss, cancelled to the residual; cross
//              surface -> corrected ~= (I+J)*divergence, which GROWS
//              (cancellation would require J ~= -I, a field contracting 100%
//              per texel). "Dilation zone replaced by background" still fires
//              at full strength: the prediction only absorbs the
//              advection-scale component (J*V < V).
//           e) the residual advection budget's miss term is capped at the
//              predicted same-surface advection * kMissCapHeadroom -- the
//              HONEST miss scale, since on a same-surface pixel the dejittered
//              step-1 error IS the advection to first order. The raw measured
//              error on a depth-missed disocclusion IS the signal; letting the
//              budget scale with it ate the signal (the last remaining
//              [FIX 3] violation). No regression on the [FIX 6] 300 kph case:
//              there miss ~= predicted, so the cap binds only cross-surface.
//   [CHG]   taaVelJitterCancel removed: the de-jitter term is verified exact
//           and is always applied (the cbuffer slot is kept as padding).
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(sceneTex,        0);
uniform_sampler2D(depthTex,        1);
uniform_sampler2D(historyTex,      2);
uniform_sampler2D(velocityTex,     3);
uniform_sampler2D(prevVelocityTex, 4);

// ============================================================================
// CBUFFER (engine-set constants -- DO NOT reorder or rename, layout is fixed)
// ============================================================================
cbuffer perDraw
{
    float  taaFeedbackMin;                  float  taaFeedbackMax;
    float  taaShadowMitigation;             float  taaShadowDarknessThreshold;
    float  taaShadowBlendStrength;          float  taaVarianceGamma;
    float  taaSoftClip;                     float  taaChromaVarianceMod;
    float  taaJitterFlickerPadding;         float  taaJitterFlickerFade;
    float  taaDepthRejection;               float  taaTanHalfFovX;
    float  taaTanHalfFovY;                  float  taaUseDepthDilation;
    float  taaLumaVariance;                 float  taaUseCovarianceClipping;
    float  taaColorSpaceOklab;              float  taaJitterAwareVariance;
    float  taaVelocityAlignedVariance;      float  taaAlignmentFeedbackDrop;
    float  taaMotionBlendDropSpeed;         float  taaUseLanczos3;
    float  taaFireflyClamp;                 float  taaShadowTemporalMult;
    float  taaShadowSpatialMult;            float  taaDirectionalVariance;
    float  taaClipDistanceRejectionEnabled; float  taaClipDistanceRejectionAmount;
    float  taaClipDistanceRejectionMinError; float taaUseKDopClipping;
    float  taaKDopVariance;                 float  taaFallbackFXAA;
    float  taaMotionBlendStart;             float  taaShadowVarianceBase;
    float  taaDebugMode;                    float  taaCurPX;
    float  taaCurPY;                        float  taaCurPZ;
    float  taaCurQX;                        float  taaCurQY;
    float  taaCurQZ;                        float  taaCurRX;
    float  taaCurRY;                        float  taaCurRZ;
    float  taaPrevPX;                       float  taaPrevPY;
    float  taaPrevPZ;                       float  taaPrevQX;
    float  taaPrevQY;                       float  taaPrevQZ;
    float  taaPrevRX;                       float  taaPrevRY;
    float  taaPrevRZ;                       float  taaHistoryOvershoot;
    float  taaLumaDriftStrength;            float  taaLumaDriftChromaTol;
    float  taaClipOvershoot;                float  taaVelRejection;
    // --- velocity disocclusion / jitter plumbing ---
    float  taaVelGradientScale;             float  taaVelPad0;  // pad (was taaVelJitterCancel; de-jitter is always on)
    float  taaCrossTestStrength;            float  taaJitPrev2Yaw;
    float  taaJitPrev2Pitch;                float  taaJitPad0;

    float2 oneOverTargetSize;
    POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#ifdef SHADER_STAGE_VS
#define mainV main
#else
#define mainP main
#endif

// ============================================================================
// CONSTANTS
// ============================================================================
static const float kEpsilon                  = 1e-5;
static const float kLargeValue               = 32000.0;
static const float kSqrt2                    = 1.41421356;
static const float kLog2E                    = 1.44269504;
static const float kMinMotionDirLengthPx     = 0.1;     // below this motion is "static"
static const float kShadowLumaFloor          = 0.01;
static const float kShadowThresholdMin       = 0.001;
static const float kFlickerPadThreshold      = 0.001;
static const float kMinMotionBlendDropSpeed  = 0.1;
static const float kMotionFullStrengthPx     = 8.0;     // motion that fully drops history feedback

static const float kMinSigma                 = 0.001;
static const float kMinSpatialContrast       = 0.001;
static const float kMinFootprintRange        = 1e-4;
static const float kFireflyClampEpsilon      = 0.001;
static const float kHistoryTapReachPx        = 1.5;     // how far a history filter footprint may reach

// Fallback FXAA
static const float kFXAAReduceMul            = 1.0 / 128.0;
static const float kFXAAReduceMin            = 1.0 / 128.0;
static const float kFXAAMaxDir               = 8.0;
static const float kFXAABlendKnee            = 0.8;
static const float kFXAABlendSharpness       = 2.0;

// Luma drift correction
static const float kLumaDriftLumaFloor       = 0.05;
static const float kLumaDriftRelThreshold    = 0.30;
static const float kLumaDriftAbsThreshold    = 0.03;
static const float kLumaDriftChromaFadeWidth = 2.0;

// Current-frame pursuit confirmation tolerance
static const float kPursuitVelBaseTolerancePx = 0.30;   // px: base divergence tolerance
static const float kMinVelCoherenceRadiusPx   = 0.25;   // floor for the velocity-coherence radius

// Smooth-field (continuous parallax) advection handling. The advection error
// is PREDICTED and subtracted (first-order Jacobian correction -- the velocity
// analogue of the depth test's kappa), not merely tolerated:
//   * ground under forward motion has a smooth, strongly curved velocity
//     field; at 150+ kph the same-surface per-frame velocity change exceeds
//     80 px -- beyond any fixed budget. J_landing * (landing miss) cancels it
//     to ~1-3 px (verified analytically for the ground-plane flow map).
//   * the dejittered step-1 error VECTOR is exactly the pursuit landing miss,
//     so the correction needs no extra state.
//   * the residual budget only covers second-order terms (field curvature
//     along the miss, camera acceleration, Jacobian noise * miss), gated by
//     continuity, single-layer depth structure AND the depth test's
//     same-surface confidence. A layer boundary -- the disocclusion signal
//     itself -- never receives either term.
static const float kVelDiscontinuityRatio    = 0.65;    // curvature/pair-gradient ratio that still counts as continuous
static const float kVelDiscontinuityAbsPx    = 0.25;    // px: velocity-curvature noise floor
static const float kVelJacobianResidualFrac  = 0.25;    // fraction of |J|*miss kept for 2nd-order terms
static const float kPursuitSnapPadPx         = 1.5;     // px: fixed part of the residual pad
static const float kPursuitMaxAdvectionPx    = 12.0;    // px: hard clamp on the RESIDUAL budget only

// [FIX 8] Layer-aware velocity selection (depth-invisible boundaries).
static const float kQuadStepGradMul          = 4.0;     // quad step must exceed coherentGrad * this to commit
static const float kMissCapHeadroom          = 1.5;     // same-surface miss cap headroom (Jacobian underestimate insurance)

// Debug views
static const float kDebugVelocityScale       = 0.1;
static const float kDebugLinearDepthRange    = 100.0;

static const float2 kOffsets3x3[9] =
{
    float2( 0,  0), float2( 0, -1), float2( 0,  1),
    float2(-1,  0), float2( 1,  0), float2(-1, -1),
    float2( 1, -1), float2(-1,  1), float2( 1,  1)
};

// Fixed Gaussian-style 3x3 weights (e^-dist^2) used when the kernel is not
// jitter-shifted. kInvLength = 1/|offset| for directional weighting.
static const float kStdWeights[9] = { 1.0, 0.36787944, 0.36787944, 0.36787944, 0.36787944, 0.13533528, 0.13533528, 0.13533528, 0.13533528 };
static const float kInvLength[9]  = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678 };

// 16 fixed axes for k-DOP neighborhood clipping.
static const float3 kDopAxes[16] =
{
    float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0),
    float3(0.820081, 0.456727, -0.344773), float3(0.540295, 0.829202, 0.143195),
    float3(0.255800, 0.841084, -0.476597), float3(-0.406935, -0.389062, 0.826459),
    float3(-0.826708, -0.382923, -0.412219), float3(0.260942, -0.577482, 0.773578),
    float3(0.254398, 0.637821, 0.726957), float3(0.310900, -0.728083, -0.610930),
    float3(0.798513, -0.556827, -0.228738), float3(0.673383, -0.163602, -0.720964),
    float3(-0.813922, 0.369658, -0.448201), float3(0.477650, -0.853722, 0.207384),
    float3(-0.554854, -0.041550, -0.830910)
};

// ============================================================================
// SMALL REUSABLE MATH HELPERS
// ============================================================================
float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

// Standard bilinear interpolation of a 2x2 quad (fraction in [0,1]).
float  Bilerp2x2(float  c00, float  c10, float  c01, float  c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}
float2 Bilerp2x2(float2 c00, float2 c10, float2 c01, float2 c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

// Soft-clip scale factor: 1.0 = hard clip; grows toward (1 + softClipAmount) as
// the requested overshoot (in box-size units) grows; motion restores hard clip.
float SoftClipUnitScale(float overshootUnits, float softClipAmount, float motionFactor)
{
    float softLimit = 1.0 + softClipAmount * (1.0 - exp2(-(overshootUnits - 1.0) * kLog2E));
    return lerp(softLimit, 1.0, motionFactor);
}

// Entry/exit interval of a ray against one slab [slab.x, slab.y] along one axis.
float2 RaySlabInterval(float origin, float dir, float2 slab)
{
    float sign   = (dir >= 0.0) ? 1.0 : -1.0;
    float invDir = 1.0 / (sign * max(abs(dir), 1e-7));
    float t0     = (slab.x - origin) * invDir;
    float t1     = (slab.y - origin) * invDir;
    return float2(min(t0, t1), max(t0, t1));
}

// Closed-form inverse of a symmetric 3x3 matrix (covariance matrices).
float3x3 InverseSymmetric3x3(float3x3 m, out bool invertible)
{
    float c00 = m[1][1] * m[2][2] - m[1][2] * m[1][2];
    float c01 = m[0][2] * m[1][2] - m[0][1] * m[2][2];
    float c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];

    float det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    invertible = (abs(det) > kEpsilon);
    if (!invertible) return float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);

    float invDet = 1.0 / det;
    float3x3 inv;
    inv[0][0] = c00 * invDet;
    inv[0][1] = c01 * invDet;
    inv[0][2] = c02 * invDet;

    inv[1][0] = c01 * invDet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[0][2]) * invDet;
    inv[1][2] = (m[0][1] * m[0][2] - m[0][0] * m[1][2]) * invDet;

    inv[2][0] = c02 * invDet;
    inv[2][1] = inv[1][2];
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[0][1]) * invDet;

    return inv;
}

// ============================================================================
// COLOR SPACES & REVERSIBLE TONEMAPPING
// ============================================================================
// Reversible luma tonemap keeps bright fireflies from dominating the clip box.
float3 Tonemap(float3 c)   { float luma = LumaRGB(c); return c / (1.0 + luma); }
float3 Untonemap(float3 c) { float luma = LumaRGB(c); return c / max(1.0 - min(luma, 0.999), 1e-4); }

static const float3x3 kRGB_TO_LMS    = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB  = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS  = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB    = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800, 2.60975740, -0.34131939, -0.00419608, -0.70341861, 1.70761470);

float3 RGBToOklab(float3 c)
{
    float3 lms = max(mul(kRGB_TO_LMS, c), 0.0);
    float3 lmsRoot;
    lmsRoot.x = (lms.x > 0.0) ? pow(lms.x, 1.0 / 3.0) : 0.0;
    lmsRoot.y = (lms.y > 0.0) ? pow(lms.y, 1.0 / 3.0) : 0.0;
    lmsRoot.z = (lms.z > 0.0) ? pow(lms.z, 1.0 / 3.0) : 0.0;
    return mul(kLMS_TO_OKLAB, lmsRoot);
}

float3 OklabToRGB(float3 c) { float3 l = mul(kOKLAB_TO_LMS, c); return mul(kLMS_TO_RGB, l * l * l); }
float3 RGBToYCoCg(float3 c) { return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
float3 YCoCgToRGB(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// RGB <-> tonemapped working space (Oklab or YCoCg, selected by config).
float3 ToSpace(float3 rgb)
{
    float3 tonemapped = Tonemap(max(0.0, rgb));
    return (taaColorSpaceOklab > 0.5) ? RGBToOklab(tonemapped) : RGBToYCoCg(tonemapped);
}
float3 FromSpace(float3 c)
{
    float3 rgb = (taaColorSpaceOklab > 0.5) ? OklabToRGB(c) : YCoCgToRGB(c);
    return Untonemap(max(0.0, rgb));
}

// Clamp the history color back into the valid RGB gamut if clipping pushed
// chroma outside (keeps hue, pulls toward the luma axis).
float3 CompressGamut(float3 historyColorSpace)
{
    if (taaColorSpaceOklab > 0.5)
    {
        float3 historyRGB = OklabToRGB(historyColorSpace);
        float minChannel = min(historyRGB.r, min(historyRGB.g, historyRGB.b));
        if (minChannel < 0.0)
        {
            float luma  = max(0.0, LumaRGB(historyRGB));
            float scale = saturate(luma / max(luma - minChannel, kEpsilon));
            historyRGB = luma + (historyRGB - luma) * scale;
            historyColorSpace = RGBToOklab(historyRGB);
        }
        return historyColorSpace;
    }

    // YCoCg: reconstruct RGB to detect gamut violation.
    float Y  = historyColorSpace.x;
    float Co = historyColorSpace.y;
    float Cg = historyColorSpace.z;
    float r = Y + Co - Cg;
    float g = Y + Cg;
    float b = Y - Co - Cg;
    float minChannel = min(r, min(g, b));
    if (minChannel < 0.0)
    {
        float scale = saturate(max(0.0, Y) / max(Y - minChannel, kEpsilon));
        historyColorSpace.yz *= scale;
    }
    return historyColorSpace;
}

float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

// ============================================================================
// CAMERA BASIS & REPROJECTION
// ============================================================================
// One frame's jittered camera axes, pre-scaled by tan(halfFov) and expressed in
// the stable output basis. Because jitter is a PHYSICAL camera rotation, these
// bases carry the jitter.
struct CameraBasis
{
    float3 rightTanFov;   // P: +X screen axis * tanHalfFovX
    float3 forward;       // Q: view forward axis
    float3 downTanFov;    // R: +Y-down screen axis * tanHalfFovY
};

CameraBasis GetCurrentFrameCameraBasis()
{
    CameraBasis c;
    c.rightTanFov = float3(taaCurPX,  taaCurPY,  taaCurPZ);
    c.forward     = float3(taaCurQX,  taaCurQY,  taaCurQZ);
    c.downTanFov  = float3(taaCurRX,  taaCurRY,  taaCurRZ);
    return c;
}

CameraBasis GetPreviousFrameCameraBasis()
{
    CameraBasis c;
    c.rightTanFov = float3(taaPrevPX, taaPrevPY, taaPrevPZ);
    c.forward     = float3(taaPrevQX, taaPrevQY, taaPrevQZ);
    c.downTanFov  = float3(taaPrevRX, taaPrevRY, taaPrevRZ);
    return c;
}

// Canonical projection of a camera-space ray to stable screen UV.
float2 ProjectRayToStableUV(float3 ray, float2 fallbackUV)
{
    if (ray.y <= kEpsilon) return fallbackUV;
    return float2((ray.x / (ray.y * max(taaTanHalfFovX, kEpsilon))) * 0.5 + 0.5,
                  0.5 - (ray.z / (ray.y * max(taaTanHalfFovY, kEpsilon))) * 0.5);
}

// Camera-space ray seen by `camera` at screen position `uv`.
float3 BuildCameraRay(float2 uv, CameraBasis camera)
{
    return uv.x * camera.rightTanFov + camera.forward - uv.y * camera.downTanFov;
}

// Screen-space position, under `camera`, of the ray that the stable output
// shows at `uv`. With a jittered camera this yields the jittered sample
// position; with the previous camera it yields the previous frame's jitter
// offset (used by jitter cancellation).
float2 ReprojectThroughCamera(float2 uv, CameraBasis camera, float2 fallbackUV)
{
    return ProjectRayToStableUV(BuildCameraRay(uv, camera), fallbackUV);
}

// ============================================================================
// VIEWPORT & PIXEL GEOMETRY
// ============================================================================
struct ViewportParams
{
    float2 texelSize;    // 1 / target size
    float2 sizePixels;   // target size
    float2 minUV;        // half-texel-inset sampling bounds
    float2 maxUV;
};

ViewportParams GetViewportParams()
{
    ViewportParams vp;
    vp.texelSize  = oneOverTargetSize;
    vp.sizePixels = 1.0 / max(oneOverTargetSize, 1e-6);
    vp.minUV      = 0.5 * vp.texelSize;
    vp.maxUV      = 1.0 - vp.minUV;
    return vp;
}

// Where the stable output pixel lands inside the jittered render: the snapped
// texel center to sample from plus the remaining sub-texel jitter.
struct SnappedPixel
{
    float2 jitteredUV;   // jittered screen position of the stable pixel's ray
    float2 snappedUV;    // clamped texel-center UV used for all current-frame taps
    float2 jitterPx;     // sub-texel jitter offset in pixels, range [-0.5, 0.5]
};

SnappedPixel SnapJitteredPixel(float2 stableUV, CameraBasis currentCamera, ViewportParams vp)
{
    SnappedPixel px;
    px.jitteredUV      = ReprojectThroughCamera(stableUV, currentCamera, stableUV);
    float2 pixelPos    = px.jitteredUV * vp.sizePixels;
    float2 texelCenter = floor(pixelPos) + 0.5;   // nearest texel center
    px.snappedUV       = clamp(texelCenter * vp.texelSize, vp.minUV, vp.maxUV);
    px.jitterPx        = pixelPos - texelCenter;
    return px;
}

// Viewport-clamped tap UVs of the 3x3 neighborhood around a snapped texel
// (index order matches kOffsets3x3).
void Build3x3TapUVs(float2 centerUV, float2 texelSize, float2 minUV, float2 maxUV, out float2 tapUVs[9])
{
    float2 down = clamp(centerUV - texelSize, minUV, maxUV);
    float2 up   = clamp(centerUV + texelSize, minUV, maxUV);
    tapUVs[0] = centerUV;
    tapUVs[1] = float2(centerUV.x, down.y);
    tapUVs[2] = float2(centerUV.x, up.y);
    tapUVs[3] = float2(down.x, centerUV.y);
    tapUVs[4] = float2(up.x,  centerUV.y);
    tapUVs[5] = float2(down.x, down.y);
    tapUVs[6] = float2(up.x,  down.y);
    tapUVs[7] = float2(down.x, up.y);
    tapUVs[8] = float2(up.x,  up.y);
}

// ============================================================================
// FALLBACK FXAA (strict viewport-clamped; takes preloaded taps)
// ============================================================================
// tapsRGB: [0]=center, [1]=NW(-1,-1), [2]=NE(+1,-1), [3]=SW(-1,+1), [4]=SE(+1,+1)
float3 ApplyFXAA(float2 centerUV, float2 texelSize, float3 tapsRGB[5], float2 minUV, float2 maxUV)
{
    float lumaNW = LumaRGB(tapsRGB[1]);
    float lumaNE = LumaRGB(tapsRGB[2]);
    float lumaSW = LumaRGB(tapsRGB[3]);
    float lumaSE = LumaRGB(tapsRGB[4]);
    float lumaM  = LumaRGB(tapsRGB[0]);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * kFXAAReduceMul), kFXAAReduceMin);
    float rcpDirMin = 1.0 / (min(abs(lumaMax - lumaMin), max(lumaMax, 1.0)) + dirReduce);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    dir = clamp(dir * rcpDirMin, float2(-kFXAAMaxDir, -kFXAAMaxDir), float2(kFXAAMaxDir, kFXAAMaxDir)) * texelSize;

    float2 uv0 = clamp(centerUV + dir * (1.0 / 3.0 - 0.5), minUV, maxUV);
    float2 uv1 = clamp(centerUV + dir * (2.0 / 3.0 - 0.5), minUV, maxUV);
    float3 rgbA = 0.5 * (
        tex2Dlod(sceneTex, float4(uv0, 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv1, 0.0, 0.0)).rgb);

    float2 uv2 = clamp(centerUV + dir * (0.0 / 3.0 - 0.5), minUV, maxUV);
    float2 uv3 = clamp(centerUV + dir * (3.0 / 3.0 - 0.5), minUV, maxUV);
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        tex2Dlod(sceneTex, float4(uv2, 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv3, 0.0, 0.0)).rgb);

    float lumaB = LumaRGB(rgbB);
    return ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
}

// ============================================================================
// HISTORY COLOR RESAMPLING
// ============================================================================
float Lanczos3Weight(float fracCoord, int tap)
{
    float d   = abs(fracCoord + float(2 - tap));
    float piD = 3.14159265 * max(d, 1e-5);
    return 3.0 * sin(piD) * sin(piD * (1.0 / 3.0)) / (piD * piD);
}

// Clamp a resampled color to the min/max of the taps it was built from, with a
// small overshoot margin. Shared by both history filters.
float3 ClampToTapFootprint(float3 color, float3 tapMin, float3 tapMax)
{
    float3 tapRange  = max(tapMax - tapMin, kMinFootprintRange);
    float3 overshoot = taaHistoryOvershoot * tapRange;
    return clamp(color, tapMin - overshoot, tapMax + overshoot);
}

float3 SampleHistoryColorLanczos3(float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    // Normalized weight sums per axis (Lanczos lobes can be negative).
    float2 weightSum = float2(0.0, 0.0);
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        weightSum.x += Lanczos3Weight(fracOffset.x, i);
        weightSum.y += Lanczos3Weight(fracOffset.y, i);
    }
    float2 invWeightSum = 1.0 / max(weightSum, 1e-5);

    float3 color  = float3(0.0, 0.0, 0.0);
    float3 tapMin = float3(kLargeValue, kLargeValue, kLargeValue);
    float3 tapMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    for (int y = 0; y < 6; ++y)
    {
        float wY = Lanczos3Weight(fracOffset.y, y) * invWeightSum.y;
        float vy = clamp((baseTexel.y + float(y - 2)) * vp.texelSize.y, minUV.y, maxUV.y);
        for (int x = 0; x < 6; ++x)
        {
            float wX = Lanczos3Weight(fracOffset.x, x) * invWeightSum.x;
            float vx = clamp((baseTexel.x + float(x - 2)) * vp.texelSize.x, minUV.x, maxUV.x);

            float3 tapColor = max(tex2Dlod(historyTex, float4(vx, vy, 0.0, 0.0)).rgb, 0.0);
            color += tapColor * (wX * wY);

            tapMin = min(tapMin, tapColor);
            tapMax = max(tapMax, tapColor);
        }
    }

    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

float3 SampleHistoryColorCatmullRom5Tap(float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;
    float2 fracSq     = fracOffset * fracOffset;

    // Catmull-Rom weights per axis (w0..w3 for the 4 taps at offsets -1,0,+1,+2).
    float2 w0 = fracOffset * (fracOffset * (-0.5 * fracOffset + 1.0) - 0.5);
    float2 w1 = 1.0 + fracSq * (1.5 * fracOffset - 2.5);
    float2 w2 = fracOffset * (fracOffset * (-1.5 * fracOffset + 2.0) + 0.5);
    float2 w3 = fracSq * (0.5 * fracOffset - 0.5);

    // Fuse the two inner taps per axis into a single bilinear-fetchable position.
    float2 w12      = w1 + w2;
    float2 offset12 = w2 / (w12 + 1e-5);

    float2 tc0  = clamp((baseTexel - 1.0) * vp.texelSize, minUV, maxUV);
    float2 tc3  = clamp((baseTexel + 2.0) * vp.texelSize, minUV, maxUV);
    float2 tc12 = clamp((baseTexel + offset12) * vp.texelSize, minUV, maxUV);

    float3 tap0 = max(tex2Dlod(historyTex, float4(tc12.x, tc0.y,  0.0, 0.0)).rgb, 0.0);
    float3 tap1 = max(tex2Dlod(historyTex, float4(tc0.x,  tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap2 = max(tex2Dlod(historyTex, float4(tc12.x, tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap3 = max(tex2Dlod(historyTex, float4(tc3.x,  tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap4 = max(tex2Dlod(historyTex, float4(tc12.x, tc3.y,  0.0, 0.0)).rgb, 0.0);

    float w0y   = w12.x * w0.y;
    float w0x   = w0.x  * w12.y;
    float w12xy = w12.x * w12.y;
    float w3x   = w3.x  * w12.y;
    float w3y   = w12.x * w3.y;

    float3 color = tap0 * w0y + tap1 * w0x + tap2 * w12xy + tap3 * w3x + tap4 * w3y;
    float weightSum = w0y + w0x + w12xy + w3x + w3y;
    color /= max(weightSum, 1e-4);

    float3 tapMin = min(tap0, min(tap1, min(tap2, min(tap3, tap4))));
    float3 tapMax = max(tap0, max(tap1, max(tap2, max(tap3, tap4))));

    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

// ============================================================================
// PIPELINE DATA STRUCTURES
// ============================================================================
// Raw, unmodified 3x3 current-frame samples plus the closest-depth tap scan.
// NOTE: these arrays are deliberately kept RAW (no dilation, no interpolation)
// so the divergence derivatives in ComputeDepthDisocclusionScore stay clean.
struct CurrentFrameNeighborhood
{
    float  depthRaw[9];               // raw depth-buffer values per tap
    float2 velocityJitteredUV[9];     // raw velocities (jittered, undilated, uninterpolated)

    // Closest-depth (foreground) tap scan:
    float  closestDepthRaw;
    float  secondClosestDepthRaw;
    float2 closestOffsetPx;           // texel offset of the closest tap
    float2 secondClosestOffsetPx;
    float2 dilatedVelocityJitteredUV; // velocity of the closest (foreground) tap
};

// Depth-derived surface state around the center pixel.
struct SurfaceEdgeState
{
    bool  isDilationZone;        // center BEHIND neighbors: background behind a foreground crest
    bool  isForegroundEdge;      // center IN FRONT of neighbors: the crest itself
    float edgeEps;               // depth separation that counts as an edge between layers
    float planeGradX;            // central-difference depth plane gradient (per texel)
    float planeGradY;
    float maxNeighborDepthSlope; // largest depth slope across neighbor pairs ("per-2-texel" units)
    float planeDepthNoise;       // depth noise estimate from the max slope
    float depthNoiseFloor;       // lower bound of measurable depth noise
    float depthQuantStep;        // depth quantization step estimate
};

// State-aware surface sample used for reprojection & disocclusion testing.
struct ResolvedSurface
{
    float2 velocityJitteredUV;   // dilated | center | bilinear (per SurfaceEdgeState)
    float  depthRaw;             // depth matching that velocity: closest | center | center
    float2 quadVelocitySpreadUV; // max-min of the jitter-aligned bilinear velocity quad
};

// Foreground crest geometry (feeds the disocclusion tolerances).
struct ForegroundGeometry
{
    float crestDrop;   // depth drop from the closest tap down to the resolved depth
    float slope;       // worst-case depth slope of the foreground crest (per texel)
};

// Reprojection of the resolved surface into the history buffer.
struct HistoryReprojection
{
    float2 sampleUV;            // where to read the history (previous STABLE output)
    float  prevCameraRayY;      // forward component of the previous-frame ray (depth scale)
    float2 subpixelPx;          // sub-texel offset of the history sample
    float  subpixelAlignment;   // 1.0 at texel centers, 0 at texel corners
    float2 jitterResidualPx;    // current jitter minus history subpixel phase
    float2 motionPx;            // history sample -> current output pixel, in pixels
    float  motionMagnitudePx;
    float  motionNormalized;    // saturate(motion / kMotionFullStrengthPx)
    float2 motionDirUnit;       // motion direction, or (1,0) when static
};

// The history landing resolved with the SAME layer semantics as the current
// frame (see SampleHistoryLandingSurface).
struct HistoryLandingSurface
{
    bool  isDilationZone;       // landing acts as the foreground it dilates to
    bool  isForegroundEdge;

    float  effectiveDepthRaw;               // depth of the effective layer
    float2 effectiveVelocityJitteredPrevUV; // prev-frame velocity of the effective layer

    // Velocity-field shape at the landing (px).
    float maxCurvaturePx;       // second difference: ~0 on a continuous field
    float maxPairGradPx;        // local gradient magnitude (anisotropy-safe)
    float coherentGradPx;       // velocity-coherent gradient (alert noise bound)
    float coherentSpreadPx;     // velocity-coherent spread ([FIX 3] layered tolerance)
    float snapDistPx;           // |exact landing - snapped texel center|
};

// Velocity rejection (pursuit) diagnostics.
struct VelocityRejectionResult
{
    bool  rejected;        // pursuit confirmed a divergent surface
    float errorRatio;      // dejittered velocity error / tolerance (>1 triggers pursuit)
    float layerGradientPx; // combined velocity-coherent gradient (current & history), clamped
    float divergencePx;    // measured current-frame divergence at the pursuit landing
};

// Color neighborhood statistics for history clipping.
struct ColorNeighborhoodStats
{
    float3 aabbMin;
    float3 aabbMax;
    float3 mean;                 // weighted mean (mu)
    float3 sigma;                // per-channel weighted stddev
    float3x3 invCov;             // inverse covariance (if validCovariance)
    bool validCovariance;
    float spatialContrast;       // luma extent of the AABB
    float3 expectedJitterShift;  // color shift expected from the sub-texel jitter
    float weights[9];            // tap weights used for the moments
};

// ============================================================================
// LAYER-COHERENT VELOCITY FIELD ANALYSIS (shared by every landing site)
// ----------------------------------------------------------------------------
// The velocity test's NATIVE layer semantics ([FIX 8]): which texels belong to
// the center's layer is decided by velocity coherence, not by depth. This is
// what keeps the test independent from the depth test's failure modes --
// a depth-invisible boundary (grazing lip on the ground, fast object skimming
// a surface) is still a perfectly visible boundary in the VELOCITY field.
// ============================================================================

// Separates "continuous parallax field" (ground, object interiors -- one
// surface's velocity varies smoothly with depth) from "layer boundary" (the
// disocclusion signal itself). The test is the SECOND difference (curvature)
// of the velocity field: a locally linear field has ~zero curvature in every
// direction, while a boundary puts the center texel on ONE layer, so the
// straddling pairs' curvature is ~half the inter-layer velocity step.
// (A first-difference / min-over-pairs test misreads smooth ANISOTROPIC
// gradients -- a strafed ground plane -- as layered, because the pair
// perpendicular to the gradient always measures zero.)
void MeasureVelocityFieldShape(
    float2 velocityJitteredUV[9], float2 sizePixels,
    out float maxCurvaturePx, out float maxPairGradPx)
{
    float2 curvH  = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[3] + velocityJitteredUV[4])) * sizePixels;
    float2 curvV  = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[1] + velocityJitteredUV[2])) * sizePixels;
    float2 curvD1 = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[5] + velocityJitteredUV[8])) * sizePixels;
    float2 curvD2 = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[6] + velocityJitteredUV[7])) * sizePixels;
    maxCurvaturePx = max(max(length(curvH), length(curvV)),
                         max(length(curvD1), length(curvD2)));

    // Opposite-pair first differences, per texel (diagonal pairs span
    // 2*sqrt(2) texels). The max over pair directions is the local gradient
    // magnitude, safe for anisotropic fields.
    float gradX  = length((velocityJitteredUV[4] - velocityJitteredUV[3]) * sizePixels) * 0.5;
    float gradY  = length((velocityJitteredUV[2] - velocityJitteredUV[1]) * sizePixels) * 0.5;
    float gradD1 = length((velocityJitteredUV[8] - velocityJitteredUV[5]) * sizePixels) * (0.25 * kSqrt2);
    float gradD2 = length((velocityJitteredUV[7] - velocityJitteredUV[6]) * sizePixels) * (0.25 * kSqrt2);
    maxPairGradPx = max(max(gradX, gradY), max(gradD1, gradD2));
}

// Continuous (locally linear) velocity field: the smooth-surface correction
// below is granted ONLY here. A layer boundary keeps the raw signal.
bool IsContinuousVelocityField(float maxCurvaturePx, float maxPairGradPx)
{
    return (maxCurvaturePx <= kVelDiscontinuityRatio * maxPairGradPx + kVelDiscontinuityAbsPx);
}

// Max velocity change per pixel between the anchor and taps that move together
// with it (|delta| within the coherence radius). Cross-surface velocity
// differences -- the disocclusion signal itself -- are excluded, so the noise
// estimate cannot eat the signal on similar-depth occlusions. The radius is
// tied to taaVelRejection: relative motion below the alert threshold is noise
// we tolerate; above it, it is the signal.
float MeasureVelocityCoherentGradientPx(
    float2 anchorVelocityJitteredUV,
    float2 neighborVelocityJitteredUV[9],
    float2 sizePixels,
    float  coherenceRadiusPx)
{
    float gradientMax = 0.0;
    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float2 deltaPx = (neighborVelocityJitteredUV[i] - anchorVelocityJitteredUV) * sizePixels;
        if (length(deltaPx) <= coherenceRadiusPx)
        {
            float distPx = max(length(kOffsets3x3[i]), 1e-3);
            gradientMax = max(gradientMax, length(deltaPx) / distPx);
        }
    }
    return gradientMax;
}

// The 2x2 quad the sub-texel phase points into (shared by the current frame's
// jitter phase and every landing's sub-texel phase).
void SelectPhaseQuad(
    float2 velocityJitteredUV[9], float2 phasePx,
    out float2 v00, out float2 v10, out float2 v01, out float2 v11)
{
    v00 = velocityJitteredUV[0];
    v10 = (phasePx.x >= 0.0) ? velocityJitteredUV[4] : velocityJitteredUV[3];
    v01 = (phasePx.y >= 0.0) ? velocityJitteredUV[2] : velocityJitteredUV[1];
    v11 = (phasePx.x >= 0.0)
        ? ((phasePx.y >= 0.0) ? velocityJitteredUV[8] : velocityJitteredUV[6])
        : ((phasePx.y >= 0.0) ? velocityJitteredUV[7] : velocityJitteredUV[5]);
}

// Max pairwise velocity step inside a 2x2 quad, in px.
float QuadVelocityStepPx(float2 v00, float2 v10, float2 v01, float2 v11, float2 sizePixels)
{
    float2 d10 = (v10 - v00) * sizePixels;
    float2 d01 = (v01 - v00) * sizePixels;
    float2 d11 = (v11 - v00) * sizePixels;
    float2 d1x = (v11 - v10) * sizePixels;
    float2 dx1 = (v11 - v01) * sizePixels;
    float2 dxx = (v10 - v01) * sizePixels;
    return max(max(length(d10), length(d01)),
               max(max(length(d11), length(d1x)), max(length(dx1), length(dxx))));
}

// Does the phase-selected 2x2 quad straddle a velocity layer step? Two
// conditions, BOTH required:
//   (a) the 3x3 velocity field is DISCONTINUOUS here (raw second differences
//       vs pair gradients): a locally LINEAR field of ANY gradient is exactly
//       reproducible by the bilinear, so magnitude alone must never commit
//       (this is what keeps steep fast ground on the bilinear path even when
//       the field's own gradient exceeds the coherence radius);
//   (b) the step is INSIDE the selected quad and beyond what the center's own
//       (velocity-coherent) layer explains. The bound uses ONLY the coherent
//       gradient -- including the step itself would scale the threshold with
//       the signal ([FIX 3] failure mode).
bool QuadStraddlesVelocityStep(
    float2 v00, float2 v10, float2 v01, float2 v11,
    float2 velocityJitteredUV[9], float2 sizePixels, float coherenceRadiusPx)
{
    float maxCurvaturePx, maxPairGradPx;
    MeasureVelocityFieldShape(velocityJitteredUV, sizePixels, maxCurvaturePx, maxPairGradPx);
    if (IsContinuousVelocityField(maxCurvaturePx, maxPairGradPx))
        return false;

    float quadStepPx     = QuadVelocityStepPx(v00, v10, v01, v11, sizePixels);
    float coherentGradPx = MeasureVelocityCoherentGradientPx(
        velocityJitteredUV[0], velocityJitteredUV, sizePixels, coherenceRadiusPx);
    return quadStepPx > kQuadStepGradMul * coherentGradPx + coherenceRadiusPx;
}

// Bilinear velocity at the exact sub-texel phase -- unless the selected quad
// straddles a velocity layer step, in which case the pixel keeps its OWN
// rendered layer (the center tap). This is the velocity test's native layer
// semantics ([FIX 8a]): a depth-INVISIBLE boundary collapses the depth gap
// below edgeEps exactly where the velocity test must take over from depth,
// and a blend there would average the inter-layer step -- the disocclusion
// signal itself -- toward zero for every pixel whose two frame-side blend
// weight sets happen to agree. Steps below the coherence floor keep the
// bilinear: their blending error is below the alert tolerance by construction
// (SNR floor, degrades gracefully).
float2 SelectLayerAwareQuadVelocity(float2 velocityJitteredUV[9], float2 phasePx, float2 sizePixels)
{
    float2 v00, v10, v01, v11;
    SelectPhaseQuad(velocityJitteredUV, phasePx, v00, v10, v01, v11);

    float coherenceRadiusPx = max(taaVelRejection, kMinVelCoherenceRadiusPx);
    if (QuadStraddlesVelocityStep(v00, v10, v01, v11, velocityJitteredUV, sizePixels, coherenceRadiusPx))
        return velocityJitteredUV[0];   // the pixel's own rendered layer

    return Bilerp2x2(v00, v10, v01, v11, abs(phasePx));
}

// ============================================================================
// CURRENT-FRAME GATHER & SURFACE ANALYSIS
// ============================================================================
CurrentFrameNeighborhood GatherCurrentFrameNeighborhood(
    float2 snappedUV,
    float2 tapUVs[9],
    float  centerDepthRaw,
    float2 centerVelocityJitteredUV,
    bool   fetchNeighborDepth,
    bool   fetchNeighborVelocity)
{
    CurrentFrameNeighborhood n;

    n.depthRaw[0]           = centerDepthRaw;
    n.velocityJitteredUV[0] = centerVelocityJitteredUV;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        // When neighbor fetches are disabled the center values are replicated so
        // everything downstream sees a perfectly flat neighborhood.
        float  depth = centerDepthRaw;
        if (fetchNeighborDepth)    { depth = tex2Dlod(depthTex,    float4(tapUVs[i], 0.0, 0.0)).r; }
        float2 velocity = centerVelocityJitteredUV;
        if (fetchNeighborVelocity) { velocity = tex2Dlod(velocityTex, float4(tapUVs[i], 0.0, 0.0)).rg; }

        n.depthRaw[i]           = depth;
        n.velocityJitteredUV[i] = velocity;
    }

    // Closest-depth (foreground) tap scan. "Closest" = largest raw depth
    // (reverse-Z: raw = 1/linear, larger raw = nearer surface).
    n.closestDepthRaw       = centerDepthRaw;
    n.secondClosestDepthRaw = 0.0;
    n.closestOffsetPx       = float2(0.0, 0.0);
    n.secondClosestOffsetPx = float2(0.0, 0.0);

    int closestIdx = 0;
    if (fetchNeighborDepth)
    {
        [unroll]
        for (int i = 1; i < 9; ++i)
        {
            float depth = n.depthRaw[i];
            if (depth > n.closestDepthRaw)
            {
                n.secondClosestDepthRaw = n.closestDepthRaw;
                n.secondClosestOffsetPx = n.closestOffsetPx;
                n.closestDepthRaw       = depth;
                n.closestOffsetPx       = kOffsets3x3[i];
                closestIdx = i;
            }
            else if (depth > n.secondClosestDepthRaw)
            {
                n.secondClosestDepthRaw = depth;
                n.secondClosestOffsetPx = kOffsets3x3[i];
            }
        }
    }

    // The foreground layer's true velocity (only consumed on dilation zones).
    n.dilatedVelocityJitteredUV = n.velocityJitteredUV[closestIdx];
    return n;
}

// Core classifier on a raw-depth 3x3, so the HISTORY landing and the PURSUIT
// landing are classified with EXACTLY the same rules as the current frame.
// Without a shared classifier, "foreground reprojecting into its own
// previous-frame dilation zone" can never validate consistently.
SurfaceEdgeState AnalyzeSurfaceEdgesCore(float depthRaw[9], bool useDepthDilation)
{
    SurfaceEdgeState s;
    float centerDepth = depthRaw[0];

    // Center-minus-neighbor-pair second differences ("curvature"): they sign
    // the center's position relative to its neighborhood.
    float curvatureH  = centerDepth - 0.5 * (depthRaw[3] + depthRaw[4]);
    float curvatureV  = centerDepth - 0.5 * (depthRaw[1] + depthRaw[2]);
    float curvatureD1 = centerDepth - 0.5 * (depthRaw[5] + depthRaw[8]);
    float curvatureD2 = centerDepth - 0.5 * (depthRaw[6] + depthRaw[7]);
    float minCurvature = min(min(curvatureH, curvatureV), min(curvatureD1, curvatureD2));
    float maxCurvature = max(max(curvatureH, curvatureV), max(curvatureD1, curvatureD2));

    // Largest slope between neighbor pairs (diagonals rescaled to consistent
    // "per-2-texel" units; thresholds tuned around this scale).
    s.maxNeighborDepthSlope = max(
        max(abs(depthRaw[4] - depthRaw[3]),
            abs(depthRaw[2] - depthRaw[1])),
        max(abs(depthRaw[8] - depthRaw[5]),
            abs(depthRaw[6] - depthRaw[7])) * 0.7071);
    s.planeDepthNoise = s.maxNeighborDepthSlope * 0.05;

    // Depth separation that counts as an edge between two layers.
    s.edgeEps = max(max(centerDepth, 1e-6) * 0.008, s.planeDepthNoise) + 1e-6;

    // Center behind its neighbors -> background behind a foreground crest:
    // velocity/depth must come from the foreground layer (dilation).
    s.isDilationZone = useDepthDilation && (minCurvature < -s.edgeEps);
    // Center in front of its neighbors -> the crest itself: the center tap is
    // the most representative sample.
    s.isForegroundEdge = !s.isDilationZone && (maxCurvature > s.edgeEps);

    // Central-difference plane fit of the depth field around the center.
    s.planeGradX = 0.5 * (depthRaw[4] - depthRaw[3]);
    s.planeGradY = 0.5 * (depthRaw[2] - depthRaw[1]);

    // Lower bound of measurable depth noise.
    float minAbsCurvature = min(min(abs(curvatureH), abs(curvatureV)),
                                min(abs(curvatureD1), abs(curvatureD2)));
    s.depthNoiseFloor = max(minAbsCurvature, s.planeDepthNoise);

    // Depth quantization step estimate: smallest nonzero curvature (0 if none).
    float minNonZeroCurvature = 1.0;
    if (abs(curvatureH)  > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureH));
    if (abs(curvatureV)  > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureV));
    if (abs(curvatureD1) > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureD1));
    if (abs(curvatureD2) > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureD2));
    s.depthQuantStep = (minNonZeroCurvature < 1.0) ? minNonZeroCurvature : 0.0;

    return s;
}

SurfaceEdgeState AnalyzeSurfaceEdges(CurrentFrameNeighborhood neighborhood, bool useDepthDilation)
{
    return AnalyzeSurfaceEdgesCore(neighborhood.depthRaw, useDepthDilation);
}

// Pick the velocity/depth pair that represents the surface at this pixel.
// The raw neighbor arrays stay untouched so the divergence derivatives in the
// depth disocclusion test remain mathematically clean.
ResolvedSurface ResolveSurfaceVelocity(
    CurrentFrameNeighborhood neighborhood, SurfaceEdgeState edge, float2 jitterPx, float2 sizePixels)
{
    ResolvedSurface s;

    // 2x2 velocity quad the sub-texel jitter points into (for bilinear use).
    float2 v00, v10, v01, v11;
    SelectPhaseQuad(neighborhood.velocityJitteredUV, jitterPx, v00, v10, v01, v11);

    // Velocity spread across the quad -> noise estimate for the tests.
    s.quadVelocitySpreadUV = max(max(v00, v10), max(v01, v11))
                           - min(min(v00, v10), min(v01, v11));

    if (edge.isDilationZone)
    {
        // Background behind a foreground crest: use the foreground layer's motion.
        s.velocityJitteredUV = neighborhood.dilatedVelocityJitteredUV;
        s.depthRaw           = neighborhood.closestDepthRaw;
    }
    else if (edge.isForegroundEdge)
    {
        // The crest itself: the center sample is the truth.
        s.velocityJitteredUV = neighborhood.velocityJitteredUV[0];
        s.depthRaw           = neighborhood.depthRaw[0];
    }
    else
    {
        // Flat area: bilinear velocity keeps sub-pixel accuracy -- unless the
        // jitter quad straddles a velocity layer step, in which case the pixel
        // keeps its OWN rendered layer ([FIX 8a]).
        s.velocityJitteredUV = SelectLayerAwareQuadVelocity(
            neighborhood.velocityJitteredUV, jitterPx, sizePixels);
        s.depthRaw           = neighborhood.depthRaw[0];
    }
    return s;
}

ForegroundGeometry ComputeForegroundGeometry(
    CurrentFrameNeighborhood neighborhood, ResolvedSurface resolved, SurfaceEdgeState edge)
{
    ForegroundGeometry fg;

    // Depth drop from the closest (foreground) tap down to the resolved depth,
    // spread over the texel distance to that tap.
    float crestDrop   = max(neighborhood.closestDepthRaw - resolved.depthRaw, 0.0);
    float crestSpanPx = max(length(neighborhood.closestOffsetPx), 1.0);
    float crestSlope  = crestDrop / crestSpanPx;

    // Slope between the two closest taps if they belong to the same object (a
    // thin foreground edge) rather than two clearly separated layers.
    float layerGap   = neighborhood.closestDepthRaw - neighborhood.secondClosestDepthRaw;
    bool  sameObject = (layerGap < edge.edgeEps * 3.5);
    float gapSpanPx  = max(length(neighborhood.closestOffsetPx - neighborhood.secondClosestOffsetPx), 1.0);
    float layerGapSlope = sameObject ? (max(layerGap, 0.0) / gapSpanPx) : 0.0;

    fg.crestDrop = crestDrop;
    fg.slope     = max(crestSlope, layerGapSlope);
    return fg;
}

// ============================================================================
// HISTORY REPROJECTION
// ============================================================================
HistoryReprojection ReprojectToHistory(
    float2 stableUV,
    float2 jitteredUV,
    float2 currentJitterPx,
    ResolvedSurface resolved,
    CameraBasis previousCamera,
    ViewportParams vp)
{
    HistoryReprojection h;

    // Ray (in the previous camera's basis) of the surface point seen at the
    // jittered position: base ray for jitteredUV plus the jittered velocity
    // applied along the previous camera's screen axes. Canonical projection of
    // that ray lands in the previous STABLE output = the history buffer.
    float3 prevRayBase = jitteredUV.x * previousCamera.rightTanFov
                       + previousCamera.forward
                       - jitteredUV.y * previousCamera.downTanFov;
    float3 prevRay = prevRayBase
                   + resolved.velocityJitteredUV.x * previousCamera.rightTanFov
                   - resolved.velocityJitteredUV.y * previousCamera.downTanFov;

    h.sampleUV       = ProjectRayToStableUV(prevRay, jitteredUV + resolved.velocityJitteredUV);
    h.prevCameraRayY = prevRay.y;

    // Net motion between the history sample and this output pixel.
    h.motionPx          = (h.sampleUV - stableUV) * vp.sizePixels;
    h.motionMagnitudePx = length(h.motionPx);
    h.motionNormalized  = saturate(h.motionMagnitudePx / kMotionFullStrengthPx);
    h.motionDirUnit     = (h.motionMagnitudePx > kMinMotionDirLengthPx)
                        ? normalize(h.motionPx) : float2(1.0, 0.0);

    // Sub-texel phase of the history sample (0 at texel centers).
    float2 historyPixelPos = h.sampleUV * vp.sizePixels;
    h.subpixelPx        = historyPixelPos - (floor(historyPixelPos) + 0.5);
    h.subpixelAlignment = 1.0 - saturate(length(h.subpixelPx) * kSqrt2);

    // Sub-texel jitter still present after snapping both frames: the residual
    // mismatch the disocclusion test has to tolerate.
    h.jitterResidualPx = currentJitterPx - h.subpixelPx;
    return h;
}

// ============================================================================
// VELOCITY-FIELD JACOBIAN & ADVECTION BUDGET
// ----------------------------------------------------------------------------
// Layer-consistent derivative estimators ([FIX 8b]). Raw central differences
// at a straddled neighborhood are HALF the inter-layer velocity step, which
// both poisoned the [FIX 6] advection prediction and made the continuity test
// misread silhouettes as curvature. On an all-coherent neighborhood these are
// bit-identical to the raw estimators, so the [FIX 4]/[FIX 5]/[FIX 6] ground
// cases are untouched.
// ============================================================================

// 2x2 Jacobian of the velocity field in px-per-texel (rows: d(vx,vy)/d(x,y)),
// from layer-consistent differences: central when both opposite taps are
// coherent with the center layer, one-sided when only one is, zero when
// neither is. The jitter content is sub-pixel and cancels in the difference.
float4 EstimateVelocityJacobianCoherentPx(float2 v[9], float2 sizePixels, float coherenceRadiusPx)
{
    float2 c3 = (v[3] - v[0]) * sizePixels;
    float2 c4 = (v[4] - v[0]) * sizePixels;
    float2 c1 = (v[1] - v[0]) * sizePixels;
    float2 c2 = (v[2] - v[0]) * sizePixels;
    bool leftOk  = (length(c3) <= coherenceRadiusPx);
    bool rightOk = (length(c4) <= coherenceRadiusPx);
    bool upOk    = (length(c1) <= coherenceRadiusPx);
    bool downOk  = (length(c2) <= coherenceRadiusPx);

    float4 J = float4(0.0, 0.0, 0.0, 0.0);
    if      (rightOk && leftOk)  { float2 d = (c4 - c3) * 0.5; J.x = d.x; J.z = d.y; }
    else if (rightOk)            { J.x = c4.x;  J.z = c4.y;  }
    else if (leftOk)             { J.x = -c3.x; J.z = -c3.y; }
    if      (downOk && upOk)     { float2 d = (c2 - c1) * 0.5; J.y = d.x; J.w = d.y; }
    else if (downOk)             { J.y = c2.x;  J.w = c2.y;  }
    else if (upOk)               { J.y = -c1.x; J.w = -c1.y; }
    return J;
}

float2 ApplyVelocityJacobian(float4 J, float2 vecPx)
{
    return float2(dot(J.xy, vecPx), dot(J.zw, vecPx));
}

// Residual (second-order) advection budget: what the first-order Jacobian
// correction cannot absorb -- field curvature along the miss, camera
// acceleration, and Jacobian noise amplified by a large miss. It does NOT
// scale with the transport distance or the error's full magnitude (see
// [FIX 3]); hard-clamped. [FIX 8e]: the miss term is capped at missCapPx --
// the predicted same-surface advection (the HONEST miss scale: on a
// same-surface pixel the dejittered step-1 error IS the advection to first
// order). On a depth-missed disocclusion the raw measured error IS the
// inter-layer step -- the signal itself -- and a budget scaling with it would
// eat the signal.
float ResidualAdvectionBudgetPx(float maxPairGradPx, float missPx, float snapDistPx, float missCapPx)
{
    float boundedMissPx = min(missPx, missCapPx);
    return min(maxPairGradPx * (kVelJacobianResidualFrac * boundedMissPx + snapDistPx + kPursuitSnapPadPx),
               kPursuitMaxAdvectionPx);
}

// Field shape with layer-consistent differences: only pairs whose BOTH taps
// are coherent with the center layer contribute. At a straddled neighborhood
// the raw second difference is half the inter-layer step, which the
// continuity ratio cannot separate from genuine curvature -- so continuities
// measured raw gate the [FIX 6] correction off exactly where dilation zones
// live. A fully isolated center (1-texel sliver) measures a trivially
// continuous field, which is correct: its own layer has no structure.
void MeasureVelocityFieldShapeCoherent(
    float2 v[9], float2 sizePixels, float coherenceRadiusPx,
    out float maxCurvaturePx, out float maxPairGradPx)
{
    bool ok[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
        ok[i] = (i == 0) || (length((v[i] - v[0]) * sizePixels) <= coherenceRadiusPx);

    float2 curvH  = (ok[3] && ok[4]) ? (v[0] - 0.5 * (v[3] + v[4])) * sizePixels : float2(0.0, 0.0);
    float2 curvV  = (ok[1] && ok[2]) ? (v[0] - 0.5 * (v[1] + v[2])) * sizePixels : float2(0.0, 0.0);
    float2 curvD1 = (ok[5] && ok[8]) ? (v[0] - 0.5 * (v[5] + v[8])) * sizePixels : float2(0.0, 0.0);
    float2 curvD2 = (ok[6] && ok[7]) ? (v[0] - 0.5 * (v[6] + v[7])) * sizePixels : float2(0.0, 0.0);
    maxCurvaturePx = max(max(length(curvH), length(curvV)), max(length(curvD1), length(curvD2)));

    float gradX  = (ok[3] && ok[4]) ? length((v[4] - v[3]) * sizePixels) * 0.5 : 0.0;
    float gradY  = (ok[1] && ok[2]) ? length((v[2] - v[1]) * sizePixels) * 0.5 : 0.0;
    float gradD1 = (ok[5] && ok[8]) ? length((v[8] - v[5]) * sizePixels) * (0.25 * kSqrt2) : 0.0;
    float gradD2 = (ok[6] && ok[7]) ? length((v[7] - v[6]) * sizePixels) * (0.25 * kSqrt2) : 0.0;
    maxPairGradPx = max(max(gradX, gradY), max(gradD1, gradD2));
}

// ============================================================================
// HISTORY LANDING SURFACE (shared by the depth & velocity disocclusion tests)
// ----------------------------------------------------------------------------
// The history landing is resolved with EXACTLY the same layer semantics as the
// current frame:
//   * dilation zone -> the FOREGROUND layer it dilates to. The dilation zone
//     IS that object: "foreground replaced by its own dilation zone" must
//     validate, and "dilation zone replaced by background" must reject -- so
//     the rendered background texel is NOT offered as a candidate.
//   * crest         -> the center tap.
//   * flat          -> bilinear at the exact sub-texel landing position (a
//     locally linear field reproduces exactly, so graded ground reads its OWN
//     velocity instead of a snapped neighbor's) -- but the VELOCITY commits to
//     the rendered layer when the quad straddles a velocity step ([FIX 8a]).
// ============================================================================
HistoryLandingSurface SampleHistoryLandingSurface(
    float2 historyUV, ViewportParams vp, bool useDepthDilation, bool gather)
{
    HistoryLandingSurface h;
    h.isDilationZone   = false;
    h.isForegroundEdge = false;
    h.maxCurvaturePx   = 0.0;
    h.maxPairGradPx    = 0.0;
    h.coherentGradPx   = 0.0;
    h.coherentSpreadPx = 0.0;
    h.snapDistPx       = 0.0;

    // Minimal default when no disocclusion test needs the landing: the single
    // nearest tap (consumed only by the debug views).
    float2 clampedUV  = clamp(historyUV, vp.minUV, vp.maxUV);
    h.effectiveDepthRaw               = tex2Dlod(historyTex,      float4(clampedUV, 0.0, 0.0)).a;
    h.effectiveVelocityJitteredPrevUV = tex2Dlod(prevVelocityTex, float4(clampedUV, 0.0, 0.0)).rg;
    if (!gather)
        return h;

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;    // sub-texel landing phase, [-0.5, 0.5]
    h.snapDistPx     = length(fracPx);

    float  depths[9];
    float2 velocities[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapUV = clamp(snappedUV + kOffsets3x3[i] * vp.texelSize, vp.minUV, vp.maxUV);
        depths[i]     = tex2Dlod(historyTex,      float4(tapUV, 0.0, 0.0)).a;
        velocities[i] = tex2Dlod(prevVelocityTex, float4(tapUV, 0.0, 0.0)).rg;
    }

    // Same classifier as the current frame. The history alpha channel stores
    // the previous frame's RAW rendered depth (never resampled), so the depth
    // structure at the landing -- crest, dilation zone, flat -- is classifiable
    // identically to the current frame.
    SurfaceEdgeState edge = AnalyzeSurfaceEdgesCore(depths, useDepthDilation);
    h.isDilationZone   = edge.isDilationZone;
    h.isForegroundEdge = edge.isForegroundEdge;

    // Closest tap = the foreground a dilation zone dilates to (reverse-Z:
    // larger raw = nearer).
    float closestDepth = depths[0];
    int   closestIdx   = 0;
    [unroll]
    for (int j = 1; j < 9; ++j)
    {
        if (depths[j] > closestDepth) { closestDepth = depths[j]; closestIdx = j; }
    }

    // Effective-layer resolution (mirrors ResolveSurfaceVelocity).
    if (h.isDilationZone)
    {
        h.effectiveDepthRaw               = closestDepth;
        h.effectiveVelocityJitteredPrevUV = velocities[closestIdx];
    }
    else if (h.isForegroundEdge)
    {
        h.effectiveDepthRaw               = depths[0];
        h.effectiveVelocityJitteredPrevUV = velocities[0];
    }
    else
    {
        float d00 = depths[0];
        float d10 = (fracPx.x >= 0.0) ? depths[4] : depths[3];
        float d01 = (fracPx.y >= 0.0) ? depths[2] : depths[1];
        float d11 = (fracPx.x >= 0.0)
            ? ((fracPx.y >= 0.0) ? depths[8] : depths[6])
            : ((fracPx.y >= 0.0) ? depths[7] : depths[5]);

        // Depth stays bilinear: at a depth-invisible boundary the two layers'
        // depths are near-equal BY DEFINITION, so there is nothing to commit
        // to. The VELOCITY commits to the rendered layer when the quad
        // straddles a step ([FIX 8a]) -- the signal lives there, not in depth.
        h.effectiveDepthRaw               = Bilerp2x2(d00, d10, d01, d11, abs(fracPx));
        h.effectiveVelocityJitteredPrevUV = SelectLayerAwareQuadVelocity(velocities, fracPx, vp.sizePixels);
    }

    // Field shape + velocity-coherent noise, anchored at the effective layer.
    // [FIX 8b]: the shape is measured with layer-consistent differences, so a
    // straddled landing measures its OWN layer's continuity instead of half
    // the inter-layer step.
    float coherenceRadiusPx = max(taaVelRejection, kMinVelCoherenceRadiusPx);
    MeasureVelocityFieldShapeCoherent(velocities, vp.sizePixels, coherenceRadiusPx, h.maxCurvaturePx, h.maxPairGradPx);

    [unroll]
    for (int m = 0; m < 9; ++m)
    {
        float2 deltaPx = (velocities[m] - h.effectiveVelocityJitteredPrevUV) * vp.sizePixels;
        float  lenPx   = length(deltaPx);
        if (lenPx <= coherenceRadiusPx)
        {
            h.coherentSpreadPx = max(h.coherentSpreadPx, lenPx);
            float distPx = max(length(kOffsets3x3[m]), 1.0);
            h.coherentGradPx = max(h.coherentGradPx, lenPx / distPx);
        }
    }
    return h;
}

// ============================================================================
// COLOR NEIGHBORHOOD STATISTICS
// ============================================================================
ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized, float2 jitterPx)
{
    ColorNeighborhoodStats stats;
    stats.validCovariance = false;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    bool covarianceEnabled    = (taaUseCovarianceClipping > 0.5) && (taaUseKDopClipping <= 0.5);
    bool jitterPaddingEnabled = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 weightedSum   = float3(0.0, 0.0, 0.0);   // first moment accumulator
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);   // second moment accumulator
    float3 weightedCross = float3(0.0, 0.0, 0.0);   // cross moments (xy, xz, yz)
    float  totalWeight   = 0.0;
    float  motionFactor  = saturate(motionNormalized);

    // Optionally center the weighting kernel on the jittered position instead
    // of the snapped texel center.
    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        stats.aabbMin = min(stats.aabbMin, tapColorSpace);
        stats.aabbMax = max(stats.aabbMax, tapColorSpace);

        // Distance of the tap from the (optionally jitter-shifted) kernel center.
        float2 offsetFromCenterPx = tapOffsetPx - weightCenterPx;
        float w = jitterCenteredWeights
            ? exp2(-dot(offsetFromCenterPx, offsetFromCenterPx) * kLog2E)
            : kStdWeights[i];

        // Down-weight dark (noisier) samples.
        if (taaLumaVariance > 0.5) { w *= (1.0 / (1.0 + max(tapColorSpace.x, 0.0))); }
        // Prefer taps along the motion direction while moving.
        if (taaVelocityAlignedVariance > 0.5 && i > 0)
        {
            w *= lerp(1.0, saturate(dot(tapOffsetPx, motionDirUnit) * kInvLength[i] * 0.5 + 0.5), motionFactor);
        }

        stats.weights[i] = w;
        weightedSum   += tapColorSpace * w;
        weightedSumSq += tapColorSpace * tapColorSpace * w;
        if (covarianceEnabled)
        {
            weightedCross += float3(tapColorSpace.x * tapColorSpace.y,
                                    tapColorSpace.x * tapColorSpace.z,
                                    tapColorSpace.y * tapColorSpace.z) * w;
        }
        totalWeight += w;
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean  = weightedSum * invTotalWeight;
    stats.sigma = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));

    // Firefly clamp: pull the AABB bounds into the mean +/- k*sigma band so a
    // single outlier cannot dominate the clip box. Clamping is monotonic, so
    // aabbMin <= aabbMax still holds and the mean (a positive-weight average of
    // the taps) stays inside the box.
    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

    // Color shift expected from the remaining sub-texel jitter (one-sided
    // gradients taken in the jitter direction).
    stats.expectedJitterShift = float3(0.0, 0.0, 0.0);
    if (jitterPaddingEnabled)
    {
        float3 gradX = jitterPx.x > 0.0 ? (neighborhoodColorSpace[4] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[3] - neighborhoodColorSpace[0]);
        float3 gradY = jitterPx.y > 0.0 ? (neighborhoodColorSpace[2] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[1] - neighborhoodColorSpace[0]);
        stats.expectedJitterShift = (gradX * abs(jitterPx.x)) + (gradY * abs(jitterPx.y));
    }

    if (covarianceEnabled)
    {
        float3x3 cov = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        cov[0][0] = weightedSumSq.x * invTotalWeight - stats.mean.x * stats.mean.x;
        cov[1][1] = weightedSumSq.y * invTotalWeight - stats.mean.y * stats.mean.y;
        cov[2][2] = weightedSumSq.z * invTotalWeight - stats.mean.z * stats.mean.z;
        cov[0][1] = weightedCross.x * invTotalWeight - stats.mean.x * stats.mean.y;
        cov[0][2] = weightedCross.y * invTotalWeight - stats.mean.x * stats.mean.z;
        cov[1][2] = weightedCross.z * invTotalWeight - stats.mean.y * stats.mean.z;

        cov[0][0] += kEpsilon; cov[1][1] += kEpsilon; cov[2][2] += kEpsilon;

        // Inflate the covariance by the expected jitter-induced color shift so
        // legit jitter shimmer does not get clipped away (flicker padding).
        if (jitterPaddingEnabled)
        {
            float paddingFade   = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - motionNormalized) : 1.0;
            float paddingAmount = taaJitterFlickerPadding * paddingFade;

            if (taaDirectionalVariance > 0.5)
            {
                // Anisotropic: pad along the expected shift direction.
                float3 directionalPad = stats.expectedJitterShift * paddingAmount;
                cov[0][0] += directionalPad.x * directionalPad.x;
                cov[1][1] += directionalPad.y * directionalPad.y;
                cov[2][2] += directionalPad.z * directionalPad.z;
                cov[0][1] += directionalPad.x * directionalPad.y;
                cov[0][2] += directionalPad.x * directionalPad.z;
                cov[1][2] += directionalPad.y * directionalPad.z;
            }
            else
            {
                // Isotropic: pad by local contrast * jitter magnitude.
                float padAmount = stats.spatialContrast * length(jitterPx) * paddingAmount;
                cov[0][0] += padAmount * padAmount;
                cov[1][1] += padAmount * padAmount;
                cov[2][2] += padAmount * padAmount;
            }
        }

        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        stats.invCov = InverseSymmetric3x3(cov, stats.validCovariance);
    }

    // Fallback when no valid covariance: add the padding to per-channel sigma.
    if (!stats.validCovariance && jitterPaddingEnabled)
    {
        float paddingFade   = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - motionNormalized) : 1.0;
        float paddingAmount = taaJitterFlickerPadding * paddingFade;

        if (taaDirectionalVariance > 0.5) { stats.sigma += abs(stats.expectedJitterShift * paddingAmount); }
        else                              { stats.sigma += (stats.spatialContrast * length(jitterPx) * paddingAmount); }
    }

    stats.sigma = max(stats.sigma, kMinSigma);
    return stats;
}

// ============================================================================
// HISTORY CLIPPING
// ============================================================================
// Clip the history->target ray to the box. If the history sits inside the box
// it is returned unchanged; otherwise it is moved to the first box entry point
// (optionally softened). The target (the neighborhood mean) is provably inside
// every box variant used below, so the entry point always exists in (0,1).
float3 ClipRayToBox(float3 history, float3 target, float3 boxMin, float3 boxMax, float softClipAmount, float motionFactor)
{
    float3 boxCenter = 0.5 * (boxMax + boxMin);
    float3 boxExtent = max(0.5 * (boxMax - boxMin), kEpsilon);

    // How far outside the box the history is, in box-size units.
    float3 historyUnit = (history - boxCenter) / boxExtent;
    float overshootUnits = max(abs(historyUnit).x, max(abs(historyUnit).y, abs(historyUnit).z));

    if (overshootUnits <= 1.0)
        return history;

    float3 rayDir = target - history;

    float2 slabX = RaySlabInterval(history.x, rayDir.x, float2(boxMin.x, boxMax.x));
    float2 slabY = RaySlabInterval(history.y, rayDir.y, float2(boxMin.y, boxMax.y));
    float2 slabZ = RaySlabInterval(history.z, rayDir.z, float2(boxMin.z, boxMax.z));
    float entryT = saturate(max(max(slabX.x, slabY.x), slabZ.x));

    float3 clipped = history + rayDir * entryT;

    if (softClipAmount > 0.0)
    {
        float softScale = SoftClipUnitScale(overshootUnits, softClipAmount, motionFactor);
        clipped = lerp(history, clipped, 1.0 / max(softScale, 1.0));
    }
    return clipped;
}

// Clip the history color into the current color neighborhood. Three mutually
// exclusive strategies (by config):
//   1. 16-axis k-DOP slab clipping (optionally variance-extended slabs)
//   2. covariance (Mahalanobis) ellipsoid clip, then box ray clip
//   3. per-channel variance box, then box ray clip
float3 ClipHistoryToNeighborhood(
    float3 historyColorSpace, ColorNeighborhoodStats stats,
    float motionNormalized, float dynamicGamma,
    float3 neighborhoodColorSpace[9], float3 clipMargin)
{
    float motionFactor = saturate(motionNormalized);

    if (taaUseKDopClipping > 0.5)
    {
        float3 rayOrigin = stats.mean;
        float3 rayDir    = historyColorSpace - rayOrigin;

        float nearHit = -kLargeValue;
        float farHit  =  kLargeValue;

        [unroll]
        for (int a = 0; a < 16; ++a)
        {
            float3 axis = kDopAxes[a];
            float centerProjection = dot(rayOrigin, axis);
            float2 slab;

            if (taaKDopVariance > 0.5)
            {
                // Weighted mean/std of the tap projections along this axis.
                float2 moments = float2(0.0, 0.0);
                float  weightSum = 0.0;
                float  projectionMin = kLargeValue;
                float  projectionMax = -kLargeValue;

                [unroll]
                for (int n = 0; n < 9; ++n)
                {
                    float projection = dot(neighborhoodColorSpace[n], axis);
                    float w = stats.weights[n];
                    moments += float2(projection, projection * projection) * w;
                    weightSum += w;
                    projectionMin = min(projectionMin, projection);
                    projectionMax = max(projectionMax, projection);
                }
                moments /= max(weightSum, kEpsilon);

                float mean          = moments.x;
                float sigma         = sqrt(max(moments.y - mean * mean, 0.0));
                float extendedSigma = sigma * dynamicGamma;

                slab = float2(mean - extendedSigma, mean + extendedSigma);

                float axisMargin = taaClipOvershoot * max(projectionMax - projectionMin, kMinFootprintRange);
                slab += float2(-axisMargin, axisMargin);
            }
            else
            {
                slab = float2(kLargeValue, -kLargeValue);
                [unroll]
                for (int n = 0; n < 9; ++n)
                {
                    float projection = dot(neighborhoodColorSpace[n], axis);
                    slab.x = min(projection, slab.x);
                    slab.y = max(projection, slab.y);
                }

                float axisMargin = taaClipOvershoot * max(slab.y - slab.x, kMinFootprintRange) + kEpsilon;
                slab += float2(-axisMargin, axisMargin);
            }

            float2 interval = RaySlabInterval(centerProjection, dot(rayDir, axis), slab);
            nearHit = max(nearHit, interval.x);
            farHit  = min(farHit,  interval.y);
        }

        // The ray origin (the mean) is always inside the k-DOP, so a history
        // outside it exits at farHit in (0,1); farHit >= 1 means history inside.
        if (nearHit <= farHit && (nearHit > 0.0 || farHit > 0.0))
        {
            float tHit = clamp(nearHit > 0.0 ? nearHit : farHit, 0.0, 1.0);

            if (tHit < 1.0)
            {
                if (taaSoftClip > 0.0)
                {
                    float overshootUnits = 1.0 / max(tHit, kEpsilon);
                    float softScale = SoftClipUnitScale(overshootUnits, taaSoftClip, motionFactor);
                    return rayOrigin + rayDir * (softScale / overshootUnits);
                }
                return rayOrigin + tHit * rayDir;
            }
            return historyColorSpace;
        }

        return historyColorSpace;
    }

    if (stats.validCovariance)
    {
        // Mahalanobis ellipsoid clip first, then the AABB as a hard backstop.
        float3 diff = historyColorSpace - stats.mean;
        float mahalanobisSq = dot(diff, mul(stats.invCov, diff));
        float gammaSq = dynamicGamma * dynamicGamma;
        float3 ellipsoidClipped = (mahalanobisSq > gammaSq && mahalanobisSq > kEpsilon)
            ? (stats.mean + diff * (dynamicGamma / sqrt(max(mahalanobisSq, kEpsilon))))
            : historyColorSpace;
        return ClipRayToBox(ellipsoidClipped, stats.mean, stats.aabbMin - clipMargin, stats.aabbMax + clipMargin, taaSoftClip, motionFactor);
    }

    // Per-channel variance box (chroma channels get their own scale).
    float3 chromaScale     = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 varianceExtents = stats.sigma * dynamicGamma * chromaScale;
    float3 boxMin = max(stats.mean - varianceExtents, stats.aabbMin - clipMargin);
    float3 boxMax = min(stats.mean + varianceExtents, stats.aabbMax + clipMargin);

    return ClipRayToBox(historyColorSpace, stats.mean, boxMin, boxMax, taaSoftClip, motionFactor);
}

// ============================================================================
// MOTION-COMPENSATED DEPTH DISOCCLUSION
// ----------------------------------------------------------------------------
// Compares the history depth against the current surface depth transported
// ("pursued") into the previous frame, correcting for perspective flow (kappa)
// and velocity divergence. Evaluated with RAW neighbor samples to guarantee
// zero false divergence. Returns a score; >= 1.0 means "history rejected".
// ============================================================================
float ComputeDepthDisocclusionScore(
    float  resolvedDepthRaw,
    float  historyDepthRaw,
    float  neighborDepthRaw[9],
    float2 neighborVelocityJitteredUV[9],
    float2 resolvedVelocityJitteredUV,
    float2 stableUV,
    float2 historySampleUV,
    float  prevCameraRayY,
    float2 jitterResidualPx,
    float2 quadVelocitySpreadUV,
    float  foregroundSlope,
    float  foregroundCrestDrop,
    float2 closestOffsetPx,
    bool   isDilationZone,
    bool   isForegroundEdge,
    float  depthNoiseFloor,
    float  depthQuantStep,
    float  depthPlaneGradX,
    float  depthPlaneGradY,
    float  surfaceLayerEps,
    ViewportParams vp)
{
    if (taaDepthRejection <= 0.001) return 0.0;

    // --- Tangent-plane coordinates of both sample positions -----------------
    float2 curTan  = float2((stableUV.x * 2.0 - 1.0) * taaTanHalfFovX,
                            (1.0 - stableUV.y * 2.0) * taaTanHalfFovY);
    float2 prevTan = float2((historySampleUV.x * 2.0 - 1.0) * taaTanHalfFovX,
                            (1.0 - historySampleUV.y * 2.0) * taaTanHalfFovY);
    float curRadiusSq   = dot(curTan, curTan);
    float curDenom      = 1.0 + curRadiusSq;
    float curRayLength  = sqrt(curDenom);
    float prevRayLength = sqrt(1.0 + dot(prevTan, prevTan));
    float rayScale      = prevRayLength / max(curRayLength, 1e-4);

    // --- Robust one-sided depth gradients (smallest magnitude side) ---------
    float gradLeft  = resolvedDepthRaw - neighborDepthRaw[3];
    float gradRight = neighborDepthRaw[4] - resolvedDepthRaw;
    float gradUp    = resolvedDepthRaw - neighborDepthRaw[1];
    float gradDown  = neighborDepthRaw[2] - resolvedDepthRaw;
    float robustGradX = (abs(gradLeft)  < abs(gradRight)) ? gradLeft : gradRight;
    float robustGradY = (abs(gradUp)    < abs(gradDown))  ? gradUp   : gradDown;

    // Plane gradient in tangent units (depth change per tangent-space unit).
    float tangentStepX = max(2.0 * taaTanHalfFovX * vp.texelSize.x, 1e-7);
    float tangentStepY = max(2.0 * taaTanHalfFovY * vp.texelSize.y, 1e-7);
    float2 planeGradTan = float2(depthPlaneGradX / tangentStepX, depthPlaneGradY / tangentStepY);

    // --- Measured velocity divergence at the center (raw neighbor samples) --
    float2 velocityTan = float2(resolvedVelocityJitteredUV.x * 2.0 * taaTanHalfFovX,
                                -resolvedVelocityJitteredUV.y * 2.0 * taaTanHalfFovY);
    float radialTerm = dot(velocityTan, curTan) / curDenom;

    float divLeft  = neighborVelocityJitteredUV[0].x - neighborVelocityJitteredUV[3].x;
    float divRight = neighborVelocityJitteredUV[4].x - neighborVelocityJitteredUV[0].x;
    float divUp    = neighborVelocityJitteredUV[0].y - neighborVelocityJitteredUV[1].y;
    float divDown  = neighborVelocityJitteredUV[2].y - neighborVelocityJitteredUV[0].y;
    float divergenceX = ((abs(divLeft) < abs(divRight)) ? divLeft : divRight) * vp.sizePixels.x;
    float divergenceY = ((abs(divUp)   < abs(divDown))  ? divUp   : divDown)  * vp.sizePixels.y;
    float divergenceMeasured = divergenceX + divergenceY;

    // --- Velocity spread across the quad -> noise estimates -----------------
    float2 velocitySpreadPx = quadVelocitySpreadUV * vp.sizePixels;
    float velocitySpreadTan = 0.5 * (quadVelocitySpreadUV.x * 2.0 * taaTanHalfFovX +
                                     quadVelocitySpreadUV.y * 2.0 * taaTanHalfFovY);
    float divergenceNoise   = quadVelocitySpreadUV.x * vp.sizePixels.x +
                              quadVelocitySpreadUV.y * vp.sizePixels.y;

    // --- Inter-layer velocity shear: velocity change per unit depth between
    //     the center layer and any clearly separated background layer --------
    float2 layerShear      = float2(0.0, 0.0);
    float  layerSeparation = 0.0;
    bool   haveLayerShear  = false;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float2 offsetPx = kOffsets3x3[i];
        float predictedDepth = resolvedDepthRaw + depthPlaneGradX * offsetPx.x + depthPlaneGradY * offsetPx.y;
        float planeResidual  = neighborDepthRaw[i] - predictedDepth;
        float depthGap       = neighborDepthRaw[i] - resolvedDepthRaw;
        if (abs(planeResidual) > surfaceLayerEps &&
            abs(depthGap) > 0.5 * surfaceLayerEps + 1e-7 &&
            abs(depthGap) > layerSeparation)
        {
            float2 neighborVelocityTan = float2(neighborVelocityJitteredUV[i].x * 2.0 * taaTanHalfFovX,
                                                -neighborVelocityJitteredUV[i].y * 2.0 * taaTanHalfFovY);
            layerShear      = (neighborVelocityTan - velocityTan) / depthGap;
            layerSeparation = abs(depthGap);
            haveLayerShear  = true;
        }
    }

    // Reject the shear estimate when its own noise (velocity spread over the
    // layer separation) is too large to be trusted.
    if (haveLayerShear)
    {
        float shearNoise       = 2.0 * velocitySpreadTan / max(layerSeparation, 1e-7);
        float shearNoiseScaled = resolvedDepthRaw * length(curTan) * shearNoise / curDenom;
        if (shearNoiseScaled > 0.25 * taaDepthRejection) haveLayerShear = false;
    }

    // --- Divergence-to-depth-scale correction (kappa) ------------------------
    // kappa ~ d(ln depth)/dt along the flow; estimated from the measured
    // divergence, refined either by the inter-layer shear or by the depth plane.
    // Sign convention check: forward dolly -> outward divergence -> kappa < 0
    // -> (1-kappa) > 1 -> expected raw depth increases, as it should (reverse-Z).
    float denomNaive = max(2.0 - curRadiusSq, 0.25);
    float kappaNaive = -(divergenceMeasured - 3.0 * radialTerm) / denomNaive;
    float kappa;        // correction factor, clamped below
    float kappaSigma;   // noise bound of the estimate

    if (haveLayerShear)
    {
        float shearTerm = resolvedDepthRaw * dot(curTan, layerShear) / curDenom;
        kappa = shearTerm
              + 0.5 * (-(divergenceMeasured - 3.0 * radialTerm)
                       - dot(layerShear, 3.0 * resolvedDepthRaw * curTan / curDenom - planeGradTan));
        float shearNoise = 2.0 * velocitySpreadTan / max(layerSeparation, 1e-7);
        kappaSigma = divergenceNoise / denomNaive
                   + resolvedDepthRaw * length(curTan) * shearNoise / curDenom
                   + velocitySpreadTan * length(curTan) / max(prevRayLength, 1e-3);
    }
    else
    {
        float denomPlane = max(2.0 - curRadiusSq + curDenom * dot(planeGradTan, curTan) / max(resolvedDepthRaw, 1e-6),
                               0.5 * denomNaive);
        float kappaPlane = -(divergenceMeasured - 3.0 * radialTerm) / denomPlane;
        kappa = kappaPlane * (5.0 - curRadiusSq) / 6.0;
        kappaSigma = divergenceNoise / denomNaive
                   + abs(kappaNaive) * (1.0 + curRadiusSq) / 6.0
                   + abs(kappaPlane - kappaNaive)
                   + velocitySpreadTan * length(curTan) / max(prevRayLength, 1e-3);
    }

    kappa = clamp(kappa, -0.25, 0.25);

    // --- Expected history depth from the transported current surface --------
    float safePrevRayY = (prevCameraRayY > 1e-4) ? prevCameraRayY : 1.0;
    float expectedDepthRaw = (resolvedDepthRaw / safePrevRayY) * rayScale * (1.0 - kappa);
    expectedDepthRaw = max(expectedDepthRaw, 1e-6);

    float invExpected = 1.0 / expectedDepthRaw;
    // Only a history surface CLOSER than the transported current surface counts
    // as a disocclusion (history occluding what we now see).
    float relativeDepthExcess = max(historyDepthRaw - expectedDepthRaw, 0.0) * invExpected;

    float baseThreshold = taaDepthRejection + kappaSigma;

    // --- State-dependent tolerances ------------------------------------------
    float allowances = 0.0;
    if (!isDilationZone && !isForegroundEdge)
    {
        // Flat interior: residual jitter mismatch + local depth curvature.
        float jitterAllowance = abs(robustGradX * jitterResidualPx.x + robustGradY * jitterResidualPx.y);
        float curvatureH = abs(resolvedDepthRaw - 0.5 * (neighborDepthRaw[3] + neighborDepthRaw[4]));
        float curvatureV = abs(resolvedDepthRaw - 0.5 * (neighborDepthRaw[1] + neighborDepthRaw[2]));
        float curvatureAllowance = max(curvatureH, curvatureV);
        allowances = (jitterAllowance + curvatureAllowance) * invExpected;
    }
    else if (isForegroundEdge)
    {
        // The center is the crest: the history footprint may legitimately reach
        // down the front face within the filter support.
        float reachPx = kHistoryTapReachPx
                      + (abs(jitterResidualPx.x) + abs(jitterResidualPx.y))
                      + 0.5 * length(velocitySpreadPx);
        allowances = (foregroundCrestDrop + foregroundSlope * reachPx) * invExpected;
    }
    else // dilation zone: background behind a crest
    {
        float reachPx = length(closestOffsetPx)
                      + kHistoryTapReachPx
                      + length(jitterResidualPx)
                      + 0.5 * length(velocitySpreadPx);
        allowances = (foregroundSlope * reachPx) * invExpected;
    }

    float threshold = baseThreshold + allowances;

    if (depthQuantStep > 0.0)
        threshold += 2.0 * depthQuantStep * (invExpected + 1.0 / max(historyDepthRaw, 1e-6));

    return relativeDepthExcess / max(threshold, 1e-6);
}

// ============================================================================
// PURSUIT CONFIRMATION (current-frame, layer-consistent velocity divergence)
// ----------------------------------------------------------------------------
// Push the historical surface forward along its own (dejittered) motion into
// the CURRENT frame, then measure how much the current-frame velocity field
// diverges between the landing position and the current pixel. Both velocities
// belong to frame t, so camera acceleration and jitter cancel; only a genuine
// surface change (disocclusion) can diverge.
//
// The landing is resolved with the SAME layer semantics as everywhere else
// (dilation zone -> the foreground it dilates to, crest -> center, flat ->
// bilinear at the exact sub-texel position, committing to the rendered layer
// when the quad straddles a velocity step [FIX 8a]).
//
// Tolerance: [FIX 3] noise floor (the landing's velocity-coherent spread) PLUS
// the second-order residual of the [FIX 6] first-order Jacobian correction,
// which itself is gated by (a) a continuous landing velocity field ([FIX 8b]:
// measured with layer-consistent differences) and (b) the kappa-corrected
// depth transport confirming the same surface. The single-layer landing
// requirement was REMOVED ([FIX 8d]): a dilation-zone or crest landing
// resolves to its effective (foreground) layer, so its divergence carries the
// SAME same-surface advection the correction cancels -- and the correction is
// self-discriminating, so a cross-surface divergence GROWS through it
// ((I+J) amplification; cancellation would require J ~= -I). Neither term
// ever scales with the transport distance or the raw divergence (both grow as
// fast as the disocclusion signal itself; see [FIX 3]).
// ============================================================================
bool PursuitConfirmsDivergence(
    float2 historySampleUV,
    float2 prevVelocityEffectiveJitteredUV,
    float2 curVelocityJitteredUV,
    float2 jitterTransportUV,
    float2 missVecPx,             // dejittered step-1 error vector == the transport's landing miss ([FIX 6])
    float  depthGate,             // [0..1] depth-transport same-surface confidence ([FIX 5])
    ViewportParams vp,
    out float divergencePx)
{
    divergencePx = 0.0;

    if (taaCrossTestStrength <= 0.001)
        return false;

    // 1) Exact landing: where the history surface is in the CURRENT render
    //    (the transport term dejitters the frame advance exactly).
    float2 pursuitUV = historySampleUV - (prevVelocityEffectiveJitteredUV + jitterTransportUV);
    if (any(pursuitUV < vp.minUV) || any(pursuitUV > vp.maxUV))
        return false;

    // 2) 3x3 depth/velocity neighborhood at the landing, resolved with the
    //    same state-aware layer rules as the current frame.
    float2 pixelPos  = pursuitUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;

    float  depths[9];
    float2 velocities[9];
    [unroll]
    for (int k = 0; k < 9; ++k)
    {
        float2 tapUV = clamp(snappedUV + kOffsets3x3[k] * vp.texelSize, vp.minUV, vp.maxUV);
        depths[k]     = tex2Dlod(depthTex,    float4(tapUV, 0.0, 0.0)).r;
        velocities[k] = tex2Dlod(velocityTex, float4(tapUV, 0.0, 0.0)).rg;
    }

    SurfaceEdgeState landingEdge = AnalyzeSurfaceEdgesCore(depths, taaUseDepthDilation > 0.5);

    float closestDepth = depths[0];
    int   closestIdx   = 0;
    [unroll]
    for (int j = 1; j < 9; ++j)
    {
        if (depths[j] > closestDepth) { closestDepth = depths[j]; closestIdx = j; }
    }

    float2 landingVelocityJitteredUV;
    if (landingEdge.isDilationZone)
    {
        // The landing is background behind a crest: it acts as the foreground.
        landingVelocityJitteredUV = velocities[closestIdx];
    }
    else if (landingEdge.isForegroundEdge)
    {
        landingVelocityJitteredUV = velocities[0];
    }
    else
    {
        // Flat: bilinear at the exact landing position, but never across a
        // velocity layer step ([FIX 8a]).
        landingVelocityJitteredUV = SelectLayerAwareQuadVelocity(velocities, fracPx, vp.sizePixels);
    }

    // 3) Divergence VECTOR between the two current-frame velocities.
    float2 divergenceVecPx = (landingVelocityJitteredUV - curVelocityJitteredUV) * vp.sizePixels;

    float coherenceRadiusPx = max(taaVelRejection, kMinVelCoherenceRadiusPx);

    // 3b) [FIX 6] First-order advection correction. On a continuous field the
    //     linear transport lands short by exactly the dejittered step-1 error
    //     (missVecPx), and the divergence that miss produces through the
    //     landing's own field is J_landing * missVecPx. Subtracting it cancels
    //     the same-surface signal to ~1-3 px even at 300 kph on the near
    //     ground, where the raw divergence is ~80 px. Gated by continuity and
    //     depth confidence; the single-layer requirement is gone ([FIX 8d]).
    //     Both the Jacobian and the continuity test are layer-consistent
    //     ([FIX 8b]): raw central differences at a straddled landing are half
    //     the inter-layer step.
    float maxCurvaturePx, maxPairGradPx;
    MeasureVelocityFieldShapeCoherent(velocities, vp.sizePixels, coherenceRadiusPx, maxCurvaturePx, maxPairGradPx);

    float landingSpreadPx = 0.0;
    [unroll]
    for (int m = 0; m < 9; ++m)
    {
        float2 deltaPx = (velocities[m] - landingVelocityJitteredUV) * vp.sizePixels;
        if (length(deltaPx) <= coherenceRadiusPx)
            landingSpreadPx = max(landingSpreadPx, length(deltaPx));
    }

    bool landingContinuous = IsContinuousVelocityField(maxCurvaturePx, maxPairGradPx);
    bool advectionGated    = landingContinuous && (depthGate > 0.001);

    float2 correctedDivergenceVecPx = divergenceVecPx;
    float  residualPx = 0.0;
    if (advectionGated)
    {
        float4 Jlanding = EstimateVelocityJacobianCoherentPx(velocities, vp.sizePixels, coherenceRadiusPx);
        correctedDivergenceVecPx -= ApplyVelocityJacobian(Jlanding, missVecPx) * depthGate;

        // [FIX 8e] the residual's miss term is capped at the landing's own
        // predicted same-surface advection: the raw miss on a depth-missed
        // disocclusion is the inter-layer step, and the budget must not scale
        // with the signal.
        float2 vLandingPx = landingVelocityJitteredUV * vp.sizePixels;
        float  missCapPx  = length(ApplyVelocityJacobian(Jlanding, vLandingPx)) * kMissCapHeadroom;
        residualPx = ResidualAdvectionBudgetPx(maxPairGradPx, length(missVecPx), length(fracPx), missCapPx) * depthGate;
    }

    // Diagnostics & test use the corrected magnitude (what drives the decision).
    divergencePx = length(correctedDivergenceVecPx);
    float tolerancePx = kPursuitVelBaseTolerancePx + landingSpreadPx + residualPx;

    return (divergencePx * saturate(taaCrossTestStrength)) > tolerancePx;
}

// ============================================================================
// JITTER CANCELLATION & TRANSPORT
// ----------------------------------------------------------------------------
// Both terms are exact first-order identities verified against the engine-side
// basis construction:
//   * Cancel:    the jitter content of (vCur - vPrev) is the second difference
//                s_t - 2 s_{t-1} + s_{t-2}; subtracting it dejitters the
//                velocity-pair comparison.
//   * Transport: advancing the history surface one frame forward requires
//                subtracting s_t - s_{t-1} + s_{t-2} from vPrev, so the pursuit
//                lands exactly on the history surface's current render position.
// ============================================================================
float2 AnalyticJitterOffsetPrev2UV()
{
    // Jitter offset of frame t-2, reconstructed analytically from its yaw/pitch
    // (the jitter is a physical camera rotation).
    return float2(-tan(taaJitPrev2Yaw)   / (2.0 * max(taaTanHalfFovX, 1e-4)),
                  -tan(taaJitPrev2Pitch) / (2.0 * max(taaTanHalfFovY, 1e-4)));
}

float2 EstimateJitterCancelUV(float2 jitterOffsetCurUV, float2 jitterOffsetPrevUV)
{
    return 2.0 * jitterOffsetPrevUV - jitterOffsetCurUV - AnalyticJitterOffsetPrev2UV();
}

float2 EstimateJitterTransportUV(float2 jitterOffsetCurUV, float2 jitterOffsetPrevUV)
{
    return jitterOffsetPrevUV - jitterOffsetCurUV - AnalyticJitterOffsetPrev2UV();
}

// ============================================================================
// VELOCITY REJECTION (step 2 of disocclusion; fully independent of depth)
// ============================================================================
VelocityRejectionResult EvaluateVelocityRejection(
    float2 historySampleUV,
    float2 resolvedVelocityJitteredUV,
    HistoryLandingSurface landing,
    bool   currentSingleLayer,      // current pixel is not a dilation zone / crest
    float  depthGate,               // [0..1] depth-transport same-surface confidence ([FIX 5])
    float2 jitterCancelUV,
    float2 jitterTransportUV,
    float2 neighborVelocityJitteredUV[9],
    ViewportParams vp)
{
    VelocityRejectionResult r;
    r.rejected        = false;
    r.errorRatio      = 0.0;
    r.layerGradientPx = 0.0;
    r.divergencePx    = 0.0;

    if (taaVelRejection <= 0.001)
        return r;

    // Dejittered error VECTOR against the landing's EFFECTIVE layer. A
    // dilation-zone landing validates exclusively against the foreground it
    // dilates to: if the current pixel is now genuine background, nothing
    // matches and the pixel is (correctly) flagged. A flat landing reads the
    // bilinear velocity at the exact sub-texel position -- committed to the
    // rendered layer when the quad straddles a step ([FIX 8a]) -- so a
    // continuous parallax field compares its OWN velocity, not a snapped
    // neighbor's, and a depth-invisible boundary compares LAYERS, not a blend.
    // On a same-surface pixel this vector IS the pursuit transport's landing
    // miss (identity), so it is passed through to the [FIX 6] correction.
    float2 velocityErrorVecPx = (resolvedVelocityJitteredUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
    float  velocityErrorPx    = length(velocityErrorVecPx);

    // Velocity-coherent gradients on both frame sides, combined and clamped
    // (feeds the ALERT noise bound only -- never the pursuit tolerance).
    float coherenceRadiusPx = max(taaVelRejection, kMinVelCoherenceRadiusPx);
    float currentGradientPx = MeasureVelocityCoherentGradientPx(
        resolvedVelocityJitteredUV, neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
    r.layerGradientPx = clamp(max(currentGradientPx, landing.coherentGradPx), 0.0, 1.0);

    float velocityNoise = r.layerGradientPx * taaVelGradientScale;

    // [FIX 8b] layer-consistent field shapes: a straddled neighborhood
    // measures only its own layer's continuity, so the smooth-field handling
    // below is no longer misgated at silhouettes.
    float curMaxCurvaturePx, curMaxPairGradPx;
    MeasureVelocityFieldShapeCoherent(neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx, curMaxCurvaturePx, curMaxPairGradPx);
    bool currentContinuous = IsContinuousVelocityField(curMaxCurvaturePx, curMaxPairGradPx);
    bool landingContinuous = IsContinuousVelocityField(landing.maxCurvaturePx, landing.maxPairGradPx);

    // [FIX 6]/[FIX 8c] Predicted same-surface advection for the alert
    // tolerance: on a continuous field the per-frame velocity change of the
    // SAME point is |J_current * v_current| (the point advects through its own
    // field). Gated by continuity, the depth transport's same-surface
    // confidence, and -- for the flat path -- single-layer structure; plus the
    // second-order residual, whose miss term is capped at the predicted
    // advection ([FIX 8e]).
    float advectionPx = 0.0;
    if (currentContinuous && currentSingleLayer)
    {
        float4 Jcur    = EstimateVelocityJacobianCoherentPx(neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
        float2 vCurPx  = resolvedVelocityJitteredUV * vp.sizePixels; // jitter is sub-pixel: irrelevant at this scale
        float  predictedPx = length(ApplyVelocityJacobian(Jcur, vCurPx));
        float  residualPx  = ResidualAdvectionBudgetPx(
            max(curMaxPairGradPx, landing.maxPairGradPx), velocityErrorPx, landing.snapDistPx,
            predictedPx * kMissCapHeadroom);
        advectionPx = (predictedPx + residualPx) * depthGate;
    }
    else if (!currentSingleLayer && landingContinuous)
    {
        // [FIX 8c] Dilation zone / crest: the resolved velocity IS a foreground
        // texel's, so the cross-frame comparison is that point's OWN advection
        // (J*V-scale on a fast, strongly graded foreground) -- previously
        // unpredicted, which was the random dilation-zone speckle. Predicted
        // here as a max-norm bound from the LANDING's field (the landing sits
        // on the foreground's interior, so its coherent stats are the
        // foreground layer's own).
        float2 vEffPx      = resolvedVelocityJitteredUV * vp.sizePixels;
        float  predictedPx = landing.maxPairGradPx * length(vEffPx);
        float  residualPx  = ResidualAdvectionBudgetPx(
            landing.maxPairGradPx, velocityErrorPx, landing.snapDistPx,
            predictedPx * kMissCapHeadroom);
        advectionPx = (predictedPx + residualPx) * depthGate;
    }

    r.errorRatio = velocityErrorPx / max(taaVelRejection + velocityNoise + advectionPx, 1e-4);

    if (r.errorRatio > 1.0)
    {
        // Error above tolerance: confirm with a pure current-frame pursuit of
        // the landing's effective (dominant) surface.
        r.rejected = PursuitConfirmsDivergence(
            historySampleUV, landing.effectiveVelocityJitteredPrevUV, resolvedVelocityJitteredUV,
            jitterTransportUV, velocityErrorVecPx, depthGate, vp, r.divergencePx);
    }
    return r;
}

// ============================================================================
// SHADOW RISK / CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
// Risk that this pixel is a freshly darkened (shadowed) area whose true color
// lies outside the temporal neighborhood: darkening vs. the spatial
// neighborhood or vs. the history sample.
float ComputeShadowRisk(float3 currentColorSpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float currentLuma      = max(currentColorSpace.x, 0.0);
    float neighborhoodLuma = max(stats.mean.x, kShadowLumaFloor);

    float spatialDarkening  = saturate((neighborhoodLuma - currentLuma) / max(taaShadowDarknessThreshold * neighborhoodLuma, kShadowThresholdMin));
    float temporalDarkening = saturate(abs(historyColorSpace.x - currentColorSpace.x) / max(taaShadowDarknessThreshold, kShadowThresholdMin));

    return saturate(max(spatialDarkening  * taaShadowSpatialMult,
                        temporalDarkening * taaShadowTemporalMult)) * saturate(taaShadowMitigation);
}

// How far the clip had to move the history, in sigmas -> rejection weight.
float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / max(stats.sigma, kMinSigma);
    float maxChannelDistance = max(clipDistance.x, max(clipDistance.y, clipDistance.z));
    return saturate((maxChannelDistance - taaClipDistanceRejectionMinError) * taaClipDistanceRejectionAmount);
}

// Pull the blended luma back toward the current sample when the history
// drifted beyond plausible neighborhood bounds (guards against luma ghosts).
void ApplyLumaDriftCorrection(inout float3 clippedHistorySpace, float3 currentColorSpace, ColorNeighborhoodStats stats)
{
    if (taaLumaDriftStrength <= 0.001)
        return;

    float lumaBias     = clippedHistorySpace.x - currentColorSpace.x;
    float relativeBias = lumaBias / max(currentColorSpace.x, kLumaDriftLumaFloor);

    // Fade the correction out on high-chroma neighborhoods (shading detail, not drift).
    float chromaSpread = max(stats.aabbMax.y - stats.aabbMin.y, stats.aabbMax.z - stats.aabbMin.z);
    float chromaGate   = 1.0 - saturate((chromaSpread - taaLumaDriftChromaTol) / kLumaDriftChromaFadeWidth);

    if (abs(relativeBias) > kLumaDriftRelThreshold && abs(lumaBias) > kLumaDriftAbsThreshold)
        clippedHistorySpace.x -= lumaBias * taaLumaDriftStrength * chromaGate;

    clippedHistorySpace.x = max(clippedHistorySpace.x, 0.0);
}

// History feedback (weight of history in the blend), reduced by motion, poor
// sub-pixel alignment, shadow risk and hard clipping distance.
// The alignment drop is applied AFTER the feedbackMin/Max clamp and ONLY on
// planar (single-layer) surfaces ([FIX 7]): there the current frame is jittered
// supersampling, so trading a little accumulation for resampling accuracy
// preserves texture sharpness; on edges and dilation zones the current sample
// is aliased and dropping history there reintroduces flicker.
float ComputeHistoryFeedback(HistoryReprojection repro, float shadowRisk, float clipDistanceRejection, bool planarSurface)
{
    float feedback = taaFeedbackMax;

    // Motion: fast-moving pixels trust the current frame more.
    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    feedback = lerp(taaFeedbackMax, taaFeedbackMin, motionDrop);
    feedback = clamp(feedback, taaFeedbackMin, taaFeedbackMax);

    // Shadowed / heavily clipped history is less trustworthy.
    feedback = lerp(feedback, taaFeedbackMin, shadowRisk * taaShadowBlendStrength);
    feedback = lerp(feedback, taaFeedbackMin, clipDistanceRejection);

    // Sub-pixel misalignment of the history sample (jitter phase mismatch).
    // Deliberately OUTSIDE the [feedbackMin, feedbackMax] clamp: inside it,
    // the subtraction was swallowed entirely whenever the two bounds are
    // equal (the default 0.97 / 0.97), which is how the drop silently died.
    if (planarSurface)
    {
        feedback -= taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment);
        feedback = max(feedback, 0.0);
    }
    return feedback;
}

// FXAA kicks in only when the temporal blend already leans strongly on the
// (noisy) current frame.
float ComputeFXAAFilterWeight(float currentBlendWeight)
{
    float weight = saturate((currentBlendWeight - kFXAABlendKnee) / max(1.0 - kFXAABlendKnee, 0.05));
    return saturate(weight * kFXAABlendSharpness);
}

// ============================================================================
// DEBUG VIEWS
// ============================================================================
float4 DebugViewVelocity(float2 velocityUV, float2 sizePixels, float velocityScale, float centerDepthRaw)
{
    float2 velocityPx = abs(velocityUV * sizePixels) * (kDebugVelocityScale * velocityScale);
    return float4(float3(saturate(velocityPx), 0.0), centerDepthRaw);
}

float4 DebugViewLinearDepth(float centerDepthRaw)
{
    float linearDepth = saturate(LinearizeDepth(centerDepthRaw) / kDebugLinearDepthRange);
    return float4(float3(linearDepth, linearDepth, linearDepth), centerDepthRaw);
}

float4 DebugViewHistoryColor(float3 historyColorSpace, float centerDepthRaw)
{
    return float4(saturate(FromSpace(historyColorSpace)), centerDepthRaw);
}

float4 DebugViewEdgeState(float3 currentColorRGB, bool isDilationZone, bool isForegroundEdge, bool velocityStraddled, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.25;
    if (isDilationZone)        debugColor = float3(1.0, 0.05, 0.05); // red: background behind a crest
    else if (isForegroundEdge) debugColor = float3(0.0, 0.85, 1.0);  // cyan: the crest itself
    else if (velocityStraddled) debugColor = float3(1.0, 0.05, 1.0); // magenta: depth-flat but velocity-straddled ([FIX 8a])
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewDisocclusionBreakdown(
    float3 currentColorRGB, bool depthRejected, bool velocityRejected, bool pursuitAlertSuppressed, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.1;
    if (depthRejected)    debugColor.r = 1.0;
    if (velocityRejected) debugColor.g = 1.0;
    if (depthRejected && velocityRejected) debugColor = float3(1.0, 1.0, 0.0);
    else if (pursuitAlertSuppressed)       debugColor.b = 0.45; // velocity error flagged but pursuit unconfirmed
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewPursuit(float3 currentColorRGB, float divergencePx, bool velocityRejected, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.15;
    debugColor += float3(saturate(divergencePx * 0.5), 0.0, 0.0);
    if (velocityRejected) debugColor = float3(0.0, 1.0, 0.2); // green: confirmed divergence
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewAlignmentDrop(float3 currentColorRGB, float dropAmount, bool planarSurface, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.2;
    debugColor += float3(dropAmount,
                         planarSurface ? 0.25 : 0.0,
                         planarSurface ? 0.0  : 0.25);
    return float4(debugColor, centerDepthRaw);
}

// ============================================================================
// MAIN PIXEL SHADER
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    ViewportParams vp          = GetViewportParams();
    CameraBasis currentCamera  = GetCurrentFrameCameraBasis();
    CameraBasis previousCamera = GetPreviousFrameCameraBasis();

    // ------------------------------------------------------------------
    // 1) Resolve the jittered render position of this stable output pixel.
    // ------------------------------------------------------------------
    SnappedPixel pixel = SnapJitteredPixel(IN.uv0, currentCamera, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    // Debug views that only need the center sample.
    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
        return DebugViewLinearDepth(centerDepthRaw);
    if (taaDebugMode > 2.5 && taaDebugMode < 3.5)
        return DebugViewVelocity(centerVelocityJitteredUV, vp.sizePixels, 1.0, centerDepthRaw);

    // ------------------------------------------------------------------
    // 2) Gather the raw 3x3 current-frame depth/velocity neighborhood.
    // ------------------------------------------------------------------
    bool useDepthDilation     = (taaUseDepthDilation > 0.5);
    bool needNeighborDepth    = useDepthDilation || (taaDepthRejection > 0.001);
    bool needNeighborVelocity = useDepthDilation || (taaDepthRejection > 0.001); // currently same condition
    bool fxaaEnabled          = (taaFallbackFXAA > 0.5);

    float2 tapUVs[9];
    Build3x3TapUVs(pixel.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, tapUVs);

    CurrentFrameNeighborhood neighborhood = GatherCurrentFrameNeighborhood(
        pixel.snappedUV, tapUVs, centerDepthRaw, centerVelocityJitteredUV,
        needNeighborDepth, needNeighborVelocity);

    // ------------------------------------------------------------------
    // 3) Classify the local depth structure, resolve velocity/depth and the
    //    foreground crest geometry.
    // ------------------------------------------------------------------
    SurfaceEdgeState edge = AnalyzeSurfaceEdges(neighborhood, useDepthDilation);

    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        // [FIX 8a] visualization: does the jitter-selected quad straddle a
        // velocity layer step even though depth classifies this pixel flat?
        bool velocityStraddled = false;
        if (!edge.isDilationZone && !edge.isForegroundEdge)
        {
            float2 v00, v10, v01, v11;
            SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.jitterPx, v00, v10, v01, v11);
            velocityStraddled = QuadStraddlesVelocityStep(
                v00, v10, v01, v11, neighborhood.velocityJitteredUV, vp.sizePixels,
                max(taaVelRejection, kMinVelCoherenceRadiusPx));
        }
        return DebugViewEdgeState(currentColorRGB, edge.isDilationZone, edge.isForegroundEdge, velocityStraddled, centerDepthRaw);
    }

    ResolvedSurface resolved      = ResolveSurfaceVelocity(neighborhood, edge, pixel.jitterPx, vp.sizePixels);
    ForegroundGeometry foreground = ComputeForegroundGeometry(neighborhood, resolved, edge);

    // Planar (single-layer) surface: not a dilation zone, not a crest -- the
    // bilinear-velocity path. The only place the sub-pixel alignment feedback
    // drop may act ([FIX 7]).
    bool currentSingleLayer = !(edge.isDilationZone || edge.isForegroundEdge);

    // Depth tolerance for taps to count as the same surface layer (used by the
    // disocclusion test's inter-layer shear scan).
    float surfaceLayerEps = taaDepthRejection * max(resolved.depthRaw, 1e-6) + 3.0 * edge.depthNoiseFloor;

    // ------------------------------------------------------------------
    // 4) Reproject the resolved surface into the history buffer.
    // ------------------------------------------------------------------
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, pixel.jitteredUV, pixel.jitterPx, resolved, previousCamera, vp);

    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
        return DebugViewVelocity(repro.motionPx, vp.sizePixels, 1.0, centerDepthRaw);

    if (taaDebugMode > 8.5 && taaDebugMode < 9.5)
    {
        float dropAmount = currentSingleLayer
            ? taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment)
            : 0.0;
        return DebugViewAlignmentDrop(currentColorRGB, dropAmount, currentSingleLayer, centerDepthRaw);
    }

    // ------------------------------------------------------------------
    // 5) Resolve the history landing surface (shared by both disocclusion
    //    tests). The landing is classified and resolved with the SAME layer
    //    semantics as the current frame: a dilation zone acts as the
    //    foreground it dilates to; a flat landing's velocity commits to its
    //    rendered layer across a velocity step ([FIX 8a]).
    // ------------------------------------------------------------------
    HistoryLandingSurface landing = SampleHistoryLandingSurface(
        repro.sampleUV, vp, useDepthDilation,
        (taaDepthRejection > 0.001) || (taaVelRejection > 0.001));

    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
        return DebugViewVelocity(landing.effectiveVelocityJitteredPrevUV, vp.sizePixels, 2.0, centerDepthRaw);

    // ------------------------------------------------------------------
    // 6) Gather the color neighborhood (for clipping & statistics).
    //    Also caches the FXAA corner taps when FXAA is enabled.
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float3 fxaaTapRGB[5]; // [0]=center, [1]=NW, [2]=NE, [3]=SW, [4]=SE
    fxaaTapRGB[0] = currentColorRGB;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
        if (fxaaEnabled && i >= 5) { fxaaTapRGB[i - 4] = tapRGB; }
    }

    // ------------------------------------------------------------------
    // 7) Validate the history sample position (filter support must fit).
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseLanczos3 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        // Offscreen / too close to the border: pure current frame (+ FXAA).
        if (fxaaEnabled)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, fxaaTapRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), centerDepthRaw);
        }
        return float4(currentColorRGB, centerDepthRaw);
    }

    // ------------------------------------------------------------------
    // 8) Bidirectional temporal attribution.
    //    - Depth disocclusion runs FIRST: its kappa-corrected transport
    //      score is the layering oracle (depthGate) for the velocity test's
    //      advection prediction ([FIX 5]/[FIX 6]).
    //    - Velocity rejection detects a history footprint occupied by a
    //      different (diverging) surface via current-frame pursuit, with the
    //      same effective-layer semantics -- now velocity-native at
    //      depth-invisible boundaries ([FIX 8]).
    //    Both are independent rejections; the final decision is their union.
    // ------------------------------------------------------------------
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized, pixel.jitterPx);
    float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

    // Jitter offsets of the last frames -> de-jitter terms (both exact).
    float2 jitterOffsetCurUV  = pixel.jitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV = IN.uv0 - ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0);
    float2 jitterCancelUV     = EstimateJitterCancelUV(jitterOffsetCurUV, jitterOffsetPrevUV);     // velocity comparison
    float2 jitterTransportUV  = EstimateJitterTransportUV(jitterOffsetCurUV, jitterOffsetPrevUV);  // pursuit landing

    // --- Step 1: motion-compensated depth disocclusion ----------------------
    // History depth for the test: the landing's EFFECTIVE layer (dilation
    // zone -> the foreground it dilates to; crest -> center tap; flat ->
    // bilinear at the exact landing position).
    float historyDepthTestRaw = landing.effectiveDepthRaw;

    float depthDisocclusionScore = ComputeDepthDisocclusionScore(
        resolved.depthRaw, historyDepthTestRaw,
        neighborhood.depthRaw, neighborhood.velocityJitteredUV,
        resolved.velocityJitteredUV,
        IN.uv0, repro.sampleUV, repro.prevCameraRayY,
        repro.jitterResidualPx, resolved.quadVelocitySpreadUV,
        foreground.slope, foreground.crestDrop, neighborhood.closestOffsetPx,
        edge.isDilationZone, edge.isForegroundEdge,
        edge.depthNoiseFloor, edge.depthQuantStep,
        edge.planeGradX, edge.planeGradY,
        surfaceLayerEps,
        vp);
    bool depthRejected = (depthDisocclusionScore >= 1.0);

    // Same-surface confidence for the velocity advection prediction ([FIX 5]):
    // full when the depth transport matches well inside its tolerance, zero
    // at rejection. On a genuine disocclusion this closes the correction and
    // the velocity test keeps its tight [FIX 3] tolerance.
    float depthGate = saturate((1.0 - depthDisocclusionScore) * 2.0);

    // --- Step 2: current-frame velocity pursuit (gated by depth confidence) -
    VelocityRejectionResult velocityRejection = EvaluateVelocityRejection(
        repro.sampleUV,
        resolved.velocityJitteredUV,
        landing,
        currentSingleLayer,             // current-side single layer
        depthGate,
        jitterCancelUV, jitterTransportUV,
        neighborhood.velocityJitteredUV,
        vp);

    // --- Step 3: strict binary union of both rejections ---------------------
    bool disoccluded = depthRejected || velocityRejection.rejected;

    if (taaDebugMode > 1.5 && taaDebugMode < 2.5)
        return DebugViewDisocclusionBreakdown(
            currentColorRGB, depthRejected, velocityRejection.rejected,
            (velocityRejection.errorRatio > 1.0 && !velocityRejection.rejected),
            centerDepthRaw);

    if (taaDebugMode > 6.5 && taaDebugMode < 7.5)
        return DebugViewPursuit(currentColorRGB, velocityRejection.divergencePx, velocityRejection.rejected, centerDepthRaw);

    // ------------------------------------------------------------------
    // 9) Resample the history color and clamp it into the valid gamut.
    // ------------------------------------------------------------------
    float3 historyColorSpace =
        (taaUseLanczos3 > 0.5)
        ? SampleHistoryColorLanczos3(repro.sampleUV, vp, historyMinUV, historyMaxUV)
        : SampleHistoryColorCatmullRom5Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    historyColorSpace = CompressGamut(historyColorSpace);

    if (taaDebugMode > 4.5 && taaDebugMode < 5.5)
        return DebugViewHistoryColor(historyColorSpace, centerDepthRaw);

    // ------------------------------------------------------------------
    // 10) Clip the history against the current color neighborhood, then
    //     correct for luma drift.
    // ------------------------------------------------------------------
    float shadowRisk   = ComputeShadowRisk(currentColorSpace, historyColorSpace, colorStats);
    float dynamicGamma = lerp(taaVarianceGamma, max(taaVarianceGamma, taaShadowVarianceBase), shadowRisk);

    float3 clippedHistorySpace = ClipHistoryToNeighborhood(
        historyColorSpace, colorStats, repro.motionNormalized, dynamicGamma,
        neighborhoodColorSpace, clipMargin);

    float clipDistanceRejection = 0.0;
    if (taaClipDistanceRejectionEnabled > 0.5)
        clipDistanceRejection = ComputeClipDistanceRejection(clippedHistorySpace, historyColorSpace, colorStats);

    ApplyLumaDriftCorrection(clippedHistorySpace, currentColorSpace, colorStats);

    // ------------------------------------------------------------------
    // 11) Blend (with optional FXAA fallback on the current-frame term).
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, shadowRisk, clipDistanceRejection, currentSingleLayer);
    float currentBlendWeight = 1.0 - historyFeedback;

    // Binary disocclusion: full rejection on trigger, zero partial credit.
    if (disoccluded)
        currentBlendWeight = 1.0;

    float3 currentFrameColorSpace = currentColorSpace;
    if (fxaaEnabled)
    {
        float fxaaWeight = ComputeFXAAFilterWeight(currentBlendWeight);
        if (fxaaWeight > 0.001)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, fxaaTapRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 12) Final blend & output (depth goes to alpha for next frame's tests).
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    // Clamp scalar luminance only; do NOT clamp signed chrominance channels.
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    return float4(outputRGB, centerDepthRaw);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}