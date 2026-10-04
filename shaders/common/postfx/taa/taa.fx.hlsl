// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect -- resolve pass
// ----------------------------------------------------------------------------
// CONTEXT: this mod produces sub-pixel jitter by PHYSICALLY ROTATING the
// in-game camera every frame (there is no projection-matrix jitter). The
// per-frame camera bases (P/Q/R below) therefore contain the jitter, and the
// velocity buffers contain the jitter motion. Every comparison that must be
// jitter-free subtracts the jitter component (s_t - 2*s_{t-1} + s_{t-2}).
// The history buffer stores the previous frame's STABLE (de-jittered) output.
//
// BASIS CONVENTION: P = the full-width right axis, R = the full-height up
// axis, Q = the top-left corner ray -- the only linear arrangement for which
// ProjectRayToStableUV(BuildCameraRay(uv, cam)) = uv exactly for a STABLE
// basis. normalize(Q) points at the corner, NOT forward.
//
// FIX -- MAP DIRECTIONS (read before touching the reprojection):
//   ReprojectThroughCamera(uv, cam)     frame-UV -> stable-UV (the stable
//                                       position of the frame pixel's content).
//   InverseReprojectThroughCamera(u, c) stable-UV -> frame-UV (where the
//                                       stable point sits in that frame).
//   They differ by 2x the frame's jitter and are NOT interchangeable. The
//   snap hides the difference for SAMPLING (both select the same texel while
//   |jitter| < 0.5 px, and fracPx remains the content's sub-texel offset
//   either way), but the REPROJECTION BASE must be the INVERSE: passing the
//   forward map leaks +2*jitter into every landing (static scenes landed at
//   IN.uv0 + 2*jitter instead of the identity), displacing every stored-field
//   read by up to a full texel -- which is what made the depth envelope fire
//   on stationary jitter while the own-history gate validated through the
//   neighbor's entry (and re-rejected the very revocations it issued).
//
// STORED MOTION FIELD (#TAA_HistMotion, written by taaMotion.fx.hlsl):
//   Per stable texel: layer-resolved effective velocity (xy), effective
//   depth (z), layer flag (w: 0 background/continuous, 1 foreground edge,
//   2 dilation zone) -- exactly the values that produced that texel's
//   history sample. The field stores the POST-VALIDATION state: a dilation
//   revoked by the landing gate is stored as background (the writer samples
//   the sign of #TAA_Result.a -- negative = revoked), so the revocation
//   PERSISTS -- a phase-artifact dilation is rejected once ("no history")
//   and stays rejected, instead of re-entering the dilation cycle from raw
//   depth every frame. Dilation zones are already baked into the stored
//   depths, so a landing inside an object's previous dilation band resolves
//   directly to the foreground depth. The revocation bit travels to the
//   writer through the sign of #TAA_Result.a (negative = revoked; the
//   magnitude is the acutance metric, recovered with abs()). EVERY return
//   path -- including all debug views -- transports the bit.
//   FIX: foreground-edge effective values are sub-texel PHASE dependent
//   (see LAYER SEMANTICS) -- both passes make the identical choice through
//   the shared classification and the identical snap, so the stored field
//   always matches what the resolve reprojected with.
//
// LAYER SEMANTICS (the disocclusion contract):
//   * A dilation zone acts as part of the object it dilates to. A pixel that
//     WAS foreground and lands in that object's previous dilation zone is
//     NOT disoccluded (the landing resolves to the foreground on both sides).
//   * Own-History Dilation Validation: a dilation candidate validates
//     through the history it would have if it stayed dilated -- its OWN
//     stored texel (the identity landing puts the candidate's reprojection
//     on its own texel; previously the +2*jitter displacement made it read
//     the neighbor's entry instead). A LAYER-MATCHED support tap validates
//     iff it is foreground-OWNED (flag >= 1), or its stored depth is at the
//     object itself (crest-anchored threshold). The support degenerates from
//     the 2x2 quad to a 1x2/2x1 pair to the center texel alone near the
//     texel center (kCenterLandingFracPx) -- static landings sit on the
//     center (the de-jittered reprojection of a static point is the
//     identity, now actually enforced by the inverse reprojection base).
//   * Dual-sided subpixel slant tolerance accounts for camera jitter phase
//     on continuous surfaces across both current and history frames.
//   * FIX -- Effective-value phase rule (foreground edges): an edge texel
//     whose jitter phase points TOWARD its own silhouette uses its ACTUAL
//     sample (own center depth + own center velocity -- the dilation's
//     point-sample rule, anchored to the own texel instead of the crest).
//     With the phase toward the flat part it takes the full sub-pixel
//     bilinear treatment (quad-interpolated velocity + minmod sub-pixel
//     depth) exactly like a continuous surface -- the sign-selected quad
//     then lies entirely on the object. The effective values never blend
//     across the silhouette in either phase. Extrapolating the minmod slope
//     ACROSS the limb was what falsely fired the depth envelope on the
//     sides of round poles: at a silhouette the minmod gradient is the
//     FLAT-side step, a 2-5x underestimate of a curved flank's outer slope
//     (a round pole's raw-depth slope blows up toward the limb), and the
//     phase-dependent error desynced current vs stored every frame.
//   * Dilated effective values stay the crest's RAW sample (closest texel's
//     depth AND velocity, never phase-interpolated): the landing is then
//     base(own stable position) + crest velocity = the T-1 RING's position,
//     whose stored entry is the T-1 CREST's raw sample -- an EXACT anchor
//     match (crest-center material vs crest-center material, tracked
//     rigidly with the object). See ClassifyLayerSurface.
//   * FIX -- Velocity disocclusion gates on the CURRENT side only
//     (continuous-background pixels); the HISTORY side is deliberately NOT
//     flag-gated. The layer-gated landing reconstruction handles mixed
//     quads: a background-centered landing excludes foreground-owned taps
//     (the comparison stays background-vs-background), while a foreground-
//     OWNED centered landing reconstructs the OCCLUDER's stored velocity --
//     exactly the thin-occluder reveal signal. The old historyIsForeground
//     gate disabled the test on every thin/curved/edge-flagged occluder
//     (wires, poles, characters), i.e. on precisely its canonical cases.
//
// OUTPUT: RGB = resolved color. A = the raw scene's local acutance energy
// (cross high-pass of RCAS-luma in SRTM space) consumed by taaFinal.fx.hlsl's
// auto-parity sharpener, SIGN-ENCODED with the revocation bit for the motion
// writer (debug views carry the bit on their depth magnitude instead).
// Debug mode 9 visualizes the gate: R = revoked candidate, G = kept
// candidate (full = flag branch, half = depth branch), B = depth-rejected.
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"

uniform_sampler2D(sceneTex,         0);
uniform_sampler2D(depthTex,         1);
uniform_sampler2D(historyTex,       2);
uniform_sampler2D(velocityTex,      3);
uniform_sampler2D(historyMotionTex, 4); // previous frame's stored motion field

// ============================================================================
// CBUFFER (engine-set constants -- fixed layout)
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
    float  taaMotionBlendDropSpeed;         float  taaUseSlepian3;
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
    float  taaVelPad1;                      float  taaVelPad0;
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
static const float kLargeValue              = 32000.0;
static const float kSqrt2                   = 1.41421356;
static const float kLog2E                   = 1.44269504;
static const float kMinMotionDirLengthPx    = 0.1;
static const float kShadowLumaFloor         = 0.01;
static const float kShadowThresholdMin      = 0.001;
static const float kFlickerPadThreshold     = 0.001;
static const float kMinMotionBlendDropSpeed = 0.1;
static const float kMotionFullStrengthPx    = 8.0;

// Landing within this fraction of the texel center (per axis; fracPx units,
// texel half-width = 0.5) counts as ON the center: the neighbors' bilinear
// weights are negligible, and the sign(frac) quad selection there is decided
// by reprojection residual noise (~0.005 px) rather than real subpixel
// position -- the full 2x2 gate is unstable exactly there. ~20x the residual.
// Any real drift >= 0.1 px/frame displaces the landing past this and
// legitimately extends the support (layer-matched).
static const float kCenterLandingFracPx     = 0.1;

// Sign-bit transport of the revocation flag: #TAA_Result.a is negative iff
// this pixel's tentative dilation was revoked (or could not be validated:
// offscreen history, or a debug view that returns before validation). The
// epsilon separates "revoked, zero magnitude" from +0.0; the magnitude is
// either the acutance metric (normal path) or the debug view's depth
// magnitude (debug paths). The writer ignores the bit for non-candidates,
// so conservative "revoked" encodings are always safe.
static const float kRevokedAlphaEpsilon     = 1e-30;

static const float kMinSigma                = 0.001;
static const float kMinSpatialContrast      = 0.001;
static const float kMinFootprintRange       = 1e-4;
static const float kFireflyClampEpsilon     = 0.001;

// Fallback FXAA
static const float kFXAAReduceMul           = 1.0 / 128.0;
static const float kFXAAReduceMin           = 1.0 / 128.0;
static const float kFXAAMaxDir              = 8.0;
static const float kFXAABlendKnee           = 0.8;
static const float kFXAABlendSharpness      = 2.0;

// Luma drift correction
static const float kLumaDriftLumaFloor       = 0.05;
static const float kLumaDriftRelThreshold    = 0.30;
static const float kLumaDriftAbsThreshold    = 0.03;
static const float kLumaDriftChromaFadeWidth = 2.0;

static const float kDebugVelocityScale      = 0.1;
static const float kDebugLinearDepthRange   = 100.0;

static const float kStdWeights[9]     = { 1.0, 0.36787944, 0.36787944, 0.36787944, 0.36787944, 0.13533528, 0.13533528, 0.13533528, 0.13533528 };
static const float kInvLength[9]      = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678 };

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
float SoftClipUnitScale(float overshootUnits, float softClipAmount, float motionFactor)
{
    float softLimit = 1.0 + softClipAmount * (1.0 - exp2(-(overshootUnits - 1.0) * kLog2E));
    return lerp(softLimit, 1.0, motionFactor);
}

float2 RaySlabInterval(float origin, float dir, float2 slab)
{
    float signVal = (dir >= 0.0) ? 1.0 : -1.0;
    float invDir  = 1.0 / (signVal * max(abs(dir), 1e-7));
    float t0      = (slab.x - origin) * invDir;
    float t1      = (slab.y - origin) * invDir;
    return float2(min(t0, t1), max(t0, t1));
}

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
// REVOCATION TRANSPORT
// ----------------------------------------------------------------------------
// EVERY return path must transport the revocation bit through the alpha sign
// -- including debug returns. The writer reads the sign every frame; a debug
// return with plain positive alpha makes it store the RAW classification,
// poisoning the motion field with unvalidated flag-2 entries that the next
// frame's own-history gate then trusts. The writer ignores the bit for
// non-candidates, so a conservative "revoked" is safe on returns that fire
// before validation has run.
// ============================================================================
float TransportAlpha(bool revoked, float magnitude)
{
    return revoked ? -(magnitude + kRevokedAlphaEpsilon) : magnitude;
}

// ============================================================================
// COLOR SPACES & REVERSIBLE TONEMAPPING
// ============================================================================
float3 Tonemap(float3 c)   { float luma = LumaRGB(c); return c / (1.0 + luma); }
float3 Untonemap(float3 c) { float luma = LumaRGB(c); return c / max(1.0 - min(luma, 0.999), 1e-4); }

static const float3x3 kRGB_TO_LMS   = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB   = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800, 2.60975740, -0.34131939, -0.00419608, -0.70341861, 1.70761470);

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

    float3 rgb = YCoCgToRGB(historyColorSpace);
    float minChannel = min(rgb.r, min(rgb.g, rgb.b));
    if (minChannel < 0.0)
    {
        float scale = saturate(max(0.0, historyColorSpace.x) / max(historyColorSpace.x - minChannel, kEpsilon));
        historyColorSpace.yz *= scale;
    }
    return historyColorSpace;
}

float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

// ============================================================================
// RAW-SCENE ACUTANCE METRIC (auto-parity sharpener input)
// ----------------------------------------------------------------------------
// FSR SRTM + RCAS luma (x2) -- the exact space taaFinal.fx.hlsl measures the
// resolved image in, so the energy ratio between the two is meaningful.
// ============================================================================
float SrtmLumaFSR(float3 rgb)
{
    float3 c = max(rgb, 0.0);
    c *= 1.0 / (max(c.r, max(c.g, c.b)) + 1.0);
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// ============================================================================
// CAMERA BASIS & REPROJECTION
// ============================================================================
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

// ============================================================================
// HISTORY REPROJECTION
// ----------------------------------------------------------------------------
// FIX: frameBaseUV must be the CURRENT-FRAME position of the stable point
// stableUV (InverseReprojectThroughCamera), NOT the forward map. With the
// inverse as the velocity base, the jitter components of the velocity cancel
// EXACTLY and a static scene lands at stableUV (the identity). For a locally
// uniform rotational flow the landing is stableUV - trueMotion regardless of
// WHICH effective velocity (own bilerp / crest / dilated) is passed:
//     landing = S_{t-1}( (g - a_t) + (a_t - a_{t-1} - m) ) = g - m
// so velocity dilation follows the object, while the background
// re-reprojection after a revocation lands back on the pixel's own texel.
// (The velocity may be sampled at ANY nearby texel -- its evaluation base
// cancels in the chain; only the frameBaseUV matters.)
// ============================================================================
struct HistoryReprojection
{
    float2 sampleUV;
    float2 motionPx;
    float  motionMagnitudePx;
    float  motionNormalized;
    float2 motionDirUnit;
    float  subpixelAlignment;
};

HistoryReprojection ReprojectToHistory(
    float2 stableUV,
    float2 frameBaseUV,        // FIX: renamed -- current-frame position of the
                               // stable point (inverse map), the velocity base.
    float2 velocityJitteredUV,
    CameraBasis previousCamera,
    ViewportParams vp)
{
    HistoryReprojection h;

    float3 prevRay    = BuildCameraRay(frameBaseUV + velocityJitteredUV, previousCamera);
    float2 fallbackUV = frameBaseUV + velocityJitteredUV;
    h.sampleUV        = ProjectRayToStableUV(prevRay, fallbackUV, taaTanHalfFovX, taaTanHalfFovY);

    h.motionPx          = (h.sampleUV - stableUV) * vp.sizePixels;
    h.motionMagnitudePx = length(h.motionPx);
    h.motionNormalized  = saturate(h.motionMagnitudePx / kMotionFullStrengthPx);
    h.motionDirUnit     = (h.motionMagnitudePx > kMinMotionDirLengthPx)
                        ? normalize(h.motionPx) : float2(1.0, 0.0);

    float2 historyPixelPos = h.sampleUV * vp.sizePixels;
    float2 subpixelPx      = historyPixelPos - (floor(historyPixelPos) + 0.5);
    h.subpixelAlignment    = 1.0 - saturate(length(subpixelPx) * kSqrt2);
    return h;
}

// ============================================================================
// JITTER CANCELLATION (EXACT ROTATIONAL FLOW)
// ============================================================================
float2 RotationFlowUV(float yaw, float pitch, float2 uv)
{
    float3 d = float3((uv.x * 2.0 - 1.0) * max(taaTanHalfFovX, 1e-4),
                       1.0,
                      (1.0 - uv.y * 2.0) * max(taaTanHalfFovY, 1e-4));

    float cy = cos(yaw), sy = sin(yaw);
    float3 r = float3(d.x * cy - d.y * sy,
                      d.x * sy + d.y * cy,
                      d.z);
    float cp = cos(pitch), sp = sin(pitch);
    float3 e = float3(r.x,
                      r.y * cp - r.z * sp,
                      r.y * sp + r.z * cp);

    float invEy = 1.0 / (abs(e.y) > 1e-5 ? e.y : 1e-5);
    float2 outUV;
    outUV.x = 0.5 + 0.5 * (e.x * invEy) / max(taaTanHalfFovX, 1e-4);
    outUV.y = 0.5 - 0.5 * (e.z * invEy) / max(taaTanHalfFovY, 1e-4);
    return outUV - uv;
}

// ============================================================================
// FALLBACK FXAA
// ============================================================================
float3 ApplyFXAA(float2 centerUV, float2 texelSize, float3 centerRGB, float3 cornersRGB[4], float2 minUV, float2 maxUV)
{
    float lumaNW = LumaRGB(cornersRGB[0]);
    float lumaNE = LumaRGB(cornersRGB[1]);
    float lumaSW = LumaRGB(cornersRGB[2]);
    float lumaSE = LumaRGB(cornersRGB[3]);
    float lumaM  = LumaRGB(centerRGB);

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
// HISTORY COLOR RESAMPLING: THE SLEPIAN SUITE
// ============================================================================
float3 ClampToTapFootprint(float3 color, float3 tapMin, float3 tapMax)
{
    float3 tapRange  = max(tapMax - tapMin, kMinFootprintRange);
    float3 overshoot = taaHistoryOvershoot * tapRange;
    return clamp(color, tapMin - overshoot, tapMax + overshoot);
}

void GetSlepian2FusedWeights_SIMD(float f, float baseCoord, out float3 uv, out float3 w)
{
    float f2 = f * f;
    float q  = 0.25 * f2;

    float4 u = float4(0.75 - 0.5 * f, 1.0, 0.75 + 0.5 * f, f) - q;
    u = max(u, 0.0);

    float4 p = 1.0 + u * (2.56 + u * (1.6384 + u * (0.466034 + u * 0.074565)));

    float4 raw;
    raw.x = -p.x * (f / (1.0 + f));
    raw.y =  p.y;
    raw.z =  p.z * (f / max(1.0 - f, 1e-4));
    raw.w = -p.w * (f / (2.0 - f));

    float raw12 = raw.y + raw.z;
    float t12   = raw.z / max(raw12, 1e-5);

    float sumW   = (raw.x + raw12) + raw.w;
    float invSum = 1.0 / max(sumW, 1e-5);

    w  = float3(raw.x, raw12, raw.w) * invSum;
    uv = float3(baseCoord - 1.0, baseCoord + t12, baseCoord + 2.0);
}

float3 SampleHistoryColor_Slepian2_Fast9Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uvX, weightX;
    float3 uvY, weightY;
    GetSlepian2FusedWeights_SIMD(fracOffset.x, baseTexel.x, uvX, weightX);
    GetSlepian2FusedWeights_SIMD(fracOffset.y, baseTexel.y, uvY, weightY);

    float3 tcX = clamp(uvX * vp.texelSize.x, minUV.x, maxUV.x);
    float3 tcY = clamp(uvY * vp.texelSize.y, minUV.y, maxUV.y);

    float3 color  = 0.0;
    float  sumW   = 0.0;
    float3 tapMin = float3( kLargeValue,  kLargeValue,  kLargeValue);
    float3 tapMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    [unroll]
    for (int y = 0; y < 3; ++y)
    {
        float wy = weightY[y];
        float vy = tcY[y];

        [unroll]
        for (int x = 0; x < 3; ++x)
        {
            float tc = tcX[x];
            float w  = weightX[x] * wy;

            if ((x != 1) && (y != 1))
                w *= 0.82;

            float3 tap = max(tex2Dlod(historyTex, float4(tc, vy, 0.0, 0.0)).rgb, 0.0);
            color += tap * w;
            sumW  += w;

            tapMin = min(tapMin, tap);
            tapMax = max(tapMax, tap);
        }
    }

    color /= max(sumW, 1e-4);
    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

void GetSlepian3FusedWeights(float frac, float baseCoord, out float3 posPt, out float2 posBi, out float w[5])
{
    float f = frac;
    float f2 = f * f;

    float d0 = 2.0 + f; float d0_2 = d0 * d0;
    float d1 = 1.0 + f; float d1_2 = d1 * d1;
    float d3 = 1.0 - f; float d3_2 = d3 * d3;
    float d4 = 2.0 - f; float d4_2 = d4 * d4;
    float d5 = 3.0 - f; float d5_2 = d5 * d5;

    float u0 = max(1.0 - d0_2 * (1.0 / 9.0), 0.0);
    float u1 = max(1.0 - d1_2 * (1.0 / 9.0), 0.0);
    float u2 = max(1.0 - f2   * (1.0 / 9.0), 0.0);
    float u3 = max(1.0 - d3_2 * (1.0 / 9.0), 0.0);
    float u4 = max(1.0 - d4_2 * (1.0 / 9.0), 0.0);
    float u5 = max(1.0 - d5_2 * (1.0 / 9.0), 0.0);

    #define SLEPIAN_I0_POLY(u) (1.0 + u * (5.1529 + u * (6.6380 + u * (3.8052 + u * 1.0858))))

    float win0 = SLEPIAN_I0_POLY(u0);
    float win1 = SLEPIAN_I0_POLY(u1);
    float win2 = SLEPIAN_I0_POLY(u2);
    float win3 = SLEPIAN_I0_POLY(u3);
    float win4 = SLEPIAN_I0_POLY(u4);
    float win5 = SLEPIAN_I0_POLY(u5);

    float raw0 =  win0 * (f / d0);
    float raw1 = -win1 * (f / d1);
    float raw2 =  win2;
    float raw3 =  win3 * (f / max(d3, 1e-4));
    float raw4 = -win4 * (f / d4);
    float raw5 =  win5 * (f / d5);

    float w23 = raw2 + raw3;
    float t23 = raw3 / max(w23, 1e-5);

    posPt.x = baseCoord - 2.0;
    posPt.y = baseCoord - 1.0;
    posPt.z = baseCoord + 2.0;
    posBi.x = baseCoord + t23;
    posBi.y = baseCoord + 3.0;

    w[0] = raw0;
    w[1] = raw1;
    w[2] = w23;
    w[3] = raw4;
    w[4] = raw5;

    float sumW = w[0] + w[1] + w[2] + w[3] + w[4];
    float invSum = 1.0 / max(sumW, 1e-5);
    [unroll]
    for (int i = 0; i < 5; ++i)
        w[i] *= invSum;
}

float3 SampleHistoryColor_Slepian3_Fused21Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uX_pt, uY_pt;
    float2 uX_bi, uY_bi;
    float wX[5], wY[5];

    GetSlepian3FusedWeights(fracOffset.x, baseTexel.x, uX_pt, uX_bi, wX);
    GetSlepian3FusedWeights(fracOffset.y, baseTexel.y, uY_pt, uY_bi, wY);

    float coordsX[5] = { uX_pt.x, uX_pt.y, uX_bi.x, uX_pt.z, uX_bi.y };
    float coordsY[5] = { uY_pt.x, uY_pt.y, uY_bi.x, uY_pt.z, uY_bi.y };

    float tcX[5], tcY[5];
    [unroll]
    for (int c = 0; c < 5; ++c)
    {
        tcX[c] = clamp(coordsX[c] * vp.texelSize.x, minUV.x, maxUV.x);
        tcY[c] = clamp(coordsY[c] * vp.texelSize.y, minUV.y, maxUV.y);
    }

    float3 color  = 0.0;
    float  sumW   = 0.0;
    float3 tapMin = float3( kLargeValue,  kLargeValue,  kLargeValue);
    float3 tapMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    [unroll]
    for (int y = 0; y < 5; ++y)
    {
        float vy = tcY[y];
        float wy = wY[y];

        [unroll]
        for (int x = 0; x < 5; ++x)
        {
            if ((x == 0 || x == 4) && (y == 0 || y == 4))
                continue;

            float vx = tcX[x];
            float w  = wX[x] * wy;

            float3 tap = max(tex2Dlod(historyTex, float4(vx, vy, 0.0, 0.0)).rgb, 0.0);
            color += tap * w;
            sumW  += w;

            if (x >= 1 && x <= 3 && y >= 1 && y <= 3)
            {
                tapMin = min(tapMin, tap);
                tapMax = max(tapMax, tap);
            }
        }
    }

    color /= max(sumW, 1e-4);
    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

// ============================================================================
// HISTORY MOTION FIELD SAMPLING (layer-resolved landing analysis)
// ----------------------------------------------------------------------------
// Four taps of the enclosing bilinear quad around the landing. The
// reconstruction is LAYER-GATED: the stored field already carries the dilated
// foreground in its zones, so a subpixel landing must never blend across an
// ownership boundary -- taps whose layer class (foreground-owned: flag >= 1)
// differs from the landing texel's are excluded from the interpolation
// (weights renormalized), from the gradients, AND from the support gate:
// a foreign-layer tap must never validate a dilation while the depth
// reconstruction excludes that same tap.
// ============================================================================
struct HistoryMotion
{
    float2 velocityUV;      // layer-gated bilinear stored effective velocity
    float  depth;           // layer-gated bilinear stored effective depth
    float  depthMax;        // enclosing-quad max (closest surface present; ~= old closestDepth)
    float  supportDepthMax; // LAYER-MATCHED bilinear-support max depth
    float  supportMaxFlag;  // LAYER-MATCHED max layer flag within the support
    float  gradX, gradY;    // layer-gated minmod gradients of the stored field
    float  spreadPx;        // max matched-tap velocity deviation from the reconstruction (px)
    float  centerFlag;      // layer flag of the snapped landing texel
    float  maxFlag;         // max layer flag in the quad
};

HistoryMotion SampleHistoryMotion(SnappedCoord landing, ViewportParams vp)
{
    HistoryMotion m;

    float sx = (landing.fracPx.x >= 0.0) ? 1.0 : -1.0;
    float sy = (landing.fracPx.y >= 0.0) ? 1.0 : -1.0;

    float2 uv10 = clamp(landing.snappedUV + float2(sx, 0.0) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv01 = clamp(landing.snappedUV + float2(0.0, sy) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv11 = clamp(landing.snappedUV + float2(sx, sy) * vp.texelSize, vp.minUV, vp.maxUV);

    float4 m00 = tex2Dlod(historyMotionTex, float4(landing.snappedUV, 0.0, 0.0));
    float4 m10 = tex2Dlod(historyMotionTex, float4(uv10, 0.0, 0.0));
    float4 m01 = tex2Dlod(historyMotionTex, float4(uv01, 0.0, 0.0));
    float4 m11 = tex2Dlod(historyMotionTex, float4(uv11, 0.0, 0.0));

    bool fgCenter = (m00.w >= 0.75);
    bool match10  = ((m10.w >= 0.75) == fgCenter);
    bool match01  = ((m01.w >= 0.75) == fgCenter);
    bool match11  = ((m11.w >= 0.75) == fgCenter);

    float2 f   = abs(landing.fracPx);
    float  w00 = (1.0 - f.x) * (1.0 - f.y);
    float  w10 = f.x * (1.0 - f.y) * (match10 ? 1.0 : 0.0);
    float  w01 = (1.0 - f.x) * f.y * (match01 ? 1.0 : 0.0);
    float  w11 = f.x * f.y * (match11 ? 1.0 : 0.0);
    float  invW = 1.0 / max(w00 + w10 + w01 + w11, 1e-4);

    m.depth      = (m00.z  * w00 + m10.z  * w10 + m01.z  * w01 + m11.z  * w11) * invW;
    m.velocityUV = (m00.xy * w00 + m10.xy * w10 + m01.xy * w01 + m11.xy * w11) * invW;

    float dx0 = match10 ? (m10.z - m00.z) : 0.0;
    float dx1 = (match01 && match11) ? (m11.z - m01.z) : 0.0;
    m.gradX = Minmod(dx0, dx1);

    float dy0 = match01 ? (m01.z - m00.z) : 0.0;
    float dy1 = (match10 && match11) ? (m11.z - m10.z) : 0.0;
    m.gradY = Minmod(dy0, dy1);

    // Quad-wide extents are deliberately NOT layer-gated: they answer "is any
    // foreground-owned surface present in the quad at all".
    m.depthMax   = max(max(m00.z, m10.z), max(m01.z, m11.z));
    m.centerFlag = m00.w;
    m.maxFlag    = max(max(m00.w, m10.w), max(m01.w, m11.w));

    // LAYER-MATCHED bilinear reconstruction support: the subset of the quad
    // the landing's reconstruction actually reads, degenerating with
    // position -- full 2x2 (landing between texels), 1x2 / 2x1 (landing on
    // a center axis: the noise-selected perpendicular neighbor's weight is
    // negligible), center texel alone (landing at the center). Foreign-
    // layer taps are excluded by the SAME match masks the depth
    // reconstruction uses.
    bool xOffCenter = (f.x >= kCenterLandingFracPx);
    bool yOffCenter = (f.y >= kCenterLandingFracPx);
    float supportDepthMax = m00.z;
    float supportMaxFlag  = m00.w;
    if (xOffCenter && match10)               { supportDepthMax = max(supportDepthMax, m10.z); supportMaxFlag = max(supportMaxFlag, m10.w); }
    if (yOffCenter && match01)               { supportDepthMax = max(supportDepthMax, m01.z); supportMaxFlag = max(supportMaxFlag, m01.w); }
    if (xOffCenter && yOffCenter && match11) { supportDepthMax = max(supportDepthMax, m11.z); supportMaxFlag = max(supportMaxFlag, m11.w); }
    m.supportDepthMax = supportDepthMax;
    m.supportMaxFlag  = supportMaxFlag;

    // Bilinear mixing error of the reconstruction (px) -- the landing-side
    // replacement for the old raw-history layer velocity spread.
    float2 d10 = (m10.xy - m.velocityUV) * vp.sizePixels * (match10 ? 1.0 : 0.0);
    float2 d01 = (m01.xy - m.velocityUV) * vp.sizePixels * (match01 ? 1.0 : 0.0);
    float2 d11 = (m11.xy - m.velocityUV) * vp.sizePixels * (match11 ? 1.0 : 0.0);
    m.spreadPx = sqrt(max(max(dot(d10, d10), dot(d01, d01)), dot(d11, d11)));

    return m;
}

// ============================================================================
// COLOR NEIGHBORHOOD STATISTICS
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 aabbMin;
    float3 aabbMax;
    float3 mean;
    float3 sigma;
    float3x3 invCov;
    bool validCovariance;
    float spatialContrast;
    float3 expectedJitterShift;
    float weights[9];
};

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized, float2 jitterPx)
{
    ColorNeighborhoodStats stats;
    stats.validCovariance = false;
    stats.aabbMin = neighborhoodColorSpace[0];
    stats.aabbMax = neighborhoodColorSpace[0];

    bool covarianceEnabled    = (taaUseCovarianceClipping > 0.5) && (taaUseKDopClipping <= 0.5);
    bool jitterPaddingEnabled = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 weightedSum   = float3(0.0, 0.0, 0.0);
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);
    float3 weightedCross = float3(0.0, 0.0, 0.0);
    float  totalWeight   = 0.0;
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

    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        // FIX: the second clamp fed aabbMin into aabbMax, collapsing the box
        // onto its min end whenever the firefly clamp was active.
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

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

        if (jitterPaddingEnabled)
        {
            float paddingFade   = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - motionNormalized) : 1.0;
            float paddingAmount = taaJitterFlickerPadding * paddingFade;

            if (taaDirectionalVariance > 0.5)
            {
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
                float padAmount = stats.spatialContrast * length(jitterPx) * paddingAmount;
                cov[0][0] += padAmount * padAmount;
                cov[1][1] += padAmount * padAmount;
                cov[2][2] += padAmount * padAmount;
            }
        }

        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        stats.invCov = InverseSymmetric3x3(cov, stats.validCovariance);
    }

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
float3 ClipRayToBox(float3 history, float3 target, float3 boxMin, float3 boxMax, float softClipAmount, float motionFactor)
{
    float3 boxCenter = 0.5 * (boxMax + boxMin);
    float3 boxExtent = max(0.5 * (boxMax - boxMin), kEpsilon);

    float3 historyUnit = (history - boxCenter) / boxExtent;
    float3 absU = abs(historyUnit);
    float overshootUnits = max(absU.x, max(absU.y, absU.z));

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
        float3 diff = historyColorSpace - stats.mean;
        float mahalanobisSq = dot(diff, mul(stats.invCov, diff));
        float gammaSq = dynamicGamma * dynamicGamma;
        float3 ellipsoidClipped = (mahalanobisSq > gammaSq && mahalanobisSq > kEpsilon)
            ? (stats.mean + diff * (dynamicGamma / sqrt(max(mahalanobisSq, kEpsilon))))
            : historyColorSpace;
        return ClipRayToBox(ellipsoidClipped, stats.mean, stats.aabbMin - clipMargin, stats.aabbMax + clipMargin, taaSoftClip, motionFactor);
    }

    float3 chromaScale     = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 varianceExtents = stats.sigma * dynamicGamma * chromaScale;
    float3 boxMin = max(stats.mean - varianceExtents, stats.aabbMin - clipMargin);
    float3 boxMax = min(stats.mean + varianceExtents, stats.aabbMax + clipMargin);

    return ClipRayToBox(historyColorSpace, stats.mean, boxMin, boxMax, taaSoftClip, motionFactor);
}

// ============================================================================
// SHADOW RISK / CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
float ComputeShadowRisk(float3 currentColorSpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float currentLuma      = max(currentColorSpace.x, 0.0);
    float neighborhoodLuma = max(stats.mean.x, kShadowLumaFloor);

    float spatialDarkening  = saturate((neighborhoodLuma - currentLuma) / max(taaShadowDarknessThreshold * neighborhoodLuma, kShadowThresholdMin));
    float temporalDarkening = saturate(abs(historyColorSpace.x - currentColorSpace.x) / max(taaShadowDarknessThreshold, kShadowThresholdMin));

    return saturate(max(spatialDarkening  * taaShadowSpatialMult,
                        temporalDarkening * taaShadowTemporalMult)) * saturate(taaShadowMitigation);
}

float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / max(stats.sigma, kMinSigma);
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

float ComputeHistoryFeedback(HistoryReprojection repro, float shadowRisk, float clipDistanceRejection, bool planarSurface)
{
    float feedback = taaFeedbackMax;

    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    feedback = lerp(taaFeedbackMax, taaFeedbackMin, motionDrop);
    feedback = clamp(feedback, taaFeedbackMin, taaFeedbackMax);

    feedback = lerp(feedback, taaFeedbackMin, shadowRisk * taaShadowBlendStrength);
    feedback = lerp(feedback, taaFeedbackMin, clipDistanceRejection);

    if (planarSurface)
    {
        feedback -= taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment);
        feedback = max(feedback, 0.0);
    }
    return feedback;
}

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

float4 DebugViewEdgeState(float3 currentColorRGB, bool wasRevoked, bool isDilationZone, bool isForegroundEdge, bool isForeground, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.25;
    if (wasRevoked)             debugColor = float3(1.0, 0.6, 0.0);  // Orange: REVOKED dilation candidate
    else if (isDilationZone)    debugColor = float3(1.0, 0.05, 0.05); // Red: kept dilation zone
    else if (isForegroundEdge)  debugColor = float3(0.0, 0.85, 1.0);  // Cyan: silhouette edge
    else if (isForeground)      debugColor = float3(0.1, 0.95, 0.1);  // Green: object body
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewDisocclusionBreakdown(
    float3 currentColorRGB, bool depthRejected, bool velocityRejected, bool alertSuppressed, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.1;
    if (depthRejected)    debugColor.r = 1.0;
    if (velocityRejected) debugColor.g = 1.0;
    if (depthRejected && velocityRejected) debugColor = float3(1.0, 1.0, 0.0);
    else if (alertSuppressed)              debugColor.b = 0.45;
    return float4(debugColor, centerDepthRaw);
}

// ============================================================================
// MAIN PIXEL SHADER
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    ViewportParams vp          = GetViewportParams(oneOverTargetSize);
    CameraBasis currentCamera  = GetCurrentFrameCameraBasis();
    CameraBasis previousCamera = GetPreviousFrameCameraBasis();

    // ------------------------------------------------------------------
    // 1) Current Pixel Geometry
    // ------------------------------------------------------------------
    // Forward map (frame-UV -> stable-UV): the stable position of the content
    // at the frame texel under IN.uv0. Used ONLY as the +b jitter offset
    // (velocity cancel + variance machinery). NEVER snapped: the rotational
    // jitter flow is amplified (1+tan^2) toward the screen edges (~2.3x at
    // the sides for 65 deg / 16:9 -> ~+-1.1 px there, vs +-0.5 px aimed at
    // screen center), and beyond half a texel a forward-map snap selects a
    // texel whose content is up to ~2 px from the output pixel.
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    // Inverse map (stable-UV -> frame-UV): where the stable grid point sits in
    // the CURRENT frame. This is BOTH the reprojection base AND the sampling
    // snap: snap(u_t(g)) is the texel whose content lies nearest to stable
    // position g, so the sampled content stays within +-0.5 px of the output
    // pixel everywhere on screen. (Restores the original mod's sampling
    // semantics -- its basis(+angles) current basis was the mirrored map,
    // an inverse stand-in, which is why its snap was correct.) fracPx =
    // stableInFrameUV - snapped is identical to the original mod's fracPx,
    // so all tuning against it still applies.
    float2 stableInFrameUV  = InverseReprojectThroughCamera(IN.uv0, currentCamera, taaTanHalfFovX, taaTanHalfFovY);
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    // Debug modes 3/4 fire before any classification exists. They still must
    // not poison the transport: encode conservative "revoked" (the writer
    // ignores the bit for non-candidates, so this only forces candidates
    // into their background state during these views -- the band rebuilds
    // one hit phase after leaving).
    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
    {
        float4 v = DebugViewLinearDepth(centerDepthRaw);
        v.a = -(centerDepthRaw + kRevokedAlphaEpsilon);
        return v;
    }
    if (taaDebugMode > 2.5 && taaDebugMode < 3.5)
    {
        float4 v = DebugViewVelocity(centerVelocityJitteredUV, vp.sizePixels, 1.0, centerDepthRaw);
        v.a = -(centerDepthRaw + kRevokedAlphaEpsilon);
        return v;
    }

    // ------------------------------------------------------------------
    // 2) Current Neighborhood Gather
    // ------------------------------------------------------------------
    bool useDepthDilation    = (taaUseDepthDilation > 0.5);
    bool depthTestEnabled    = (taaDepthRejection > 0.001);
    bool velocityTestEnabled = (taaVelRejection > 0.001);
    bool fxaaEnabled         = (taaFallbackFXAA > 0.5);

    float2 tapUVs[9];
    Build3x3TapUVs(pixel.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, tapUVs);

    float  curDepths[9];
    float2 curVelocities[9];
    curDepths[0]     = centerDepthRaw;
    curVelocities[0] = centerVelocityJitteredUV;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        curDepths[i]     = tex2Dlod(depthTex,    float4(tapUVs[i], 0.0, 0.0)).r;
        curVelocities[i] = tex2Dlod(velocityTex, float4(tapUVs[i], 0.0, 0.0)).rg;
    }

    // ------------------------------------------------------------------
    // 3) Current Layer Classification (Tentative Candidate Dilation)
    // ------------------------------------------------------------------
    LayerSurface currentLayer = ClassifyLayerSurface(
        curDepths, curVelocities,
        pixel.fracPx, vp.sizePixels,
        useDepthDilation, taaDepthRejection,
        velocityTestEnabled);

    // ------------------------------------------------------------------
    // 4) Reprojection to History Buffer
    // ------------------------------------------------------------------
    // FIX: base = stableInFrameUV (the inverse map). Static scenes now land
    // on their own texel (the identity); motion is measured jitter-free.
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, currentLayer.effectiveVelocityUV, previousCamera, vp);

    // Debug mode 1 fires after classification but before validation:
    // conservative per-candidate encoding.
    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
    {
        // FIX: motionPx is already in PIXELS; DebugViewVelocity multiplies by
        // sizePixels internally (its other callers pass UV-space vectors), so
        // convert to UV first -- otherwise the view amplifies ~150x and
        // saturates at the 0.005 px reprojection noise floor.
        float4 v = DebugViewVelocity(repro.motionPx * vp.texelSize, vp.sizePixels, 1.0, centerDepthRaw);
        v.a = TransportAlpha(currentLayer.isDilationZone, centerDepthRaw);
        return v;
    }

    // ------------------------------------------------------------------
    // 5) Exact Jitter Plumbing
    // ------------------------------------------------------------------
    // All offsets in the SAME sense: a_tau = S_tau(g) - g (the per-frame
    // content shift). jitterOffsetCurUV = S_t(g) - g is exactly that.
    float2 jitterOffsetCurUV  = currentJitteredUV - IN.uv0;
    // FIX: jitterOffsetPrevUV was computed NEGATED (IN.uv0 - map), flipping
    // the middle term of the second difference below: the cancel subtracted
    // a_t + 2*a_{t-1} + a_{t-2} instead of a_t - 2*a_{t-1} + a_{t-2}, leaving
    // 4*a_{t-1} of uncancelled jitter in the velocity test on static scenes.
    float2 jitterOffsetPrevUV = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterCancelUV     = float2(0.0, 0.0);
    float2 jitterTransportUV  = float2(0.0, 0.0);

    bool needJitterCancel = velocityTestEnabled || (taaDebugMode > 1.5 && taaDebugMode < 2.5)
                                                || (taaDebugMode > 11.5 && taaDebugMode < 12.5);
    if (needJitterCancel)
    {
        // a_{t-2} from the t-2 jitter angles. MUST match the sense of the two
        // offsets above (= S_{t-2}(g) - g); RotationFlowUV(theta) equals that
        // iff the engine's per-frame basis rotation matches RotationFlowUV's
        // rotation convention. VERIFY with debug mode 12 on a static scene:
        // the residual must collapse to noise -- if it DOUBLES, negate the
        // angles on the C++ side (or negate this term here).
        float2 jitterOffsetPrev2UV = RotationFlowUV(taaJitPrev2Yaw, taaJitPrev2Pitch, IN.uv0);

        // Static-scene velocity difference (both frames' jitter in V):
        //   V_t - V_{t-1} = (a_t - a_{t-1}) - (a_{t-1} - a_{t-2})
        //                 =  a_t - 2*a_{t-1} + a_{t-2}
        jitterCancelUV    = jitterOffsetCurUV - 2.0 * jitterOffsetPrevUV + jitterOffsetPrev2UV;
        jitterTransportUV = jitterOffsetPrev2UV - jitterOffsetPrevUV;
    }

    // ------------------------------------------------------------------
    // 6) Color Neighborhood Gather (Zero-Redundancy Shared FXAA Corners)
    //    + raw-scene acutance energy for the auto-parity sharpener
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float3 fxaaCornersRGB[4]; // 0:NW(5), 1:NE(6), 2:SW(7), 3:SE(8)
    float  rawCrossLumaSum = 0.0; // cross taps 1..4, RCAS-luma of SRTM'd raw

    [unroll]
    for (int c = 1; c < 9; ++c)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[c], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[c] = ToSpace(tapRGB);
        if (c <= 4)
            rawCrossLumaSum += SrtmLumaFSR(tapRGB);
        if (fxaaEnabled && c >= 5)
            fxaaCornersRGB[c - 5] = tapRGB;
    }

    // Parity target for taaFinal.fx.hlsl: local high-pass energy of the raw
    // scene, in the same SRTM/RCAS-luma space the final pass measures the
    // resolved image in. Transported via the alpha channel (sign-encoded
    // with the revocation bit -- see the final return).
    float rawHighPass        = SrtmLumaFSR(currentColorRGB) - rawCrossLumaSum * 0.25;
    float rawSharpnessEnergy = rawHighPass * rawHighPass;

    // ------------------------------------------------------------------
    // 7) Validate History Sample Bounds
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseSlepian3 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        // Offscreen history = no history support: a tentative dilation here
        // can never be validated, so encode it as revoked for the writer.
        float earlyAlpha = currentLayer.isDilationZone
            ? -(rawSharpnessEnergy + kRevokedAlphaEpsilon)
            : rawSharpnessEnergy;

        if (fxaaEnabled)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), earlyAlpha);
        }
        return float4(currentColorRGB, earlyAlpha);
    }

    // ------------------------------------------------------------------
    // 8) History Landing Analysis (stored layer-resolved motion field)
    // ------------------------------------------------------------------
    SnappedCoord landing = SnapUVToTexel(repro.sampleUV, vp);
    HistoryMotion histMotion = SampleHistoryMotion(landing, vp);

    float rayLenCur  = RayLengthFromUV(IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
    float perspScale = rayLenPrev / max(rayLenCur, 1e-6);

    // Set when the tentative dilation is revoked below; travels to the
    // motion writer through the alpha sign so the revocation PERSISTS in
    // the stored field.
    bool dilationRevoked = false;

    // Debug bookkeeping (mode 9): pre-mutation candidacy and which gate
    // branch kept the dilation.
    bool dilationCandidate = false;
    bool gateViaFlag       = false;

    if (currentLayer.isDilationZone)
    {
        dilationCandidate = true;

        // Own-History Dilation Validation: the candidate validates through
        // the history it would have if it stayed dilated -- its own stored
        // texel. (With the fixed base, a static candidate lands EXACTLY on
        // its own texel; under motion it follows the object.)
        float rawExpClosest = currentLayer.closestDepth * perspScale;

        // Object-side slant allowance: the depth spread across the object
        // between the candidate and its crest, from the slope-limited
        // gradients. The minmod limiter zeroes the gradients across a
        // silhouette cliff, so background-side candidates (own center =
        // background -- every silhouette's outside neighbor) get NO slant
        // allowance and a strict crest anchor.
        float2 crestOffset = kOffsets3x3[currentLayer.closestIdx];
        float  objectSlant = (abs(currentLayer.gradX * crestOffset.x) + abs(currentLayer.gradY * crestOffset.y)) * perspScale;
        float  tolObject   = max(taaDepthRejection * rawExpClosest, 1e-5) + objectSlant;
        float  fgThreshold = rawExpClosest - tolObject;

        // A) Own-history support gate (LAYER-MATCHED). A support tap
        //    validates the dilation iff it is foreground-OWNED (flag >= 1:
        //    sampled edge or persisted dilation band -- the record of
        //    "stayed dilated"), or its stored depth is at the object itself
        //    (crest-anchored). The candidate's own background history fails
        //    both -> revoke. Foreign-layer taps are excluded by the same
        //    match masks the depth reconstruction uses.
        gateViaFlag = (histMotion.supportMaxFlag >= 0.75);
        bool quadTouchesForeground = gateViaFlag
                                  || (histMotion.supportDepthMax >= fgThreshold);

        // C) The landing itself was inside the object's dilation band.
        bool touchesAlreadyDilated = (histMotion.centerFlag >= 1.5) && (histMotion.depth >= fgThreshold);

        bool historyHasForeground = quadTouchesForeground
                                  || touchesAlreadyDilated;

        if (!historyHasForeground)
        {
            // Own history contains neither ownership nor object depth:
            // revoke dilation. The revocation persists via the stored field
            // (taaMotion consumes the alpha sign and writes this texel as
            // background), so a phase-artifact dilation is rejected once
            // and stays rejected.
            dilationRevoked = true;
            RevokeDilation(currentLayer, centerDepthRaw, curVelocities, pixel.fracPx);

            // Re-reproject using the true continuous background velocity.
            // FIX: same base -- the revoked candidate now lands back on its
            // OWN texel, so CASE 2 compares background against background
            // instead of landing on the previous dilation band's baked
            // crest depth (occludedByDilationZone) and rejecting.
            repro      = ReprojectToHistory(IN.uv0, stableInFrameUV, currentLayer.effectiveVelocityUV, previousCamera, vp);
            landing    = SnapUVToTexel(repro.sampleUV, vp);
            histMotion = SampleHistoryMotion(landing, vp);

            rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
            perspScale = rayLenPrev / max(rayLenCur, 1e-6);
        }
    }

    bool landingIsDilationZone = (histMotion.centerFlag >= 1.5);

    // Debug modes 8 and 6 fire after validation: transport the actual
    // verdict.
    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        float4 v = DebugViewEdgeState(currentColorRGB, dilationRevoked, currentLayer.isDilationZone, currentLayer.isForegroundEdge, currentLayer.isForeground, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
    {
        float4 v = DebugViewVelocity(histMotion.velocityUV, vp.sizePixels, 2.0, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    bool  depthRejected      = false;
    bool  velocityRejected   = false;
    float velocityErrorRatio = 0.0;

    if (depthTestEnabled)
    {
        float rawExp = currentLayer.effectiveDepth * perspScale;

        // Exact dual-sided subpixel slant tolerance:
        // Accounts for subpixel jitter phase on slanted/curved surfaces in
        // both current and history frames.
        float tolCurSlant  = (abs(currentLayer.gradX * pixel.fracPx.x) + abs(currentLayer.gradY * pixel.fracPx.y)) * perspScale;
        float tolHistSlant = abs(histMotion.gradX * landing.fracPx.x) + abs(histMotion.gradY * landing.fracPx.y);
        float tolSlant     = tolCurSlant + tolHistSlant;
        float tolDepth     = max(taaDepthRejection * rawExp, 1e-5) + tolSlant;

        if (currentLayer.isForeground)
        {
            // CASE 1: Current pixel is FOREGROUND (Silhouette Edge or Confirmed
            // Dilation Zone). Symmetrized Physical Depth Envelope.
            // Because the stored field bakes the dilated crest depth into last
            // frame's dilation zones, the landing sample alone resolves the
            // envelope: a landing in the object's previous dilation band reads
            // foreground depth directly (NOT disoccluded), a landing on pure
            // background reads background depth (the object was not there).
            float curLimb  = min(currentLayer.effectiveDepth, currentLayer.closestDepth) * perspScale;
            float curCrest = max(currentLayer.effectiveDepth, currentLayer.closestDepth) * perspScale;

            bool occludedInFront = (histMotion.depth > curCrest + tolDepth); // an occluder was present at t-1
            bool missingBehind   = (histMotion.depth < curLimb - tolDepth);  // the object was not present at t-1

            depthRejected = occludedInFront || missingBehind;
        }
        else
        {
            // CASE 2: Current pixel is BACKGROUND / CONTINUOUS SURFACE
            // Contract: "a dilation zone gets replaced with background on the
            // next frame this pixel is disoccluded." Only rejected if the
            // current pixel is strictly farther than the historical
            // occluder/dilation zone.
            bool occludedByDilationZone = landingIsDilationZone && (histMotion.depthMax > rawExp + tolDepth);
            bool occludedByGeometry     = (histMotion.depth - rawExp) > tolDepth;

            depthRejected = occludedByDilationZone || occludedByGeometry;
        }
    }

    // Velocity rejection is restricted to continuous background surfaces ON THE
    // CURRENT SIDE ONLY: silhouette edges experience motion boundary cliffs
    // and angular parallax acceleration; their presence is validated by the
    // geometric depth envelope, not by velocity continuity.
    //
    // FIX -- THE HISTORY SIDE IS NO LONGER FLAG-GATED. Skipping the test when
    // the landing quad contained any foreground-owned stored texel disabled
    // the test exactly on its canonical cases: thin / curved / edge-flagged
    // occluders (wires, poles, characters -- their stored texels are all
    // flag 1/2), so a revealed background pixel's landing always read a
    // flagged texel and velocity disocclusion never fired where it mattered
    // (the depth test misses those at similar depths, where its threshold is
    // deliberately loose to spare curved flanks). The layer-gated
    // reconstruction already handles the mixed case correctly: a
    // background-centered landing excludes foreground-owned taps (the
    // comparison stays background-vs-background -- no silhouette
    // velocity-cliff mixing), while a foreground-OWNED centered landing
    // reconstructs the OCCLUDER's stored velocity, so the error term becomes
    // the relative motion and the reveal rejects. Pixels landing in stale
    // dilation bands were already rejected by the depth test's
    // occludedByDilationZone; the extra coverage is strictly the previously
    // missed disocclusions.
    if (!depthRejected && velocityTestEnabled && !currentLayer.isForeground)
    {
        float2 errVecPx = (currentLayer.effectiveVelocityUV - histMotion.velocityUV - jitterCancelUV) * vp.sizePixels;
        float  errPx    = length(errVecPx);

        velocityErrorRatio = errPx / max(taaVelRejection, 1e-4);

        float maxGrad = max(currentLayer.layerVelSpreadPx, histMotion.spreadPx);
        float velTol  = taaVelRejection + maxGrad * landing.fracDist;

        if (errPx > velTol)
        {
            if (taaCrossTestStrength > 0.001)
            {
                // Exact round-trip pursuit confirmation. On coherent motion:
                //   landing    = g - m
                //   storedVel  = a_{t-1} - a_{t-2} - m
                //   transport  = a_{t-2} - a_{t-1}
                //   pursuitUV  = g - m - (storedVel + transport) = g
                // FIX: the round trip returns EXACTLY to the stable pixel
                // IN.uv0 (it previously chased currentJitteredUV, leaving a
                // systematic jitter-magnitude bias in the divergence).
                float2 pursuitUV = repro.sampleUV - (histMotion.velocityUV + jitterTransportUV);
                if (all(pursuitUV >= vp.minUV) && all(pursuitUV <= vp.maxUV))
                {
                    float roundTripDivergencePx = length((pursuitUV - IN.uv0) * vp.sizePixels);
                    velocityRejected = (roundTripDivergencePx * saturate(taaCrossTestStrength) > (1.0 + landing.fracDist));
                }
                else
                {
                    velocityRejected = true;
                }
            }
            else
            {
                velocityRejected = true;
            }
        }
    }

    bool disoccluded = depthRejected || velocityRejected;

    // Debug modes 2 and 12 fire after the tests: transport the actual
    // verdict.
    if (taaDebugMode > 1.5 && taaDebugMode < 2.5)
    {
        float4 v = DebugViewDisocclusionBreakdown(
            currentColorRGB, depthRejected, velocityRejected,
            (velocityErrorRatio > 1.0 && !velocityRejected),
            centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    if (taaDebugMode > 11.5 && taaDebugMode < 12.5)
    {
        float2 residualPx = (currentLayer.effectiveVelocityUV - histMotion.velocityUV - jitterCancelUV) * vp.sizePixels;
        float4 v = float4(saturate(length(residualPx) * 0.5), saturate(length(residualPx) * 0.5), 0.0, 0.0);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    // Debug mode 9: the dilation gate breakdown.
    //   R = revoked candidate
    //   G = kept candidate (full = via flag branch, half = via depth branch)
    //   B = depth test rejected this pixel
    // Red ring at a silhouette's outside edge = revocation working.
    // Cyan (G+B) = kept AND rejected -- a gate/test divergence.
    // Blue only = rejected non-candidate -- the boundary texel's own CASE 1.
    if (taaDebugMode > 8.5 && taaDebugMode < 9.5)
    {
        float3 debugColor = currentColorRGB * 0.1;
        if (dilationRevoked)                                  debugColor.r = 1.0;
        if (dilationCandidate && !dilationRevoked)
            debugColor.g = gateViaFlag ? 1.0 : 0.5;
        if (depthRejected)                                    debugColor.b = 1.0;
        return float4(debugColor, TransportAlpha(dilationRevoked, centerDepthRaw));
    }

    // ------------------------------------------------------------------
    // 9) History Color Reconstruction (Slepian Suite)
    // ------------------------------------------------------------------
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized, pixel.fracPx);
    float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

    float3 historyColorSpace =
        (taaUseSlepian3 > 0.5)
        ? SampleHistoryColor_Slepian3_Fused21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV)
        : SampleHistoryColor_Slepian2_Fast9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    historyColorSpace = CompressGamut(historyColorSpace);

    if (taaDebugMode > 4.5 && taaDebugMode < 5.5)
    {
        float4 v = DebugViewHistoryColor(historyColorSpace, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    // ------------------------------------------------------------------
    // 10) Color Clipping & Luma Drift Correction
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
    // 11) Feedback & Temporal Blending
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, shadowRisk, clipDistanceRejection, !currentLayer.isForeground);
    float currentBlendWeight = disoccluded ? 1.0 : (1.0 - historyFeedback);

    float3 currentFrameColorSpace = currentColorSpace;
    if (fxaaEnabled)
    {
        float fxaaWeight = ComputeFXAAFilterWeight(currentBlendWeight);
        if (fxaaWeight > 0.001)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 12) Final Blend Output (alpha = acutance metric for the auto-parity
    //     sharpener in taaFinal.fx.hlsl, SIGN-ENCODED with the revocation
    //     bit for the motion writer: negative = this pixel's tentative
    //     dilation was revoked)
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    float outAlpha = dilationRevoked
        ? -(rawSharpnessEnergy + kRevokedAlphaEpsilon)
        : rawSharpnessEnergy;
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}