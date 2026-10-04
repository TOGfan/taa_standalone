// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect -- resolve pass
// ----------------------------------------------------------------------------
// CONTEXT: sub-pixel jitter by PHYSICALLY ROTATING the camera; the per-frame
// bases (P/Q/R) contain the jitter; the velocity buffers contain the jitter
// motion; every jitter-free comparison subtracts s_t - 2*s_{t-1} + s_{t-2}
// (all s in the forward-map-minus-identity sense; see EstimateJitter*).
// The history buffer stores the previous frame's STABLE output.
//
// BASIS / MAP DIRECTIONS: see taaShared.h.hlsl. The resolve SAMPLES and
// REPROJECTS through the INVERSE map (static landings = the identity); the
// forward map is only the s_t offset source.
//
// STORED MOTION FIELD (#TAA_HistMotion, written by taaMotion.fx.hlsl):
//   Per stable texel: the previous frame's RAW velocity (xy) / depth (z) +
//   flag (w), with kept dilation bands carrying the crest's raw sample and
//   revoked candidates reverted to their own raw background sample.
//
// LAYER CLASSIFICATION (merged):
//   * The DECISION (dilation candidate / foreground edge / flat) is the OLD
//     depth-curvature classifier (AnalyzeSurfaceEdgesCore), shared by the
//     current frame, the history landing, the pursuit landing and the writer.
//   * The EFFECTIVE VALUES keep the refactor's upgrades (the foreground-edge
//     PHASE rule, the crest-raw depth anchoring) plus the PORTED two-tap
//     similarity extrapolation for the velocity (see [PORT] below).
//   * DILATION REVOCATION is kept: a candidate validates through the history
//     it would have if it stayed dilated -- its own stored texel (the ANCHOR
//     landing: the crest's raw velocity) -- via a LAYER-MATCHED support
//     gate. Revocation downgrades to the texel's own raw background sample
//     and PERSISTS through the stored field.
//   * The history landing is classified with the SAME rules and additionally
//     respects the stored flag as the POST-VALIDATION ownership record
//     ([FIX 9]): a background-owned center resolves strictly as background.
//
// TWO-TAP SIMILARITY VELOCITY EXTRAPOLATION ([PORT] from the older lineage):
//   A dilation zone acts as the foreground it dilates to, and its velocity is
//   that layer's FIELD EVALUATED AT THIS PIXEL. Interpolation cannot reach
//   it, so the field is extrapolated from two same-surface foreground taps
//   under a similarity (rotation + uniform scale) prior -- two point
//   correspondences determine a 2D similarity exactly, and it reproduces the
//   perpendicular/rotational gradient of the parallax field that a plain
//   linear extrapolation misses. Treating 2D px vectors as complex numbers,
//   with tap offsets f1, f2 and velocities v1, v2:
//       q    = (v2 - v1) / (f2 - f1)      complex quotient: rotation+scale
//       v(B) = v1 + q * (B - f1)          B = exact sub-texel position
//   BOUNDED GAIN by construction: |v(B) - v1| = |q| * |B - f1| exactly.
//   JITTER-CLEAN: q is a same-frame difference; static scenes give q ~ 0,
//   so the identity landing and the gate's anchor are bit-stable. Gates:
//   the geometric same-surface rule and the rigid parallax magnitude
//   ceiling -- both measured -- return the previous behavior (the closest
//   tap's raw velocity / the own center sample) on any failure. Applied at
//   the DILATION branch and the FOREGROUND-EDGE cliff phase, at every site
//   (current frame, history landing, pursuit) via the shared
//   ClassifyLayerSurface. kept
//   candidates' tests and history color use the extrapolated landing, with
//   the shift charged into the depth reach and the velocity advection
//   bound. The writer is unchanged: the stored band is a plateau of the
//   crest's raw sample, so the landing-side extrapolation degenerates to
//   the anchor value there automatically (w ~ 0 -> q ~ 0).
//
// OUTPUT: RGB = resolved color. A = the raw scene's acutance energy for
// taaFinal's auto-parity sharpener, SIGN-ENCODED with the revocation bit
// (negative = this pixel's tentative dilation was revoked).
//
// DEBUG MODES (taaDebugMode):
//   0 off | 1 frame motion | 2 disocclusion breakdown (R=depth, G=velocity,
//   B=suppressed pursuit alert) | 3 center velocity | 4 linearized depth |
//   5 history color | 6 landing effective velocity | 7 pursuit divergence |
//   8 layer state (orange=revoked candidate, red=kept dilation zone, cyan=
//   crest, magenta=depth-flat but velocity-straddled) | 9 dilation-gate
//   breakdown (R=revoked, G=kept candidate: full=flag branch, half=depth
//   branch, B=depth-rejected) | 10 alignment-drop activity | 12 dejittered
//   residual (jitter-cancel verification).
//
// CHANGELOG: [MERGE] classification reverted to the old curvature rules;
// [KEPT] dilation revocation + the foreground-edge phase rule; [RESTORE]
// velocity disocclusion + the kappa-corrected depth score (with depthGate);
// [FIX 9] the revocations apply to the history side (the landing's
// ownership gate); [PORT] the two-tap similarity velocity extrapolation
// (this entry); [KEPT] infra: inverse-map snap & reprojection base, fixed
// jitter offset signs, RotationFlowUV prev-2, Slepian filters, acutance
// alpha, FXAA corners.
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"

uniform_sampler2D(sceneTex,         0);
uniform_sampler2D(depthTex,         1);
uniform_sampler2D(historyTex,       2);
uniform_sampler2D(velocityTex,      3);
uniform_sampler2D(historyMotionTex, 4); // previous frame's motion field

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

// Smooth-field (continuous parallax) advection handling ([FIX 4..6]): the
// advection error is PREDICTED and subtracted (first-order Jacobian
// correction), the residual budget covers second-order terms only.
static const float kVelDiscontinuityRatio    = 0.65;    // curvature/pair-gradient ratio that still counts as continuous
static const float kVelDiscontinuityAbsPx    = 0.25;    // px: velocity-curvature noise floor
static const float kVelJacobianResidualFrac  = 0.25;    // fraction of |J|*miss kept for 2nd-order terms
static const float kPursuitSnapPadPx         = 1.5;     // px: fixed part of the residual pad
static const float kPursuitMaxAdvectionPx    = 12.0;    // px: hard clamp on the RESIDUAL budget only

// Layer-aware velocity selection (depth-invisible boundaries).
static const float kQuadStepGradMul          = 4.0;     // quad step must exceed coherentGrad * this to commit
static const float kMissCapHeadroom          = 1.5;     // same-surface miss cap headroom (Jacobian underestimate insurance)

// Velocity quantization fallback floor for the extrapolation gates: the
// quantization step is measured per neighborhood (MeasureVelocityQuantStepPx);
// this is the fallback when no step is observable (a locally constant field
// reads identical bins). For an fp16 RG velocity buffer at 1080p the rounding
// step at ~100 px motion is ~0.06 px; set this to your format's encoding step
// if it is larger.
static const float kVelQuantFloorPx          = 0.0625;

// Dilation-revocation gate: landing within this fraction of the texel center
// counts as ON the center (the sign(frac) quad selection there is decided by
// reprojection residual noise, not real subpixel position). ~20x the residual.
static const float kCenterLandingFracPx      = 0.1;

// Sign-bit transport of the revocation flag through the alpha channel:
// negative = revoked.
static const float kRevokedAlphaEpsilon      = 1e-30;

// Debug views
static const float kDebugVelocityScale       = 0.1;
static const float kDebugLinearDepthRange    = 100.0;

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
// REVOCATION TRANSPORT
// ----------------------------------------------------------------------------
// EVERY return path must transport the revocation bit through the alpha sign
// -- including debug returns. The writer ignores the bit for non-candidates,
// so a conservative "revoked" is safe on returns that fire before validation.
// ============================================================================
float TransportAlpha(bool revoked, float magnitude)
{
    return revoked ? -(magnitude + kRevokedAlphaEpsilon) : magnitude;
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
// CAMERA BASIS (from cbuffer) & JITTER FLOW
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

// Exact rotational flow offset of a yaw/pitch rotation at uv. In the shared
// offset sense (s_tau = F_tau^{-1}(g) - g) this is exactly s_{t-2} when fed
// the t-2 jitter angles -- position-exact. VERIFY with debug mode 12 on a
// static scene: the residual must collapse to noise -- if it is large, negate
// the angles on the C++ side (or negate this term here).
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

float RayLengthFromUV(float2 uv, float tanHalfFovX, float tanHalfFovY)
{
    float2 tanXY = float2((uv.x * 2.0 - 1.0) * tanHalfFovX, (1.0 - uv.y * 2.0) * tanHalfFovY);
    return sqrt(1.0 + dot(tanXY, tanXY));
}

// ============================================================================
// FALLBACK FXAA (strict viewport-clamped; center + preloaded corner taps)
// ============================================================================
// cornersRGB: [0]=NW(-1,-1), [1]=NE(+1,-1), [2]=SW(-1,+1), [3]=SE(+1,+1)
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
// Clamp a resampled color to the min/max of the taps it was built from, with a
// small overshoot margin. Shared by both history filters.
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
// LAYER-COHERENT VELOCITY FIELD ANALYSIS (layer-aware quad selection)
// ----------------------------------------------------------------------------
// The velocity test's NATIVE layer semantics ([FIX 8]): which texels belong
// to the center's layer is decided by velocity coherence, not by depth.
// ============================================================================
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
// differences -- the disocclusion signal itself -- are excluded.
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

// Does the phase-selected 2x2 quad straddle a velocity layer step? BOTH
// required: (a) the 3x3 field is DISCONTINUOUS here; (b) the step is INSIDE
// the selected quad and beyond what the center's own (velocity-coherent)
// layer explains -- never the step itself ([FIX 3]).
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
// rendered layer (the center tap).
float2 SelectLayerAwareQuadVelocity(float2 velocityJitteredUV[9], float2 phasePx, float2 sizePixels, float coherenceRadiusPx)
{
    float2 v00, v10, v01, v11;
    SelectPhaseQuad(velocityJitteredUV, phasePx, v00, v10, v01, v11);

    if (QuadStraddlesVelocityStep(v00, v10, v01, v11, velocityJitteredUV, sizePixels, coherenceRadiusPx))
        return velocityJitteredUV[0];   // the pixel's own rendered layer ([FIX 8a])

    return Bilerp2x2(v00, v10, v01, v11, abs(phasePx));
}

float2 BilerpVelocityQuad(float2 velocityUV[9], float2 fracPx)
{
    float2 v00 = velocityUV[0];
    float2 v10 = (fracPx.x >= 0.0) ? velocityUV[4] : velocityUV[3];
    float2 v01 = (fracPx.y >= 0.0) ? velocityUV[2] : velocityUV[1];
    float2 v11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? velocityUV[8] : velocityUV[6])
        : ((fracPx.y >= 0.0) ? velocityUV[7] : velocityUV[5]);
    return Bilerp2x2(v00, v10, v01, v11, abs(fracPx));
}

// ============================================================================
// MEASURED PAIR RULES & THE TWO-TAP SIMILARITY EXTRAPOLATION ([PORT])
// ----------------------------------------------------------------------------
// The gates and the extrapolation from the older lineage, verbatim in spirit:
//   * ForegroundPairSameSurface: two foreground taps are ONE surface iff
//     their raw gap is explained by the surface's own measured slope over
//     their distance, plus quantization.
//   * ForegroundPairRigidMagnitude: a pair's velocity difference is
//     legitimate iff it is explained by parallax through the pair's OWN
//     relative depth gap, plus the field's coherent gradient, plus the
//     measurement floor. A self-consistent rigid-motion test.
//   * MeasureShallowForegroundGeometry: the foreground object's shallow-end
//     geometry from ADJACENT tap pairs (span <= sqrt2). On a DILATION zone
//     the center is the background -- strictly foreground-side only; on a
//     CREST / flat pixel the center IS the surface's silhouette-side tap --
//     foreground-INCLUSIVE (the center itself joins the set).
//   * ResolveForegroundFieldVelocityUV: the complex-quotient extrapolation
//     (see the header's [PORT] note). Gate failures return the anchor tap's
//     raw velocity -- the exact pre-port behavior.
// ============================================================================
float2 ComplexMul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

bool ForegroundPairSameSurface(
    float rawA, float2 offA, float rawB, float2 offB,
    float slopePerTexel, float depthQuantStep)
{
    float spanPx = max(length(offA - offB), 1.0);
    return abs(rawA - rawB) <= slopePerTexel * spanPx + 2.0 * depthQuantStep;
}

bool ForegroundPairRigidMagnitude(
    float2 wPx, float2 dPx, float fgVelocityPx,
    float rawA, float rawB, float coherentGradPx, float velQuantPx)
{
    float spanPx = max(sqrt(dot(dPx, dPx)), 1e-3);
    float relGap = abs(rawA - rawB) / max(min(rawA, rawB), 1e-6);
    float ceilingPx = coherentGradPx + fgVelocityPx * relGap / spanPx;
    return length(wPx) <= ceilingPx * spanPx + 2.0 * velQuantPx;
}

// The smallest nonzero pairwise velocity step in the 3x3, in px: the honest
// local noise scale of the buffer. On a quantized encoding this IS the
// encoding step; on a locally constant field nothing is observable and the
// format floor applies.
float MeasureVelocityQuantStepPx(float2 velocityJitteredUV[9], float2 sizePixels)
{
    float minStepPx = kLargeValue;
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        [unroll]
        for (int j = i + 1; j < 9; ++j)
        {
            float stepPx = length((velocityJitteredUV[j] - velocityJitteredUV[i]) * sizePixels);
            if (stepPx > 1e-6)
                minStepPx = min(minStepPx, stepPx);
        }
    }
    return (minStepPx < kLargeValue) ? minStepPx : kVelQuantFloorPx;
}

void MeasureShallowForegroundGeometry(
    float  depthRaw[9], float2 velocityJitteredUV[9], float2 sizePixels,
    float  fgCoherentGradPx, float velQuantPx,
    bool   centerOnObject,
    out float shallowSlope, out float shallowVelGradPx)
{
    shallowSlope     = 0.0;
    shallowVelGradPx = 0.0;

    [unroll]
    for (int a = 0; a < 9; ++a)
    {
        if (centerOnObject ? (depthRaw[a] < depthRaw[0])   // behind the crest's own surface
                           : (depthRaw[a] <= depthRaw[0]))  // not foreground-side of the background
            continue;
        [unroll]
        for (int b = a + 1; b < 9; ++b)
        {
            if (centerOnObject ? (depthRaw[b] < depthRaw[0])
                               : (depthRaw[b] <= depthRaw[0]))
                continue;
            float2 spanVec = kOffsets3x3[b] - kOffsets3x3[a];
            float  spanPx  = length(spanVec);
            if (spanPx > 1.5) continue;             // adjacent pairs only
            float  gap  = abs(depthRaw[a] - depthRaw[b]);
            float2 wPx  = (velocityJitteredUV[b] - velocityJitteredUV[a]) * sizePixels;
            float  vAPx = length(velocityJitteredUV[a] * sizePixels);
            if (!ForegroundPairRigidMagnitude(wPx, spanVec, vAPx, depthRaw[a], depthRaw[b],
                                              fgCoherentGradPx, velQuantPx))
                continue;
            shallowSlope     = max(shallowSlope, gap / spanPx);
            shallowVelGradPx = max(shallowVelGradPx, length(wPx) / spanPx);
        }
    }
}

// The foreground layer's velocity field evaluated at an exact position, via
// the two-tap similarity (rotation + uniform scale) prior:
//     q = (v2 - v1) / (f2 - f1)  as a complex quotient;  v(B) = v1 + q*(B - f1)
// Bounded gain EXACTLY (|v(B) - v1| = |q| * |B - f1|); jitter-clean (q is a
// same-frame difference; static scenes give q ~ 0). Any gate failure returns
// the anchor tap's raw velocity and zeroes the diagnostics.
//   * DILATION call:  anchor = the closest (crest) tap, second = the
//     second-closest, and the second must additionally be foreground-side of
//     the background center (secondMustBeCloserThanRef / secondRefRaw).
//   * EDGE cliff-phase call: anchor = the center itself; the same-surface
//     gate (slope reference = the center-inclusive shallow measurement)
//     admits the along-silhouette pairs -- which under the similarity prior
//     determine the full first-order field -- and rejects steep interior
//     steps (indistinguishable from a near-layer gap; the fallback holds).
float2 ResolveForegroundFieldVelocityUV(
    float2 anchorOffsetPx,  float  anchorDepthRaw,  float2 anchorVelocityUV,
    float2 secondOffsetPx,  float  secondDepthRaw,  float2 secondVelocityUV,
    bool   secondMustBeCloserThanRef,
    float  secondRefRaw,
    float2 evalPx,
    float  slopePerTexel,
    float  depthQuantStep,
    float  coherentGradPx,
    float  velQuantPx,
    float2 sizePixels,
    out float pairGradPx,       // |w|/|d| of the active pair (0 = fallback)
    out float dispPx)           // |q|*|eval-anchor| of the active extrapolation
{
    pairGradPx = 0.0;
    dispPx     = 0.0;

    if (secondMustBeCloserThanRef && !(secondDepthRaw > secondRefRaw))
        return anchorVelocityUV;

    if (!ForegroundPairSameSurface(anchorDepthRaw, anchorOffsetPx,
                                   secondDepthRaw, secondOffsetPx,
                                   slopePerTexel, depthQuantStep))
        return anchorVelocityUV;

    float2 dPx   = secondOffsetPx - anchorOffsetPx;
    float2 wPx   = (secondVelocityUV - anchorVelocityUV) * sizePixels;
    float  dSqPx = dot(dPx, dPx);
    if (dSqPx < 1e-4)                            // coincident taps: no pair geometry
        return anchorVelocityUV;

    float fgVelocityPx = length(anchorVelocityUV * sizePixels);
    if (!ForegroundPairRigidMagnitude(wPx, dPx, fgVelocityPx,
                                      anchorDepthRaw, secondDepthRaw,
                                      coherentGradPx, velQuantPx))
        return anchorVelocityUV;

    // q = w / d as a complex quotient: w * conj(d) / |d|^2.
    float2 q = ComplexMul(wPx, float2(dPx.x, -dPx.y)) / dSqPx;

    float2 evalOffsetPx = evalPx - anchorOffsetPx;
    pairGradPx = length(wPx) / sqrt(dSqPx);
    dispPx     = length(ComplexMul(q, evalOffsetPx));

    float2 vPx = anchorVelocityUV * sizePixels + ComplexMul(q, evalOffsetPx);
    return vPx * (1.0 / sizePixels);
}

// ============================================================================
// MERGED LAYER RESOLUTION (old classification + kept upgrades + the [PORT])
// ============================================================================
float Minmod(float a, float b)
{
    return (a * b > 0.0) ? ((abs(a) < abs(b)) ? a : b) : 0.0;
}

void ComputeSurfaceGradients(float depthRaw[9], out float gradX, out float gradY)
{
    float centerDepth = depthRaw[0];
    float dxL = centerDepth - depthRaw[3];
    float dxR = depthRaw[4] - centerDepth;
    gradX = Minmod(dxL, dxR);

    float dyD = centerDepth - depthRaw[1];
    float dyU = depthRaw[2] - centerDepth;
    gradY = Minmod(dyD, dyU);
}

struct LayerSurface
{
    bool   isDilationZone;     // OLD classification
    bool   isForegroundEdge;   // OLD classification
    bool   isForeground;       // dilation || edge (cleared on revocation)
    bool   edgeTowardCliff;    // KEPT upgrade: phase rule selector
    int    closestIdx;
    float  closestDepth;
    float  gradX, gradY;       // minmod (cliff plane + sub-pixel edge depth)
    float  effectiveDepth;
    float2 effectiveVelocityUV;
    // Two-tap similarity extrapolation diagnostics (all 0 when inactive /
    // revoked -- zero tolerance charges, the exact pre-port behavior):
    float  pairGradPx;           // the active pair's per-texel velocity gradient
    float  shallowVelGradPx;     // the foreground's measured shallow-end velocity gradient
    float  extrapolationDispPx;  // |q|*|eval-anchor|: the landing shift it causes
};

LayerSurface ClassifyLayerSurface(
    float  depthRaw[9],
    float2 velocityUV[9],
    float2 fracPx,
    float2 sizePixels,
    float  coherenceRadiusPx,
    bool   useDepthDilation,
    bool   isLandingSite,
    SurfaceEdgeState edge)
{
    LayerSurface s;
    s.isDilationZone      = edge.isDilationZone;
    s.isForegroundEdge    = edge.isForegroundEdge;
    s.isForeground        = s.isDilationZone || s.isForegroundEdge;
    s.pairGradPx          = 0.0;
    s.shallowVelGradPx    = 0.0;
    s.extrapolationDispPx = 0.0;

    ComputeSurfaceGradients(depthRaw, s.gradX, s.gradY);

    // Closest + second-closest (foreground) tap scan -- reverse-Z: larger
    // raw = nearer.
    s.closestDepth = depthRaw[0];
    s.closestIdx   = 0;
    float secondDepth = 0.0;
    int   secondIdx   = 0;
    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        if (depthRaw[i] > s.closestDepth)
        {
            secondDepth   = s.closestDepth;
            secondIdx     = s.closestIdx;
            s.closestDepth = depthRaw[i];
            s.closestIdx   = i;
        }
        else if (depthRaw[i] > secondDepth)
        {
            secondDepth = depthRaw[i];
            secondIdx   = i;
        }
    }
    float2 secondOffset = kOffsets3x3[secondIdx];

    // Cliff sides: which sides of the texel hold an off-layer FARTHER
    // (background) neighbor, below the minmod plane by edgeEps. The phase
    // test is PER-AXIS (diagonals caught, ridges fall to the point sample).
    bool cliffPosX = false, cliffNegX = false, cliffPosY = false, cliffNegY = false;
    [unroll]
    for (int k = 1; k < 9; ++k)
    {
        float predDepth = depthRaw[0] + s.gradX * kOffsets3x3[k].x + s.gradY * kOffsets3x3[k].y;
        if (depthRaw[k] < predDepth - edge.edgeEps)
        {
            if (kOffsets3x3[k].x > 0)      cliffPosX = true;
            else if (kOffsets3x3[k].x < 0) cliffNegX = true;
            if (kOffsets3x3[k].y > 0)      cliffPosY = true;
            else if (kOffsets3x3[k].y < 0) cliffNegY = true;
        }
    }
    s.edgeTowardCliff = s.isForegroundEdge &&
        ( (cliffPosX && fracPx.x > 0.0) || (cliffNegX && fracPx.x < 0.0) ||
          (cliffPosY && fracPx.y > 0.0) || (cliffNegY && fracPx.y < 0.0) );

    // Extrapolation inputs, measured only where the extrapolation can run
    // (dilation zones and crest cliff-phases; everything else pays nothing).
    float velQuantPx     = kVelQuantFloorPx;
    float shallowSlope   = 0.0;
    float shallowVelGrad = 0.0;
    float fgCoherentGrad = 0.0;
    if (s.isDilationZone || s.edgeTowardCliff)
    {
        velQuantPx = MeasureVelocityQuantStepPx(velocityUV, sizePixels);
        float gateRadiusPx = max(taaVelRejection, velQuantPx);
        fgCoherentGrad = MeasureVelocityCoherentGradientPx(
            velocityUV[s.closestIdx], velocityUV, sizePixels, gateRadiusPx);
        MeasureShallowForegroundGeometry(
            depthRaw, velocityUV, sizePixels,
            fgCoherentGrad, velQuantPx,
            !s.isDilationZone,          // centerOnObject
            shallowSlope, shallowVelGrad);
    }
    s.shallowVelGradPx = shallowVelGrad;

    // Effective values.
    float subpixelDepth = depthRaw[0] + s.gradX * fracPx.x + s.gradY * fracPx.y;

    if (s.isDilationZone)
    {
        // Background behind a foreground crest: the pixel acts as the
        // FOREGROUND, and its velocity is that layer's field EVALUATED AT
        // THIS PIXEL ([PORT]: the two-tap similarity extrapolation; gate
        // failures fall back to the closest tap's raw sample -- the exact
        // pre-port behavior).
        s.effectiveDepth = s.closestDepth;
        float2 anchorOff = kOffsets3x3[s.closestIdx];
        s.effectiveVelocityUV = ResolveForegroundFieldVelocityUV(
            anchorOff, s.closestDepth, velocityUV[s.closestIdx],
            secondOffset, secondDepth, velocityUV[secondIdx],
            true, depthRaw[0],      // the second tap must be foreground-side of the background center
            fracPx,
            shallowSlope, edge.depthQuantStep,
            fgCoherentGrad, velQuantPx,
            sizePixels,
            s.pairGradPx, s.extrapolationDispPx);
    }
    else if (s.edgeTowardCliff)
    {
        // Phase toward the silhouette ([PORT]): the content position hangs
        // at/past the limb; the velocity is the object's field at the exact
        // phase -- the same extrapolation anchored at the center, with the
        // second-closest tap. Gate failures fall back to the own center raw
        // sample (the previous behavior). The depth stays the own center.
        s.effectiveDepth = depthRaw[0];
        s.effectiveVelocityUV = ResolveForegroundFieldVelocityUV(
            float2(0.0, 0.0), depthRaw[0], velocityUV[0],
            secondOffset, secondDepth, velocityUV[secondIdx],
            false, 0.0,
            fracPx,
            shallowSlope, edge.depthQuantStep,
            fgCoherentGrad, velQuantPx,
            sizePixels,
            s.pairGradPx, s.extrapolationDispPx);
    }
    else if (s.isForegroundEdge)
    {
        // Phase toward the flat part: the on-object quad, full sub-pixel
        // treatment (KEPT upgrade).
        s.effectiveDepth      = isLandingSite ? Bilerp2x2(
            depthRaw[0],
            (fracPx.x >= 0.0) ? depthRaw[4] : depthRaw[3],
            (fracPx.y >= 0.0) ? depthRaw[2] : depthRaw[1],
            (fracPx.x >= 0.0)
                ? ((fracPx.y >= 0.0) ? depthRaw[8] : depthRaw[6])
                : ((fracPx.y >= 0.0) ? depthRaw[7] : depthRaw[5]),
            abs(fracPx)) : subpixelDepth;
        s.effectiveVelocityUV = BilerpVelocityQuad(velocityUV, fracPx);
    }
    else
    {
        // Flat: OLD depth semantics (center at the current frame / bilinear at
        // landing sites) + the OLD layer-aware velocity selection ([FIX 8a]).
        s.effectiveDepth      = isLandingSite ? Bilerp2x2(
            depthRaw[0],
            (fracPx.x >= 0.0) ? depthRaw[4] : depthRaw[3],
            (fracPx.y >= 0.0) ? depthRaw[2] : depthRaw[1],
            (fracPx.x >= 0.0)
                ? ((fracPx.y >= 0.0) ? depthRaw[8] : depthRaw[6])
                : ((fracPx.y >= 0.0) ? depthRaw[7] : depthRaw[5]),
            abs(fracPx)) : depthRaw[0];
        s.effectiveVelocityUV = SelectLayerAwareQuadVelocity(velocityUV, fracPx, sizePixels, coherenceRadiusPx);
    }
    return s;
}

// Shared post-validation downgrade: the texel's OWN raw background sample.
// The extrapolation diagnostics are zeroed -- no tolerance charges.
void RevokeDilation(inout LayerSurface s, float centerDepthRaw, float2 centerVelocityUV)
{
    s.isDilationZone      = false;
    s.isForegroundEdge    = false;
    s.isForeground        = false;
    s.edgeTowardCliff     = false;
    s.pairGradPx          = 0.0;
    s.shallowVelGradPx    = 0.0;
    s.extrapolationDispPx = 0.0;
    s.effectiveDepth      = centerDepthRaw;
    s.effectiveVelocityUV = centerVelocityUV;
}

// ============================================================================
// PIPELINE DATA STRUCTURES
// ============================================================================
// Raw, unmodified 3x3 current-frame samples plus the closest-depth tap scan.
// Kept RAW (no dilation, no interpolation) so the divergence derivatives in
// ComputeDepthDisocclusionScore stay clean.
struct CurrentFrameNeighborhood
{
    float  depthRaw[9];
    float2 velocityJitteredUV[9];

    float  closestDepthRaw;
    float  secondClosestDepthRaw;
    float2 closestOffsetPx;
    float2 secondClosestOffsetPx;
    float2 dilatedVelocityJitteredUV;
};

// Foreground crest geometry (feeds the disocclusion tolerances).
struct ForegroundGeometry
{
    float crestDrop;
    float slope;
};

// Reprojection of the resolved surface into the history buffer.
struct HistoryReprojection
{
    float2 sampleUV;            // where to read the history (previous STABLE output)
    float  prevCameraRayY;      // forward component of the previous-frame ray (depth scale)
    float2 subpixelPx;          // sub-texel offset of the history sample
    float2 subpixelAlignment;   // 1.0 at texel centers, 0 at texel corners
    float2 jitterResidualPx;    // current jitter minus history subpixel phase
    float2 motionPx;
    float  motionMagnitudePx;
    float  motionNormalized;    // saturate(motion / kMotionFullStrengthPx)
    float2 motionDirUnit;       // motion direction, or (1,0) when static
};

// The history landing resolved with the SAME layer semantics as the current
// frame (see SampleHistoryLandingSurface).
struct HistoryLandingSurface
{
    bool  isDilationZone;
    bool  isForegroundEdge;

    float  effectiveDepthRaw;
    float2 effectiveVelocityJitteredPrevUV;

    // Velocity-field shape at the landing (px).
    float maxCurvaturePx;
    float maxPairGradPx;
    float coherentGradPx;
    float coherentSpreadPx;
    float snapDistPx;

    // [PORT] extrapolation diagnostics (landing side).
    float pairGradPx;        // the active landing extrapolation pair's gradient
    float shallowVelGradPx;  // the landing's measured shallow-end velocity gradient
};

// Velocity rejection (pursuit) diagnostics.
struct VelocityRejectionResult
{
    bool  rejected;
    float errorRatio;
    float layerGradientPx;
    float divergencePx;
};

// Color neighborhood statistics for history clipping.
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

// ============================================================================
// CURRENT-FRAME GATHER
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
    // (reverse-Z). NOTE: this scan is identical to the one inside
    // ClassifyLayerSurface -- neighborhood.dilatedVelocityJitteredUV is
    // always the crest tap's raw velocity that ClassifyLayerSurface picked.
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

    // The foreground layer's true raw velocity (the ANCHOR -- consumed by the
    // dilation-revocation gate's landing).
    n.dilatedVelocityJitteredUV = n.velocityJitteredUV[closestIdx];
    return n;
}

SurfaceEdgeState AnalyzeSurfaceEdges(CurrentFrameNeighborhood neighborhood, bool useDepthDilation)
{
    return AnalyzeSurfaceEdgesCore(neighborhood.depthRaw, useDepthDilation);
}

ForegroundGeometry ComputeForegroundGeometry(
    CurrentFrameNeighborhood neighborhood, float resolvedDepthRaw, SurfaceEdgeState edge)
{
    ForegroundGeometry fg;

    float crestDrop   = max(neighborhood.closestDepthRaw - resolvedDepthRaw, 0.0);
    float crestSpanPx = max(length(neighborhood.closestOffsetPx), 1.0);
    float crestSlope  = crestDrop / crestSpanPx;

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
// ----------------------------------------------------------------------------
// frameBaseUV must be the CURRENT-FRAME position of the stable point (the
// INVERSE map): with it as the velocity base, the jitter components of the
// velocity cancel EXACTLY and a static scene lands at stableUV (the identity).
// (The ray construction is linear in uv, so base + velocity along the
// previous camera's screen axes is EXACTLY Ray_prev(frameBaseUV + velocity).)
// ============================================================================
HistoryReprojection ReprojectToHistory(
    float2 stableUV,
    float2 frameBaseUV,
    float2 currentJitterPx,
    float2 velocityJitteredUV,
    CameraBasis previousCamera,
    ViewportParams vp)
{
    HistoryReprojection h;

    float3 prevRayBase = frameBaseUV.x * previousCamera.rightTanFov
                       + previousCamera.forward
                       - frameBaseUV.y * previousCamera.downTanFov;
    float3 prevRay = prevRayBase
                   + velocityJitteredUV.x * previousCamera.rightTanFov
                   - velocityJitteredUV.y * previousCamera.downTanFov;

    h.sampleUV       = ProjectRayToStableUV(prevRay, frameBaseUV + velocityJitteredUV, taaTanHalfFovX, taaTanHalfFovY);
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

    // Sub-texel jitter still present after snapping both frames.
    h.jitterResidualPx = currentJitterPx - h.subpixelPx;
    return h;
}

// ============================================================================
// VELOCITY-FIELD JACOBIAN & ADVECTION BUDGET ([FIX 6]/[FIX 8])
// ============================================================================
// 2x2 Jacobian of the velocity field in px-per-texel, from layer-consistent
// differences.
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
// correction cannot absorb. Does NOT scale with the transport distance or the
// error's full magnitude ([FIX 3]); hard-clamped. [FIX 8e]: the miss term is
// capped at the predicted same-surface advection.
float ResidualAdvectionBudgetPx(float maxPairGradPx, float missPx, float snapDistPx, float missCapPx)
{
    float boundedMissPx = min(missPx, missCapPx);
    return min(maxPairGradPx * (kVelJacobianResidualFrac * boundedMissPx + snapDistPx + kPursuitSnapPadPx),
               kPursuitMaxAdvectionPx);
}

// Field shape with layer-consistent differences.
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
// DILATION-REVOCATION GATE READER (the stored quad at the candidate's landing)
// ----------------------------------------------------------------------------
// Own-History Dilation Validation. The landing's NEIGHBORHOOD record decides:
//   * DEAD-CENTER landing (|frac| < kCenterLandingFracPx per axis): the
//     support is the center texel alone -- the own-texel check. Static
//     landings are the EXACT identity (the inverse base cancels the jitter;
//     the extrapolation's static residual is ~0.002 px), so static
//     candidates validate through their OWN record: phase artifacts (own
//     texel flag 0) revoke and STAY revoked; kept-band texels (own flag 2)
//     stay kept. This alone carries the revocation persistence.
//   * OFF-CENTER landing: ANY tap of the degenerated 2x2 quad counts -- by
//     flag (>= 1: edge, or kept dilation band -- "dilated non-revoked
//     history counts as foreground") OR by depth (at the object,
//     crest-anchored). NO layer-matching: a foreign foreground tap DOES
//     validate. (The old layer-matched support gated the band's bootstrap
//     on the sub-texel motion phase: a landing on the background texel
//     just outside the band excluded the crest/band neighbor as foreign,
//     the band never formed there, and the verdict flickered with the
//     motion's fractional part.)
// gateDepth stays layer-gated: its only consumer (touchesAlreadyDilated)
// requires the center to be flag 2, where the gating reads crest depths.
// ============================================================================
struct HistoryMotionGate
{
    float  centerFlag;       // stored flag of the snapped landing texel
    float  gateDepth;        // layer-gated bilinear stored depth
    float  supportMaxFlag;   // support max flag (ANY quad tap)
    float  supportDepthMax;  // support max depth (ANY quad tap)
};

HistoryMotionGate SampleHistoryMotionGate(float2 historyUV, ViewportParams vp)
{
    HistoryMotionGate g;

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;

    float sx = (fracPx.x >= 0.0) ? 1.0 : -1.0;
    float sy = (fracPx.y >= 0.0) ? 1.0 : -1.0;

    float2 uv10 = clamp(snappedUV + float2(sx, 0.0) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv01 = clamp(snappedUV + float2(0.0, sy) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv11 = clamp(snappedUV + float2(sx, sy) * vp.texelSize, vp.minUV, vp.maxUV);

    float4 m00 = tex2Dlod(historyMotionTex, float4(snappedUV, 0.0, 0.0));
    float4 m10 = tex2Dlod(historyMotionTex, float4(uv10, 0.0, 0.0));
    float4 m01 = tex2Dlod(historyMotionTex, float4(uv01, 0.0, 0.0));
    float4 m11 = tex2Dlod(historyMotionTex, float4(uv11, 0.0, 0.0));

    // Layer-gated bilinear depth (gateDepth's consumer requires center flag 2).
    bool fgCenter = (m00.w >= 0.75);
    bool match10  = ((m10.w >= 0.75) == fgCenter);
    bool match01  = ((m01.w >= 0.75) == fgCenter);
    bool match11  = ((m11.w >= 0.75) == fgCenter);

    float2 f   = abs(fracPx);
    float  w00 = (1.0 - f.x) * (1.0 - f.y);
    float  w10 = f.x * (1.0 - f.y) * (match10 ? 1.0 : 0.0);
    float  w01 = (1.0 - f.x) * f.y * (match01 ? 1.0 : 0.0);
    float  w11 = f.x * f.y * (match11 ? 1.0 : 0.0);
    float  invW = 1.0 / max(w00 + w10 + w01 + w11, 1e-4);
    g.gateDepth  = (m00.z * w00 + m10.z * w10 + m01.z * w01 + m11.z * w11) * invW;
    g.centerFlag = m00.w;

    // The support: ANY tap of the degenerated 2x2 quad -- flag OR depth, no
    // layer-matching. Dead-center landings degenerate to the center alone
    // (the static own-texel check).
    bool xOffCenter = (f.x >= kCenterLandingFracPx);
    bool yOffCenter = (f.y >= kCenterLandingFracPx);
    float supportDepthMax = m00.z;
    float supportMaxFlag  = m00.w;
    if (xOffCenter)               { supportDepthMax = max(supportDepthMax, m10.z); supportMaxFlag = max(supportMaxFlag, m10.w); }
    if (yOffCenter)               { supportDepthMax = max(supportDepthMax, m01.z); supportMaxFlag = max(supportMaxFlag, m01.w); }
    if (xOffCenter && yOffCenter) { supportDepthMax = max(supportDepthMax, m11.z); supportMaxFlag = max(supportMaxFlag, m11.w); }
    g.supportDepthMax = supportDepthMax;
    g.supportMaxFlag  = supportMaxFlag;

    return g;
}

// ============================================================================
// HISTORY LANDING SURFACE (shared by the depth & velocity disocclusion tests)
// ----------------------------------------------------------------------------
// Classified and resolved with the SAME rules as the current frame ([FIX 4a]
// / [SEMANTICS] / [PORT]: the same extrapolation, on the stored field), PLUS
// the OWNERSHIP GATE ([FIX 9]): the stored flag is the POST-VALIDATION layer
// record. The stored band is a plateau of the crest's raw sample, so the
// landing-side extrapolation degenerates to the anchor value there
// automatically (w ~ 0 -> q ~ 0) -- the [SEMANTICS] behavior is unchanged.
// ============================================================================
// Ownership-gated bilinear depth at the landing ([FIX 9]): never blend across
// an ownership boundary.
float LandingOwnedDepth(float depths[9], float flags[9], float2 fracPx)
{
    float d00 = depths[0];
    float d10 = (fracPx.x >= 0.0) ? depths[4] : depths[3];
    float d01 = (fracPx.y >= 0.0) ? depths[2] : depths[1];
    float d11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? depths[8] : depths[6])
        : ((fracPx.y >= 0.0) ? depths[7] : depths[5]);
    float f00 = flags[0];
    float f10 = (fracPx.x >= 0.0) ? flags[4] : flags[3];
    float f01 = (fracPx.y >= 0.0) ? flags[2] : flags[1];
    float f11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? flags[8] : flags[6])
        : ((fracPx.y >= 0.0) ? flags[7] : flags[5]);

    float2 f = abs(fracPx);
    bool  fg = (f00 >= 0.75);
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y) * (((f10 >= 0.75) == fg) ? 1.0 : 0.0);
    float w01 = (1.0 - f.x) * f.y * (((f01 >= 0.75) == fg) ? 1.0 : 0.0);
    float w11 = f.x * f.y * (((f11 >= 0.75) == fg) ? 1.0 : 0.0);
    float invW = 1.0 / max(w00 + w10 + w01 + w11, 1e-4);
    return (d00 * w00 + d10 * w10 + d01 * w01 + d11 * w11) * invW;
}

HistoryLandingSurface SampleHistoryLandingSurface(
    float2 historyUV, ViewportParams vp, bool useDepthDilation, bool gather,
    float coherenceRadiusPx, bool currentIsForeground)
{
    HistoryLandingSurface h;
    h.isDilationZone   = false;
    h.isForegroundEdge = false;
    h.maxCurvaturePx   = 0.0;
    h.maxPairGradPx    = 0.0;
    h.coherentGradPx   = 0.0;
    h.coherentSpreadPx = 0.0;
    h.snapDistPx       = 0.0;
    h.pairGradPx       = 0.0;
    h.shallowVelGradPx = 0.0;

    // Minimal default when no disocclusion test needs the landing: the single
    // nearest tap (debug views only). The stored value there is already the
    // post-validation state, so this default is revocation-correct.
    float2 clampedUV = clamp(historyUV, vp.minUV, vp.maxUV);
    float4 mCenter   = tex2Dlod(historyMotionTex, float4(clampedUV, 0.0, 0.0));
    h.effectiveDepthRaw               = mCenter.z;
    h.effectiveVelocityJitteredPrevUV = mCenter.xy;
    if (!gather)
        return h;

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;    // sub-texel landing phase, [-0.5, 0.5]
    h.snapDistPx     = length(fracPx);

    float  depths[9];
    float2 velocities[9];
    float  flags[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapUV = clamp(snappedUV + kOffsets3x3[i] * vp.texelSize, vp.minUV, vp.maxUV);
        float4 m     = tex2Dlod(historyMotionTex, float4(tapUV, 0.0, 0.0));
        depths[i]     = m.z;    // previous frame's depth (raw / baked band / revoked)
        velocities[i] = m.xy;   // previous frame's velocity
        flags[i]      = m.w;    // post-validation layer record
    }

    // THE classifier on the stored structure.
    SurfaceEdgeState landingEdge = AnalyzeSurfaceEdgesCore(depths, useDepthDilation);

    // --- Ownership gate ([FIX 9]) --------------------------------------------
    bool centerFg = (flags[0] >= 0.75);
    if (!currentIsForeground && !centerFg)
    {
        // Background-resolved center (flat or REVOKED): resolve strictly as
        // background -- suppress the structural foreground pulls (and the
        // extrapolation with them).
        landingEdge.isDilationZone   = false;
        landingEdge.isForegroundEdge = false;
    }

    LayerSurface landingLayer = ClassifyLayerSurface(
        depths, velocities, fracPx, vp.sizePixels, coherenceRadiusPx,
        useDepthDilation, true, landingEdge);

    if (!currentIsForeground && !centerFg)
    {
        // The depth must never blend across the ownership boundary.
        landingLayer.effectiveDepth = LandingOwnedDepth(depths, flags, fracPx);
    }

    h.isDilationZone                  = landingLayer.isDilationZone;
    h.isForegroundEdge                = landingLayer.isForegroundEdge;
    h.effectiveDepthRaw               = landingLayer.effectiveDepth;
    h.effectiveVelocityJitteredPrevUV = landingLayer.effectiveVelocityUV;
    h.pairGradPx                      = landingLayer.pairGradPx;
    h.shallowVelGradPx                = landingLayer.shallowVelGradPx;

    // Field shape + velocity-coherent noise, anchored at the effective layer.
    // [FIX 8b]: layer-consistent differences.
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
// JITTER CANCELLATION & TRANSPORT
// ----------------------------------------------------------------------------
// Both terms are exact first-order identities of the rotational jitter flow.
// All offsets are in the shared sense: s_tau = the forward map of the stable
// pixel through frame tau's camera, minus the stable pixel.
//   * Cancel:    the jitter content of (vCur - vPrev) is s_t - 2 s_{t-1} + s_{t-2}.
//   * Transport: advancing the history surface one frame forward requires
//                subtracting s_t - s_{t-1} + s_{t-2} from vPrev, so the pursuit
//                lands EXACTLY on the surface's CURRENT-FRAME position.
// ============================================================================
float2 EstimateJitterCancelUV(float2 sCur, float2 sPrev, float2 sPrev2)
{
    return sCur - 2.0 * sPrev + sPrev2;
}

float2 EstimateJitterTransportUV(float2 sCur, float2 sPrev, float2 sPrev2)
{
    return sCur - sPrev + sPrev2;
}

// ============================================================================
// MOTION-COMPENSATED DEPTH DISOCCLUSION (RESTORED)
// ----------------------------------------------------------------------------
// Kappa-corrected transport: expected history depth = current depth, ray-
// normalized into the previous frame's parameterization, times (1-kappa),
// kappa estimated from the measured velocity divergence. One-sided (only a
// history surface CLOSER than the transported current surface rejects).
// Returns a continuous score; >= 1.0 rejects, and saturate((1-score)*2) is
// the depthGate. [PORT]: extrapolationDispPx charges the active velocity
// extrapolation's landing shift into the crest / dilation reach terms.
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
    float  extrapolationDispPx,
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

    // Plane gradient in tangent units.
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

    // --- Inter-layer velocity shear ------------------------------------------
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

    // Reject the shear estimate when its own noise is too large to be trusted.
    if (haveLayerShear)
    {
        float shearNoise       = 2.0 * velocitySpreadTan / max(layerSeparation, 1e-7);
        float shearNoiseScaled = resolvedDepthRaw * length(curTan) * shearNoise / curDenom;
        if (shearNoiseScaled > 0.25 * taaDepthRejection) haveLayerShear = false;
    }

    // --- Divergence-to-depth-scale correction (kappa) ------------------------
    float denomNaive = max(2.0 - curRadiusSq, 0.25);
    float kappaNaive = -(divergenceMeasured - 3.0 * radialTerm) / denomNaive;
    float kappa;
    float kappaSigma;

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
        // down the front face within the filter support. The extrapolation's
        // landing shift is charged into the reach.
        float reachPx = kHistoryTapReachPx
                      + (abs(jitterResidualPx.x) + abs(jitterResidualPx.y))
                      + 0.5 * length(velocitySpreadPx)
                      + extrapolationDispPx;
        allowances = (foregroundCrestDrop + foregroundSlope * reachPx) * invExpected;
    }
    else // dilation zone: background behind a crest
    {
        float reachPx = length(closestOffsetPx)
                      + kHistoryTapReachPx
                      + length(jitterResidualPx)
                      + 0.5 * length(velocitySpreadPx)
                      + extrapolationDispPx;
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
// diverges between the landing position and the current pixel. Both
// velocities are FIELD SAMPLES of the same content (the shared [PORT]
// extrapolation on both sides), so the coherent-motion round trip still
// cancels to ~0. The landing is resolved with ClassifyLayerSurface -- the
// SAME rules as everywhere else.
// ============================================================================
bool PursuitConfirmsDivergence(
    float2 historySampleUV,
    float2 prevVelocityEffectiveJitteredUV,
    float2 curVelocityJitteredUV,
    float2 jitterTransportUV,
    float2 missVecPx,             // dejittered step-1 error vector == the transport's landing miss ([FIX 6])
    float  depthGate,             // [0..1] depth-transport same-surface confidence ([FIX 5])
    float  coherenceRadiusPx,
    ViewportParams vp,
    out float divergencePx)
{
    divergencePx = 0.0;

    if (taaCrossTestStrength <= 0.001)
        return false;

    // 1) Exact landing: where the history surface is in the CURRENT render.
    float2 pursuitUV = historySampleUV - (prevVelocityEffectiveJitteredUV + jitterTransportUV);
    if (any(pursuitUV < vp.minUV) || any(pursuitUV > vp.maxUV))
        return false;

    // 2) 3x3 depth/velocity neighborhood at the landing, resolved with the
    //    same layer rules as the current frame.
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
    LayerSurface landingLayer = ClassifyLayerSurface(
        depths, velocities, fracPx, vp.sizePixels, coherenceRadiusPx,
        taaUseDepthDilation > 0.5, true, landingEdge);
    float2 landingVelocityJitteredUV = landingLayer.effectiveVelocityUV;

    // 3) Divergence VECTOR between the two current-frame velocities.
    float2 divergenceVecPx = (landingVelocityJitteredUV - curVelocityJitteredUV) * vp.sizePixels;

    // 3b) [FIX 6] first-order advection correction, [FIX 8b] layer-consistent
    //     estimators, [FIX 8d] no single-layer requirement, [FIX 8e] honest
    //     miss cap.
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

        float2 vLandingPx = landingVelocityJitteredUV * vp.sizePixels;
        float  missCapPx  = length(ApplyVelocityJacobian(Jlanding, vLandingPx)) * kMissCapHeadroom;
        residualPx = ResidualAdvectionBudgetPx(maxPairGradPx, length(missVecPx), length(fracPx), missCapPx) * depthGate;
    }

    divergencePx = length(correctedDivergenceVecPx);
    float tolerancePx = kPursuitVelBaseTolerancePx + landingSpreadPx + residualPx;

    return (divergencePx * saturate(taaCrossTestStrength)) > tolerancePx;
}

// ============================================================================
// VELOCITY REJECTION (step 2 of disocclusion; fully independent of depth)
// ============================================================================
VelocityRejectionResult EvaluateVelocityRejection(
    float2 historySampleUV,
    float2 resolvedVelocityJitteredUV,
    HistoryLandingSurface landing,
    bool   currentSingleLayer,
    float  currentDilationPairGradPx,   // [PORT] the current side's active extrapolation pair gradient
    float  currentShallowVelGradPx,     // [PORT] the current side's shallow-end velocity gradient
    float  depthGate,
    float2 jitterCancelUV,
    float2 jitterTransportUV,
    float2 neighborVelocityJitteredUV[9],
    float  coherenceRadiusPx,
    ViewportParams vp)
{
    VelocityRejectionResult r;
    r.rejected        = false;
    r.errorRatio      = 0.0;
    r.layerGradientPx = 0.0;
    r.divergencePx    = 0.0;

    if (taaVelRejection <= 0.001)
        return r;

    // Dejittered error VECTOR against the landing's EFFECTIVE layer. On a
    // same-surface pixel this vector IS the pursuit transport's landing miss
    // (identity), so it is passed through to the [FIX 6] correction.
    float2 velocityErrorVecPx = (resolvedVelocityJitteredUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
    float  velocityErrorPx    = length(velocityErrorVecPx);

    // Velocity-coherent gradients on both frame sides, combined and clamped.
    float currentGradientPx = MeasureVelocityCoherentGradientPx(
        resolvedVelocityJitteredUV, neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
    r.layerGradientPx = clamp(max(currentGradientPx, landing.coherentGradPx), 0.0, 1.0);

    float velocityNoise = r.layerGradientPx * taaVelGradientScale;

    // [FIX 8b] layer-consistent field shapes.
    float curMaxCurvaturePx, curMaxPairGradPx;
    MeasureVelocityFieldShapeCoherent(neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx, curMaxCurvaturePx, curMaxPairGradPx);
    bool currentContinuous = IsContinuousVelocityField(curMaxCurvaturePx, curMaxPairGradPx);
    bool landingContinuous = IsContinuousVelocityField(landing.maxCurvaturePx, landing.maxPairGradPx);

    // [FIX 6]/[FIX 8c] predicted same-surface advection for the alert
    // tolerance. For !currentSingleLayer, the bound takes the max over every
    // gradient source: the stored band replicates the crest's per-row
    // samples, so the landing-side pair gradient loses the perpendicular
    // component; the current side still measures both; and the [PORT]'s
    // ACTIVE extrapolation pairs + both sides' shallow-end geometry carry
    // the near-silhouette gradients -- the extrapolation's own delta (the
    // current field sample vs the landing's anchor value) is exactly this
    // scale, so without them kept dilation candidates would false-alert.
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
        float boundGradPx = max(max(landing.maxPairGradPx, curMaxPairGradPx),
                                max(max(landing.pairGradPx, currentDilationPairGradPx),
                                    max(landing.shallowVelGradPx, currentShallowVelGradPx)));
        float2 vEffPx      = resolvedVelocityJitteredUV * vp.sizePixels;
        float  predictedPx = boundGradPx * length(vEffPx);
        float  residualPx  = ResidualAdvectionBudgetPx(
            boundGradPx, velocityErrorPx, landing.snapDistPx,
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
            jitterTransportUV, velocityErrorVecPx, depthGate, coherenceRadiusPx, vp, r.divergencePx);
    }
    return r;
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

    // Firefly clamp: pull the AABB bounds into the mean +/- k*sigma band.
    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

    // Color shift expected from the remaining sub-texel jitter.
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

        // Inflate the covariance by the expected jitter-induced color shift.
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

// History feedback. The alignment drop is applied AFTER the feedbackMin/Max
// clamp and ONLY on planar (single-layer) surfaces ([FIX 7]).
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

float4 DebugViewEdgeState(float3 currentColorRGB, bool wasRevoked, bool isDilationZone, bool isForegroundEdge, bool velocityStraddled, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.25;
    if (wasRevoked)            debugColor = float3(1.0, 0.6, 0.0);  // orange: REVOKED dilation candidate
    else if (isDilationZone)   debugColor = float3(1.0, 0.05, 0.05); // red: kept dilation zone
    else if (isForegroundEdge) debugColor = float3(0.0, 0.85, 1.0);  // cyan: crest
    else if (velocityStraddled) debugColor = float3(1.0, 0.05, 1.0); // magenta: flat but velocity-straddled ([FIX 8a])
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewDisocclusionBreakdown(
    float3 currentColorRGB, bool depthRejected, bool velocityRejected, bool alertSuppressed, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.1;
    if (depthRejected)    debugColor.r = 1.0;
    if (velocityRejected) debugColor.g = 1.0;
    if (depthRejected && velocityRejected) debugColor = float3(1.0, 1.0, 0.0);
    else if (alertSuppressed)              debugColor.b = 0.45; // velocity error flagged but pursuit unconfirmed
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
    ViewportParams vp          = GetViewportParams(oneOverTargetSize);
    CameraBasis currentCamera  = GetCurrentFrameCameraBasis();
    CameraBasis previousCamera = GetPreviousFrameCameraBasis();
    float  coherenceRadiusPx   = max(taaVelRejection, kMinVelCoherenceRadiusPx);

    // ------------------------------------------------------------------
    // 1) Resolve the jittered render position of this stable output pixel.
    //    Forward map: ONLY the s_t jitter offset source. Inverse map: the
    //    sampling snap AND the reprojection base (static landings = identity).
    // ------------------------------------------------------------------
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float2 stableInFrameUV   = InverseReprojectThroughCamera(IN.uv0, currentCamera, taaTanHalfFovX, taaTanHalfFovY);
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    // Debug modes 3/4 fire before any classification exists: conservative
    // "revoked" (safe: the writer ignores the bit for non-candidates).
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
    // 3) Classify (OLD rules) and resolve the effective surface (merged
    //    rules: crest anchor / edge phase rule + the [PORT] extrapolation /
    //    flat [FIX 8a]).
    // ------------------------------------------------------------------
    SurfaceEdgeState edge = AnalyzeSurfaceEdges(neighborhood, useDepthDilation);

    LayerSurface currentLayer = ClassifyLayerSurface(
        neighborhood.depthRaw, neighborhood.velocityJitteredUV, pixel.fracPx,
        vp.sizePixels, coherenceRadiusPx, useDepthDilation, false, edge);

    ForegroundGeometry foreground = ComputeForegroundGeometry(
        neighborhood, currentLayer.effectiveDepth, edge);

    // Velocity spread across the jitter-aligned phase quad (raw values) -- a
    // noise estimate for the depth test.
    float2 qv00, qv10, qv01, qv11;
    SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, qv00, qv10, qv01, qv11);
    float2 quadVelocitySpreadUV = max(max(qv00, qv10), max(qv01, qv11))
                                - min(min(qv00, qv10), min(qv01, qv11));

    // ------------------------------------------------------------------
    // 4) Reproject into the history buffer (base = inverse map) with the
    //    EFFECTIVE velocity (the extrapolated field sample for a dilation
    //    candidate / a crest cliff-phase). This ONE landing feeds the gate,
    //    the tests and the history color -- the gate validates exactly what
    //    everything else consumes.
    // ------------------------------------------------------------------
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);

    // Debug mode 1 fires after classification but before validation:
    // conservative per-candidate encoding.
    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
    {
        // motionPx is already in PIXELS; DebugViewVelocity multiplies by
        // sizePixels internally, so convert to UV first.
        float4 v = DebugViewVelocity(repro.motionPx * vp.texelSize, vp.sizePixels, 1.0, centerDepthRaw);
        v.a = TransportAlpha(currentLayer.isDilationZone, centerDepthRaw);
        return v;
    }

    // ------------------------------------------------------------------
    // 5) Color neighborhood gather + FXAA corners + acutance energy.
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float3 fxaaCornersRGB[4]; // 0:NW(5), 1:NE(6), 2:SW(7), 3:SE(8)
    float  rawCrossLumaSum = 0.0; // cross taps 1..4, RCAS-luma of SRTM'd raw

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
        if (i <= 4)
            rawCrossLumaSum += SrtmLumaFSR(tapRGB);
        if (fxaaEnabled && i >= 5)
            fxaaCornersRGB[i - 5] = tapRGB;
    }

    float rawHighPass        = SrtmLumaFSR(currentColorRGB) - rawCrossLumaSum * 0.25;
    float rawSharpnessEnergy = rawHighPass * rawHighPass;

    // ------------------------------------------------------------------
    // 6) Validate the history sample position (filter support must fit).
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseSlepian3 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        // Offscreen history = no support: a tentative dilation here can never
        // be validated, so encode it as revoked for the writer.
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
    // 7) Exact jitter plumbing (all offsets in the shared s_tau sense).
    // ------------------------------------------------------------------
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterOffsetPrev2UV = RotationFlowUV(taaJitPrev2Yaw, taaJitPrev2Pitch, IN.uv0);
    float2 jitterCancelUV     = EstimateJitterCancelUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);
    float2 jitterTransportUV  = EstimateJitterTransportUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);

    // ------------------------------------------------------------------
    // 8) Own-history dilation validation (KEPT): the candidate validates
    //    through the history it would have if it stayed dilated -- the
    //    landing of its EFFECTIVE (extrapolated) velocity. Staying dilated
    //    means moving with the foreground field at this pixel, and the gate
    //    must validate the SAME landing the tests and the history color
    //    read: a split (gate at the anchor landing, tests at the
    //    extrapolated one) can keep a candidate whose landing misses the
    //    band -- kept by the gate, rejected by the velocity test, and only
    //    the gate's revocation persists -> flicker. Static scenes: q ~ 0,
    //    the extrapolated landing IS the anchor -> identical verdicts.
    //    Revocation downgrades to the texel's own raw background sample
    //    and re-projects with it (matching the writer's stored state).
    // ------------------------------------------------------------------
    bool dilationRevoked   = false;
    bool dilationCandidate = false;
    bool gateViaFlag       = false;

    if (currentLayer.isDilationZone)
    {
        dilationCandidate = true;

        float rayLenCur  = RayLengthFromUV(IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
        float rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
        float perspScale = rayLenPrev / max(rayLenCur, 1e-6);

        float rawExpClosest = currentLayer.closestDepth * perspScale;

        // Object-side slant allowance: minmod zeroes the gradients across a
        // silhouette cliff, so background-side candidates get NO slant
        // allowance and a strict crest anchor. The allowance's scale (the
        // object's slope over the crest offset) is the same magnitude and
        // shape as the extrapolation's landing displacement, so it covers it.
        float2 crestOffset = kOffsets3x3[currentLayer.closestIdx];
        float  objectSlant = (abs(currentLayer.gradX * crestOffset.x) + abs(currentLayer.gradY * crestOffset.y)) * perspScale;
        float  tolObject   = max(taaDepthRejection * rawExpClosest, 1e-5) + objectSlant;
        float  fgThreshold = rawExpClosest - tolObject;

        HistoryMotionGate gate = SampleHistoryMotionGate(repro.sampleUV, vp);

        // A) Own-history support gate (LAYER-MATCHED): a tap validates iff it
        //    is foreground-OWNED (flag >= 1) or its stored depth is at the
        //    object itself.
        gateViaFlag = (gate.supportMaxFlag >= 0.75);
        bool quadTouchesForeground = gateViaFlag || (gate.supportDepthMax >= fgThreshold);

        // C) The landing itself was inside the object's dilation band.
        bool touchesAlreadyDilated = (gate.centerFlag >= 1.5) && (gate.gateDepth >= fgThreshold);

        bool historyHasForeground = quadTouchesForeground || touchesAlreadyDilated;

        if (!historyHasForeground)
        {
            // Revoke: the revocation persists via the stored field.
            dilationRevoked = true;
            RevokeDilation(currentLayer, centerDepthRaw, centerVelocityJitteredUV);

            // Re-reproject with the true background velocity: the revoked
            // candidate lands back on its own texel and the tests compare
            // background against background.
            repro = ReprojectToHistory(
                IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);
        }
    }

    // ------------------------------------------------------------------
    // 9) History landing analysis: shared classifier + merged resolution +
    //    the [PORT] extrapolation on the stored field, gated by the stored
    //    ownership record ([FIX 9]). currentLayer.isForeground is the
    //    POST-revocation state: a revoked candidate is background.
    // ------------------------------------------------------------------
    HistoryLandingSurface landing = SampleHistoryLandingSurface(
        repro.sampleUV, vp, useDepthDilation,
        (taaDepthRejection > 0.001) || (taaVelRejection > 0.001),
        coherenceRadiusPx,
        currentLayer.isForeground);

    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
    {
        float4 v = DebugViewVelocity(landing.effectiveVelocityJitteredPrevUV, vp.sizePixels, 2.0, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    // Post-revocation single-layer state (a revoked candidate acts flat).
    bool currentSingleLayer = !(currentLayer.isDilationZone || currentLayer.isForegroundEdge);
    float surfaceLayerEps = taaDepthRejection * max(currentLayer.effectiveDepth, 1e-6) + 3.0 * edge.depthNoiseFloor;

    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        // Layer state + [FIX 8a] straddle visualization.
        bool velocityStraddled = false;
        if (!currentLayer.isDilationZone && !currentLayer.isForegroundEdge)
        {
            float2 v00, v10, v01, v11;
            SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, v00, v10, v01, v11);
            velocityStraddled = QuadStraddlesVelocityStep(
                v00, v10, v01, v11, neighborhood.velocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
        }
        float4 v = DebugViewEdgeState(currentColorRGB, dilationRevoked,
            currentLayer.isDilationZone, currentLayer.isForegroundEdge, velocityStraddled, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    if (taaDebugMode > 9.5 && taaDebugMode < 10.5)
    {
        float dropAmount = currentSingleLayer
            ? taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment)
            : 0.0;
        float4 v = DebugViewAlignmentDrop(currentColorRGB, dropAmount, currentSingleLayer, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    // ------------------------------------------------------------------
    // 10) Disocclusion tests: RESTORED kappa-corrected depth score (the
    //     layering oracle / depthGate) + the RESTORED velocity rejection.
    //     Binary union of both rejections.
    // ------------------------------------------------------------------
    float depthDisocclusionScore = ComputeDepthDisocclusionScore(
        currentLayer.effectiveDepth, landing.effectiveDepthRaw,
        neighborhood.depthRaw, neighborhood.velocityJitteredUV,
        currentLayer.effectiveVelocityUV,
        IN.uv0, repro.sampleUV, repro.prevCameraRayY,
        repro.jitterResidualPx, quadVelocitySpreadUV,
        foreground.slope, foreground.crestDrop, neighborhood.closestOffsetPx,
        currentLayer.isDilationZone, currentLayer.isForegroundEdge,
        edge.depthNoiseFloor, edge.depthQuantStep,
        edge.planeGradX, edge.planeGradY,
        surfaceLayerEps,
        currentLayer.extrapolationDispPx,
        vp);
    bool depthRejected = (depthDisocclusionScore >= 1.0);

    // Same-surface confidence for the velocity advection prediction ([FIX 5]).
    float depthGate = saturate((1.0 - depthDisocclusionScore) * 2.0);

    VelocityRejectionResult velocityRejection = EvaluateVelocityRejection(
        repro.sampleUV,
        currentLayer.effectiveVelocityUV,
        landing,
        currentSingleLayer,
        currentLayer.pairGradPx,        // [PORT] the active extrapolation pair's gradient
        currentLayer.shallowVelGradPx,  // [PORT] the foreground's shallow-end velocity gradient
        depthGate,
        jitterCancelUV, jitterTransportUV,
        neighborhood.velocityJitteredUV,
        coherenceRadiusPx,
        vp);

    bool disoccluded = depthRejected || velocityRejection.rejected;

    if (taaDebugMode > 1.5 && taaDebugMode < 2.5)
    {
        float4 v = DebugViewDisocclusionBreakdown(
            currentColorRGB, depthRejected, velocityRejection.rejected,
            (velocityRejection.errorRatio > 1.0 && !velocityRejection.rejected),
            centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    if (taaDebugMode > 6.5 && taaDebugMode < 7.5)
    {
        float4 v = DebugViewPursuit(currentColorRGB, velocityRejection.divergencePx, velocityRejection.rejected, centerDepthRaw);
        v.a = TransportAlpha(dilationRevoked, centerDepthRaw);
        return v;
    }

    if (taaDebugMode > 11.5 && taaDebugMode < 12.5)
    {
        // Jitter-cancel verification (see RotationFlowUV).
        float2 residualPx = (currentLayer.effectiveVelocityUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
        float rr = saturate(length(residualPx) * 0.5);
        return float4(rr, rr, 0.0, TransportAlpha(dilationRevoked, centerDepthRaw));
    }

    if (taaDebugMode > 8.5 && taaDebugMode < 9.5)
    {
        // Dilation-gate breakdown: R=revoked, G=kept candidate (full=flag
        // branch, half=depth branch), B=depth-rejected.
        float3 debugColor = currentColorRGB * 0.1;
        if (dilationRevoked)                                  debugColor.r = 1.0;
        if (dilationCandidate && !dilationRevoked)
            debugColor.g = gateViaFlag ? 1.0 : 0.5;
        if (depthRejected)                                    debugColor.b = 1.0;
        return float4(debugColor, TransportAlpha(dilationRevoked, centerDepthRaw));
    }

    // ------------------------------------------------------------------
    // 11) Color stats, history color resampling, clipping, drift.
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
    // 12) Feedback & temporal blending.
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
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 13) Final blend & output: alpha = acutance metric, SIGN-ENCODED with
    //     the revocation bit (negative = this texel's tentative dilation
    //     was revoked).
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    // Clamp scalar luminance only; do NOT clamp signed chrominance channels.
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    float outAlpha = TransportAlpha(dilationRevoked, rawSharpnessEnergy);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}