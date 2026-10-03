// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect
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
// LAYER SEMANTICS (the disocclusion contract):
//   * A dilation zone acts as part of the object it dilates to. A pixel that
//     WAS foreground and lands in that object's previous dilation zone is
//     NOT disoccluded (the landing resolves to the foreground on both sides).
//   * 2x2 Enclosing Quad Gating: Dilation is strictly disabled if the 2x2
//     bilinear reconstruction quad in history contains zero foreground.
//   * Dual-sided subpixel slant tolerance accounts for camera jitter phase
//     on continuous surfaces across both current and history frames.
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(sceneTex,        0);
uniform_sampler2D(depthTex,        1);
uniform_sampler2D(historyTex,      2);
uniform_sampler2D(velocityTex,     3);
uniform_sampler2D(prevVelocityTex, 4);

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
static const float kEpsilon                 = 1e-6;
static const float kLargeValue              = 32000.0;
static const float kSqrt2                   = 1.41421356;
static const float kLog2E                   = 1.44269504;
static const float kMinMotionDirLengthPx    = 0.1;
static const float kShadowLumaFloor         = 0.01;
static const float kShadowThresholdMin      = 0.001;
static const float kFlickerPadThreshold     = 0.001;
static const float kMinMotionBlendDropSpeed = 0.1;
static const float kMotionFullStrengthPx    = 8.0;

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

static const float2 kOffsets3x3[9] =
{
    float2( 0,  0), float2( 0, -1), float2( 0,  1),
    float2(-1,  0), float2( 1,  0), float2(-1, -1),
    float2( 1, -1), float2(-1,  1), float2( 1,  1)
};

static const float kInvOffsetLenSq[9] = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5 };
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
float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

float2 Bilerp2x2(float2 c00, float2 c10, float2 c01, float2 c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

float Minmod(float a, float b)
{
    return (a * b > 0.0) ? ((abs(a) < abs(b)) ? a : b) : 0.0;
}

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
// CAMERA BASIS & REPROJECTION
// ============================================================================
struct CameraBasis
{
    float3 rightTanFov;   // P: full-width right axis
    float3 forward;       // Q: top-left corner ray
    float3 downTanFov;    // R: full-height up axis
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

float2 ProjectRayToStableUV(float3 ray, float2 fallbackUV)
{
    if (ray.y <= kEpsilon) return fallbackUV;
    return float2((ray.x / (ray.y * max(taaTanHalfFovX, kEpsilon))) * 0.5 + 0.5,
                  0.5 - (ray.z / (ray.y * max(taaTanHalfFovY, kEpsilon))) * 0.5);
}

float3 BuildCameraRay(float2 uv, CameraBasis camera)
{
    return uv.x * camera.rightTanFov + camera.forward - uv.y * camera.downTanFov;
}

float2 ReprojectThroughCamera(float2 uv, CameraBasis camera, float2 fallbackUV)
{
    return ProjectRayToStableUV(BuildCameraRay(uv, camera), fallbackUV);
}

float RayLengthFromUV(float2 uv, float tanHalfFovX, float tanHalfFovY)
{
    float2 tanXY = float2((uv.x * 2.0 - 1.0) * tanHalfFovX, (1.0 - uv.y * 2.0) * tanHalfFovY);
    return sqrt(1.0 + dot(tanXY, tanXY));
}

// ============================================================================
// VIEWPORT & PIXEL GEOMETRY
// ============================================================================
struct ViewportParams
{
    float2 texelSize;
    float2 sizePixels;
    float2 minUV;
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

struct SnappedCoord
{
    float2 snappedUV;
    float2 fracPx;
    float  fracDist;
};

SnappedCoord SnapUVToTexel(float2 uv, ViewportParams vp)
{
    SnappedCoord sc;
    float2 pixelPos  = uv * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    sc.snappedUV     = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    sc.fracPx        = pixelPos - baseTexel;
    sc.fracDist      = length(sc.fracPx);
    return sc;
}

void Build3x3TapUVs(float2 centerUV, float2 texelSize, float2 minUV, float2 maxUV, out float2 tapUVs[9])
{
    float2 uvMinus = clamp(centerUV - texelSize, minUV, maxUV);
    float2 uvPlus  = clamp(centerUV + texelSize, minUV, maxUV);
    tapUVs[0] = centerUV;
    tapUVs[1] = float2(centerUV.x, uvMinus.y);
    tapUVs[2] = float2(centerUV.x, uvPlus.y);
    tapUVs[3] = float2(uvMinus.x, centerUV.y);
    tapUVs[4] = float2(uvPlus.x,  centerUV.y);
    tapUVs[5] = float2(uvMinus.x, uvMinus.y);
    tapUVs[6] = float2(uvPlus.x,  uvMinus.y);
    tapUVs[7] = float2(uvMinus.x, uvPlus.y);
    tapUVs[8] = float2(uvPlus.x,  uvPlus.y);
}

// ============================================================================
// GEOMETRIC LAYER CLASSIFICATION & SURFACE ANALYSIS
// ============================================================================
struct LayerSurface
{
    bool   isDilationZone;
    bool   isForegroundEdge;
    bool   isForeground;
    int    closestIdx;
    float  closestDepth;
    float  gradX;
    float  gradY;
    float  effectiveDepth;
    float2 effectiveVelocityUV;
    float  layerVelSpreadPx;
};

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

LayerSurface ClassifyLayerSurface(
    float  depthRaw[9],
    float2 velocityUV[9],
    float2 fracPx,
    float2 sizePixels,
    bool   useDepthDilation,
    float  depthRejectionThresh,
    bool   measureVelSpread)
{
    LayerSurface s;
    float centerDepth = depthRaw[0];

    // 1. Surface gradients via minmod limiter across cardinal neighbors
    ComputeSurfaceGradients(depthRaw, s.gradX, s.gradY);

    // 2. Identify closest (foreground) neighbor and detect discontinuities in one pass
    s.closestDepth = centerDepth;
    s.closestIdx   = 0;
    float depthEps = depthRejectionThresh * centerDepth;
    bool hasCloserNeighbor  = false;
    bool hasFartherNeighbor = false;
    float depthDiff[9];

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        if (depthRaw[i] > s.closestDepth)
        {
            s.closestDepth = depthRaw[i];
            s.closestIdx   = i;
        }

        float predDepth = centerDepth + s.gradX * kOffsets3x3[i].x + s.gradY * kOffsets3x3[i].y;
        float diff      = depthRaw[i] - predDepth;
        depthDiff[i]    = diff;
        hasCloserNeighbor  = hasCloserNeighbor  || (diff > depthEps);
        hasFartherNeighbor = hasFartherNeighbor || (diff < -depthEps);
    }

    s.isDilationZone   = useDepthDilation && hasCloserNeighbor;
    s.isForegroundEdge = !s.isDilationZone && hasFartherNeighbor;
    s.isForeground     = s.isDilationZone || s.isForegroundEdge;

    // 3. Resolve effective layer depth and velocity
    float subpixelDepth = centerDepth + s.gradX * fracPx.x + s.gradY * fracPx.y;
    s.effectiveDepth = s.isDilationZone ? s.closestDepth : subpixelDepth;

    s.effectiveVelocityUV = s.isDilationZone   ? velocityUV[s.closestIdx]
                          : (s.isForegroundEdge ? velocityUV[0]
                          : BilerpVelocityQuad(velocityUV, fracPx));

    // 4. Measure layer-coherent spatial velocity spread (bypassed if velocity testing is disabled)
    s.layerVelSpreadPx = 0.0;
    if (measureVelSpread)
    {
        float maxVelDiffSq = 0.0;
        [unroll]
        for (int k = 1; k < 9; ++k)
        {
            bool sameLayer = s.isDilationZone ? (depthRaw[k] > centerDepth) : (abs(depthDiff[k]) <= depthEps);
            if (sameLayer)
            {
                float2 diffPx = (velocityUV[k] - s.effectiveVelocityUV) * sizePixels;
                maxVelDiffSq = max(maxVelDiffSq, dot(diffPx, diffPx) * kInvOffsetLenSq[k]);
            }
        }
        s.layerVelSpreadPx = sqrt(maxVelDiffSq);
    }

    return s;
}

// ============================================================================
// HISTORY REPROJECTION
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
    float2 jitteredUV,
    float2 velocityJitteredUV,
    CameraBasis previousCamera,
    ViewportParams vp)
{
    HistoryReprojection h;

    float3 prevRay    = BuildCameraRay(jitteredUV + velocityJitteredUV, previousCamera);
    float2 fallbackUV = jitteredUV + velocityJitteredUV;
    h.sampleUV        = ProjectRayToStableUV(prevRay, fallbackUV);

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
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMin, fireflyMin, fireflyMax);
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

float4 DebugViewEdgeState(float3 currentColorRGB, bool isDilationZone, bool isForegroundEdge, bool isForeground, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.25;
    if (isDilationZone)        debugColor = float3(1.0, 0.05, 0.05); // Red: confirmed dilation zone
    else if (isForegroundEdge) debugColor = float3(0.0, 0.85, 1.0);  // Cyan: silhouette edge
    else if (isForeground)     debugColor = float3(0.1, 0.95, 0.1);  // Green: object body
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
    ViewportParams vp          = GetViewportParams();
    CameraBasis currentCamera  = GetCurrentFrameCameraBasis();
    CameraBasis previousCamera = GetPreviousFrameCameraBasis();

    // ------------------------------------------------------------------
    // 1) Current Pixel Geometry
    // ------------------------------------------------------------------
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0);
    SnappedCoord pixel       = SnapUVToTexel(currentJitteredUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
        return DebugViewLinearDepth(centerDepthRaw);
    if (taaDebugMode > 2.5 && taaDebugMode < 3.5)
        return DebugViewVelocity(centerVelocityJitteredUV, vp.sizePixels, 1.0, centerDepthRaw);

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
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, currentJitteredUV, currentLayer.effectiveVelocityUV, previousCamera, vp);

    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
        return DebugViewVelocity(repro.motionPx, vp.sizePixels, 1.0, centerDepthRaw);

    // ------------------------------------------------------------------
    // 5) Exact Jitter Plumbing
    // ------------------------------------------------------------------
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = IN.uv0 - ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0);
    float2 jitterCancelUV      = float2(0.0, 0.0);
    float2 jitterTransportUV   = float2(0.0, 0.0);

    bool needJitterCancel = velocityTestEnabled || (taaDebugMode > 1.5 && taaDebugMode < 2.5)
                                                || (taaDebugMode > 11.5 && taaDebugMode < 12.5);
    if (needJitterCancel)
    {
        float2 jitterOffsetPrev2UV = RotationFlowUV(taaJitPrev2Yaw, taaJitPrev2Pitch, IN.uv0);
        jitterCancelUV    = jitterOffsetCurUV - 2.0 * jitterOffsetPrevUV + jitterOffsetPrev2UV;
        jitterTransportUV = jitterOffsetPrev2UV - jitterOffsetPrevUV;
    }

    // ------------------------------------------------------------------
    // 6) Color Neighborhood Gather (Zero-Redundancy Shared FXAA Corners)
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float3 fxaaCornersRGB[4]; // 0:NW(5), 1:NE(6), 2:SW(7), 3:SE(8)

    [unroll]
    for (int c = 1; c < 9; ++c)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[c], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[c] = ToSpace(tapRGB);
        if (fxaaEnabled && c >= 5)
            fxaaCornersRGB[c - 5] = tapRGB;
    }

    // ------------------------------------------------------------------
    // 7) Validate History Sample Bounds
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseLanczos3 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        if (fxaaEnabled)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), centerDepthRaw);
        }
        return float4(currentColorRGB, centerDepthRaw);
    }

    // ------------------------------------------------------------------
    // 8) Geometrically Exact Disocclusion & Shared History Landing
    // ------------------------------------------------------------------
    SnappedCoord landing = SnapUVToTexel(repro.sampleUV, vp);

    float2 histTapUVs[9];
    Build3x3TapUVs(landing.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, histTapUVs);

    float  histDepths[9];
    float2 histVelocities[9];
    bool   needLandingVelocity = velocityTestEnabled || (taaDebugMode > 1.5 && taaDebugMode < 2.5)
                                                  || (taaDebugMode > 5.5 && taaDebugMode < 6.5)
                                                  || (taaDebugMode > 11.5 && taaDebugMode < 12.5);

    [unroll]
    for (int h = 0; h < 9; ++h)
    {
        histDepths[h] = tex2Dlod(historyTex, float4(histTapUVs[h], 0.0, 0.0)).a;
        float2 velUV  = clamp(histTapUVs[h] + jitterOffsetPrevUV, vp.minUV, vp.maxUV);
        histVelocities[h] = needLandingVelocity ? tex2Dlod(prevVelocityTex, float4(velUV, 0.0, 0.0)).rg : float2(0.0, 0.0);
    }

    // ------------------------------------------------------------------
    // History 2x2 Enclosing Quad + Paired Dilation Validation:
    // A continuous subpixel landing is bilinearly reconstructed strictly
    // from the 4 texels of the 2x2 quad enclosing it. If none of those 4 texels
    // touches foreground, nor does the paired relative dilated tap touch foreground,
    // the history sample contains 0.0% foreground history. Dilation is revoked,
    // preventing camera jitter from activating phantom disocclusions.
    // ------------------------------------------------------------------
    float rayLenCur  = RayLengthFromUV(IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
    float perspScale = rayLenPrev / max(rayLenCur, 1e-6);

    // Initial tentative classification at landing
    LayerSurface tentativeHistLayer = ClassifyLayerSurface(
        histDepths, histVelocities,
        landing.fracPx, vp.sizePixels,
        useDepthDilation, taaDepthRejection,
        velocityTestEnabled);

    if (currentLayer.isDilationZone)
    {
        // 1. Enclosing 2x2 bilinear quad depths around repro.sampleUV
        float d00 = histDepths[0];
        float d10 = (landing.fracPx.x >= 0.0) ? histDepths[4] : histDepths[3];
        float d01 = (landing.fracPx.y >= 0.0) ? histDepths[2] : histDepths[1];
        float d11 = (landing.fracPx.x >= 0.0)
            ? ((landing.fracPx.y >= 0.0) ? histDepths[8] : histDepths[6])
            : ((landing.fracPx.y >= 0.0) ? histDepths[7] : histDepths[5]);

        float maxHistDepth2x2 = max(max(d00, d10), max(d01, d11));

        float rawExpClosest = currentLayer.closestDepth * perspScale;
        float rawExpLimb    = centerDepthRaw * perspScale;
        float sagitta       = abs(rawExpClosest - rawExpLimb);
        float tolObject     = max(taaDepthRejection * rawExpClosest, 1e-5) + sagitta;
        float fgThreshold   = min(rawExpLimb, rawExpClosest) - tolObject;

        // Conservative test:
        // A) Does any of the 4 enclosing 2x2 bilinear texels touch foreground?
        bool quadTouchesForeground = (maxHistDepth2x2 >= fgThreshold);
        // B) Does the paired relative dilated tap in history touch foreground?
        bool pairedTapTouchesForeground = (histDepths[currentLayer.closestIdx] >= fgThreshold);
        // C) Did the landing touch an already-dilated zone of this foreground object?
        bool touchesAlreadyDilated = tentativeHistLayer.isDilationZone && (tentativeHistLayer.closestDepth >= fgThreshold);

        bool historyHasForeground = quadTouchesForeground || pairedTapTouchesForeground || touchesAlreadyDilated;

        if (!historyHasForeground)
        {
            // History bilinear quad has zero foreground: revoke dilation
            currentLayer.isDilationZone      = false;
            currentLayer.isForeground        = false;
            currentLayer.effectiveDepth      = centerDepthRaw + currentLayer.gradX * pixel.fracPx.x + currentLayer.gradY * pixel.fracPx.y;
            currentLayer.effectiveVelocityUV = BilerpVelocityQuad(curVelocities, pixel.fracPx);

            // Re-reproject using true continuous background velocity
            repro   = ReprojectToHistory(IN.uv0, currentJitteredUV, currentLayer.effectiveVelocityUV, previousCamera, vp);
            landing = SnapUVToTexel(repro.sampleUV, vp);
            Build3x3TapUVs(landing.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, histTapUVs);

            [unroll]
            for (int h2 = 0; h2 < 9; ++h2)
            {
                histDepths[h2] = tex2Dlod(historyTex, float4(histTapUVs[h2], 0.0, 0.0)).a;
                float2 velUV2  = clamp(histTapUVs[h2] + jitterOffsetPrevUV, vp.minUV, vp.maxUV);
                histVelocities[h2] = needLandingVelocity ? tex2Dlod(prevVelocityTex, float4(velUV2, 0.0, 0.0)).rg : float2(0.0, 0.0);
            }

            rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
            perspScale = rayLenPrev / max(rayLenCur, 1e-6);

            // Re-classify history layer at the true background landing
            tentativeHistLayer = ClassifyLayerSurface(
                histDepths, histVelocities,
                landing.fracPx, vp.sizePixels,
                useDepthDilation, taaDepthRejection,
                velocityTestEnabled);
        }
    }

    LayerSurface historyLayer = tentativeHistLayer;

    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
        return DebugViewEdgeState(currentColorRGB, currentLayer.isDilationZone, currentLayer.isForegroundEdge, currentLayer.isForeground, centerDepthRaw);

    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
        return DebugViewVelocity(historyLayer.effectiveVelocityUV, vp.sizePixels, 2.0, centerDepthRaw);

    bool  depthRejected      = false;
    bool  velocityRejected   = false;
    float velocityErrorRatio = 0.0;

    if (depthTestEnabled)
    {
        float rawExp = currentLayer.effectiveDepth * perspScale;

        // Exact dual-sided subpixel slant tolerance:
        // Accounts for subpixel jitter phase on slanted/curved surfaces in both current and history frames
        float tolCurSlant  = (abs(currentLayer.gradX * pixel.fracPx.x) + abs(currentLayer.gradY * pixel.fracPx.y)) * perspScale;
        float tolHistSlant = abs(historyLayer.gradX * landing.fracPx.x) + abs(historyLayer.gradY * landing.fracPx.y);
        float tolSlant     = tolCurSlant + tolHistSlant;
        float tolDepth     = max(taaDepthRejection * rawExp, 1e-5) + tolSlant;

        if (currentLayer.isForeground)
        {
            // CASE 1: Current pixel is FOREGROUND (Silhouette Edge or Confirmed Dilation Zone)
            // Symmetrized Physical Depth Envelope:
            // When subpixel jitter shifts a steep surface between a direct limb hit and a dilated
            // crest hit in either direction, taking the bounds across both frames ensures consistency.
            float curLimb  = min(currentLayer.effectiveDepth, currentLayer.closestDepth) * perspScale;
            float curCrest = max(currentLayer.effectiveDepth, currentLayer.closestDepth) * perspScale;

            // An occluder in front of the object was present at t-1:
            bool occludedInFront = (historyLayer.effectiveDepth > curCrest + tolDepth);

            // The foreground object was not present at t-1 (history closest depth is strictly behind limb):
            bool missingBehind = (historyLayer.closestDepth < curLimb - tolDepth);

            if (occludedInFront || missingBehind)
            {
                depthRejected = true;
            }
            else
            {
                float histLimb  = min(historyLayer.effectiveDepth, historyLayer.closestDepth);
                float histCrest = max(historyLayer.effectiveDepth, historyLayer.closestDepth);

                float fgMin = min(curLimb, histLimb) - tolDepth;
                float fgMax = max(curCrest, histCrest) + tolDepth;

                bool histMatches = (historyLayer.effectiveDepth >= fgMin && historyLayer.effectiveDepth <= fgMax) ||
                                   (historyLayer.closestDepth   >= fgMin && historyLayer.closestDepth   <= fgMax);

                depthRejected = !histMatches;
            }
        }
        else
        {
            // CASE 2: Current pixel is BACKGROUND / CONTINUOUS SURFACE
            // Contract: "a dilation zone gets replaced with background on the next frame this pixel is disoccluded."
            // Only rejected if the current pixel is strictly farther than the historical occluder/dilation zone.
            bool occludedByDilationZone = historyLayer.isDilationZone && (historyLayer.closestDepth > rawExp + tolDepth);
            bool occludedByGeometry     = (historyLayer.effectiveDepth - rawExp) > tolDepth;
            depthRejected = occludedByDilationZone || occludedByGeometry;
        }
    }

    // Velocity rejection is restricted strictly to continuous background surfaces.
    // Silhouette edges experience motion boundary cliffs and angular parallax acceleration;
    // their presence is validated by the geometric depth envelope, not by velocity continuity.
    if (!depthRejected && velocityTestEnabled && !currentLayer.isForeground && !historyLayer.isForeground)
    {
        float2 errVecPx = (currentLayer.effectiveVelocityUV - historyLayer.effectiveVelocityUV - jitterCancelUV) * vp.sizePixels;
        float  errPx    = length(errVecPx);

        velocityErrorRatio = errPx / max(taaVelRejection, 1e-4);

        float maxGrad = max(currentLayer.layerVelSpreadPx, historyLayer.layerVelSpreadPx);
        float velTol  = taaVelRejection + maxGrad * landing.fracDist;

        if (errPx > velTol)
        {
            if (taaCrossTestStrength > 0.001)
            {
                // Exact round-trip pursuit confirmation
                float2 pursuitUV = repro.sampleUV - (historyLayer.effectiveVelocityUV + jitterTransportUV);
                if (all(pursuitUV >= vp.minUV) && all(pursuitUV <= vp.maxUV))
                {
                    float roundTripDivergencePx = length(pursuitUV - currentJitteredUV) * vp.sizePixels;
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

    if (taaDebugMode > 1.5 && taaDebugMode < 2.5)
        return DebugViewDisocclusionBreakdown(
            currentColorRGB, depthRejected, velocityRejected,
            (velocityErrorRatio > 1.0 && !velocityRejected),
            centerDepthRaw);

    if (taaDebugMode > 11.5 && taaDebugMode < 12.5)
    {
        float2 residualPx = (currentLayer.effectiveVelocityUV - historyLayer.effectiveVelocityUV - jitterCancelUV) * vp.sizePixels;
        return float4(saturate(length(residualPx) * 0.5), saturate(length(residualPx) * 0.5), 0.0, centerDepthRaw);
    }

    // ------------------------------------------------------------------
    // 9) History Color Reconstruction (Slepian Suite)
    // ------------------------------------------------------------------
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized, pixel.fracPx);
    float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

    float3 historyColorSpace =
        (taaUseLanczos3 > 0.5)
        ? SampleHistoryColor_Slepian3_Fused21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV)
        : SampleHistoryColor_Slepian2_Fast9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    historyColorSpace = CompressGamut(historyColorSpace);

    if (taaDebugMode > 4.5 && taaDebugMode < 5.5)
        return DebugViewHistoryColor(historyColorSpace, centerDepthRaw);

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
    // 12) Final Blend Output
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
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