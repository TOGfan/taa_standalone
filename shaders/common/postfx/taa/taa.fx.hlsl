#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(sceneTex,     0);
uniform_sampler2D(depthTex,     1);
uniform_sampler2D(historyTex,   2);
uniform_sampler2D(velocityTex,  3);

// ============================================================================
// CBUFFER
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
    float  taaClipOvershoot;
    float  taaPadding0;                     // Aligns oneOverTargetSize to an 8-byte boundary for std140/Vulkan/Metal

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
static const float kEpsilon                 = 1e-5;
static const float kLargeValue              = 32000.0;
static const float kSqrt2                   = 1.41421356;
static const float kLog2E                   = 1.44269504;
static const float taaMinMotionDir          = 0.1;
static const float kShadowLumaFloor         = 0.01;
static const float kShadowThresholdMin      = 0.001;
static const float kFlickerPadThreshold     = 0.001;
static const float kMinMotionBlendDropSpeed = 0.1;

static const float kMotionFullStrengthPx    = 8.0;

static const float kMinSigma                = 0.001;
static const float kMinSpatialContrast      = 0.001;
static const float kMinFootprintRange       = 1e-4;
static const float kFireflyClampEpsilon     = 0.001;

// Tightened physical residuals
static const float kForegroundSlopeRadius   = 1.5;   // px: true 3x3 footprint reach (was 3.0)
static const float kReprojResidualPerTan    = 0.5;  // rotation slack per unit motion
static const float kReprojResidualMax       = 0.05; // cap of reprojection residual (was 0.05)
static const float kDollyResidualMax        = 0.1;  // cap of translation depth-scale residual

// Fallback FXAA
static const float kFXAAReduceMul           = 1.0 / 128.0;
static const float kFXAAReduceMin           = 1.0 / 128.0;
static const float kFXAAMaxDir              = 8.0;
static const float kFXAABlendKnee           = 0.8;
static const float kFXAABlendSharpness      = 2.0;

// Luma drift correction
static const float kLumaDriftLumaFloor      = 0.05;
static const float kLumaDriftRelThreshold   = 0.30;
static const float kLumaDriftAbsThreshold   = 0.03;
static const float kLumaDriftChromaFadeWidth = 2.0;

// Debug views
static const float kDebugVelocityScale      = 0.1;
static const float kDebugLinearDepthRange   = 100.0;

static const float2 kOffsets3x3[9] =
{
    float2( 0,  0), float2( 0, -1), float2( 0,  1),
    float2(-1,  0), float2( 1,  0), float2(-1, -1),
    float2( 1, -1), float2(-1,  1), float2( 1,  1)
};

static const float kStdWeights[9] = { 1.0, 0.36787944, 0.36787944, 0.36787944, 0.36787944, 0.13533528, 0.13533528, 0.13533528, 0.13533528 };
static const float kInvLength[9]  = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678 };

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
// COLOR SPACES & REVERSIBLE TONEMAPPING
// ============================================================================
float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

float3 Tonemap(float3 c)
{
    float luma = LumaRGB(c);
    return c / (1.0 + luma);
}

float3 Untonemap(float3 c)
{
    float luma = LumaRGB(c);
    return c / max(1.0 - min(luma, 0.999), 1e-4);
}

static const float3x3 kRGB_TO_LMS = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800,  2.60975740, -0.34131939, -0.00419608, -0.70341861,  1.70761470);

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
    float3 cTonemapped = Tonemap(max(0.0, rgb));
    return (taaColorSpaceOklab > 0.5) ? RGBToOklab(cTonemapped) : RGBToYCoCg(cTonemapped); 
}

float3 FromSpace(float3 c) 
{ 
    float3 cRGB = (taaColorSpaceOklab > 0.5) ? OklabToRGB(c) : YCoCgToRGB(c); 
    return Untonemap(max(0.0, cRGB)); 
}

// ============================================================================
// MATRIX UTILITIES
// ============================================================================
float3x3 InverseSymmetric3x3(float3x3 m, out bool success)
{
    float c00 = m[1][1] * m[2][2] - m[1][2] * m[1][2];
    float c01 = m[0][2] * m[1][2] - m[0][1] * m[2][2];
    float c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    
    float det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    success = (abs(det) > kEpsilon);
    if (!success) return float3x3(0,0,0, 0,0,0, 0,0,0);

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
// REPROJECTION & DEPTH
// ============================================================================
float2 ReprojFinish(float3 r2, float2 fallbackUV)
{
    if (r2.y <= kEpsilon) return fallbackUV;
    return float2((r2.x / (r2.y * max(taaTanHalfFovX, kEpsilon))) * 0.5 + 0.5,
                  0.5 - (r2.z / (r2.y * max(taaTanHalfFovY, kEpsilon))) * 0.5);
}

float2 ReprojectUV(float2 uv, float3 P, float3 Q, float3 R)
{
    return ReprojFinish(uv.x * P + Q - uv.y * R, uv);
}

float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

// ============================================================================
// FALLBACK FXAA
// ============================================================================
float3 ApplyFXAACached(float2 uv, float2 texel, float3 fxaaRGB[5])
{
    float lumaNW = LumaRGB(fxaaRGB[1]);
    float lumaNE = LumaRGB(fxaaRGB[2]);
    float lumaSW = LumaRGB(fxaaRGB[3]);
    float lumaSE = LumaRGB(fxaaRGB[4]);
    float lumaM  = LumaRGB(fxaaRGB[0]);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * kFXAAReduceMul), kFXAAReduceMin);
    float rcpDirMin = 1.0 / (min(abs(lumaMax - lumaMin), max(lumaMax, 1.0)) + dirReduce);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    dir = clamp(dir * rcpDirMin, float2(-kFXAAMaxDir, -kFXAAMaxDir), float2(kFXAAMaxDir, kFXAAMaxDir)) * texel;

    float3 rgbA = 0.5 * (
        tex2Dlod(sceneTex, float4(uv + dir * (1.0/3.0 - 0.5), 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv + dir * (2.0/3.0 - 0.5), 0.0, 0.0)).rgb);
        
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        tex2Dlod(sceneTex, float4(uv + dir * (0.0/3.0 - 0.5), 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv + dir * (3.0/3.0 - 0.5), 0.0, 0.0)).rgb);

    float lumaB = LumaRGB(rgbB);
    return ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
}

// ============================================================================
// HISTORY SAMPLING
// ============================================================================
float Lanczos3Weight(float fc, int tap)
{
    float d = abs(fc + float(2 - tap));
    float piD = 3.14159265 * max(d, 1e-5);
    return 3.0 * sin(piD) * sin(piD * (1.0 / 3.0)) / (piD * piD);
}

float3 SampleHistoryLanczos3(float2 uv, float2 texSize, float2 invTexSize, float2 minUV, float2 maxUV)
{
    float2 samplePos = uv * texSize;
    float2 tc = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc;

    float2 sumXY = float2(0.0, 0.0);
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        sumXY.x += Lanczos3Weight(f.x, i);
        sumXY.y += Lanczos3Weight(f.y, i);
    }
    float2 invSum = 1.0 / max(sumXY, 1e-5);

    float3 color   = float3(0.0, 0.0, 0.0);
    float3 footMin = float3(kLargeValue, kLargeValue, kLargeValue);
    float3 footMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    for (int y = 0; y < 6; ++y)
    {
        float wY = Lanczos3Weight(f.y, y) * invSum.y;
        float vy = clamp((tc.y + float(y - 2)) * invTexSize.y, minUV.y, maxUV.y);
        for (int x = 0; x < 6; ++x)
        {
            float wX = Lanczos3Weight(f.x, x) * invSum.x;
            float vx = clamp((tc.x + float(x - 2)) * invTexSize.x, minUV.x, maxUV.x);

            float3 tapColor = max(tex2Dlod(historyTex, float4(vx, vy, 0.0, 0.0)).rgb, 0.0);
            color += tapColor * (wX * wY);

            footMin = min(footMin, tapColor);
            footMax = max(footMax, tapColor);
        }
    }

    float3 footRng = max(footMax - footMin, kMinFootprintRange);
    float3 footMargin = taaHistoryOvershoot * footRng;
    color = clamp(color, footMin - footMargin, footMax + footMargin);

    return ToSpace(max(color, 0.0));
}

float3 SampleHistoryCatmullRom5Tap(float2 uv, float2 texSize, float2 invTexSize, float2 minUV, float2 maxUV)
{
    float2 samplePos = uv * texSize;
    float2 tc = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc;
    float2 f2 = f * f;

    float2 w0 = f * (f * (-0.5 * f + 1.0) - 0.5);
    float2 w1 = 1.0 + f2 * (1.5 * f - 2.5);
    float2 w2 = f * (f * (-1.5 * f + 2.0) + 0.5);
    float2 w3 = f2 * (0.5 * f - 0.5);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w12 + 1e-5);

    float2 tc0  = clamp((tc - 1.0) * invTexSize, minUV, maxUV);
    float2 tc3  = clamp((tc + 2.0) * invTexSize, minUV, maxUV);
    float2 tc12 = clamp((tc + offset12) * invTexSize, minUV, maxUV);

    float3 tap0 = max(tex2Dlod(historyTex, float4(tc12.x, tc0.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap1 = max(tex2Dlod(historyTex, float4(tc0.x, tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap2 = max(tex2Dlod(historyTex, float4(tc12.x, tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap3 = max(tex2Dlod(historyTex, float4(tc3.x, tc12.y, 0.0, 0.0)).rgb, 0.0);
    float3 tap4 = max(tex2Dlod(historyTex, float4(tc12.x, tc3.y, 0.0, 0.0)).rgb, 0.0);

    float w0y   = w12.x * w0.y;
    float w0x   = w0.x * w12.y;
    float w12xy = w12.x * w12.y;
    float w3x   = w3.x * w12.y;
    float w3y   = w12.x * w3.y;

    float3 color = tap0 * w0y + tap1 * w0x + tap2 * w12xy + tap3 * w3x + tap4 * w3y;
    float wsum = w0y + w0x + w12xy + w3x + w3y;
    color /= max(wsum, 1e-4);

    float3 footMin = float3(kLargeValue, kLargeValue, kLargeValue);
    float3 footMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);
    footMin = min(footMin, min(tap0, min(tap1, min(tap2, min(tap3, tap4)))));
    footMax = max(footMax, max(tap0, max(tap1, max(tap2, max(tap3, tap4)))));

    float3 footRng = max(footMax - footMin, kMinFootprintRange);
    float3 footMargin = taaHistoryOvershoot * footRng;
    color = clamp(color, footMin - footMargin, footMax + footMargin);

    return ToSpace(max(color, 0.0));
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct DepthVelocityStats { 
    float  closestRawDepth;
    float  secondClosestRaw;
    float2 closestOffset;
    float2 secondOffset;
    float2 bestVel; 
};

struct NeighborhoodStats { 
    float3 aabbMin;
    float3 aabbMax; 
    float3 mu; 
    float3 sigma; 
    float3x3 invCov; 
    bool validCovariance; 
    float spatialContrast; 
    float3 expectedColorShift; 
    float weights[9]; 
};

struct HistoryData {
    bool valid;
    float disocclusion;
    float shadowRisk;
    float clipDistanceRejection;
    float3 colorSpace;
};

// ============================================================================
// STATE-AWARE HISTORY DEPTH SAMPLING
// ============================================================================
void SampleHistoryDepth(float2 historyUV, float2 passTexel, float2 passTexSize, float2 minUV, float2 maxUV,
                        out float bilinearDepth, out float closestDepth)
{
    float2 samplePos = historyUV * passTexSize - 0.5;
    float2 tc00 = floor(samplePos);
    float2 f = saturate(samplePos - tc00);

    float2 uvs[4];
    uvs[0] = clamp((tc00 + float2(0.5, 0.5)) * passTexel, minUV, maxUV);
    uvs[1] = clamp((tc00 + float2(1.5, 0.5)) * passTexel, minUV, maxUV);
    uvs[2] = clamp((tc00 + float2(0.5, 1.5)) * passTexel, minUV, maxUV);
    uvs[3] = clamp((tc00 + float2(1.5, 1.5)) * passTexel, minUV, maxUV);

    float d0 = tex2Dlod(historyTex, float4(uvs[0], 0.0, 0.0)).a;
    float d1 = tex2Dlod(historyTex, float4(uvs[1], 0.0, 0.0)).a;
    float d2 = tex2Dlod(historyTex, float4(uvs[2], 0.0, 0.0)).a;
    float d3 = tex2Dlod(historyTex, float4(uvs[3], 0.0, 0.0)).a;

    closestDepth = max(max(d0, d1), max(d2, d3));

    float2 w1 = f;
    float2 w0 = 1.0 - f;
    bilinearDepth = d0 * (w0.x * w0.y)
                  + d1 * (w1.x * w0.y)
                  + d2 * (w0.x * w1.y)
                  + d3 * (w1.x * w1.y);
}

// ============================================================================
// NEIGHBORHOOD STATISTICS
// ============================================================================
NeighborhoodStats ComputeNeighborhoodStats(float3 cachedSpace[9], float2 velocityDir, float normalizedMotion, float2 localJitterPx)
{
    NeighborhoodStats stats; stats.validCovariance = false;
    stats.aabbMin  = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax  = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    bool wantCovariance = (taaUseCovarianceClipping > 0.5) && (taaUseKDopClipping <= 0.5);
    bool needExpectedShift = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 m1 = float3(0.0, 0.0, 0.0), m2 = float3(0.0, 0.0, 0.0);
    float3 mCross = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    float motionFactor = saturate(normalizedMotion);

    bool useShiftedCenter = (taaJitterAwareVariance > 0.5);
    float2 centerShift = useShiftedCenter ? localJitterPx : float2(0.0, 0.0);

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 pixelOffset = kOffsets3x3[i]; 
        float3 cSpace = cachedSpace[i];
        
        stats.aabbMin = min(stats.aabbMin, cSpace); 
        stats.aabbMax = max(stats.aabbMax, cSpace);

        float2 shiftOffset = pixelOffset - centerShift;
        float w = useShiftedCenter ? exp2(-dot(shiftOffset, shiftOffset) * kLog2E) : kStdWeights[i];

        if (taaLumaVariance > 0.5) { 
            w *= (1.0 / (1.0 + max(cSpace.x, 0.0))); 
        }
        if (taaVelocityAlignedVariance > 0.5 && i > 0) { 
            w *= lerp(1.0, saturate(dot(pixelOffset, velocityDir) * kInvLength[i] * 0.5 + 0.5), motionFactor); 
        }

        stats.weights[i] = w;
        m1 += cSpace * w; 
        m2 += cSpace * cSpace * w;
        if (wantCovariance) { mCross += float3(cSpace.x * cSpace.y, cSpace.x * cSpace.z, cSpace.y * cSpace.z) * w; }
        weightSum += w;
    }

    float invWeightSum = 1.0 / max(weightSum, kEpsilon);
    stats.mu = m1 * invWeightSum;
    stats.sigma = sqrt(max(m2 * invWeightSum - stats.mu * stats.mu, 0.0));

    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mu - taaFireflyClamp * stats.sigma; 
        float3 fireflyMax = stats.mu + taaFireflyClamp * stats.sigma;
        stats.aabbMin  = clamp(stats.aabbMin, fireflyMin, fireflyMax); 
        stats.aabbMax  = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

    stats.expectedColorShift = float3(0.0, 0.0, 0.0);
    if (needExpectedShift) {
        float3 gradX = localJitterPx.x > 0.0 ? (cachedSpace[4] - cachedSpace[0]) : (cachedSpace[3] - cachedSpace[0]);
        float3 gradY = localJitterPx.y > 0.0 ? (cachedSpace[2] - cachedSpace[0]) : (cachedSpace[1] - cachedSpace[0]);
        stats.expectedColorShift = (gradX * abs(localJitterPx.x)) + (gradY * abs(localJitterPx.y));
    }

    if (wantCovariance)
    {
        float3x3 cov = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        cov[0][0] = m2.x * invWeightSum - stats.mu.x * stats.mu.x;
        cov[1][1] = m2.y * invWeightSum - stats.mu.y * stats.mu.y;
        cov[2][2] = m2.z * invWeightSum - stats.mu.z * stats.mu.z;
        cov[0][1] = mCross.x * invWeightSum - stats.mu.x * stats.mu.y;
        cov[0][2] = mCross.y * invWeightSum - stats.mu.x * stats.mu.z;
        cov[1][2] = mCross.z * invWeightSum - stats.mu.y * stats.mu.z;

        cov[0][0] += kEpsilon; cov[1][1] += kEpsilon; cov[2][2] += kEpsilon;

        if (taaJitterFlickerPadding > kFlickerPadThreshold)
        {
            float paddingFade = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - normalizedMotion) : 1.0;
            float padMult = taaJitterFlickerPadding * paddingFade;
            
            if (taaDirectionalVariance > 0.5)
            {
                float3 dirPad = stats.expectedColorShift * padMult;
                cov[0][0] += dirPad.x * dirPad.x;
                cov[1][1] += dirPad.y * dirPad.y;
                cov[2][2] += dirPad.z * dirPad.z;
                cov[0][1] += dirPad.x * dirPad.y;
                cov[0][2] += dirPad.x * dirPad.z;
                cov[1][2] += dirPad.y * dirPad.z;
            }
            else
            {
                float padAmt = stats.spatialContrast * length(localJitterPx) * padMult;
                cov[0][0] += padAmt * padAmt;
                cov[1][1] += padAmt * padAmt;
                cov[2][2] += padAmt * padAmt;
            }
        }
        
        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        stats.invCov = InverseSymmetric3x3(cov, stats.validCovariance);
    }

    if (!stats.validCovariance && taaJitterFlickerPadding > kFlickerPadThreshold)
    {
        float paddingFade = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - normalizedMotion) : 1.0;
        float padMult = taaJitterFlickerPadding * paddingFade;
        
        if (taaDirectionalVariance > 0.5) { stats.sigma += abs(stats.expectedColorShift * padMult); }
        else { stats.sigma += (stats.spatialContrast * length(localJitterPx) * padMult); }
    }

    stats.sigma = max(stats.sigma, kMinSigma);
    return stats;
}

// ============================================================================
// EXACT RAY-AABB INTERSECTION CLIPPING
// ============================================================================
float3 IntersectRayAABB(float3 history, float3 target, float3 boxMin, float3 boxMax, float softClip, float motionFactor)
{
    float3 pClip = 0.5 * (boxMax + boxMin);
    float3 eClip = max(0.5 * (boxMax - boxMin), kEpsilon);

    float3 vClip = history - pClip;
    float3 vUnit = vClip / eClip;
    float3 aUnit = abs(vUnit);
    float maUnit = max(aUnit.x, max(aUnit.y, aUnit.z));

    if (maUnit > 1.0)
    {
        float3 rayDir = target - history;
        float3 s = step(0.0, rayDir) * 2.0 - 1.0;
        float3 invDir = 1.0 / (s * max(abs(rayDir), 1e-7));
        
        float3 t0 = (boxMin - history) * invDir;
        float3 t1 = (boxMax - history) * invDir;
        
        float3 tMin = min(t0, t1);
        float tHit = saturate(max(max(tMin.x, tMin.y), tMin.z));

        float3 clipped = history + rayDir * tHit;
        
        if (softClip > 0.0)
        {
            float softLimit = 1.0 + softClip * (1.0 - exp2(-(maUnit - 1.0) * kLog2E));
            float blendSoft = lerp(softLimit, 1.0, motionFactor);
            clipped = lerp(history, clipped, 1.0 / max(blendSoft, 1.0));
        }
        return clipped;
    }
    return history;
}

// ============================================================================
// HISTORY CLIPPING
// ============================================================================
float3 ClipHistory(float3 historySpace, NeighborhoodStats stats, float normalizedMotion, float dynamicGamma, float3 cachedSpace[9], float3 clipMargin)
{
    float motionFactor = saturate(normalizedMotion);

    if (taaUseKDopClipping > 0.5)
    {
        float3 rayCenter = stats.mu;
        float3 dir = historySpace - rayCenter;
        
        float nearHit = -kLargeValue; 
        float farHit  =  kLargeValue;

        [unroll]
        for (int a = 0; a < 16; ++a)
        {
            float3 axis = kDopAxes[a];
            float proj_pos = dot(rayCenter, axis);
            float2 extents;

            if (taaKDopVariance > 0.5)
            {
                float2 moments = float2(0.0, 0.0);
                float wSum = 0.0;
                float pMin = kLargeValue;
                float pMax = -kLargeValue;

                [unroll]
                for (int n = 0; n < 9; ++n)
                {
                    float t = dot(cachedSpace[n], axis);
                    float w = stats.weights[n];
                    moments += float2(t, t * t) * w;
                    wSum += w;
                    pMin = min(pMin, t);
                    pMax = max(pMax, t);
                }
                moments /= max(wSum, kEpsilon);
                
                float mu = moments.x;
                float sigma = sqrt(max(moments.y - mu * mu, 0.0));
                float expandedSigma = sigma * dynamicGamma;

                extents.x = mu - expandedSigma;
                extents.y = mu + expandedSigma;

                float axisMargin = taaClipOvershoot * max(pMax - pMin, kMinFootprintRange);
                extents += float2(-axisMargin, axisMargin);
            }
            else
            {
                extents = float2(kLargeValue, -kLargeValue);
                [unroll]
                for (int n = 0; n < 9; ++n)
                {
                    float t = dot(cachedSpace[n], axis);
                    extents.x = min(t, extents.x);
                    extents.y = max(t, extents.y);
                }
                
                float axisMargin = taaClipOvershoot * max(extents.y - extents.x, kMinFootprintRange) + kEpsilon;
                extents += float2(-axisMargin, axisMargin);
            }

            float dir_dot = dot(dir, axis);
            float s_dot   = (dir_dot >= 0.0 ? 1.0 : -1.0);
            float inv_dir = 1.0 / (s_dot * max(abs(dir_dot), 1e-7));

            float t0 = (extents.x - proj_pos) * inv_dir;
            float t1 = (extents.y - proj_pos) * inv_dir;

            nearHit = max(nearHit, min(t0, t1));
            farHit  = min(farHit,  max(t0, t1));
        }

        if (nearHit <= farHit && (nearHit > 0.0 || farHit > 0.0))
        {
            float t_hit = clamp(nearHit > 0.0 ? nearHit : farHit, 0.0, 1.0);

            if (t_hit < 1.0)
            {
                if (taaSoftClip > 0.0)
                {
                    float maxUnit = 1.0 / max(t_hit, kEpsilon);
                    float softLimit = 1.0 + taaSoftClip * (1.0 - exp2(-(maxUnit - 1.0) * kLog2E));
                    return rayCenter + dir * (lerp(softLimit, 1.0, motionFactor) / maxUnit);
                }
                return rayCenter + t_hit * dir;
            }
            return historySpace;
        }

        return historySpace;
    }

    if (stats.validCovariance)
    {
        float3 diff = historySpace - stats.mu; 
        float d2 = dot(diff, mul(stats.invCov, diff)); 
        float gamma2 = dynamicGamma * dynamicGamma;
        float3 clipped = (d2 > gamma2 && d2 > kEpsilon) ? (stats.mu + diff * (dynamicGamma / sqrt(max(d2, kEpsilon)))) : historySpace;
        return IntersectRayAABB(clipped, stats.mu, stats.aabbMin - clipMargin, stats.aabbMax + clipMargin, taaSoftClip, motionFactor);
    }

    float3 chromaWeights = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 extents = stats.sigma * dynamicGamma * chromaWeights;
    float3 bMin = max(stats.mu - extents, stats.aabbMin - clipMargin);
    float3 bMax = min(stats.mu + extents, stats.aabbMax + clipMargin);

    return IntersectRayAABB(historySpace, stats.mu, bMin, bMax, taaSoftClip, motionFactor);
}

// ============================================================================
// MOTION-COMPENSATED DEPTH DISOCCLUSION (Optimized for Tilted Planes)
// ============================================================================
float ComputeDisocclusion(
    float  resolvedDepth,
    float  histRawDepth,
    float  cachedDepth[9],
    float2 cachedVel[9],
    float2 pixelVel,
    float  rPrevY,
    float2 exactJitterDelta,
    float  foregroundSlope,
    float  crestDrop,
    bool   isDilationZone,
    bool   isForegroundEdge)
{
    if (taaDepthRejection <= 0.001) return 0.0;

    float safeRPrevY = (rPrevY > 1e-4) ? rPrevY : 1.0;
    float expectedPrevRawDepth = resolvedDepth / safeRPrevY;

    float depthDiff = histRawDepth - expectedPrevRawDepth;
    if (depthDiff <= 0.0) return 0.0;

    float invExpected  = 1.0 / max(expectedPrevRawDepth, 1e-6);
    float relativeDiff = depthDiff * invExpected;
    float threshold    = taaDepthRejection;

    if (!isDilationZone && !isForegroundEdge)
    {
        // State 1: Continuous Surface / Tilted Plane
        float dxLeft   = resolvedDepth - cachedDepth[3];
        float dxRight  = cachedDepth[4] - resolvedDepth;
        float dyTop    = resolvedDepth - cachedDepth[1];
        float dyBottom = cachedDepth[2] - resolvedDepth;

        float gradX = (abs(dxLeft) < abs(dxRight)) ? dxLeft : dxRight;
        float gradY = (abs(dyTop)  < abs(dyBottom)) ? dyTop  : dyBottom;

        // Exact directional subpixel jitter slack
        float directionalJitterSlack = abs(gradX * exactJitterDelta.x + gradY * exactJitterDelta.y);

        // Directional velocity slack: projects slope onto actual screen motion vector
        // Eliminates the huge isotropic L1 padding on tilted planes
        float2 vDir = (length(pixelVel) > 1e-4) ? normalize(pixelVel) : float2(0.0, 0.0);
        float dirSlope   = abs(gradX * vDir.x + gradY * vDir.y);
        float crossSlope = abs(gradX * -vDir.y + gradY * vDir.x);
        float velSlack   = (dirSlope * 0.75 + crossSlope * 0.25);

        threshold += (directionalJitterSlack + velSlack) * invExpected;

        // Divergence scale
        float2 texSize = 1.0 / max(oneOverTargetSize, 1e-6);
        float divX = ( (cachedVel[4].x - cachedVel[3].x)
                     + (cachedVel[6].x - cachedVel[5].x)
                     + (cachedVel[8].x - cachedVel[7].x) ) * (1.0 / 3.0);
        float divY = ( (cachedVel[2].y - cachedVel[1].y)
                     + (cachedVel[7].y - cachedVel[5].y)
                     + (cachedVel[8].y - cachedVel[6].y) ) * (1.0 / 3.0);
        float divField = divX * texSize.x + divY * texSize.y;
        float t = saturate(max(divField, 0.0) * 0.25);
        threshold += min(t / max(1.0 - t, 0.25), kDollyResidualMax);
    }
    else if (isForegroundEdge)
    {
        // State 3: Foreground Crest / Geometric Horizon
        float jitterL1 = abs(exactJitterDelta.x) + abs(exactJitterDelta.y);
        float curveSlack = crestDrop + foregroundSlope * (1.0 + jitterL1);
        threshold += curveSlack * invExpected;
    }
    else
    {
        // State 2: Dilation Zone (Background pixel adjacent to low-profile foreground)
        float jitterL1 = abs(exactJitterDelta.x) + abs(exactJitterDelta.y);
        threshold += foregroundSlope * (kForegroundSlopeRadius + jitterL1) * invExpected;
    }

    // Motion-scaled reprojection residual (tightened cap)
    float motionTan = length(pixelVel) * oneOverTargetSize.y * 2.0 * taaTanHalfFovY;
    threshold += min(motionTan * kReprojResidualPerTan, kReprojResidualMax);

    return (relativeDiff > threshold) ? 1.0 : 0.0;
}

// ============================================================================
// GAMUT COMPRESSION
// ============================================================================
float3 CompressGamut(float3 historySpace)
{
    if (taaColorSpaceOklab > 0.5)
    {
        float3 historyRGB = OklabToRGB(historySpace);
        float minChannel = min(historyRGB.r, min(historyRGB.g, historyRGB.b));
        if (minChannel < 0.0)
        {
            float luma = max(0.0, LumaRGB(historyRGB));
            float alpha = saturate(luma / max(luma - minChannel, kEpsilon));
            historyRGB = luma + (historyRGB - luma) * alpha;
            historySpace = RGBToOklab(historyRGB);
        }
        return historySpace;
    }

    float Y  = historySpace.x;
    float Co = historySpace.y;
    float Cg = historySpace.z;

    float r = Y + Co - Cg;
    float g = Y + Cg;
    float b = Y - Co - Cg;

    float minCh = min(r, min(g, b));
    if (minCh < 0.0)
    {
        float alpha = saturate(max(0.0, Y) / max(Y - minCh, kEpsilon));
        historySpace.yz *= alpha;
    }
    return historySpace;
}

// ============================================================================
// MAIN PIXEL SHADER
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    // 1. Setup resolution values
    float2 passTexel = oneOverTargetSize;
    float2 passTexSize = 1.0 / max(passTexel, 1e-6);
    float2 minRenderUV = 0.5 * passTexel;
    float2 maxRenderUV = 1.0 - minRenderUV;

    // 2. CPU-precomputed reprojection bases
    float3 Pcurr = float3(taaCurPX,  taaCurPY,  taaCurPZ);
    float3 Qcurr = float3(taaCurQX,  taaCurQY,  taaCurQZ);
    float3 Rcurr = float3(taaCurRX,  taaCurRY,  taaCurRZ);
    float3 Pprev = float3(taaPrevPX, taaPrevPY, taaPrevPZ);
    float3 Qprev = float3(taaPrevQX, taaPrevQY, taaPrevQZ);
    float3 Rprev = float3(taaPrevRX, taaPrevRY, taaPrevRZ);

    // 3. Compute Jitter UVs
    float2 jitterUV = ReprojectUV(IN.uv0, Pcurr, Qcurr, Rcurr);
    float2 jitterPixelPos = jitterUV * passTexSize;
    float2 baseRenderTC = floor(jitterPixelPos) + 0.5;
    float2 snappedRenderUV = clamp(baseRenderTC * passTexel, minRenderUV, maxRenderUV);
    float2 localJitterPx = jitterPixelPos - baseRenderTC;

    // 4. Fetch center pixel samples (tonemapped in ToSpace)
    float3 currentColor = max(tex2Dlod(sceneTex, float4(snappedRenderUV, 0.0, 0.0)).rgb, 0.0);
    float centerRawDepth = tex2Dlod(depthTex, float4(snappedRenderUV, 0.0, 0.0)).r;
    float3 centerColorSpace = ToSpace(currentColor);
    float2 centerVel = tex2Dlod(velocityTex, float4(snappedRenderUV, 0.0, 0.0)).rg;

    // Debug: linear depth view
    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
    {
        float debugLinDepth = saturate(LinearizeDepth(centerRawDepth) / kDebugLinearDepthRange);
        return float4(float3(debugLinDepth, debugLinDepth, debugLinDepth), centerRawDepth);
    }

    // 5. Gather 3x3 depth + velocity
    bool useDilation = (taaUseDepthDilation > 0.5);
    bool needNeighborDepth = useDilation || (taaDepthRejection > 0.001);
    bool needNeighborVel   = useDilation || (taaDepthRejection > 0.001);
    bool storeRGB = (taaFallbackFXAA > 0.5);

    DepthVelocityStats dvStats;
    dvStats.closestRawDepth  = centerRawDepth;
    dvStats.secondClosestRaw = 0.0;
    dvStats.closestOffset    = float2(0.0, 0.0);
    dvStats.secondOffset     = float2(0.0, 0.0);
    dvStats.bestVel = centerVel;

    float4 uvLRBT = float4(
        clamp(snappedRenderUV - passTexel, minRenderUV, maxRenderUV),
        clamp(snappedRenderUV + passTexel, minRenderUV, maxRenderUV)
    );

    float2 tapUVs[9];
    tapUVs[0] = snappedRenderUV;
    tapUVs[1] = float2(snappedRenderUV.x, uvLRBT.y);
    tapUVs[2] = float2(snappedRenderUV.x, uvLRBT.w);
    tapUVs[3] = float2(uvLRBT.x, snappedRenderUV.y);
    tapUVs[4] = float2(uvLRBT.z, snappedRenderUV.y);
    tapUVs[5] = uvLRBT.xy;
    tapUVs[6] = uvLRBT.zy;
    tapUVs[7] = uvLRBT.xw;
    tapUVs[8] = uvLRBT.zw;

    float  cachedDepth[9];
    float2 cachedVel[9];
    cachedDepth[0] = centerRawDepth;
    cachedVel[0]   = centerVel;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float2 offsetUV = tapUVs[i];
        float dRaw = centerRawDepth;
        if (needNeighborDepth) { dRaw = tex2Dlod(depthTex, float4(offsetUV, 0.0, 0.0)).r; }
        float2 v = centerVel;
        if (needNeighborVel) { v = tex2Dlod(velocityTex, float4(offsetUV, 0.0, 0.0)).rg; }

        cachedDepth[i] = dRaw;
        cachedVel[i]   = v;

        if (needNeighborDepth) {
            if (dRaw > dvStats.closestRawDepth) {
                dvStats.secondClosestRaw = dvStats.closestRawDepth;
                dvStats.secondOffset     = dvStats.closestOffset;
                dvStats.closestRawDepth  = dRaw;
                dvStats.closestOffset    = kOffsets3x3[i];
                if (useDilation) { dvStats.bestVel = v; }
            } else if (dRaw > dvStats.secondClosestRaw) {
                dvStats.secondClosestRaw = dRaw;
                dvStats.secondOffset     = kOffsets3x3[i];
            }
        }
    }

    // 5b. Sensitive Signed Midpoint Curvature Velocity & Depth Selection
    float2 v00 = cachedVel[0];
    float2 v10 = (localJitterPx.x >= 0.0) ? cachedVel[4] : cachedVel[3];
    float2 v01 = (localJitterPx.y >= 0.0) ? cachedVel[2] : cachedVel[1];
    float2 v11 = (localJitterPx.x >= 0.0)
        ? ((localJitterPx.y >= 0.0) ? cachedVel[8] : cachedVel[6])
        : ((localJitterPx.y >= 0.0) ? cachedVel[7] : cachedVel[5]);

    // Adaptive Planar Epsilon: 0.6% base relative scale + plane slope noise
    // 4x more sensitive to objects skimming above roads than the old 2.5%
    float planeNoise = (abs(cachedDepth[4] - cachedDepth[3]) + abs(cachedDepth[2] - cachedDepth[1])) * 0.04;
    float eps = max(max(centerRawDepth, 1e-6) * 0.006, planeNoise) + 1e-6;

    float deltaH  = centerRawDepth - 0.5 * (cachedDepth[3] + cachedDepth[4]);
    float deltaV  = centerRawDepth - 0.5 * (cachedDepth[1] + cachedDepth[2]);
    float deltaD1 = centerRawDepth - 0.5 * (cachedDepth[5] + cachedDepth[8]);
    float deltaD2 = centerRawDepth - 0.5 * (cachedDepth[6] + cachedDepth[7]);

    float minDelta = min(min(deltaH, deltaV), min(deltaD1, deltaD2));
    float maxDelta = max(max(deltaH, deltaV), max(deltaD1, deltaD2));

    bool isDilationZone   = useDilation && (minDelta < -eps);
    bool isForegroundEdge = !isDilationZone && (maxDelta > eps);

    float2 subpixelF = abs(localJitterPx);
    float2 bilinearVel = lerp(lerp(v00, v10, subpixelF.x), lerp(v01, v11, subpixelF.x), subpixelF.y);

    float2 resolvedVel;
    float  resolvedDepth;

    if (isDilationZone)
    {
        resolvedVel   = dvStats.bestVel;
        resolvedDepth = dvStats.closestRawDepth;
    }
    else if (isForegroundEdge)
    {
        resolvedVel   = centerVel;
        resolvedDepth = centerRawDepth;
    }
    else
    {
        resolvedVel   = bilinearVel;
        resolvedDepth = centerRawDepth;
    }

    // 5c. True Curvature Drop & Foreground Slope Estimation
    float crestDrop = max(dvStats.closestRawDepth - resolvedDepth, 0.0);
    float crestSpan = max(length(dvStats.closestOffset), 1.0);
    float crestSlope = crestDrop / crestSpan;

    // Disconnect cliff step from interior object slope when only 1-2 foreground pixels exist
    float rawDiff = dvStats.closestRawDepth - dvStats.secondClosestRaw;
    bool secondIsSameObject = (rawDiff < eps * 3.5);
    float fgSpan = max(length(dvStats.closestOffset - dvStats.secondOffset), 1.0);
    float baseSlope = secondIsSameObject ? (max(rawDiff, 0.0) / fgSpan) : 0.0;

    float foregroundSlope = max(crestSlope, baseSlope);

    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        float3 stateDebugColor = currentColor * 0.25;
        if (isDilationZone)
            stateDebugColor = float3(1.0, 0.05, 0.05);
        else if (isForegroundEdge)
            stateDebugColor = float3(0.0, 0.85, 1.0);
        return float4(stateDebugColor, centerRawDepth);
    }

    // 6. History UV and Previous Ray Formulation
    float3 rPrevBase = jitterUV.x * Pprev + Qprev - jitterUV.y * Rprev;
    float3 rPrev = rPrevBase + resolvedVel.x * Pprev - resolvedVel.y * Rprev;
    float2 historyStableUV = ReprojFinish(rPrev, jitterUV + resolvedVel);

    // 7. Motion metrics
    float2 pixelVel = (historyStableUV - IN.uv0) * passTexSize;
    float totalPixelMotion = length(pixelVel);
    float normalizedMotion = saturate(totalPixelMotion / kMotionFullStrengthPx);
    float2 velocityDir = (totalPixelMotion > taaMinMotionDir) ? normalize(pixelVel) : float2(1.0, 0.0);

    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
        return float4(float3(saturate(abs(pixelVel) * kDebugVelocityScale), 0.0), centerRawDepth);

    // 8. Pixel-grid alignment and exact subpixel sampling displacement
    float2 historyPixelCoord = historyStableUV * passTexSize;
    float2 histSubpixel = historyPixelCoord - (floor(historyPixelCoord) + 0.5);
    float pixelAlignment = 1.0 - saturate(length(histSubpixel) * kSqrt2);
    float2 exactJitterDelta = localJitterPx - histSubpixel;

    // 8b. History validity
    float2 support = (taaUseLanczos3 > 0.5 ? 3.0 : 2.0) * passTexel;
    HistoryData hist;
    hist.valid = all(historyStableUV >= support) && all(historyStableUV <= 1.0 - support);
    hist.disocclusion = 1.0; 
    hist.shadowRisk = 0.0; 
    hist.clipDistanceRejection = 0.0;
    hist.colorSpace = centerColorSpace;

    float3 fxaaRGB[5];
    fxaaRGB[0] = currentColor;
    float blend = 0.0;

    if (hist.valid)
    {
        // 9. Fetch neighborhood colors
        float3 cachedSpace[9];
        cachedSpace[0] = centerColorSpace;
        [unroll]
        for (int i = 1; i < 9; ++i)
        {
            float3 cRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
            cachedSpace[i] = ToSpace(cRGB);
            if (storeRGB && i >= 5) { fxaaRGB[i - 4] = cRGB; }
        }

        NeighborhoodStats colorStats = ComputeNeighborhoodStats(cachedSpace, velocityDir, normalizedMotion, localJitterPx);
        float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

        // 10. Fetch and evaluate temporal history 
        float histRawDepth = 1.0;
        if (taaDepthRejection > 0.001)
        {
            float histBilinearDepth;
            float histClosestDepth;
            SampleHistoryDepth(historyStableUV, passTexel, passTexSize, minRenderUV, maxRenderUV,
                               histBilinearDepth, histClosestDepth);

            histRawDepth = (isDilationZone || isForegroundEdge) ? histClosestDepth : histBilinearDepth;
        }

        hist.disocclusion = ComputeDisocclusion(resolvedDepth, histRawDepth, cachedDepth, cachedVel,
                                                pixelVel, rPrev.y, exactJitterDelta, foregroundSlope,
                                                crestDrop, isDilationZone, isForegroundEdge);

        if (taaUseLanczos3 > 0.5) {
            hist.colorSpace = SampleHistoryLanczos3(historyStableUV, passTexSize, passTexel, minRenderUV, maxRenderUV);
        } else {
            hist.colorSpace = SampleHistoryCatmullRom5Tap(historyStableUV, passTexSize, passTexel, minRenderUV, maxRenderUV);
        }

        // 10b. Luminance drift correction
        if (taaLumaDriftStrength > 0.001) {
            float2 hChi = hist.colorSpace.yz / max(abs(hist.colorSpace.x), kLumaDriftLumaFloor);
            float2 cChi = centerColorSpace.yz / max(abs(centerColorSpace.x), kLumaDriftLumaFloor);
            float  chromaDist = length(hChi - cChi);

            float tol = max(taaLumaDriftChromaTol, 1e-3);
            float sameSurface = 1.0 - smoothstep(tol, tol * kLumaDriftChromaFadeWidth, chromaDist);

            if (sameSurface > 0.001) {
                float muY = colorStats.mu.x;
                float hY  = hist.colorSpace.x;
                if (abs(muY - hY) > max(kLumaDriftRelThreshold * abs(hY), kLumaDriftAbsThreshold)) {
                    hist.colorSpace.x = max(hY + (muY - hY) * taaLumaDriftStrength * sameSurface, 1e-3);
                }
            }
        }

        if (taaShadowMitigation > 0.5) {
            hist.shadowRisk = (1.0 - smoothstep(0.0, max(kShadowThresholdMin, taaShadowDarknessThreshold), centerColorSpace.x)) 
                            * saturate(abs(max(centerColorSpace.x, kShadowLumaFloor) - max(hist.colorSpace.x, kShadowLumaFloor)) * taaShadowTemporalMult) 
                            * (1.0 - saturate(colorStats.spatialContrast * taaShadowSpatialMult));
        }

        float dynamicGamma = max(taaVarianceGamma, 0.0);
        if (taaShadowMitigation > 0.5) { 
            dynamicGamma += (hist.shadowRisk * max(taaVarianceGamma, taaShadowVarianceBase)) * (1.0 - hist.disocclusion); 
        }

        // 11. History clipping and relative metrics
        float3 unclippedHistorySpace = hist.colorSpace;
        hist.colorSpace = ClipHistory(hist.colorSpace, colorStats, normalizedMotion, dynamicGamma, cachedSpace, clipMargin);

        if (taaClipDistanceRejectionEnabled > 0.5) { 
            float clipDistance = length(unclippedHistorySpace - hist.colorSpace);
            float rejectionAmount = max(taaClipDistanceRejectionAmount, 0.05);
            hist.clipDistanceRejection = smoothstep(
                taaClipDistanceRejectionMinError, 
                taaClipDistanceRejectionMinError + rejectionAmount, 
                clipDistance
            );
        }

        // 12. Gamut compression
        hist.colorSpace = CompressGamut(hist.colorSpace);

        // 13. Final Blend factor mapping
        float motionBlendEnd = max(taaMotionBlendDropSpeed, taaMotionBlendStart + kMinMotionBlendDropSpeed);
        blend = lerp(taaFeedbackMax, taaFeedbackMin, smoothstep(taaMotionBlendStart, motionBlendEnd, totalPixelMotion));
        blend = lerp(blend, blend * taaAlignmentFeedbackDrop, 1.0 - pixelAlignment);
        
        if (taaShadowMitigation > 0.5) { blend = lerp(blend, taaShadowBlendStrength, hist.shadowRisk); }
        if (taaClipDistanceRejectionEnabled > 0.5) { blend = lerp(blend, 0.0, hist.clipDistanceRejection); }
        blend = lerp(blend, 0.0, hist.disocclusion);
    }
    else if (storeRGB)
    {
        [unroll]
        for (int i = 5; i < 9; ++i)
        {
            fxaaRGB[i - 4] = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        }
    }

    // 14. Fallback edge smoothing
    float fxaaBlend = saturate((kFXAABlendKnee - blend) * kFXAABlendSharpness);
    if (taaFallbackFXAA > 0.5 && fxaaBlend > 0.0)
    {
        float3 fxaaColor = ApplyFXAACached(snappedRenderUV, passTexel, fxaaRGB);
        centerColorSpace = lerp(centerColorSpace, ToSpace(fxaaColor), fxaaBlend);
    }

    // 15. Direct Debug Output
    if (taaDebugMode > 0.5)
    {
        float3 debugColor = float3(0.0, 0.0, 0.0);
        if (taaDebugMode < 2.5) {
            debugColor = max(FromSpace(hist.colorSpace), 0.0);
        } else if (taaDebugMode < 3.5) {
            float3 disoccHighlight = float3(1.0, 0.1, 0.2);
            debugColor = (hist.disocclusion > 0.5) ? lerp(currentColor, disoccHighlight, 0.75) : (currentColor * 0.4);
        } else if (taaDebugMode < 5.5) {
            debugColor = float3(0.0, hist.shadowRisk, hist.shadowRisk);
        } else if (taaDebugMode < 6.5) {
            debugColor = float3(0.0, 0.0, 0.0);
        } else if (taaDebugMode < 7.5) {
            debugColor = float3(blend, blend, blend);
        }

        return float4(debugColor, centerRawDepth);
    }

    // 16. Resolve outputs
    float3 blendedSpace = lerp(centerColorSpace, hist.colorSpace, blend);
    return float4(max(FromSpace(blendedSpace), 0.0), centerRawDepth);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}