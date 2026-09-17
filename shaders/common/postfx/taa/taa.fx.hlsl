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
static const float kLargeValue              = 1e5;
static const float kSqrt2                   = 1.41421356;
static const float kLog2E                   = 1.44269504;  // exp(x) == exp2(x * kLog2E)
static const float taaMinMotionDir          = 0.1;
static const float kShadowLumaFloor         = 0.01;
static const float kShadowThresholdMin      = 0.001;
static const float kFlickerPadThreshold     = 0.001;
static const float kMinMotionBlendDropSpeed = 0.1;

// Motion, in pixels, at which motion-gated effects reach full strength.
static const float kMotionFullStrengthPx    = 8.0;

static const float kMinSigma                = 0.001;  // sigma floor for clipping
static const float kMinSpatialContrast      = 0.001;  // spatial contrast floor
static const float kMinFootprintRange       = 1e-4;   // history footprint / slab range floor
static const float kFireflyClampEpsilon     = 0.001;  // below this the firefly clamp counts as "off"

// Fallback FXAA
static const float kFXAAReduceMul           = 1.0 / 128.0;
static const float kFXAAReduceMin           = 1.0 / 128.0;
static const float kFXAAMaxDir              = 8.0;
static const float kFXAABlendKnee           = 0.8;    // blend level below which FXAA engages
static const float kFXAABlendSharpness      = 2.0;    // steepness of the FXAA engage ramp

// Luma drift correction
static const float kLumaDriftLumaFloor      = 0.05;   // luma floor for the chroma-ratio test
static const float kLumaDriftRelThreshold   = 0.30;   // relative luma error that triggers correction
static const float kLumaDriftAbsThreshold   = 0.03;   // absolute luma error that triggers correction
static const float kLumaDriftChromaFadeWidth = 2.0;   // chroma tolerance fade, in multiples of tol

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
// COLOR SPACES
// ============================================================================
float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

static const float3x3 kRGB_TO_LMS = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800,  2.60975740, -0.34131939, -0.00419608, -0.70341861,  1.70761470);

float3 RGBToOklab(float3 c) { return mul(kLMS_TO_OKLAB, pow(max(mul(kRGB_TO_LMS, c), 0.0), 1.0 / 3.0)); }
float3 OklabToRGB(float3 c) { float3 l = mul(kOKLAB_TO_LMS, c); return mul(kLMS_TO_RGB, l * l * l); }
float3 RGBToYCoCg(float3 c) { return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
float3 YCoCgToRGB(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

float3 ToSpace(float3 rgb) { return (taaColorSpaceOklab > 0.5) ? RGBToOklab(rgb) : RGBToYCoCg(rgb); }
float3 FromSpace(float3 c) { return (taaColorSpaceOklab > 0.5) ? OklabToRGB(c) : YCoCgToRGB(c); }

// ============================================================================
// MATRIX UTILITIES (Symmetric Optimized)
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
float2 ReprojFinish(float3 r2)
{
    if (r2.y <= kEpsilon) return float2(-1.0, -1.0);
    return float2((r2.x / (r2.y * taaTanHalfFovX)) * 0.5 + 0.5,
                  0.5 - (r2.z / (r2.y * taaTanHalfFovY)) * 0.5);
}

float2 ReprojectUV(float2 uv, float3 P, float3 Q, float3 R)
{
    return ReprojFinish(uv.x * P + Q - uv.y * R);
}

float2 ReprojectPrev(float3 rBase, float2 vel, float3 P, float3 R)
{
    return ReprojFinish(rBase + vel.x * P - vel.y * R);
}

float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

float3 RayFromUV(float2 uv)
{
    return float3((uv.x * 2.0 - 1.0) * taaTanHalfFovX, 1.0, (1.0 - uv.y * 2.0) * taaTanHalfFovY);
}

// ============================================================================
// FALLBACK FXAA
// ============================================================================
// fxaaRGB layout: [0] = M, [1] = NW, [2] = NE, [3] = SW, [4] = SE
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
float3 SampleHistoryLanczos3(float2 uv, float2 texSize, float2 invTexSize, float2 minUV, float2 maxUV)
{
    float2 samplePos = uv * texSize;
    float2 tc = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc;

    float wX[6], wY[6]; 
    float2 sumXY = 0.0;

    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        float2 d = abs(f.xy + float(2 - i));
        float2 piD = 3.14159265 * max(d, 1e-5);
        float2 w = 3.0 * sin(piD) * sin(piD * (1.0 / 3.0)) / (piD * piD);
        wX[i] = w.x;
        wY[i] = w.y;
        sumXY += w;
    }

    float2 invSum = 1.0 / max(sumXY, 1e-5);
    [unroll] for (int j = 0; j < 6; ++j) { wX[j] *= invSum.x; wY[j] *= invSum.y; }

    float uvsX[6], uvsY[6];
    [unroll]
    for (int k = 0; k < 6; ++k)
    {
        uvsX[k] = clamp((tc.x + float(k - 2)) * invTexSize.x, minUV.x, maxUV.x);
        uvsY[k] = clamp((tc.y + float(k - 2)) * invTexSize.y, minUV.y, maxUV.y);
    }

    float3 color = 0.0;
    float3 footMin = kLargeValue;
    float3 footMax = -kLargeValue;

    [unroll]
    for (int y = 0; y < 6; ++y)
    {
        [unroll]
        for (int x = 0; x < 6; ++x)
        {
            float3 tapColor = max(0.0, tex2Dlod(historyTex, float4(uvsX[x], uvsY[y], 0.0, 0.0)).rgb);
            color += tapColor * (wX[x] * wY[y]);

            footMin = min(footMin, tapColor);
            footMax = max(footMax, tapColor);
        }
    }

    float3 footRng = max(footMax - footMin, kMinFootprintRange);
    float3 footMargin = taaHistoryOvershoot * footRng;
    color = clamp(color, footMin - footMargin, footMax + footMargin);

    return ToSpace(max(0.0, color));
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

    float3 tap0 = max(0.0, tex2Dlod(historyTex, float4(tc12.x, tc0.y, 0.0, 0.0)).rgb);
    float3 tap1 = max(0.0, tex2Dlod(historyTex, float4(tc0.x, tc12.y, 0.0, 0.0)).rgb);
    float3 tap2 = max(0.0, tex2Dlod(historyTex, float4(tc12.x, tc12.y, 0.0, 0.0)).rgb);
    float3 tap3 = max(0.0, tex2Dlod(historyTex, float4(tc3.x, tc12.y, 0.0, 0.0)).rgb);
    float3 tap4 = max(0.0, tex2Dlod(historyTex, float4(tc12.x, tc3.y, 0.0, 0.0)).rgb);

    float w0y   = w12.x * w0.y;
    float w0x   = w0.x * w12.y;
    float w12xy = w12.x * w12.y;
    float w3x   = w3.x * w12.y;
    float w3y   = w12.x * w3.y;

    float3 color = tap0 * w0y + tap1 * w0x + tap2 * w12xy + tap3 * w3x + tap4 * w3y;
    float wsum = w0y + w0x + w12xy + w3x + w3y;
    color /= max(wsum, 1e-4);

    float3 footMin = min(tap0, min(tap1, min(tap2, min(tap3, tap4))));
    float3 footMax = max(tap0, max(tap1, max(tap2, max(tap3, tap4))));

    float3 footRng = max(footMax - footMin, kMinFootprintRange);
    float3 footMargin = taaHistoryOvershoot * footRng;
    color = clamp(color, footMin - footMargin, footMax + footMargin);

    return ToSpace(max(0.0, color));
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct DepthVelocityStats { 
    float closestRawDepth;
    float2 bestVel; 
    float2 closestTapUV;
    float rightRawDepth;
    float2 rightTapUV;
    float downRawDepth;
    float2 downTapUV;
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
// 2x2 DILATED HISTORY DEPTH LOOKUP
// ============================================================================
float SampleDilatedHistoryDepth(float2 historyUV, float2 passTexel, float2 passTexSize, float2 minUV, float2 maxUV)
{
    float2 samplePos = historyUV * passTexSize - 0.5;
    float2 tc00 = floor(samplePos);
    
    float2 uvs[4];
    uvs[0] = clamp((tc00 + float2(0.5, 0.5)) * passTexel, minUV, maxUV);
    uvs[1] = clamp((tc00 + float2(1.5, 0.5)) * passTexel, minUV, maxUV);
    uvs[2] = clamp((tc00 + float2(0.5, 1.5)) * passTexel, minUV, maxUV);
    uvs[3] = clamp((tc00 + float2(1.5, 1.5)) * passTexel, minUV, maxUV);
    
    float d0 = tex2Dlod(historyTex, float4(uvs[0], 0.0, 0.0)).a;
    float d1 = tex2Dlod(historyTex, float4(uvs[1], 0.0, 0.0)).a;
    float d2 = tex2Dlod(historyTex, float4(uvs[2], 0.0, 0.0)).a;
    float d3 = tex2Dlod(historyTex, float4(uvs[3], 0.0, 0.0)).a;
    
    return max(max(d0, d1), max(d2, d3));
}

// ============================================================================
// NEIGHBORHOOD STATISTICS
// ============================================================================
NeighborhoodStats ComputeNeighborhoodStats(float3 cachedSpace[9], float2 velocityDir, float normalizedMotion, float2 localJitterPx)
{
    NeighborhoodStats stats; stats.validCovariance = false;
    stats.aabbMin  = kLargeValue; stats.aabbMax  = -kLargeValue;

    bool wantCovariance = (taaUseCovarianceClipping > 0.5) && (taaUseKDopClipping <= 0.5);
    bool needExpectedShift = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 m1 = 0.0, m2 = 0.0;
    float3 mCross = 0.0;
    float weightSum = 0.0;
    float motionFactor = saturate(normalizedMotion);

    bool useShiftedCenter = (taaJitterAwareVariance > 0.5);
    float2 centerShift = useShiftedCenter ? localJitterPx : 0.0;

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

    stats.expectedColorShift = 0.0;
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
        float farHit = kLargeValue;

        [unroll]
        for (int a = 0; a < 16; ++a)
        {
            float3 axis = kDopAxes[a]; 
            float2 extents = float2(kLargeValue, -kLargeValue);
            
            float proj[9];
            [unroll]
            for (int n = 0; n < 9; ++n) { proj[n] = dot(cachedSpace[n], axis); }
            
            float proj_pos = dot(rayCenter, axis);

            if (taaKDopVariance > 0.5)
            {
                float2 moments = 0.0;
                float wSum = 0.0;
                float pMin = kLargeValue; 
                float pMax = -kLargeValue;
                [unroll]
                for (int n = 0; n < 9; ++n) 
                { 
                    float w = stats.weights[n];
                    moments += float2(proj[n], proj[n] * proj[n]) * w; 
                    wSum += w;
                    pMin = min(pMin, proj[n]);
                    pMax = max(pMax, proj[n]);
                }
                moments /= wSum;
                
                float mu = moments.x; 
                float sigma = sqrt(max(moments.y - mu * mu, 0.0));
                float expandedSigma = sigma * dynamicGamma;
                
                extents.x = mu - expandedSigma;
                extents.y = mu + expandedSigma;

                float axisMargin = taaClipOvershoot * max(pMax - pMin, kMinFootprintRange);
                extents.x -= axisMargin;
                extents.y += axisMargin;
            }
            else
            {
                [unroll]
                for (int n = 0; n < 9; ++n) { extents.x = min(proj[n], extents.x); extents.y = max(proj[n], extents.y); }
                extents.x -= kEpsilon;
                extents.y += kEpsilon;

                float axisMargin = taaClipOvershoot * max(extents.y - extents.x, kMinFootprintRange);
                extents.x -= axisMargin;
                extents.y += axisMargin;
            }

            float dir_dot = dot(dir, axis); 
            float s_dot = (dir_dot >= 0.0 ? 1.0 : -1.0);
            float inv_dir = 1.0 / (s_dot * max(abs(dir_dot), 1e-7));
            float t0 = (extents.x - proj_pos) * inv_dir; 
            float t1 = (extents.y - proj_pos) * inv_dir;

            nearHit = max(nearHit, min(t0, t1)); 
            farHit = min(farHit, max(t0, t1));
        }

        if (nearHit <= farHit && (nearHit > 0.0 || farHit > 0.0))
        {
            float t_hit = clamp(nearHit > 0.0 ? nearHit : farHit, 0.0, 1.0);
            if (t_hit < 1.0)
            {
                float maxUnit = 1.0 / max(t_hit, kEpsilon);
                float softLimit = 1.0 + taaSoftClip * (1.0 - exp2(-(maxUnit - 1.0) * kLog2E));
                return rayCenter + dir * (lerp(softLimit, 1.0, motionFactor) / maxUnit);
            }
        }
        return historySpace;
    }

    if (stats.validCovariance)
    {
        float3 diff = historySpace - stats.mu; 
        float d2 = dot(diff, mul(stats.invCov, diff)); 
        float gamma2 = dynamicGamma * dynamicGamma;
        float3 clipped = (d2 > gamma2 && d2 > kEpsilon) ? (stats.mu + diff * (dynamicGamma / sqrt(d2))) : historySpace;
        return IntersectRayAABB(clipped, stats.mu, stats.aabbMin - clipMargin, stats.aabbMax + clipMargin, taaSoftClip, motionFactor);
    }

    float3 chromaWeights = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 extents = stats.sigma * dynamicGamma * chromaWeights;
    float3 bMin = max(stats.mu - extents, stats.aabbMin - clipMargin);
    float3 bMax = min(stats.mu + extents, stats.aabbMax + clipMargin);

    return IntersectRayAABB(historySpace, stats.mu, bMin, bMax, taaSoftClip, motionFactor);
}

// ============================================================================
// DISOCCLUSION
// ============================================================================
float ComputeDisocclusion(DepthVelocityStats dv, float histRawDepth, float centerRawDepth, float2 centerUV, float2 historyUV)
{
    if (taaDepthRejection <= 0.001) return 0.0;

    float linC = LinearizeDepth(centerRawDepth);
    float histLin = LinearizeDepth(histRawDepth);

    float3 pC = RayFromUV(centerUV) * linC;
    float3 pX = RayFromUV(dv.rightTapUV) * LinearizeDepth(dv.rightRawDepth);
    float3 pY = RayFromUV(dv.downTapUV) * LinearizeDepth(dv.downRawDepth);

    float3 n = cross(pX - pC, pY - pC);
    float nLen = length(n);
    if (nLen < 1e-6) return 0.0;
    n /= nLen;

    float3 pA = RayFromUV(dv.closestTapUV) * LinearizeDepth(dv.closestRawDepth);
    if (dot(n, pA) > 0.0) { n = -n; }

    float3 pH = RayFromUV(historyUV) * histLin;
    float planeDist = dot(n, pH - pA);

    return (planeDist > taaDepthRejection * histLin) ? 1.0 : 0.0;
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
    float2 passTexSize = 1.0 / passTexel;
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

    // 4. Fetch center pixel samples
    float3 currentColor = max(0.0, tex2Dlod(sceneTex, float4(snappedRenderUV, 0.0, 0.0)).rgb);
    float centerRawDepth = tex2Dlod(depthTex, float4(snappedRenderUV, 0.0, 0.0)).r;
    float3 centerColorSpace = ToSpace(currentColor);
    float2 centerVel = tex2Dlod(velocityTex, float4(snappedRenderUV, 0.0, 0.0)).rg;

    // Debug: linear depth view
    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
        return float4(saturate(LinearizeDepth(centerRawDepth) / kDebugLinearDepthRange).xxx, centerRawDepth);

    // 5. Gather 3x3 depth + velocity. Color taps are deferred to step 9 so
    //    they can be skipped entirely when the history is invalid and the
    //    FXAA fallback is disabled.
    bool useDilation = (taaUseDepthDilation > 0.5);
    bool needNeighborDepth = useDilation || (taaDepthRejection > 0.001);
    bool storeRGB = (taaFallbackFXAA > 0.5);

    DepthVelocityStats dvStats;
    dvStats.closestRawDepth = centerRawDepth;
    dvStats.bestVel = centerVel;
    dvStats.closestTapUV = snappedRenderUV;
    dvStats.rightRawDepth = centerRawDepth;
    dvStats.rightTapUV = snappedRenderUV;
    dvStats.downRawDepth = centerRawDepth;
    dvStats.downTapUV = snappedRenderUV;

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
        if (useDilation) { v = tex2Dlod(velocityTex, float4(offsetUV, 0.0, 0.0)).rg; }

        cachedDepth[i] = dRaw;
        cachedVel[i]   = v;

        if (needNeighborDepth) {
            if (i == 2) { dvStats.downRawDepth  = dRaw; dvStats.downTapUV  = offsetUV; }
            if (i == 4) { dvStats.rightRawDepth = dRaw; dvStats.rightTapUV = offsetUV; }
            if (dRaw > dvStats.closestRawDepth) {
                dvStats.closestRawDepth = dRaw;
                dvStats.closestTapUV = offsetUV;
                if (useDilation) { dvStats.bestVel = v; }
            }
        }
    }

// 5b. Signed Midpoint Curvature Velocity Selection
    int idxX = (localJitterPx.x >= 0.0) ? 4 : 3;
    int idxY = (localJitterPx.y >= 0.0) ? 2 : 1;
    int idxCorner;
    if (localJitterPx.x >= 0.0)
        idxCorner = (localJitterPx.y >= 0.0) ? 8 : 6;
    else
        idxCorner = (localJitterPx.y >= 0.0) ? 7 : 5;

    float2 v00 = cachedVel[0];
    float2 v10 = cachedVel[idxX];
    float2 v01 = cachedVel[idxY];
    float2 v11 = cachedVel[idxCorner];

    // 1. Distance-invariant scale (2.5% of local perspective depth)
    float eps = max(centerRawDepth, 1e-6) * 0.025 + 1e-7;

    // 2. Measure signed midpoint deviation across all 4 axes:
    // delta = center - midpoint(opposite_neighbors)
    float deltaH  = centerRawDepth - 0.5 * (cachedDepth[3] + cachedDepth[4]); // Left / Right
    float deltaV  = centerRawDepth - 0.5 * (cachedDepth[1] + cachedDepth[2]); // Top / Bottom
    float deltaD1 = centerRawDepth - 0.5 * (cachedDepth[5] + cachedDepth[8]); // TL / BR
    float deltaD2 = centerRawDepth - 0.5 * (cachedDepth[6] + cachedDepth[7]); // TR / BL

    float minDelta = min(min(deltaH, deltaV), min(deltaD1, deltaD2));
    float maxDelta = max(max(deltaH, deltaV), max(deltaD1, deltaD2));

    // 3. Geometric State Classification:
    // - minDelta < -eps: Center is depressed below neighbors (Background next to incoming foreground)
    // - maxDelta > +eps: Center is elevated above neighbors (Foreground edge / 1-px wire)
    // - within [-eps, +eps]: Center lies on the planar midpoint (Continuous surface)
    bool isDilationZone   = useDilation && (minDelta < -eps);
    bool isForegroundEdge = !isDilationZone && (maxDelta > eps);

    // 4. Compute subpixel bilinear velocity for State 1
    float2 subpixelF = abs(localJitterPx);
    float2 bilinearVel = lerp(lerp(v00, v10, subpixelF.x), lerp(v01, v11, subpixelF.x), subpixelF.y);

    // 5. Resolve velocity based on the 3 states
    float2 resolvedVel;
    if (isDilationZone)
    {
        // State 2: Dilate discrete foreground velocity
        resolvedVel = dvStats.bestVel;
    }
    else if (isForegroundEdge)
    {
        // State 3: Pure discrete center velocity (no bilinear mixing across edge)
        resolvedVel = centerVel;
    }
    else
    {
        // State 1: Continuous planar surface; use subpixel bilinear velocity
        resolvedVel = bilinearVel;
    }

    // Debug: Velocity State Classification (Mode 8)
    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        float3 stateDebugColor = currentColor * 0.25; // Continuous = Dimmed Scene
        if (isDilationZone)
            stateDebugColor = float3(1.0, 0.05, 0.05); // Dilation Zone = Bright Red
        else if (isForegroundEdge)
            stateDebugColor = float3(0.0, 0.85, 1.0);  // Foreground Edge / 1-px Wire = Bright Cyan
        return float4(stateDebugColor, centerRawDepth);
    }

    // 6. History UV from the resolved velocity
    float3 rPrevBase = jitterUV.x * Pprev + Qprev - jitterUV.y * Rprev;
    float2 historyStableUV = ReprojectPrev(rPrevBase, resolvedVel, Pprev, Rprev);

    // 7. Motion metrics
    float2 pixelVel = (historyStableUV - IN.uv0) * passTexSize;
    float totalPixelMotion = length(pixelVel);
    float normalizedMotion = saturate(totalPixelMotion / kMotionFullStrengthPx);
    float2 velocityDir = (totalPixelMotion > taaMinMotionDir) ? normalize(pixelVel) : float2(1.0, 0.0);

    // Debug: velocity view
    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
        return float4(float3(saturate(abs(pixelVel) * kDebugVelocityScale), 0.0), centerRawDepth);

    // 8. Pixel-grid alignment of the history sample
    float2 historyPixelCoord = historyStableUV * passTexSize;
    float pixelAlignment = 1.0 - saturate(length(historyPixelCoord - (floor(historyPixelCoord) + 0.5)) * kSqrt2);

    // 8b. History validity (hoisted above the color work so steps 9+ can be gated)
    float2 support = (taaUseLanczos3 > 0.5 ? 3.0 : 2.0) * passTexel;
    HistoryData hist;
    hist.valid = all(historyStableUV >= support) && all(historyStableUV <= 1.0 - support);
    hist.disocclusion = 1.0; 
    hist.shadowRisk = 0.0; 
    hist.clipDistanceRejection = 0.0;
    hist.colorSpace = centerColorSpace;

    // FXAA corner cache: [0] = M, [1] = NW, [2] = NE, [3] = SW, [4] = SE
    float3 fxaaRGB[5];
    fxaaRGB[0] = currentColor;

    float blend = 0.0;

    if (hist.valid)
    {
        // 9. Fetch neighborhood colors and convert. Only runs when the
        //    history is valid; the 4 FXAA corner taps are cached alongside.
        float3 cachedSpace[9];
        cachedSpace[0] = centerColorSpace;
        [unroll]
        for (int i = 1; i < 9; ++i)
        {
            float3 cRGB = max(0.0, tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb);
            cachedSpace[i] = ToSpace(cRGB);
            if (storeRGB && i >= 5) { fxaaRGB[i - 4] = cRGB; }
        }

        // 9a. Compute spatial neighborhood statistics
        NeighborhoodStats colorStats = ComputeNeighborhoodStats(cachedSpace, velocityDir, normalizedMotion, localJitterPx);

        // 9b. CLIP OVERSHOOT margin
        float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

        // 10. Fetch and evaluate temporal history 
        float histRawDepth = 1.0;
        if (taaDepthRejection > 0.001)
        {
            if (useDilation) {
                histRawDepth = SampleDilatedHistoryDepth(historyStableUV, passTexel, passTexSize, minRenderUV, maxRenderUV);
            } else {
                histRawDepth = tex2Dlod(historyTex, float4(clamp((floor(historyPixelCoord) + 0.5) * passTexel, minRenderUV, maxRenderUV), 0.0, 0.0)).a;
            }
        }

        hist.disocclusion = ComputeDisocclusion(dvStats, histRawDepth, centerRawDepth, snappedRenderUV, historyStableUV);

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

        // 11. History clipping and metrics
        float3 unclippedHistorySpace = hist.colorSpace;
        hist.colorSpace = ClipHistory(hist.colorSpace, colorStats, normalizedMotion, dynamicGamma, cachedSpace, clipMargin);

        if (taaClipDistanceRejectionEnabled > 0.5) { 
            float clipDistance = length(unclippedHistorySpace - hist.colorSpace);
            hist.clipDistanceRejection = saturate((clipDistance - taaClipDistanceRejectionMinError) / max(taaClipDistanceRejectionAmount, 1e-5)); 
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
        // History invalid but FXAA active: fetch only the 4 corner taps.
        [unroll]
        for (int i = 5; i < 9; ++i)
        {
            fxaaRGB[i - 4] = max(0.0, tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb);
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
        float3 debugColor = 0.0;
        if (taaDebugMode < 2.5) {
            debugColor = max(FromSpace(hist.colorSpace), 0.0);
        } else if (taaDebugMode < 3.5) {
            float3 disoccHighlight = float3(1.0, 0.1, 0.2);
            debugColor = (hist.disocclusion > 0.5) ? lerp(currentColor, disoccHighlight, 0.75) : (currentColor * 0.4);
        } else if (taaDebugMode < 5.5) {
            debugColor = float3(0.0, hist.shadowRisk, hist.shadowRisk);
        } else if (taaDebugMode < 6.5) {
            debugColor = 0.0; // reserved: was the confidence view (removed)
        } else if (taaDebugMode < 7.5) {
            debugColor = blend.xxx;
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