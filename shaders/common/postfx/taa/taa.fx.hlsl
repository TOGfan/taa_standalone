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
    float  taaFeedbackMin;              float  taaFeedbackMax;              
    float  taaShadowMitigation;         float  taaShadowDarknessThreshold;
    float  taaShadowBlendStrength;      float  taaVarianceGamma;            float  taaSoftClip;
    float  taaChromaVarianceMod;        float  taaJitterFlickerPadding;     float  taaJitterFlickerFade;
    float  taaDepthRejection;           float  taaVelDisocclusion;          float  taaTanHalfFovX;
    float  taaTanHalfFovY;              float  taaJitterYaw;                float  taaJitterPitch;              
    float  taaPrevJitterYaw;            float  taaPrevJitterPitch;          float  taaUseDepthDilation;         
    float  taaAdaptiveVariance;         float  taaLumaVariance;             float  taaUseCovarianceClipping;    
    float  taaColorSpaceOklab;          float  taaJitterAwareVariance;      float  taaVelocityAlignedVariance;
    float  taaAlignmentFeedbackDrop;    float  taaMotionBlendDropSpeed;     float  taaBilinearHistoryVel;       
    float  taaRoundedAABB;              float  taaUseLanczos3;              float  taaFireflyClamp;             
    float  taaAdaptiveVarStart;         float  taaAdaptiveVarEnd;           float  taaShadowTemporalMult;       
    float  taaShadowSpatialMult;        float  taaDirectionalVariance;      
    float  taaClipDistanceRejectionEnabled; float  taaClipDistanceRejectionAmount; float  taaClipDistanceRejectionMinError;
    float  taaUseKDopClipping;          float  taaKDopVariance;             float  taaFallbackFXAA;
    float  taaDepthRejRelStatic;        float  taaDepthRejRelMoving;        float  taaDepthRejAbs;
    float  taaVelRejBaseStatic;         float  taaVelRejBaseMoving;         float  taaVelRejMotionScale;
    float  taaMotionBlendStart;         float  taaShadowVarianceBase;       float  taaCollapseRatioMin;
    float  taaCollapseRatioMax;

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
static const float kEpsilon               = 1e-5;
static const float kLargeValue            = 1e5;
static const float kSqrt2                 = 1.41421356;
static const float taaMinMotionDir        = 0.1;
static const float kShadowLumaFloor       = 0.01;
static const float kShadowThresholdMin    = 0.001;
static const float kFlickerPadThreshold   = 0.001;
static const float kVelRejToleranceOffset = 0.01;
static const float kMinMotionBlendDropSpeed = 0.1;
static const float kFSRConfidenceThreshold= 0.3;

static const float2 kOffsets3x3[9] =
{
    float2( 0,  0), float2( 0, -1), float2( 0,  1),
    float2(-1,  0), float2( 1,  0), float2(-1, -1),
    float2( 1, -1), float2(-1,  1), float2( 1,  1)
};

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
float PerceptualLuma(float3 c) { return LumaRGB(c); }

static const float3x3 kRGB_TO_LMS = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800,  2.60975740, -0.34131939, -0.00419608, -0.70341861,  1.70761470);

float3 RGBToOklab(float3 c) { return mul(kLMS_TO_OKLAB, pow(max(mul(kRGB_TO_LMS, c), 0.0), 1.0 / 3.0)); }
float3 OklabToRGB(float3 c) { float3 l = mul(kOKLAB_TO_LMS, c); return mul(kLMS_TO_RGB, l * l * l); }
float3 RGBToYCoCg(float3 c) { return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r  - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
float3 YCoCgToRGB(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

float3 ToSpace(float3 rgb) { return (taaColorSpaceOklab > 0.5) ? RGBToOklab(rgb) : RGBToYCoCg(rgb); }
float3 FromSpace(float3 c) { return (taaColorSpaceOklab > 0.5) ? OklabToRGB(c) : YCoCgToRGB(c); }

// ============================================================================
// MATRIX UTILITIES
// ============================================================================
float3x3 Inverse3x3(float3x3 m, out bool success)
{
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    success = (abs(det) > kEpsilon);
    if (!success) return float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float invDet = 1.0 / det; float3x3 inv;
    inv[0][0] = (m[1][1]*m[2][2] - m[2][1]*m[1][2]) * invDet; inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet; inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet;
    inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet; inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet; inv[1][2] = (m[1][0]*m[0][2] - m[0][0]*m[1][2]) * invDet;
    inv[2][0] = (m[1][0]*m[2][1] - m[2][0]*m[1][1]) * invDet; inv[2][1] = (m[2][0]*m[0][1] - m[0][0]*m[2][1]) * invDet; inv[2][2] = (m[0][0]*m[1][1] - m[1][0]*m[0][1]) * invDet;
    return inv;
}

// ============================================================================
// REPROJECTION & DEPTH
// ============================================================================
// Optimized: sp, cp, sy, cy are now precalculated to save 8 sincos() instructions per pixel
float2 ReprojectUV(float2 uv, float sp, float cp, float sy, float cy)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 ray = float3(ndc.x * taaTanHalfFovX, 1.0, ndc.y * taaTanHalfFovY);
    
    float3 r1 = float3(ray.x, ray.y * cp - ray.z * sp, ray.y * sp + ray.z * cp);
    float3 r2 = float3(r1.x * cy - r1.y * sy, r1.x * sy + r1.y * cy, r1.z);
    
    if (r2.y <= kEpsilon) return float2(-1.0, -1.0);
    return float2((r2.x / (r2.y * taaTanHalfFovX)) * 0.5 + 0.5, 0.5 - (r2.z / (r2.y * taaTanHalfFovY)) * 0.5);
}
float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

// ============================================================================
// FALLBACK FXAA
// ============================================================================
float3 ApplyFXAA(float2 uv, float2 texel, float3 centerColor) 
{
    float lumaNW = PerceptualLuma(tex2Dlod(sceneTex, float4(uv + float2(-1.0, -1.0) * texel, 0.0, 0.0)).rgb);
    float lumaNE = PerceptualLuma(tex2Dlod(sceneTex, float4(uv + float2( 1.0, -1.0) * texel, 0.0, 0.0)).rgb);
    float lumaSW = PerceptualLuma(tex2Dlod(sceneTex, float4(uv + float2(-1.0,  1.0) * texel, 0.0, 0.0)).rgb);
    float lumaSE = PerceptualLuma(tex2Dlod(sceneTex, float4(uv + float2( 1.0,  1.0) * texel, 0.0, 0.0)).rgb);
    float lumaM  = PerceptualLuma(centerColor);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * (1.0/128.0)), (1.0/128.0));
    float rcpDirMin = 1.0 / (min(abs(lumaMax - lumaMin), max(lumaMax, 1.0)) + dirReduce);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    dir = clamp(dir * rcpDirMin, float2(-8.0, -8.0), float2(8.0, 8.0)) * texel;

    float3 rgbA = 0.5 * (
        tex2Dlod(sceneTex, float4(uv + dir * (1.0/3.0 - 0.5), 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv + dir * (2.0/3.0 - 0.5), 0.0, 0.0)).rgb);
        
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        tex2Dlod(sceneTex, float4(uv + dir * (0.0/3.0 - 0.5), 0.0, 0.0)).rgb +
        tex2Dlod(sceneTex, float4(uv + dir * (3.0/3.0 - 0.5), 0.0, 0.0)).rgb);

    float lumaB = PerceptualLuma(rgbB);
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

    float wX[6], wY[6]; float sumX = 0.0, sumY = 0.0;
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        float dx = abs(f.x + float(2 - i));
        if (dx < 1e-5) wX[i] = 1.0; else if (dx >= 3.0) wX[i] = 0.0;
        else { float pix = 3.14159265 * dx; wX[i] = 3.0 * sin(pix) * sin(pix / 3.0) / (pix * pix); }
        sumX += wX[i];

        float dy = abs(f.y + float(2 - i));
        if (dy < 1e-5) wY[i] = 1.0; else if (dy >= 3.0) wY[i] = 0.0;
        else { float piy = 3.14159265 * dy; wY[i] = 3.0 * sin(piy) * sin(piy / 3.0) / (piy * piy); }
        sumY += wY[i];
    }
    [unroll] for (int i = 0; i < 6; ++i) { wX[i] /= sumX; wY[i] /= sumY; }

    float3 color = 0.0;
    [unroll]
    for (int y = 0; y < 6; ++y)
    {
        [unroll]
        for (int x = 0; x < 6; ++x)
        {
            float2 tapUV = clamp((tc + float2(float(x) - 2.0, float(y) - 2.0)) * invTexSize, minUV, maxUV);
            color += tex2Dlod(historyTex, float4(tapUV, 0.0, 0.0)).rgb * (wX[x] * wY[y]);
        }
    }
    return max(0.0, color);
}

float3 SampleHistoryCatmullRom5Tap(float2 uv, float2 texSize, float2 invTexSize)
{
    float2 samplePos = uv * texSize;
    float2 tc = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc;

    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w12 + 1e-5);

    float2 tc0 = (tc - 1.0) * invTexSize;
    float2 tc3 = (tc + 2.0) * invTexSize;
    float2 tc12 = (tc + offset12) * invTexSize;

    float3 color = 0.0;
    color += tex2Dlod(historyTex, float4(tc12.x, tc0.y, 0.0, 0.0)).rgb * (w12.x * w0.y);
    color += tex2Dlod(historyTex, float4(tc0.x, tc12.y, 0.0, 0.0)).rgb * (w0.x * w12.y);
    color += tex2Dlod(historyTex, float4(tc12.x, tc12.y, 0.0, 0.0)).rgb * (w12.x * w12.y);
    color += tex2Dlod(historyTex, float4(tc3.x, tc12.y, 0.0, 0.0)).rgb * (w3.x * w12.y);
    color += tex2Dlod(historyTex, float4(tc12.x, tc3.y, 0.0, 0.0)).rgb * (w12.x * w3.y);

    return max(0.0, color);
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct DepthVelocityStats { float closestRawDepth; float minRawDepthSearch; float maxRawDepthSearch; float2 bestVel; float2 minVel; float2 maxVel; };
struct NeighborhoodStats { float3 aabbMin; float3 aabbMax; float3 aabbMin5; float3 aabbMax5; float3 mu; float3 sigma; float3x3 invCov; bool validCovariance; float spatialContrast; float3 expectedColorShift; float weights[9]; };

struct HistoryData {
    bool valid;
    float disocclusion;
    float shadowRisk;
    float clipDistanceRejection;
    float confidence;
    float3 colorSpace;
};

// ============================================================================
// NEIGHBORHOOD STATISTICS
// ============================================================================
NeighborhoodStats ComputeNeighborhoodStats(float3 cachedSpace[9], float2 velocityDir, float pixelMotion, float effectiveMotion, float2 localJitterPx)
{
    NeighborhoodStats stats; stats.validCovariance = false;
    stats.aabbMin  = kLargeValue; stats.aabbMax  = -kLargeValue; stats.aabbMin5 = kLargeValue; stats.aabbMax5 = -kLargeValue;

    float3 m1Std = 0.0, m2Std = 0.0, m1Ctr = 0.0, m2Ctr = 0.0;
    float weightSumStd = 0.0, weightSumCtr = 0.0;
    float motionFactor = saturate(pixelMotion);
    float cachedWeight[9], cachedCenterWeight[9];

    bool useShiftedCenter = (taaJitterAwareVariance > 0.5);
    float2 centerShift = useShiftedCenter ? localJitterPx : 0.0;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 pixelOffset = kOffsets3x3[i]; float3 cSpace = cachedSpace[i];
        stats.aabbMin = min(stats.aabbMin, cSpace); stats.aabbMax = max(stats.aabbMax, cSpace);
        if (i < 5) { stats.aabbMin5 = min(stats.aabbMin5, cSpace); stats.aabbMax5 = max(stats.aabbMax5, cSpace); }

        float stdWeight = exp(-dot(pixelOffset, pixelOffset));
        float centerWeight = exp(-dot(pixelOffset - centerShift, pixelOffset - centerShift) * 0.5);

        if (taaLumaVariance > 0.5) { float lumaMod = 1.0 / (1.0 + cSpace.x); stdWeight *= lumaMod; centerWeight *= lumaMod; }
        if (taaVelocityAlignedVariance > 0.5 && i > 0) { float velMod = lerp(1.0, saturate(dot(normalize(pixelOffset), velocityDir) * 0.5 + 0.5), motionFactor); stdWeight *= velMod; centerWeight *= velMod; }

        cachedWeight[i] = stdWeight; m1Std += cSpace * stdWeight; m2Std += cSpace * cSpace * stdWeight; weightSumStd += stdWeight;
        cachedCenterWeight[i] = centerWeight; m1Ctr += cSpace * centerWeight; m2Ctr += cSpace * cSpace * centerWeight; weightSumCtr += centerWeight;
        
        stats.weights[i] = useShiftedCenter ? centerWeight : stdWeight;
    }

    float3 muEarly = m1Std / weightSumStd;
    float3 sigmaEarly = sqrt(max(m2Std / weightSumStd - muEarly * muEarly, 0.0));

    float3 fireflyMin = muEarly - taaFireflyClamp * sigmaEarly; float3 fireflyMax = muEarly + taaFireflyClamp * sigmaEarly;
    stats.aabbMin  = clamp(stats.aabbMin, fireflyMin, fireflyMax); stats.aabbMax  = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    stats.aabbMin5 = clamp(stats.aabbMin5, fireflyMin, fireflyMax); stats.aabbMax5 = clamp(stats.aabbMax5, fireflyMin, fireflyMax);

    stats.mu = useShiftedCenter ? (m1Ctr / max(weightSumCtr, kEpsilon)) : muEarly;
    stats.sigma = useShiftedCenter ? sqrt(max(m2Ctr / max(weightSumCtr, kEpsilon) - stats.mu * stats.mu, 0.0)) : sigmaEarly;
    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, 0.001);

    float3 gradX = localJitterPx.x > 0.0 ? (cachedSpace[4] - cachedSpace[0]) : (cachedSpace[3] - cachedSpace[0]);
    float3 gradY = localJitterPx.y > 0.0 ? (cachedSpace[2] - cachedSpace[0]) : (cachedSpace[1] - cachedSpace[0]);
    stats.expectedColorShift = (gradX * abs(localJitterPx.x)) + (gradY * abs(localJitterPx.y));

    if (taaUseCovarianceClipping > 0.5)
    {
        float3x3 cov = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        [unroll]
        for (int j = 0; j < 9; ++j)
        {
            float3 d = cachedSpace[j] - stats.mu; float w = useShiftedCenter ? cachedCenterWeight[j] : cachedWeight[j];
            cov[0][0] += d.x * d.x * w; cov[0][1] += d.x * d.y * w; cov[0][2] += d.x * d.z * w;
            cov[1][1] += d.y * d.y * w; cov[1][2] += d.y * d.z * w; cov[2][2] += d.z * d.z * w;
        }

        float cWeightSum = useShiftedCenter ? weightSumCtr : weightSumStd;
        cov[0][0] /= cWeightSum; cov[0][1] /= cWeightSum; cov[0][2] /= cWeightSum;
        cov[1][1] /= cWeightSum; cov[1][2] /= cWeightSum; cov[2][2] /= cWeightSum;

        cov[0][0] += kEpsilon; cov[1][1] += kEpsilon; cov[2][2] += kEpsilon;

        if (taaJitterFlickerPadding > kFlickerPadThreshold)
        {
            float paddingFade = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - effectiveMotion) : 1.0;
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
        stats.invCov = Inverse3x3(cov, stats.validCovariance);
    }

    if (!stats.validCovariance && taaJitterFlickerPadding > kFlickerPadThreshold)
    {
        float paddingFade = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - effectiveMotion) : 1.0;
        float padMult = taaJitterFlickerPadding * paddingFade;
        
        if (taaDirectionalVariance > 0.5) { stats.sigma += abs(stats.expectedColorShift * padMult); }
        else { stats.sigma += (stats.spatialContrast * length(localJitterPx) * padMult); }
    }

    stats.sigma = max(stats.sigma, 0.001);
    return stats;
}

// ============================================================================
// AABB CLIPPING
// ============================================================================
float3 ClipAABB(float3 history, float3 boxMin, float3 boxMax, float softClip, float motionFactor)
{
    float3 center = 0.5 * (boxMax + boxMin);
    float3 extents = max(0.5 * (boxMax - boxMin), kEpsilon);
    float3 offset = history - center;
    float maxUnit = max(abs(offset.x / extents.x), max(abs(offset.y / extents.y), abs(offset.z / extents.z)));

    if (maxUnit > 1.0)
    {
        float softLimit = 1.0 + softClip * (1.0 - exp(-(maxUnit - 1.0)));
        return center + (offset / maxUnit) * lerp(softLimit, 1.0, motionFactor);
    }
    return history;
}

// ============================================================================
// HISTORY CLIPPING
// ============================================================================
float3 ClipHistory(float3 historySpace, NeighborhoodStats stats, float effectiveMotion, float dynamicGamma, float3 cachedSpace[9], float2 localJitterPx)
{
    if (taaUseKDopClipping > 0.5)
    {
        float3 rayCenter = stats.mu; float3 dir = historySpace - rayCenter;
        float nearHit = -kLargeValue; float farHit = kLargeValue;
        
        float padMult = (taaJitterFlickerPadding > kFlickerPadThreshold) ? (taaJitterFlickerPadding * ((taaJitterFlickerFade > 0.5) ? saturate(1.0 - effectiveMotion) : 1.0)) : 0.0;

        [unroll]
        for (int a = 0; a < 16; ++a)
        {
            float3 axis = kDopAxes[a]; float2 extents = float2(kLargeValue, -kLargeValue);
            
            float proj[9];
            [unroll]
            for (int n = 0; n < 9; ++n) { proj[n] = dot(cachedSpace[n], axis); }
            
            float proj_pos = dot(rayCenter, axis);
            float expectedAxisShift = 0.0;
            
            if (padMult > 0.0)
            {
                if (taaDirectionalVariance > 0.5)
                {
                    expectedAxisShift = dot(stats.expectedColorShift * padMult, axis);
                }
                else
                {
                    float crossMin = min(proj[0], min(min(proj[1], proj[2]), min(proj[3], proj[4])));
                    float crossMax = max(proj[0], max(max(proj[1], proj[2]), max(proj[3], proj[4])));
                    float padAmt = (crossMax - crossMin) * length(localJitterPx) * padMult;
                    extents.x -= padAmt;
                    extents.y += padAmt;
                }
            }

            if (taaKDopVariance > 0.5)
            {
                float2 moments = 0.0;
                float wSum = 0.0;
                [unroll]
                for (int n = 0; n < 9; ++n) 
                { 
                    float w = stats.weights[n];
                    moments += float2(proj[n], proj[n] * proj[n]) * w; 
                    wSum += w;
                }
                moments /= wSum;
                
                float mu = moments.x; float sigma = sqrt(max(moments.y - mu * mu, 0.0));
                float expandedSigma = sigma * dynamicGamma;
                
                extents.x = min(mu - expandedSigma, proj_pos);
                extents.y = max(mu + expandedSigma, proj_pos);

                if (taaDirectionalVariance > 0.5 && padMult > 0.0) 
                {
                    extents.x += min(0.0, expectedAxisShift); 
                    extents.y += max(0.0, expectedAxisShift); 
                }
            }
            else
            {
                [unroll]
                for (int n = 0; n < 9; ++n) { extents.x = min(proj[n], extents.x); extents.y = max(proj[n], extents.y); }
                extents.x -= kEpsilon;
                extents.y += kEpsilon;

                if (taaDirectionalVariance > 0.5 && padMult > 0.0) 
                {
                    extents.x += min(0.0, expectedAxisShift);
                    extents.y += max(0.0, expectedAxisShift);
                }
            }

            float dir_dot = dot(dir, axis); float inv_dir = 1.0 / (abs(dir_dot) > 1e-7 ? dir_dot : 1e-7 * sign(dir_dot + 1e-8));
            float t0 = (extents.x - proj_pos) * inv_dir; float t1 = (extents.y - proj_pos) * inv_dir;

            nearHit = max(nearHit, min(t0, t1)); farHit  = min(farHit, max(t0, t1));
        }

        if (nearHit <= farHit && (nearHit > 0.0 || farHit > 0.0))
        {
            float t_hit = clamp(nearHit > 0.0 ? nearHit : farHit, 0.0, 1.0);
            if (t_hit < 1.0)
            {
                float maxUnit = 1.0 / max(t_hit, kEpsilon);
                float softLimit = 1.0 + taaSoftClip * (1.0 - exp(-(maxUnit - 1.0)));
                return rayCenter + dir * (lerp(softLimit, 1.0, saturate(effectiveMotion / max(kMinMotionBlendDropSpeed, 0.1))) / maxUnit);
            }
        }
        return historySpace;
    }

    float motionFactor = saturate(effectiveMotion / max(kMinMotionBlendDropSpeed, 0.1));

    if (stats.validCovariance)
    {
        float3 diff = historySpace - stats.mu; float d2 = dot(diff, mul(stats.invCov, diff)); float gamma2 = dynamicGamma * dynamicGamma;
        float3 clipped = (d2 > gamma2 && d2 > kEpsilon) ? (stats.mu + diff * (dynamicGamma / sqrt(d2))) : historySpace;
        return ClipAABB(clipped, stats.aabbMin, stats.aabbMax, taaSoftClip, motionFactor);
    }

    float3 chromaWeights = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 extents = stats.sigma * dynamicGamma * chromaWeights;

    return ClipAABB(historySpace, max(stats.mu - extents, stats.aabbMin), min(stats.mu + extents, stats.aabbMax), taaSoftClip, motionFactor);
}

// ============================================================================
// DISOCCLUSION
// ============================================================================
float ComputeDisocclusion(DepthVelocityStats dv, float histRawDepth, float2 histVelMem, float2 renderSize)
{
    float depthDisocclusion = 0.0; float velDisocclusion = 0.0;
    if (taaDepthRejection > 0.001)
    {
        float minLin = LinearizeDepth(dv.maxRawDepthSearch); float maxLin = LinearizeDepth(dv.minRawDepthSearch); float histLin = LinearizeDepth(histRawDepth);
        float depthTol = (maxLin - minLin) + (histLin * lerp(taaDepthRejRelStatic, taaDepthRejRelMoving, saturate(taaDepthRejection))) + taaDepthRejAbs;
        depthDisocclusion = saturate(max(0.0, max(minLin - histLin, histLin - maxLin)) / max(depthTol, 1e-4));
    }
    if (taaVelDisocclusion > 0.001)
    {
        float velMag = max(length(dv.bestVel * renderSize), 1.0);
        float velDistToBox = length(max(0.0, max(dv.minVel - histVelMem, histVelMem - dv.maxVel)) * renderSize);
        velDisocclusion = saturate(velDistToBox / max(lerp(taaVelRejBaseStatic, taaVelRejBaseMoving, saturate(taaVelDisocclusion)) + (velMag * taaVelRejMotionScale) + kVelRejToleranceOffset, 1e-5));
    }
    return max(depthDisocclusion, velDisocclusion);
}

// ============================================================================
// GAMUT COMPRESSION
// ============================================================================
float3 CompressGamut(float3 historySpace, float3 currentRGB)
{
    float3 historyRGB = (taaColorSpaceOklab > 0.5) ? OklabToRGB(historySpace) : YCoCgToRGB(historySpace);
    float minChannel = min(historyRGB.r, min(historyRGB.g, historyRGB.b));

    if (minChannel < 0.0)
    {
        float luma = LumaRGB(historyRGB);
        historyRGB = luma + (historyRGB - luma) * (luma / max(luma - minChannel, kEpsilon));
        historySpace = (taaColorSpaceOklab > 0.5) ? RGBToOklab(historyRGB) : RGBToYCoCg(historyRGB);
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

    // 2. Precompute trigonometric functions ONCE per pixel for reprojection
    float sp, cp, sy, cy; sincos(taaJitterPitch, sp, cp); sincos(taaJitterYaw, sy, cy);
    float psp, pcp, psy, pcy; sincos(-taaPrevJitterPitch, psp, pcp); sincos(-taaPrevJitterYaw, psy, pcy);

    // 3. Compute Jitter UVs
    float2 jitterUV = ReprojectUV(IN.uv0, sp, cp, sy, cy);
    float2 jitterPixelPos = jitterUV * passTexSize;
    float2 baseRenderTC = floor(jitterPixelPos) + 0.5;
    float2 snappedRenderUV = clamp(baseRenderTC * passTexel, minRenderUV, maxRenderUV);
    float2 localJitterPx = jitterPixelPos - (clamp(baseRenderTC * passTexel, 0.0, 1.0 - passTexel) * passTexSize);

    // 4. Fetch center pixel samples
    float3 currentColor = max(0.0, tex2Dlod(sceneTex, float4(snappedRenderUV, 0.0, 0.0)).rgb);
    float centerRawDepth = tex2Dlod(depthTex, float4(snappedRenderUV, 0.0, 0.0)).r;
    float3 centerColorSpace = ToSpace(currentColor);
    float2 centerVel = tex2Dlod(velocityTex, float4(snappedRenderUV, 0.0, 0.0)).rg;

    // 5. Compute current pixel motion
    float2 pixelVel = (ReprojectUV(jitterUV + centerVel, psp, pcp, psy, pcy) - IN.uv0) * passTexSize;
    float totalPixelMotion = length(pixelVel);
    float2 velocityDir = (totalPixelMotion > taaMinMotionDir) ? normalize(pixelVel) : float2(1.0, 0.0);

    // 6. Gather 3x3 neighborhood data
    DepthVelocityStats dvStats;
    dvStats.closestRawDepth = centerRawDepth;
    dvStats.bestVel = centerVel;
    dvStats.minVel = centerVel; 
    dvStats.maxVel = centerVel; 
    dvStats.minRawDepthSearch = centerRawDepth; 
    dvStats.maxRawDepthSearch = centerRawDepth;

    float3 cachedSpace[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        if (i == 0) 
        {
            cachedSpace[0] = centerColorSpace;
        } 
        else 
        {
            float2 offsetUV = clamp((baseRenderTC + kOffsets3x3[i]) * passTexel, minRenderUV, maxRenderUV);
            float dRaw = tex2Dlod(depthTex, float4(offsetUV, 0.0, 0.0)).r; 
            float2 v = tex2Dlod(velocityTex, float4(offsetUV, 0.0, 0.0)).rg;
            float3 cSpace = ToSpace(max(0.0, tex2Dlod(sceneTex, float4(offsetUV, 0.0, 0.0)).rgb));

            dvStats.minVel = min(dvStats.minVel, v); 
            dvStats.maxVel = max(dvStats.maxVel, v);
            dvStats.minRawDepthSearch = min(dvStats.minRawDepthSearch, dRaw); 
            dvStats.maxRawDepthSearch = max(dvStats.maxRawDepthSearch, dRaw);

            if (taaUseDepthDilation > 0.5 && dRaw > dvStats.closestRawDepth) { 
                dvStats.closestRawDepth = dRaw; 
                dvStats.bestVel = v; 
            }
            cachedSpace[i] = cSpace;
        }
    }

    // 7. Evaluate history UVs and stored velocity
    float2 historyStableUV = ReprojectUV(jitterUV + dvStats.bestVel, psp, pcp, psy, pcy);
    float2 currentJitterHistUV = clamp(ReprojectUV(historyStableUV, sp, cp, sy, cy), minRenderUV, maxRenderUV);
    
    float2 histVelMem;
    if (taaBilinearHistoryVel > 0.5) {
        histVelMem = tex2Dlod(velocityTex, float4(currentJitterHistUV, 0.0, 0.0)).rg;
    } else {
        float2 snappedHistUV = clamp((floor(currentJitterHistUV * passTexSize) + 0.5) * passTexel, minRenderUV, maxRenderUV);
        histVelMem = tex2Dlod(velocityTex, float4(snappedHistUV, 0.0, 0.0)).rg;
    }

    float effectivePixelMotion = max(totalPixelMotion, length(histVelMem * passTexSize));
    float2 historyPixelCoord = historyStableUV * passTexSize;
    float pixelAlignment = 1.0 - saturate(length(historyPixelCoord - (floor(historyPixelCoord) + 0.5)) * kSqrt2);

    // 8. Compute spatial neighborhood statistics
    NeighborhoodStats colorStats = ComputeNeighborhoodStats(cachedSpace, velocityDir, totalPixelMotion, effectivePixelMotion, localJitterPx);

    // 9. Fetch and evaluate temporal history 
    HistoryData hist;
    hist.valid = all(historyStableUV >= 0.0) && all(historyStableUV <= 1.0);
    hist.disocclusion = 1.0; 
    hist.shadowRisk = 0.0; 
    hist.clipDistanceRejection = 0.0;
    hist.confidence = 0.0;
    hist.colorSpace = centerColorSpace;

    if (hist.valid)
    {
        hist.disocclusion = ComputeDisocclusion(dvStats, tex2Dlod(historyTex, float4((floor(historyPixelCoord) + 0.5) * passTexel, 0.0, 0.0)).a, histVelMem, passTexSize);

        if (taaUseLanczos3 > 0.5) {
            hist.colorSpace = ToSpace(SampleHistoryLanczos3(historyStableUV, passTexSize, passTexel, minRenderUV, maxRenderUV));
        } else {
            hist.colorSpace = ToSpace(SampleHistoryCatmullRom5Tap(historyStableUV, passTexSize, passTexel));
        }

        hist.colorSpace = clamp(hist.colorSpace, colorStats.aabbMin, colorStats.aabbMax);

        float3 histDelta = hist.colorSpace - colorStats.mu;
        float3 normDelta = histDelta / max(colorStats.sigma * max(taaVarianceGamma, 0.001), 0.001);
        hist.confidence = exp(-dot(normDelta, normDelta) * 0.5);

        if (taaShadowMitigation > 0.5) {
            hist.shadowRisk = (1.0 - smoothstep(0.0, max(kShadowThresholdMin, taaShadowDarknessThreshold), centerColorSpace.x)) 
                            * saturate(abs(max(centerColorSpace.x, kShadowLumaFloor) - max(hist.colorSpace.x, kShadowLumaFloor)) * taaShadowTemporalMult) 
                            * (1.0 - saturate(colorStats.spatialContrast * taaShadowSpatialMult));
        }
    }

    // 10. Execute AABB Adaptation
    float3 range9 = colorStats.aabbMax - colorStats.aabbMin;
    float3 range5 = colorStats.aabbMax5 - colorStats.aabbMin5;
    float collapseRatio = max(range5.x / max(range9.x, 1e-4), max(range5.y / max(range9.y, 1e-4), range5.z / max(range9.z, 1e-4)));
    float allowCollapse = smoothstep(taaCollapseRatioMin, taaCollapseRatioMax, collapseRatio);

    float3 adaptiveAABBMin = colorStats.aabbMin5; 
    float3 adaptiveAABBMax = colorStats.aabbMax5;
    
    if (taaRoundedAABB > 0.5) { 
        adaptiveAABBMin = lerp(adaptiveAABBMin, (colorStats.aabbMin + colorStats.aabbMin5) * 0.5, pixelAlignment); 
        adaptiveAABBMax = lerp(adaptiveAABBMax, (colorStats.aabbMax + colorStats.aabbMax5) * 0.5, pixelAlignment); 
    }
    if (taaAdaptiveVariance > 0.5) { 
        float mAF = smoothstep(taaAdaptiveVarStart, taaAdaptiveVarEnd, effectivePixelMotion); 
        adaptiveAABBMin = lerp(adaptiveAABBMin, colorStats.aabbMin5, mAF * allowCollapse); 
        adaptiveAABBMax = lerp(adaptiveAABBMax, colorStats.aabbMax5, mAF * allowCollapse); 
    }
    
    colorStats.aabbMin = adaptiveAABBMin; 
    colorStats.aabbMax = adaptiveAABBMax;

    float dynamicGamma = max(taaVarianceGamma, 0.0);
    if (taaShadowMitigation > 0.5) { 
        dynamicGamma += (hist.shadowRisk * max(taaVarianceGamma, taaShadowVarianceBase)) * (1.0 - hist.disocclusion); 
    }

    // 11. History Clipping and metrics
    float3 unclippedHistorySpace = hist.colorSpace;
    hist.colorSpace = ClipHistory(hist.colorSpace, colorStats, effectivePixelMotion, dynamicGamma, cachedSpace, localJitterPx);

    if (taaClipDistanceRejectionEnabled > 0.5 && hist.valid) { 
        float clipDistance = length(unclippedHistorySpace - hist.colorSpace);
        hist.clipDistanceRejection = saturate((clipDistance - taaClipDistanceRejectionMinError) / max(taaClipDistanceRejectionAmount, 1e-5)); 
    }

    // 12. Gamut compression
    hist.colorSpace = CompressGamut(hist.colorSpace, currentColor);

    // 13. Final Blend factor mapping
    float blend = 0.0;
    if (hist.valid) {
        float motionBlendEnd = max(taaMotionBlendDropSpeed, taaMotionBlendStart + kMinMotionBlendDropSpeed);
        blend = lerp(taaFeedbackMax, taaFeedbackMin, smoothstep(taaMotionBlendStart, motionBlendEnd, effectivePixelMotion));
        
        if (hist.confidence > kFSRConfidenceThreshold) { blend = lerp(blend, blend * taaAlignmentFeedbackDrop, 1.0 - pixelAlignment); }
        if (taaShadowMitigation > 0.5) { blend = lerp(blend, taaShadowBlendStrength, hist.shadowRisk); }
        if (taaClipDistanceRejectionEnabled > 0.5) { blend = lerp(blend, 0.0, hist.clipDistanceRejection); }
        blend = lerp(blend, 0.0, hist.disocclusion);
    }

    // 14. Fallback edge smoothing
    float fxaaBlend = saturate((0.8 - blend) * 2.0);
    if (taaFallbackFXAA > 0.5 && fxaaBlend > 0.0)
    {
        float3 fxaaColor = ApplyFXAA(snappedRenderUV, passTexel, currentColor);
        centerColorSpace = lerp(centerColorSpace, ToSpace(fxaaColor), fxaaBlend);
    }

    // Resolve outputs
    float3 blendedSpace = lerp(centerColorSpace, hist.colorSpace, blend);
    return float4(FromSpace(blendedSpace), centerRawDepth);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}