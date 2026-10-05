// ============================================================================
// TAA history color resampling: the Kaiser suite, plus the fallback FXAA
// ----------------------------------------------------------------------------
// Kaiser-windowed sincs, evaluated EXACTLY. The window p(u) = I0(beta*sqrt(u))
// is computed as its exact series in u,
//   p(u) = sum_k (beta^2/4)^k / (k!)^2 * u^k, via the recurrence
//   c_k = c_{k-1} * (beta^2/4) / k^2. All terms positive: no cancellation,
//   the truncation error is bounded by the first dropped term.
//
// PER-TIER BETA (single-line knobs; the window stays exact at any value):
//   * kKaiserBeta4Tap = 3.2: the 4-tap's wide transition band makes droop,
//     not ripple, the dominant capacity loss.
//   * kKaiserBeta6Tap = 5.2: in-band capacity is saturated at 12-bit
//     precision; only the pi/2 droop and the in-loop bias remain.
//
// The 9-tap kernel is exactly the separable 4x4 logical-tap kernel (the two
// same-signed center taps merge into one weighted fetch per axis). The
// 21-tap is the 6x6 logical-tap kernel with the four corner fetches skipped
// (fetch budget); the invSum renormalization keeps DC exact over the fetched
// support, and the clamp hull covers the FULL fetched set -- the same
// "taps it was built from" rule the 9-tap path uses.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// samplers (historyTex, sceneTex), cbuffer perDraw (taaHistoryOvershoot),
// postFx macros (tex2Dlod). Requires fragments included before:
// taaShared.h.hlsl, taaConstants.h.hlsl, taaColor.h.hlsl (ToSpace).
// ============================================================================
#ifndef TAA_RESAMPLE_H_HLSL
#define TAA_RESAMPLE_H_HLSL

// I0(beta * sqrt(u)) for u in [0, 1], exact series through u^10. Covers the
// 6-tap window: at beta 5.2 the first dropped term is < 1e-5.
float KaiserI0Series(float u, float beta)
{
    float t    = 0.25 * beta * beta;
    float term = 1.0;
    float sum  = 1.0;
    float up   = u;
    [unroll]
    for (int k = 1; k <= 10; ++k)
    {
        term *= t / float(k * k);
        sum  += term * up;
        up   *= u;
    }
    return sum;
}

// float4 lane-parallel variant (the 4-tap's SIMD window). 8 terms: at the
// 4-tap's beta (3.2) the first dropped term is < 4e-8.
float4 KaiserI0Series4(float4 u, float beta)
{
    float  t    = 0.25 * beta * beta;
    float4 term = 1.0;
    float4 sum  = 1.0;
    float4 up   = u;
    [unroll]
    for (int k = 1; k <= 8; ++k)
    {
        term *= t / float(k * k);
        sum  += term * up;
        up   *= u;
    }
    return sum;
}

// Clamp a resampled color to the min/max of the taps it was built from, with a
// small overshoot margin. Shared by both history filters.
float3 ClampToTapFootprint(float3 color, float3 tapMin, float3 tapMax)
{
    float3 tapRange  = max(tapMax - tapMin, kMinFootprintRange);
    float3 overshoot = taaHistoryOvershoot * tapRange;
    return clamp(color, tapMin - overshoot, tapMax + overshoot);
}

// Kaiser-4 fused weights: the separable 4x4 logical-tap kernel collapsed to
// 3 fetches per axis. u_k = 1 - (d_k / 2)^2, d = -(1+f), -f, 1-f, 2-f
// (exact, shared q); sinc(d_k) with the common sin(pi*f)/pi replaced by f --
// exact after the invSum normalization (all g-factors scale together).
void GetKaiser4FusedWeights(float f, float baseCoord, out float3 uv, out float3 w)
{
    float f2 = f * f;
    float q  = 0.25 * f2;
    float4 u = max(float4(0.75 - 0.5 * f, 1.0, 0.75 + 0.5 * f, f) - q, 0.0);

    float4 p = KaiserI0Series4(u, kKaiserBeta4Tap);

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

float3 SampleHistoryColor_Kaiser4_9Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uvX, weightX;
    float3 uvY, weightY;
    GetKaiser4FusedWeights(fracOffset.x, baseTexel.x, uvX, weightX);
    GetKaiser4FusedWeights(fracOffset.y, baseTexel.y, uvY, weightY);

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

// Kaiser-6 fused weights: the separable 6x6 logical-tap kernel collapsed to
// 5 fetches per axis (the one same-signed adjacent pair merges).
void GetKaiser6FusedWeights(float frac, float baseCoord, out float3 posPt, out float2 posBi, out float w[5])
{
    float f  = frac;
    float f2 = f * f;

    // |d_k| of the six logical taps, R = 3 support.
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

    float win0 = KaiserI0Series(u0, kKaiserBeta6Tap);
    float win1 = KaiserI0Series(u1, kKaiserBeta6Tap);
    float win2 = KaiserI0Series(u2, kKaiserBeta6Tap);
    float win3 = KaiserI0Series(u3, kKaiserBeta6Tap);
    float win4 = KaiserI0Series(u4, kKaiserBeta6Tap);
    float win5 = KaiserI0Series(u5, kKaiserBeta6Tap);

    // sinc(d_k), sin(pi*f)/pi replaced by f (exact after normalization).
    // Signs for d = -(2+f), -(1+f), -f, 1-f, 2-f, 3-f:  +, -, +, +, -, +.
    float raw0 =  win0 * (f / d0);
    float raw1 = -win1 * (f / d1);
    float raw2 =  win2;
    float raw3 =  win3 * (f / max(d3, 1e-4));
    float raw4 = -win4 * (f / d4);
    float raw5 =  win5 * (f / d5);

    // The one same-signed adjacent pair: the taps at base+0 / base+1.
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

    // The invSum renormalization absorbs the skipped corner fetches' mass
    // (~0.08% of the kernel); DC stays exact over the fetched support.
    float sumW = w[0] + w[1] + w[2] + w[3] + w[4];
    float invSum = 1.0 / max(sumW, 1e-5);
    [unroll]
    for (int i = 0; i < 5; ++i)
        w[i] *= invSum;
}

float3 SampleHistoryColor_Kaiser6_21Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uX_pt, uY_pt;
    float2 uX_bi, uY_bi;
    float wX[5], wY[5];

    GetKaiser6FusedWeights(fracOffset.x, baseTexel.x, uX_pt, uX_bi, wX);
    GetKaiser6FusedWeights(fracOffset.y, baseTexel.y, uY_pt, uY_bi, wY);

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
            // The four corner fetches are skipped (fetch budget); the sumW
            // renormalization above keeps DC exact.
            if ((x == 0 || x == 4) && (y == 0 || y == 4))
                continue;

            float vx = tcX[x];
            float w  = wX[x] * wy;

            float3 tap = max(tex2Dlod(historyTex, float4(vx, vy, 0.0, 0.0)).rgb, 0.0);
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

float ComputeFXAAFilterWeight(float currentBlendWeight)
{
    float weight = saturate((currentBlendWeight - kFXAABlendKnee) / max(1.0 - kFXAABlendKnee, 0.05));
    return saturate(weight * kFXAABlendSharpness);
}

#endif // TAA_RESAMPLE_H_HLSL