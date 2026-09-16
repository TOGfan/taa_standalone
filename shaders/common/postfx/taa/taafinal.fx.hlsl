#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(taaResultTex, 0);

cbuffer perDraw {
    float taaSharpness;
    float taaDebugMode;

    float2 oneOverTargetSize;
    POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#ifdef SHADER_STAGE_VS
#define mainV main
#else
#define mainP main
#endif

// Fast perceptual curve for HDR-safe spatial operations
float3 LinearToPerceptual(float3 c) {
    return sqrt(max(0.0, c));
}

float3 PerceptualToLinear(float3 c) {
    return c * c;
}

float3 SampleResultPerceptual(float2 uv) {
    float3 linearCol = max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
    return LinearToPerceptual(linearCol);
}

float3 ApplyRCAS(float3 e, float2 pixel_uv, float2 texel, float2 min_uv, float2 max_uv, float sharpness) {
    if (sharpness <= 0.001) return PerceptualToLinear(e);

    // 5-tap neighborhood in perceptual space
    //      b
    //   d  e  f
    //      h
    float3 b = SampleResultPerceptual(clamp(pixel_uv + float2( 0.0, -texel.y), min_uv, max_uv));
    float3 d = SampleResultPerceptual(clamp(pixel_uv + float2(-texel.x,  0.0), min_uv, max_uv));
    float3 f = SampleResultPerceptual(clamp(pixel_uv + float2( texel.x,  0.0), min_uv, max_uv));
    float3 h = SampleResultPerceptual(clamp(pixel_uv + float2( 0.0,  texel.y), min_uv, max_uv));

    // 1. Full 5-tap bounding box
    float3 minRGB = min(min(b, d), min(e, min(f, h)));
    float3 maxRGB = max(max(b, d), max(e, max(f, h)));

    // 2. FidelityFX RCAS Luma Noise Protection (prevents sharpening single-pixel noise/TAA jitter)
    float bL = b.g + 0.5 * (b.r + b.b);
    float dL = d.g + 0.5 * (d.r + d.b);
    float eL = e.g + 0.5 * (e.r + e.b);
    float fL = f.g + 0.5 * (f.r + f.b);
    float hL = h.g + 0.5 * (h.r + h.b);

    float minL = min(min(bL, dL), min(eL, min(fL, hL)));
    float maxL = max(max(bL, dL), max(eL, max(fL, hL)));

    float nz = 0.25 * (bL + dL + fL + hL) - eL;
    float rangeL = maxL - minL;
    nz = saturate(abs(nz) / max(rangeL, 1e-4));
    nz = 1.0 - 0.5 * nz; // Denoise attenuation factor

    // 3. Contrast-adaptive negative lobe calculation (FidelityFX RCAS style)
    // Limits the filter lobe to prevent clipping against local min/max
    float3 hitMin = minRGB / (4.0 * max(maxRGB, 1e-4));
    float3 hitMax = (1.0 - maxRGB) / (4.0 * minRGB + 1e-4);
    float3 lobeRGB = max(-hitMin, hitMax);
    
    // Select peak weight across channels
    float lobe = min(0.0, max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)));
    
    // Scale by user sharpness and noise attenuation (limit negative lobe to -0.25)
    float w = max(-0.20, lobe) * sharpness * nz;

    // 4. Resolve filter
    float3 outColor = (w * (b + d + f + h) + e) / (1.0 + 4.0 * w);

    // 5. Anti-ringing clamp
    outColor = clamp(outColor, minRGB, maxRGB);

    return PerceptualToLinear(outColor);
}

// ============================================================================
// MAIN PASS
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    float2 passTexel = oneOverTargetSize;
    float2 minUV     = 0.5 * passTexel;
    float2 maxUV     = 1.0 - minUV;

    if (taaDebugMode > 0.5) {
        return float4(max(0.0, tex2Dlod(taaResultTex, float4(IN.uv0, 0.0, 0.0)).rgb), 1.0);
    }

    float3 center = SampleResultPerceptual(IN.uv0);
    float totalSharp = saturate(taaSharpness);

    float3 result = ApplyRCAS(center, IN.uv0, passTexel, minUV, maxUV, totalSharp);
    return float4(result, 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }