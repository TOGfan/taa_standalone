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

// ============================================================================
// CONSTANTS (Direct from AMD FidelityFX FSR1: ffx_fsr1.h)
// ============================================================================
// #define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))
static const float FSR_RCAS_LIMIT = 0.1875;

// Peak range calculation constant: AF2 peakC = AF2(1.0, -4.0);
static const float2 peakC = float2(1.0, -4.0);

// Fast perceptual space conversions for linear/HDR input (FsrRcasInputF)
float3 LinearToPerceptual(float3 c) {
    return sqrt(max(0.0, c));
}

float3 PerceptualToLinear(float3 c) {
    return c * c;
}

float3 SamplePerceptual(float2 uv) {
    float3 linearCol = max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
    return LinearToPerceptual(linearCol);
}

// ============================================================================
// AMD FIDELITYFX RCAS (FsrRcasF implementation from ffx_fsr1.h)
// ============================================================================
float3 FsrRcasF(float2 uv, float2 texel, float2 minUV, float2 maxUV, float sharpness) {
    if (sharpness <= 0.001) {
        return max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
    }

    // 1. Fetch raw HDR colors (unclamped)
    float3 b = max(0.0, tex2Dlod(taaResultTex, float4(clamp(uv + float2( 0.0, -texel.y), minUV, maxUV), 0.0, 0.0)).rgb);
    float3 d = max(0.0, tex2Dlod(taaResultTex, float4(clamp(uv + float2(-texel.x,  0.0), minUV, maxUV), 0.0, 0.0)).rgb);
    float3 e = max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
    float3 f = max(0.0, tex2Dlod(taaResultTex, float4(clamp(uv + float2( texel.x,  0.0), minUV, maxUV), 0.0, 0.0)).rgb);
    float3 h = max(0.0, tex2Dlod(taaResultTex, float4(clamp(uv + float2( 0.0,  texel.y), minUV, maxUV), 0.0, 0.0)).rgb);

    // 2. Compute local peak luminance to normalize the neighborhood into [0, 1]
    float maxLuma = max(max(max(b.g, d.g), max(e.g, f.g)), h.g);
    float normFactor = 1.0 / max(maxLuma, 1e-4);

    // 3. Normalised perceptual space for lobe calculation (safe in [0, 1])
    float3 bNorm = sqrt(saturate(b * normFactor));
    float3 dNorm = sqrt(saturate(d * normFactor));
    float3 eNorm = sqrt(saturate(e * normFactor));
    float3 fNorm = sqrt(saturate(f * normFactor));
    float3 hNorm = sqrt(saturate(h * normFactor));

    // 4. Min and max of 4-tap ring
    float3 mn4 = min(min(bNorm, dNorm), min(fNorm, hNorm));
    float3 mx4 = max(max(bNorm, dNorm), max(fNorm, hNorm));

    // 5. AMD RCAS peak limit search (now guaranteed safe from division by zero)
    float3 hitMin = mn4 / (4.0 * max(mx4, 1e-4));
    float3 hitMaxDenom = 4.0 * mn4 + peakC.y; // peakC.y is -4.0
    // Prevent division by zero near 1.0
    hitMaxDenom = min(hitMaxDenom, -1e-4);
    float3 hitMax = (peakC.x - mx4) / hitMaxDenom;

    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(0.0, max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b))));

    // 6. AMD Noise mitigation
    float nz = 0.25 * (bNorm.g + dNorm.g + fNorm.g + hNorm.g) - eNorm.g;
    float rangeL = max(mx4.g - mn4.g, 1e-4);
    nz = saturate(abs(nz) / rangeL);
    nz = -0.5 * nz + 1.0;
    
    lobe *= sharpness * nz;

    // 7. Resolve using the ORIGINAL HDR values (retains 100% full HDR range)
    float3 outColor = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);

    return max(0.0, outColor);
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

    float totalSharp = saturate(taaSharpness);
    float3 result = FsrRcasF(IN.uv0, passTexel, minUV, maxUV, totalSharp);

    return float4(result, 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }