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

    // 1. Fetch 5-tap neighborhood:
    //      b
    //   d  e  f
    //      h
    float3 b = SamplePerceptual(clamp(uv + float2( 0.0, -texel.y), minUV, maxUV));
    float3 d = SamplePerceptual(clamp(uv + float2(-texel.x,  0.0), minUV, maxUV));
    float3 e = SamplePerceptual(uv);
    float3 f = SamplePerceptual(clamp(uv + float2( texel.x,  0.0), minUV, maxUV));
    float3 h = SamplePerceptual(clamp(uv + float2( 0.0,  texel.y), minUV, maxUV));

    // 2. Min and max of 4-tap ring (excluding center e)
    // AF3 mn4 = min(AMin3F3(b.rgb, d.rgb, f.rgb), h.rgb);
    // AF3 mx4 = max(AMax3F3(b.rgb, d.rgb, f.rgb), h.rgb);
    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));

    // 3. Exact AMD peak limit search
    // AF3 hitMin = mn4 / (4.0 * mx4);
    // AF3 hitMax = (peakC.x - mx4) / (4.0 * mn4 + peakC.y);
    float3 hitMin = mn4 / (4.0 * max(mx4, 1e-4));
    float3 hitMax = (peakC.x - mx4) / (4.0 * mn4 + peakC.y);
    float3 lobeRGB = max(-hitMin, hitMax);

    // AF1 lobe = max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * con;
    float lobe = max(-FSR_RCAS_LIMIT, min(0.0, max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b))));

    // 4. AMD Noise mitigation (FSR_RCAS_DENOISE) using green channel / approximate luma
    float nz = 0.25 * (b.g + d.g + f.g + h.g) - e.g;
    float rangeL = max(mx4.g - mn4.g, 1e-4);
    nz = saturate(abs(nz) / rangeL);
    nz = -0.5 * nz + 1.0;
    
    // Apply user sharpness scaling and noise attenuation factor
    lobe *= sharpness * nz;

    // 5. Official AMD RCAS Resolve
    // pix = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
    float3 outColor = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);

    // Transform back to linear color (NO final bounding clamp, exact to AMD SDK)
    return max(0.0, PerceptualToLinear(outColor));
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