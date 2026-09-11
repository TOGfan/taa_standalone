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
// FIDELITYFX RCAS
// ============================================================================
float3 SampleResult(float2 uv) {
    return max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
}

float3 ApplyRCAS(float3 center, float2 pixel_uv, float2 texel, float2 max_uv, float totalSharp) {
    if (totalSharp <= 0.001) return center;
    
    float3 b = SampleResult(clamp(pixel_uv + float2( 0, -texel.y), 0.0, max_uv));
    float3 d = SampleResult(clamp(pixel_uv + float2(-texel.x,  0), 0.0, max_uv));
    float3 e = center;
    float3 f = SampleResult(clamp(pixel_uv + float2( texel.x,  0), 0.0, max_uv));
    float3 h = SampleResult(clamp(pixel_uv + float2( 0,  texel.y), 0.0, max_uv));
    
    float3 minRGB = min(min(b, d), min(e, min(f, h)));
    float3 maxRGB = max(max(b, d), max(e, max(f, h)));
    
    float w = max(-0.249, -0.25 * totalSharp);
    float3 limit = min(minRGB, max(0.0, 1.0 - maxRGB)) / max(maxRGB, 1e-5);
    w = max(w, -0.25 * min(limit.r, min(limit.g, limit.b)) * totalSharp);
    
    return (b * w + d * w + e + f * w + h * w) / max(1.0 + 4.0 * w, 1e-5);
}

// ============================================================================
// MAIN PASS
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    float2 passTexel = oneOverTargetSize;
    float2 maxUV     = 1.0 - passTexel;
    
    float3 center = SampleResult(IN.uv0);
    
    if (taaDebugMode > 0.5) {
        return float4(center, 1.0);
    }
    
    float totalSharp = saturate(taaSharpness);
    return float4(ApplyRCAS(center, IN.uv0, passTexel, maxUV, totalSharp), 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }