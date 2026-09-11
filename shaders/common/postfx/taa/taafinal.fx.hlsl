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
// FIDELITYFX-STYLE RCAS (bug-fixed, HDR-safe)
// ============================================================================
// Changes vs the previous version:
//  1. maxRGB used min(f, h) -- the limiter was wrong whenever the local max
//     was the right or down tap.
//  2. The LDR limiter term (1 - maxRGB) zeroed the lobe for any channel above
//     1.0. This pass runs on linear HDR values, so sky/highlights got NO
//     sharpening. Replaced with a range-based limiter.
//  3. Added a final clamp to the neighborhood min/max: the filtered result
//     can no longer overshoot (ringing) at higher strengths.
//  4. Lobe capped at -0.2 (max ~4x edge gain) instead of -0.249 (gain
//     explodes toward sharpness = 1.0). At the default 0.25 nothing changes.
// ============================================================================
float3 SampleResult(float2 uv) {
    return max(0.0, tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0)).rgb);
}

float3 ApplyRCAS(float3 center, float2 pixel_uv, float2 texel, float2 min_uv, float2 max_uv, float totalSharp) {
    if (totalSharp <= 0.001) return center;

    float3 b = SampleResult(clamp(pixel_uv + float2( 0.0, -texel.y), min_uv, max_uv));
    float3 d = SampleResult(clamp(pixel_uv + float2(-texel.x,  0.0), min_uv, max_uv));
    float3 e = center;
    float3 f = SampleResult(clamp(pixel_uv + float2( texel.x,  0.0), min_uv, max_uv));
    float3 h = SampleResult(clamp(pixel_uv + float2( 0.0,  texel.y), min_uv, max_uv));

    // 5-tap min/max (FIX: max was max(max(b,d), max(e, min(f,h))))
    float3 minRGB = min(min(b, d), min(e, min(f, h)));
    float3 maxRGB = max(max(b, d), max(e, max(f, h)));

    // Negative lobe, capped at -0.2 (max ~4x edge gain)
    float w = max(-0.2, -0.25 * totalSharp);

    // Soft limiter (HDR-safe): preserves shadow detail / avoids amplifying
    // near-black; the old '1 - maxRGB' highlight gate is gone.
    float3 level = max(maxRGB, 1e-4);
    float  limit = min(minRGB.r / level.r, min(minRGB.g / level.g, minRGB.b / level.b));
    w = max(w, -0.25 * limit * totalSharp);

    float3 outColor = (b * w + d * w + e + f * w + h * w) / max(1.0 + 4.0 * w, 1e-5);

    // Anti-ringing: never leave the neighborhood range
    return clamp(outColor, minRGB, maxRGB);
}

// ============================================================================
// MAIN PASS
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    float2 passTexel = oneOverTargetSize;
    float2 minUV     = 0.5 * passTexel;
    float2 maxUV     = 1.0 - minUV;

    float3 center = SampleResult(IN.uv0);

    if (taaDebugMode > 0.5) {
        return float4(center, 1.0);
    }

    float totalSharp = saturate(taaSharpness);
    return float4(ApplyRCAS(center, IN.uv0, passTexel, minUV, maxUV, totalSharp), 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }