#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(taaResultTex, 0);
uniform_sampler2D(velocityTex,  1);
uniform_sampler2D(historyTex,   2);
uniform_sampler2D(depthTex,     3);

cbuffer perDraw {
    float taaSharpness; float taaAdaptiveSharp; float taaDebugMode;
    float taaTanHalfFovX; float taaTanHalfFovY;
    float taaJitterYaw; float taaJitterPitch; float taaPrevJitterYaw; float taaPrevJitterPitch;
    float taaUseDepthDilation; float taaShadowDarknessThreshold; float taaDepthRejection; 
    float taaVelDisocclusion; float taaAlignmentRCASBoost;
    float taaShadowTemporalMult; float taaShadowSpatialMult;

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
// UTILITIES
// ============================================================================

float GetLinearDistance(float rawDepth) { return 1.0 / max(rawDepth, 1e-6); }

float2 ReprojectUV(float2 uv, float yaw, float pitch) {
    float ndc_x = uv.x * 2.0 - 1.0; float ndc_y = 1.0 - uv.y * 2.0; 
    float3 pos = float3(ndc_x * taaTanHalfFovX, 1.0, ndc_y * taaTanHalfFovY);
    float sp, cp, sy, cy; sincos(pitch, sp, cp); sincos(yaw, sy, cy);
    float3 r1; r1.x = pos.x; r1.y = pos.y * cp - pos.z * sp; r1.z = pos.y * sp + pos.z * cp;
    float3 r2; r2.x = r1.x * cy - r1.y * sy; r2.y = r1.x * sy + r1.y * cy; r2.z = r1.z;
    if (r2.y <= 1e-4) return float2(-1.0, -1.0);
    return float2((r2.x / (r2.y * taaTanHalfFovX)) * 0.5 + 0.5, 0.5 - (r2.z / (r2.y * taaTanHalfFovY)) * 0.5);
}

float3 SampleResult(float2 uv) {
    return max(0.0, tex2Dlod(taaResultTex, float4(uv, 0, 0)).rgb);
}

// ============================================================================
// DEBUG VIEWS
// ============================================================================

float4 GetDebugDisocclusionMask(float2 snapped_jitter_tc, float2 texel, float2 texSize, float2 max_uv, float2 jitter_uv_reproj, float exactSnappedDepth, float totalPixelMotion) {
    float closestRawDepth = exactSnappedDepth;
    float2 best_pixel_uv = clamp(snapped_jitter_tc * texel, 0.0, max_uv);
    float2 minVel = 999.0, maxVel = -999.0;
    float minRawDepthSearch = 999.0, maxRawDepthSearch = -999.0;

    [unroll]
    for (int vy = -1; vy <= 1; ++vy) {
        [unroll]
        for (int vx = -1; vx <= 1; ++vx) {
            float2 sample_uv = clamp((snapped_jitter_tc + float2(vx, vy)) * texel, 0.0, max_uv);
            float d_raw = saturate(tex2Dlod(depthTex, float4(sample_uv, 0, 0)).r);
            float2 v = tex2Dlod(velocityTex, float4(sample_uv, 0, 0)).rg;
            minVel = min(minVel, v); maxVel = max(maxVel, v);
            minRawDepthSearch = min(minRawDepthSearch, d_raw); maxRawDepthSearch = max(maxRawDepthSearch, d_raw);
            if (taaUseDepthDilation > 0.5 && d_raw > closestRawDepth) { closestRawDepth = d_raw; best_pixel_uv = sample_uv; }
        }
    }
    
    float2 bestRawVel = tex2Dlod(velocityTex, float4(best_pixel_uv, 0, 0)).rg;
    float2 history_stable_uv = ReprojectUV(jitter_uv_reproj + bestRawVel, -taaPrevJitterYaw, -taaPrevJitterPitch);
    
    float hist_raw_depth = tex2Dlod(historyTex, float4(clamp((floor(history_stable_uv * texSize) + 0.5) * texel, 0.0, max_uv), 0, 0)).a;
    if (hist_raw_depth >= 1024.0) hist_raw_depth = exactSnappedDepth; 
    
    float2 current_jitter_hist_uv = ReprojectUV(history_stable_uv, taaJitterYaw, taaJitterPitch);
    float2 hist_vel = tex2Dlod(velocityTex, float4(clamp((floor(current_jitter_hist_uv * texSize) + 0.5) * texel, 0.0, max_uv), 0, 0)).rg;

    float depthDisocclusion = 0.0, velDisocclusionFactor = 0.0;
    if (taaDepthRejection > 0.001) {
        float histLin = GetLinearDistance(hist_raw_depth); float minLin = GetLinearDistance(maxRawDepthSearch); float maxLin = GetLinearDistance(minRawDepthSearch); 
        float depthTolerance = (maxLin - minLin) + (histLin * lerp(0.1, 0.02, saturate(taaDepthRejection))) + 0.01;
        depthDisocclusion = saturate(max(0.0, max(minLin - histLin, histLin - maxLin)) / max(depthTolerance, 1e-4));
    }
    if (taaVelDisocclusion > 0.001) {
        float velDistToBox = length(max(0.0, max(minVel - hist_vel, hist_vel - maxVel)) * texSize);
        velDisocclusionFactor = saturate(velDistToBox / max(lerp(0.5, 0.05, saturate(taaVelDisocclusion)) + (totalPixelMotion * 0.5) + 0.01, 1e-5));
    }
    return float4(velDisocclusionFactor > 0.5 ? 1.0 : 0.0, depthDisocclusion > 0.5 ? 1.0 : 0.0, 0.0, 1.0);
}

float4 GetDebugView(int mode, float2 pixel_uv, float2 texel, float2 texSize, float2 max_uv, float3 centerColor, float2 stable_motion, float exactSnappedDepth, float2 snapped_jitter_tc, float2 jitter_uv_reproj) {
    if (mode == 1) return float4(saturate(abs(stable_motion * 50.0)), 0, 1);
    if (mode == 2) return float4(max(0.0, tex2Dlod(historyTex, float4(clamp(pixel_uv, 0.0, max_uv), 0, 0)).rgb), 1);
    if (mode == 3) return GetDebugDisocclusionMask(snapped_jitter_tc, texel, texSize, max_uv, jitter_uv_reproj, exactSnappedDepth, length(stable_motion * texSize));
    if (mode == 4) return float4(saturate(GetLinearDistance(exactSnappedDepth) / 100.0).xxx, 1.0);
    if (mode == 5) { 
        float temporalContrast = abs(dot(centerColor, float3(0.2126, 0.7152, 0.0722)) - dot(max(0.0, tex2Dlod(historyTex, float4(clamp(pixel_uv, 0.0, max_uv), 0, 0)).rgb), float3(0.2126, 0.7152, 0.0722)));
        float3 minRGB = min(min(min(centerColor, SampleResult(clamp(pixel_uv + float2(0, -texel.y), 0.0, max_uv))), SampleResult(clamp(pixel_uv + float2(0, texel.y), 0.0, max_uv))), min(SampleResult(clamp(pixel_uv + float2(-texel.x, 0), 0.0, max_uv)), SampleResult(clamp(pixel_uv + float2(texel.x, 0), 0.0, max_uv))));
        float3 maxRGB = max(max(max(centerColor, SampleResult(clamp(pixel_uv + float2(0, -texel.y), 0.0, max_uv))), SampleResult(clamp(pixel_uv + float2(0, texel.y), 0.0, max_uv))), max(SampleResult(clamp(pixel_uv + float2(-texel.x, 0), 0.0, max_uv)), SampleResult(clamp(pixel_uv + float2(texel.x, 0), 0.0, max_uv))));
        float shadowProxy = 1.0 - smoothstep(0.0, max(0.001, taaShadowDarknessThreshold), dot(centerColor, float3(0.2126, 0.7152, 0.0722)));
        return float4(shadowProxy * saturate(temporalContrast * taaShadowTemporalMult) * (1.0 - saturate((maxRGB.g - minRGB.g) * taaShadowSpatialMult)), 0.0, shadowProxy * 0.35, 1.0);
    }
    
    return float4(0,0,0,1);
}

// ============================================================================
// FIDELITYFX CAS
// ============================================================================

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
    float2 stable_uv = IN.uv0;
    float2 texel     = oneOverTargetSize;
    float2 texSize   = 1.0 / texel;
    float2 max_uv    = 1.0 - texel;
    
    float2 center_tc = floor(stable_uv * texSize) + 0.5;
    float2 pixel_uv  = center_tc * texel;

    float3 center = SampleResult(pixel_uv);
    
    float2 jitter_uv_reproj = ReprojectUV(stable_uv, taaJitterYaw, taaJitterPitch);
    float2 snapped_jitter_tc = floor(jitter_uv_reproj * texSize) + 0.5;
    float2 rawCenterVel = tex2Dlod(velocityTex, float4(clamp(snapped_jitter_tc * texel, 0.0, max_uv), 0, 0)).rg;
    
    float2 prev_center_stable = ReprojectUV(jitter_uv_reproj + rawCenterVel, -taaPrevJitterYaw, -taaPrevJitterPitch);
    float2 center_stable_motion = prev_center_stable - stable_uv;
    
    // Debug Mode Overlay
    if (taaDebugMode > 0.5 && taaDebugMode < 5.5) {
        float exactDepth = saturate(tex2Dlod(depthTex, float4(clamp(snapped_jitter_tc * texel, 0.0, max_uv), 0, 0)).r);
        return GetDebugView((int)floor(taaDebugMode + 0.5), pixel_uv, texel, texSize, max_uv, center, center_stable_motion, exactDepth, snapped_jitter_tc, jitter_uv_reproj);
    }
    
    if (taaDebugMode > 5.5 && taaDebugMode < 7.5) {
        return float4(center, 1.0);
    }
    
    float motionFactor = saturate(length(center_stable_motion * texSize) * 2.0);
    
    float2 historyPixelCoord = prev_center_stable * texSize;
    float subPixelAlignment = 1.0 - saturate(length(historyPixelCoord - (floor(historyPixelCoord) + 0.5)) / 0.7071);
    float sharpnessBoost = lerp(taaAlignmentRCASBoost, 0.0, subPixelAlignment) * motionFactor; 
    
    float totalSharp = saturate(taaSharpness + (taaAdaptiveSharp * motionFactor) + sharpnessBoost);

    return float4(max(0.0, ApplyRCAS(center, pixel_uv, texel, max_uv, totalSharp)), 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }