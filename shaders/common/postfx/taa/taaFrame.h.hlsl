// ============================================================================
// TAA frame geometry: camera bases, jitter flow, history reprojection
// ----------------------------------------------------------------------------
// The per-frame camera bases from the cbuffer, the exact rotational jitter
// flow, the jitter cancel/transport identities, and the reprojection of the
// resolved surface into the history buffer.
//
// JITTER PLUMBING (all offsets in the shared s_tau sense, see taaShared):
//   * Cancel:    the jitter content of (vCur - vPrev) is s_t - 2 s_{t-1} + s_{t-2}.
//   * Transport: advancing the history surface one frame forward requires
//                subtracting s_t - s_{t-1} + s_{t-2} from vPrev, so the pursuit
//                lands EXACTLY on the surface's CURRENT-FRAME position.
//   * Reprojection base: frameBaseUV must be the CURRENT-FRAME position of
//                the stable point (the INVERSE map). With it as the velocity
//                base, the jitter components of the velocity cancel and a
//                static scene lands at stableUV (the identity, up to the
//                inverse map's ~1e-4 px reflection residual). The ray
//                construction is linear in uv, so base + velocity along the
//                previous camera's screen axes is EXACTLY
//                Ray_prev(frameBaseUV + velocity).
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (camera bases, taaTanHalfFovX/Y). Requires fragments
// included before: taaShared.h.hlsl, taaConstants.h.hlsl.
// ============================================================================
#ifndef TAA_FRAME_H_HLSL
#define TAA_FRAME_H_HLSL

CameraBasis GetCurrentFrameCameraBasis()
{
    CameraBasis c;
    c.rightTanFov = float3(taaCurPX,  taaCurPY,  taaCurPZ);
    c.forward     = float3(taaCurQX,  taaCurQY,  taaCurQZ);
    c.downTanFov  = float3(taaCurRX,  taaCurRY,  taaCurRZ);
    return c;
}

CameraBasis GetPreviousFrameCameraBasis()
{
    CameraBasis c;
    c.rightTanFov = float3(taaPrevPX, taaPrevPY, taaPrevPZ);
    c.forward     = float3(taaPrevQX, taaPrevQY, taaPrevQZ);
    c.downTanFov  = float3(taaPrevRX, taaPrevRY, taaPrevRZ);
    return c;
}

// Exact rotational flow offset of a yaw/pitch rotation at uv. In the shared
// offset sense (s_tau = F_tau^{-1}(g) - g) this is exactly s_{t-2} when fed
// the t-2 jitter rotation -- position-exact. VERIFY with debug mode 12 on a
// static scene: the residual must collapse to noise.
// rotSinCos = (sin(yaw), cos(yaw), sin(pitch), cos(pitch)): the HOST
// pre-evaluates these -- they are per-draw uniforms, and the per-pixel
// sin/cos this function used to do was 4 wasted transcendentals per pixel.
float2 RotationFlowUV(float4 rotSinCos, float2 uv)
{
    float3 d = float3((uv.x * 2.0 - 1.0) * max(taaTanHalfFovX, 1e-4),
                       1.0,
                      (1.0 - uv.y * 2.0) * max(taaTanHalfFovY, 1e-4));

    float sy = rotSinCos.x, cy = rotSinCos.y;
    float3 r = float3(d.x * cy - d.y * sy,
                      d.x * sy + d.y * cy,
                      d.z);
    float sp = rotSinCos.z, cp = rotSinCos.w;
    float3 e = float3(r.x,
                      r.y * cp - r.z * sp,
                      r.y * sp + r.z * cp);

    float invEy = 1.0 / (abs(e.y) > 1e-5 ? e.y : 1e-5);
    float2 outUV;
    outUV.x = 0.5 + 0.5 * (e.x * invEy) / max(taaTanHalfFovX, 1e-4);
    outUV.y = 0.5 - 0.5 * (e.z * invEy) / max(taaTanHalfFovY, 1e-4);
    return outUV - uv;
}

float RayLengthFromUV(float2 uv, float tanHalfFovX, float tanHalfFovY)
{
    float2 tanXY = float2((uv.x * 2.0 - 1.0) * tanHalfFovX, (1.0 - uv.y * 2.0) * tanHalfFovY);
    return sqrt(1.0 + dot(tanXY, tanXY));
}

// Reprojection of the resolved surface into the history buffer.
struct HistoryReprojection
{
    float2 sampleUV;            // where to read the history (previous STABLE output)
    float2 subpixelPx;          // sub-texel offset of the history sample
    float  subpixelAlignment;   // 1.0 at texel centers, 0 at texel corners
    float2 jitterResidualPx;    // current jitter minus history subpixel phase
    float2 motionPx;
    float  motionMagnitudePx;
    float  motionNormalized;    // saturate(motion / kMotionFullStrengthPx)
    float2 motionDirUnit;       // motion direction, or (1,0) when static
};

HistoryReprojection ReprojectToHistory(
    float2 stableUV,
    float2 frameBaseUV,
    float2 currentJitterPx,
    float2 velocityJitteredUV,
    CameraBasis previousCamera,
    ViewportParams vp)
{
    HistoryReprojection h;

    float3 prevRayBase = frameBaseUV.x * previousCamera.rightTanFov
                       + previousCamera.forward
                       - frameBaseUV.y * previousCamera.downTanFov;
    float3 prevRay = prevRayBase
                   + velocityJitteredUV.x * previousCamera.rightTanFov
                   - velocityJitteredUV.y * previousCamera.downTanFov;

    h.sampleUV       = ProjectRayToStableUV(prevRay, frameBaseUV + velocityJitteredUV, taaTanHalfFovX, taaTanHalfFovY);

    // Net motion between the history sample and this output pixel.
    h.motionPx          = (h.sampleUV - stableUV) * vp.sizePixels;
    h.motionMagnitudePx = length(h.motionPx);
    h.motionNormalized  = saturate(h.motionMagnitudePx / kMotionFullStrengthPx);
    h.motionDirUnit     = (h.motionMagnitudePx > kMinMotionDirLengthPx)
                        ? normalize(h.motionPx) : float2(1.0, 0.0);

    // Sub-texel phase of the history sample (0 at texel centers).
    float2 historyPixelPos = h.sampleUV * vp.sizePixels;
    h.subpixelPx        = historyPixelPos - (floor(historyPixelPos) + 0.5);
    h.subpixelAlignment = 1.0 - saturate(length(h.subpixelPx) * kSqrt2);

    // Sub-texel jitter still present after snapping both frames.
    h.jitterResidualPx = currentJitterPx - h.subpixelPx;
    return h;
}

float2 EstimateJitterCancelUV(float2 sCur, float2 sPrev, float2 sPrev2)
{
    return sCur - 2.0 * sPrev + sPrev2;
}

float2 EstimateJitterTransportUV(float2 sCur, float2 sPrev, float2 sPrev2)
{
    return sCur - sPrev + sPrev2;
}

#endif // TAA_FRAME_H_HLSL