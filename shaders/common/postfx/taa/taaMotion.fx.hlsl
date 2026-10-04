// ============================================================================
// TAA motion-field writer (#TAA_HistMotion)
// ----------------------------------------------------------------------------
// Writes, per stable texel, the layer-resolved (dilated / bilinear) effective
// velocity (xy), effective depth (z) and layer flag (w) of the CURRENT frame:
//     flag 0.0 = background / continuous surface
//     flag 1.0 = foreground edge (silhouette crest texel)
//     flag 2.0 = dilation zone (owned by the foreground crest)
// These are exactly the values the resolve pass used to reproject this texel,
// so next frame's resolve reads the authoritative layer data at the history
// landing instead of re-deriving it from raw history depth/velocity.
//
// FIX -- POST-VALIDATION STATE: the field now stores the state AFTER the
// resolve's own-history dilation gate. A dilation candidate the resolve
// REVOKED is detected through the SIGN of #TAA_Result.a at this texel
// (negative = revoked -- every resolve return path transports the bit,
// including debug views) and is stored as BACKGROUND here, so the revocation
// PERSISTS: the next frame's gate re-reads background at that texel and
// stays revoked, instead of re-entering the dilation cycle from raw depth
// every frame. The bit is ignored for non-candidates, so the conservative
// "revoked" encodings of early/debug resolve returns are always safe.
//
// FIX -- PHASE-DEPENDENT EDGE VALUES: foreground-edge effective values follow
// the sub-texel phase rule in taaShared.h.hlsl (bilinear toward the flat
// part, the pixel's own center samples toward the silhouette). This pass
// snaps the SAME inverse-map position as the resolve and calls the SAME
// classification with the SAME fracPx, so it stores exactly what the resolve
// reprojected with -- the stored field cannot disagree.
//
// Runs as a child AFTER the resolve (it must not overwrite the field before
// the resolve has read the previous frame's, and it needs the resolve's
// output for the revocation bit). Shares its classification code with
// taa.fx.hlsl via taaShared.h.hlsl -- identical inputs, identical logic:
// both passes snap the SAME forward-map position, so the classifications are
// guaranteed identical (the resolve only uses the INVERSE map as its
// reprojection base -- never for sampling).
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"

uniform_sampler2D(depthTex,    0);
uniform_sampler2D(velocityTex, 1);
uniform_sampler2D(resultTex,   2); // FIX: #TAA_Result -- the resolve's output
                                   // from THIS frame; sign of .a carries the
                                   // revocation bit (negative = this texel's
                                   // tentative dilation was revoked).

cbuffer perDraw
{
    float taaUseDepthDilation;  float taaDepthRejection;
    float taaCurPX;             float taaCurPY;
    float taaCurPZ;             float taaCurQX;
    float taaCurQY;             float taaCurQZ;
    float taaCurRX;             float taaCurRY;
    float taaCurRZ;             float taaTanHalfFovX;
    float taaTanHalfFovY;       float taaPad0;

    float2 oneOverTargetSize;
    POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#ifdef SHADER_STAGE_VS
  #define mainV main
#else
  #define mainP main
#endif

float EncodeLayerFlag(LayerSurface layer)
{
    if (layer.isDilationZone)   return 2.0;
    if (layer.isForegroundEdge) return 1.0;
    return 0.0;
}

float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    ViewportParams vp = GetViewportParams(oneOverTargetSize);

    CameraBasis currentCamera;
    currentCamera.rightTanFov = float3(taaCurPX, taaCurPY, taaCurPZ);
    currentCamera.forward     = float3(taaCurQX, taaCurQY, taaCurQZ);
    currentCamera.downTanFov  = float3(taaCurRX, taaCurRY, taaCurRZ);

    // Inverse map (stable-UV -> frame-UV), snapped -- the SAME value the
    // resolve snaps, so this pass classifies the identical texel with the
    // identical fracPx. (Snapping the forward map breaks wherever the
    // rotational jitter exceeds half a texel -- the screen perimeter.)
    float2 stableInFrameUV = InverseReprojectThroughCamera(IN.uv0, currentCamera, taaTanHalfFovX, taaTanHalfFovY);
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float2 tapUVs[9];
    Build3x3TapUVs(pixel.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, tapUVs);

    float  depths[9];
    float2 velocities[9];
    depths[0]     = tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    velocities[0] = tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        depths[i]     = tex2Dlod(depthTex,    float4(tapUVs[i], 0.0, 0.0)).r;
        velocities[i] = tex2Dlod(velocityTex, float4(tapUVs[i], 0.0, 0.0)).rg;
    }

    LayerSurface layer = ClassifyLayerSurface(
        depths, velocities,
        pixel.fracPx, vp.sizePixels,
        (taaUseDepthDilation > 0.5), taaDepthRejection,
        false); // spread is measured at the landing, never stored

    // FIX: consume the revocation bit and store the POST-VALIDATION state.
    // A revoked candidate is stored as background (flag 0, subpixel bg
    // depth, bilerp bg velocity -- the exact values the resolve re-projected
    // with after revoking, via the shared RevokeDilation helper).
    bool revoked = (layer.isDilationZone &&
                    (tex2Dlod(resultTex, float4(IN.uv0, 0.0, 0.0)).a < 0.0));
    if (revoked)
        RevokeDilation(layer, depths[0], velocities, pixel.fracPx);

    return float4(layer.effectiveVelocityUV, layer.effectiveDepth, EncodeLayerFlag(layer));
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }