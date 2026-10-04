// ============================================================================
// TAA motion-field writer (#TAA_HistMotion)
// ----------------------------------------------------------------------------
// Writes, per STABLE texel, the CURRENT frame's RAW layer-classification
// inputs, re-indexed into stable texel space (via the inverse-map snap):
//     xy = raw jittered velocity of the content at that stable texel
//     z  = raw rendered depth of that content
//     w  = layer flag (0 background/flat, 1 foreground edge,
//          2 dilation zone) from the SHARED old classifier
// EXCEPT where the resolve's post-validation state must persist:
//     * a KEPT dilation candidate stores the CREST's raw sample (velocity AND
//       depth of the closest tap, flag 2): the anchor the next frame's
//       own-history gate validates through, and the baked foreground the
//       history landing resolves to inside the object's dilation band.
//     * a REVOKED candidate is reverted to its OWN raw background sample
//       (flag 0) -- exactly the values the resolve re-projected with after
//       revoking (RevokeDilation's background state), so the revocation
//       PERSISTS: the next frame's gate re-reads background at that texel and
//       stays revoked instead of re-entering the dilation cycle every frame.
//
// The field is therefore raw everywhere except dilation bands (crest
// re-attribution) and revoked texels (background reversion) -- clean input for
// the resolve's landing-side re-classification with the old depth-curvature
// rules, with the layer/revocation semantics baked in.
//
// Runs as a child AFTER the resolve: it must not overwrite the field before
// the resolve has read the previous frame's, and it needs the resolve's
// output for the revocation bit (#TAA_Result.a < 0 = this texel's tentative
// dilation was revoked; the writer ignores the bit for non-candidates, so the
// conservative "revoked" encodings of the resolve's early/debug returns are
// always safe).
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"

uniform_sampler2D(depthTex,    0);
uniform_sampler2D(velocityTex, 1);
uniform_sampler2D(resultTex,   2); // #TAA_Result -- the resolve's output from
                                   // THIS frame; sign of .a = revocation bit.

cbuffer perDraw
{
    float taaUseDepthDilation;  float taaDepthRejection;  // slot kept; unused
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

float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    ViewportParams vp = GetViewportParams(oneOverTargetSize);

    CameraBasis currentCamera;
    currentCamera.rightTanFov = float3(taaCurPX, taaCurPY, taaCurPZ);
    currentCamera.forward     = float3(taaCurQX, taaCurQY, taaCurQZ);
    currentCamera.downTanFov  = float3(taaCurRX, taaCurRY, taaCurRZ);

    // Inverse map (stable-UV -> frame-UV), snapped -- the SAME value the
    // resolve snaps, so this pass classifies the identical texel and picks the
    // identical crest tap. (Snapping the forward map breaks wherever the
    // rotational jitter exceeds half a texel -- the screen perimeter.)
    float2 stableInFrameUV = InverseReprojectThroughCamera(IN.uv0, currentCamera, taaTanHalfFovX, taaTanHalfFovY);
    SnappedCoord pixel     = SnapUVToTexel(stableInFrameUV, vp);

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

    // THE classifier (old depth-curvature rules) -- identical to the resolve's.
    SurfaceEdgeState edge = AnalyzeSurfaceEdgesCore(depths, (taaUseDepthDilation > 0.5));

    // Closest (foreground) tap -- identical scan to the resolve's.
    float closestDepth = depths[0];
    int   closestIdx   = 0;
    [unroll]
    for (int j = 1; j < 9; ++j)
    {
        if (depths[j] > closestDepth) { closestDepth = depths[j]; closestIdx = j; }
    }

    // Consume the revocation bit and store the POST-VALIDATION state.
    bool revoked = edge.isDilationZone &&
                   (tex2Dlod(resultTex, float4(IN.uv0, 0.0, 0.0)).a < 0.0);

    if (revoked)
    {
        // RevokeDilation's background state: the texel's OWN raw sample
        // (the center always was the background layer; the dilation only
        // borrowed the crest's values). The resolve re-projected with exactly
        // these values after revoking.
        return float4(velocities[0], depths[0], 0.0);
    }
    if (edge.isDilationZone)
    {
        // Kept candidate: the crest's raw sample -- the anchor.
        return float4(velocities[closestIdx], depths[closestIdx], 2.0);
    }

    // Flat / foreground edge: the raw center sample (the landing re-classifies
    // and re-resolves with its own phase; no pre-resolution needed here).
    return float4(velocities[0], depths[0], edge.isForegroundEdge ? 1.0 : 0.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }