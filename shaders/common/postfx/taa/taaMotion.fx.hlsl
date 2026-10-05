// ============================================================================
// TAA motion-field writer (#TAA_HistMotion)
// ----------------------------------------------------------------------------
// Writes, per STABLE texel, the CURRENT frame's RAW layer-classification
// inputs, re-indexed into stable texel space (via the inverse-map snap):
//     xy = raw jittered velocity of the content at that stable texel
//     z  = raw rendered depth of that content
//     w  = layer flag (0 background/flat, 1 foreground edge, 2 dilation zone)
//          from the SHARED depth-curvature classifier
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
// the resolve's landing-side re-classification with the shared depth-curvature
// rules, with the layer/revocation semantics baked in.
//
// SCHEDULING: a standalone ROOT pass at PFXAfterBin, not a child of the
// resolve. This pass needs three sampled textures, and child passes only
// reliably bind two texture slots on the Vulkan backend (slots 0/1 bind; the
// third fails its descriptor update every frame). As a root it binds like the
// resolve does. PFXAfterBin guarantees, without priority guesswork:
//   * the whole resolve chain (PFXBeforeBin, children included) has finished,
//     so #TAA_Result holds THIS frame's output and the revocation bit in its
//     sign-encoded alpha is fresh;
//   * this pass overwrites #TAA_HistMotion strictly AFTER this frame's
//     resolve read the previous contents, and strictly before the next
//     frame's resolve reads the new one;
//   * #prepass[Depth] and #velocitybuffer are persistent frame resources,
//     valid at any phase.
// The writer ignores the revocation bit for non-candidates, so the
// conservative "revoked" encodings of the resolve's early/debug returns are
// always safe. The pass is enabled/disabled together with the resolve chain
// AND the useMotionField setting.
//
// Fetch budget: the classifier needs all nine depths, but the velocities are
// fetched lazily -- the center tap always, the crest tap only inside dilation
// zones -- and the resolve output is fetched only where a revocation could be
// recorded.
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"

uniform_sampler2D(depthTex,         0); // #prepass[Depth]
uniform_sampler2D(velocityTex,      1); // #velocitybuffer
// #TAA_History: the history-copy child's bit-exact copy of the resolve
// output (alpha sign included). The resolve's own #TAA_Result target is only
// a live resource inside the resolve's pass chain, so this root pass reads
// the persistent copy instead -- the same binding the resolve itself uses
// for #TAA_History at its slot 2.
uniform_sampler2D(resolveOutputTex, 2);

// ============================================================================
// CBUFFER (all constants are set BY NAME from client/postFx/taa.lua)
// ============================================================================
cbuffer perDraw
{
    float taaUseDepthDilation;  float taaTanHalfFovX;
    float taaTanHalfFovY;       float taaCurPX;
    float taaCurPY;             float taaCurPZ;
    float taaCurQX;             float taaCurQY;
    float taaCurQZ;             float taaCurRX;
    float taaCurRY;             float taaCurRZ;

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

    // The classifier needs the full depth 3x3.
    float depths[9];
    depths[0] = tex2Dlod(depthTex, float4(pixel.snappedUV, 0.0, 0.0)).r;
    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        depths[i] = tex2Dlod(depthTex, float4(tapUVs[i], 0.0, 0.0)).r;
    }

    // THE classifier (depth-curvature rules) -- identical to the resolve's.
    SurfaceEdgeState edge = AnalyzeSurfaceEdgesCore(depths, (taaUseDepthDilation > 0.5));

    // The center velocity is always stored.
    float2 centerVelocity = tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    if (edge.isDilationZone)
    {
        // Closest (foreground) tap -- identical scan to the resolve's. The
        // scan tracks the tap's UV directly so every array access stays
        // statically indexed (a dynamic index would force the tapUVs array
        // onto the indexable-temp path, which some drivers map to private
        // memory or spill).
        float  closestDepth = depths[0];
        float2 closestUV    = pixel.snappedUV;
        [unroll]
        for (int j = 1; j < 9; ++j)
        {
            if (depths[j] > closestDepth)
            {
                closestDepth = depths[j];
                closestUV    = tapUVs[j];
            }
        }

        // Consume the revocation bit (negative alpha = this texel's tentative
        // dilation was revoked) and store the POST-VALIDATION state.
        bool revoked = tex2Dlod(resolveOutputTex, float4(IN.uv0, 0.0, 0.0)).a < 0.0;

        if (revoked)
        {
            // RevokeDilation's background state: the texel's OWN raw sample
            // (the center always was the background layer; the dilation only
            // borrowed the crest's values). The resolve re-projected with
            // exactly these values after revoking.
            return float4(centerVelocity, depths[0], 0.0);
        }

        // Kept candidate: the crest's raw sample -- the anchor.
        float2 crestVelocity = tex2Dlod(velocityTex, float4(closestUV, 0.0, 0.0)).rg;
        return float4(crestVelocity, closestDepth, 2.0);
    }

    // Flat / foreground edge: the raw center sample (the landing re-classifies
    // and re-resolves with its own phase; no pre-resolution needed here).
    return float4(centerVelocity, depths[0], edge.isForegroundEdge ? 1.0 : 0.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }