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
// SCHEDULING: a CHILD of the resolve pass (TAA_PreFx), executing inside the
// resolve's pass chain after the history-copy sibling. As a child it binds
// its three texture slots the same way the resolve does. The pass is NEVER
// touched after creation -- no enable()/disable(), no scheduling fields of
// its own; any lifecycle manipulation re-registers it outside the resolve's
// pass chain on the Vulkan backend and its named-target bindings then fail
// ("missing vulkan resource" spam). useMotionField therefore gates the
// resolve's stored-field ANALYSIS only; this pass always runs (a single
// fullscreen write whose output is ignored when the resolve does not read
// the field).
// The writer ignores the revocation bit for non-candidates, so the
// conservative "revoked" encodings of the resolve's early/debug returns are
// always safe.
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
// #TAA_Result: the resolve wrote it THIS frame; as a child of the resolve we
// run inside its pass chain, after the final/history-copy siblings, so the
// revocation sign in its alpha is fresh.
uniform_sampler2D(resolveOutputTex, 2);

// ============================================================================
// CBUFFER (all constants are set BY NAME from client/postFx/taa.lua)
// ============================================================================
cbuffer perDraw
{
    float taaUseDepthDilation;  float taaUseMotionField;
    float taaTanHalfFovX;       float taaTanHalfFovY;
    float taaCurPX;             float taaCurPY;
    float taaCurPZ;             float taaCurQX;
    float taaCurQY;             float taaCurQZ;
    float taaCurRX;             float taaCurRY;
    float taaCurRZ;

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

    // Fast path: when the resolve does not read the stored field
    // (useMotionField off), write the CHEAP VALID center sample instead of
    // the raw classification -- the output is never consumed, and writing a
    // valid (not garbage) field means toggling the setting back on needs no
    // warm-up. Saves the 9-tap depth gather + the classifier + the crest
    // scan on every pixel. The PASS itself still runs: its lifecycle must
    // never be touched (see the header).
    if (taaUseMotionField < 0.5)
    {
        float2 centerVelocity = tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;
        float  centerDepth    = tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
        return float4(centerVelocity, centerDepth, 0.0);
    }

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