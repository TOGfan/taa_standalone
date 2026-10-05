// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect -- resolve pass (pipeline host)
// ----------------------------------------------------------------------------
// CONTEXT: sub-pixel jitter by PHYSICALLY ROTATING the camera; the per-frame
// bases (P/Q/R) contain the jitter; the velocity buffers contain the jitter
// motion; every jitter-free comparison subtracts s_t - 2*s_{t-1} + s_{t-2}
// (all s in the forward-map-minus-identity sense; see taaFrame.h.hlsl).
// The history buffer stores the previous frame's STABLE output.
//
// BASIS / MAP DIRECTIONS: see taaShared.h.hlsl. The resolve SAMPLES and
// REPROJECTS through the INVERSE map (static landings = the identity); the
// forward map is only the s_t offset source.
//
// STORED MOTION FIELD (#TAA_HistMotion, written by taaMotion.fx.hlsl,
// enabled by taaUseMotionField): per stable texel, the previous frame's RAW
// velocity (xy) / depth (z) + layer flag (w), with kept dilation bands
// carrying the crest's raw sample and revoked candidates reverted to their
// own raw background sample. When the motion field is disabled the field is
// neither written nor read: dilation runs unvalidated, the disocclusion
// tests are skipped, and disocclusion relies on color clipping alone.
//
// HISTORY CLIPPING: the exact-hull simplex clipper (taaClip.h.hlsl) with a
// checked exactness certificate; the fallback is a safe upper bound, so the
// clip can never over-clip.
//
// OUTPUT: RGB = the resolved color -- ALWAYS, also while a debug mode is
// active (the history-copy child stores this RGB into #TAA_History
// unconditionally, so a debug color here would poison the accumulation).
// A = the raw scene's acutance energy for taaFinal's auto-parity sharpener,
// SIGN-ENCODED with the revocation bit; while a debug mode is active, A
// instead carries the packed view payload (PackDebugAlpha, taaShared) --
// the sign still encodes the revocation (the motion writer reads the sign
// only) and the auto-sharpener is bypassed in debug.
//
// PIPELINE (mainP): (1) resolve the jittered render position via the
// inverse map + snap; (2) gather the raw current-frame neighborhood; (3)
// classify the effective surface (taaLayers); (4) reproject into history
// with the effective velocity (taaFrame); (5) gather the color neighborhood
// + FXAA corners + acutance; (6) validate the history landing position
// (filter support must fit); (7) exact jitter plumbing (taaFrame);
// (8) own-history dilation validation through the stored field
// (taaDisocclusion); (9) history landing analysis (taaDisocclusion);
// (10) disocclusion tests (taaDisocclusion); (11) color stats, history
// resampling (taaResample), clipping (taaClip), drift; (12) feedback and
// the temporal blend; (13) output.
//
// DEBUG MODES (taaDebugMode): rendered by taaFinal from a payload the
// resolve stashes into #TAA_Result.a; the resolve's RGB is always the real
// blend, so the accumulation continues while debugging and debug switches
// off with zero recovery time. View bases are the resolved output:
//   0 off | 1 frame motion | 2 disocclusion breakdown (R=depth, G=velocity,
//   B=suppressed pursuit alert) | 3 center velocity | 4 linearized depth |
//   5 history color (the STORED #TAA_History buffer, read by taaFinal) |
//   6 landing effective velocity | 7 pursuit divergence | 8 layer state
//   (orange=revoked candidate, red=kept dilation zone, cyan=crest,
//   magenta=depth-flat but velocity-straddled) | 9 dilation-gate breakdown
//   (R=revoked, G=kept candidate: full=flag branch, half=depth branch,
//   B=depth-rejected) | 10 alignment-drop activity | 11 hull clipper state
//   (G=exact -- simplex certificate or the collinear 1D solve, half-G=
//   certified dual-descent bound, R=defensive fallback that should never
//   fire, B=iterations used) | 12 dejittered residual (jitter-cancel
//   verification).
//
// FILE ARCHITECTURE (flat includes, this file is the only include issuer):
//   taaShared.h.hlsl     pure math shared with taaMotion (map directions,
//                         the depth-curvature classifier)
//   taaConstants.h.hlsl  all static constants
//   taaColor.h.hlsl      working color spaces, tonemapping, acutance
//   taaFrame.h.hlsl      camera bases, jitter flow, history reprojection
//   taaVelocity.h.hlsl   velocity-field math, pair rules, extrapolation
//   taaResample.h.hlsl   Kaiser history filters, fallback FXAA
//   taaLayers.h.hlsl     layer classification, current-frame gather
//   taaDisocclusion.h.hlsl motion-field gate, landing, parallax fit, pursuit
//   taaClip.h.hlsl       color stats, exact-hull clip, feedback, drift
//   taaDebug.h.hlsl      debug view rendering from the packed alpha payload
//                         (included by taaFinal.fx.hlsl, NOT the resolve)
// The fragment headers are NOT standalone shaders; they compile only in this
// file's context (the samplers and cbuffer below) and are included in
// dependency order.
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"
#include "shaders/common/postFx/taa/taaConstants.h.hlsl"

uniform_sampler2D(sceneTex,         0);
uniform_sampler2D(depthTex,         1);
uniform_sampler2D(historyTex,       2);
uniform_sampler2D(velocityTex,      3);
uniform_sampler2D(historyMotionTex, 4); // previous frame's motion field

// ============================================================================
// CBUFFER (all constants are set BY NAME from client/postFx/taa.lua)
// ============================================================================
cbuffer perDraw
{
    float  taaFeedbackMin;                  float  taaFeedbackMax;
    float  taaVarianceGamma;                float  taaSoftClip;
    float  taaChromaVarianceMod;            float  taaJitterFlickerPadding;
    float  taaJitterFlickerFade;            float  taaDepthRejection;
    float  taaTanHalfFovX;                  float  taaTanHalfFovY;
    float  taaUseDepthDilation;             float  taaUseMotionField;
    float  taaLumaVariance;                 float  taaUseHullClipping;
    float  taaColorSpaceOklab;              float  taaJitterAwareVariance;
    float  taaVelocityAlignedVariance;      float  taaAlignmentFeedbackDrop;
    float  taaMotionBlendDropSpeed;         float  taaUseKaiser6;
    float  taaFireflyClamp;                 float  taaFallbackFXAA;
    float  taaMotionBlendStart;             float  taaClipDistanceRejectionEnabled;
    float  taaClipDistanceRejectionAmount;  float  taaClipDistanceRejectionMinError;
    float  taaDirectionalVariance;          float  taaDebugMode;
    float  taaCurPX;                        float  taaCurPY;
    float  taaCurPZ;                        float  taaCurQX;
    float  taaCurQY;                        float  taaCurQZ;
    float  taaCurRX;                        float  taaCurRY;
    float  taaCurRZ;                        float  taaPrevPX;
    float  taaPrevPY;                       float  taaPrevPZ;
    float  taaPrevQX;                       float  taaPrevQY;
    float  taaPrevQZ;                       float  taaPrevRX;
    float  taaPrevRY;                       float  taaPrevRZ;
    float  taaHistoryOvershoot;             float  taaLumaDriftStrength;
    float  taaLumaDriftChromaTol;           float  taaClipOvershoot;
    float  taaVelRejection;                 float  taaVelGradientScale;
    // taaDepthParallaxStep: the camera's forward displacement this frame
    // along the PREVIOUS frame's forward axis, in the units of 1/rawDepth.
    // 0 = not provided; the shader then measures T_y locally.
    float  taaDepthParallaxStep;            float  taaCrossTestStrength;
    float  taaJitPrev2Yaw;                  float  taaJitPrev2Pitch;

    float2 oneOverTargetSize;
    POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#include "shaders/common/postFx/taa/taaColor.h.hlsl"
#include "shaders/common/postFx/taa/taaFrame.h.hlsl"
#include "shaders/common/postFx/taa/taaVelocity.h.hlsl"
#include "shaders/common/postFx/taa/taaResample.h.hlsl"
#include "shaders/common/postFx/taa/taaLayers.h.hlsl"
#include "shaders/common/postFx/taa/taaDisocclusion.h.hlsl"
#include "shaders/common/postFx/taa/taaClip.h.hlsl"

#ifdef SHADER_STAGE_VS
#define mainV main
#else
#define mainP main
#endif

// ============================================================================
// REVOCATION TRANSPORT
// ----------------------------------------------------------------------------
// EVERY return path must transport the revocation bit through the alpha's
// SIGN -- including the debug paths, where PackDebugAlpha carries it in the
// same bit. The writer ignores the bit for non-candidates, so a
// conservative "revoked" is safe on returns that fire before validation.
// ============================================================================
float TransportAlpha(bool revoked, float magnitude)
{
    return revoked ? -(magnitude + kRevokedAlphaEpsilon) : magnitude;
}

// ============================================================================
// MAIN PIXEL SHADER
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    ViewportParams vp          = GetViewportParams(oneOverTargetSize);
    CameraBasis currentCamera  = GetCurrentFrameCameraBasis();
    CameraBasis previousCamera = GetPreviousFrameCameraBasis();
    float  coherenceRadiusPx   = max(taaVelRejection, kMinVelCoherenceRadiusPx);

    // ------------------------------------------------------------------
    // DEBUG PAYLOAD STASH: while a debug mode is active the output's RGB
    // must stay the REAL blended color (the history-copy child stores it
    // into #TAA_History unconditionally), so no view early-returns. Each
    // view stashes its state into (dbgCode, dbgA, dbgB) as soon as its data
    // exists; the exits compose it into the alpha via PackDebugAlpha
    // (which preserves the revocation sign for the motion writer), and
    // taaFinal renders the view. dbgCode 0 = nothing stashed (taaFinal
    // shows the resolved color). The pipeline therefore always runs to the
    // blend during debug, the accumulation stays live, and the views
    // observe (and switch off from) the true converged state.
    // ------------------------------------------------------------------
    bool  debugActive = (taaDebugMode > 0.5);
    float dbgCode = 0.0, dbgA = 0.0, dbgB = 0.0;

    // ------------------------------------------------------------------
    // 1) Resolve the jittered render position of this stable output pixel.
    //    Forward map: ONLY the s_t jitter offset source. Inverse map: the
    //    sampling snap AND the reprojection base (static landings = identity).
    // ------------------------------------------------------------------
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float2 stableInFrameUV   = InverseReprojectThroughCamera(IN.uv0, currentCamera, taaTanHalfFovX, taaTanHalfFovY);
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    // Modes 3/4 need nothing beyond step 1 -- stash and fall through.
    if (taaDebugMode > 3.5 && taaDebugMode < 4.5)
    {
        dbgCode = 4.0;
        dbgA    = saturate(LinearizeDepth(centerDepthRaw) / kDebugLinearDepthRange);
    }
    else if (taaDebugMode > 2.5 && taaDebugMode < 3.5)
    {
        float2 vPx = abs(centerVelocityJitteredUV * vp.sizePixels) * kDebugVelocityScale;
        dbgCode = 3.0;
        dbgA    = saturate(vPx.x);
        dbgB    = saturate(vPx.y);
    }

    // ------------------------------------------------------------------
    // 2) Gather the raw 3x3 current-frame depth/velocity neighborhood.
    //    Neighbor fetches are needed for depth dilation, the velocity-
    //    extrapolation gates, and the stored-field disocclusion tests
    //    (landing + pursuit classification); when none of those are active
    //    the neighborhood is replicated flat.
    // ------------------------------------------------------------------
    bool useDepthDilation   = (taaUseDepthDilation > 0.5);
    bool useMotionField     = (taaUseMotionField > 0.5);
    bool depthTestActive    = useMotionField && (taaDepthRejection > 0.001);
    bool disocclusionActive = useMotionField && ((taaDepthRejection > 0.001) || (taaVelRejection > 0.001));
    bool needNeighbors      = useDepthDilation || disocclusionActive;
    bool fxaaEnabled        = (taaFallbackFXAA > 0.5);

    float2 tapUVs[9];
    Build3x3TapUVs(pixel.snappedUV, vp.texelSize, vp.minUV, vp.maxUV, tapUVs);

    CurrentFrameNeighborhood neighborhood = GatherCurrentFrameNeighborhood(
        pixel.snappedUV, tapUVs, centerDepthRaw, centerVelocityJitteredUV,
        needNeighbors, needNeighbors);

    // ------------------------------------------------------------------
    // 3) Classify and resolve the effective surface (crest anchor / edge
    //    phase rule / two-tap extrapolation / flat layer-aware selection).
    // ------------------------------------------------------------------
    SurfaceEdgeState edge = AnalyzeSurfaceEdges(neighborhood, useDepthDilation);

    LayerSurface currentLayer = ClassifyLayerSurface(
        neighborhood.depthRaw, neighborhood.velocityJitteredUV, pixel.fracPx,
        vp.sizePixels, coherenceRadiusPx, useDepthDilation, false, edge);

    // Velocity spread across the jitter-aligned phase quad (raw values) -- a
    // noise estimate consumed only by the depth disocclusion test.
    float2 quadVelocitySpreadUV = float2(0.0, 0.0);
    if (depthTestActive)
    {
        float2 qv00, qv10, qv01, qv11;
        SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, qv00, qv10, qv01, qv11);
        quadVelocitySpreadUV = max(max(qv00, qv10), max(qv01, qv11))
                             - min(min(qv00, qv10), min(qv01, qv11));
    }

    // Foreground crest geometry: consumed only by the depth disocclusion test.
    ForegroundGeometry foreground;
    foreground.crestDrop = 0.0;
    foreground.slope     = 0.0;
    if (depthTestActive)
        foreground = ComputeForegroundGeometry(neighborhood, currentLayer.effectiveDepth, edge);

    // ------------------------------------------------------------------
    // 4) Reproject into the history buffer (base = inverse map) with the
    //    EFFECTIVE velocity (the extrapolated field sample for a dilation
    //    candidate / a crest cliff-phase). This ONE landing feeds the gate,
    //    the tests and the history color -- the gate validates exactly what
    //    everything else consumes.
    // ------------------------------------------------------------------
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);

    // Mode 1 (frame motion): repro exists; motionPx is already in pixels.
    if (taaDebugMode > 0.5 && taaDebugMode < 1.5)
    {
        float2 vPx = abs(repro.motionPx) * kDebugVelocityScale;
        dbgCode = 1.0;
        dbgA    = saturate(vPx.x);
        dbgB    = saturate(vPx.y);
    }

    // ------------------------------------------------------------------
    // 5) Color neighborhood gather + FXAA corners + acutance energy.
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float3 fxaaCornersRGB[4]; // 0:NW(5), 1:NE(6), 2:SW(7), 3:SE(8)
    float  rawCrossLumaSum = 0.0; // cross taps 1..4, RCAS-luma of SRTM'd raw

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
        if (i <= 4)
            rawCrossLumaSum += SrtmLumaFSR(tapRGB);
        if (fxaaEnabled && i >= 5)
            fxaaCornersRGB[i - 5] = tapRGB;
    }

    float rawHighPass        = SrtmLumaFSR(currentColorRGB) - rawCrossLumaSum * 0.25;
    float rawSharpnessEnergy = rawHighPass * rawHighPass;

    // ------------------------------------------------------------------
    // 6) Validate the history sample position (filter support must fit).
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseKaiser6 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        // Offscreen history = no support: a tentative dilation here can
        // never be validated, so encode it as revoked for the writer. The
        // payloads stashed so far (modes 1/3/4 all predate this point) still
        // travel through the packed alpha; later modes have no data on this
        // path and taaFinal falls back to showing the resolved color.
        bool  earlyRevoked = currentLayer.isDilationZone;
        float earlyAlpha  = debugActive
            ? PackDebugAlpha(earlyRevoked, dbgCode, saturate(dbgA), saturate(dbgB))
            : TransportAlpha(earlyRevoked, rawSharpnessEnergy);

        if (fxaaEnabled)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), earlyAlpha);
        }
        return float4(currentColorRGB, earlyAlpha);
    }

    // ------------------------------------------------------------------
    // 7) Exact jitter plumbing (all offsets in the shared s_tau sense).
    // ------------------------------------------------------------------
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterOffsetPrev2UV = RotationFlowUV(taaJitPrev2Yaw, taaJitPrev2Pitch, IN.uv0);
    float2 jitterCancelUV     = EstimateJitterCancelUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);
    float2 jitterTransportUV  = EstimateJitterTransportUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);

    // ------------------------------------------------------------------
    // 8) Own-history dilation validation (requires the stored motion field):
    //    the candidate validates through the history it would have if it
    //    stayed dilated -- the landing of its EFFECTIVE (extrapolated)
    //    velocity. Revocation downgrades to the texel's own raw background
    //    sample and re-projects with it (matching the writer's stored
    //    state). With the motion field disabled, candidates stay dilated
    //    unvalidated.
    // ------------------------------------------------------------------
    bool dilationRevoked   = false;
    bool dilationCandidate = false;
    bool gateViaFlag       = false;

    if (useMotionField && currentLayer.isDilationZone)
    {
        dilationCandidate = true;

        float rayLenCur  = RayLengthFromUV(IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
        float rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
        float perspScale = rayLenPrev / max(rayLenCur, 1e-6);

        float rawExpClosest = currentLayer.closestDepth * perspScale;

        // Object-side slant allowance: minmod zeroes the gradients across a
        // silhouette cliff, so background-side candidates get NO slant
        // allowance and a strict crest anchor. The allowance's scale (the
        // object's slope over the crest offset) is the same magnitude and
        // shape as the extrapolation's landing displacement, so it covers it.
        float2 crestOffset = currentLayer.closestOffsetPx;
        float  objectSlant = (abs(currentLayer.gradX * crestOffset.x) + abs(currentLayer.gradY * crestOffset.y)) * perspScale;
        float  tolObject   = max(taaDepthRejection * rawExpClosest, 1e-5) + objectSlant;
        float  fgThreshold = rawExpClosest - tolObject;

        HistoryMotionGate gate = SampleHistoryMotionGate(repro.sampleUV, vp);

        // A) Own-history support gate: a tap validates iff it is
        //    foreground-OWNED (flag >= 1) or its stored depth is at the
        //    object itself.
        gateViaFlag = (gate.supportMaxFlag >= 0.75);
        bool quadTouchesForeground = gateViaFlag || (gate.supportDepthMax >= fgThreshold);

        // C) The landing itself was inside the object's dilation band.
        bool touchesAlreadyDilated = (gate.centerFlag >= 1.5) && (gate.gateDepth >= fgThreshold);

        bool historyHasForeground = quadTouchesForeground || touchesAlreadyDilated;

        if (!historyHasForeground)
        {
            // Revoke: the revocation persists via the stored field.
            dilationRevoked = true;
            RevokeDilation(currentLayer, centerDepthRaw, centerVelocityJitteredUV);

            // Re-reproject with the true background velocity: the revoked
            // candidate lands back on its own texel and the tests compare
            // background against background.
            repro = ReprojectToHistory(
                IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);
        }
    }

    // ------------------------------------------------------------------
    // 9) History landing analysis: shared classifier + merged resolution +
    //    the extrapolation on the stored field, gated by the stored
    //    ownership record. currentLayer.isForeground is the POST-revocation
    //    state: a revoked candidate is background. The landing is only
    //    analyzed when a disocclusion test (or a landing debug view)
    //    consumes it.
    // ------------------------------------------------------------------
    bool landingDebug = (taaDebugMode > 5.5 && taaDebugMode < 6.5) ||
                        (taaDebugMode > 11.5 && taaDebugMode < 12.5);
    bool needLanding  = disocclusionActive || landingDebug;
    HistoryLandingSurface landing;
    landing.effectiveDepthRaw               = 0.0;
    landing.effectiveVelocityJitteredPrevUV = float2(0.0, 0.0);
    landing.gradRaw          = float2(0.0, 0.0);
    landing.maxCurvaturePx   = 0.0;
    landing.maxPairGradPx    = 0.0;
    landing.coherentGradPx   = 0.0;
    landing.snapDistPx       = 0.0;
    landing.pairGradPx       = 0.0;
    landing.shallowVelGradPx = 0.0;
    if (needLanding)
    {
        landing = SampleHistoryLandingSurface(
            repro.sampleUV, vp, useDepthDilation, disocclusionActive || landingDebug,
            coherenceRadiusPx,
            currentLayer.isForeground);
    }

    // Mode 6 (landing effective velocity): the landing exists now.
    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
    {
        float2 vPx = abs(landing.effectiveVelocityJitteredPrevUV * vp.sizePixels) * (kDebugVelocityScale * 2.0);
        dbgCode = 6.0;
        dbgA    = saturate(vPx.x);
        dbgB    = saturate(vPx.y);
    }

    // Post-revocation single-layer state (a revoked candidate acts flat).
    bool currentSingleLayer = !(currentLayer.isDilationZone || currentLayer.isForegroundEdge);

    // Mode 8 (layer state + straddle). A bit field: 4 = dilation zone,
    // 2 = foreground edge, 1 = velocity-straddled; the revocation rides the
    // alpha sign.
    if (taaDebugMode > 7.5 && taaDebugMode < 8.5)
    {
        bool velocityStraddled = false;
        if (!currentLayer.isDilationZone && !currentLayer.isForegroundEdge)
        {
            float2 v00, v10, v01, v11;
            SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, v00, v10, v01, v11);
            velocityStraddled = QuadStraddlesVelocityStep(
                v00, v10, v01, v11, neighborhood.velocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
        }
        float f = (currentLayer.isDilationZone   ? 4.0 : 0.0)
                + (currentLayer.isForegroundEdge ? 2.0 : 0.0)
                + (velocityStraddled              ? 1.0 : 0.0);
        dbgCode = 8.0;
        dbgA    = f * 0.125;
    }

    // Mode 10 (alignment-drop activity).
    if (taaDebugMode > 9.5 && taaDebugMode < 10.5)
    {
        float dropAmount = currentSingleLayer
            ? taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment)
            : 0.0;
        dbgCode = 10.0;
        dbgA    = saturate(dropAmount);
        dbgB    = currentSingleLayer ? 1.0 : 0.0;
    }

    // ------------------------------------------------------------------
    // 10) Disocclusion tests (require the stored motion field): the
    //     geometric depth score (the layering oracle / depthGate) + the
    //     velocity rejection. Binary union of both rejections. With the
    //     motion field disabled both are skipped and disocclusion relies on
    //     color clipping alone.
    // ------------------------------------------------------------------
    float depthDisocclusionScore = 0.0;
    if (depthTestActive)
    {
        depthDisocclusionScore = ComputeDepthDisocclusionScore(
            currentLayer.effectiveDepth, currentLayer.effectiveVelocityUV,
            stableInFrameUV, IN.uv0,
            neighborhood.depthRaw, neighborhood.velocityJitteredUV, tapUVs,
            landing.effectiveDepthRaw, landing.gradRaw,
            repro.jitterResidualPx, quadVelocitySpreadUV,
            foreground.slope, neighborhood.closestOffsetPx,
            currentLayer.extrapolationDispPx,
            currentLayer.isDilationZone, currentLayer.isForegroundEdge,
            currentLayer.gradX, currentLayer.gradY,
            edge.depthNoiseFloor, edge.depthQuantStep,
            currentCamera, previousCamera,
            coherenceRadiusPx, historySupportTexels,
            vp);
    }
    bool  depthRejected = (depthDisocclusionScore >= 1.0);

    // Same-surface confidence for the velocity advection prediction.
    float depthGate = saturate((1.0 - depthDisocclusionScore) * 2.0);

    VelocityRejectionResult velocityRejection;
    velocityRejection.rejected     = false;
    velocityRejection.errorRatio   = 0.0;
    velocityRejection.divergencePx = 0.0;
    if (disocclusionActive)
    {
        velocityRejection = EvaluateVelocityRejection(
            repro.sampleUV,
            currentLayer.effectiveVelocityUV,
            landing,
            currentSingleLayer,
            currentLayer.pairGradPx,        // the active extrapolation pair's gradient
            currentLayer.shallowVelGradPx,  // the foreground's shallow-end velocity gradient
            depthGate,
            jitterCancelUV, jitterTransportUV,
            neighborhood.velocityJitteredUV,
            coherenceRadiusPx,
            vp);
    }

    bool disoccluded = depthRejected || velocityRejection.rejected;

    // Mode 2 (disocclusion breakdown). A bit field: 4 = depth rejected,
    // 2 = velocity rejected, 1 = suppressed pursuit alert.
    if (taaDebugMode > 1.5 && taaDebugMode < 2.5)
    {
        float f = (depthRejected ? 4.0 : 0.0)
                + (velocityRejection.rejected ? 2.0 : 0.0)
                + ((velocityRejection.errorRatio > 1.0 && !velocityRejection.rejected) ? 1.0 : 0.0);
        dbgCode = 2.0;
        dbgA    = f * 0.125;
    }

    // Mode 7 (pursuit divergence).
    if (taaDebugMode > 6.5 && taaDebugMode < 7.5)
    {
        dbgCode = 7.0;
        dbgA    = saturate(velocityRejection.divergencePx * 0.5);
        dbgB    = velocityRejection.rejected ? 1.0 : 0.0;
    }

    // Mode 12 (dejittered residual -- jitter-cancel verification).
    if (taaDebugMode > 11.5 && taaDebugMode < 12.5)
    {
        float2 residualPx = (currentLayer.effectiveVelocityUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
        dbgCode = 12.0;
        dbgA    = saturate(length(residualPx) * 0.5);
    }

    // Mode 9 (dilation-gate breakdown). A bit field: 4 = kept candidate,
    // 2 = kept via the flag branch (else the depth branch), 1 = depth
    // rejected; the revocation rides the alpha sign.
    if (taaDebugMode > 8.5 && taaDebugMode < 9.5)
    {
        float f = ((dilationCandidate && !dilationRevoked) ? 4.0 : 0.0)
                + (gateViaFlag ? 2.0 : 0.0)
                + (depthRejected ? 1.0 : 0.0);
        dbgCode = 9.0;
        dbgA    = f * 0.125;
    }

    // ------------------------------------------------------------------
    // 11) Color stats, history color resampling, clipping, drift.
    // ------------------------------------------------------------------
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized, pixel.fracPx);
    float3 clipMargin = taaClipOvershoot * max(colorStats.aabbMax - colorStats.aabbMin, 0.0);

    float3 historyColorSpace =
        (taaUseKaiser6 > 0.5)
        ? SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV)
        : SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);

    // With zero resampling-overshoot margin the tap-footprint clamp keeps the
    // resampled color inside the convex hull of valid taps -- provably in
    // gamut -- so the gamut recompression (an Oklab/RGB round trip) is
    // skipped.
    if (taaHistoryOvershoot > 0.001)
        historyColorSpace = CompressGamut(historyColorSpace);

    // Mode 5 (history color) carries no numeric payload: taaFinal renders it
    // directly from the stored #TAA_History buffer (the exact input of this
    // frame's blend). The code only routes the dispatch.
    if (taaDebugMode > 4.5 && taaDebugMode < 5.5)
    {
        dbgCode = 5.0;
    }

    float2 hullDiag = float2(0.0, 0.0);
    float3 clippedHistorySpace = ClipHistoryToNeighborhood(
        historyColorSpace, colorStats, repro.motionNormalized, taaVarianceGamma,
        neighborhoodColorSpace, clipMargin, hullDiag);

    // Mode 11 (hull clipper state): A = hullDiag.x / 4, B = pivots.
    if (taaDebugMode > 10.5 && taaDebugMode < 11.5)
    {
        dbgCode = 11.0;
        dbgA    = hullDiag.x * 0.25;
        dbgB    = saturate(hullDiag.y * 0.2);
    }

    float clipDistanceRejection = 0.0;
    if (taaClipDistanceRejectionEnabled > 0.5)
        clipDistanceRejection = ComputeClipDistanceRejection(clippedHistorySpace, historyColorSpace, colorStats);

    ApplyLumaDriftCorrection(clippedHistorySpace, currentColorSpace, colorStats);

    // ------------------------------------------------------------------
    // 12) Feedback & temporal blending.
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, clipDistanceRejection, currentSingleLayer);
    float currentBlendWeight = 1.0 - historyFeedback;

    // Binary disocclusion: full rejection on trigger, zero partial credit.
    if (disoccluded)
        currentBlendWeight = 1.0;

    float3 currentFrameColorSpace = currentColorSpace;
    if (fxaaEnabled)
    {
        float fxaaWeight = ComputeFXAAFilterWeight(currentBlendWeight);
        if (fxaaWeight > 0.001)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 13) Final blend & output: RGB is ALWAYS the real resolved color (also
    //     during debug -- this is what makes the history store
    //     poison-proof). Alpha = the acutance metric, SIGN-ENCODED with the
    //     revocation bit; while a debug mode is active it carries the packed
    //     view payload instead (PackDebugAlpha keeps the sign, so the motion
    //     writer is unaffected; the magnitude's only other consumer,
    //     taaFinal's auto-sharpener, is bypassed in debug).
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    // Clamp scalar luminance only; do NOT clamp signed chrominance channels.
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    float outAlpha = debugActive
        ? PackDebugAlpha(dilationRevoked, dbgCode, saturate(dbgA), saturate(dbgB))
        : TransportAlpha(dilationRevoked, rawSharpnessEnergy);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}