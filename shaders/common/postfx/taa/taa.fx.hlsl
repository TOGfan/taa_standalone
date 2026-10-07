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
// STORED MOTION FIELD (#TAA_HistMotion, written by taaMotion.fx.hlsl):
// per stable texel, the previous frame's RAW velocity (xy) / depth (z) +
// layer flag (w). Read by the resolve for dilation validation and the
// disocclusion tests.
//
// HISTORY CLIPPING: the Mahalanobis statistic gate (taaClip.h.hlsl) -- now
// PER-CHANNEL normalized (the record's TOTAL split by the live spatial
// covariance's diagonal; the previous isotropic form paid the full total to
// every channel: 3x the true per-channel variance under isotropy, chi = 2.8
// actually gating at ~4.85 sigma), with the PHASE-CORRECTED test vector
// d_corr = h - mu + beta*(centroid - u), the exact transient/Student
// accounting, and the consumer-split innovation.
//
// OUTPUT: RGB = the resolved color -- ALWAYS, also while a debug mode is
// active. A = the packed CLIP STATE: [31] the revocation sign, [30:27] tag
// 0110, [26:20] the state sigma code (0 = cold), [19:8] the acutance target
// (12 bits, SQRT-COMPRESSED -- energy range [0,9] -- and TEMPORALLY
// STABILIZED against the previous frame's value at the landing),
// [7:1] the record's age, [0] the fiction flag. While a debug mode is
// active, A carries the packed debug payload (tag 0111) with the sigma
// embedded in its low 7 bits.
//
// PIPELINE (mainP): (1) resolve the jittered render position; (2) gather
// the raw current-frame neighborhood; (3) classify the effective surface;
// (4) reproject into history; (5) gather the color neighborhood + FXAA
// corners + acutance; (6) validate the history landing position; (7) exact
// jitter plumbing; (8) own-history dilation validation; (9) history landing
// analysis; (10) disocclusion tests; (11) color stats (full tap set) with
// the centroid/LS-residual estimators, the temporal clip-state
// read/reset/advance, the acutance EWMA, the anti-alignment (corrected
// vector, standardized guards), the statistic gate; (12) feedback and the
// temporal blend; (13) output.
//
// DEBUG MODES (taaDebugMode): rendered by taaFinal from the alpha payload.
//   0 off | 1 frame motion | 2 disocclusion breakdown | 3 center velocity |
//   4 linearized depth | 5 history color | 6 landing effective velocity |
//   7 pursuit divergence | 8 layer state | 9 dilation-gate breakdown |
//   10 alignment-drop activity | 11 clip gate state | 12 dejittered residual.
//
// FILE ARCHITECTURE (flat includes, this file is the only include issuer):
//   taaShared / taaConstants / taaColor / taaFrame / taaVelocity /
//   taaResample / taaLayers / taaDisocclusion / taaClip / taaDebug
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
// #TAA_History bound a SECOND time, through a POINT sampler: the bit-packed
// clip state in the alpha must never see bilinear weights. Slot 2 stays
// linear for the fused Kaiser color taps. (A linear sampler is NOT bit-exact
// even at a snapped texel center: the center (k+0.5)/N is not
// float-representable for non-power-of-two sizes.)
uniform_sampler2D(historyStateTex, 5);

// ============================================================================
// CBUFFER (all constants are set BY NAME from client/postFx/taa.lua)
// ----------------------------------------------------------------------------
// taaUseHullClipping is a LEGACY FLAG NAME: the temporal clip-state switch.
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
    float  taaFallbackFXAA;
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
    // The t-2 jitter rotation, pre-evaluated on the CPU (per-draw uniforms;
    // the shader's per-pixel sin/cos of them was 4 wasted transcendentals).
    float  taaJitPrev2YawSin;               float  taaJitPrev2YawCos;
    float  taaJitPrev2PitchSin;             float  taaJitPrev2PitchCos;
    // The Studentization fit, computed BY THE HOST from the effective
    // radius (studentFitAB in client/postFx/taa.lua):
    //     S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2
    // exact at nu = 4 (age 0) and nu = 12.33 (converged); see taaClip.
    // Unset constants read 0: S = 1 (the pre-Student gate, graceful).
    float  taaStudentA;                     float  taaStudentB;
    // The gate's mu-share scoping blend (see taaClip.h.hlsl): 0 = the
    // conservative full-E payment, 1 = residual-scoped (the exact mu
    // variance). Set from client/postFx/taa.lua (clipScopedMu).
    float  taaClipScopedMu;
    // 1 when the final pass will actually consume the acutance transport
    // (auto sharpening on AND a nonzero sharpness) -- set by the host. Gates
    // the raw-scene acutance metric, the transport EWMA and -- when the
    // clip memory is also off -- the entire historyStateTex fetch.
    float  taaAcutanceActive;

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

// The FXAA corner taps (the diagonal scene taps 5..8), fetched at the point
// of use -- see the step-5 note in mainP. Bit-identical to the values the
// step-5 loop used to retain (same UVs, same sampler, same max()).
void FetchFxaaCorners(float2 tapUVs[9], out float3 cornersRGB[4])
{
    [unroll]
    for (int c = 0; c < 4; ++c)
        cornersRGB[c] = max(tex2Dlod(sceneTex, float4(tapUVs[c + 5], 0.0, 0.0)).rgb, 0.0);
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
    // must stay the REAL blended color; the view state rides the alpha.
    // ------------------------------------------------------------------
    bool  debugActive = (taaDebugMode > 0.5);
    float dbgCode = 0.0, dbgA = 0.0, dbgB = 0.0;

    // ------------------------------------------------------------------
    // 1) Resolve the jittered render position of this stable output pixel.
    // ------------------------------------------------------------------
    // One forward evaluation serves both maps: the inverse is 2u - F(u), and
    // F(u) IS currentJitteredUV -- the previous form evaluated the identical
    // forward map (same arguments, same fallback) twice per pixel.
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float2 stableInFrameUV   = 2.0 * IN.uv0 - currentJitteredUV;
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;



    // ------------------------------------------------------------------
    // 2) Gather the raw 3x3 current-frame depth/velocity neighborhood.
    // ------------------------------------------------------------------
    bool useDepthDilation   = (taaUseDepthDilation > 0.5);
    bool useMotionField     = (taaUseMotionField > 0.5);
    bool acutanceActive     = (taaAcutanceActive > 0.5);
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
    // 3) Classify and resolve the effective surface.
    // ------------------------------------------------------------------
    SurfaceEdgeState edge = AnalyzeSurfaceEdges(neighborhood, useDepthDilation);

    LayerSurface currentLayer = ClassifyLayerSurface(
        neighborhood.depthRaw, neighborhood.velocityJitteredUV, pixel.fracPx,
        vp.sizePixels, coherenceRadiusPx, useDepthDilation, false, edge);

    // Velocity spread across the jitter-aligned phase quad (raw values).
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
    foreground.slope = 0.0;
    if (depthTestActive)
        foreground = ComputeForegroundGeometry(neighborhood, currentLayer.effectiveDepth, edge);

    // ------------------------------------------------------------------
    // 4) Reproject into the history buffer with the EFFECTIVE velocity.
    // ------------------------------------------------------------------
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);



    // ------------------------------------------------------------------
    // 5) Color neighborhood gather + FXAA corners + acutance energy.
    // ------------------------------------------------------------------
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float  rawCrossLumaSum = 0.0; // cross taps 1..4, RCAS-luma of SRTM'd raw

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
        if (acutanceActive && !debugActive && i <= 4)
            rawCrossLumaSum += SrtmLumaFSR(tapRGB);
        // The FXAA corner taps are NOT retained here: holding four raw
        // float3s live across the whole shader cost ~12 registers on every
        // pixel. FetchFxaaCorners() re-fetches them at the two points of
        // use -- extra fetches only on FXAA-engaging and border pixels,
        // none in steady state.
    }

    // The metric runs only when the transport has a consumer (auto
    // sharpening active) and debug is off (PackDebugAlpha carries no
    // acutance field). Gated to zero, both pack sites naturally write the
    // zero-energy field.
    float rawHighPass        = (acutanceActive && !debugActive)
                              ? SrtmLumaFSR(currentColorRGB) - rawCrossLumaSum * 0.25
                              : 0.0;
    float rawSharpnessEnergy = rawHighPass * rawHighPass;

    // ------------------------------------------------------------------
    // 6) Validate the history sample position (filter support must fit).
    //    Border pixels pack the FRESH acutance value: the stabilized EWMA
    //    reads the previous alpha at the landing (step 11), which is not
    //    meaningful at an out-of-support landing anyway.
    // ------------------------------------------------------------------
    float  historySupportTexels = (taaUseKaiser6 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        bool  earlyRevoked = currentLayer.isDilationZone;
        float earlyAlpha  = debugActive
            ? PackDebugAlpha(earlyRevoked, dbgCode, saturate(dbgA), saturate(dbgB), 0.0)
            : PackClipStateAlpha(earlyRevoked, 0.0, 0.0,
                                 saturate(sqrt(rawSharpnessEnergy)) * (1.0 / 3.0), false);

        if (fxaaEnabled)
        {
            float3 fxaaCornersRGB[4];
            FetchFxaaCorners(tapUVs, fxaaCornersRGB);
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), earlyAlpha);
        }
        return float4(currentColorRGB, earlyAlpha);
    }

    // ------------------------------------------------------------------
    // 7) Exact jitter plumbing. jitterOffsetCurUV is the per-pixel jitter
    //    offset s_t (the velocity tests' cancel term).
    // ------------------------------------------------------------------
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterOffsetPrev2UV = RotationFlowUV(
        float4(taaJitPrev2YawSin, taaJitPrev2YawCos, taaJitPrev2PitchSin, taaJitPrev2PitchCos),
        IN.uv0);
    float2 jitterCancelUV     = EstimateJitterCancelUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);
    float2 jitterTransportUV  = EstimateJitterTransportUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);

    // ------------------------------------------------------------------
    // 8) Own-history dilation validation (requires the stored motion field).
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

        float2 crestOffset = currentLayer.closestOffsetPx;
        float  objectSlant = (abs(currentLayer.gradX * crestOffset.x) + abs(currentLayer.gradY * crestOffset.y)) * perspScale;
        float  tolObject   = max(taaDepthRejection * rawExpClosest, 1e-5) + objectSlant;
        float  fgThreshold = rawExpClosest - tolObject;

        HistoryMotionGate gate = SampleHistoryMotionGate(repro.sampleUV, vp);

        gateViaFlag = (gate.supportMaxFlag >= 0.75);
        bool quadTouchesForeground = gateViaFlag || (gate.supportDepthMax >= fgThreshold);
        bool touchesAlreadyDilated = (gate.centerFlag >= 1.5) && (gate.gateDepth >= fgThreshold);

        if (!(quadTouchesForeground || touchesAlreadyDilated))
        {
            dilationRevoked = true;
            RevokeDilation(currentLayer, centerDepthRaw, centerVelocityJitteredUV);

            repro = ReprojectToHistory(
                IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);
        }
    }

    // ------------------------------------------------------------------
    // 9) History landing analysis (only when consumed).
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



    // Post-revocation single-layer state (a revoked candidate acts flat).
    bool currentSingleLayer = !(currentLayer.isDilationZone || currentLayer.isForegroundEdge);





    // ------------------------------------------------------------------
    // 10) Disocclusion tests (require the stored motion field).
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
            currentLayer.pairGradPx,
            currentLayer.shallowVelGradPx,
            depthGate,
            jitterCancelUV, jitterTransportUV,
            neighborhood.velocityJitteredUV,
            coherenceRadiusPx,
            vp);
    }

    bool disoccluded = depthRejected || velocityRejection.rejected;









    // ------------------------------------------------------------------
    // 11) Color stats (full tap set), the temporal clip state (sigma +
    //     age + fiction), the acutance EWMA, the anti-alignment, the
    //     statistic gate.
    // ------------------------------------------------------------------
    // The stats run on the FULL tap set, always. The phase bit's regressor
    // is the sample's snap offset; the GATE's test vector additionally
    // carries the centroid correction (see taaClip).
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized,
        pixel.fracPx);

    // BRANCH, not ternary: ?: materializes BOTH kernels (30 history fetches
    // instead of 21, plus both weight suites). The condition is a per-draw
    // uniform and everything here is tex2Dlod/pure math -- no gradient
    // hazard, the branch is free.
    float3 historyColorSpace;
    if (taaUseKaiser6 > 0.5)
        historyColorSpace = SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    else
        historyColorSpace = SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);

    // NaN/Inf guard: the strict form catches NaN (NaN < x is false) AND both
    // infinities (the old `>= 0` test passed +Inf, and Inf*tGate=0 = NaN
    // poisoned the blend for a frame). Collapse to the neighborhood mean.
    if (!(dot(historyColorSpace, historyColorSpace) < kLargeValue))
        historyColorSpace = colorStats.mean;

    // Gamut recompression: required with resampling-overshoot margin, and
    // in OKLAB mode ALWAYS (Oklab is nonlinear -- convexity is a
    // linear-space argument).
    if (taaHistoryOvershoot > 0.001 || taaColorSpaceOklab > 0.5)
        historyColorSpace = CompressGamut(historyColorSpace);



    // ---- temporal clip state + acutance transport -------------------------
    bool  temporalStateEnabled = (taaUseHullClipping > 0.5);
    float sigmaPrevSq = -1.0;
    float agePrev     = 0.0;
    bool  fictionPrev = false;

    // The previous stable output's packed alpha, read AT THE LANDING through
    // the DEDICATED POINT-SAMPLED binding (slot 5). ONE fetch serves BOTH
    // consumers (the clip record and the acutance EWMA) and is skipped
    // entirely when neither is active: with the clip memory off AND
    // sharpening off/inactive, a whole history-buffer fetch per pixel
    // disappears.
    float prevStateAlpha = 0.0;
    if (temporalStateEnabled || acutanceActive)
    {
        float2 stateUV = SnapUVToTexel(repro.sampleUV, vp).snappedUV;
        prevStateAlpha = tex2Dlod(historyStateTex, float4(stateUV, 0.0, 0.0)).a;
    }
    if (temporalStateEnabled)
        DecodeClipState(prevStateAlpha, sigmaPrevSq, agePrev, fictionPrev);

    // The acutance target, TEMPORALLY STABILIZED: the fresh measurement is
    // taken at the CURRENT jitter phase and swings ~2x across the phase
    // cycle on edges; packing it raw made the sharpener's boost itself
    // flicker. EWMA against the previous frame's decoded value AT THE
    // LANDING (the same advection semantics as the record). Foreign/debug
    // tags decode 0, so the target ramps in over ~4 frames after debug or
    // on reveals -- no sharpening pops. Gated on the transport having a
    // consumer; inactive packs the zero-energy field.
    float acutanceStabLin = 0.0;
    if (acutanceActive)
    {
        acutanceStabLin = saturate(sqrt(rawSharpnessEnergy)) * (1.0 / 3.0);
        float ePrev = DecodeAcutanceLinear(prevStateAlpha);
        acutanceStabLin = lerp(ePrev, acutanceStabLin, kSharpEwmaRate);
    }

    // This frame's own fiction state: a TRAVELING dilation band (dilation
    // zone with real motion), DIRECTION-BLIND. Static dilation is NOT
    // fiction.
    bool dilationFictionNow = currentLayer.isDilationZone
                            && (repro.motionMagnitudePx > 0.5);

    // FICTION-ADVected record (traveling-band writer) = foreign content:
    // killed unconditionally by every non-band reader (the sentinel gate).
    if (fictionPrev && !dilationFictionNow)
    {
        sigmaPrevSq = -1.0;
    }

    // The innovation against the stored accumulator, PRE-clip, and its
    // PHASE-CORRECTED form (the mean-structure test input).
    float3 innov       = currentColorSpace - historyColorSpace;
    float  innovSq     = dot(innov, innov);
    float3 innovCorr   = innov - colorStats.phaseShift;
    float  innovCorrSq = dot(innovCorr, innovCorr);

    // The spatial prior: the taps' TOTAL variance tr(Cov) -- the re-seed and
    // the change-point tests' scale (NEVER the record: the record is exactly
    // what a ghost arms).
    float spatialSigmaSq = dot(colorStats.sigma, colorStats.sigma);
    float spatialScaleSq = max(spatialSigmaSq, kClipSigmaRecordFloorSq);
    bool  hadRecord      = (sigmaPrevSq >= 0.0);
    bool  innovFinite    = (innovSq < kLargeValue);

    // (a) THE ANTI-ALIGNMENT RESET (the ghost signature; see taaClip): the
    //     mean lies BETWEEN the current sample and the history, the
    //     PHASE-CLEAN corrected innovation corroborates, the history is
    //     meaningfully outside, and DILATION ZONES are exempt. The test
    //     vector is the PHASE-CORRECTED d (the centroid correction -- the
    //     un-corrected d carried the kernel-truncation wobble, perturbing
    //     the direction test), and the guards are STANDARDIZED per channel
    //     by the aniso-capped spatial sigma: the old trCov-normalized forms
    //     were luma-scaled on luma-dominant content, so a pure-chroma ghost
    //     had to clear a LUMA-scaled corroboration AND distance bar. The
    //     standardized forms are identical under isotropy.
    // (b) THE SHOCK: innovation > kClipSpikeRatio times BOTH the record
    //     and the neighborhood's squared AABB range.
    // (c) GEOMETRY AS EVIDENCE, NOT COMMAND: a geometric reject resets only
    //     when the color side corroborates (chi^2_3 at 95%).
    float3 dAlign        = historyColorSpace - colorStats.mean + colorStats.gatePhaseShift;
    float  dAlignSq      = dot(dAlign, dAlign);
    bool   antiAligned   = dot(innovCorr, dAlign)
                           < -kAlignCos * sqrt(innovCorrSq * dAlignSq);
    float3 guardSigma    = AnisoClampSigma(colorStats.sigma);
    float3 zAlign        = dAlign / guardSigma;
    float3 zInnov        = innovCorr / guardSigma;
    bool   farEnough     = dot(zAlign, zAlign) > kAlignDistChiSq;
    bool   corroborated  = dot(zInnov, zInnov) > kAlignSpatialChiSq;
    // SINGLE-LAYER SANCTITY (supersedes the dilation-only exemption): the
    // anti-alignment's premise -- the neighborhood mean tracking one
    // surface -- requires single-layer statistics. On dilation zones AND
    // foreground-edge (crest) texels the 3x3 spans an ownership boundary:
    // the mixture mean lies between the layers BY CONSTRUCTION, which is
    // the ghost geometry the test reads. The sweep transient at a moving
    // crest fires it solidly (|z_d|^2 ~ 1-3, |z_i|^2 ~ 10+), producing a
    // reset wave riding every moving silhouette. currentSingleLayer is the
    // same post-revocation predicate the alignment drop and the velocity
    // rejection already use for the same reason; a revoked dilation
    // candidate re-arms the test (it acts flat).
    bool   antiAlignReset = hadRecord && currentSingleLayer
                         && antiAligned && farEnough && corroborated;

    float rangeSq = dot(colorStats.aabbMax - colorStats.aabbMin,
                        colorStats.aabbMax - colorStats.aabbMin);
    bool  spike   = hadRecord
        && (innovSq > kClipSpikeRatio * max(sigmaPrevSq, rangeSq));

    bool  resetRecord = hadRecord
        && (spike || antiAlignReset || !innovFinite
            || (disoccluded && innovSq > kClipCorroborateRatio * sigmaPrevSq));
    bool  carryRecord = hadRecord && !resetRecord;

    // FULL-RATE, WINSORIZED ingest of the RAW second moment -- no motion
    // fade. THE SEED IS THE SPATIAL PRIOR (the new regime's measured
    // spread). DILATION-FICTION FREEZE (traveling bands): never ingested,
    // the record and its age frozen, the flag written.
    bool fictionFreeze = carryRecord && dilationFictionNow;

    float winsorCap   = kClipWinsorC * kClipWinsorC * max(sigmaPrevSq, kClipSigmaRecordFloorSq);
    float innovIngest = min(innovSq, winsorCap);

    float sigmaStatSq, gateSigmaSq, ageNext;
    if (fictionFreeze)
    {
        sigmaStatSq = max(sigmaPrevSq, kClipSigmaRecordFloorSq);   // frozen, stored + flagged
        gateSigmaSq = sigmaStatSq;
        ageNext     = agePrev;                                     // frozen: the dof do not grow
    }
    else if (carryRecord)
    {
        sigmaStatSq = max(lerp(sigmaPrevSq, innovIngest, kClipSigmaEmaRate), kClipSigmaRecordFloorSq);
        gateSigmaSq = sigmaStatSq;
        ageNext     = min(agePrev + 1.0, kClipMaxAge);
    }
    else
    {
        sigmaStatSq = spatialScaleSq;          // the seed: the new regime's measured spread
        gateSigmaSq = -1.0;                    // cold gate (reset or no record)
        ageNext     = 0.0;                     // the seed IS the reset
    }

    // The flag travels with every traveling-band output.
    bool fictionAdvected = dilationFictionNow;

    // The accumulator's blend weight at clip time (motion only). The
    // clip-rejection and alignment drops raise the TRUE accumulator alpha,
    // biasing Var(h) low -- the tight, anti-ghost direction.
    float statAlpha = 1.0 - ComputeMotionFeedback(repro);

    // ---- the statistic gate -----------------------------------------------
    // Per-channel normalized, phase-corrected test vector, transient +
    // Student accounting, conservative-or-scoped mu share (taaClip).
    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, colorStats, gateSigmaSq, ageNext, statAlpha, repro.motionNormalized);
    float3 clippedHistorySpace = clipGate.clippedColorSpace;

    // ---- DEBUG PAYLOAD STASH (consolidated) ---------------------------------
    // All modes stash here, at the end of the pipeline, under ONE uniform
    // branch: previously twelve separate mode checks ran per pixel in
    // production and kept dbgCode/dbgA/dbgB live across the entire shader.
    // Values are identical to the scattered stashes (mode 6 reads the same
    // default-constructed zeros when the landing was not gathered). The two
    // deliberate debug-only deltas: the !historyValid early-out now packs
    // code 0 for every mode (border pixels uniformly show the resolved
    // color), and mode 1 on revoked dilation candidates shows the
    // post-revocation motion -- the motion actually used.
    if (debugActive)
    {
        if (taaDebugMode > 0.5 && taaDebugMode < 1.5)          // 1: frame motion
        {
            float2 vPx = abs(repro.motionPx) * kDebugVelocityScale;
            dbgCode = 1.0; dbgA = saturate(vPx.x); dbgB = saturate(vPx.y);
        }
        else if (taaDebugMode > 1.5 && taaDebugMode < 2.5)     // 2: disocclusion
        {
            float f = (depthRejected ? 4.0 : 0.0)
                    + (velocityRejection.rejected ? 2.0 : 0.0)
                    + ((velocityRejection.errorRatio > 1.0 && !velocityRejection.rejected) ? 1.0 : 0.0);
            dbgCode = 2.0; dbgA = f * 0.125;
        }
        else if (taaDebugMode > 2.5 && taaDebugMode < 3.5)     // 3: center velocity
        {
            float2 vPx = abs(centerVelocityJitteredUV * vp.sizePixels) * kDebugVelocityScale;
            dbgCode = 3.0; dbgA = saturate(vPx.x); dbgB = saturate(vPx.y);
        }
        else if (taaDebugMode > 3.5 && taaDebugMode < 4.5)     // 4: linearized depth
        {
            dbgCode = 4.0;
            dbgA = saturate(LinearizeDepth(centerDepthRaw) / kDebugLinearDepthRange);
        }
        else if (taaDebugMode > 4.5 && taaDebugMode < 5.5)     // 5: history color
        {
            dbgCode = 5.0;
        }
        else if (taaDebugMode > 5.5 && taaDebugMode < 6.5)     // 6: landing velocity
        {
            float2 vPx = abs(landing.effectiveVelocityJitteredPrevUV * vp.sizePixels) * (kDebugVelocityScale * 2.0);
            dbgCode = 6.0; dbgA = saturate(vPx.x); dbgB = saturate(vPx.y);
        }
        else if (taaDebugMode > 6.5 && taaDebugMode < 7.5)     // 7: pursuit divergence
        {
            dbgCode = 7.0;
            dbgA = saturate(velocityRejection.divergencePx * 0.5);
            dbgB = velocityRejection.rejected ? 1.0 : 0.0;
        }
        else if (taaDebugMode > 7.5 && taaDebugMode < 8.5)     // 8: layer state
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
            dbgCode = 8.0; dbgA = f * 0.125;
        }
        else if (taaDebugMode > 8.5 && taaDebugMode < 9.5)     // 9: dilation gate
        {
            float f = ((dilationCandidate && !dilationRevoked) ? 4.0 : 0.0)
                    + (gateViaFlag ? 2.0 : 0.0)
                    + (depthRejected ? 1.0 : 0.0);
            dbgCode = 9.0; dbgA = f * 0.125;
        }
        else if (taaDebugMode > 9.5 && taaDebugMode < 10.5)    // 10: alignment drop
        {
            float dropAmount = currentSingleLayer
                ? taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment)
                : 0.0;
            dbgCode = 10.0; dbgA = saturate(dropAmount); dbgB = currentSingleLayer ? 1.0 : 0.0;
        }
        else if (taaDebugMode > 10.5 && taaDebugMode < 11.5)   // 11: clip gate state
        {
            dbgCode = 11.0;
            dbgA = carryRecord ? 1.0 : (hadRecord ? 0.35 : 0.15);
            dbgB = 1.0 - clipGate.tGate;
        }
        else if (taaDebugMode > 11.5 && taaDebugMode < 12.5)   // 12: dejittered residual
        {
            float2 residualPx = (currentLayer.effectiveVelocityUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
            dbgCode = 12.0;
            dbgA = saturate(length(residualPx) * 0.5);
        }
    }

    float clipDistanceRejection = 0.0;
    if (taaClipDistanceRejectionEnabled > 0.5)
        clipDistanceRejection = ComputeClipDistanceRejection(
            clippedHistorySpace, historyColorSpace, colorStats);

    ApplyLumaDriftCorrection(clippedHistorySpace, currentColorSpace, colorStats);

    // ------------------------------------------------------------------
    // 12) Feedback (clip-distance responsive) & the temporal blend.
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
            float3 fxaaCornersRGB[4];
            FetchFxaaCorners(tapUVs, fxaaCornersRGB);
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 13) Final blend & output: RGB is ALWAYS the real resolved color.
    //     Alpha = the packed CLIP STATE (sign / tag 0110 / 7-bit state
    //     sigma / sqrt-compressed stabilized acutance / 7-bit record age /
    //     fiction flag).
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    // Clamp scalar luminance only; do NOT clamp signed chrominance channels.
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    float outAlpha = debugActive
        ? PackDebugAlpha(dilationRevoked, dbgCode, saturate(dbgA), saturate(dbgB), sqrt(sigmaStatSq))
        : PackClipStateAlpha(dilationRevoked, sqrt(sigmaStatSq), ageNext, acutanceStabLin, fictionAdvected);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}