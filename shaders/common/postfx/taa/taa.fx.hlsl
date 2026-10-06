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
// HISTORY CLIPPING: the Mahalanobis statistic gate (taaClip.h.hlsl), with
// the exact estimation-uncertainty accounting and the consumer-split
// innovation:
//     raw  i = x(S + u) - h  -> the variance path (the record E, the gate
//          scale: the accumulator carries the full phase spread; u is the
//          sample's snap offset, u = -pixel.fracPx)
//     corr i_corr = i - beta*u  -> the mean-structure path (the
//          anti-alignment; the accumulator field is PHASE-FREE, so the
//          phase bit is the sample's alone -- landings between pixels
//          enter as the resample bit the record absorbs, first-class)
//     gate_var = cStat(t) * S(nu(t)) * E   (live; the transient + the
//          host-fit Student inflation, slider-coupled)
//     cold     = invNeff * sigma_c^2       (the change-point replacement)
//     seed     = tr(Cov_taps)              (the new regime's MEASURED
//          spread -- never the change-point's own squared residual, which
//          re-armed the gate to the ghost's scale: the faint sky ghost)
// plus the anti-alignment reset, corroborated at chi^2_3(0.80) of the
// spatial trace. Mixtures are content: the stats run on the FULL tap set,
// always.
//
// OUTPUT: RGB = the resolved color -- ALWAYS, also while a debug mode is
// active. A = the packed CLIP STATE: [31] the revocation sign, [30:27] tag
// 0110, [26:20] the state sigma code (0 = cold), [19:8] the 12-bit acutance
// energy, [7:1] the record's age, [0] the fiction flag. While a debug mode
// is active, A carries the packed debug payload (tag 0111) with the sigma
// embedded in its low 7 bits.
//
// PIPELINE (mainP): (1) resolve the jittered render position; (2) gather
// the raw current-frame neighborhood; (3) classify the effective surface;
// (4) reproject into history; (5) gather the color neighborhood + FXAA
// corners + acutance; (6) validate the history landing position; (7) exact
// jitter plumbing; (8) own-history dilation validation; (9) history landing
// analysis; (10) disocclusion tests; (11) color stats (full tap set), the
// temporal clip-state read/reset/advance (sigma + age + fiction), the
// anti-alignment test, the statistic gate; (12) feedback and the temporal
// blend; (13) output.
//
// DEBUG MODES (taaDebugMode): rendered by taaFinal from the alpha payload.
//   0 off | 1 frame motion | 2 disocclusion breakdown | 3 center velocity |
//   4 linearized depth | 5 history color | 6 landing effective velocity |
//   7 pursuit divergence | 8 layer state | 9 dilation-gate breakdown |
//   10 alignment-drop activity | 11 clip gate state (G = record carried,
//   half-G = reset this frame, dim-R = no record, B = applied shrink) |
//   12 dejittered residual.
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
    // The Studentization fit, computed BY THE HOST from the effective
    // radius (studentFitAB in client/postFx/taa.lua):
    //     S(nu) = 1 + taaStudentA/nu + taaStudentB/nu^2
    // exact at nu = 4 (age 0) and nu = 12.33 (converged); see taaClip.
    // Unset constants read 0: S = 1 (the pre-Student gate, graceful).
    float  taaStudentA;                     float  taaStudentB;

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
    foreground.crestDrop = 0.0;
    foreground.slope     = 0.0;
    if (depthTestActive)
        foreground = ComputeForegroundGeometry(neighborhood, currentLayer.effectiveDepth, edge);

    // ------------------------------------------------------------------
    // 4) Reproject into the history buffer with the EFFECTIVE velocity.
    // ------------------------------------------------------------------
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);

    // Mode 1 (frame motion).
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
        bool  earlyRevoked = currentLayer.isDilationZone;
        float earlyAlpha  = debugActive
            ? PackDebugAlpha(earlyRevoked, dbgCode, saturate(dbgA), saturate(dbgB), 0.0)
            : PackClipStateAlpha(earlyRevoked, 0.0, 0.0, rawSharpnessEnergy, false);

        if (fxaaEnabled)
        {
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), earlyAlpha);
        }
        return float4(currentColorRGB, earlyAlpha);
    }

    // ------------------------------------------------------------------
    // 7) Exact jitter plumbing. jitterOffsetCurUV is the per-pixel jitter
    //    offset s_t (the velocity tests' cancel term). The PHASE
    //    REGRESSION does NOT use it -- it uses the sample's snap offset,
    //    pixel.fracPx (step 11); the accumulator field is phase-free.
    // ------------------------------------------------------------------
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterOffsetPrev2UV = RotationFlowUV(taaJitPrev2Yaw, taaJitPrev2Pitch, IN.uv0);
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

    // Mode 6 (landing effective velocity).
    if (taaDebugMode > 5.5 && taaDebugMode < 6.5)
    {
        float2 vPx = abs(landing.effectiveVelocityJitteredPrevUV * vp.sizePixels) * (kDebugVelocityScale * 2.0);
        dbgCode = 6.0;
        dbgA    = saturate(vPx.x);
        dbgB    = saturate(vPx.y);
    }

    // Post-revocation single-layer state (a revoked candidate acts flat).
    bool currentSingleLayer = !(currentLayer.isDilationZone || currentLayer.isForegroundEdge);

    // Mode 8 (layer state + straddle).
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
    // 10) Disocclusion tests (require the stored motion field). With the
    //     motion field disabled both are skipped and disocclusion relies
    //     on the color side alone (the statistic gate + the change-point
    //     guards in step 11).
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

    // Mode 2 (disocclusion breakdown).
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

    // Mode 9 (dilation-gate breakdown).
    if (taaDebugMode > 8.5 && taaDebugMode < 9.5)
    {
        float f = ((dilationCandidate && !dilationRevoked) ? 4.0 : 0.0)
                + (gateViaFlag ? 2.0 : 0.0)
                + (depthRejected ? 1.0 : 0.0);
        dbgCode = 9.0;
        dbgA    = f * 0.125;
    }

    // ------------------------------------------------------------------
    // 11) Color stats (full tap set), the temporal clip state (sigma +
    //     age + fiction), the anti-alignment, the statistic gate.
    // ------------------------------------------------------------------
    // The stats run on the FULL tap set, always: an anti-aliased texel IS
    // a coverage mixture, and the mixture is the sub-texel signal the
    // record is supposed to measure. The phase bit's regressor is the
    // SAMPLE's snap offset (pixel.fracPx, negated inside the stats): the
    // accumulator field is phase-free, so the corrected innovation is
    // null-clean for any landing, and the landing's own fraction enters
    // as the resample bit the record absorbs (see the taaClip header).
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized,
        pixel.fracPx);

    float3 historyColorSpace =
        (taaUseKaiser6 > 0.5)
        ? SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV)
        : SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);

    // NaN guard: a non-finite history value would sail THROUGH the gate and
    // permanently poison the accumulation. Collapse to the neighborhood
    // mean; the record side is covered by the innovSq reset below.
    if (!(dot(historyColorSpace, historyColorSpace) >= 0.0))
        historyColorSpace = colorStats.mean;

    // Gamut recompression: required with resampling-overshoot margin, and
    // in OKLAB mode ALWAYS (Oklab is nonlinear -- convexity is a
    // linear-space argument).
    if (taaHistoryOvershoot > 0.001 || taaColorSpaceOklab > 0.5)
        historyColorSpace = CompressGamut(historyColorSpace);

    // Mode 5 (history color) carries no numeric payload.
    if (taaDebugMode > 4.5 && taaDebugMode < 5.5)
    {
        dbgCode = 5.0;
    }

    // ---- temporal clip state: read, reset, advance -----------------------
    // The previous stable output's packed alpha, read AT THE LANDING through
    // the DEDICATED POINT-SAMPLED binding (slot 5). The state is sigma
    // (E[i^2]), the record's AGE (the Student dof and the accumulator
    // transient) and the fiction flag.
    bool  temporalStateEnabled = (taaUseHullClipping > 0.5);
    float sigmaPrevSq = -1.0;
    float agePrev     = 0.0;
    bool  fictionPrev = false;
    if (temporalStateEnabled)
    {
        float2 stateUV = SnapUVToTexel(repro.sampleUV, vp).snappedUV;
        DecodeClipState(tex2Dlod(historyStateTex, float4(stateUV, 0.0, 0.0)).a,
                        sigmaPrevSq, agePrev, fictionPrev);
    }

    // This frame's own fiction state: a TRAVELING dilation band (dilation
    // zone with real motion), DIRECTION-BLIND. Static dilation -- a thin
    // fence at rest -- is NOT fiction. Both sides of a traveling band are
    // the DILATION's to own; the clip never distinguishes them.
    bool dilationFictionNow = currentLayer.isDilationZone
                            && (repro.motionMagnitudePx > 0.5);

    // FICTION-ADVected record (traveling-band writer) = foreign content:
    // killed unconditionally by every non-band reader (the sentinel gate).
    if (fictionPrev && !dilationFictionNow)
    {
        sigmaPrevSq = -1.0;
    }

    // The innovation against the stored accumulator, PRE-clip, and its
    // PHASE-CORRECTED form. The RAW second moment feeds the record (the
    // accumulator carries the full phase oscillation -- the sub-texel and
    // resampler-footprint room); the CORRECTED form (the phase bit
    // subtracted: beta * u, u = -fracPx) feeds the mean-structure test
    // only.
    float3 innov       = currentColorSpace - historyColorSpace;
    float  innovSq     = dot(innov, innov);
    float3 innovCorr   = innov - colorStats.phaseShift;
    float  innovCorrSq = dot(innovCorr, innovCorr);

    // The spatial prior: the taps' TOTAL variance tr(Cov) -- the
    // anti-alignment's corroboration reference, the re-seed, and the
    // change-point tests' scale (NEVER the record: the record is exactly
    // what a ghost arms).
    float spatialSigmaSq = dot(colorStats.sigma, colorStats.sigma);
    float spatialScaleSq = max(spatialSigmaSq, kClipSigmaRecordFloorSq);
    bool  hadRecord      = (sigmaPrevSq >= 0.0);
    bool  innovFinite    = (innovSq < kLargeValue);

    // (a) THE ANTI-ALIGNMENT RESET (the ghost signature; see taaClip): the
    //     mean lies BETWEEN the current sample and the history; the
    //     PHASE-CLEAN corrected innovation corroborates against the
    //     neighborhood's trace at chi^2_3(0.80); the history is
    //     meaningfully outside; and DILATION ZONES are exempt -- the band
    //     is the silhouette's feather, both sides; the clip's domain
    //     begins beyond it. A vacated texel still carrying object color
    //     (the ghost past the feather) fires this and is reset in one
    //     frame.
    // (b) THE SHOCK: innovation > kClipSpikeRatio times BOTH the record
    //     and the neighborhood's squared AABB range.
    // (c) GEOMETRY AS EVIDENCE, NOT COMMAND: a geometric reject resets only
    //     when the color side corroborates (chi^2_3 at 95%).
    float3 dAlign        = historyColorSpace - colorStats.mean;
    float  dAlignSq      = dot(dAlign, dAlign);
    bool   antiAligned   = dot(innovCorr, dAlign)
                           < -kAlignCos * sqrt(innovCorrSq * dAlignSq);
    bool   farEnough     = dAlignSq > kAlignDist * kAlignDist * spatialScaleSq;
    bool   corroborated  = innovCorrSq > kAlignSpatial * spatialScaleSq;
    bool   antiAlignReset = hadRecord && !currentLayer.isDilationZone
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
    // fade. THE SEED IS THE SPATIAL PRIOR: a change-point's first squared
    // residual is a draw from the OLD error, not the new regime's spread --
    // seeding E with it armed the gate to the change-point's own magnitude
    // and left the post-reset remnant riding unclipped at blend speed (the
    // faint, clip-shaped ghost). The spatial prior is the new regime's
    // MEASURED spread, and the Student factor (nu = 4 at age 0) prices its
    // own uncertainty. The AGE resets with the seed (the dof and the
    // accumulator transient restart).
    //
    // DILATION-FICTION FREEZE (traveling bands): the innovation is
    // borrowed-content fiction -- never ingested, the record and its age
    // frozen, the flag written.
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

    // The flag travels with every traveling-band output (the reader rule
    // exempts band readers, so this marks the wake, nothing else).
    bool fictionAdvected = dilationFictionNow;

    // The accumulator's blend weight at clip time (motion only). The
    // clip-rejection and alignment drops raise the TRUE accumulator alpha,
    // biasing Var(h) low -- the tight, anti-ghost direction (the mu share
    // does not depend on it).
    float statAlpha = 1.0 - ComputeMotionFeedback(repro);

    // ---- the statistic gate ------------------------------------------------
    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, colorStats, gateSigmaSq, ageNext, statAlpha, repro.motionNormalized);
    float3 clippedHistorySpace = clipGate.clippedColorSpace;

    // Mode 11 (clip gate state): A = 1.0 carried / 0.35 reset this frame
    // (anti-alignment, shock, corroborated geometry, NaN) / 0.15 no record;
    // B = applied shrink. The ghost-past-the-feather kill reads as a
    // one-frame 0.35 flash just beyond the band's trailing edge, then
    // quiet; the post-reset gate re-warms from the spatial scale.
    if (taaDebugMode > 10.5 && taaDebugMode < 11.5)
    {
        dbgCode = 11.0;
        dbgA    = carryRecord ? 1.0 : (hadRecord ? 0.35 : 0.15);
        dbgB    = 1.0 - clipGate.tGate;
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
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            currentFrameColorSpace = lerp(currentFrameColorSpace, ToSpace(fxaaColorRGB), fxaaWeight);
        }
    }

    // ------------------------------------------------------------------
    // 13) Final blend & output: RGB is ALWAYS the real resolved color.
    //     Alpha = the packed CLIP STATE (sign / tag 0110 / 7-bit state
    //     sigma / 12-bit acutance / 7-bit record age / fiction flag).
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    // Clamp scalar luminance only; do NOT clamp signed chrominance channels.
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    float outAlpha = debugActive
        ? PackDebugAlpha(dilationRevoked, dbgCode, saturate(dbgA), saturate(dbgB), sqrt(sigmaStatSq))
        : PackClipStateAlpha(dilationRevoked, sqrt(sigmaStatSq), ageNext, rawSharpnessEnergy, fictionAdvected);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}