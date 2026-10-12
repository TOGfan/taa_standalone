// ============================================================================
// TAA (Temporal Anti-Aliasing) post effect -- resolve pass (pipeline host)
// ----------------------------------------------------------------------------
// CONTEXT: sub-pixel jitter by PHYSICALLY ROTATING the camera; the per-frame
// bases (P/Q/R) contain the jitter; the velocity buffers contain the jitter
// motion; every jitter-free comparison subtracts s_t - 2*s_{t-1} + s_{t-2}.
// The history buffer stores the previous frame's STABLE output.
//
// PIPELINE (per output pixel):
//   1-3)   snap the stable position; gather the raw current-frame 3x3
//          (depth/velocity/color); classify the effective surface layer
//          (dilation zone / foreground edge / flat) and resolve its
//          effective depth + velocity. [GEOMETRY]
//   4)     reproject into the history through the previous camera's basis
//          with the EFFECTIVE velocity.
//   7-10)  exact jitter plumbing, the own-history dilation validation, the
//          history landing analysis, the depth/velocity disocclusion tests.
//          [GEOMETRY]
//   11)    THE FLAT MODEL (taaClip v5.3): the coupled-plane statistics,
//          the plain record, the structure split's transport, the level
//          gate (the hierarchical posterior + the soft clip), the blend,
//          the VarH recursion.
//
// OUTPUT: RGB = the resolved color. A = the packed CLIP STATE (tag 101).
// Debug modes carry tag 001.
//
// DEBUG MODES: 0 off | 1 frame motion | 2 disocclusion breakdown |
//   3 center velocity | 4 linearized depth | 5 history color | 6 landing
//   velocity | 7 pursuit divergence | 8 layer state | 9 dilation gate |
//   10 alignment drop | 11 clip gate state | 12 dejittered residual |
//   13 gate null law (calibration: dbgA = saturate(mdd/16)).
// ============================================================================

#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"
#include "shaders/common/postFx/taa/taaShared.h.hlsl"
#include "shaders/common/postFx/taa/taaConstants.h.hlsl"

uniform_sampler2D(sceneTex,         0);
uniform_sampler2D(depthTex,         1);
uniform_sampler2D(historyTex,       2);
uniform_sampler2D(velocityTex,      3);
uniform_sampler2D(historyMotionTex, 4);
uniform_sampler2D(historyStateTex, 5);

// ============================================================================
// CBUFFER (all constants are set BY NAME from client/postFx/taa.lua)
// ============================================================================
cbuffer perDraw
{
    float  taaFeedbackMin;                  float  taaFeedbackMax;
    float  taaVarianceGamma;                float  taaSoftClip;
    float  taaChromaVarianceMod;            float  taaDepthRejection;
    float  taaTanHalfFovX;                  float  taaTanHalfFovY;
    float  taaUseDepthDilation;             float  taaUseMotionField;
    float  taaLumaVariance;                 float  taaUseHullClipping;
    float  taaColorSpaceOklab;              float  taaJitterAwareVariance;
    float  taaVelocityAlignedVariance;      float  taaAlignmentFeedbackDrop;
    float  taaMotionBlendDropSpeed;         float  taaUseKaiser6;
    float  taaFallbackFXAA;
    float  taaMotionBlendStart;             float  taaDebugMode;
    float  taaCurPX;                        float  taaCurPY;
    float  taaCurPZ;                        float  taaCurQX;
    float  taaCurQY;                        float  taaCurQZ;
    float  taaCurRX;                        float  taaCurRY;
    float  taaCurRZ;                        float  taaPrevPX;
    float  taaPrevPY;                       float  taaPrevPZ;
    float  taaPrevQX;                       float  taaPrevQY;
    float  taaPrevQZ;                       float  taaPrevRX;
    float  taaPrevRY;                       float  taaPrevRZ;
    float  taaHistoryOvershoot;             float  taaClipOvershoot;
    float  taaVelRejection;                 float  taaVelGradientScale;
    float  taaDepthParallaxStep;            float  taaCrossTestStrength;
    float  taaJitPrev2YawSin;               float  taaJitPrev2YawCos;
    float  taaJitPrev2PitchSin;             float  taaJitPrev2PitchCos;
    float  taaStudentA;                     float  taaStudentB;
    float  taaStudentC;
    float  taaArmingA;                      float  taaArmingB;
    float  taaJitterPhaseEx;                float  taaJitterPhaseEy;
    float  taaJitterPhaseExy;
    float  taaVarhResampleLoss;             float  taaVarhWhiteLoss;

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

    bool  debugActive = (taaDebugMode > 0.5);
    float dbgCode = 0.0, dbgA = 0.0, dbgB = 0.0;

    // 1) The jittered render position.
    float2 currentJitteredUV = ReprojectThroughCamera(IN.uv0, currentCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
    float2 stableInFrameUV   = 2.0 * IN.uv0 - currentJitteredUV;
    SnappedCoord pixel       = SnapUVToTexel(stableInFrameUV, vp);

    float3 currentColorRGB          = max(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb, 0.0);
    float  centerDepthRaw           =       tex2Dlod(depthTex,    float4(pixel.snappedUV, 0.0, 0.0)).r;
    float3 currentColorSpace        = ToSpace(currentColorRGB);
    float2 centerVelocityJitteredUV =       tex2Dlod(velocityTex, float4(pixel.snappedUV, 0.0, 0.0)).rg;

    // 2) The raw 3x3 current-frame depth/velocity neighborhood.
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

    // 3) Classify and resolve the effective surface.
    SurfaceEdgeState edge = AnalyzeSurfaceEdges(neighborhood, useDepthDilation);

    LayerSurface currentLayer = ClassifyLayerSurface(
        neighborhood.depthRaw, neighborhood.velocityJitteredUV, pixel.fracPx,
        vp.sizePixels, coherenceRadiusPx, useDepthDilation, false, edge);

    float2 quadVelocitySpreadUV = float2(0.0, 0.0);
    if (depthTestActive)
    {
        float2 qv00, qv10, qv01, qv11;
        SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, qv00, qv10, qv01, qv11);
        quadVelocitySpreadUV = max(max(qv00, qv10), max(qv01, qv11))
                             - min(min(qv00, qv10), min(qv01, qv11));
    }

    ForegroundGeometry foreground;
    foreground.slope = 0.0;
    if (depthTestActive)
        foreground = ComputeForegroundGeometry(neighborhood, currentLayer.effectiveDepth, edge);

    // 4) Reproject into history with the EFFECTIVE velocity.
    HistoryReprojection repro = ReprojectToHistory(
        IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);

    // 5) Color neighborhood gather + FXAA corners.
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    [unroll]
    for (int i = 1; i < 8; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
    }
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[8], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[8] = ToSpace(tapRGB);
    }

    // 6) Validate the history landing position.
    float  historySupportTexels = (taaUseKaiser6 > 0.5) ? 3.0 : 2.0;
    float2 historyMinUV = historySupportTexels * vp.texelSize;
    float2 historyMaxUV = 1.0 - historyMinUV;
    bool historyValid = all(repro.sampleUV >= historyMinUV) && all(repro.sampleUV <= historyMaxUV);

    if (!historyValid)
    {
        bool  earlyRevoked = currentLayer.isDilationZone;
        float earlyAlpha  = debugActive
            ? PackDebugAlpha(earlyRevoked, dbgCode, saturate(dbgA), saturate(dbgB), 0.0)
            : PackClipStateAlpha(earlyRevoked, 0.0, 1820.0, 0.0, 32.0);

        if (fxaaEnabled)
        {
            float3 fxaaCornersRGB[4];
            FetchFxaaCorners(tapUVs, fxaaCornersRGB);
            float3 fxaaColorRGB = ApplyFXAA(pixel.snappedUV, vp.texelSize, currentColorRGB, fxaaCornersRGB, vp.minUV, vp.maxUV);
            return float4(max(fxaaColorRGB, 0.0), earlyAlpha);
        }
        return float4(currentColorRGB, earlyAlpha);
    }

    // 7) Exact jitter plumbing.
    float2 jitterOffsetCurUV   = currentJitteredUV - IN.uv0;
    float2 jitterOffsetPrevUV  = ReprojectThroughCamera(IN.uv0, previousCamera, IN.uv0, taaTanHalfFovX, taaTanHalfFovY) - IN.uv0;
    float2 jitterOffsetPrev2UV = RotationFlowUV(
        float4(taaJitPrev2YawSin, taaJitPrev2YawCos, taaJitPrev2PitchSin, taaJitPrev2PitchCos),
        IN.uv0);
    float2 jitterCancelUV     = EstimateJitterCancelUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);
    float2 jitterTransportUV  = EstimateJitterTransportUV(jitterOffsetCurUV, jitterOffsetPrevUV, jitterOffsetPrev2UV);

    // 8) Own-history dilation validation.
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

        if (!quadTouchesForeground)
        {
            dilationRevoked = true;
            RevokeDilation(currentLayer, centerDepthRaw, centerVelocityJitteredUV);

            repro = ReprojectToHistory(
                IN.uv0, stableInFrameUV, pixel.fracPx, currentLayer.effectiveVelocityUV, previousCamera, vp);
        }
    }

    // 9) History landing analysis.
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

    bool currentSingleLayer = !(currentLayer.isDilationZone || currentLayer.isForegroundEdge);

    // 10) Disocclusion tests.
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
    // 11) THE FLAT MODEL: the state, the statistics, the split, the
    //     record, the gate.
    // ------------------------------------------------------------------
    bool  temporalStateEnabled = (taaUseHullClipping > 0.5);
    float sigmaPrevSq = -1.0;
    float varHPrevSq  = 0.0;
    float agePrev     = 0.0;
    float splitPrev   = 0.0;

    if (temporalStateEnabled)
    {
        float2 stateUV = SnapUVToTexel(repro.sampleUV, vp).snappedUV;
        DecodeClipState(tex2Dlod(historyStateTex, float4(stateUV, 0.0, 0.0)).a,
                        sigmaPrevSq, varHPrevSq, agePrev, splitPrev);
    }

    bool hadRecord = (sigmaPrevSq >= 0.0);

    // The uniform-velocity pre-test: the transport floor's cheap gate (a
    // uniform 3x3 provably measures the quantization floor).
    bool velUniform3x3 = true;
    [unroll]
    for (int vu = 1; vu < 9; ++vu)
        velUniform3x3 = velUniform3x3
                     && all(neighborhood.velocityJitteredUV[vu] == centerVelocityJitteredUV);

    // The transport scatter's static floor is MEASURED -- the velocity
    // buffer's own quantization/noise scale (the smallest nonzero
    // pairwise step of the 3x3).
    float transportFloorPx = 0.0;
    if (needNeighbors && repro.motionMagnitudePx < kTransportGatePx)
        transportFloorPx = velUniform3x3 ? kVelQuantFloorPx
                                         : MeasureVelocityQuantStepPx(neighborhood.velocityJitteredUV, vp.sizePixels);

    // The gated landing-error vector: the transport scatter's input.
    float2 transportErrPx;
    {
        float transportMotionPx = saturate(repro.motionNormalized) * kMotionFullStrengthPx;
        float motionShare        = saturate(transportMotionPx / kTransportGatePx);
        transportErrPx.x = max(abs(repro.jitterResidualPx.x) * motionShare, transportFloorPx);
        transportErrPx.y = max(abs(repro.jitterResidualPx.y) * motionShare, transportFloorPx);
    }

    // ---- the statistics (the coupled 2x2 plane; no masks, no flags) ----
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized,
        pixel.fracPx, transportErrPx);

    // ---- the record's split (PRE-update; the model's own composition) ----
    float nuPrev = StudentEffectiveDof(agePrev, kRecordIngestDof);
    RecordVarianceParts partsLive;
    partsLive.ePer           = float3(0.0, 0.0, 0.0);
    partsLive.varHPer        = float3(0.0, 0.0, 0.0);
    partsLive.sPer           = float3(0.0, 0.0, 0.0);
    partsLive.spatialPriorSq = 0.0;
    if (hadRecord)
        partsLive = SplitRecordVariance(colorStats, sigmaPrevSq, varHPrevSq, nuPrev);

    // ---- the history read (Kaiser) ----
    float3 historyColorSpace;
    if (taaUseKaiser6 > 0.5)
        historyColorSpace = SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    else
        historyColorSpace = SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);

    if (!(dot(historyColorSpace, historyColorSpace) < kLargeValue))
        historyColorSpace = colorStats.plane0;

    if (taaHistoryOvershoot > 0.001 || taaColorSpaceOklab > 0.5)
        historyColorSpace = CompressGamut(historyColorSpace);

    // ---- the innovation (the stream the record and the spike price) ----
    float3 innov   = currentColorSpace - historyColorSpace;
    float  innovSq = dot(innov, innov);

    // ---- change-points ----
    float spikeRefSq = max(sigmaPrevSq, colorStats.rangeSq)
                     + dot(colorStats.transportVar, float3(1.0, 1.0, 1.0));
    bool  spike   = hadRecord
        && (innovSq > kClipSpikeRatio * spikeRefSq);

    bool geoCorroborated = false;
    if (hadRecord && disoccluded)
    {
        float rXYg, rXZg, rYZg;
        ShrunkenCorrelations(colorStats, rXYg, rXZg, rYZg);
        geoCorroborated = (WhitenedInnovSq(innov, partsLive, colorStats.transportVar, rXYg, rXZg, rYZg)
                           > kGeoCorroborateChiSq);
    }

    bool  resetRecord = hadRecord
        && (spike || !(innovSq < kLargeValue) || geoCorroborated);
    bool  carryRecord = hadRecord && !resetRecord;

    // ---- THE STRUCTURE SPLIT'S TRANSPORT (v5.3) ----
    // The signed normalized fraction f = (R - s)/max(R, s): the spatial
    // residual total against the record's per-sample total. EMA'd, 6-bit
    // packed; the gate reconstructs tauB2 = max(f,0)*residSq. The clamp
    // lives on the STABLE side (see taaClip's header). A fresh record
    // certifies nothing (the seed >= R by construction, so f <= 0 cold);
    // a disocclusion restarts the split with the new content.
    float splitFracNext;
    {
        float residTotal = dot(colorStats.residSq, float3(1.0, 1.0, 1.0));
        float sPerTotal  = hadRecord
            ? max(sigmaPrevSq - varHPrevSq, kClipSigmaRecordFloorSq)
            : max(SeedRecordTotalSq(colorStats), kClipSigmaRecordFloorSq);
        float fRaw = clamp((residTotal - sPerTotal)
                           / max(max(residTotal, sPerTotal), kEpsilon), -1.0, 1.0);
        splitFracNext = (carryRecord && !disoccluded)
            ? lerp(splitPrev, fRaw, kSplitEmaRate)
            : fRaw;
    }

    // ---- the record: the plain single-sample ingest ----
    float winsorCap  = kClipWinsorC * kClipWinsorC
                     * max(partsLive.spatialPriorSq, kClipSigmaRecordFloorSq);

    float sigmaStatSq, ageNext;
    if (carryRecord)
    {
        sigmaStatSq = max(lerp(sigmaPrevSq, min(innovSq, winsorCap), kClipSigmaEmaRate), kClipSigmaRecordFloorSq);
        ageNext     = min(agePrev + 1.0, kClipMaxAge);
    }
    else
    {
        sigmaStatSq = max(SeedRecordTotalSq(colorStats), kClipSigmaRecordFloorSq);
        ageNext     = 0.0;
    }

    // ---- THE LEVEL GATE ----
    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, currentColorSpace, colorStats, partsLive,
        carryRecord, carryRecord ? agePrev : 0.0,
        splitPrev,
        repro.motionNormalized,
        neighborhoodColorSpace);
    float3 clippedHistorySpace = clipGate.clippedColorSpace;

    // ---- DEBUG PAYLOAD STASH ----------------------------------------------
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
        else if (taaDebugMode > 12.5 && taaDebugMode < 13.5)   // 13: gate null law
        {
            float chiDbg = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));
            dbgCode = 13.0;
            dbgA = saturate(clipGate.mdd * (1.0 / 16.0));
            dbgB = (clipGate.mdd > chiDbg * chiDbg) ? 1.0 : 0.0;
        }
    }

    // ------------------------------------------------------------------
    // 12) Feedback & the temporal blend.
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, currentSingleLayer);
    float currentBlendWeight = 1.0 - historyFeedback;

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
    // 13) The blend, the VarH recursion, the packed output.
    // ------------------------------------------------------------------
    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);

    // ---- the VarH recursion (READ semantics; the host-measured gains;
    //      12-bit transport -- the quantization is no longer an error
    //      source) ----
    float varHCodeNext = 1820.0;
    {
        float aBlend = currentBlendWeight;
        float dampGain  = 1.0 - saturate(taaVarhResampleLoss);
        float freshGain = 1.0 - saturate(taaVarhWhiteLoss);
        if (carryRecord)
        {
            float varHPrevCapped = dot(partsLive.varHPer, float3(1.0, 1.0, 1.0));
            float vxFresh  = max(sigmaStatSq - varHPrevCapped, 0.0);
            float varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * varHPrevCapped
                           + aBlend * aBlend * freshGain * vxFresh;
            varHCodeNext = EncodeVarHRatio(varHNext, max(sigmaStatSq, kEpsilon));
        }
        else
        {
            float3 dCorrSeed = historyColorSpace - clipGate.center;
            float  tC = clipGate.tGate;
            float3 varH0Per = clipGate.rawGateVar + (1.0 - tC) * (1.0 - tC) * (dCorrSeed * dCorrSeed);
            float  varH0 = varH0Per.x + varH0Per.y + varH0Per.z;
            float  s0 = max(sigmaStatSq - varH0, 0.0);
            float  varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * varH0
                            + aBlend * aBlend * freshGain * s0;
            varHCodeNext = EncodeVarHRatio(varHNext, max(sigmaStatSq, kEpsilon));
        }
    }

    // ---- the packed output ----
    float splitCodeNext = clamp(round((splitFracNext + 1.0) * 32.0), 0.0, 63.0);
    float outAlpha = debugActive
        ? PackDebugAlpha(dilationRevoked, dbgCode, saturate(dbgA), saturate(dbgB), sqrt(sigmaStatSq))
        : PackClipStateAlpha(dilationRevoked, sqrt(sigmaStatSq), varHCodeNext, ageNext, splitCodeNext);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}