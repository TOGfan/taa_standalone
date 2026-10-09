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
//          effective depth + velocity.
//   4)     reproject into the history through the previous camera's basis
//          with the EFFECTIVE velocity.
//   7-8)   exact jitter plumbing (cancel/transport identities) and the
//          own-history dilation validation through the stored motion
//          field.
//   9-10)  history landing analysis and the depth/velocity disocclusion
//          tests.
//   11)    the statistical core: the 9-tap spatial record/drift estimator
//          (same-layer mask minus the stored edge flags), the drift
//          predict step, the persistence detector, the record update and
//          the Mahalanobis gate (taaClip.h.hlsl).
//   12-13) feedback, the temporal blend, the VarH transport recursion and
//          the packed output.
//
// THE 9-TAP SPATIAL ESTIMATOR consumes the full 3x3 same-layer subset,
// weighted by the kernel weights (which downweight the diagonals and
// thereby bound their ~2x larger parallax residual in the variance). The
// LS plane fit uses the COUPLED 2x2 normal equations -- with the diagonals
// present, sum(w*x*y) is nonzero and a decoupled per-axis fit is biased.
// The estimator exports the LS phase term (gS*u) so the innovation's phase
// correction uses the same, ~2.7x tighter gradient estimate.
//
// OUTPUT: RGB = the resolved color, ALWAYS. A = the packed CLIP STATE
// (tag 101). Debug modes carry tag 001.
//
// REQUIRES: HistoryLandingSurface.tapBandFlag[9] (taaDisocclusion.h.hlsl)
// -- the per-tap w channel of the landing 3x3 motion-field gather.
//
// DEBUG MODES: 0 off | 1 frame motion | 2 disocclusion breakdown |
//   3 center velocity | 4 linearized depth | 5 history color | 6 landing
//   velocity | 7 pursuit divergence | 8 layer state | 9 dilation gate |
//   10 alignment drop | 11 clip gate state (telemetry: dbgB = the fitted
//   coverage on engaged steps, |T|/alarm elsewhere) | 12 dejittered residual
//   | 13 gate null law (calibration: dbgA = saturate(mdd/16); park on a
//   static scene -- the nominal 95th percentile sits at chi^2/16; dbgB =
//   gate engagement).
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
    float  taaDepthParallaxStep;            float  taaCrossTestStrength;
    float  taaJitPrev2YawSin;               float  taaJitPrev2YawCos;
    float  taaJitPrev2PitchSin;            float  taaJitPrev2PitchCos;
    float  taaStudentA;                     float  taaStudentB;
    float  taaClipScopedMu;
    float  taaAcutanceActive;               float  taaDriftCompensation;
    float  taaDriftMaxGain;
    float  taaJitterPhaseEx;                float  taaJitterPhaseEy;
    float  taaJitterPhaseExy;
    float  taaClipGhostReset;
    float  taaCusumFlipAcc;                 float  taaCusumNoiseAcc;
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
// THE DRIFT-IMMUNE RECORD INGEST: the spatial second difference
// ----------------------------------------------------------------------------
// NINE taps: the center + the full 3x3 same-layer subset, weighted by the
// kernel weights (which downweight the diagonals and thereby bound their
// ~2x larger parallax residual in the variance). The LS plane fit uses
// the COUPLED 2x2 normal equations -- with the diagonals present,
// sum(w*x*y) is nonzero and a decoupled per-axis fit is biased. The
// estimator also exports the LS phase term (gS*u) so the innovation's
// phase correction uses the same, ~2.7x tighter gradient estimate.
//
// The common mode iMean = delta - u_t*grad (the taps carry the current
// jitter phase; the accumulated history sits at the phase average) -- the
// exported driftMean is PHASE-CORRECTED. The drift estimate's variance
// carries every noise source it actually has: the weighted mean's noise
// (floored at the record's honesty floor), the motion-gated transport
// scatter, and the phase correction's own error from the exact LS
// covariance (sigma^2 * [u]^T N^{-1} [u] with N the weighted second-moment
// matrix).
//
// THE EDGE-FLAG TAP MASK (historyTapIsBand): a same-depth neighbor whose
// stored flag marks it an edge texel carries the foreground-edge band's
// accumulation in its history (that is the band's job) -- its (current -
// history) mismatch is the BAND's state, not this texel's. Left in the
// estimator it inflates varSpatial (the record's spatial seed AND the
// drift estimate's noise) and biases iMean by the neighbor's trail: the
// persistence detector's t-statistic drops ~2x on the wake pixels behind
// a moving foreground, stretching the very trail it exists to evict. The
// CENTER tap is never masked: its mismatch IS the signal.
//
// The reductions run in ONE accumulation pass with closed-form finishes
// (the mean-subtracted identities below); the per-tap values stay loop
// locals -- no iVal/wVal/used arrays.
// ============================================================================
void EstimateSpatialRecord(
    float3 tapsSpace[9],
    float2 landingUV,
    float2 jitterPx,               // pixel.fracPx -- the SAME value the stats consume
    ViewportParams vp,
    bool sameLayerMask[9],
    bool layerMaskValid,
    bool historyTapIsBand[9],      // the stored-flag edge mask (see mainP)
    ColorNeighborhoodStats stats,
    out float recordSq,            // -1.0 = invalid (the legacy ingest runs)
    out float3 driftMean,          // the PHASE-CORRECTED local drift estimate
    out float3 driftVar,           // the drift estimate's noise+scatter variance
    out float3 phaseLS)            // gS*u -- the unified phase correction term
{
    recordSq  = -1.0;
    driftMean = float3(0.0, 0.0, 0.0);
    driftVar  = float3(0.0, 0.0, 0.0);
    phaseLS   = float3(0.0, 0.0, 0.0);

    if (!layerMaskValid) return;

    // ONE accumulation pass. The varSpatial and coupled-plane reductions
    // below are closed forms of these moments:
    //   sum w*(i - iMean)^2 = sum w*i^2 - (sum w*i)^2 / W
    //   sum w*ox*(c - cMean) = sum w*ox*c - cMean * sum w*ox
    // (the mean-subtracted identities; the raw second moments feed the
    // coupled normal equations directly). The cancellation is bounded:
    // the closed forms lose 1-2 digits against mean-subtracted loops when
    // the common mode dominates, leaving 5-6 significant digits -- far
    // beneath the record transport's 6-bit sigma quantization and every
    // consumer's scale.
    float  wSum = 0.0, wSq = 0.0, usedF = 0.0;
    float  mXX = 0.0, mYY = 0.0, mXY = 0.0, mX = 0.0, mY = 0.0;
    float3 iSum  = float3(0.0, 0.0, 0.0);
    float3 iSum2 = float3(0.0, 0.0, 0.0);
    float3 cSum  = float3(0.0, 0.0, 0.0);
    float3 cXSum = float3(0.0, 0.0, 0.0);
    float3 cYSum = float3(0.0, 0.0, 0.0);

    [unroll]
    for (int t = 0; t < 9; ++t)
    {
        if ((t == 0) || (sameLayerMask[t] && !historyTapIsBand[t]))
        {
            float3 hRGB = max(tex2Dlod(historyTex, float4(landingUV + kOffsets3x3[t] * vp.texelSize, 0.0, 0.0)).rgb, 0.0);
            float3 iV   = tapsSpace[t] - ToSpace(hRGB);
            float  wv   = stats.mixWeights[t];
            float2 o    = kOffsets3x3[t];
            wSum  += wv;
            wSq   += wv * wv;
            iSum  += iV * wv;
            iSum2 += (iV * iV) * wv;
            cSum  += tapsSpace[t] * wv;
            cXSum += o.x * tapsSpace[t] * wv;
            cYSum += o.y * tapsSpace[t] * wv;
            mXX += wv * o.x * o.x;
            mYY += wv * o.y * o.y;
            mXY += wv * o.x * o.y;
            mX  += wv * o.x;
            mY  += wv * o.y;
            usedF += 1.0;
        }
    }
    if (usedF < 2.5 || wSum < 1e-6) return;

    float  invW  = 1.0 / wSum;
    float3 iMean = iSum * invW;

    float invNeffSub = wSq * invW * invW;
    float dofCorr = 1.0 / max(1.0 - invNeffSub, 0.2);
    float3 varSpatial = max((iSum2 - iSum * iSum * invW) * invW * dofCorr, float3(0.0, 0.0, 0.0));

    // THE COUPLED 2x2 LS PLANE (the diagonals make mXY nonzero).
    float3 cMean  = cSum * invW;
    float3 gSxNum = cXSum - cMean * mX;
    float3 gSyNum = cYSum - cMean * mY;
    float detN = mXX * mYY - mXY * mXY;
    float3 gSx, gSy;
    if (detN > 1e-6)
    {
        gSx = (mYY * gSxNum - mXY * gSyNum) / detN;
        gSy = (mXX * gSyNum - mXY * gSxNum) / detN;
    }
    else
    {
        // degenerate subset (no 2D spread): the decoupled fit, exact when
        // mXY = 0 and the spread is 1D.
        gSx = gSxNum / max(mXX, 1e-4);
        gSy = gSyNum / max(mYY, 1e-4);
    }

    // The record's phase energy, the full quadratic form.
    float3 phaseSq = taaJitterPhaseEx  * (gSx * gSx)
                   + taaJitterPhaseEy  * (gSy * gSy)
                   + 2.0 * taaJitterPhaseExy * (gSx * gSy);

    recordSq  = dot(varSpatial, float3(1.0, 1.0, 1.0)) + dot(phaseSq, float3(1.0, 1.0, 1.0));

    // The phase-corrected drift (mirrors stats.phaseShift's convention
    // exactly: the raw common mode is delta - u*grad, corrected by
    // ADDING gS*u).
    driftMean = iMean + gSx * jitterPx.x + gSy * jitterPx.y;
    phaseLS   = gSx * jitterPx.x + gSy * jitterPx.y;

    // The drift estimate's variance. The phase-correction error:
    // Var(u_x*eps_gx + u_y*eps_gy) = varSpatial * [u]^T N^{-1} [u], with
    // N = [mXX mXY; mXY mYY] the weighted second-moment matrix (N is PSD,
    // det > 0 on the non-degenerate branch). Capped at 4.0 to keep
    // adversarial subsets finite.
    float3 varFloor = float3(kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq, kClipSigmaRecordFloorSq);
    float phaseErrScale = (jitterPx.x * jitterPx.x * mYY
                         + jitterPx.y * jitterPx.y * mXX
                         - 2.0 * jitterPx.x * jitterPx.y * mXY) / max(detN, 1e-6);
    phaseErrScale = min(max(phaseErrScale, 0.0), 4.0);
    driftVar = max(varSpatial, varFloor) * invNeffSub
             + stats.transportVar
             + phaseErrScale * varSpatial;
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

    // 5) Color neighborhood gather + FXAA corners + acutance energy.
    float3 neighborhoodColorSpace[9];
    neighborhoodColorSpace[0] = currentColorSpace;

    float  rawCrossLumaSum = 0.0;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float3 tapRGB = max(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb, 0.0);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
        if (acutanceActive && !debugActive && i <= 4)
            rawCrossLumaSum += SrtmLumaFSR(tapRGB);
    }

    float rawHighPass        = (acutanceActive && !debugActive)
                              ? SrtmLumaFSR(currentColorRGB) - rawCrossLumaSum * 0.25
                              : 0.0;
    float rawSharpnessEnergy = rawHighPass * rawHighPass;

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
            : PackClipStateAlpha(earlyRevoked, 0.0, 28.0, 0.0,
                                 saturate(sqrt(rawSharpnessEnergy)) * (1.0 / 3.0), 32.0);

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
        bool touchesAlreadyDilated = (gate.centerFlag >= 1.5) && (gate.gateDepth >= fgThreshold);

        if (!(quadTouchesForeground || touchesAlreadyDilated))
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
    // 11) The temporal clip state, the layer mask, the stats, the
    //     persistence detector, the change-point decisions, the record
    //     update, the statistic gate.
    // ------------------------------------------------------------------
    bool  temporalStateEnabled = (taaUseHullClipping > 0.5);
    float sigmaPrevSq = -1.0;
    float varHPrevSq  = 0.0;
    float agePrev     = 0.0;
    float ghostTPrev  = 0.0;   // the transported persistence statistic T

    float prevStateAlpha = 0.0;
    if (temporalStateEnabled || acutanceActive)
    {
        float2 stateUV = SnapUVToTexel(repro.sampleUV, vp).snappedUV;
        prevStateAlpha = tex2Dlod(historyStateTex, float4(stateUV, 0.0, 0.0)).a;
    }
    if (temporalStateEnabled)
        DecodeClipState(prevStateAlpha, sigmaPrevSq, varHPrevSq, agePrev, ghostTPrev);

    float acutanceStabLin = 0.0;
    if (acutanceActive)
    {
        acutanceStabLin = saturate(sqrt(rawSharpnessEnergy)) * (1.0 / 3.0);
        float ePrev = DecodeAcutanceLinear(prevStateAlpha);
        acutanceStabLin = lerp(ePrev, acutanceStabLin, kSharpEwmaRate);
    }

    bool hadRecord = (sigmaPrevSq >= 0.0);

    // The same-depth-layer mask (the step's level partition).
    bool sameLayerMask[9];
    bool layerMaskValid = needNeighbors;
    [unroll]
    for (int m = 0; m < 9; ++m)
    {
        sameLayerMask[m] = layerMaskValid
            ? (abs(neighborhood.depthRaw[m] - centerDepthRaw) <= edge.edgeEps)
            : true;
    }

    bool ghostModeOn    = (taaClipGhostReset > 0.5);
    bool ghostSoft      = ((taaClipGhostReset > 0.5) && (taaClipGhostReset < 1.5)) || (taaClipGhostReset > 2.5);
    bool ghostTelemetry = (taaClipGhostReset > 1.5) && (taaClipGhostReset < 2.5);
    bool ghostHard      = (taaClipGhostReset > 2.5);

    // The uniform-velocity pre-test: if all eight neighbors carry exactly
    // the center's velocity, every pairwise step is zero -- the quant-step
    // measurement provably returns kVelQuantFloorPx (its > 1e-12 test
    // rejects all-zero deltas) and the edge sweep is provably exactly 0
    // (every dPx is zero). Covers static scenes and rigid camera pans.
    bool velUniform3x3 = true;
    [unroll]
    for (int vu = 1; vu < 9; ++vu)
        velUniform3x3 = velUniform3x3
                     && all(neighborhood.velocityJitteredUV[vu] == centerVelocityJitteredUV);

    // The edge's SWEEP RATE.
    float2 sweepNrm = float2(length(0.5 * (neighborhoodColorSpace[4] - neighborhoodColorSpace[3])),
                              length(0.5 * (neighborhoodColorSpace[2] - neighborhoodColorSpace[1])));
    float  edgeSweepPx = 0.0;
    if (!velUniform3x3 && dot(sweepNrm, sweepNrm) > 1e-12)
    {
        sweepNrm = normalize(sweepNrm);
        [unroll]
        for (int sw = 1; sw < 9; ++sw)
        {
            float2 dPx = (neighborhood.velocityJitteredUV[sw] - centerVelocityJitteredUV) * vp.sizePixels;
            edgeSweepPx = max(edgeSweepPx, abs(dot(dPx, sweepNrm)));
        }
    }

    // The transport scatter's static floor is MEASURED -- the velocity
    // buffer's own quantization/noise scale (the smallest nonzero
    // pairwise step of the 3x3, the same honest noise scale the
    // foreground paths use). Computed only where it can bind (sub-gate
    // motion: at/above the gate motion the jitter-residual proxy is
    // charged at full weight) and only when velocity neighbors were
    // gathered (a no-motion configuration charges no transport scatter by
    // construction).
    float transportFloorPx = 0.0;
    if (needNeighbors && repro.motionMagnitudePx < kTransportGatePx)
        transportFloorPx = velUniform3x3 ? kVelQuantFloorPx
                                         : MeasureVelocityQuantStepPx(neighborhood.velocityJitteredUV, vp.sizePixels);

    // ---- the estimator's edge-tap mask ----------------------------------------
    // The stored motion field's flag channel (w) at the estimator's own
    // history-tap positions. BOTH stored edge flags mark stable texels
    // whose accumulated history is the edge TRANSITION's mix, not a
    // layer's flat content: flag 1 = foreground edge (the crest's own
    // accumulation), flag 2 = kept dilation candidate (the band's). The
    // flag is read at the LANDING position, so flag 1 cannot be left to
    // the depth mask: the depth mask is evaluated at the CURRENT 3x3's
    // positions -- a landing-side crest texel can sit under a
    // currently-background tap and pass it while its history is
    // foreground-mixed. With both masked, the dilation band's own
    // estimator correctly bails (its landing 3x3 is edge/band-flagged) --
    // the band's statistics are the step model's business, and the drift
    // corrector / persistence detector do not act on the band's
    // legitimate accumulation. The CENTER tap is never masked (its
    // mismatch is the signal).
    // When the landing analysis ran, its gather already read the motion
    // field over this exact 3x3 -- the flags ride along at zero fetch
    // cost (tapBandFlag, taaDisocclusion). The explicit fetches remain
    // only for the configuration with the motion field on but no landing
    // gather (both disocclusion tests off, depth dilation on). The range
    // check (<= 2.5) keeps a garbage/NaN flag channel (first frame after
    // a resize) from spuriously masking. With useMotionField off the
    // writer stores flag 0 everywhere -- the mask is simply skipped (the
    // detector's wake latency roughly doubles in that configuration: the
    // documented cost of the Performance preset).
    bool historyTapIsBand[9];
    [unroll]
    for (int hbm = 0; hbm < 9; ++hbm)
        historyTapIsBand[hbm] = false;
    if (useMotionField && layerMaskValid)
    {
        if (needLanding)
        {
            [unroll]
            for (int hfl = 1; hfl < 9; ++hfl)
                historyTapIsBand[hfl] = (landing.tapBandFlag[hfl] >= 0.5)
                                     && (landing.tapBandFlag[hfl] <= 2.5);
        }
        else
        {
            [unroll]
            for (int hbf = 1; hbf < 9; ++hbf)
            {
                float tapFlag = tex2Dlod(historyMotionTex,
                    float4(repro.sampleUV + kOffsets3x3[hbf] * vp.texelSize, 0.0, 0.0)).w;
                historyTapIsBand[hbf] = (tapFlag >= 0.5) && (tapFlag <= 2.5);
            }
        }
    }

    // ---- color stats ----
    ColorNeighborhoodStats colorStats = ComputeColorNeighborhoodStats(
        neighborhoodColorSpace, repro.motionDirUnit, repro.motionNormalized,
        pixel.fracPx, repro.jitterResidualPx, edgeSweepPx, sameLayerMask, layerMaskValid,
        transportFloorPx, historyTapIsBand);

    // ---- the 9-tap spatial record estimate + the drift predict ----
    // (computed BEFORE the history fetch: the correction applies to the
    // history before the innovation, the gate and the detector see it; the
    // phase term is held for the unified innovation correction below.)
    float  recordEstSpatial = -1.0;
    float3 driftCorr        = float3(0.0, 0.0, 0.0);
    float3 estPhaseLS       = float3(0.0, 0.0, 0.0);
    float3 driftMeanEst     = float3(0.0, 0.0, 0.0);   // hoisted: the persistence detector consumes it
    float3 driftVarEst      = float3(0.0, 0.0, 0.0);
    if (layerMaskValid)
    {
        EstimateSpatialRecord(neighborhoodColorSpace, repro.sampleUV, pixel.fracPx, vp,
                              sameLayerMask, layerMaskValid, historyTapIsBand, colorStats,
                              recordEstSpatial, driftMeanEst, driftVarEst, estPhaseLS);

        if (taaDriftCompensation > 0.5)
        {
            // The soft-thresholded tracker: the noise gate is the sparse-
            // prior Bayes action on the estimator's own sampling law; the
            // SNR-shaped gain capped by taaDriftMaxGain is the speed/noise
            // dial (a correction applied at gain g injects ~g*Var/2 of
            // noise power into the history vs the accumulator's own
            // (a/2)Var).
            //
            // THE PERSISTENCE UNLOCK: unconfirmed, the noise gate stands at
            // kDriftThreshSigmas (2.0 estimator sigmas) -- a persistent
            // sub-threshold offset stays invisible forever. When the
            // persistence detector confirms (|T| over kGhostConfirmT,
            // ~1e-3 false rate, EMA-smooth ramp), the threshold drops to
            // kGhostUnlockThrSigmas and the event guard below yields:
            // temporal evidence substitutes for per-frame SNR. THIS -- not
            // the blend floor -- is the ghost-trail eviction engine.
            float  confirmW  = ghostSoft
                ? saturate((abs(ghostTPrev) - kGhostConfirmT) / (kGhostAlarmT - kGhostConfirmT))
                : 0.0;
            float  threshEff = lerp(kDriftThreshSigmas, kGhostUnlockThrSigmas, confirmW);
            float3 dAbs      = abs(driftMeanEst);

            // The event guard's whole posterior machinery is inert unless
            // some channel clears the soft threshold or the detector has
            // confirmed (dThr = 0 -> snr = 0 -> gain = 0, and
            // max(pDrift, confirmW) scales a zero gain). The test runs in
            // the squared domain so the skip path pays no sqrt. The output
            // is identical on both paths; ~90% of pixels take the skip (a
            // 2-sigma one-sided exceedance is ~2%/channel).
            bool anyExcess = (dAbs.x * dAbs.x > threshEff * threshEff * driftVarEst.x)
                          || (dAbs.y * dAbs.y > threshEff * threshEff * driftVarEst.y)
                          || (dAbs.z * dAbs.z > threshEff * threshEff * driftVarEst.z);
            if (anyExcess || confirmW > 0.0)
            {
                float3 sigmaD    = sqrt(driftVarEst);
                float3 dThr      = max(dAbs - threshEff * sigmaD, float3(0.0, 0.0, 0.0));
                float3 snr       = dThr * dThr / max(dThr * dThr + driftVarEst, float3(1e-12, 1e-12, 1e-12));
                float3 gain      = saturate(taaDriftMaxGain) * snr;

                // The event guard is a per-channel POSTERIOR, not a clamp.
                // H_drift: m ~ N(0, v + tau^2), tau =
                // kDriftPriorScaleSigmas * sigmaClean (the lighting-rate
                // prior); H_step: m ~ N(0, v + sigmaClean^2) (a reveal
                // draws from the content marginal). Equal priors; the
                // posterior scales the GAIN per channel. On CONFIRMATION
                // the guard yields (max(pDrift, confirmW)) -- a persistent
                // offset that survived the geometric rejection is a ghost
                // or a lighting change, and both want tracking. The hard
                // magnitude cap stays as the belt. The content scale is
                // the pixel's OWN layer's (sigmaClean; = sigma
                // unstraddled): the full-set sigma carries the off-layer
                // step, a ~dFB-scaled lighting prior that stands the
                // drift posterior down exactly on the wake pixels.
                float3 sigmaC = AnisoClampSigma(colorStats.sigmaClean);
                float3 tauD   = (kDriftPriorScaleSigmas * sigmaC) * (kDriftPriorScaleSigmas * sigmaC);
                float3 vD     = tauD + driftVarEst;
                float3 vS     = sigmaC * sigmaC + driftVarEst;
                float3 lrLog  = 0.5 * (log(vS / vD) + (driftMeanEst * driftMeanEst) * (1.0 / vS - 1.0 / vD));
                float3 pDrift = 1.0 / (1.0 + exp(-lrLog));
                gain = gain * max(pDrift, float3(confirmW, confirmW, confirmW));

                float3 driftCap = kDriftCapSigmas * sigmaC;
                driftCorr = clamp(sign(driftMeanEst) * dThr * gain, -driftCap, driftCap);
            }
        }
    }

    // ---- the history color (Kaiser) ----------------------------------------
    float3 historyColorSpace;
    if (taaUseKaiser6 > 0.5)
        historyColorSpace = SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);
    else
        historyColorSpace = SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, historyMinUV, historyMaxUV);

    if (!(dot(historyColorSpace, historyColorSpace) < kLargeValue))
        historyColorSpace = colorStats.mean;

    // THE PREDICT STEP: the measured common-mode correction, applied before
    // the innovation / gate / detector. The stored history absorbs the
    // correction, so next frame's estimator measures the residual: geometric
    // convergence with no transported state.
    historyColorSpace += driftCorr;

    if (taaHistoryOvershoot > 0.001 || taaColorSpaceOklab > 0.5)
        historyColorSpace = CompressGamut(historyColorSpace);

    // ---- the innovation, its corrected form ----------------------------------
    float3 innov       = currentColorSpace - historyColorSpace;
    float  innovSq     = dot(innov, innov);
    // The UNIFIED phase correction: the estimator's same-layer 9-tap
    // coupled-LS gradient (~2.7x tighter than the central difference),
    // with the CD form as the fallback when the estimator is invalid. The
    // convention is identical (the phase component of a raw innovation is
    // -grad*u; it is removed by ADDING grad*u).
    float3 phaseCorr   = (recordEstSpatial >= 0.0) ? estPhaseLS : (-colorStats.phaseShift);
    float3 innovCorr   = innov + phaseCorr;
    float  innovCorrSq = dot(innovCorr, innovCorr);

    float statAlpha = 1.0 - ComputeMotionFeedback(repro);

    // ---- the record's per-channel split --------------------------------------
    RecordVarianceParts partsLive;
    partsLive.ePer        = float3(0.0, 0.0, 0.0);
    partsLive.varHPer     = float3(0.0, 0.0, 0.0);
    partsLive.sPer        = float3(0.0, 0.0, 0.0);
    partsLive.muVarFull   = float3(0.0, 0.0, 0.0);
    partsLive.muVarScoped = float3(0.0, 0.0, 0.0);
    partsLive.muVar       = float3(0.0, 0.0, 0.0);
    partsLive.centerW     = 0.0;
    partsLive.spatialPriorSq = 0.0;
    if (hadRecord)
        partsLive = SplitRecordVariance(colorStats, sigmaPrevSq, varHPrevSq, statAlpha);

    // The record's information clock.
    float nuPrev = StudentEffectiveDof(agePrev, colorStats.recordSampleDof);

    // ---- THE PERSISTENCE DETECTOR ---------------------------------------------
    // One scalar, transported in the alpha's [5:0] field: T = the EMA of
    // the drift estimator's per-frame spatial t (the LUMA common mode,
    // driftMeanEst.x / sqrt(driftVarEst.x)), scaled by kGhostEmaNorm so
    // the H0 law is ~N(0,1) BY CONSTRUCTION -- the ratio is
    // self-normalized, so a variance-model error cancels between
    // numerator and denominator. Under a persistent offset (a ghost
    // trail: history stuck at stale content) the per-frame t is constant
    // and T converges to t * kGhostEmaNorm in ~13 frames. The per-frame t
    // is clamped at +-6: a single degenerate-subset spike cannot confirm
    // (one frame contributes at most ~3.2 to T).
    float ghostT = ghostTPrev;
    if (ghostModeOn && layerMaskValid)
    {
        float tFrame = driftMeanEst.x / sqrt(max(driftVarEst.x, 1e-12));
        tFrame = clamp(tFrame, -6.0, 6.0);
        ghostT = (1.0 - kGhostEmaRate) * ghostTPrev
               + (kGhostEmaRate * kGhostEmaNorm) * tFrame;
        ghostT = clamp(ghostT, -7.75, 7.75);
    }
    // The confirmed evidence in nats -- the exact tau = sigma0
    // Gaussian-prior Bayes factor against the detector's REALIZED null,
    //     ln BF = T^2 / (4 sigma0^2) - 0.5*ln(2),
    // with sigma0^2 = kGhostNullStdSq (1.5: the t_6 input's nu/(nu-2)
    // inflation -- kGhostEmaNorm normalizes iid UNIT-variance inputs, so
    // T's null std is ~1.22, not 1; the |T| thresholds price against the
    // same null). ZERO below the confirmation threshold -- the detector
    // contributes nothing anywhere until confirmed.
    float ghostEvNats = (ghostSoft && abs(ghostT) > kGhostConfirmT)
        ? (kGhostEvCoef * ghostT * ghostT - 0.34657)
        : 0.0;
    bool ghostAlarm = ghostHard && (abs(ghostT) >= kGhostAlarmT);

    // ---- change-points, in order of authority --------------------------------
    // The plausible-range reference at a straddle is the own-layer range
    // PLUS the coverage span -- the exact slice of what the full AABB
    // carries: at w = 0 (unresolvable step) the span is the full dFB0 and
    // this recovers the full-range behavior; at w = 1 it is the measured
    // bracket (and the record is step-scale there and dominates the
    // max() anyway). The wake is unstraddled and never sees this.
    float rangeSqSpike = colorStats.rangeSq;
    if (layerMaskValid && colorStats.straddled)
        rangeSqSpike = colorStats.rangeSqClean + colorStats.covSpanSq;
    bool  spike   = hadRecord
        && (innovCorrSq > kClipSpikeRatio * max(sigmaPrevSq, rangeSqSpike));

    bool geoCorroborated = false;
    if (hadRecord && disoccluded)
        geoCorroborated = (WhitenedInnovSq(innovCorr, partsLive) > kGeoCorroborateChiSq);

    // The alarm does not reset the record (a confirmed trail does not need
    // the Student dof drop; the corrector and floor evict it), and a
    // record reset does not drain T (see the transport below).
    bool  resetRecord = hadRecord
        && (spike || !(innovSq < kLargeValue) || geoCorroborated);
    bool  carryRecord = hadRecord && !resetRecord;

    // FICTION FREEZE: one-line A/B toggle for the record/transport fiction
    // models.
    bool fictionFreeze = false;

    // The cap references the CONTENT scale, not the record.
    float winsorCap  = kClipWinsorC * kClipWinsorC
                     * max(partsLive.spatialPriorSq, kClipSigmaRecordFloorSq);

    float sigmaStatSq, ageNext;
    if (fictionFreeze)
    {
        sigmaStatSq = max(sigmaPrevSq, kClipSigmaRecordFloorSq);
        ageNext     = agePrev;
    }
    else if (carryRecord)
    {
        float innovIngest;
        if (colorStats.stepW > 0.5)
        {
            // MODEL-REFERENCED (the stable temporal average is exactly the
            // step's variance model).
            float3 dObs   = currentColorSpace - colorStats.stepXModel;
            float3 dPhase = colorStats.stepXModel - colorStats.stepTargetStable;
            innovIngest = min(dot(dObs, dObs) + dot(dPhase, dPhase), winsorCap);
        }
        else
        {
            innovIngest = (recordEstSpatial >= 0.0)
                ? min(recordEstSpatial, winsorCap)
                : min(innovSq, winsorCap);
        }
        sigmaStatSq = max(lerp(sigmaPrevSq, innovIngest, kClipSigmaEmaRate), kClipSigmaRecordFloorSq);
        ageNext     = min(agePrev + 1.0, kClipMaxAge);
    }
    else
    {
        // Coherence note: the reset seed is the Var(x) model (engaged
        // steps) or the spatial E[i^2]-flavored estimate (flats), while
        // the carried record converges toward E[i^2] -- a 2/(2-a) - 1
        // inconsistency (~1.5% at default feedback) at the reset boundary;
        // exact at a = 1 (full-replacement frames, where resets
        // concentrate).
        sigmaStatSq = (colorStats.stepW > 0.5 || recordEstSpatial < 0.0)
            ? SeedRecordTotalSq(colorStats)
            : max(recordEstSpatial, kClipSigmaRecordFloorSq);
        ageNext     = 0.0;
    }

    // ---- the statistic gate ---------------------------------------------------
    // The gate runs ONCE: resetRecord is fully determined before it (the
    // spike test needs the corrected innovation and the range reference,
    // the geometric corroboration needs the record split -- none need the
    // gate), and the cold path never reads the record age, so the cold
    // re-entry on reset pixels folds into this single call.
    float nuHonest = (colorStats.stepW > 0.5) ? colorStats.cusumStepDof : nuPrev;

    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, colorStats, partsLive, carryRecord,
        carryRecord ? agePrev : 0.0, statAlpha, repro.motionNormalized,
        neighborhoodColorSpace,
        ghostEvNats, nuHonest);
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
            // Telemetry: the fitted coverage on engaged steps; the
            // persistence statistic |T|/alarm elsewhere (dim on static
            // scenes, |T| < ~1 against the 1.22-sigma null; brightness
            // blooming along a real trail, then decaying as the corrector
            // evicts it -- THE detector verification view).
            dbgB = ghostTelemetry
                ? ((colorStats.stepW > 0.5)
                    ? saturate(colorStats.stepC)
                    : saturate(abs(ghostT) * (1.0 / kGhostAlarmT)))
                : 1.0 - clipGate.tGate;
        }
        else if (taaDebugMode > 11.5 && taaDebugMode < 12.5)   // 12: dejittered residual
        {
            float2 residualPx = (currentLayer.effectiveVelocityUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
            dbgCode = 12.0;
            dbgA = saturate(length(residualPx) * 0.5);
        }
        else if (taaDebugMode > 12.5 && taaDebugMode < 13.5)   // 13: gate null law
        {
            // The realized-null calibration view. Park on a STATIC scene
            // and read A's brightness distribution: the nominal 95th
            // percentile sits at chi^2/16 (mid-gray at the default
            // chi 2.8); brightness beyond that is the gate running hot,
            // and the EMPIRICAL Student factor is (A's 95th percentile) *
            // 16 / chi^2. B = engagement (mdd > chi^2, blue tint).
            float chiDbg = max(taaVarianceGamma, kEpsilon) * (1.0 + max(taaClipOvershoot, 0.0));
            dbgCode = 13.0;
            dbgA = saturate(clipGate.mdd * (1.0 / 16.0));
            dbgB = (clipGate.mdd > chiDbg * chiDbg) ? 1.0 : 0.0;
        }
    }

    float clipDistanceRejection = 0.0;
    if (taaClipDistanceRejectionEnabled > 0.5)
        clipDistanceRejection = ComputeClipDistanceRejection(
            clippedHistorySpace, historyColorSpace, colorStats);

    ApplyLumaDriftCorrection(clippedHistorySpace, currentColorSpace, colorStats);

    // ------------------------------------------------------------------
    // 12) Feedback & the temporal blend.
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, clipDistanceRejection, currentSingleLayer);
    float currentBlendWeight = 1.0 - historyFeedback;

    // The replacement posterior floors the blend: under H1 the Bayes
    // action is (1-p1)*h_clipped + p1*x_current. p1 carries the CONFIRMED
    // persistence evidence (zero below confirmation -- the detector
    // contributes nothing to the blend on ~99.9% of pixels; the floor
    // exists for sub-radius persistent ghosts the gate cannot see).
    currentBlendWeight = max(currentBlendWeight, clipGate.p1);

    if (disoccluded)
        currentBlendWeight = 1.0;

    if (ghostAlarm)
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
    // 13) The VarH recursion, the final blend & the output.
    // ------------------------------------------------------------------
    float varHCodeNext = 12.0;
    {
        float aBlend = currentBlendWeight;
        float vTotal = 3.0 * kResampleVarSq;
        // READ semantics: varH is the variance of the READ (the
        // Kaiser-resampled history that the gate tests), not the stored
        // field; the recursion's input is the implied Var(x) (the record
        // E[i^2] MINUS the read variance); and the two gains are the
        // host-measured kernel constants -- dampGain = the fixed-point
        // read gain for the (smooth, re-read) history term, freshGain =
        // the white-input read gain for the (white across texels) fresh-x
        // term.
        float dampGain  = 1.0 - saturate(taaVarhResampleLoss);
        float freshGain = 1.0 - saturate(taaVarhWhiteLoss);
        if (fictionFreeze)
        {
            varHCodeNext = EncodeVarHRatio(varHPrevSq, max(sigmaStatSq, kEpsilon));
        }
        else if (carryRecord)
        {
            // The recursion consumes the SAME prior-capped share the gate
            // does (kArmingCap * spatialPriorSq, see SplitRecordVariance)
            // -- otherwise the band-carryover state self-sustains: varH >
            // record zeroes vxFresh every frame and the only decay is the
            // (1-a)^2 damp, which is SLOWER than the record's own EMA --
            // the transported ratio then grows frame over frame exactly
            // on the texels leaving a dilation band.
            float varHPrevCapped = min(varHPrevSq, kArmingCap * partsLive.spatialPriorSq);
            float vxFresh  = max(sigmaStatSq - varHPrevCapped, 0.0);
            float varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * varHPrevCapped
                           + aBlend * aBlend * freshGain * vxFresh;
            varHCodeNext = EncodeVarHRatio(varHNext, max(sigmaStatSq, kEpsilon));
        }
        else
        {
            float3 dCorrSeed = historyColorSpace - colorStats.mean + colorStats.gatePhaseShift;
            float  tC = clipGate.tGate;
            float3 varH0Per = clipGate.rawGateVar + (1.0 - tC) * (1.0 - tC) * (dCorrSeed * dCorrSeed);
            float  varH0 = varH0Per.x + varH0Per.y + varH0Per.z;
            float  s0 = max(sigmaStatSq - varH0 - vTotal, 0.0);
            float  varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * (varH0 + vTotal)
                            + aBlend * aBlend * freshGain * s0;
            varHCodeNext = EncodeVarHRatio(varHNext, max(sigmaStatSq, kEpsilon));
        }
    }

    float3 blendedColorSpace = lerp(clippedHistorySpace, currentFrameColorSpace, currentBlendWeight);
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);
    // The persistence statistic's transport: 0.25-quanta code = round(4T)
    // + 32 (T in [-7.75, 7.75] = codes [1, 63]). Never force-drained on a
    // record reset: the EMA's memory is bounded, and a reset
    // (spike/geometry) is often the moment a trail is FRESHEST.
    // Full-replacement frames (disocclusion, the alarm) erase the offset
    // itself, so T decays naturally through the estimator.
    float ghostCodeNext = clamp(round(ghostT * 4.0) + 32.0, 1.0, 63.0);
    float outAlpha = debugActive
        ? PackDebugAlpha(dilationRevoked, dbgCode, saturate(dbgA), saturate(dbgB), sqrt(sigmaStatSq))
        : PackClipStateAlpha(dilationRevoked, sqrt(sigmaStatSq), varHCodeNext, ageNext, acutanceStabLin, ghostCodeNext);
    return float4(outputRGB, outAlpha);
}

// ============================================================================
// MAIN VERTEX SHADER
// ============================================================================
PFXVertToPix mainV(PFXVert IN)
{
    return processPostFxVert(IN);
}