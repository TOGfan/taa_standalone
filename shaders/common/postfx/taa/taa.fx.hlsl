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
//   11)    THE FLAT MODEL (taaClip): the robust coupled-plane statistics
//          with the exact hat-matrix bookkeeping, the H-free record, the
//          structure split's transport, the level gate (the hierarchical
//          posterior + the measured-miss soft clip), the blend, the VarH
//          recursion.
//
// OUTPUT: RGB = the resolved color. A = the packed CLIP STATE (tag 101).
// Debug modes carry tag 001. Nothing non-finite is ever written (the
// write-side guard) -- a NaN frame would otherwise be feedback-locked and
// neighbor-visible through the resample footprint forever.
//
// DEBUG MODES: 0 off | 1 frame motion | 2 disocclusion breakdown |
//   3 center velocity | 4 linearized depth | 5 history color | 6 landing
//   velocity | 7 pursuit divergence | 8 layer state | 9 dilation gate |
//   10 alignment drop | 11 clip gate state | 12 dejittered residual |
//   13 gate null law (calibration) | 14 deployment canary.
//
// DEPTH EXACTNESS: the dilation-validation gate and the depth test share
// the layer's fitted forward displacement T_y; the expected-crest depth is
// transported with the exact relation w' = w*B + T_y composed with the
// rotation-exact raw form (raw(u) = rayLen(u)/D, so 1/rawExpRot = w*B
// exactly under pure rotation; translation adds T_y). Revoked candidates
// drop the measurement and the depth test re-fits internally.
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
    float  taaAccNoiseFactor;
    float  taaClipBlendDrop;                float  taaClipBlendDropStart;
    float  taaClipBlendDropWidth;

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
        cornersRGB[c] = ScrubRawRGB(tex2Dlod(sceneTex, float4(tapUVs[c + 5], 0.0, 0.0)).rgb);
}

// ============================================================================
// THE CLIP STATE, DECODE-THEN-BLENDED
// ----------------------------------------------------------------------------
// The state must describe the content the Kaiser read actually blended --
// a ~9-21 texel footprint -- not one coin-flipped texel. Four point-sampled
// fetches (the bilinear quad around the landing), each DECODED, then:
//   * the CONTINUOUS fields (sigma^2, varH -- linear space; varH's ratio
//     code decodes against ITS OWN texel's sigma first) blend with the
//     bilinear weights, renormalized over the valid texels;
//   * the DISCRETE transports (age: 4 bits, split: 6 bits -- blends of
//     packed fields are meaningless) come from the maximum-weight VALID
//     texel -- exactly the nearest, i.e. exactly the snapped read's choice;
//   * no valid texel -> COLD.
// The raw bits are never interpolated and nothing is re-encoded. At an
// exact texel-center landing (the static identity) the weights degenerate
// to (1,0,0,0) and the read is BIT-IDENTICAL to the snapped form -- the
// blend engages only with sub-texel phase, never with motion state.
// Known approximation (accepted): a linear blend of varH is the smooth-
// field interpolation of READ-semantics values -- neighboring records
// share overlapping 3x3 neighborhoods and each transported value already
// carries the read's own correlation structure; far closer than a snap.
// ============================================================================
void SampleHistoryClipState(float2 historyUV, ViewportParams vp,
                            out float sigmaSq, out float varHSq,
                            out float recordAge, out float splitFrac)
{
    sigmaSq = -1.0;
    varHSq = 0.0;
    recordAge = 0.0;
    splitFrac = 0.0;

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos - 0.5) + 0.5;   // the bilinear quad's base
    float2 f         = pixelPos - baseTexel;           // [0,1)^2

    float2 tc0  = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 tcX  = clamp((baseTexel + float2(1.0, 0.0)) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 tcY  = clamp((baseTexel + float2(0.0, 1.0)) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 tcXY = clamp((baseTexel + float2(1.0, 1.0)) * vp.texelSize, vp.minUV, vp.maxUV);

    float a0  = tex2Dlod(historyStateTex, float4(tc0,  0.0, 0.0)).a;
    float aX  = tex2Dlod(historyStateTex, float4(tcX,  0.0, 0.0)).a;
    float aY  = tex2Dlod(historyStateTex, float4(tcY,  0.0, 0.0)).a;
    float aXY = tex2Dlod(historyStateTex, float4(tcXY, 0.0, 0.0)).a;

    float s0, v0, ag0, sp0;
    float sX, vX, agX, spX;
    float sY, vY, agY, spY;
    float sXY, vXY, agXY, spXY;
    DecodeClipState(a0,  s0,  v0,  ag0,  sp0);
    DecodeClipState(aX,  sX,  vX,  agX,  spX);
    DecodeClipState(aY,  sY,  vY,  agY,  spY);
    DecodeClipState(aXY, sXY, vXY, agXY, spXY);

    float w0  = (1.0 - f.x) * (1.0 - f.y);
    float wX  = f.x * (1.0 - f.y);
    float wY  = (1.0 - f.x) * f.y;
    float wXY = f.x * f.y;

    float wSum = 0.0;
    float sSum = 0.0;
    float vSum = 0.0;
    float wBest = 0.0;
    if (s0 >= 0.0)  { wSum += w0;  sSum += w0 * s0;  vSum += w0 * v0;   if (w0  >= wBest) { wBest = w0;  recordAge = ag0;  splitFrac = sp0; } }
    if (sX >= 0.0)  { wSum += wX;  sSum += wX * sX;  vSum += wX * vX;   if (wX  >= wBest) { wBest = wX;  recordAge = agX;  splitFrac = spX; } }
    if (sY >= 0.0)  { wSum += wY;  sSum += wY * sY;  vSum += wY * vY;   if (wY  >= wBest) { wBest = wY;  recordAge = agY;  splitFrac = spY; } }
    if (sXY >= 0.0) { wSum += wXY; sSum += wXY * sXY; vSum += wXY * vXY; if (wXY >= wBest) { wBest = wXY; recordAge = agXY; splitFrac = spXY; } }

    if (wSum > 1e-6)
    {
        sigmaSq = sSum / wSum;
        varHSq  = vSum / wSum;
    }
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

    float3 currentColorRGB          = ScrubRawRGB(tex2Dlod(sceneTex,    float4(pixel.snappedUV, 0.0, 0.0)).rgb);
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
        float3 tapRGB = ScrubRawRGB(tex2Dlod(sceneTex, float4(tapUVs[i], 0.0, 0.0)).rgb);
        neighborhoodColorSpace[i] = ToSpace(tapRGB);
    }
    {
        float3 tapRGB = ScrubRawRGB(tex2Dlod(sceneTex, float4(tapUVs[8], 0.0, 0.0)).rgb);
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

    // 8) Own-history dilation validation: the layer's forward displacement
    //    T_y is fitted BEFORE the gate for dilation candidates and REUSED
    //    by the depth test (identical inputs -> bit-identical values for
    //    every non-revoked pixel); on revocation the measurement is dropped
    //    and the depth test re-fits inside its own early-out-gated path.
    //    The pre-gate fit measures the FOREGROUND layer -- for a kept
    //    dilation candidate that mask IS the foreground (the crest and its
    //    kin), exactly the layer the crest depth belongs to.
    float  ty = 0.0, tySigma = 0.0;
    bool   tyMeasured = false;

    bool dilationRevoked   = false;
    bool dilationCandidate = false;
    bool gateViaFlag       = false;

    if (useMotionField && currentLayer.isDilationZone)
    {
        dilationCandidate = true;

        if (depthTestActive)
        {
            float3 prevFwd = CameraForwardAxis(previousCamera);
            tyMeasured = EstimateLayerForwardParallax(
                currentLayer.effectiveVelocityUV, stableInFrameUV, IN.uv0,
                neighborhood.depthRaw, neighborhood.velocityJitteredUV, tapUVs,
                currentLayer.isDilationZone, currentLayer.isForegroundEdge,
                currentCamera, previousCamera, prevFwd,
                coherenceRadiusPx, vp, ty, tySigma);
        }

        float rayLenCur  = RayLengthFromUV(IN.uv0, taaTanHalfFovX, taaTanHalfFovY);
        float rayLenPrev = RayLengthFromUV(repro.sampleUV, taaTanHalfFovX, taaTanHalfFovY);
        float perspScale = rayLenPrev / max(rayLenCur, 1e-6);

        // The expected stored raw of the crest, WITH the translation: the
        // stored field is last frame's and the camera TRANSLATES -- the
        // exact relation is w' = w*B + T_y, hence
        // rawExp = 1/(1/rawExpRot + ty). ty = 0 (fit off or failed)
        // degenerates to the rotation-only value.
        float rawExpRot     = currentLayer.closestDepth * perspScale;
        float rawExpClosest = 1.0 / max(1.0 / max(rawExpRot, 1e-6) + ty, kEpsilon);

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

            // The revoked pixel now transports its OWN background surface;
            // the pre-gate foreground measurement no longer applies.
            tyMeasured = false;
            ty = 0.0;
            tySigma = 0.0;
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
            vp,
            ty, tySigma, tyMeasured);
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
        // Decode-then-blend: the state describes the footprint the Kaiser
        // read actually blended. Bit-identical to the snapped read at exact
        // texel centers (the static identity).
        SampleHistoryClipState(repro.sampleUV, vp,
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

    // THE LANDING-MISS MODEL. The soft clip's H1 is priced by the measured
    // miss distribution. sigma^2 = the gated transport scatter (the known
    // residual -- per-frame noise-like, wandering with the jitter, so it is
    // SPREAD, not offset: the mixture stays centered on the target and
    // there is no sign convention to get wrong) + the velocity field's
    // measured noise scale. Where the quant-step scan was skipped (high
    // motion), the uniform-field floor stands in -- at high motion the
    // miss scale is dominated by the measured residual anyway.
    //
    // THE SCALE'S DOF (missModel.z): the residual component of the scale
    // is EXACT arithmetic (the jitter is known, the landing phase is
    // known); the velocity component is an ESTIMATE -- the 36-pair
    // quant-step scan over nine heavily correlated taps (~6 effective
    // dof), or a convention (the fixed 1/16-px floor, the uniform-field
    // classification -- numerically Gaussian by nu ~ 256). The
    // Satterthwaite blend over the variance components,
    //     nu = nuVel * (1 + exact^2 / estimated^2)^2,
    // sends certain scales to the Gaussian law and velocity-dominated
    // scales to nuVel; the tail law and the cell masses marginalize it
    // (taaClip). A point-Gaussian prior collapses artificially hard at
    // tight sigma -- the plug-in scale's artifact.
    //
    // THE CONDITIONAL MISS RATE (missModel.w, a log2 multiplier on the
    // prior odds): the SCALE prices how FAR a miss lands, not how OFTEN
    // one happens -- and landings are not exchangeable. Two measured
    // high-risk classes, keyed on state the resolve already computed:
    // the similarity-extrapolated landings (dilation zones and crest
    // cliff-phases -- the extrapolation's error is first-order in the
    // field curvature, far above the quant floor the scale charges) and
    // the velocity-straddled phase quads (the resolved velocity can be
    // the wrong layer's by a full step). The straddle test is recomputed
    // rather than exported (the mode-8 precedent); its pre-screen makes
    // it a few ops on quiet quads. The boost lifts the prior odds only --
    // the scales, the masses and the tail stay measured. The dials are
    // the constants in taaClip, never a cap on the measured scale.
    // Consumed ONLY by the soft clip's posteriorPath: with taaSoftClip off
    // the whole block is dead work (no debug mode reads it -- mode 8
    // recomputes its own straddle test), so the uniform branch skips it.
    // transportErrPx above stays -- the gate's transportVar consumes it in
    // every configuration.
    float4 missModel = float4(0.0, 0.0, 2.0, 0.0);   // xy = the miss sigmas (texels), z = the scale's dof, w = the rate boost (log2)
    if (taaSoftClip > 0.001)
    {
        float velNoisePx = (transportFloorPx > 0.0) ? transportFloorPx : kVelQuantFloorPx;
        missModel.x = sqrt(transportErrPx.x * transportErrPx.x + velNoisePx * velNoisePx);
        missModel.y = sqrt(transportErrPx.y * transportErrPx.y + velNoisePx * velNoisePx);

        float nuVel = (transportFloorPx > 0.0 && !velUniform3x3) ? kMissNuVelMeasured : kMissNuVelFloor;
        float evSq = velNoisePx * velNoisePx;
        float ratioX = 1.0 + (transportErrPx.x * transportErrPx.x) / evSq;
        float ratioY = 1.0 + (transportErrPx.y * transportErrPx.y) / evSq;
        missModel.z = clamp(min(nuVel * ratioX * ratioX, nuVel * ratioY * ratioY), 2.0, 256.0);

        float missRateBoost2 = 0.0;
        if (currentLayer.extrapolationDispPx > 0.0)
            missRateBoost2 += kMissRateExtrapolated2;
        if (!currentLayer.isDilationZone && !currentLayer.isForegroundEdge)
        {
            float2 v00, v10, v01, v11;
            SelectPhaseQuad(neighborhood.velocityJitteredUV, pixel.fracPx, v00, v10, v01, v11);
            if (QuadStraddlesVelocityStep(v00, v10, v01, v11, neighborhood.velocityJitteredUV, vp.sizePixels, coherenceRadiusPx))
                missRateBoost2 += kMissRateStraddled2;
        }
        missModel.w = min(missRateBoost2, kMissRateBoostMax2);
    }

    // ---- the statistics (the robust coupled 2x2 plane, the exact
    //      hat-matrix bookkeeping; no masks, no flags) ----
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

    float3 historyColorSpace;
    if (taaUseKaiser6 > 0.5)
        historyColorSpace = SampleHistoryColor_Kaiser6_21Tap(repro.sampleUV, vp, vp.minUV, vp.maxUV);
    else
        historyColorSpace = SampleHistoryColor_Kaiser4_9Tap(repro.sampleUV, vp, vp.minUV, vp.maxUV);

    if (!(dot(historyColorSpace, historyColorSpace) < kLargeValue))
        historyColorSpace = colorStats.plane0;

    if (taaHistoryOvershoot > 0.001 || taaColorSpaceOklab > 0.5)
        historyColorSpace = CompressGamut(historyColorSpace);

    // ---- the accumulated neighborhood: the soft clip's H1 components --
    //      the history texels at the landing's 8 neighbors, what a
    //      landing-miss ghost actually holds. Fetched only when the soft
    //      clip can run (~8 fetches = ~10% of the resolve's fetch traffic
    //      with the posterior enabled). Exact texel centers through the
    //      linear slot-2 sampler; NO CompressGamut (raw stored texels are
    //      in-gamut by construction); NO phase correction (stable-space
    //      content). Motion-agnostic: the prior odds do the gating. ----
    float3 accumulatedNeighbors[8];
    if (taaSoftClip > 0.001)
    {
        float2 landingTexel = floor(repro.sampleUV * vp.sizePixels) + 0.5;
        [unroll]
        for (int aj = 1; aj < 9; ++aj)
        {
            float2 uv = clamp((landingTexel + kOffsets3x3[aj]) * vp.texelSize, vp.minUV, vp.maxUV);
            float3 acc = ToSpace(ScrubRawRGB(tex2Dlod(historyTex, float4(uv, 0.0, 0.0)).rgb));
            if (!(dot(acc, acc) < kLargeValue))
                acc = colorStats.plane0;    // the NaN guard's policy, as the center read
            accumulatedNeighbors[aj - 1] = acc;
        }
    }
    else
    {
        [unroll]
        for (int ajOff = 0; ajOff < 8; ++ajOff)
            accumulatedNeighbors[ajOff] = float3(0.0, 0.0, 0.0);
    }

    // ---- the innovation (the stream the spike and the corroboration
    //      price) ----
    float3 innov   = currentColorSpace - historyColorSpace;
    float  innovSq = dot(innov, innov);

    // ---- the shrunk channel correlations (ONE evaluation: the spike, the
    //      winsor, the corroboration test and the statistic gate all
    //      consume the same values) ----
    float rXYg, rXZg, rYZg;
    ShrunkenCorrelations(colorStats, rXYg, rXZg, rYZg);

    // THE SHARED METRIC, built once: the whitened innovation AND the
    // statistic gate's every quadratic form consume the same R^-1 at unit
    // scale (the gate's internal cofactor block is gone -- it reads this
    // struct now).
    SharedCorrelationMetric sharedMetric = BuildSharedCorrelationMetric(rXYg, rXZg, rYZg);

    // ONE evaluation of the whitened innovation: the spike's record branch
    // and the corroboration reset both consume it (the former double
    // evaluation is gone).
    float wInnovSq = 0.0;
    if (hadRecord)
        wInnovSq = WhitenedInnovSq(innov, partsLive, colorStats.transportVar, sharedMetric);

    // ---- change-points ----
    // The spike: two branches, both explicit. The record branch is a
    // WHITENED chi^2_3 test at ~1e-6 per frame through the shrunk full
    // metric -- the same channel law as the corroboration reset, with the
    // transport scatter charged per-channel inside the metric (a scalar-
    // total form would assume exchangeable uncorrelated channels: on
    // luminance-dominated YCoCg content the 3-channel sum's effective dof
    // is ~1, and the priced ~1e-6 level would run at ~1e-3 there). The
    // range branch is the distribution-free garbage backstop (16x the
    // observed AABB range -- effectively never on the null by design: the
    // H-free ingest absorbed the level-change job, geoCorroboration owns
    // reveals, and the NaN guard is separate).
    bool  spike = hadRecord
        && (innovSq > 16.0 * colorStats.rangeSq
            || wInnovSq > kChiSq3_1e6);

    bool geoCorroborated = false;
    if (hadRecord && disoccluded)
    {
        geoCorroborated = (wInnovSq > kChiSq3_95);
    }

    bool  resetRecord = hadRecord
        && (spike || !(innovSq < kLargeValue) || geoCorroborated);
    bool  carryRecord = hadRecord && !resetRecord;

    // ---- THE STRUCTURE SPLIT'S TRANSPORT ----
    // The signed normalized fraction f = (R - s)/max(R, s): the spatial
    // residual total against the record (which IS s). With the exact
    // residual dof this anchor is unbiased -- the ingest's fixed point is
    // e = 0. A fresh record certifies nothing (the seed >= R by
    // construction); a disocclusion restarts the split with the new
    // content.
    float splitFracNext;
    {
        float residTotal = dot(colorStats.residSq, float3(1.0, 1.0, 1.0));
        float sPerTotal  = hadRecord
            ? max(sigmaPrevSq, kClipSigmaRecordFloorSq)
            : max(SeedRecordTotalSq(colorStats), kClipSigmaRecordFloorSq);
        float fRaw = clamp((residTotal - sPerTotal)
                           / max(max(residTotal, sPerTotal), kEpsilon), -1.0, 1.0);
        splitFracNext = (carryRecord && !disoccluded)
            ? lerp(splitPrev, fRaw, kSplitEmaRate)
            : fRaw;
        // Transport scrub: NaN here would reach round() and uint() in the
        // packer. abs(NaN) <= 1.0 is false -- the strict-compare idiom.
        splitFracNext = (abs(splitFracNext) <= 1.0) ? splitFracNext : 0.0;
    }

    // ---- THE RECORD: THE H-FREE INGEST ----
    // Noise lives in dev = x0c - plane0 (H does not appear); change lives
    // in dCorr (it does not appear here). The de-bias is the exact
    // identity E[dev^2] = sPer + vPrior - 2*cPT. The winsor caps the
    // ingest at its OWN Satterthwaite dof's 0.1% point. The circularity
    // (record -> split -> tauB2 -> vPrior -> ingest -> record) has loop
    // gain (1 - q) < 1: convergent, honest fixed point e = 0 on the R > s
    // branch -- the exact residual dof is that statement's stability
    // condition.
    float sigmaStatSq, ageNext;
    {
        if (carryRecord)
        {
            // The honest 0.1% censoring point of the 3-channel correlated
            // sum (a chi^2_3/3 stand-in would run the point at ~2% on gray
            // content and systematically under-ingest legitimate variance).
            float winsorCap = WinsorCapNuStar(partsLive, colorStats, splitPrev, rXYg, rXZg, rYZg);
            // The de-bias is taaClip's law now: RecordIngestPerChannel is
            // the exact identity E[dev^2] = sPer + vPrior - 2*cPT -- the
            // same null the gate and the winsor consume (bit-identical to
            // the historical inline form).
            float3 sIngestPer = RecordIngestPerChannel(currentColorSpace, colorStats, partsLive, splitPrev);
            float  sIngest    = max(dot(sIngestPer, float3(1.0, 1.0, 1.0)), 0.0);
            sigmaStatSq = max(lerp(sigmaPrevSq, min(sIngest, winsorCap), kClipSigmaEmaRate),
                              kClipSigmaRecordFloorSq);
            ageNext     = min(agePrev + 1.0, kClipMaxAge);
        }
        else
        {
            sigmaStatSq = max(SeedRecordTotalSq(colorStats), kClipSigmaRecordFloorSq);
            ageNext     = 0.0;
        }
        // Transport scrub: NaN here would reach sqrt() in the packer, the
        // VarH recursion and next frame's record.
        sigmaStatSq = ScrubF(sigmaStatSq, kClipSigmaRecordFloorSq);
    }

    // ---- THE LEVEL GATE ----
    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, currentColorSpace, colorStats, partsLive,
        carryRecord, carryRecord ? agePrev : 0.0,
        splitPrev,
        missModel,
        accumulatedNeighbors,
        sharedMetric);
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
        else if (taaDebugMode > 13.5 && taaDebugMode < 14.5)   // 14: deployment canary
        {
            // Three one-bit probes, settings-independent: 4 = the pushed
            // non-setting constants are live (taaArmingA is a host-FIT
            // value ~4.35; a stale/no-op'ed cbuffer reads 0, and no user
            // setting can push it below 1); 2 = the viewport uniforms are
            // sane (1/size in (0,1] -- the solid-color failure is 1/size
            // = 0, every tap sampling texel (0,0)); 1 = the transported
            // state decodes (tag 5 live, tag 1 while ANY debug mode is on
            // -- both mean the store chain ran).
            float sa = tex2Dlod(historyStateTex, float4(clamp(IN.uv0, vp.minUV, vp.maxUV), 0.0, 0.0)).a;
            uint  tg = (asuint(sa) >> 28) & 0x7u;
            float f = (taaArmingA > 1.0 ? 4.0 : 0.0)
                    + ((oneOverTargetSize.x > 1e-6 && oneOverTargetSize.x <= 1.0) ? 2.0 : 0.0)
                    + ((tg == 0x5u || tg == 0x1u) ? 1.0 : 0.0);
            dbgCode = 14.0; dbgA = f * 0.125;
        }
    }

    // ------------------------------------------------------------------
    // 12) Feedback & the temporal blend.
    // ------------------------------------------------------------------
    float historyFeedback    = ComputeHistoryFeedback(repro, currentSingleLayer);

    // The clip-excess blend reduction: how far the history landed outside
    // the gate (in gate radii) linearly drops the history blend to zero
    // past a start margin. Purely blend-side -- the clamp / soft clip fix
    // the STORED value, this fixes how much of it survives this frame; the
    // record and the split are untouched by policy, and the VarH recursion
    // sees the raised blend weight and correctly restarts the accumulator
    // on dropped pixels through its own semantics.
    historyFeedback *= 1.0 - ComputeClipExcessBlendDrop(clipGate);

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
    // The write-side guard: NOTHING non-finite is ever stored. One NaN
    // frame here is feedback-locked and neighbor-visible through the
    // resample footprint; the fallback is guaranteed finite (every scene
    // read is scrubbed; the gate chain is total on finite input).
    if (!(dot(blendedColorSpace, blendedColorSpace) < kLargeValue))
        blendedColorSpace = currentFrameColorSpace;
    blendedColorSpace.x = max(blendedColorSpace.x, 0.0);
    float3 outputRGB = max(FromSpace(blendedColorSpace), 0.0);

    // ---- the VarH recursion (READ semantics; the host-measured gains;
    //      12-bit transport). sigmaStatSq IS sPer: the fresh-x term
    //      injects it directly. ----
    float varHCodeNext = 1820.0;
    {
        float aBlend = currentBlendWeight;
        float dampGain  = 1.0 - saturate(taaVarhResampleLoss);
        float freshGain = 1.0 - saturate(taaVarhWhiteLoss);
        if (carryRecord)
        {
            float varHPrevCapped = dot(partsLive.varHPer, float3(1.0, 1.0, 1.0));
            float varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * varHPrevCapped
                           + aBlend * aBlend * freshGain * max(sigmaStatSq, 0.0);
            varHCodeNext = EncodeVarHRatio(varHNext, max(sigmaStatSq, kEpsilon));
        }
        else
        {
            float3 dCorrSeed = historyColorSpace - clipGate.center;
            float  tC = clipGate.tGate;
            float3 varH0Per = clipGate.rawGateVar + (1.0 - tC) * (1.0 - tC) * (dCorrSeed * dCorrSeed);
            float  varH0 = varH0Per.x + varH0Per.y + varH0Per.z;
            float  varHNext = (1.0 - aBlend) * (1.0 - aBlend) * dampGain * varH0
                            + aBlend * aBlend * freshGain * max(sigmaStatSq, 0.0);
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
