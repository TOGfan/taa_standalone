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
//   11)    THE FLAT MODEL (taaClip v5.6): the robust coupled-plane
//          statistics with the exact hat-matrix bookkeeping, the H-free
//          record, the structure split's transport, the level gate (the
//          hierarchical posterior + the measured-miss soft clip), the
//          blend, the VarH recursion.
//
// OUTPUT: RGB = the resolved color. A = the packed CLIP STATE (tag 101).
// Debug modes carry tag 001.
//
// DEBUG MODES: 0 off | 1 frame motion | 2 disocclusion breakdown |
//   3 center velocity | 4 linearized depth | 5 history color | 6 landing
//   velocity | 7 pursuit divergence | 8 layer state | 9 dilation gate |
//   10 alignment drop | 11 clip gate state | 12 dejittered residual |
//   13 gate null law (calibration: dbgA = saturate(mdd/16)).
//
// v5.6.2 (depth-disocclusion exactness): the dilation-validation gate's
//   expected-crest depth now carries the fitted T_y (w' = w*B + T_y composed
//   with the rotation-exact raw form). The layer's forward displacement is
//   fitted BEFORE the gate for dilation candidates and SHARED with the depth
//   test -- identical inputs, bit-identical values for every non-revoked
//   pixel; revoked pixels drop the measurement and re-fit inside the depth
//   test's early-out-gated path exactly as before. The only behavioral
//   change anywhere: the gate's fgThreshold under camera translation, now
//   the exact value instead of the rotation-only one.
//
// v5.6.3 (clipper, hypothesis-side): the spike's record branch is whitened
//   through the shrunk full metric; the winsor censors at the ingest's own
//   Satterthwaite dof (WinsorCapNuStar); the shrunk correlations are
//   computed ONCE and shared by the spike / winsor / corroboration / gate.
//   The metric, cell-mass and Studentization changes live in taaClip.h.hlsl
//   (see its v5.6.3 header). No depth/velocity/dilation behavior change.
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
        cornersRGB[c] = max(tex2Dlod(sceneTex, float4(tapUVs[c + 5], 0.0, 0.0)).rgb, 0.0);
}

// ============================================================================
// (v5.6.6) THE CLIP STATE, DECODE-THEN-BLENDED
// ----------------------------------------------------------------------------
// The state must describe the content the Kaiser read actually blended --
// a ~9-21 texel footprint -- not one coin-flipped texel. Four point-sampled
// fetches (the bilinear quad around the landing), each DECODED, then:
//   * the CONTINUOUS fields (sigma^2, varH -- linear space; varH's ratio
//     code decodes against ITS OWN texel's sigma first) blend with the
//     bilinear weights, renormalized over the valid texels;
//   * the DISCRETE transports (age: 4 bits, split: 6 bits -- blends of
//     packed fields are meaningless) come from the maximum-weight VALID
//     texel -- which is exactly the nearest, i.e. exactly the old snapped
//     read's choice;
//   * no valid texel -> COLD, as before.
// The raw bits are never interpolated and nothing is re-encoded. At an
// exact texel-center landing (the static identity) the weights degenerate
// to (1,0,0,0) and the read is BIT-IDENTICAL to the old snapped form --
// the blend engages only with sub-texel phase, never with motion state.
// Known approximation (accepted): a linear blend of varH still understates
// the blended accumulator's read variance (neighboring accumulators are
// correlated through the resample chain -- the same structure
// taaAccNoiseFactor prices); far closer than a snap.
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
    //
    // v5.6.2: the layer's forward displacement T_y is fitted BEFORE the gate
    // for dilation candidates: the gate's expected-crest depth is transported
    // with the SAME exact relation the depth test uses (w' = w*B + T_y; see
    // the composed rawExpClosest below). The fit runs on the PRE-revocation
    // layer -- for a dilation candidate that mask IS the foreground (the
    // crest and its kin), exactly the layer the crest depth belongs to. The
    // measurement is REUSED by the depth test (identical inputs -> identical
    // value), so non-revoked candidates pay ONE fit, not two; on revocation
    // the measurement is dropped and the depth test re-fits on the background
    // layer inside its own early-out-gated path (as it always did).
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

        // v5.6.2 (exactness): the expected stored raw of the crest, WITH the
        // translation. The rayLen ratio is the EXACT raw transport under a
        // pure rotation (raw(u) = rayLen(u)/D with D the Euclidean distance,
        // invariant under rotation about the eye -- so 1/rawExpRot = w*B
        // exactly), but the stored field is last frame's and the camera
        // TRANSLATES: the exact relation is w' = w*B + T_y, hence
        // rawExp = 1/(1/rawExpRot + ty). The old rotation-only form biased
        // fgThreshold by |ty|/w -- up to ~10% at close range under speed,
        // twice the default 5% tolerance: keep-biased under forward camera
        // motion (stale dilation bands surviving), revoke-biased under
        // retreat (silhouette shimmer). ty = 0 (fit off or failed)
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

            // The pre-gate fit measured the FOREGROUND layer; the revoked
            // pixel now transports its OWN background surface. Drop the
            // measurement: the depth test re-fits internally (behind its
            // early-out, exactly the pre-v5.6.2 code path).
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
        // (v5.6.6) decode-then-blend: the state describes the footprint
        // the Kaiser read actually blended. Bit-identical to the old
        // snapped read at exact texel centers (the static identity).
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

    // THE LANDING-MISS MODEL'S SCALE (v5.6): the soft clip's H1 is priced
    // by the measured miss distribution. sigma^2 = the gated transport
    // scatter (the known residual -- the same quantity the null's variance
    // charges; it is per-frame noise-like, wandering with the jitter, so
    // it is SPREAD, not offset: the mixture stays centered on the target
    // and there is no sign convention to get wrong) + the velocity
    // field's measured noise scale. Where the quant-step scan was skipped
    // (high motion), the uniform-field floor stands in -- at high motion
    // the miss scale is dominated by the measured residual anyway.
    float2 missSigmaPx;
    {
        float velNoisePx = (transportFloorPx > 0.0) ? transportFloorPx : kVelQuantFloorPx;
        missSigmaPx.x = sqrt(transportErrPx.x * transportErrPx.x + velNoisePx * velNoisePx);
        missSigmaPx.y = sqrt(transportErrPx.y * transportErrPx.y + velNoisePx * velNoisePx);
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

    // ---- the accumulated neighborhood (v5.6.6 ITEM 3): the soft clip's
    //      H1 components -- the history texels at the landing's 8
    //      neighbors, what a landing-miss ghost actually holds. Fetched
    //      only when the soft clip can run (its enable flag): ~8 fetches
    //      = ~10% of the resolve's fetch traffic with the posterior
    //      enabled. Exact texel centers through the linear slot-2
    //      sampler (the documented invariant); NO CompressGamut (raw
    //      stored texels are in-gamut by construction, unlike the
    //      Kaiser read); NO phase correction (stable-space content).
    //      Motion-agnostic: the prior odds do the gating, as before. ----
    float3 accumulatedNeighbors[8];
    if (taaSoftClip > 0.001)
    {
        float2 landingTexel = floor(repro.sampleUV * vp.sizePixels) + 0.5;
        [unroll]
        for (int aj = 1; aj < 9; ++aj)
        {
            float2 uv = clamp((landingTexel + kOffsets3x3[aj]) * vp.texelSize, vp.minUV, vp.maxUV);
            float3 acc = ToSpace(max(tex2Dlod(historyTex, float4(uv, 0.0, 0.0)).rgb, 0.0));
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
    //      consume the same values -- v5.6.3) ----
    float rXYg, rXZg, rYZg;
    ShrunkenCorrelations(colorStats, rXYg, rXZg, rYZg);

    // ---- change-points ----
    // The spike: two branches, both explicit. The record branch is a
    // WHITENED chi^2_3 test at ~1e-6 per frame through the shrunk full
    // metric (v5.6.3) -- the same channel law as the corroboration reset,
    // with the transport scatter charged per-channel inside the metric.
    // The old scalar-total form assumed exchangeable uncorrelated
    // channels: on luminance-dominated YCoCg content the 3-channel sum's
    // effective dof is ~1, and the priced ~1e-6 level ran at ~1e-3 there
    // (records resetting ~1000x the priced rate). The range branch is the
    // distribution-free garbage backstop (16x the observed AABB range --
    // effectively never on the null by design: the H-free ingest absorbed
    // the level-change job, geoCorroboration owns reveals, and the NaN
    // guard is separate).
    bool  spike = hadRecord
        && (innovSq > 16.0 * colorStats.rangeSq
            || WhitenedInnovSq(innov, partsLive, colorStats.transportVar, rXYg, rXZg, rYZg) > kChiSq3_1e6);

    bool geoCorroborated = false;
    if (hadRecord && disoccluded)
    {
        geoCorroborated = (WhitenedInnovSq(innov, partsLive, colorStats.transportVar, rXYg, rXZg, rYZg)
                           > kChiSq3_95);
    }

    bool  resetRecord = hadRecord
        && (spike || !(innovSq < kLargeValue) || geoCorroborated);
    bool  carryRecord = hadRecord && !resetRecord;

    // ---- THE STRUCTURE SPLIT'S TRANSPORT ----
    // The signed normalized fraction f = (R - s)/max(R, s): the spatial
    // residual total against the record (which IS s). With the exact
    // residual dof this anchor is unbiased -- the v5.5 ingest's fixed
    // point is e = 0. A fresh record certifies nothing (the seed >= R by
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
    }

    // ---- THE RECORD: THE H-FREE INGEST ----
    // Noise lives in dev = x0c - plane0 (H does not appear); change lives
    // in dCorr (it does not appear here). The de-bias is the exact
    // identity E[dev^2] = sPer + vPrior - 2*cPT. The winsor caps the
    // ingest at its OWN Satterthwaite dof's 0.1% point (v5.6.3 --
    // WinsorCapNuStar): the null bias is ~0.6% and a center-tap firefly's
    // record impact is half the old 9x cap's. The circularity (record ->
    // split -> tauB2 -> vPrior -> ingest -> record) has loop gain
    // (1 - q) < 1: convergent, honest fixed point e = 0 on the R > s
    // branch -- the exact residual dof is that statement's stability
    // condition.
    float sigmaStatSq, ageNext;
    {
        if (carryRecord)
        {
            // (v5.6.3) The honest 0.1% censoring point of the 3-channel
            // correlated sum -- the old chi^2_3/3 stand-in ran the point
            // at ~2% on gray content and systematically under-ingested
            // legitimate variance.
            float winsorCap = WinsorCapNuStar(partsLive, colorStats, splitPrev, rXYg, rXZg, rYZg);
            float3 devIng    = (currentColorSpace - colorStats.phaseShift) - colorStats.plane0;
            float3 tauB2Ing  = max(splitPrev, 0.0) * colorStats.residSq;
            float3 vPriorIng = tauB2Ing + partsLive.sPer * colorStats.evalVar;
            float3 sIngestPer = devIng * devIng - vPriorIng + 2.0 * colorStats.priorTapCov;
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
    }

    // ---- THE LEVEL GATE ----
    ClipGateResult clipGate = ClipHistoryToStatisticGate(
        historyColorSpace, currentColorSpace, colorStats, partsLive,
        carryRecord, carryRecord ? agePrev : 0.0,
        splitPrev,
        missSigmaPx,
        accumulatedNeighbors,
        rXYg, rXZg, rYZg);
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

    // (v5.6.7) The clip-excess blend reduction: how far the history
    // landed outside the gate (in gate radii) linearly drops the history
    // blend to zero past a start margin. Purely blend-side -- the clamp /
    // soft clip fix the STORED value, this fixes how much of it survives
    // this frame; the record and the split are untouched by policy, and
    // the VarH recursion sees the raised blend weight and correctly
    // restarts the accumulator on dropped pixels through its own
    // semantics. With the motion field off this is the disocclusion
    // path: ~2e-8 on the nominal null at the defaults, routine on true
    // reveals (which sit at 3-10+ radii).
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