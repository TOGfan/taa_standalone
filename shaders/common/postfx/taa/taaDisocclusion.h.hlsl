// ============================================================================
// TAA disocclusion: motion-field gate, history landing, layer forward-
// parallax fit, depth transport score, pursuit confirmation, velocity
// rejection
// ----------------------------------------------------------------------------
// DEPTH DISOCCLUSION: the previous-frame forward depth of a point at forward
// depth w on a current-frame ray d is EXACTLY w' = w * B + T_y, where B is
// the full relative-rotation factor from the two cbuffer bases and T_y is
// the layer's relative forward displacement. LATERAL translation provably
// never appears in w' (the measured velocity already carries its screen
// effect), so the whole transport has ONE unknown scalar. T_y is fitted
// per layer from the neighborhood's dejittered parallax (the pointwise
// relation K*p = T_perp - T_y*s' holds exactly at any motion magnitude),
// with an honest sigma from the fit residual. The one-sided test (only a
// history surface IN FRONT rejects) is asymmetric-safe: relative recession
// can never false-reject; only approach can -- exactly what T_y corrects.
// The tolerance is a measured error budget: depth quanta, surface slope x
// tracked-point position error (incl. the extrapolation displacement and
// the ACTUAL Kaiser filter footprint), plane curvature, and 2*sigma_Ty.
//
// PURSUIT: push the historical surface forward along its own (dejittered)
// motion into the CURRENT frame, then measure how much the current-frame
// velocity field diverges between the landing position and the current
// pixel. Both velocities are FIELD SAMPLES of the same content, so the
// coherent-motion round trip still cancels to ~0. The landing is resolved
// with ClassifyLayerSurface -- the SAME rules as everywhere else.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// samplers (historyMotionTex, depthTex, velocityTex), cbuffer perDraw
// (rejection thresholds, taaUseDepthDilation, taaUseMotionField, tan fov),
// postFx macros (tex2Dlod). Requires fragments included before:
// taaShared.h.hlsl, taaConstants.h.hlsl, taaVelocity.h.hlsl,
// taaLayers.h.hlsl.
// ============================================================================
#ifndef TAA_DISOCCLUSION_H_HLSL
#define TAA_DISOCCLUSION_H_HLSL

// ============================================================================
// DILATION-REVOCATION GATE READER (the stored quad at the candidate's landing)
// ----------------------------------------------------------------------------
// Own-History Dilation Validation. The landing's NEIGHBORHOOD record decides:
//   * DEAD-CENTER landing: the support is the center texel alone -- the
//     own-texel check. Static landings are the identity, so static
//     candidates validate through their OWN record: phase artifacts revoke
//     and STAY revoked; kept-band texels stay kept. This alone carries the
//     revocation persistence.
//   * OFF-CENTER landing: ANY tap of the degenerated 2x2 quad counts -- by
//     flag (>= 1: edge, or kept dilation band) OR by depth (at the object,
//     crest-anchored). NO layer-matching: a foreign foreground tap DOES
//     validate.
// gateDepth stays layer-gated: its only consumer (touchesAlreadyDilated)
// requires the center to be flag 2, where the gating reads crest depths.
// ============================================================================
struct HistoryMotionGate
{
    float  centerFlag;       // stored flag of the snapped landing texel
    float  gateDepth;        // layer-gated bilinear stored depth
    float  supportMaxFlag;   // support max flag (ANY quad tap)
    float  supportDepthMax;  // support max depth (ANY quad tap)
};

HistoryMotionGate SampleHistoryMotionGate(float2 historyUV, ViewportParams vp)
{
    HistoryMotionGate g;

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;

    float sx = (fracPx.x >= 0.0) ? 1.0 : -1.0;
    float sy = (fracPx.y >= 0.0) ? 1.0 : -1.0;

    float2 uv10 = clamp(snappedUV + float2(sx, 0.0) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv01 = clamp(snappedUV + float2(0.0, sy) * vp.texelSize, vp.minUV, vp.maxUV);
    float2 uv11 = clamp(snappedUV + float2(sx, sy) * vp.texelSize, vp.minUV, vp.maxUV);

    float4 m00 = tex2Dlod(historyMotionTex, float4(snappedUV, 0.0, 0.0));
    float4 m10 = tex2Dlod(historyMotionTex, float4(uv10, 0.0, 0.0));
    float4 m01 = tex2Dlod(historyMotionTex, float4(uv01, 0.0, 0.0));
    float4 m11 = tex2Dlod(historyMotionTex, float4(uv11, 0.0, 0.0));

    // Layer-gated bilinear depth (gateDepth's consumer requires center flag 2).
    bool fgCenter = (m00.w >= 0.75);
    bool match10  = ((m10.w >= 0.75) == fgCenter);
    bool match01  = ((m01.w >= 0.75) == fgCenter);
    bool match11  = ((m11.w >= 0.75) == fgCenter);

    float2 f   = abs(fracPx);
    float  w00 = (1.0 - f.x) * (1.0 - f.y);
    float  w10 = f.x * (1.0 - f.y) * (match10 ? 1.0 : 0.0);
    float  w01 = (1.0 - f.x) * f.y * (match01 ? 1.0 : 0.0);
    float  w11 = f.x * f.y * (match11 ? 1.0 : 0.0);
    float  invW = 1.0 / max(w00 + w10 + w01 + w11, 1e-4);
    g.gateDepth  = (m00.z * w00 + m10.z * w10 + m01.z * w01 + m11.z * w11) * invW;
    g.centerFlag = m00.w;

    // The support: ANY tap of the degenerated 2x2 quad -- flag OR depth, no
    // layer-matching. Dead-center landings degenerate to the center alone
    // (the static own-texel check).
    bool xOffCenter = (f.x >= kCenterLandingFracPx);
    bool yOffCenter = (f.y >= kCenterLandingFracPx);
    float supportDepthMax = m00.z;
    float supportMaxFlag  = m00.w;
    if (xOffCenter)               { supportDepthMax = max(supportDepthMax, m10.z); supportMaxFlag = max(supportMaxFlag, m10.w); }
    if (yOffCenter)               { supportDepthMax = max(supportDepthMax, m01.z); supportMaxFlag = max(supportMaxFlag, m01.w); }
    if (xOffCenter && yOffCenter) { supportDepthMax = max(supportDepthMax, m11.z); supportMaxFlag = max(supportMaxFlag, m11.w); }
    g.supportDepthMax = supportDepthMax;
    g.supportMaxFlag  = supportMaxFlag;

    return g;
}

// ============================================================================
// HISTORY LANDING SURFACE (shared by the depth & velocity disocclusion tests)
// ----------------------------------------------------------------------------
// Classified and resolved with the SAME rules as the current frame (the same
// extrapolation, on the stored field), PLUS the OWNERSHIP GATE: the stored
// flag is the POST-VALIDATION layer record. The stored band is a plateau of
// the crest's raw sample, so the landing-side extrapolation degenerates to
// the anchor value there automatically (w ~ 0 -> q ~ 0).
// ============================================================================
// Ownership-gated bilinear depth at the landing: never blend across an
// ownership boundary.
float LandingOwnedDepth(float depths[9], float flags[9], float2 fracPx)
{
    float d00 = depths[0];
    float d10 = (fracPx.x >= 0.0) ? depths[4] : depths[3];
    float d01 = (fracPx.y >= 0.0) ? depths[2] : depths[1];
    float d11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? depths[8] : depths[6])
        : ((fracPx.y >= 0.0) ? depths[7] : depths[5]);
    float f00 = flags[0];
    float f10 = (fracPx.x >= 0.0) ? flags[4] : flags[3];
    float f01 = (fracPx.y >= 0.0) ? flags[2] : flags[1];
    float f11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? flags[8] : flags[6])
        : ((fracPx.y >= 0.0) ? flags[7] : flags[5]);

    float2 f = abs(fracPx);
    bool  fg = (f00 >= 0.75);
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y) * (((f10 >= 0.75) == fg) ? 1.0 : 0.0);
    float w01 = (1.0 - f.x) * f.y * (((f01 >= 0.75) == fg) ? 1.0 : 0.0);
    float w11 = f.x * f.y * (((f11 >= 0.75) == fg) ? 1.0 : 0.0);
    float invW = 1.0 / max(w00 + w10 + w01 + w11, 1e-4);
    return (d00 * w00 + d10 * w10 + d01 * w01 + d11 * w11) * invW;
}

// The history landing resolved with the SAME layer semantics as the current
// frame (see SampleHistoryLandingSurface).
struct HistoryLandingSurface
{
    float  effectiveDepthRaw;
    float2 effectiveVelocityJitteredPrevUV;
    float2 gradRaw;              // stored-side minmod plane gradient (raw-z per texel)

    // Velocity-field shape at the landing (px).
    float maxCurvaturePx;
    float maxPairGradPx;
    float coherentGradPx;
    float snapDistPx;

    // Extrapolation diagnostics (landing side).
    float pairGradPx;        // the active landing extrapolation pair's gradient
    float shallowVelGradPx;  // the landing's measured shallow-end velocity gradient
};

HistoryLandingSurface SampleHistoryLandingSurface(
    float2 historyUV, ViewportParams vp, bool useDepthDilation, bool gather,
    float coherenceRadiusPx, bool currentIsForeground)
{
    HistoryLandingSurface h;
    h.effectiveDepthRaw               = 0.0;
    h.effectiveVelocityJitteredPrevUV = float2(0.0, 0.0);
    h.gradRaw          = float2(0.0, 0.0);
    h.maxCurvaturePx   = 0.0;
    h.maxPairGradPx    = 0.0;
    h.coherentGradPx   = 0.0;
    h.snapDistPx       = 0.0;
    h.pairGradPx       = 0.0;
    h.shallowVelGradPx = 0.0;

    // Minimal default when no disocclusion test needs the landing (debug
    // views only): the single nearest tap of the stored field. With the
    // motion field disabled nothing consumes the landing -- skip the fetch
    // entirely.
    if (!gather)
    {
        if (taaUseMotionField > 0.5)
        {
            float2 clampedUV = clamp(historyUV, vp.minUV, vp.maxUV);
            float4 mCenter   = tex2Dlod(historyMotionTex, float4(clampedUV, 0.0, 0.0));
            h.effectiveDepthRaw               = mCenter.z;
            h.effectiveVelocityJitteredPrevUV = mCenter.xy;
        }
        return h;
    }

    float2 pixelPos  = historyUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;    // sub-texel landing phase, [-0.5, 0.5]
    h.snapDistPx     = length(fracPx);

    float  depths[9];
    float2 velocities[9];
    float  flags[9];
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapUV = clamp(snappedUV + kOffsets3x3[i] * vp.texelSize, vp.minUV, vp.maxUV);
        float4 m     = tex2Dlod(historyMotionTex, float4(tapUV, 0.0, 0.0));
        depths[i]     = m.z;    // previous frame's depth (raw / baked band / revoked)
        velocities[i] = m.xy;   // previous frame's velocity
        flags[i]      = m.w;    // post-validation layer record
    }

    // THE classifier on the stored structure.
    SurfaceEdgeState landingEdge = AnalyzeSurfaceEdgesCore(depths, useDepthDilation);

    // --- Ownership gate ------------------------------------------------------
    bool centerFg = (flags[0] >= 0.75);
    if (!currentIsForeground && !centerFg)
    {
        // Background-resolved center (flat or REVOKED): resolve strictly as
        // background -- suppress the structural foreground pulls (and the
        // extrapolation with them).
        landingEdge.isDilationZone   = false;
        landingEdge.isForegroundEdge = false;
    }

    LayerSurface landingLayer = ClassifyLayerSurface(
        depths, velocities, fracPx, vp.sizePixels, coherenceRadiusPx,
        useDepthDilation, true, landingEdge);

    if (!currentIsForeground && !centerFg)
    {
        // The depth must never blend across the ownership boundary.
        landingLayer.effectiveDepth = LandingOwnedDepth(depths, flags, fracPx);
    }

    h.effectiveDepthRaw               = landingLayer.effectiveDepth;
    h.effectiveVelocityJitteredPrevUV = landingLayer.effectiveVelocityUV;
    h.pairGradPx                      = landingLayer.pairGradPx;
    h.shallowVelGradPx                = landingLayer.shallowVelGradPx;
    h.gradRaw                         = float2(landingLayer.gradX, landingLayer.gradY);

    // Field shape + velocity-coherent noise, anchored at the effective layer,
    // with layer-consistent differences.
    MeasureVelocityFieldShapeCoherent(velocities, vp.sizePixels, coherenceRadiusPx, h.maxCurvaturePx, h.maxPairGradPx);

    [unroll]
    for (int m = 0; m < 9; ++m)
    {
        float2 deltaPx = (velocities[m] - h.effectiveVelocityJitteredPrevUV) * vp.sizePixels;
        float  lenPx   = length(deltaPx);
        if (lenPx <= coherenceRadiusPx)
        {
            float distPx = max(length(kOffsets3x3[m]), 1.0);
            h.coherentGradPx = max(h.coherentGradPx, lenPx / distPx);
        }
    }
    return h;
}

// ============================================================================
// LAYER FORWARD-PARALLAX FIT (the depth transport's single unknown)
// ----------------------------------------------------------------------------
// T_y is measured from the layer's own texels: each texel's dejittered
// parallax p_i = v_i - g_i (g: the exact relative-rotation offset field,
// g(u) = G(u) - u, G = F_prev^-1 o F_cur) satisfies the pointwise linear
// relation K_i * p_i = T_perp - T_y * s'_i (K = w*B, s' = prev-frame pos)
// exactly at any motion magnitude. Regressing (K*p) against s' over the
// layer's texels yields T_y (the slope) plus an honest sigma from the fit
// residual. Run in PIXEL units: the per-axis angular scale cancels through
// the per-channel intercept, so T_y comes out in world units. Layer mask:
// velocity coherence with the resolved layer's effective field plus the same
// depth-side rule as MeasureShallowForegroundGeometry -- dilation centers
// are background (foreground-side taps only), edge centers are the crest
// (not-behind taps only), flat is velocity-only.
// ============================================================================
bool EstimateLayerForwardParallax(
    float2 refVelocityUV,          // the resolved layer's effective (jittered) velocity
    float2 frameBaseUV,            // the stable point's current-frame position
    float2 stableUV,
    float  depthRaw[9],
    float2 velocityUV[9],
    float2 tapUVs[9],
    bool   isDilationZone,
    bool   isForegroundEdge,
    CameraBasis currentCamera,
    CameraBasis previousCamera,
    float  fitRadiusPx,
    ViewportParams vp,
    out float ty,
    out float tySigma)
{
    ty = 0.0;
    tySigma = 0.0;

    // Exact relative-rotation offset field, linearized at the pixel.
    // G(u0) is exact (F_cur(frameBase) = stableUV); the neighborhood uses the
    // analytic Jacobian (2nd-order error ~1e-4 px, far below quantization).
    float2 uRot0 = InverseReprojectThroughCamera(stableUV, previousCamera, taaTanHalfFovX, taaTanHalfFovY);
    float4 jG    = Mul2x2(Inv2x2(CameraForwardJacobian(uRot0,     previousCamera, taaTanHalfFovX, taaTanHalfFovY)),
                               CameraForwardJacobian(frameBaseUV, currentCamera, taaTanHalfFovX, taaTanHalfFovY));
    float4 mRot  = float4(jG.x - 1.0, jG.y, jG.z, jG.w - 1.0);   // J_G - I
    float2 g0    = uRot0 - frameBaseUV;                           // g(u0)

    float3 prevForward = CameraForwardAxis(previousCamera);
    float2 px = vp.sizePixels;

    // dot(BuildCameraRay(u), prevForward) is linear in u: the per-tap dot
    // collapses to two mads against these hoisted axis projections.
    float pfP = dot(currentCamera.rightTanFov, prevForward);
    float pfQ = dot(currentCamera.forward,     prevForward);
    float pfR = dot(currentCamera.downTanFov,  prevForward);

    // Center tap: the subtraction origin. Any fixed point works -- the
    // closed-form OLS below is shift-invariant -- but removing the common
    // offset conditions the sums: without it, sumS2 - n*mean^2 cancels
    // ~1e6-magnitude terms down to ~10 and float32 loses the variance.
    float  dotC = tapUVs[0].x * pfP + pfQ - tapUVs[0].y * pfR;
    float  kC   = (1.0 / max(depthRaw[0], kEpsilon)) * dotC;
    float2 gC   = g0 + Apply2x2(mRot, tapUVs[0] - frameBaseUV);
    float2 sC   = (tapUVs[0] + velocityUV[0]) * px;
    float2 GC   = kC * ((velocityUV[0] - gC) * px);

    float  n = 0.0;
    float2 sumS = float2(0.0, 0.0), sumG = float2(0.0, 0.0);
    float  sumS2 = 0.0, sumSG = 0.0, sumG2 = 0.0;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        // Layer mask, part 1: velocity coherence with the resolved layer.
        // NaN-robust form: garbage input must fall OUT of the mask -- with
        // NaN, (len > r) evaluates false and would INCLUDE the tap,
        // poisoning the fit; (!(len <= r)) excludes it.
        if (!(length((velocityUV[i] - refVelocityUV) * px) <= fitRadiusPx))
            continue;
        // Layer mask, part 2: the depth-side rule (see the section header).
        if (isDilationZone)        { if (i == 0 || depthRaw[i] <= depthRaw[0]) continue; }
        else if (isForegroundEdge) { if (depthRaw[i] <  depthRaw[0])           continue; }

        // B_i's cur-forward denominator is 1 +- the sub-pixel jitter only.
        float  dotD = tapUVs[i].x * pfP + pfQ - tapUVs[i].y * pfR;
        float  k    = (1.0 / max(depthRaw[i], kEpsilon)) * dotD;
        float2 g    = g0 + Apply2x2(mRot, tapUVs[i] - frameBaseUV);
        float2 s    = (tapUVs[i] + velocityUV[i]) * px - sC;
        float2 G    = k * ((velocityUV[i] - g) * px) - GC;

        sumS += s;  sumG += G;
        sumS2 += dot(s, s);
        sumSG += dot(s, G);
        sumG2 += dot(G, G);
        n += 1.0;
    }

    if (n < 3.0) return false;

    float  invN  = 1.0 / n;
    float2 meanS = sumS * invN;
    float2 meanG = sumG * invN;
    float  varSum = sumS2 - n * dot(meanS, meanS);
    if (varSum < kMinParallaxInfoPx2) return false;

    float  covSum = sumSG - n * dot(meanS, meanG);
    float  yySum  = sumG2 - n * dot(meanG, meanG);

    // The model: G = a - T_y * s (a: per-channel intercept = rescaled T_perp).
    ty = -covSum / varSum;

    float residSS = max(yySum - covSum * covSum / varSum, 0.0);
    float dof     = max(2.0 * n - 3.0, 1.0);
    tySigma       = sqrt(residSS / dof) / sqrt(varSum);
    return true;
}

// ============================================================================
// MOTION-COMPENSATED DEPTH DISOCCLUSION (geometric transport)
// ----------------------------------------------------------------------------
// Returns a continuous score; >= 1.0 rejects, and saturate((1-score)*2) is
// the depthGate.
// ============================================================================
float ComputeDepthDisocclusionScore(
    float  effectiveDepthRaw,
    float2 effectiveVelocityUV,
    float2 frameBaseUV,
    float2 stableUV,
    float  depthRaw[9],
    float2 velocityJitteredUV[9],
    float2 tapUVs[9],
    float  landingDepthRaw,
    float2 landingGradRaw,
    float2 jitterResidualPx,
    float2 quadVelocitySpreadUV,
    float  foregroundSlope,
    float2 closestOffsetPx,
    float  extrapolationDispPx,
    bool   isDilationZone,
    bool   isForegroundEdge,
    float  layerGradX,
    float  layerGradY,
    float  depthNoiseFloor,
    float  depthQuantStep,
    CameraBasis currentCamera,
    CameraBasis previousCamera,
    float  fitRadiusPx,
    float  historySupportTexels,
    ViewportParams vp)
{
    if (taaDepthRejection <= 0.001) return 0.0;

    // --- the transport -----------------------------------------------------
    float  wCur = 1.0 / max(effectiveDepthRaw, kEpsilon);
    float3 dCur = frameBaseUV.x * currentCamera.rightTanFov + currentCamera.forward
                - frameBaseUV.y * currentCamera.downTanFov;
    float  bCur = dot(dCur, CameraForwardAxis(previousCamera))
               / max(dot(dCur, CameraForwardAxis(currentCamera)), 1e-6);
    float  wHist = 1.0 / max(landingDepthRaw, kEpsilon);

    // --- early-out ----------------------------------------------------------
    // T_y is clamped to [-wCur, wCur], so wExp <= wCur * bCur + wCur
    // unconditionally. If even that cannot exceed wHist, the history is
    // provably too far BEHIND to ever reject (the test is one-sided: only a
    // history IN FRONT rejects) -- skip the parallax fit entirely.
    if (wCur * bCur + wCur <= wHist) return 0.0;

    float ty, tySigma;
    bool  tyMeasured = EstimateLayerForwardParallax(
        effectiveVelocityUV, frameBaseUV, stableUV,
        depthRaw, velocityJitteredUV, tapUVs,
        isDilationZone, isForegroundEdge,
        currentCamera, previousCamera, fitRadiusPx, vp, ty, tySigma);

    if (!tyMeasured)
    {
        // Unmeasurable (isolated fragment): the CPU prior if provided, else
        // assume the motion is below the fit's own detection floor.
        ty      = taaDepthParallaxStep;
        tySigma = kTyUnmeasuredFrac * wCur;
    }
    ty = clamp(ty, -wCur, wCur);      // degenerate-guard, not a tuning knob
    float wExp = max(wCur * bCur + ty, 1e-4);

    // --- the one-sided occlusion gap ----------------------------------------
    float gap = wExp - wHist;         // > 0: the history surface is IN FRONT
    if (gap <= 0.0) return 0.0;

    // --- tolerance: the measured error budget --------------------------------
    bool isFlat = !isDilationZone && !isForegroundEdge;

    // (a)+(b) representation noise: both sides' depth quanta.
    float noiseZ = max(depthQuantStep, depthNoiseFloor);
    float quantW = 2.0 * noiseZ * wExp * wExp;

    // (c) tracked-point position uncertainty x the surface's slope.
    float slopeZ = isFlat ? max(abs(layerGradX), abs(layerGradY)) : foregroundSlope;
    slopeZ = max(slopeZ, max(abs(landingGradRaw.x), abs(landingGradRaw.y)));
    float slopeW = slopeZ * wExp * wExp;   // raw-z per px -> forward units per px

    float2 velSpreadPx = quadVelocitySpreadUV * vp.sizePixels;
    float reachPx = length(jitterResidualPx)
                  + 0.5 * length(velSpreadPx)
                  + extrapolationDispPx
                  + (isDilationZone ? length(closestOffsetPx) : 0.0)
                  + (isFlat ? 0.0 : historySupportTexels - 0.5);

    // (d) deviation from the local plane (curved interiors).
    float curvZ = 0.0;
    if (isFlat)
    {
        float curvH = abs(depthRaw[0] - 0.5 * (depthRaw[3] + depthRaw[4]));
        float curvV = abs(depthRaw[0] - 0.5 * (depthRaw[1] + depthRaw[2]));
        curvZ = max(curvH, curvV);
    }
    float curvW = curvZ * wExp * wExp;

    // (e) the parallax estimate's own error (2 sigma, ~97% one-sided).
    float sigmaW = 2.0 * abs(tySigma);

    float tolW = taaDepthRejection * wExp + quantW + slopeW * reachPx + curvW + sigmaW;
    return gap / max(tolW, 1e-9);
}

// ============================================================================
// PURSUIT CONFIRMATION (current-frame, layer-consistent velocity divergence)
// ============================================================================
bool PursuitConfirmsDivergence(
    float2 historySampleUV,
    float2 prevVelocityEffectiveJitteredUV,
    float2 curVelocityJitteredUV,
    float2 jitterTransportUV,
    float2 missVecPx,             // dejittered step-1 error vector == the transport's landing miss
    float  depthGate,             // [0..1] depth-transport same-surface confidence
    float  coherenceRadiusPx,
    ViewportParams vp,
    out float divergencePx)
{
    divergencePx = 0.0;

    if (taaCrossTestStrength <= 0.001)
        return false;

    // 1) Exact landing: where the history surface is in the CURRENT render.
    float2 pursuitUV = historySampleUV - (prevVelocityEffectiveJitteredUV + jitterTransportUV);
    if (any(pursuitUV < vp.minUV) || any(pursuitUV > vp.maxUV))
        return false;

    // 2) 3x3 depth/velocity neighborhood at the landing, resolved with the
    //    same layer rules as the current frame.
    float2 pixelPos  = pursuitUV * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    float2 snappedUV = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    float2 fracPx    = pixelPos - baseTexel;

    float  depths[9];
    float2 velocities[9];
    [unroll]
    for (int k = 0; k < 9; ++k)
    {
        float2 tapUV = clamp(snappedUV + kOffsets3x3[k] * vp.texelSize, vp.minUV, vp.maxUV);
        depths[k]     = tex2Dlod(depthTex,    float4(tapUV, 0.0, 0.0)).r;
        velocities[k] = tex2Dlod(velocityTex, float4(tapUV, 0.0, 0.0)).rg;
    }

    SurfaceEdgeState landingEdge = AnalyzeSurfaceEdgesCore(depths, taaUseDepthDilation > 0.5);
    LayerSurface landingLayer = ClassifyLayerSurface(
        depths, velocities, fracPx, vp.sizePixels, coherenceRadiusPx,
        taaUseDepthDilation > 0.5, true, landingEdge);
    float2 landingVelocityJitteredUV = landingLayer.effectiveVelocityUV;

    // 3) Divergence VECTOR between the two current-frame velocities.
    float2 divergenceVecPx = (landingVelocityJitteredUV - curVelocityJitteredUV) * vp.sizePixels;

    // 3b) First-order advection correction (layer-consistent estimators, no
    //     single-layer requirement, honest miss cap).
    float maxCurvaturePx, maxPairGradPx;
    MeasureVelocityFieldShapeCoherent(velocities, vp.sizePixels, coherenceRadiusPx, maxCurvaturePx, maxPairGradPx);

    float landingSpreadPx = 0.0;
    [unroll]
    for (int m = 0; m < 9; ++m)
    {
        float2 deltaPx = (velocities[m] - landingVelocityJitteredUV) * vp.sizePixels;
        if (length(deltaPx) <= coherenceRadiusPx)
            landingSpreadPx = max(landingSpreadPx, length(deltaPx));
    }

    bool landingContinuous = IsContinuousVelocityField(maxCurvaturePx, maxPairGradPx);
    bool advectionGated    = landingContinuous && (depthGate > 0.001);

    float2 correctedDivergenceVecPx = divergenceVecPx;
    float  residualPx = 0.0;
    if (advectionGated)
    {
        float4 Jlanding = EstimateVelocityJacobianCoherentPx(velocities, vp.sizePixels, coherenceRadiusPx);
        correctedDivergenceVecPx -= ApplyVelocityJacobian(Jlanding, missVecPx) * depthGate;

        float2 vLandingPx = landingVelocityJitteredUV * vp.sizePixels;
        float  missCapPx  = length(ApplyVelocityJacobian(Jlanding, vLandingPx)) * kMissCapHeadroom;
        residualPx = ResidualAdvectionBudgetPx(maxPairGradPx, length(missVecPx), length(fracPx), missCapPx) * depthGate;
    }

    divergencePx = length(correctedDivergenceVecPx);
    float tolerancePx = kPursuitVelBaseTolerancePx + landingSpreadPx + residualPx;

    return (divergencePx * saturate(taaCrossTestStrength)) > tolerancePx;
}

// ============================================================================
// VELOCITY REJECTION (step 2 of disocclusion; fully independent of depth)
// ============================================================================
// Velocity rejection (pursuit) diagnostics.
struct VelocityRejectionResult
{
    bool  rejected;
    float errorRatio;
    float divergencePx;
};

VelocityRejectionResult EvaluateVelocityRejection(
    float2 historySampleUV,
    float2 resolvedVelocityJitteredUV,
    HistoryLandingSurface landing,
    bool   currentSingleLayer,
    float  currentDilationPairGradPx,   // the current side's active extrapolation pair gradient
    float  currentShallowVelGradPx,     // the current side's shallow-end velocity gradient
    float  depthGate,
    float2 jitterCancelUV,
    float2 jitterTransportUV,
    float2 neighborVelocityJitteredUV[9],
    float  coherenceRadiusPx,
    ViewportParams vp)
{
    VelocityRejectionResult r;
    r.rejected     = false;
    r.errorRatio   = 0.0;
    r.divergencePx = 0.0;

    if (taaVelRejection <= 0.001)
        return r;

    // Dejittered error VECTOR against the landing's EFFECTIVE layer. On a
    // same-surface pixel this vector IS the pursuit transport's landing miss
    // (identity), so it is passed through to the first-order correction.
    float2 velocityErrorVecPx = (resolvedVelocityJitteredUV - landing.effectiveVelocityJitteredPrevUV - jitterCancelUV) * vp.sizePixels;
    float  velocityErrorPx    = length(velocityErrorVecPx);

    // Velocity-coherent gradients on both frame sides, combined and clamped.
    // The current-frame scan is deferred: the combined noise is the max of
    // the two sides, so when the raw error is below the landing-side bound
    // the ratio cannot exceed 1 regardless of the current side (and the
    // pursuit cannot fire) -- quiet pixels skip the scan.
    float layerGradientPx = clamp(landing.coherentGradPx, 0.0, 1.0);
    if (velocityErrorPx > taaVelRejection + layerGradientPx * taaVelGradientScale)
    {
        float currentGradientPx = MeasureVelocityCoherentGradientPx(
            resolvedVelocityJitteredUV, neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
        layerGradientPx = max(layerGradientPx, clamp(currentGradientPx, 0.0, 1.0));
    }

    float velocityNoise = layerGradientPx * taaVelGradientScale;

    // Predicted same-surface advection for the alert tolerance. For
    // !currentSingleLayer, the bound takes the max over every gradient
    // source: the stored band replicates the crest's per-row samples, so the
    // landing-side pair gradient loses the perpendicular component; the
    // current side still measures both; and the ACTIVE extrapolation pairs +
    // both sides' shallow-end geometry carry the near-silhouette gradients.
    // The bound only matters when the raw error can exceed the base
    // tolerance: below that the ratio stays under 1 for any advection
    // value, so the field-shape measurements and the Jacobian are skipped
    // on quiet pixels.
    float advectionPx = 0.0;
    if (velocityErrorPx > taaVelRejection + velocityNoise)
    {
        // Layer-consistent field shapes.
        float curMaxCurvaturePx, curMaxPairGradPx;
        MeasureVelocityFieldShapeCoherent(neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx, curMaxCurvaturePx, curMaxPairGradPx);
        bool currentContinuous = IsContinuousVelocityField(curMaxCurvaturePx, curMaxPairGradPx);
        bool landingContinuous = IsContinuousVelocityField(landing.maxCurvaturePx, landing.maxPairGradPx);

        if (currentContinuous && currentSingleLayer)
        {
            float4 Jcur    = EstimateVelocityJacobianCoherentPx(neighborVelocityJitteredUV, vp.sizePixels, coherenceRadiusPx);
            float2 vCurPx  = resolvedVelocityJitteredUV * vp.sizePixels; // jitter is sub-pixel: irrelevant at this scale
            float  predictedPx = length(ApplyVelocityJacobian(Jcur, vCurPx));
            float  residualPx  = ResidualAdvectionBudgetPx(
                max(curMaxPairGradPx, landing.maxPairGradPx), velocityErrorPx, landing.snapDistPx,
                predictedPx * kMissCapHeadroom);
            advectionPx = (predictedPx + residualPx) * depthGate;
        }
        else if (!currentSingleLayer && landingContinuous)
        {
            float boundGradPx = max(max(landing.maxPairGradPx, curMaxPairGradPx),
                                    max(max(landing.pairGradPx, currentDilationPairGradPx),
                                        max(landing.shallowVelGradPx, currentShallowVelGradPx)));
            float2 vEffPx      = resolvedVelocityJitteredUV * vp.sizePixels;
            float  predictedPx = boundGradPx * length(vEffPx);
            float  residualPx  = ResidualAdvectionBudgetPx(
                boundGradPx, velocityErrorPx, landing.snapDistPx,
                predictedPx * kMissCapHeadroom);
            advectionPx = (predictedPx + residualPx) * depthGate;
        }
    }

    r.errorRatio = velocityErrorPx / max(taaVelRejection + velocityNoise + advectionPx, 1e-4);

    if (r.errorRatio > 1.0)
    {
        // Error above tolerance: confirm with a pure current-frame pursuit of
        // the landing's effective (dominant) surface.
        r.rejected = PursuitConfirmsDivergence(
            historySampleUV, landing.effectiveVelocityJitteredPrevUV, resolvedVelocityJitteredUV,
            jitterTransportUV, velocityErrorVecPx, depthGate, coherenceRadiusPx, vp, r.divergencePx);
    }
    return r;
}

#endif // TAA_DISOCCLUSION_H_HLSL