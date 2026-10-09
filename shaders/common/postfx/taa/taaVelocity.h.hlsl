// ============================================================================
// TAA velocity-field math: layer coherence, quad selection, the Jacobian
// advection budget, the measured pair rules and the two-tap similarity
// extrapolation
// ----------------------------------------------------------------------------
// PURE: takes everything as parameters -- no cbuffer, no samplers. The layer
// policy (taLayers.h.hlsl) builds on this; the disocclusion tests consume the
// field-shape estimators.
//
// TWO-TAP SIMILARITY VELOCITY EXTRAPOLATION:
//   A dilation zone acts as the foreground it dilates to, and its velocity is
//   that layer's FIELD EVALUATED AT THIS PIXEL. Interpolation cannot reach
//   it, so the field is extrapolated from two same-surface foreground taps
//   under a similarity (rotation + uniform scale) prior -- two point
//   correspondences determine a 2D similarity exactly, and it reproduces the
//   perpendicular/rotational gradient of the parallax field that a plain
//   linear extrapolation misses. Treating 2D px vectors as complex numbers,
//   with tap offsets f1, f2 and velocities v1, v2:
//       q    = (v2 - v1) / (f2 - f1)      complex quotient: rotation+scale
//       v(B) = v1 + q * (B - f1)          B = exact sub-texel position
//   BOUNDED GAIN by construction: |v(B) - v1| = |q| * |B - f1| exactly.
//   JITTER-CLEAN: q is a same-frame difference; static scenes give q ~ 0,
//   so the identity landing and the gate's anchor are stable. Gates:
//   the geometric same-surface rule and the rigid parallax magnitude
//   ceiling -- both measured -- return the fallback (the anchor tap's raw
//   velocity) on any failure.
//   EVALUATION ENDPOINT: v(B) is evaluated from the pair endpoint NEAREST B.
//   Under the similarity prior both endpoints give the identical v(B) (two
//   correspondences determine the field exactly), so this is a no-op inside
//   the model; under real field curvature the extrapolation error grows
//   with the lever arm |B - f_e|, so the nearer endpoint is never worse and
//   usually tighter. The dilation call benefits most: its anchor (the crest)
//   can sit ~2 texels from B while the partner is nearer.
//
// MEASURED PAIR RULES:
//   * ForegroundPairSameSurface: two foreground taps are ONE surface iff
//     their raw gap is explained by the surface's own measured slope over
//     their distance, plus quantization.
//   * ForegroundPairRigidMagnitude: a pair's velocity difference is
//     legitimate iff it is explained by parallax through the pair's OWN
//     relative depth gap, plus the field's coherent gradient, plus the
//     measurement floor.
//   * MeasureShallowForegroundGeometry: the foreground object's shallow-end
//     geometry from ADJACENT tap pairs (span <= sqrt2). On a DILATION zone
//     the center is the background -- strictly foreground-side only; on a
//     CREST / flat pixel the center IS the surface's silhouette-side tap --
//     foreground-INCLUSIVE.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires fragments
// included before: taaShared.h.hlsl (kOffsets3x3, Bilerp2x2),
// taaConstants.h.hlsl.
// ============================================================================
#ifndef TAA_VELOCITY_H_HLSL
#define TAA_VELOCITY_H_HLSL

// ============================================================================
// Field shape & layer-coherent gradients
// ============================================================================
void MeasureVelocityFieldShape(
    float2 velocityJitteredUV[9], float2 sizePixels,
    out float maxCurvaturePx, out float maxPairGradPx)
{
    float2 curvH  = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[3] + velocityJitteredUV[4])) * sizePixels;
    float2 curvV  = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[1] + velocityJitteredUV[2])) * sizePixels;
    float2 curvD1 = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[5] + velocityJitteredUV[8])) * sizePixels;
    float2 curvD2 = (velocityJitteredUV[0] - 0.5 * (velocityJitteredUV[6] + velocityJitteredUV[7])) * sizePixels;
    // Squared-domain reductions: the max of lengths is the sqrt of the max
    // of dots (monotone); the pair grads' constant scale factors fold into
    // their squared forms ((0.5)^2 = 1/4; (0.25*sqrt2)^2 = 1/8).
    maxCurvaturePx = sqrt(max(max(dot(curvH, curvH), dot(curvV, curvV)),
                             max(dot(curvD1, curvD1), dot(curvD2, curvD2))));

    float2 dX  = (velocityJitteredUV[4] - velocityJitteredUV[3]) * sizePixels;
    float2 dY  = (velocityJitteredUV[2] - velocityJitteredUV[1]) * sizePixels;
    float2 dD1 = (velocityJitteredUV[8] - velocityJitteredUV[5]) * sizePixels;
    float2 dD2 = (velocityJitteredUV[7] - velocityJitteredUV[6]) * sizePixels;
    maxPairGradPx = sqrt(max(max(dot(dX, dX) * 0.25, dot(dY, dY) * 0.25),
                             max(dot(dD1, dD1) * 0.125, dot(dD2, dD2) * 0.125)));
}

// Continuous (locally linear) velocity field: the smooth-surface correction
// below is granted ONLY here. A layer boundary keeps the raw signal.
bool IsContinuousVelocityField(float maxCurvaturePx, float maxPairGradPx)
{
    return (maxCurvaturePx <= kVelDiscontinuityRatio * maxPairGradPx + kVelDiscontinuityAbsPx);
}

// Max velocity change per pixel between the anchor and taps that move together
// with it (|delta| within the coherence radius). Cross-surface velocity
// differences -- the disocclusion signal itself -- are excluded.
float MeasureVelocityCoherentGradientPx(
    float2 anchorVelocityJitteredUV,
    float2 neighborVelocityJitteredUV[9],
    float2 sizePixels,
    float  coherenceRadiusPx)
{
    // Squared-domain coherence test AND reduction: the per-tap gradient is
    // sqrt(deltaSq) / dist with dist a compile-time stencil constant
    // (kOffsets3x3InvDistSq, taaShared), and the max over monotone
    // transforms collapses -- accumulate deltaSq * invDistSq and pay ONE
    // sqrt at the end. NaN deltas compare false and are excluded.
    float rSq = coherenceRadiusPx * coherenceRadiusPx;
    float gradientSqMax = 0.0;
    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float2 deltaPx = (neighborVelocityJitteredUV[i] - anchorVelocityJitteredUV) * sizePixels;
        float  deltaSq = dot(deltaPx, deltaPx);
        if (deltaSq <= rSq)
            gradientSqMax = max(gradientSqMax, deltaSq * kOffsets3x3InvDistSq[i]);
    }
    return sqrt(gradientSqMax);
}

// Field shape with layer-consistent differences.
void MeasureVelocityFieldShapeCoherent(
    float2 v[9], float2 sizePixels, float coherenceRadiusPx,
    out float maxCurvaturePx, out float maxPairGradPx)
{
    // Squared-domain coherence mask; the reductions below are squared too
    // (the max of lengths is the sqrt of the max of dots -- monotone; the
    // masked terms are zero vectors and contribute nothing).
    float rSq = coherenceRadiusPx * coherenceRadiusPx;
    bool ok[9];
    ok[0] = true;
    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        float2 dPx = (v[i] - v[0]) * sizePixels;
        ok[i] = dot(dPx, dPx) <= rSq;
    }

    float2 curvH  = (ok[3] && ok[4]) ? (v[0] - 0.5 * (v[3] + v[4])) * sizePixels : float2(0.0, 0.0);
    float2 curvV  = (ok[1] && ok[2]) ? (v[0] - 0.5 * (v[1] + v[2])) * sizePixels : float2(0.0, 0.0);
    float2 curvD1 = (ok[5] && ok[8]) ? (v[0] - 0.5 * (v[5] + v[8])) * sizePixels : float2(0.0, 0.0);
    float2 curvD2 = (ok[6] && ok[7]) ? (v[0] - 0.5 * (v[6] + v[7])) * sizePixels : float2(0.0, 0.0);
    maxCurvaturePx = sqrt(max(max(dot(curvH, curvH), dot(curvV, curvV)),
                             max(dot(curvD1, curvD1), dot(curvD2, curvD2))));

    // The pair grads' constant scale factors fold into their squared forms
    // ((0.5)^2 = 1/4; (0.25*sqrt2)^2 = 1/8); masked pairs are zero vectors.
    float2 dX  = (ok[3] && ok[4]) ? (v[4] - v[3]) * sizePixels : float2(0.0, 0.0);
    float2 dY  = (ok[1] && ok[2]) ? (v[2] - v[1]) * sizePixels : float2(0.0, 0.0);
    float2 dD1 = (ok[5] && ok[8]) ? (v[8] - v[5]) * sizePixels : float2(0.0, 0.0);
    float2 dD2 = (ok[6] && ok[7]) ? (v[7] - v[6]) * sizePixels : float2(0.0, 0.0);
    maxPairGradPx = sqrt(max(max(dot(dX, dX) * 0.25, dot(dY, dY) * 0.25),
                             max(dot(dD1, dD1) * 0.125, dot(dD2, dD2) * 0.125)));
}

// ============================================================================
// Phase-quad selection (the sub-texel velocity interpolation)
// ============================================================================
// The 2x2 quad the sub-texel phase points into (shared by the current frame's
// jitter phase and every landing's sub-texel phase).
void SelectPhaseQuad(
    float2 velocityJitteredUV[9], float2 phasePx,
    out float2 v00, out float2 v10, out float2 v01, out float2 v11)
{
    v00 = velocityJitteredUV[0];
    v10 = (phasePx.x >= 0.0) ? velocityJitteredUV[4] : velocityJitteredUV[3];
    v01 = (phasePx.y >= 0.0) ? velocityJitteredUV[2] : velocityJitteredUV[1];
    v11 = (phasePx.x >= 0.0)
        ? ((phasePx.y >= 0.0) ? velocityJitteredUV[8] : velocityJitteredUV[6])
        : ((phasePx.y >= 0.0) ? velocityJitteredUV[7] : velocityJitteredUV[5]);
}

// Max pairwise velocity step inside a 2x2 quad, SQUARED, in px^2. The max of
// lengths equals the sqrt of the max of squared lengths (monotone), so the
// straddle pre-screen compares against the squared radius and pays a single
// sqrt only when it trips: six sqrt -> six dot on EVERY flat pixel.
float QuadVelocityStepSq(float2 v00, float2 v10, float2 v01, float2 v11, float2 sizePixels)
{
    float2 d10 = (v10 - v00) * sizePixels;
    float2 d01 = (v01 - v00) * sizePixels;
    float2 d11 = (v11 - v00) * sizePixels;
    float2 d1x = (v11 - v10) * sizePixels;
    float2 dx1 = (v11 - v01) * sizePixels;
    float2 dxx = (v10 - v01) * sizePixels;
    return max(max(dot(d10, d10), dot(d01, d01)),
               max(max(dot(d11, d11), dot(d1x, d1x)), max(dot(dx1, dx1), dot(dxx, dxx))));
}

// Does the phase-selected 2x2 quad straddle a velocity layer step? BOTH
// required: (a) the 3x3 field is DISCONTINUOUS here; (b) the step is INSIDE
// the selected quad and beyond what the center's own (velocity-coherent)
// layer explains -- never the step itself. Cheap pre-screen first: the
// straddle condition requires quadStep to exceed the coherence radius on its
// own, so a quiet quad skips the field-shape measurement entirely. NaN-safe:
// a NaN quadStep falls through to the full test, which rejects it the same
// way as before.
bool QuadStraddlesVelocityStep(
    float2 v00, float2 v10, float2 v01, float2 v11,
    float2 velocityJitteredUV[9], float2 sizePixels, float coherenceRadiusPx)
{
    float quadStepSq = QuadVelocityStepSq(v00, v10, v01, v11, sizePixels);
    if (quadStepSq <= coherenceRadiusPx * coherenceRadiusPx)
        return false;
    // NaN-safe exactly as before: a NaN quadStep fails the <= test and falls
    // through to the full test, which rejects it identically.
    float quadStepPx = sqrt(quadStepSq);

    float maxCurvaturePx, maxPairGradPx;
    MeasureVelocityFieldShape(velocityJitteredUV, sizePixels, maxCurvaturePx, maxPairGradPx);
    if (IsContinuousVelocityField(maxCurvaturePx, maxPairGradPx))
        return false;

    float coherentGradPx = MeasureVelocityCoherentGradientPx(
        velocityJitteredUV[0], velocityJitteredUV, sizePixels, coherenceRadiusPx);
    return quadStepPx > kQuadStepGradMul * coherentGradPx + coherenceRadiusPx;
}

// Bilinear velocity at the exact sub-texel phase -- unless the selected quad
// straddles a velocity layer step, in which case the pixel keeps its OWN
// rendered layer (the center tap).
float2 SelectLayerAwareQuadVelocity(float2 velocityJitteredUV[9], float2 phasePx, float2 sizePixels, float coherenceRadiusPx)
{
    float2 v00, v10, v01, v11;
    SelectPhaseQuad(velocityJitteredUV, phasePx, v00, v10, v01, v11);

    if (QuadStraddlesVelocityStep(v00, v10, v01, v11, velocityJitteredUV, sizePixels, coherenceRadiusPx))
        return velocityJitteredUV[0];   // the pixel's own rendered layer

    return Bilerp2x2(v00, v10, v01, v11, abs(phasePx));
}

float2 BilerpVelocityQuad(float2 velocityUV[9], float2 fracPx)
{
    float2 v00, v10, v01, v11;
    SelectPhaseQuad(velocityUV, fracPx, v00, v10, v01, v11);
    return Bilerp2x2(v00, v10, v01, v11, abs(fracPx));
}

// ============================================================================
// Jacobian & advection budget (smooth-surface correction)
// ============================================================================
// 2x2 Jacobian of the velocity field in px-per-texel, from layer-consistent
// differences.
float4 EstimateVelocityJacobianCoherentPx(float2 v[9], float2 sizePixels, float coherenceRadiusPx)
{
    float2 c3 = (v[3] - v[0]) * sizePixels;
    float2 c4 = (v[4] - v[0]) * sizePixels;
    float2 c1 = (v[1] - v[0]) * sizePixels;
    float2 c2 = (v[2] - v[0]) * sizePixels;
    bool leftOk  = (length(c3) <= coherenceRadiusPx);
    bool rightOk = (length(c4) <= coherenceRadiusPx);
    bool upOk    = (length(c1) <= coherenceRadiusPx);
    bool downOk  = (length(c2) <= coherenceRadiusPx);

    float4 J = float4(0.0, 0.0, 0.0, 0.0);
    if      (rightOk && leftOk)  { float2 d = (c4 - c3) * 0.5; J.x = d.x; J.z = d.y; }
    else if (rightOk)            { J.x = c4.x;  J.z = c4.y;  }
    else if (leftOk)             { J.x = -c3.x; J.z = -c3.y; }
    if      (downOk && upOk)     { float2 d = (c2 - c1) * 0.5; J.y = d.x; J.w = d.y; }
    else if (downOk)             { J.y = c2.x;  J.w = c2.y;  }
    else if (upOk)               { J.y = -c1.x; J.w = -c1.y; }
    return J;
}

float2 ApplyVelocityJacobian(float4 J, float2 vecPx)
{
    return float2(dot(J.xy, vecPx), dot(J.zw, vecPx));
}

// Residual (second-order) advection budget: what the first-order Jacobian
// correction cannot absorb. Hard-clamped; the miss term is capped at the
// predicted same-surface advection.
float ResidualAdvectionBudgetPx(float maxPairGradPx, float missPx, float snapDistPx, float missCapPx)
{
    float boundedMissPx = min(missPx, missCapPx);
    return min(maxPairGradPx * (kVelJacobianResidualFrac * boundedMissPx + snapDistPx + kPursuitSnapPadPx),
               kPursuitMaxAdvectionPx);
}

// ============================================================================
// Measured pair rules & the two-tap similarity extrapolation
// ============================================================================
float2 ComplexMul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

bool ForegroundPairSameSurface(
    float rawA, float2 offA, float rawB, float2 offB,
    float slopePerTexel, float depthQuantStep)
{
    float spanPx = max(length(offA - offB), 1.0);
    return abs(rawA - rawB) <= slopePerTexel * spanPx + 2.0 * depthQuantStep;
}

bool ForegroundPairRigidMagnitude(
    float2 wPx, float2 dPx, float fgVelocityPx,
    float rawA, float rawB, float coherentGradPx, float velQuantPx)
{
    float spanPx = max(sqrt(dot(dPx, dPx)), 1e-3);
    float relGap = abs(rawA - rawB) / max(min(rawA, rawB), 1e-6);
    float ceilingPx = coherentGradPx + fgVelocityPx * relGap / spanPx;
    return length(wPx) <= ceilingPx * spanPx + 2.0 * velQuantPx;
}

// The smallest nonzero pairwise velocity step in the 3x3, in px: the honest
// local noise scale of the buffer. Squared-domain min (sqrt is monotone and
// the >0 test squares its floor); the per-tap pixel conversion is hoisted
// out of the pair loop. One sqrt instead of 36.
float MeasureVelocityQuantStepPx(float2 velocityJitteredUV[9], float2 sizePixels)
{
    float2 vPx[9];
    [unroll]
    for (int k = 0; k < 9; ++k) { vPx[k] = velocityJitteredUV[k] * sizePixels; }

    float kLargeSq  = kLargeValue * kLargeValue;
    float minStepSq = kLargeSq;
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        [unroll]
        for (int j = i + 1; j < 9; ++j)
        {
            float2 d = vPx[j] - vPx[i];
            float stepSq = dot(d, d);
            if (stepSq > 1e-12)
                minStepSq = min(minStepSq, stepSq);
        }
    }
    return (minStepSq < kLargeSq) ? sqrt(minStepSq) : kVelQuantFloorPx;
}

void MeasureShallowForegroundGeometry(
    float  depthRaw[9], float2 velocityJitteredUV[9], float2 sizePixels,
    float  fgCoherentGradPx, float velQuantPx,
    bool   centerOnObject,
    out float shallowSlope, out float shallowVelGradPx)
{
    shallowSlope     = 0.0;
    shallowVelGradPx = 0.0;

    [unroll]
    for (int a = 0; a < 9; ++a)
    {
        if (centerOnObject ? (depthRaw[a] < depthRaw[0])   // behind the crest's own surface
                           : (depthRaw[a] <= depthRaw[0]))  // not foreground-side of the background
            continue;
        [unroll]
        for (int b = a + 1; b < 9; ++b)
        {
            if (centerOnObject ? (depthRaw[b] < depthRaw[0])
                               : (depthRaw[b] <= depthRaw[0]))
                continue;
            float2 spanVec = kOffsets3x3[b] - kOffsets3x3[a];
            float  spanPx  = length(spanVec);
            if (spanPx > 1.5) continue;             // adjacent pairs only
            float  gap  = abs(depthRaw[a] - depthRaw[b]);
            float2 wPx  = (velocityJitteredUV[b] - velocityJitteredUV[a]) * sizePixels;
            float  vAPx = length(velocityJitteredUV[a] * sizePixels);
            if (!ForegroundPairRigidMagnitude(wPx, spanVec, vAPx, depthRaw[a], depthRaw[b],
                                              fgCoherentGradPx, velQuantPx))
                continue;
            shallowSlope     = max(shallowSlope, gap / spanPx);
            shallowVelGradPx = max(shallowVelGradPx, length(wPx) / spanPx);
        }
    }
}

// The foreground layer's velocity field evaluated at an exact position, via
// the two-tap similarity (rotation + uniform scale) prior:
//     q = (v2 - v1) / (f2 - f1)  as a complex quotient;  v(B) = v1 + q*(B - f1)
// Bounded gain EXACTLY (|v(B) - v1| = |q| * |B - f1|); jitter-clean (q is a
// same-frame difference; static scenes give q ~ 0). Any gate failure returns
// the anchor tap's raw velocity and zeroes the diagnostics.
//   * DILATION call:  anchor = the closest (crest) tap, second = the
//     second-closest, and the second must additionally be foreground-side of
//     the background center.
//   * EDGE cliff-phase call: anchor = the center itself; the same-surface
//     gate (slope reference = the center-inclusive shallow measurement)
//     admits the along-silhouette pairs and rejects steep interior steps.
float2 ResolveForegroundFieldVelocityUV(
    float2 anchorOffsetPx,  float  anchorDepthRaw,  float2 anchorVelocityUV,
    float2 secondOffsetPx,  float  secondDepthRaw,  float2 secondVelocityUV,
    bool   secondMustBeCloserThanRef,
    float  secondRefRaw,
    float2 evalPx,
    float  slopePerTexel,
    float  depthQuantStep,
    float  coherentGradPx,
    float  velQuantPx,
    float2 sizePixels,
    out float pairGradPx,       // |w|/|d| of the active pair (0 = fallback)
    out float dispPx)           // |q|*|eval-anchor| of the active extrapolation
{
    pairGradPx = 0.0;
    dispPx     = 0.0;

    if (secondMustBeCloserThanRef && !(secondDepthRaw > secondRefRaw))
        return anchorVelocityUV;

    if (!ForegroundPairSameSurface(anchorDepthRaw, anchorOffsetPx,
                                   secondDepthRaw, secondOffsetPx,
                                   slopePerTexel, depthQuantStep))
        return anchorVelocityUV;

    float2 dPx   = secondOffsetPx - anchorOffsetPx;
    float2 wPx   = (secondVelocityUV - anchorVelocityUV) * sizePixels;
    float  dSqPx = dot(dPx, dPx);
    if (dSqPx < 1e-4)                            // coincident taps: no pair geometry
        return anchorVelocityUV;

    float fgVelocityPx = length(anchorVelocityUV * sizePixels);
    if (!ForegroundPairRigidMagnitude(wPx, dPx, fgVelocityPx,
                                      anchorDepthRaw, secondDepthRaw,
                                      coherentGradPx, velQuantPx))
        return anchorVelocityUV;

    // q = w / d as a complex quotient: w * conj(d) / |d|^2.
    float2 q = ComplexMul(wPx, float2(dPx.x, -dPx.y)) / dSqPx;

    // Evaluate from the pair endpoint NEAREST the evaluation point. Under the
    // similarity prior both endpoints give the identical v(B) (two
    // correspondences determine the field exactly), so this is a no-op inside
    // the model's world -- but under real field curvature the extrapolation
    // error grows with the lever arm |B - f_e|, so the nearer endpoint is
    // never worse and usually tighter. The dilation call benefits most: its
    // anchor (the crest) can sit ~2 texels from B while the partner is
    // nearer.
    bool   fromSecond   = dot(evalPx - secondOffsetPx, evalPx - secondOffsetPx)
                        < dot(evalPx - anchorOffsetPx, evalPx - anchorOffsetPx);
    float2 baseOffsetPx = fromSecond ? secondOffsetPx    : anchorOffsetPx;
    float2 baseVelPx    = (fromSecond ? secondVelocityUV : anchorVelocityUV) * sizePixels;

    pairGradPx = length(wPx) / sqrt(dSqPx);
    // Tolerance displacement stays measured from the ANCHOR tap: consumers
    // size tolerances against the fallback (the anchor's raw velocity), and
    // |v(B) - v_anchor| = |q| * |B - f_anchor| EXACTLY, independent of which
    // endpoint the evaluation is anchored at.
    dispPx = length(ComplexMul(q, evalPx - anchorOffsetPx));

    float2 vPx = baseVelPx + ComplexMul(q, evalPx - baseOffsetPx);
    return vPx * (1.0 / sizePixels);
}

#endif // TAA_VELOCITY_H_HLSL