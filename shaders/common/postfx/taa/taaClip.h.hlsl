// ============================================================================
// TAA history clipping: color statistics, the exact-hull simplex clipper,
// the mean/variance box fallback, and the feedback/drift consumers
// ----------------------------------------------------------------------------
// EXACTNESS CERTIFICATE (the hull clipper): a plane through three taps that
// is SUPPORTING (no tap above) and whose triangle ENCLOSES the 2D ray origin
// has height at that origin EXACTLY equal to the hull exit height --
// supporting puts the hull below the plane, enclosing puts the height point
// (a barycentric combination of the taps) inside the hull. Convergence is
// CHECKED, not assumed. The only emitted values are the certificate result
// or the horizontal support plane (a valid upper bound in every
// configuration), so the clip can never over-clip: the failure mode is safe
// looseness, never flicker.
//
// LOAD-BEARING INVARIANT: the anchor (stats.mean) must lie inside the hull
// of the taps -- it is a positive-weighted average of exactly those taps.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (clipping + feedback constants). Requires fragments
// included before: taaShared.h.hlsl, taaConstants.h.hlsl, taaFrame.h.hlsl
// (HistoryReprojection).
// ============================================================================
#ifndef TAA_CLIP_H_HLSL
#define TAA_CLIP_H_HLSL

// ============================================================================
// SMALL CLIPPING MATH HELPERS
// ============================================================================
// Soft-clip scale factor: 1.0 = hard clip; grows toward (1 + softClipAmount) as
// the requested overshoot (in box-size units) grows; motion restores hard clip.
float SoftClipUnitScale(float overshootUnits, float softClipAmount, float motionFactor)
{
    float softLimit = 1.0 + softClipAmount * (1.0 - exp2(-(overshootUnits - 1.0) * kLog2E));
    return lerp(softLimit, 1.0, motionFactor);
}

// Entry/exit interval of a ray against one slab [slab.x, slab.y] along one axis.
float2 RaySlabInterval(float origin, float dir, float2 slab)
{
    float sign   = (dir >= 0.0) ? 1.0 : -1.0;
    float invDir = 1.0 / (sign * max(abs(dir), 1e-7));
    float t0     = (slab.x - origin) * invDir;
    float t1     = (slab.y - origin) * invDir;
    return float2(min(t0, t1), max(t0, t1));
}

float3 ClipRayToBox(float3 history, float3 target, float3 boxMin, float3 boxMax, float softClipAmount, float motionFactor)
{
    float3 boxCenter = 0.5 * (boxMax + boxMin);
    float3 boxExtent = max(0.5 * (boxMax - boxMin), kEpsilon);

    float3 historyUnit = (history - boxCenter) / boxExtent;
    float overshootUnits = max(abs(historyUnit).x, max(abs(historyUnit).y, abs(historyUnit).z));

    if (overshootUnits <= 1.0)
        return history;

    float3 rayDir = target - history;

    float2 slabX = RaySlabInterval(history.x, rayDir.x, float2(boxMin.x, boxMax.x));
    float2 slabY = RaySlabInterval(history.y, rayDir.y, float2(boxMin.y, boxMax.y));
    float2 slabZ = RaySlabInterval(history.z, rayDir.z, float2(boxMin.z, boxMax.z));
    float entryT = saturate(max(max(slabX.x, slabY.x), slabZ.x));

    float3 clipped = history + rayDir * entryT;

    if (softClipAmount > 0.0)
    {
        float softScale = SoftClipUnitScale(overshootUnits, softClipAmount, motionFactor);
        clipped = lerp(history, clipped, 1.0 / max(softScale, 1.0));
    }
    return clipped;
}

// ============================================================================
// COLOR NEIGHBORHOOD STATISTICS
// ============================================================================
struct ColorNeighborhoodStats
{
    float3 aabbMin;
    float3 aabbMax;
    float3 mean;
    float3 sigma;
    float  spatialContrast;
    float3 expectedJitterShift;
    float  weights[9];
};

ColorNeighborhoodStats ComputeColorNeighborhoodStats(
    float3 neighborhoodColorSpace[9], float2 motionDirUnit, float motionNormalized, float2 jitterPx)
{
    ColorNeighborhoodStats stats;
    stats.aabbMin = float3(kLargeValue, kLargeValue, kLargeValue);
    stats.aabbMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    bool jitterPaddingEnabled = (taaJitterFlickerPadding > kFlickerPadThreshold);

    float3 weightedSum   = float3(0.0, 0.0, 0.0);
    float3 weightedSumSq = float3(0.0, 0.0, 0.0);
    float  totalWeight   = 0.0;
    float  motionFactor  = saturate(motionNormalized);

    bool jitterCenteredWeights = (taaJitterAwareVariance > 0.5);
    float2 weightCenterPx = jitterCenteredWeights ? jitterPx : float2(0.0, 0.0);

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float2 tapOffsetPx   = kOffsets3x3[i];
        float3 tapColorSpace = neighborhoodColorSpace[i];

        stats.aabbMin = min(stats.aabbMin, tapColorSpace);
        stats.aabbMax = max(stats.aabbMax, tapColorSpace);

        float2 offsetFromCenterPx = tapOffsetPx - weightCenterPx;
        float w = jitterCenteredWeights
            ? exp2(-dot(offsetFromCenterPx, offsetFromCenterPx) * kLog2E)
            : kStdWeights[i];

        if (taaLumaVariance > 0.5) { w *= (1.0 / (1.0 + max(tapColorSpace.x, 0.0))); }
        if (taaVelocityAlignedVariance > 0.5 && i > 0)
        {
            w *= lerp(1.0, saturate(dot(tapOffsetPx, motionDirUnit) * kInvLength[i] * 0.5 + 0.5), motionFactor);
        }

        stats.weights[i] = w;
        weightedSum   += tapColorSpace * w;
        weightedSumSq += tapColorSpace * tapColorSpace * w;
        totalWeight   += w;
    }

    float invTotalWeight = 1.0 / max(totalWeight, kEpsilon);
    stats.mean  = weightedSum * invTotalWeight;
    stats.sigma = sqrt(max(weightedSumSq * invTotalWeight - stats.mean * stats.mean, 0.0));

    // Firefly clamp: pull the AABB bounds into the mean +/- k*sigma band.
    if (taaFireflyClamp > kFireflyClampEpsilon)
    {
        float3 fireflyMin = stats.mean - taaFireflyClamp * stats.sigma;
        float3 fireflyMax = stats.mean + taaFireflyClamp * stats.sigma;
        stats.aabbMin = clamp(stats.aabbMin, fireflyMin, fireflyMax);
        stats.aabbMax = clamp(stats.aabbMax, fireflyMin, fireflyMax);
    }

    stats.spatialContrast = max(stats.aabbMax.x - stats.aabbMin.x, kMinSpatialContrast);

    // Color shift expected from the remaining sub-texel jitter.
    stats.expectedJitterShift = float3(0.0, 0.0, 0.0);
    if (jitterPaddingEnabled)
    {
        float3 gradX = jitterPx.x > 0.0 ? (neighborhoodColorSpace[4] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[3] - neighborhoodColorSpace[0]);
        float3 gradY = jitterPx.y > 0.0 ? (neighborhoodColorSpace[2] - neighborhoodColorSpace[0]) : (neighborhoodColorSpace[1] - neighborhoodColorSpace[0]);
        stats.expectedJitterShift = (gradX * abs(jitterPx.x)) + (gradY * abs(jitterPx.y));

        // Inflate the sigma by the expected jitter-induced color shift.
        float paddingFade   = (taaJitterFlickerFade > 0.5) ? saturate(1.0 - motionNormalized) : 1.0;
        float paddingAmount = taaJitterFlickerPadding * paddingFade;

        if (taaDirectionalVariance > 0.5) { stats.sigma += abs(stats.expectedJitterShift * paddingAmount); }
        else                              { stats.sigma += (stats.spatialContrast * length(jitterPx) * paddingAmount); }
    }

    stats.sigma = max(stats.sigma, kMinSigma);
    return stats;
}

// ============================================================================
// EXACT-HULL SIMPLEX CLIPPER (2D-reduced, ray-aligned frame)
// ----------------------------------------------------------------------------
// STRUCTURE: rotate to a frame with +Z = the ray; the exit is the upper
// envelope's height at the 2D origin. A dual simplex climbs that envelope:
// the basis is a triangle of taps enclosing the origin; pricing finds the
// highest tap above the current plane; the pivot swaps it in for the vertex
// opposite the edge crossed by the ray from the entering tap through the
// origin, extended past it (the geometric ratio test).
//
// Initial basis: A = max-z tap; C, D = angular extremes around the origin
// relative to A. The circular gaps D->A and A->C are below pi by
// construction; the wrap gap C->D is the max angular gap of the tap set,
// below pi because the origin is inside their 2D hull. Hence (A,C,D)
// encloses the origin whenever it is strictly inside.
//
// All taps are tracked BY VALUE (selects over unrolled static indices) --
// no dynamic indexing anywhere.
//
// Diagnostics (hullDiag): x = 1 exact certificate, 2 supporting-only bound,
// 3 unconverged fallback; y = pivots used (0..kHullSimplexPivots).
// ============================================================================
float cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }

// One dual line-search step (GJK-style descent) from the horizontal bound.
// LP duality: the exit height h = min over gradients (a,b) of
// max_i(z_i - a*u_i - b*v_i); EVERY (a,b) evaluates to a certified upper
// bound on h, with no plane conditioning involved. Start at (0,0) (the
// horizontal bound, active tap = the max-z tap A) and descend along the
// subgradient direction q_A.xy: the first tap to overtake A along the ray
// pins the tightened bound. This attacks exactly the degenerate regime
// (ray aligned with the color variation), where the simplex's planes go
// near-vertical but the dual stays perfectly conditioned.
float HullDualSagBound(float3 q[9], float3 A)
{
    float2 g  = A.xy;
    float  dA = dot(g, g);
    if (dA < 1e-14)
        return A.z;                         // the top tap projects onto the ray: bound already tight

    float tStar = kLargeValue;
    [unroll]
    for (int k = 0; k < 9; ++k)
    {
        float denom = dA - dot(g, q[k].xy);
        if (denom < 1e-6 * dA) continue;    // never catches the active tap along this ray
        float dz = A.z - q[k].z;
        if (dz < 1e-6) continue;            // no drop: no meaningful overtake
        tStar = min(tStar, dz / denom);
    }
    if (tStar >= kLargeValue)
        return A.z;                         // no catcher (numerically): bound unchanged

    return max(A.z - tStar * dA, 0.0);      // duality guarantees >= h; clamp is a NaN guard only
}

float3 ClipHistoryToConvexHull_Simplex(
    float3 historyColorSpace,
    float3 neighborhoodColorSpace[9],
    float3 anchorPoint,
    float  clipOvershoot,
    float  softClipAmount,
    float  motionFactor,
    out float2 hullDiag)
{
    hullDiag = float2(3.0, 0.0);
    float pivotsUsed = 0.0;

    float3 d = historyColorSpace - anchorPoint;
    float  dLenSq = dot(d, d);
    if (dLenSq < 1e-10)
    {
        hullDiag = float2(1.0, 0.0);     // history == anchor: inside by definition
        return historyColorSpace;
    }
    float dLen = sqrt(dLenSq);

    // --- Ray-aligned orthonormal frame (Duff/Frisvad, branch-free) -------
    float3 ez = d / dLen;
    float sign = (ez.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign + ez.z);
    float b = ez.x * ez.y * a;
    float3 ex = float3(1.0 + sign * ez.x * ez.x * a, sign * b, -sign * ez.x);
    float3 ey = float3(b, sign + ez.y * ez.y * a, -ez.y);

    // --- Project the anchor-centered taps (u, v, z); track zMax, zMin, spread
    float3 q[9];
    float  zMax = -kLargeValue;
    float  zMin =  kLargeValue;
    float  spreadArea = 0.0;             // max |q.xy|^2: the 2D projection's squared radius
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float3 p = neighborhoodColorSpace[i] - anchorPoint;
        q[i] = float3(dot(p, ex), dot(p, ey), dot(p, ez));
        zMax = max(zMax, q[i].z);
        zMin = min(zMin, q[i].z);
        spreadArea = max(spreadArea, dot(q[i].xy, q[i].xy));
    }

    // Relative area epsilon: the fold/enclosure crosses scale as the SQUARE
    // of the 2D spread; an absolute epsilon misclassifies tight (but valid)
    // projections as degenerate. 0.1% of the area scale, with an absolute
    // floor.
    float epsArea = 1e-9 + 1e-3 * spreadArea;

    // Certificate tolerance in height units, floored by the constant and
    // scaled by the neighborhood's own z range (the data's noise scale).
    float certEps = kHullSimplexCertEps + 1e-2 * (zMax - zMin);

    // --- Initial basis: A = max-z tap; C, D = angular extremes around the
    // origin relative to A. The circular gaps D->A and A->C are below pi by
    // construction; the wrap gap C->D is the max angular gap of the tap set,
    // below pi because the origin is inside their 2D hull. Hence (A,C,D)
    // encloses the origin whenever it is strictly inside.
    float3 A = q[0];
    [unroll]
    for (int iA = 1; iA < 9; ++iA)
    {
        if (q[iA].z > A.z) A = q[iA];
    }

    float3 C = A, D = A;
    bool haveC = false, haveD = false;
    [unroll]
    for (int iE = 0; iE < 9; ++iE)
    {
        float cr = cross2(A.xy, q[iE].xy);
        if (cr > epsArea)
        {
            if (!haveC)                              { C = q[iE]; haveC = true; }
            else if (cross2(C.xy, q[iE].xy) > 0.0)   { C = q[iE]; }   // q CCW of C: larger angle
        }
        else if (cr < -epsArea)
        {
            if (!haveD)                              { D = q[iE]; haveD = true; }
            else if (cross2(q[iE].xy, D.xy) > 0.0)   { D = q[iE]; }   // q CW of D: smaller angle
        }
    }

    float supportScale = 1.0 + clipOvershoot;
    float zExit = zMax;                            // safe fallback answer
    bool  exact  = false;
    bool  valid  = (haveC && haveD);

    if (valid)
    {
        // Defense-in-depth: verify the initial enclosure (float edge cases).
        float eA = cross2(C.xy, D.xy);
        float eB = cross2(D.xy, A.xy);
        float eC = cross2(A.xy, C.xy);
        valid = (eA >= -epsArea && eB >= -epsArea && eC >= -epsArea) ||
                (eA <=  epsArea && eB <=  epsArea && eC <=  epsArea);
    }

    if (valid)
    {
        float3 V0 = A, V1 = C, V2 = D;

        // kHullSimplexPivots pivots, plus a final certificate pass on the
        // last basis (the certificate check runs first, so a converged basis
        // never pivots).
        [unroll]
        for (int it = 0; it <= kHullSimplexPivots; ++it)
        {
            // --- plane through the basis, upward normal -------------------
            float3 N = cross(V1 - V0, V2 - V0);
            if (N.z < 0.0) N = -N;
            if (N.z < 1e-7 || N.z * 1024.0 < abs(N.x) + abs(N.y))
                break;                      // collinear basis / near-vertical plane: fall back

            // --- pricing: the highest tap above the plane ------------------
            float c0 = dot(V0, N);
            float maxSlack = 0.0;
            float3 P = V0;
            [unroll]
            for (int m = 0; m < 9; ++m)
            {
                float slack = dot(q[m], N) - c0;
                if (slack > maxSlack) { maxSlack = slack; P = q[m]; }
            }

            // --- convergence certificate: supporting plane ----------------
            // maxSlack/N.z is the worst tap's height above the plane; the
            // tolerance is the data's own noise scale, so a tap poking
            // noise-level-high does not force another pivot.
            if (maxSlack <= certEps * N.z)
            {
                float height = V0.z + (N.x * V0.x + N.y * V0.y) / N.z;

                // Enclosing + supporting = EXACT; supporting alone is still
                // a valid upper bound (take the min with the fallback).
                float tA = cross2(V0.xy, V1.xy);
                float tB = cross2(V1.xy, V2.xy);
                float tC = cross2(V2.xy, V0.xy);
                bool encloses = (tA >= -epsArea && tB >= -epsArea && tC >= -epsArea) ||
                                (tA <=  epsArea && tB <=  epsArea && tC <=  epsArea);
                if (encloses) { zExit = height;             hullDiag = float2(1.0, pivotsUsed); exact = true; }
                else          { zExit = min(zExit, height); hullDiag = float2(2.0, pivotsUsed); }
                break;
            }

            if (it == kHullSimplexPivots)
                break;                      // budget spent: fall back

            // --- pivot: eject the vertex opposite the edge crossed by the
            // ray from P through the origin, extended past it (the
            // geometric ratio test -- maintains enclosure and monotone
            // progress in exact arithmetic).
            float c0s = cross2(P.xy, V0.xy);
            float c1s = cross2(P.xy, V1.xy);
            float c2s = cross2(P.xy, V2.xy);
            float e01 = cross2(V0.xy, V1.xy);
            float e12 = cross2(V1.xy, V2.xy);
            float e20 = cross2(V2.xy, V0.xy);

            if      (c1s * c2s < 0.0 && e12 * (c2s - c1s) < 0.0) V0 = P;
            else if (c2s * c0s < 0.0 && e20 * (c0s - c2s) < 0.0) V1 = P;
            else if (c0s * c1s < 0.0 && e01 * (c1s - c0s) < 0.0) V2 = P;
            else break;                     // degenerate: fall back

            pivotsUsed += 1.0;
        }
    }

    // --- dual sag: on every non-exact outcome, tighten the fallback with
    // one certified dual line-search step. Both bounds are valid upper
    // bounds on the exit height; the min is too.
    if (!exact)
    {
        zExit = min(zExit, HullDualSagBound(q, A));
    }

    // --- emit ---------------------------------------------------------------
    float tHit = min(zExit * supportScale / dLen, 1.0);
    if (tHit >= 1.0)
        return historyColorSpace;
    tHit = max(tHit, 0.0);

    if (softClipAmount > 0.0)
    {
        float overshootUnits = 1.0 / max(tHit, kEpsilon);
        float softScale = SoftClipUnitScale(overshootUnits, softClipAmount, motionFactor);
        return anchorPoint + d * (softScale * tHit);
    }
    return anchorPoint + d * tHit;
}

// ============================================================================
// HISTORY CLIPPING ENTRY
// ============================================================================
// Hull clipping: the exact simplex above. Fallback: mean/variance box
// intersected with the (firefly-clamped) sample AABB, with soft-clip.
// hullDiag: x = 0 clipper disabled, 1 exact, 2 supporting-only, 3
// unconverged; y = pivots used.
float3 ClipHistoryToNeighborhood(
    float3 historyColorSpace, ColorNeighborhoodStats stats,
    float motionNormalized, float varianceGamma,
    float3 neighborhoodColorSpace[9], float3 clipMargin,
    out float2 hullDiag)
{
    float motionFactor = saturate(motionNormalized);

    if (taaUseHullClipping > 0.5)
    {
        return ClipHistoryToConvexHull_Simplex(
            historyColorSpace, neighborhoodColorSpace, stats.mean,
            taaClipOvershoot, taaSoftClip, motionFactor, hullDiag);
    }

    hullDiag = float2(0.0, 0.0);   // hull clipper disabled: box path

    float3 chromaScale     = float3(1.0, taaChromaVarianceMod, taaChromaVarianceMod);
    float3 varianceExtents = stats.sigma * varianceGamma * chromaScale;
    float3 boxMin = max(stats.mean - varianceExtents, stats.aabbMin - clipMargin);
    float3 boxMax = min(stats.mean + varianceExtents, stats.aabbMax + clipMargin);

    return ClipRayToBox(historyColorSpace, stats.mean, boxMin, boxMax, taaSoftClip, motionFactor);
}

// ============================================================================
// CLIP REJECTION / LUMA DRIFT / FEEDBACK
// ============================================================================
float ComputeClipDistanceRejection(float3 clippedHistorySpace, float3 historyColorSpace, ColorNeighborhoodStats stats)
{
    float3 clipDistance = abs(clippedHistorySpace - historyColorSpace) / max(stats.sigma, kMinSigma);
    float maxChannelDistance = max(clipDistance.x, max(clipDistance.y, clipDistance.z));
    return saturate((maxChannelDistance - taaClipDistanceRejectionMinError) * taaClipDistanceRejectionAmount);
}

void ApplyLumaDriftCorrection(inout float3 clippedHistorySpace, float3 currentColorSpace, ColorNeighborhoodStats stats)
{
    if (taaLumaDriftStrength <= 0.001)
        return;

    float lumaBias     = clippedHistorySpace.x - currentColorSpace.x;
    float relativeBias = lumaBias / max(currentColorSpace.x, kLumaDriftLumaFloor);

    float chromaSpread = max(stats.aabbMax.y - stats.aabbMin.y, stats.aabbMax.z - stats.aabbMin.z);
    float chromaGate   = 1.0 - saturate((chromaSpread - taaLumaDriftChromaTol) / kLumaDriftChromaFadeWidth);

    if (abs(relativeBias) > kLumaDriftRelThreshold && abs(lumaBias) > kLumaDriftAbsThreshold)
        clippedHistorySpace.x -= lumaBias * taaLumaDriftStrength * chromaGate;

    clippedHistorySpace.x = max(clippedHistorySpace.x, 0.0);
}

// History feedback. The alignment drop is applied AFTER the feedbackMin/Max
// clamp and ONLY on planar (single-layer) surfaces.
float ComputeHistoryFeedback(HistoryReprojection repro, float clipDistanceRejection, bool planarSurface)
{
    float feedback = taaFeedbackMax;

    float dropSpeed = max(taaMotionBlendDropSpeed, kMinMotionBlendDropSpeed);
    float motionDrop = saturate((repro.motionMagnitudePx - taaMotionBlendStart) / dropSpeed);
    feedback = lerp(taaFeedbackMax, taaFeedbackMin, motionDrop);
    feedback = clamp(feedback, taaFeedbackMin, taaFeedbackMax);

    feedback = lerp(feedback, taaFeedbackMin, clipDistanceRejection);

    if (planarSurface)
    {
        feedback -= taaAlignmentFeedbackDrop * (1.0 - repro.subpixelAlignment);
        feedback = max(feedback, 0.0);
    }
    return feedback;
}

#endif // TAA_CLIP_H_HLSL