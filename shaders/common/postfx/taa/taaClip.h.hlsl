// ============================================================================
// TAA history clipping: color statistics, the exact-hull clipper, the
// mean/variance box fallback, and the feedback/drift consumers
// ----------------------------------------------------------------------------
// EXACTNESS CERTIFICATE (the hull clipper): four certified properties.
// (1) Collinear / flat tap sets -- every 1D color neighborhood, i.e. every
// edge -- are solved by dual axis walks (the perpendicular dual coordinate
// is degenerate for a truly collinear set, so the 1D dual IS the dual; the
// old simplex could not even form its enclosing triangle there). (2)
// Genuinely 2D sets run the dual simplex: a plane through three taps that
// is SUPPORTING (no tap above) and whose triangle ENCLOSES the 2D ray
// origin has height at that origin EXACTLY equal to the hull exit height;
// convergence is CHECKED, not assumed. (3) Any non-exact outcome is
// tightened by further dual axis walks. Every emitted value is the minimum
// of certified upper bounds on the exit height (dual points, supporting
// planes, the horizontal plane zMax), so the clip can never over-clip: the
// failure mode is safe looseness, never flicker.
// (4) PHASE-TRANSPORT PADDING (the hull's jitter accounting): the history
// is an EWMA of PAST-PHASE samples of the signal; the 9 taps are
// CURRENT-phase point samples. The convexity argument above bounds
// averages of THE TAPS -- it says nothing about averages of other phases'
// samples, and for sub-texel content (values no single phase's tap set
// spans: thin features, sparkle, per-texel noise) those differ. Clipping
// to the single-phase hull every frame constrains the converged
// accumulator to the INTERSECTION of the per-phase hulls, while the
// anti-aliased value needs the UNION's room -- that gap is the hull's
// flicker. The hull is therefore padded by the measured phase-transport
// SEGMENT [-w, +w]: w = expectedJitterShift (the same first-order gradient
// model, magnitude rule and motion fade the box path's sigma inflation
// uses -- one policy, both paths). The padded hull is hull(taps)
// Minkowski-summed with the segment, whose support function is ADDITIVE:
// every certified dual bound F(a,b) lifts by exactly
// |w_z - a*w_u - b*w_v|, every supporting plane by |dot(w,N)|/N.z, and
// zMax by |w_z|. The min of lifted bounds is a certified upper bound on
// the PADDED exit height: never-over-clip survives w.r.t. the jitter
// model (whose error is second order in the local curvature -- the same
// footing as every practical variance clipper). The lifts are identically
// zero when the padding is off, the content flat, or motion has faded it,
// and they never touch a history that already lands inside (only the exit
// height rises, and tHit >= 1 still early-returns). With an active
// non-ray-parallel segment, exactness labels are demoted to "certified":
// the padded optimum may sit at a different dual point than the one
// certified above. A ray-parallel segment (wSeg.xy == 0) lifts F by the
// constant |w_z| and moves no minimizer -- labels stay exact.
//
// LOAD-BEARING INVARIANT: the anchor (stats.mean) must lie inside the hull
// of the taps -- it is a positive-weighted average of exactly those taps.
// (Consequence used by the collinear path: the projected origin lies in
// the projected hull, so a collinear projected set has its line THROUGH
// the origin and every perpendicular coordinate is zero.)
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
// DUAL DESCENT (axis walks)
// ----------------------------------------------------------------------------
// The exit height is the primal LP  max { sum lz : sum l = 1, l >= 0,
// sum l*(u,v) = 0 }, whose dual is  h = min over (a,b) of
// F(a,b) = max_i (z_i - a*u_i - b*v_i).  EVERY dual point evaluates to a
// certified upper bound on h -- the never-over-clip property. The walks
// minimize F along one axis at a time in the (w, w-perp) reparameterized
// dual (the w-scaling cancels in the arithmetic).
//
// Walk mechanics: the governing line is the argmax at the current dual
// point (its slope is F's slope). The walk steps to the FIRST breakpoint --
// the nearest line that overtakes the governing one in the walk direction
// (cross-multiplied minimum: one division per step) -- and hands off to it.
// The minimum of the convex piecewise-linear F along the axis is exactly
// the breakpoint where the handoff line's slope stops descending (zero or
// ascending): the walk flags that crossing as exact. A budget stop or a
// missing overtaker still returns F at the walked-to dual point:
// certified.
//
// For a tap set collinear in (u,v), F is invariant along the perpendicular
// dual axis, so ONE w-walk solves the problem exactly: the clipper's fast
// path for edges and flat regions.
//
// PHASE-PADDED OBJECTIVE: with the transport segment active the true
// objective is F_pad = F + |W| (W = w_z - a*w_u - b*w_v; see the header,
// item 4), whose extra kink at W = 0 is NOT a walk breakpoint -- the walks
// still minimize F. That is looseness only: the lift is added to each
// walk's value at ITS OWN final dual point by the caller, and every lifted
// value is still a certified upper bound on the padded exit.
// ============================================================================
static const float kHullTieBreakEps    = kHullSimplexCertEps * 0.03125; // index-linear z perturbation: hull-inflating, < certEps/4 total
static const float kHullCollinearExact = 1.0e-6;     // float-noise-level collinearity: only below this is the 1D solve labeled exact
static const float kHullCollinearLoose = 1.0 / 64.0; // .. below this the 1D bound is used and the simplex skipped
static const float kHullDualStepCap   = 1e20;       // runaway guard for near-flat slopes

float cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }

// Segment lift of a dual bound: for the phase-transport segment [-w, +w]
// (w in the ray frame), F_pad(a,b) = F(a,b) + |W| with
// W = w.z - a*w.x - b*w.y -- the segment's support function is additive and
// index-independent (max_i (f_i + |W|) = F + |W| exactly).
float HullSegmentLift(float2 mn, float3 wSeg)
{
    return abs(wSeg.z - mn.x * wSeg.x - mn.y * wSeg.y);
}

float HullDualAxisWalk(
    inout float2 mn,            // dual position (x = w-axis, y = w-perp-axis); advanced in place
    bool         alongS,         // axis of this walk: true = w, false = w-perp
    int          maxSteps,
    float3       qArr[9],       // projected taps (z already tie-perturbed)
    float        sArr[9],       // w-axis coordinates  (dot(q.xy, w))
    float        tArr[9],       // w-perp coordinates  (cross(w, q.xy))
    out bool     exactHit,
    out float    stepsUsed)
{
    exactHit  = false;
    stepsUsed = 0.0;

    // Governing line at the start: the argmax at the current dual point.
    // The index-linear z perturbation makes this unique at the dual origin;
    // a mid-descent restart can land ON a crossing (tie) -- any pick yields
    // a certified result, only the label can loosen.
    float fa = -kLargeValue;
    int   g  = 0;
    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        float f = qArr[i].z - mn.x * sArr[i] - mn.y * tArr[i];
        if (f > fa) { fa = f; g = i; }
    }

    float dg = alongS ? sArr[g] : tArr[g];

    if (dg == 0.0)
    {
        exactHit = true;                 // F's slope along this axis is zero: already at the axis optimum
    }
    else
    {
        float sigma = (dg > 0.0) ? 1.0 : -1.0;

        [unroll]
        for (int it = 0; it < 8; ++it)
        {
            if (it >= maxSteps) break;

            // First overtaker of the governing line in the walk direction
            // (cross-multiplied minimum: one division per step). Lines at
            // or above the governing one and lines that never catch up are
            // skipped.
            float numB = 0.0, denB = 1.0;
            int   k = -1;
            [unroll]
            for (int j = 0; j < 9; ++j)
            {
                float dj  = alongS ? sArr[j] : tArr[j];
                float den = sigma * (dg - dj);
                if (den > 1e-30)
                {
                    float fj  = qArr[j].z - mn.x * sArr[j] - mn.y * tArr[j];
                    float num = fa - fj;
                    if (num > 0.0 && (k < 0 || num * denB < numB * den))
                    { numB = num; denB = den; k = j; }
                }
            }
            if (k < 0) break;                            // no overtaker (numerical: the dual is bounded when the origin is inside the hull)

            float step = sigma * (numB / denB);
            if (!(abs(step) < kHullDualStepCap)) break; // runaway guard (NaN-safe)

            if (alongS) mn.x += step; else mn.y += step;
            stepsUsed += 1.0;

            // Handoff: past the crossing the overtaker governs; its slope
            // along the walk is -sigma*d_k. Still descending (sigma*d_k >
            // 0): continue. Zero or ascending: THIS crossing is the
            // minimum -- exact stop.
            float dk = alongS ? sArr[k] : tArr[k];
            if (sigma * dk <= 0.0) { exactHit = true; break; }

            dg = dk;
            fa = qArr[k].z - mn.x * sArr[k] - mn.y * tArr[k];
        }
    }

    // F at the final dual point: certified regardless of how the walk
    // terminated (every dual point is an upper bound on h).
    float val = -kLargeValue;
    [unroll]
    for (int j2 = 0; j2 < 9; ++j2)
        val = max(val, qArr[j2].z - mn.x * sArr[j2] - mn.y * tArr[j2]);
    return val;
}

// ============================================================================
// EXACT-HULL SIMPLEX CLIPPER (2D-reduced, ray-aligned frame, phase-padded)
// ----------------------------------------------------------------------------
// STRUCTURE: rotate to a frame with +Z = the ray; the exit is the upper
// envelope's height at the 2D origin. Near-collinear / flat sets are
// solved by the dual axis walks above (the common case -- edges). Genuinely
// 2D sets run a dual simplex that climbs the envelope: the basis is a
// triangle of taps enclosing the origin; pricing finds the highest tap
// above the current plane; the pivot swaps it in for the vertex opposite
// the edge crossed by the ray from the entering tap through the origin,
// extended past it (the geometric ratio test).
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
// PHASE TRANSPORT: the simplex and the walks run on the UNPADDED tap set
// (the anchor-invariant and the enclosure arguments belong to it); every
// emitted bound is then lifted by the transport segment at the dual point
// / plane normal that produced it. The lifts are zero whenever the jitter
// padding is off, flat, or motion-faded -- bit-identical to the unpadded
// clipper.
//
// Diagnostics (hullDiag): x = 1 exact (simplex certificate, or the
// collinear 1D solve -- requires the segment inactive or ray-parallel),
// 2 certified bound (dual descent / supporting-only plane / any active
// non-ray-parallel segment), 3 defensive fallback (every path ends
// certified, so 3 should be unreachable -- treat it as a bug alarm);
// y = iterations used.
// ============================================================================
float3 ClipHistoryToConvexHull_Simplex(
    float3 historyColorSpace,
    float3 neighborhoodColorSpace[9],
    float3 anchorPoint,
    float3 jitterShift,           // stats.expectedJitterShift (working color space)
    float  clipOvershoot,
    float  softClipAmount,
    float  motionFactor,
    out float2 hullDiag)
{
    hullDiag = float2(3.0, 0.0);

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

    // --- Phase-transport segment, projected into the ray frame -----------
    // w = the measured expected jitter shift, faded by motion EXACTLY like
    // the box path's sigma inflation (one policy, both paths). Zero when
    // the padding is off / content flat / motion-faded: every lift below is
    // then a no-op and the clipper is bit-identical to the unpadded one.
    // The segment is symmetric ([-w, +w]): past phases sit on both sides of
    // the current one, and the history is their average.
    float  padFade = (taaJitterFlickerFade > 0.5) ? (1.0 - motionFactor) : 1.0;
    float3 wColor  = jitterShift * (taaJitterFlickerPadding * padFade);
    float3 wSeg    = float3(dot(wColor, ex), dot(wColor, ey), dot(wColor, ez));

    // --- Project the anchor-centered taps (u, v, z); track the max-z tap
    // (A), the max-radius direction (w) and the z range. z carries the
    // tie-break perturbation: unique active/basis selections keep the
    // simplex and the walks monotone (no cycling on duplicated colors), and
    // the perturbation only ever INFLATES the hull (<= 8*eps, well inside
    // certEps) -- the safe direction for a never-over-clip bound.
    float3 q[9];
    float3 A = float3(0.0, 0.0, -kLargeValue);
    float2 w = float2(0.0, 0.0);
    float  zMax = -kLargeValue;
    float  zMin =  kLargeValue;
    float  spreadArea = 0.0;             // max |q.xy|^2: the 2D projection's squared radius
    [unroll]
    for (int ip = 0; ip < 9; ++ip)
    {
        float3 p = neighborhoodColorSpace[ip] - anchorPoint;
        q[ip] = float3(dot(p, ex), dot(p, ey), dot(p, ez));
        q[ip].z += ip * kHullTieBreakEps;
        zMax = max(zMax, q[ip].z);
        zMin = min(zMin, q[ip].z);
        if (q[ip].z > A.z) A = q[ip];
        float r2 = dot(q[ip].xy, q[ip].xy);
        if (r2 > spreadArea) { spreadArea = r2; w = q[ip].xy; }
    }

    // Relative area epsilon: the fold/enclosure crosses scale as the SQUARE
    // of the 2D spread; an absolute epsilon misclassifies tight (but valid)
    // projections as degenerate. 0.1% of the area scale, with an absolute
    // floor.
    float epsArea = 1e-9 + 1e-3 * spreadArea;

    // Certificate tolerance in height units, floored by the constant and
    // scaled by the neighborhood's own z range (the data's noise scale).
    float certEps = kHullSimplexCertEps + 1e-2 * (zMax - zMin);

    // --- 1D structure along the max-spread direction ----------------------
    // s = the w-axis coordinate, t = the w-perp coordinate (= the signed
    // deviation from the w-line). All thresholds are relative: the walk
    // arithmetic is invariant under the scaling of w. A perfectly flat set
    // gives w = 0, every s = 0, devRel = 0. By the anchor-invariant, a
    // truly collinear projected set has its line THROUGH the origin, so
    // every t is zero up to float noise -- devRel sits at the noise level
    // (~1e-7), which is exactly what kHullCollinearExact admits as exact.
    float sArr[9];
    float tArr[9];
    float devMax = 0.0;
    [unroll]
    for (int iS = 0; iS < 9; ++iS)
    {
        sArr[iS] = dot(q[iS].xy, w);
        tArr[iS] = cross2(w, q[iS].xy);
        devMax   = max(devMax, abs(tArr[iS]));
    }
    float devRel = devMax / max(spreadArea, 1e-30);   // <= 1 by construction

    float supportScale = 1.0 + clipOvershoot;
    float zExit = zMax + abs(wSeg.z);                 // horizontal plane (+ segment lift): always a valid upper bound
    bool  exact  = false;
    float itersUsed = 0.0;
    float2 mnD = float2(0.0, 0.0);                     // shared dual descent position

    if (devRel <= kHullCollinearLoose)
    {
        // ---- 1D / collinear / flat: one axis walk solves it exactly ----
        // (A flat set has every s = 0: the walk stops immediately on the
        // flat-slope test with F = zMax, which IS the exact answer when the
        // whole 2D hull is the single point at the origin.) Sets in the
        // band above the noise level keep the walk result as a certified
        // bound and get perpendicular tightening below -- two taps with
        // nearly identical projections but a large z-gap can defeat a pure
        // 1D label there, and the w-perp walk resolves exactly that
        // cancellation pattern.
        bool  ex1; float st1;
        float h1 = HullDualAxisWalk(mnD, true, 8, q, sArr, tArr, ex1, st1);
        zExit     = min(h1 + HullSegmentLift(mnD, wSeg), zExit);   // lift at h1's own dual point
        itersUsed = st1;
        exact     = ex1 && (devRel <= kHullCollinearExact);
    }
    else
    {
        // ---- 2D: the dual simplex ----
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

        bool valid = (haveC && haveD);

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
            float pivotsUsed = 0.0;

            // kHullSimplexPivots + 2 pivots, plus a final certificate pass
            // on the last basis (the certificate check runs first, so a
            // converged basis never pivots).
            [unroll]
            for (int it = 0; it <= kHullSimplexPivots + 2; ++it)
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
                    // Segment lift of this plane: the padded hull's support
                    // plane with the same normal sits |dot(w,N)|/N.z higher
                    // (support functions add), so height + lift is a
                    // certified upper bound on the PADDED exit height. With a
                    // ray-parallel segment (lift = |wSeg.z| for every plane
                    // and every dual point alike) the ordering of all bounds
                    // is unchanged, so exactness survives exactly.
                    float lift = abs(dot(wSeg, N)) / N.z;

                    // Enclosing + supporting = EXACT; supporting alone is still
                    // a valid upper bound. Both take the min with the other
                    // lifted bounds -- the lifted bounds are mutually
                    // unordered, and the min of certified upper bounds is a
                    // certified upper bound.
                    float tA = cross2(V0.xy, V1.xy);
                    float tB = cross2(V1.xy, V2.xy);
                    float tC = cross2(V2.xy, V0.xy);
                    bool encloses = (tA >= -epsArea && tB >= -epsArea && tC >= -epsArea) ||
                                    (tA <=  epsArea && tB <=  epsArea && tC <=  epsArea);
                    if (encloses) { exact = true; }
                    zExit = min(zExit, height + lift);
                    break;
                }

                if (it == kHullSimplexPivots + 2)
                    break;                      // budget spent: fall back

                // --- pivot: eject the vertex opposite the edge crossed by the
                // ray from P through the origin, extended past it (the
                // geometric ratio test -- maintains enclosure and monotone
                // progress; the z perturbation keeps the selections unique).
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

            itersUsed = pivotsUsed;
        }
    }

    // ---- dual-descent tightening: every non-exact outcome gets further
    // axis walks -- the collinear path continues its endpoint across the
    // perpendicular axis (resolving near-duplicated projections with a
    // z-gap), the 2D path descends from the horizontal bound. Each value
    // is a certified upper bound (any dual point is), so the min is too.
    // Each walk's lift is captured at ITS OWN final dual point -- mnD
    // advances between calls. This covers the simplex's failure modes:
    // thin color needles along the ray (near-vertical planes) and
    // degenerate pivots.
    if (!exact)
    {
        bool  exF; float stF;
        if (devRel <= kHullCollinearLoose)
        {
            float hp   = HullDualAxisWalk(mnD, false, 4, q, sArr, tArr, exF, stF);
            float hpP  = hp + HullSegmentLift(mnD, wSeg);
            float hw2  = HullDualAxisWalk(mnD, true,  2, q, sArr, tArr, exF, stF);
            float hw2P = hw2 + HullSegmentLift(mnD, wSeg);
            zExit = min(zExit, min(hpP, hw2P));
        }
        else
        {
            float hw  = HullDualAxisWalk(mnD, true,  4, q, sArr, tArr, exF, stF);
            float hwP = hw + HullSegmentLift(mnD, wSeg);
            float hp  = HullDualAxisWalk(mnD, false, 4, q, sArr, tArr, exF, stF);
            float hpP = hp + HullSegmentLift(mnD, wSeg);
            float hw2  = HullDualAxisWalk(mnD, true,  2, q, sArr, tArr, exF, stF);
            float hw2P = hw2 + HullSegmentLift(mnD, wSeg);
            zExit = min(zExit, min(min(hwP, hpP), hw2P));
        }
        itersUsed += stF;
    }

    // ---- segment honesty: with an active NON-ray-parallel segment the
    // padded optimum can sit at a different dual point than the one the
    // machinery certified above, so equality with the PADDED exit is no
    // longer provable -- cap the label at "certified bound". A ray-parallel
    // segment lifts F by the constant |wSeg.z|, moves no minimizer, and
    // lifts every plane by the same constant: exact labels survive exactly.
    // (Bit-exact zero test is intentional: only a truly ray-parallel
    // projection keeps the label.)
    if (!(wSeg.x == 0.0 && wSeg.y == 0.0))
        exact = false;

    hullDiag = exact ? float2(1.0, itersUsed) : float2(2.0, itersUsed);

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
// Hull clipping: the exact clipper above, phase-padded by the measured
// expected jitter shift (the same model and motion fade as the box path's
// sigma inflation -- taaJitterFlickerPadding drives BOTH paths; at padding
// 0 the hull clipper is bit-identical to the strict single-phase version).
// Fallback: mean/variance box intersected with the (firefly-clamped) sample
// AABB, with soft-clip.
// hullDiag: x = 0 clipper disabled, 1 exact (simplex certificate or the
// collinear 1D solve -- requires the segment inactive or ray-parallel),
// 2 certified bound (supporting-only plane and/or dual descent and/or an
// active non-ray-parallel segment), 3 defensive fallback (unreachable);
// y = iterations used.
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
            stats.expectedJitterShift,
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