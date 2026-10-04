// ============================================================================
// TAA shared math (included by taa.fx.hlsl and taaMotion.fx.hlsl)
// ----------------------------------------------------------------------------
// PURE functions only -- no cbuffer or sampler access. Everything the layer
// classification needs is passed in, so the resolve pass and the motion-field
// writer are guaranteed to classify the current frame IDENTICALLY (a stored
// field that disagrees with the resolve's own classification would lie to the
// next frame's disocclusion tests).
//
// The foreground-edge effective values are sub-texel PHASE dependent (the
// jitter direction relative to the silhouette selects the bilinear treatment
// or the pixel's own sample) -- ClassifyLayerSurface below documents the
// anchoring contract each layer class must satisfy; do not break it piecemeal.
// ============================================================================
#ifndef TAA_SHARED_H_HLSL
#define TAA_SHARED_H_HLSL

static const float kEpsilon = 1e-6;

static const float2 kOffsets3x3[9] =
{
    float2( 0,  0), float2( 0, -1), float2( 0,  1),
    float2(-1,  0), float2( 1,  0), float2(-1, -1),
    float2( 1, -1), float2(-1,  1), float2( 1,  1)
};

static const float kInvOffsetLenSq[9] = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5 };

float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

float2 Bilerp2x2(float2 c00, float2 c10, float2 c01, float2 c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

float Minmod(float a, float b)
{
    return (a * b > 0.0) ? ((abs(a) < abs(b)) ? a : b) : 0.0;
}

// ============================================================================
// VIEWPORT & PIXEL GEOMETRY
// ============================================================================
struct ViewportParams
{
    float2 texelSize;
    float2 sizePixels;
    float2 minUV;
    float2 maxUV;
};

ViewportParams GetViewportParams(float2 oneOverSize)
{
    ViewportParams vp;
    vp.texelSize  = oneOverSize;
    vp.sizePixels = 1.0 / max(oneOverSize, 1e-6);
    vp.minUV      = 0.5 * vp.texelSize;
    vp.maxUV      = 1.0 - vp.minUV;
    return vp;
}

struct SnappedCoord
{
    float2 snappedUV;
    float2 fracPx;
    float  fracDist;
};

SnappedCoord SnapUVToTexel(float2 uv, ViewportParams vp)
{
    SnappedCoord sc;
    float2 pixelPos  = uv * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;
    sc.snappedUV     = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    sc.fracPx        = pixelPos - baseTexel;
    sc.fracDist      = length(sc.fracPx);
    return sc;
}

void Build3x3TapUVs(float2 centerUV, float2 texelSize, float2 minUV, float2 maxUV, out float2 tapUVs[9])
{
    float2 uvMinus = clamp(centerUV - texelSize, minUV, maxUV);
    float2 uvPlus  = clamp(centerUV + texelSize, minUV, maxUV);
    tapUVs[0] = centerUV;
    tapUVs[1] = float2(centerUV.x, uvMinus.y);
    tapUVs[2] = float2(centerUV.x, uvPlus.y);
    tapUVs[3] = float2(uvMinus.x, centerUV.y);
    tapUVs[4] = float2(uvPlus.x,  centerUV.y);
    tapUVs[5] = float2(uvMinus.x, uvMinus.y);
    tapUVs[6] = float2(uvPlus.x,  uvMinus.y);
    tapUVs[7] = float2(uvMinus.x, uvPlus.y);
    tapUVs[8] = float2(uvPlus.x,  uvPlus.y);
}

// ============================================================================
// CAMERA BASIS & REPROJECTION
// ============================================================================
struct CameraBasis
{
    float3 rightTanFov;   // P: full-width right axis
    float3 forward;       // Q: top-left corner ray
    float3 downTanFov;    // R: full-height up axis
};

float3 BuildCameraRay(float2 uv, CameraBasis camera)
{
    return uv.x * camera.rightTanFov + camera.forward - uv.y * camera.downTanFov;
}

float2 ProjectRayToStableUV(float3 ray, float2 fallbackUV, float tanHalfFovX, float tanHalfFovY)
{
    if (ray.y <= kEpsilon) return fallbackUV;
    return float2((ray.x / (ray.y * max(tanHalfFovX, kEpsilon))) * 0.5 + 0.5,
                  0.5 - (ray.z / (ray.y * max(tanHalfFovY, kEpsilon))) * 0.5);
}

// FIX: documented explicitly -- this is the FORWARD map: frame-UV -> stable-UV
// (the stable position of the frame pixel's content). It is NOT its own
// inverse: the two directions differ by 2x the frame's jitter offset.
float2 ReprojectThroughCamera(float2 uv, CameraBasis camera, float2 fallbackUV, float tanHalfFovX, float tanHalfFovY)
{
    return ProjectRayToStableUV(BuildCameraRay(uv, camera), fallbackUV, tanHalfFovX, tanHalfFovY);
}

// FIX: the INVERSE map: stable-UV -> frame-UV (where the stable point sits in
// that frame). This is the reprojection base: with it, the velocity's jitter
// components cancel EXACTLY and a static scene reprojects to the identity.
// First-order reflection + one Newton refinement; residual ~1e-7 px for
// rotational sub-pixel jitter (the reflection alone is ~1e-4 px, already 20x
// below the 0.005 px reprojection noise floor).
float2 InverseReprojectThroughCamera(float2 stableUV, CameraBasis camera, float tanHalfFovX, float tanHalfFovY)
{
    float2 inv  = 2.0 * stableUV - ReprojectThroughCamera(stableUV, camera, stableUV, tanHalfFovX, tanHalfFovY);
    float2 fwd2 = ReprojectThroughCamera(inv, camera, inv, tanHalfFovX, tanHalfFovY);
    return inv - (fwd2 - stableUV);
}

float RayLengthFromUV(float2 uv, float tanHalfFovX, float tanHalfFovY)
{
    float2 tanXY = float2((uv.x * 2.0 - 1.0) * tanHalfFovX, (1.0 - uv.y * 2.0) * tanHalfFovY);
    return sqrt(1.0 + dot(tanXY, tanXY));
}

// ============================================================================
// GEOMETRIC LAYER CLASSIFICATION & SURFACE ANALYSIS
// ============================================================================
struct LayerSurface
{
    bool   isDilationZone;
    bool   isForegroundEdge;
    bool   isForeground;
    bool   edgeTowardCliff;      // foreground edge whose sub-texel phase points at its own silhouette
    int    closestIdx;
    float  closestDepth;
    float  gradX;
    float  gradY;
    float  effectiveDepth;
    float2 effectiveVelocityUV;
    float  layerVelSpreadPx;
};

void ComputeSurfaceGradients(float depthRaw[9], out float gradX, out float gradY)
{
    float centerDepth = depthRaw[0];
    float dxL = centerDepth - depthRaw[3];
    float dxR = depthRaw[4] - centerDepth;
    gradX = Minmod(dxL, dxR);

    float dyD = centerDepth - depthRaw[1];
    float dyU = depthRaw[2] - centerDepth;
    gradY = Minmod(dyD, dyU);
}

float2 BilerpVelocityQuad(float2 velocityUV[9], float2 fracPx)
{
    float2 v00 = velocityUV[0];
    float2 v10 = (fracPx.x >= 0.0) ? velocityUV[4] : velocityUV[3];
    float2 v01 = (fracPx.y >= 0.0) ? velocityUV[2] : velocityUV[1];
    float2 v11 = (fracPx.x >= 0.0)
        ? ((fracPx.y >= 0.0) ? velocityUV[8] : velocityUV[6])
        : ((fracPx.y >= 0.0) ? velocityUV[7] : velocityUV[5]);
    return Bilerp2x2(v00, v10, v01, v11, abs(fracPx));
}

LayerSurface ClassifyLayerSurface(
    float  depthRaw[9],
    float2 velocityUV[9],
    float2 fracPx,
    float2 sizePixels,
    bool   useDepthDilation,
    float  depthRejectionThresh,
    bool   measureVelSpread)
{
    LayerSurface s;
    float centerDepth = depthRaw[0];

    // 1. Surface gradients via minmod limiter across cardinal neighbors
    ComputeSurfaceGradients(depthRaw, s.gradX, s.gradY);

    // 2. Identify closest (foreground) neighbor and detect discontinuities in
    //    one pass. The per-axis cliff sides (which sides of the texel hold a
    //    farther-than-plane, off-layer neighbor) feed the foreground-edge
    //    phase test in step 3.
    s.closestDepth = centerDepth;
    s.closestIdx   = 0;
    float depthEps = depthRejectionThresh * centerDepth;
    bool hasCloserNeighbor  = false;
    bool hasFartherNeighbor = false;
    bool cliffPosX = false;
    bool cliffNegX = false;
    bool cliffPosY = false;
    bool cliffNegY = false;
    float depthDiff[9];

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        if (depthRaw[i] > s.closestDepth)
        {
            s.closestDepth = depthRaw[i];
            s.closestIdx   = i;
        }

        float predDepth = centerDepth + s.gradX * kOffsets3x3[i].x + s.gradY * kOffsets3x3[i].y;
        float diff      = depthRaw[i] - predDepth;
        depthDiff[i]    = diff;
        hasCloserNeighbor = hasCloserNeighbor || (diff > depthEps);
        if (diff < -depthEps)
        {
            hasFartherNeighbor = true;
            if (kOffsets3x3[i].x > 0)      cliffPosX = true;
            else if (kOffsets3x3[i].x < 0) cliffNegX = true;
            if (kOffsets3x3[i].y > 0)      cliffPosY = true;
            else if (kOffsets3x3[i].y < 0) cliffNegY = true;
        }
    }

    s.isDilationZone   = useDepthDilation && hasCloserNeighbor;
    s.isForegroundEdge = !s.isDilationZone && hasFartherNeighbor;
    s.isForeground     = s.isDilationZone || s.isForegroundEdge;

    // 3. Resolve effective layer depth and velocity.
    //
    //    EFFECTIVE-VALUE ANCHORING (the invariant the stored field relies on
    //    -- read before touching this):
    //      * DILATION ZONE: borrow the crest's raw sample outright -- the
    //        closest texel's depth AND velocity. The landing is then
    //        base(own stable position) + crest velocity = own position -
    //        object motion = the T-1 RING's position, and the T-1 ring stored
    //        the T-1 CREST's raw sample (the writer makes the identical
    //        choice): the comparison is crest-center material vs crest-center
    //        material (the crest translates rigidly with the object) -- an
    //        EXACT anchor match. Never phase-interpolate the dilated values
    //        toward fracPx: the ring's own quad is foreign-layer, and the
    //        mirrored crest anchoring is what makes the stored field line up
    //        at the landing.
    //      * FOREGROUND EDGE, phase TOWARD THE FLAT PART: the content position
    //        stays on the object's continuous surface, and the sign-selected
    //        bilinear quad lies entirely on the object. Full sub-pixel
    //        resolution: quad-interpolated velocity + minmod sub-pixel depth
    //        (the minmod slope at an edge IS the flat-side one-sided step --
    //        the cliff side's difference is the off-layer jump and minmod
    //        discards it, so the flat-direction extension uses the correct
    //        slope).
    //      * FOREGROUND EDGE, phase TOWARD THE SILHOUETTE: the content
    //        position hangs at/past the limb crossing. The minmod slope is
    //        the flat-side step -- a large underestimate of the outer flank
    //        of a curved silhouette (a round pole's raw-depth slope blows up
    //        toward the limb), so extrapolating it across the limb desyncs
    //        from the stored field by (slope_out - slope_in) * phase delta
    //        every frame: that is what falsely fired the depth envelope
    //        along the sides of round poles. Use the pixel's ACTUAL sample
    //        (own center depth + own center velocity) -- the dilation's
    //        point-sample rule anchored to the own texel. The sub-texel
    //        anchor offset is what the resolve's dual-sided slant tolerance
    //        covers. (The bilinear quad is never used in this phase either:
    //        it would mix the background's velocity across the limb and
    //        displace the landing by up to half the object's relative speed,
    //        mis-sampling the history entirely.)
    //      The effective values never blend across the silhouette in either
    //      phase. The phase test is PER-AXIS (a cliff on the -x side only
    //      matters for fracPx.x < 0), so diagonal phases toward a cliff are
    //      caught, ridges (cliffs on both sides) fall to the point sample
    //      for any nonzero cross-axis phase, and a phase away from a single
    //      cliff correctly keeps the bilinear.
    float subpixelDepth = centerDepth + s.gradX * fracPx.x + s.gradY * fracPx.y;

    s.edgeTowardCliff = s.isForegroundEdge &&
        ( (cliffPosX && fracPx.x > 0.0) || (cliffNegX && fracPx.x < 0.0) ||
          (cliffPosY && fracPx.y > 0.0) || (cliffNegY && fracPx.y < 0.0) );

    if (s.isDilationZone)
    {
        s.effectiveDepth      = s.closestDepth;
        s.effectiveVelocityUV = velocityUV[s.closestIdx];
    }
    else if (s.edgeTowardCliff)
    {
        s.effectiveDepth      = centerDepth;
        s.effectiveVelocityUV = velocityUV[0];
    }
    else
    {
        s.effectiveDepth      = subpixelDepth;
        s.effectiveVelocityUV = BilerpVelocityQuad(velocityUV, fracPx);
    }

    // 4. Measure layer-coherent spatial velocity spread (bypassed if velocity testing is disabled)
    s.layerVelSpreadPx = 0.0;
    if (measureVelSpread)
    {
        float maxVelDiffSq = 0.0;
        [unroll]
        for (int k = 1; k < 9; ++k)
        {
            bool sameLayer = s.isDilationZone ? (depthRaw[k] > centerDepth) : (abs(depthDiff[k]) <= depthEps);
            if (sameLayer)
            {
                float2 diffPx = (velocityUV[k] - s.effectiveVelocityUV) * sizePixels;
                maxVelDiffSq = max(maxVelDiffSq, dot(diffPx, diffPx) * kInvOffsetLenSq[k]);
            }
        }
        s.layerVelSpreadPx = sqrt(maxVelDiffSq);
    }

    return s;
}

// FIX: shared post-validation downgrade. The exact background-state values the
// resolve uses after its landing gate revokes a tentative dilation. Shared so
// the motion-field writer stores the SAME post-validation state the resolve
// actually resolved with: flag 0, subpixel background depth, bilerp
// background velocity.
void RevokeDilation(inout LayerSurface s, float centerDepthRaw, float2 velocityUV[9], float2 fracPx)
{
    s.isDilationZone   = false;
    s.isForegroundEdge = false;
    s.isForeground     = false;
    s.edgeTowardCliff  = false;
    s.effectiveDepth      = centerDepthRaw + s.gradX * fracPx.x + s.gradY * fracPx.y;
    s.effectiveVelocityUV = BilerpVelocityQuad(velocityUV, fracPx);
}

#endif // TAA_SHARED_H_HLSL