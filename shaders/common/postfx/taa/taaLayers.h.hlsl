// ============================================================================
// TAA layer classification & the current-frame gather
// ----------------------------------------------------------------------------
// LAYER CLASSIFICATION: the DECISION (dilation candidate / foreground edge /
// flat) is the depth-curvature classifier (AnalyzeSurfaceEdgesCore in
// taaShared, shared with the motion-field writer), and the EFFECTIVE VALUES
// apply the foreground-edge PHASE rule, the crest-raw depth anchoring, and
// the two-tap similarity extrapolation (taaVelocity). The closest/second
// tap scan tracks VALUES, not indices: every subsequent access stays
// statically indexed, so the neighborhood arrays can live in registers
// instead of indexable-temp memory.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// samplers (depthTex, velocityTex), cbuffer perDraw (taaVelRejection),
// postFx macros (tex2Dlod). Requires fragments included before:
// taaShared.h.hlsl, taaConstants.h.hlsl, taaVelocity.h.hlsl.
// ============================================================================
#ifndef TAA_LAYERS_H_HLSL
#define TAA_LAYERS_H_HLSL

float Minmod(float a, float b)
{
    return (a * b > 0.0) ? ((abs(a) < abs(b)) ? a : b) : 0.0;
}

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

// The phase-selected 2x2 depth quad (shared by the foreground-edge and flat
// landing-depth paths).
float BilerpDepthQuad(float depthRaw[9], float2 fracPx)
{
    return Bilerp2x2(depthRaw[0],
        (fracPx.x >= 0.0) ? depthRaw[4] : depthRaw[3],
        (fracPx.y >= 0.0) ? depthRaw[2] : depthRaw[1],
        (fracPx.x >= 0.0)
            ? ((fracPx.y >= 0.0) ? depthRaw[8] : depthRaw[6])
            : ((fracPx.y >= 0.0) ? depthRaw[7] : depthRaw[5]),
        abs(fracPx));
}

struct LayerSurface
{
    bool   isDilationZone;
    bool   isForegroundEdge;
    bool   isForeground;       // dilation || edge (cleared on revocation)
    bool   edgeTowardCliff;    // phase rule selector
    float2 closestOffsetPx;    // offset of the closest (foreground) tap, in texels
    float  closestDepth;
    float  gradX, gradY;       // minmod (cliff plane + sub-pixel edge depth)
    float  effectiveDepth;
    float2 effectiveVelocityUV;
    // Two-tap similarity extrapolation diagnostics (all 0 when inactive /
    // revoked -- zero tolerance charges):
    float  pairGradPx;           // the active pair's per-texel velocity gradient
    float  shallowVelGradPx;     // the foreground's measured shallow-end velocity gradient
    float  extrapolationDispPx;  // |q|*|eval-anchor|: the landing shift it causes
};

LayerSurface ClassifyLayerSurface(
    float  depthRaw[9],
    float2 velocityUV[9],
    float2 fracPx,
    float2 sizePixels,
    float  coherenceRadiusPx,
    bool   useDepthDilation,
    bool   isLandingSite,
    SurfaceEdgeState edge)
{
    LayerSurface s;
    s.isDilationZone      = edge.isDilationZone;
    s.isForegroundEdge    = edge.isForegroundEdge;
    s.isForeground        = s.isDilationZone || s.isForegroundEdge;
    s.pairGradPx          = 0.0;
    s.shallowVelGradPx    = 0.0;
    s.extrapolationDispPx = 0.0;

    ComputeSurfaceGradients(depthRaw, s.gradX, s.gradY);

    // Cliff sides: which sides of the texel hold an off-layer FARTHER
    // (background) neighbor, below the minmod plane by edgeEps. The phase
    // test is PER-AXIS (diagonals caught, ridges fall to the point sample).
    // Only the foreground-edge branch consumes the result -- flats (the
    // majority of pixels) and dilation zones skip the scan.
    s.edgeTowardCliff = false;
    if (s.isForegroundEdge)
    {
        bool cliffPosX = false, cliffNegX = false, cliffPosY = false, cliffNegY = false;
        [unroll]
        for (int k = 1; k < 9; ++k)
        {
            float predDepth = depthRaw[0] + s.gradX * kOffsets3x3[k].x + s.gradY * kOffsets3x3[k].y;
            if (depthRaw[k] < predDepth - edge.edgeEps)
            {
                if (kOffsets3x3[k].x > 0)      cliffPosX = true;
                else if (kOffsets3x3[k].x < 0) cliffNegX = true;
                if (kOffsets3x3[k].y > 0)      cliffPosY = true;
                else if (kOffsets3x3[k].y < 0) cliffNegY = true;
            }
        }
        s.edgeTowardCliff =
            ( (cliffPosX && fracPx.x > 0.0) || (cliffNegX && fracPx.x < 0.0) ||
              (cliffPosY && fracPx.y > 0.0) || (cliffNegY && fracPx.y < 0.0) );
    }

    // Closest + second-closest (foreground) tap scan -- reverse-Z: larger
    // raw = nearer. The scan only feeds the foreground branches (the
    // dilation anchor, the cliff-phase extrapolation pair and its gates), so
    // flat and flat-phase edge pixels skip it.
    s.closestDepth      = depthRaw[0];
    s.closestOffsetPx   = float2(0.0, 0.0);
    float2 closestVelocityUV = velocityUV[0];
    float  secondDepth        = 0.0;
    float2 secondOffsetPx     = float2(0.0, 0.0);
    float2 secondVelocityUV   = velocityUV[0];
    if (s.isDilationZone || s.edgeTowardCliff)
    {
        [unroll]
        for (int i = 1; i < 9; ++i)
        {
            float depth = depthRaw[i];
            if (depth > s.closestDepth)
            {
                secondDepth       = s.closestDepth;
                secondOffsetPx    = s.closestOffsetPx;
                secondVelocityUV  = closestVelocityUV;
                s.closestDepth    = depth;
                s.closestOffsetPx = kOffsets3x3[i];
                closestVelocityUV = velocityUV[i];
            }
            else if (depth > secondDepth)
            {
                secondDepth      = depth;
                secondOffsetPx   = kOffsets3x3[i];
                secondVelocityUV = velocityUV[i];
            }
        }
    }

    // Extrapolation inputs, measured only where the extrapolation can run
    // (dilation zones and crest cliff-phases; everything else pays nothing).
    float velQuantPx     = kVelQuantFloorPx;
    float shallowSlope   = 0.0;
    float shallowVelGrad = 0.0;
    float fgCoherentGrad = 0.0;
    if (s.isDilationZone || s.edgeTowardCliff)
    {
        velQuantPx = MeasureVelocityQuantStepPx(velocityUV, sizePixels);
        float gateRadiusPx = max(taaVelRejection, velQuantPx);
        fgCoherentGrad = MeasureVelocityCoherentGradientPx(
            closestVelocityUV, velocityUV, sizePixels, gateRadiusPx);
        MeasureShallowForegroundGeometry(
            depthRaw, velocityUV, sizePixels,
            fgCoherentGrad, velQuantPx,
            !s.isDilationZone,          // centerOnObject
            shallowSlope, shallowVelGrad);
    }
    s.shallowVelGradPx = shallowVelGrad;

    // Effective values.
    float subpixelDepth = depthRaw[0] + s.gradX * fracPx.x + s.gradY * fracPx.y;

    if (s.isDilationZone)
    {
        // Background behind a foreground crest: the pixel acts as the
        // FOREGROUND, and its velocity is that layer's field EVALUATED AT
        // THIS PIXEL (the two-tap similarity extrapolation; gate failures
        // fall back to the closest tap's raw sample).
        s.effectiveDepth = s.closestDepth;
        s.effectiveVelocityUV = ResolveForegroundFieldVelocityUV(
            s.closestOffsetPx, s.closestDepth, closestVelocityUV,
            secondOffsetPx, secondDepth, secondVelocityUV,
            true, depthRaw[0],      // the second tap must be foreground-side of the background center
            fracPx,
            shallowSlope, edge.depthQuantStep,
            fgCoherentGrad, velQuantPx,
            sizePixels,
            s.pairGradPx, s.extrapolationDispPx);
    }
    else if (s.edgeTowardCliff)
    {
        // Phase toward the silhouette: the content position hangs at/past
        // the limb; the velocity is the object's field at the exact phase --
        // the same extrapolation anchored at the center. Gate failures fall
        // back to the own center raw sample. The depth stays the own center.
        //
        // PARTNER SELECTION: the pair partner must be a tap OTHER than the
        // center. When the center is the scan's closest tap (the typical
        // crest) the second-closest is that partner. When some neighbor is
        // deeper still, the demotion chain leaves the second-closest AT the
        // center itself -- a degenerate (center, center) pair whose zero
        // span always trips the dSqPx gate and silently disables the
        // extrapolation on exactly the branch that exists to use it. In
        // that case the closest tap (the deepest foreground neighbor) is
        // the partner: temporally stable, admitted through the identical
        // gates, and resolving depth ties in scan order -- the same
        // stability class as the original second-closest pick.
        bool   closestIsCenter = (s.closestOffsetPx.x == 0.0) && (s.closestOffsetPx.y == 0.0);
        float2 partnerOffsetPx = closestIsCenter ? secondOffsetPx    : s.closestOffsetPx;
        float  partnerDepth    = closestIsCenter ? secondDepth       : s.closestDepth;
        float2 partnerVelocity = closestIsCenter ? secondVelocityUV  : closestVelocityUV;

        s.effectiveDepth = depthRaw[0];
        s.effectiveVelocityUV = ResolveForegroundFieldVelocityUV(
            float2(0.0, 0.0), depthRaw[0], velocityUV[0],
            partnerOffsetPx, partnerDepth, partnerVelocity,
            false, 0.0,
            fracPx,
            shallowSlope, edge.depthQuantStep,
            fgCoherentGrad, velQuantPx,
            sizePixels,
            s.pairGradPx, s.extrapolationDispPx);
    }
    else if (s.isForegroundEdge)
    {
        // Phase toward the flat part: the on-object quad, full sub-pixel
        // treatment.
        s.effectiveDepth      = isLandingSite ? Bilerp2x2(
            depthRaw[0],
            (fracPx.x >= 0.0) ? depthRaw[4] : depthRaw[3],
            (fracPx.y >= 0.0) ? depthRaw[2] : depthRaw[1],
            (fracPx.x >= 0.0)
                ? ((fracPx.y >= 0.0) ? depthRaw[8] : depthRaw[6])
                : ((fracPx.y >= 0.0) ? depthRaw[7] : depthRaw[5]),
            abs(fracPx)) : subpixelDepth;
        s.effectiveVelocityUV = BilerpVelocityQuad(velocityUV, fracPx);
    }
    else
    {
        // Flat: center depth at the current frame / bilinear at landing
        // sites, plus the layer-aware velocity selection.
        s.effectiveDepth      = isLandingSite ? BilerpDepthQuad(depthRaw, fracPx) : depthRaw[0];
        s.effectiveVelocityUV = SelectLayerAwareQuadVelocity(velocityUV, fracPx, sizePixels, coherenceRadiusPx);
    }
    return s;
}

// Shared post-validation downgrade: the texel's OWN raw background sample.
// The extrapolation diagnostics are zeroed -- no tolerance charges.
void RevokeDilation(inout LayerSurface s, float centerDepthRaw, float2 centerVelocityUV)
{
    s.isDilationZone      = false;
    s.isForegroundEdge    = false;
    s.isForeground        = false;
    s.edgeTowardCliff     = false;
    s.pairGradPx          = 0.0;
    s.shallowVelGradPx    = 0.0;
    s.extrapolationDispPx = 0.0;
    s.effectiveDepth      = centerDepthRaw;
    s.effectiveVelocityUV = centerVelocityUV;
}

// ============================================================================
// CURRENT-FRAME GATHER
// ============================================================================
// Raw, unmodified 3x3 current-frame samples plus the closest-depth tap scan.
// Kept RAW (no dilation, no interpolation) so the divergence derivatives in
// the disocclusion tests stay clean.
struct CurrentFrameNeighborhood
{
    float  depthRaw[9];
    float2 velocityJitteredUV[9];

    float  closestDepthRaw;
    float  secondClosestDepthRaw;
    float2 closestOffsetPx;
    float2 secondClosestOffsetPx;
};

CurrentFrameNeighborhood GatherCurrentFrameNeighborhood(
    float2 snappedUV,
    float2 tapUVs[9],
    float  centerDepthRaw,
    float2 centerVelocityJitteredUV,
    bool   fetchNeighborDepth,
    bool   fetchNeighborVelocity)
{
    CurrentFrameNeighborhood n;

    n.depthRaw[0]           = centerDepthRaw;
    n.velocityJitteredUV[0] = centerVelocityJitteredUV;

    [unroll]
    for (int i = 1; i < 9; ++i)
    {
        // When neighbor fetches are disabled the center values are replicated so
        // everything downstream sees a perfectly flat neighborhood.
        float  depth = centerDepthRaw;
        if (fetchNeighborDepth)    { depth = tex2Dlod(depthTex,    float4(tapUVs[i], 0.0, 0.0)).r; }
        float2 velocity = centerVelocityJitteredUV;
        if (fetchNeighborVelocity) { velocity = tex2Dlod(velocityTex, float4(tapUVs[i], 0.0, 0.0)).rg; }

        n.depthRaw[i]           = depth;
        n.velocityJitteredUV[i] = velocity;
    }

    // Closest-depth (foreground) tap scan. "Closest" = largest raw depth
    // (reverse-Z). This scan is identical to the one inside
    // ClassifyLayerSurface.
    n.closestDepthRaw       = centerDepthRaw;
    n.secondClosestDepthRaw = 0.0;
    n.closestOffsetPx       = float2(0.0, 0.0);
    n.secondClosestOffsetPx = float2(0.0, 0.0);

    if (fetchNeighborDepth)
    {
        [unroll]
        for (int i = 1; i < 9; ++i)
        {
            float depth = n.depthRaw[i];
            if (depth > n.closestDepthRaw)
            {
                n.secondClosestDepthRaw = n.closestDepthRaw;
                n.secondClosestOffsetPx = n.closestOffsetPx;
                n.closestDepthRaw       = depth;
                n.closestOffsetPx       = kOffsets3x3[i];
            }
            else if (depth > n.secondClosestDepthRaw)
            {
                n.secondClosestDepthRaw = depth;
                n.secondClosestOffsetPx = kOffsets3x3[i];
            }
        }
    }

    return n;
}

SurfaceEdgeState AnalyzeSurfaceEdges(CurrentFrameNeighborhood neighborhood, bool useDepthDilation)
{
    return AnalyzeSurfaceEdgesCore(neighborhood.depthRaw, useDepthDilation);
}

// Foreground crest geometry (feeds the disocclusion tolerances).
struct ForegroundGeometry
{
    float slope;
};

ForegroundGeometry ComputeForegroundGeometry(
    CurrentFrameNeighborhood neighborhood, float resolvedDepthRaw, SurfaceEdgeState edge)
{
    ForegroundGeometry fg;

    float crestDrop   = max(neighborhood.closestDepthRaw - resolvedDepthRaw, 0.0);
    float crestSpanPx = max(length(neighborhood.closestOffsetPx), 1.0);
    float crestSlope  = crestDrop / crestSpanPx;

    float layerGap   = neighborhood.closestDepthRaw - neighborhood.secondClosestDepthRaw;
    bool  sameObject = (layerGap < edge.edgeEps * 3.5);
    float gapSpanPx  = max(length(neighborhood.closestOffsetPx - neighborhood.secondClosestOffsetPx), 1.0);
    float layerGapSlope = sameObject ? (max(layerGap, 0.0) / gapSpanPx) : 0.0;

    fg.slope = max(crestSlope, layerGapSlope);
    return fg;
}

#endif // TAA_LAYERS_H_HLSL