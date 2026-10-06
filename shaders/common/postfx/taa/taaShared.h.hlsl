// ============================================================================
// TAA shared math (included by taa.fx.hlsl, taaMotion.fx.hlsl and
// taaFinal.fx.hlsl)
// ----------------------------------------------------------------------------
// PURE functions only -- no cbuffer or sampler access. Everything the layer
// classification needs is passed in, so the resolve pass and the motion-field
// writer are guaranteed to classify the current frame IDENTICALLY (a stored
// field that disagrees with the resolve's own classification would lie to the
// next frame's disocclusion tests).
//
// BASIS CONVENTION: P = the full-width right axis, R = the full-height up
// axis, Q = the top-left corner ray -- the only linear arrangement for which
// ProjectRayToStableUV(BuildCameraRay(uv, cam)) = uv exactly for a STABLE
// basis. normalize(Q) points at the corner, NOT forward.
//
// MAP DIRECTIONS (read before touching the reprojection):
//   ReprojectThroughCamera(uv, cam)     frame-UV -> stable-UV (the stable
//                                       position of the frame pixel's content).
//   InverseReprojectThroughCamera(u, c) stable-UV -> frame-UV (where the stable
//                                       point sits in that frame).
//   They differ by 2x the frame's jitter and are NOT interchangeable. The
//   resolve SAMPLES and REPROJECTS through the INVERSE map (the stable
//   pixel's content sits there; with it as the velocity base a static scene
//   reprojects to the identity); the FORWARD map is used only to derive the
//   jitter offset s_t.
//
// LAYER CLASSIFICATION: the depth-curvature classifier (dilation zone /
// foreground edge / flat) is THE classifier -- shared verbatim by the current
// frame, the history landing and the motion-field writer's flag channel, so
// every landing site resolves with identical layer semantics.
//
// CLIP-STATE SIGMA HELPERS: the 7-bit log2 encoding of the temporal clip
// record's sigma lives HERE (not taaClip) because TWO transports carry it:
// the clip-state alpha (taaClip) and the debug payload (below). One encoding,
// two packings; DecodeClipSigmaSq accepts both tags.
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

float LumaRGB(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

// Standard bilinear interpolation of a 2x2 quad (fraction in [0,1]).
float  Bilerp2x2(float  c00, float  c10, float  c01, float  c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}
float2 Bilerp2x2(float2 c00, float2 c10, float2 c01, float2 c11, float2 fraction)
{
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

// ============================================================================
// DEBUG PAYLOAD TRANSPORT (through #TAA_Result.a while a debug mode is on)
// ----------------------------------------------------------------------------
// The resolve's single render target must always carry the REAL blended
// color (the history-copy child stores it into #TAA_History unconditionally
// -- debug colors in RGB would poison the accumulation). The view state
// rides the alpha instead: the magnitude's only consumers are taaFinal's
// auto-sharpener (bypassed in debug) and the writer's revocation SIGN (kept
// genuine). Bit layout: 31 = revocation sign; 30..27 = 0111 (finite,
// non-subnormal, never NaN/Inf); 26..23 = view code; 22..13 = payload A
// [0,1] (10 bits); 12..7 = payload B [0,1] (6 bits); 6..0 = the temporal
// clip-state sigma code (7 bits, the same encoding as the clip-state
// alpha's field -- see taaClip.h.hlsl).
//
// Carrying the clip state INSIDE the debug payload is what makes the state
// OBSERVABLE: without it, every debug frame overwrites the alpha with a
// foreign tag, the next frame's record reads cold, and debug mode 11 shows
// permanently red -- the view destroying the state it displays. With it the
// record chain runs uninterrupted through debugging. Payload B's coarser
// 6 bits are display-only in every view (a boolean or a coarse channel);
// the 0.125-quantized bit fields decode exactly through round(b*63)/63.
// Exact through an RGBA32F target with point sampling (no filtering on the
// store path).
// ============================================================================
static const float kClipSigmaRef = 1.0 / 255.0;   // the clip-state sigma's unit

float ClipPackSigmaCode(float s)
{
    return clamp((log2(max(s, 1e-6) / kClipSigmaRef) + 8.0) * 8.0, 0.0, 127.0);
}

float ClipUnpackSigmaCode(uint code)
{
    return kClipSigmaRef * exp2(code * 0.125 - 8.0);
}

float PackDebugAlpha(bool revoked, float code, float a, float b, float sigma)
{
    uint u = 0x38000000u
           | ((uint(code + 0.5) & 0xFu)        << 23)
           | ((uint(a * 1023.0 + 0.5) & 0x3FFu) << 13)
           | ((uint(b * 63.0 + 0.5)   & 0x3Fu)  << 7)
           |  (uint(ClipPackSigmaCode(sigma) + 0.5) & 0x7Fu);
    return asfloat(revoked ? (u | 0x80000000u) : u);
}

// ============================================================================
// ACUTANCE TRANSPORT DECODE (the clip-state alpha's [19:8] field)
// ----------------------------------------------------------------------------
// The resolve packs the raw-scene acutance target SQRT-COMPRESSED (packed
// linear [0,1] <-> energy [0,9]; the old linear packing saturated at E = 1
// -- every full-contrast LDR edge and all HDR content) and TEMPORALLY
// STABILIZED (an EWMA at kSharpEwmaRate against the previous frame's
// decoded value at the landing; the raw measurement swings ~2x across the
// jitter phase cycle on edges and would flicker the sharpener's boost).
// DecodeAcutanceEnergy returns the ENERGY; a foreign tag (debug payload,
// cleared buffer, NaN) decodes as zero: no boost, never a spurious one.
// ============================================================================
float DecodeAcutanceLinear(float alphaValue)
{
    uint u = asuint(alphaValue);
    if (((u >> 27) & 0xFu) == 0x6u)                     // clip-state tag
        return (float)((u >> 8) & 0xFFFu) * (1.0 / 4095.0);
    return 0.0;
}

float DecodeAcutanceEnergy(float alphaValue)
{
    float lin = DecodeAcutanceLinear(alphaValue);
    return 9.0 * lin * lin;                             // energy domain
}



void UnpackDebugAlpha(float alpha, out uint code, out float a, out float b)
{
    uint u = asuint(alpha);
    code = (u >> 23) & 0xFu;
    a    = (float)((u >> 13) & 0x3FFu) * (1.0 / 1023.0);
    b    = (float)((u >> 7) & 0x3Fu) * (1.0 / 63.0);
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
    float2 snappedUV;   // clamped texel-center UV used for all current-frame taps
    float2 fracPx;      // sub-texel phase of the snapped position, range [-0.5, 0.5]
};

SnappedCoord SnapUVToTexel(float2 uv, ViewportParams vp)
{
    SnappedCoord sc;
    float2 pixelPos  = uv * vp.sizePixels;
    float2 baseTexel = floor(pixelPos) + 0.5;   // nearest texel center
    sc.snappedUV     = clamp(baseTexel * vp.texelSize, vp.minUV, vp.maxUV);
    sc.fracPx        = pixelPos - baseTexel;
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

// FORWARD map: frame-UV -> stable-UV (the stable position of the frame pixel's
// content). It is NOT its own inverse: the two directions differ by 2x the
// frame's jitter offset.
float2 ReprojectThroughCamera(float2 uv, CameraBasis camera, float2 fallbackUV, float tanHalfFovX, float tanHalfFovY)
{
    return ProjectRayToStableUV(BuildCameraRay(uv, camera), fallbackUV, tanHalfFovX, tanHalfFovY);
}

// INVERSE map: stable-UV -> frame-UV (where the stable point sits in that
// frame). First-order reflection; residual ~1e-4 px for rotational sub-pixel
// jitter -- 20x below the 0.005 px reprojection noise floor and orders below
// every consumer's quantization (the snap grid is 0.5 px, the velocity
// encoding floor is ~0.06 px).
float2 InverseReprojectThroughCamera(float2 stableUV, CameraBasis camera, float tanHalfFovX, float tanHalfFovY)
{
    return 2.0 * stableUV - ReprojectThroughCamera(stableUV, camera, stableUV, tanHalfFovX, tanHalfFovY);
}

// ============================================================================
// CAMERA DIFFERENTIALS (depth-disocclusion support)
// ----------------------------------------------------------------------------
// Exact differential quantities of the camera maps, used by the resolve's
// forward-parallax fit. All pure; the stable basis yields CameraForwardAxis
// = (0,1,0) and CameraForwardJacobian = identity -- usable as unit checks.
// ============================================================================

// The camera's forward (principal) direction: the center ray of the basis.
float3 CameraForwardAxis(CameraBasis camera)
{
    return normalize(camera.rightTanFov * 0.5 + camera.forward - camera.downTanFov * 0.5);
}

// Row-major 2x2 helpers, packed as (m00, m01, m10, m11).
float2 Apply2x2(float4 m, float2 v)
{
    return float2(m.x * v.x + m.y * v.y, m.z * v.x + m.w * v.y);
}

float4 Mul2x2(float4 a, float4 b)
{
    return float4(a.x * b.x + a.y * b.z, a.x * b.y + a.y * b.w,
                  a.z * b.x + a.w * b.z, a.z * b.y + a.w * b.w);
}

float4 Inv2x2(float4 m)
{
    float det = m.x * m.w - m.y * m.z;
    if (abs(det) < 1e-12) return float4(1.0, 0.0, 0.0, 1.0);
    float inv = 1.0 / det;
    return float4(m.w * inv, -m.y * inv, -m.z * inv, m.x * inv);
}

// Row-major 2x2 Jacobian of the FORWARD map u -> ReprojectThroughCamera(u,
// camera), analytic and exact for any rotation. With r = BuildCameraRay(u):
//   F.x = 0.5 + 0.5*r.x/(r.y*tanX),  F.y = 0.5 - 0.5*r.z/(r.y*tanY)
//   dr/du.x = P,  dr/du.y = -R
float4 CameraForwardJacobian(float2 u, CameraBasis camera, float tanHalfFovX, float tanHalfFovY)
{
    float3 r  = BuildCameraRay(u, camera);
    float  ry = max(r.y, 1e-4);
    float3 P  = camera.rightTanFov;
    float3 R  = camera.downTanFov;
    float invX = 1.0 / (ry * ry * max(tanHalfFovX, 1e-4));
    float invY = 1.0 / (ry * ry * max(tanHalfFovY, 1e-4));
    return float4(
        0.5 * ( P.x * ry - r.x * P.y) * invX,   // dF.x/du.x
        0.5 * ( r.x * R.y - R.x * ry ) * invX,  // dF.x/du.y
       -0.5 * ( P.z * ry - r.z * P.y) * invY,   // dF.y/du.x
        0.5 * ( R.z * ry - r.z * R.y) * invY);  // dF.y/du.y
}

// ============================================================================
// LAYER CLASSIFICATION (the depth-curvature classifier -- THE classifier)
// ----------------------------------------------------------------------------
// Core classifier on a raw-depth 3x3, shared by the CURRENT frame, the
// HISTORY landing and the PURSUIT landing (and the motion writer's flag), so
// every frame side resolves with EXACTLY the same layer semantics. Without a
// shared classifier, "foreground reprojecting into its own previous-frame
// dilation zone" can never validate consistently.
// ============================================================================
struct SurfaceEdgeState
{
    bool  isDilationZone;        // center BEHIND neighbors: background behind a foreground crest
    bool  isForegroundEdge;      // center IN FRONT of neighbors: the crest itself
    float edgeEps;               // depth separation that counts as an edge between layers
    float maxNeighborDepthSlope; // largest depth slope across neighbor pairs ("per-2-texel" units)
    float planeDepthNoise;       // depth noise estimate from the max slope
    float depthNoiseFloor;       // lower bound of measurable depth noise
    float depthQuantStep;        // depth quantization step estimate
};

SurfaceEdgeState AnalyzeSurfaceEdgesCore(float depthRaw[9], bool useDepthDilation)
{
    SurfaceEdgeState s;
    float centerDepth = depthRaw[0];

    // Center-minus-neighbor-pair second differences ("curvature"): they sign
    // the center's position relative to its neighborhood.
    float curvatureH  = centerDepth - 0.5 * (depthRaw[3] + depthRaw[4]);
    float curvatureV  = centerDepth - 0.5 * (depthRaw[1] + depthRaw[2]);
    float curvatureD1 = centerDepth - 0.5 * (depthRaw[5] + depthRaw[8]);
    float curvatureD2 = centerDepth - 0.5 * (depthRaw[6] + depthRaw[7]);
    float minCurvature = min(min(curvatureH, curvatureV), min(curvatureD1, curvatureD2));
    float maxCurvature = max(max(curvatureH, curvatureV), max(curvatureD1, curvatureD2));

    // Largest slope between neighbor pairs (diagonals rescaled to consistent
    // "per-2-texel" units; thresholds tuned around this scale).
    s.maxNeighborDepthSlope = max(
        max(abs(depthRaw[4] - depthRaw[3]),
            abs(depthRaw[2] - depthRaw[1])),
        max(abs(depthRaw[8] - depthRaw[5]),
            abs(depthRaw[6] - depthRaw[7])) * 0.7071);
    s.planeDepthNoise = s.maxNeighborDepthSlope * 0.05;

    // Depth separation that counts as an edge between two layers.
    s.edgeEps = max(max(centerDepth, 1e-6) * 0.008, s.planeDepthNoise) + 1e-6;

    // Center behind its neighbors -> background behind a foreground crest:
    // velocity/depth must come from the foreground layer (dilation).
    s.isDilationZone = useDepthDilation && (minCurvature < -s.edgeEps);
    // Center in front of its neighbors -> the crest itself: the center tap is
    // the most representative sample.
    s.isForegroundEdge = !s.isDilationZone && (maxCurvature > s.edgeEps);

    // Lower bound of measurable depth noise.
    float minAbsCurvature = min(min(abs(curvatureH), abs(curvatureV)),
                                min(abs(curvatureD1), abs(curvatureD2)));
    s.depthNoiseFloor = max(minAbsCurvature, s.planeDepthNoise);

    // Depth quantization step estimate: smallest nonzero curvature (0 if none).
    float minNonZeroCurvature = 1.0;
    if (abs(curvatureH)  > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureH));
    if (abs(curvatureV)  > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureV));
    if (abs(curvatureD1) > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureD1));
    if (abs(curvatureD2) > 1e-9) minNonZeroCurvature = min(minNonZeroCurvature, abs(curvatureD2));
    s.depthQuantStep = (minNonZeroCurvature < 1.0) ? minNonZeroCurvature : 0.0;

    return s;
}

#endif // TAA_SHARED_H_HLSL