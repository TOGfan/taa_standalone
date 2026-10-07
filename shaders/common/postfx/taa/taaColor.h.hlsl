// ============================================================================
// TAA working color spaces & reversible tonemapping
// ----------------------------------------------------------------------------
// Oklab / YCoCg conversions, the reversible luma tonemap, gamut recompression
// and the raw-scene acutance metric shared with taaFinal's sharpener.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// cbuffer perDraw (taaColorSpaceOklab). Requires fragments included before:
// taaShared.h.hlsl (LumaRGB, kEpsilon), taaConstants.h.hlsl.
// ============================================================================
#ifndef TAA_COLOR_H_HLSL
#define TAA_COLOR_H_HLSL

// Reversible luma tonemap keeps bright fireflies from dominating the clip box.
float3 Tonemap(float3 c)   { float luma = LumaRGB(c); return c / (1.0 + luma); }
float3 Untonemap(float3 c) { float luma = LumaRGB(c); return c / max(1.0 - min(luma, 0.999), 1e-4); }

static const float3x3 kRGB_TO_LMS    = float3x3(0.41222147, 0.53633253, 0.05144599, 0.21190349, 0.68069954, 0.10739695, 0.08830246, 0.28171883, 0.62997870);
static const float3x3 kLMS_TO_OKLAB  = float3x3(0.21045425,  0.79361778, -0.00407204, 1.97799849, -2.42859220,  0.45059370, 0.02590403,  0.78277176, -0.80867576);
static const float3x3 kOKLAB_TO_LMS  = float3x3(1.0,  0.39633777,  0.21580375, 1.0, -0.10556134, -0.06385417, 1.0, -0.08948417, -1.29148554);
static const float3x3 kLMS_TO_RGB    = float3x3(4.07674166, -3.30771159,  0.23096992, -1.26843800, 2.60975740, -0.34131939, -0.00419608, -0.70341861, 1.70761470);

float3 RGBToOklab(float3 c)
{
    float3 lms = max(mul(kRGB_TO_LMS, c), 0.0);
    // pow(0, y) is NaN on some drivers: guard each channel to x > 0.
    float3 lmsRoot;
    lmsRoot.x = (lms.x > 0.0) ? pow(lms.x, 1.0 / 3.0) : 0.0;
    lmsRoot.y = (lms.y > 0.0) ? pow(lms.y, 1.0 / 3.0) : 0.0;
    lmsRoot.z = (lms.z > 0.0) ? pow(lms.z, 1.0 / 3.0) : 0.0;
    return mul(kLMS_TO_OKLAB, lmsRoot);
}

float3 OklabToRGB(float3 c) { float3 l = mul(kOKLAB_TO_LMS, c); return mul(kLMS_TO_RGB, l * l * l); }
float3 RGBToYCoCg(float3 c) { return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
float3 YCoCgToRGB(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// RGB <-> tonemapped working space (Oklab or YCoCg, selected by config).
// BRANCH, not ternary: ?: materializes BOTH conversions and selects -- in
// YCoCg mode that wasted ~27 pow (cube roots) + 27 matrix applies per pixel.
// The condition is a per-draw uniform; a coherent branch skips the dead side.
float3 ToSpace(float3 rgb)
{
    float3 t = Tonemap(max(0.0, rgb));
    if (taaColorSpaceOklab > 0.5)
        return RGBToOklab(t);
    return RGBToYCoCg(t);
}
float3 FromSpace(float3 c)
{
    if (taaColorSpaceOklab > 0.5)
        return Untonemap(max(0.0, OklabToRGB(c)));
    return Untonemap(max(0.0, YCoCgToRGB(c)));
}

// Clamp the history color back into the valid RGB gamut if clipping pushed
// chroma outside (keeps hue, pulls toward the luma axis).
float3 CompressGamut(float3 historyColorSpace)
{
    if (taaColorSpaceOklab > 0.5)
    {
        float3 historyRGB = OklabToRGB(historyColorSpace);
        float minChannel = min(historyRGB.r, min(historyRGB.g, historyRGB.b));
        if (minChannel < 0.0)
        {
            float luma  = max(0.0, LumaRGB(historyRGB));
            float scale = saturate(luma / max(luma - minChannel, kEpsilon));
            historyRGB = luma + (historyRGB - luma) * scale;
            historyColorSpace = RGBToOklab(historyRGB);
        }
        return historyColorSpace;
    }

    // YCoCg: reconstruct RGB to detect gamut violation.
    float Y  = historyColorSpace.x;
    float Co = historyColorSpace.y;
    float Cg = historyColorSpace.z;
    float r = Y + Co - Cg;
    float g = Y + Cg;
    float b = Y - Co - Cg;
    float minChannel = min(r, min(g, b));
    if (minChannel < 0.0)
    {
        float scale = saturate(max(0.0, Y) / max(Y - minChannel, kEpsilon));
        historyColorSpace.yz *= scale;
    }
    return historyColorSpace;
}

float LinearizeDepth(float rawDepth) { return 1.0 / max(rawDepth, kEpsilon); }

// ============================================================================
// RAW-SCENE ACUTANCE METRIC (auto-parity sharpener input)
// ----------------------------------------------------------------------------
// FSR SRTM + RCAS luma (x2) -- the exact space taaFinal.fx.hlsl measures the
// resolved image in, so the energy ratio between the two is meaningful.
// ============================================================================
float SrtmLumaFSR(float3 rgb)
{
    float3 c = max(rgb, 0.0);
    c *= 1.0 / (max(c.r, max(c.g, c.b)) + 1.0);
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

#endif // TAA_COLOR_H_HLSL