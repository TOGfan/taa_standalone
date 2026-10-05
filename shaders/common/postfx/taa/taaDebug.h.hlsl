// ============================================================================
// TAA debug views
// ----------------------------------------------------------------------------
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires fragments
// included before: taaShared.h.hlsl, taaConstants.h.hlsl, taaColor.h.hlsl
// (FromSpace).
// ============================================================================
#ifndef TAA_DEBUG_H_HLSL
#define TAA_DEBUG_H_HLSL

float4 DebugViewVelocity(float2 velocityUV, float2 sizePixels, float velocityScale, float centerDepthRaw)
{
    float2 velocityPx = abs(velocityUV * sizePixels) * (kDebugVelocityScale * velocityScale);
    return float4(float3(saturate(velocityPx), 0.0), centerDepthRaw);
}

float4 DebugViewLinearDepth(float centerDepthRaw)
{
    float linearDepth = saturate(LinearizeDepth(centerDepthRaw) / kDebugLinearDepthRange);
    return float4(float3(linearDepth, linearDepth, linearDepth), centerDepthRaw);
}

float4 DebugViewHistoryColor(float3 historyColorSpace, float centerDepthRaw)
{
    return float4(saturate(FromSpace(historyColorSpace)), centerDepthRaw);
}

float4 DebugViewEdgeState(float3 currentColorRGB, bool wasRevoked, bool isDilationZone, bool isForegroundEdge, bool velocityStraddled, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.25;
    if (wasRevoked)            debugColor = float3(1.0, 0.6, 0.0);  // orange: REVOKED dilation candidate
    else if (isDilationZone)   debugColor = float3(1.0, 0.05, 0.05); // red: kept dilation zone
    else if (isForegroundEdge) debugColor = float3(0.0, 0.85, 1.0);  // cyan: crest
    else if (velocityStraddled) debugColor = float3(1.0, 0.05, 1.0); // magenta: flat but velocity-straddled
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewDisocclusionBreakdown(
    float3 currentColorRGB, bool depthRejected, bool velocityRejected, bool alertSuppressed, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.1;
    if (depthRejected)    debugColor.r = 1.0;
    if (velocityRejected) debugColor.g = 1.0;
    if (depthRejected && velocityRejected) debugColor = float3(1.0, 1.0, 0.0);
    else if (alertSuppressed)              debugColor.b = 0.45; // velocity error flagged but pursuit unconfirmed
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewPursuit(float3 currentColorRGB, float divergencePx, bool velocityRejected, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.15;
    debugColor += float3(saturate(divergencePx * 0.5), 0.0, 0.0);
    if (velocityRejected) debugColor = float3(0.0, 1.0, 0.2); // green: confirmed divergence
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewAlignmentDrop(float3 currentColorRGB, float dropAmount, bool planarSurface, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.2;
    debugColor += float3(dropAmount,
                         planarSurface ? 0.25 : 0.0,
                         planarSurface ? 0.0  : 0.25);
    return float4(debugColor, centerDepthRaw);
}

float4 DebugViewHullClip(float3 currentColorRGB, float2 hullDiag, float centerDepthRaw)
{
    float3 debugColor = currentColorRGB * 0.1;
    if (hullDiag.x > 2.5)      debugColor.r = 1.0;    // unconverged: safe fallback bound
    else if (hullDiag.x > 1.5) debugColor.g = 0.5;    // supporting-only bound
    else if (hullDiag.x > 0.5) debugColor.g = 1.0;    // exact certificate fired
    debugColor.b = saturate(hullDiag.y * 0.2);       // pivots used
    return float4(debugColor, centerDepthRaw);
}

#endif // TAA_DEBUG_H_HLSL