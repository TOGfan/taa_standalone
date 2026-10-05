// ============================================================================
// TAA debug view rendering (included by taaFinal.fx.hlsl)
// ----------------------------------------------------------------------------
// The resolve NEVER writes debug colors: its RGB is always the real blended
// color (the history-copy child stores it into #TAA_History unconditionally)
// and the view state rides #TAA_Result.a instead, packed by PackDebugAlpha
// (taaShared.h.hlsl). This file turns that payload back into the on-screen
// view, so debug colors exist ONLY in $backBuffer and the temporal
// accumulation continues undisturbed while debugging.
//
// SEMANTIC NOTES vs. the old early-return views:
//   * View bases use the RESOLVED output (the blend), not the raw current
//     frame -- under convergence they coincide, and the resolved base keeps
//     the views stable frame-to-frame.
//   * Mode 5 shows the STORED #TAA_History buffer (the exact input of this
//     frame's blend), read directly by taaFinal -- previously it showed the
//     Kaiser-resampled sample at the landing position. Showing the stored
//     buffer is what you want when auditing the accumulation itself.
//   * Flag payloads pack small integer bit fields into payload A as f/8
//     (10-bit quantization sits far below the rounding threshold).
//
// FRAGMENT HEADER: compiled only inside taaFinal.fx.hlsl. Requires fragments
// included before: taaShared.h.hlsl (UnpackDebugAlpha).
// ============================================================================
#ifndef TAA_DEBUG_H_HLSL
#define TAA_DEBUG_H_HLSL

float3 RenderDebugView(uint code, float a, float b, bool revoked,
                       float3 resultRGB, float3 historyRGB)
{
    switch (code)
    {
    // 1 / 3 / 6: velocity views. A and B are the pre-saturated display
    // channels (|v| * kDebugVelocityScale * modeScale) that the old
    // DebugViewVelocity produced.
    case 1:
    case 3:
    case 6:
        return float3(a, b, 0.0);

    // 2: disocclusion breakdown. A bits: 4 = depth rejected, 2 = velocity
    // rejected, 1 = suppressed pursuit alert.
    case 2: {
        uint f = (uint)(a * 8.0 + 0.5);
        bool depthRejected = (f & 4u) != 0u;
        bool velRejected   = (f & 2u) != 0u;
        bool alert         = (f & 1u) != 0u;
        float3 c = resultRGB * 0.1;
        if (depthRejected) c.r = 1.0;
        if (velRejected)   c.g = 1.0;
        if (depthRejected && velRejected) c = float3(1.0, 1.0, 0.0);
        else if (alert)                    c.b = 0.45;
        return c;
    }

    // 4: linearized depth (pre-saturated grayscale value in A).
    case 4:
        return float3(a, a, a);

    // 5: the stored history buffer.
    case 5:
        return saturate(historyRGB);

    // 7: pursuit. A = saturate(divergencePx * 0.5), B = confirmed rejection.
    case 7: {
        float3 c = resultRGB * 0.15 + float3(a, 0.0, 0.0);
        if (b > 0.5) c = float3(0.0, 1.0, 0.2);
        return c;
    }

    // 8: layer state. A bits: 4 = dilation zone, 2 = foreground edge,
    // 1 = velocity-straddled; the revocation rides the alpha SIGN.
    case 8: {
        uint f = (uint)(a * 8.0 + 0.5);
        if (revoked)        return float3(1.0, 0.6, 0.0);
        if ((f & 4u) != 0u) return float3(1.0, 0.05, 0.05);
        if ((f & 2u) != 0u) return float3(0.0, 0.85, 1.0);
        if ((f & 1u) != 0u) return float3(1.0, 0.05, 1.0);
        return resultRGB * 0.25;
    }

    // 9: dilation-gate breakdown. A bits: 4 = kept candidate, 2 = kept via
    // the stored-flag branch (else the depth branch), 1 = depth rejected;
    // the revocation rides the alpha SIGN.
    case 9: {
        uint f = (uint)(a * 8.0 + 0.5);
        float3 c = resultRGB * 0.1;
        if (revoked)        c.r = 1.0;
        if ((f & 4u) != 0u) c.g = ((f & 2u) != 0u) ? 1.0 : 0.5;
        if ((f & 1u) != 0u) c.b = 1.0;
        return c;
    }

    // 10: alignment-drop activity. A = drop amount, B = planar surface.
    case 10: {
        bool planar = (b > 0.5);
        return resultRGB * 0.2 + float3(a, planar ? 0.25 : 0.0, planar ? 0.0 : 0.25);
    }

    // 11: hull clipper state. A decodes to the code: 1 = exact (simplex
    // certificate or the collinear 1D solve), 2 = certified dual bound,
    // 3 = defensive (should never fire). B = iterations used (pre-scaled).
    case 11: {
        float3 c = resultRGB * 0.1;
        float x = a * 4.0;
        if (x > 2.5)      c.r = 1.0;    // unconverged: safe fallback bound
        else if (x > 1.5) c.g = 0.5;    // supporting-only bound
        else if (x > 0.5) c.g = 1.0;    // exact certificate fired
        c.b = b;
        return c;
    }

    // 12: dejittered residual (jitter-cancel verification).
    case 12:
        return float3(a, a, 0.0);

    // No payload (the history-support early-out, or a mode whose data was
    // never produced for this pixel): show the resolved output.
    default:
        return resultRGB;
    }
}

#endif // TAA_DEBUG_H_HLSL