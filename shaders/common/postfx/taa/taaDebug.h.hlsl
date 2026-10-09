// ============================================================================
// TAA debug view rendering (included by taaFinal.fx.hlsl)
// ----------------------------------------------------------------------------
// The resolve NEVER writes debug colors: its RGB is always the real blended
// color (the history-copy child stores it into #TAA_History unconditionally)
// and the view state rides #TAA_Result.a instead, packed by PackDebugAlpha
// (taaShared.h.hlsl -- which also embeds the temporal clip-state sigma in
// the payload's low 7 bits, so the record chain survives debugging). This
// file turns that payload back into the on-screen view, so debug colors
// exist ONLY in $backBuffer and the temporal accumulation continues
// undisturbed while debugging.
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
//     (10-bit quantization sits far below the rounding threshold). Payload
//     B is 6 bits -- display-only in every view; the 0.125-quantized bit
//     fields decode exactly through it.
//
// v3.9: mode 13 -- the gate's REALIZED NULL LAW (the audit's calibration
// view). A = saturate(mdd/16) as a grayscale ramp: park on a STATIC scene
// and read the brightness distribution; the nominal 95th percentile sits at
// chi^2/16 (~0.49 mid-gray at the default chi 2.8). Brightness beyond that
// on static content is the gate running hot, and the EMPIRICAL Student
// factor is (A's 95th percentile) * 16 / chi^2. B marks engagement
// (mdd > chi^2) with a blue tint.
//
// v3.10: mode 11's B channel, in clipGhostReset=2 telemetry mode on flats,
// is the persistence statistic |T|/alarm -- dim and stable on static scenes
// (|T| < ~1), blooming along a real trail then decaying as the unlocked
// corrector evicts it.
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

    // 11: clip gate state. A: the record's SURVIVING VARIANCE FRACTION when
    // carried (1.0 = clean record, full room; dimming toward 0.55 = the
    // bias is latching, the gate is entering bias-removal mode -- the
    // ghost-kill state), 0.35 = record RESET this frame (identity reset,
    // shock, corroborated geometry, or the NaN guard), 0.15 = no record at
    // all. B = how far the gate shrank the history toward the neighborhood
    // mean, post-soft-clip (0 = untouched); in clipGhostReset=2 telemetry
    // mode, B = the fitted coverage on engaged steps and the persistence
    // statistic |T|/alarm on flats (dim and stable on static scenes;
    // blooming along a real trail, then decaying as the corrector evicts
    // it -- THE detector verification view). The reveal repair reads as
    // either a one-frame reset flash at the departing edge or the green
    // dimming over ~4 frames with B bright while the gate dismantles the
    // remnant; a persistent dim-green + bright-B region means a ghost the
    // latch is still dismantling (watch it clear in 2-3 frames).
    case 11: {
        float3 c = resultRGB * 0.1;
        if (a > 0.45)      c.g = saturate((a - 0.55) * 2.2);  // carried: brightness = surviving variance fraction
        else if (a > 0.25) c.g = 0.5;                         // reset this frame (A = 0.35)
        else               c.r = 0.5;                         // no record (A = 0.15)
        c.b = b;                         // applied shrink fraction / detector telemetry
        return c;
    }

    // 12: dejittered residual (jitter-cancel verification).
    case 12:
        return float3(a, a, 0.0);

    // 13: the gate's realized null law (v3.9, the audit's calibration view).
    // A = saturate(mdd/16) as the histogram channel (grayscale); the nominal
    // 95th percentile sits at chi^2/16 -- mid-gray at the default chi 2.8.
    // Park on a STATIC scene: brightness beyond the nominal point is the
    // gate running hot, and the empirical Student factor is (A's 95th
    // percentile) * 16 / chi^2. B = engagement (mdd > chi^2), blue tint.
    case 13:
        return float3(a, a, b * 0.9);

    // No payload (the history-support early-out, or a mode whose data was
    // never produced for this pixel): show the resolved output.
    default:
        return resultRGB;
    }
}

#endif // TAA_DEBUG_H_HLSL