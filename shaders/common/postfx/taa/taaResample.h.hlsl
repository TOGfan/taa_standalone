// ============================================================================
// TAA history color resampling: the Wiener suite, plus the fallback FXAA
// ----------------------------------------------------------------------------
// Two tiers, both per-phase closed-form Wiener (MMSE) solves for a flat
// in-band PSD on [-pi/2, pi/2]: the least-squares fit of e^{i w f} by the
// tap exponentials { e^{i w j} }. The normal equations have a Toeplitz sinc
// Gram with LITERAL entries (closed forms in W below) and a right-hand side
// of band-limited autocorrelations at fractional lags, g(x) = sin(W x)/x.
// Everything is trigonometric in f: no LUTs, no offline steps, and (via the
// quarter-band collapse) no transcendentals.
//
//   9-TAP  (4 logical taps/axis, 3 fused fetches): the exact 4-DOF solve.
//     Even block: LS, DC pinned. Odd block: complete and unconstrained.
//   21-TAP (6 logical taps/axis, 5 fused fetches, corners skipped): the
//     SHARP tier. Even block: DC and band edge BOTH pinned (a - b - c =
//     C + S makes A(pi/2) exact at every phase), one LS degree of freedom
//     left. Odd block: edge-pinned. Plus the corner redistribution.
//
// WHY THE SPLIT: the edge pin's price is set by the slack left after DC.
// The 4-tap even block has 2 DOF; DC + edge consumes both, and the residual
// piles up as a +4.4% mid-band hump (measured; reads as oversharpening).
// The 6-tap has 3; the residual spreads to +0.6% and the pin is free
// (phase-avg MSE statistically indistinguishable from the unpinned solve).
//
// THE 9-TAP EVEN BLOCK IS ONE CURVE: 2 DOF, DC is one linear constraint,
// and the edge response E(f) parameterizes the entire DC-exact family:
//     a = (1 + sqrt(2) E) / 2,    b = (1 - sqrt(2) E) / 2.
// Every 4-tap DC-exact even design is a point on this line; the band MSE
// is a parabola in E whose minimum E_c(f) is the constrained-LS solve.
// The invSum rescale sits at E_r(f) = E_LS(f)/d(f) > E_c(f) at every
// interior phase, and is therefore the EXACT optimum of
//     "MMSE subject to DC exact and band edge >= E_r(f)".
// It is not an approximation of the constrained solve; it is the solution
// of a sharpness-floored criterion. The unconstrained-DC optimum (1.5x
// lower band MSE, worst-phase edge 0.953 vs 0.973) is two mads away:
// TAA_WIENER4_DC_EXACT below.
//
// THE QUARTER-BAND COLLAPSE: at W = pi/2 the literals sin W = 1, cos W = 0,
// sin 2W = 0, cos 2W = -1, sin 3W = -1, cos 3W = 0 are EXACT, and the six
// fractional lags collapse to S and C over linear denominators:
//     g(f)   = S/f        g(f-1) =  C/(1-f)    g(f+1) =  C/(1+f)
//     g(f-2) = S/(2-f)    g(f+2) = -S/(2+f)    g(f-3) = -C/(3-f)
// with S = sin(pi f / 2), C = cos(pi f / 2). The phase is MIRRORED to
// f in [0, 1/2] (the LS problem reflects about the tap-set center: swap
// the outer weights, t -> 1-t), which keeps every denominator >= 1/2 and
// restricts the series argument to x = (pi/2) f in [0, pi/4] -- where
// 5-term series for sin(x)/x and cos(x) replace every transcendental.
// sin(x)/x is formed DIRECTLY from the series bracket, so the removable
// singularity of S/f at f = 0 never exists as a division.
//
// CORNER REDISTRIBUTION (21-tap): the four skipped fetches carry a rank-1
// block (w0 far-pair partial responses, outer product). The executed
// 21-fetch kernel with the sigma correction below is the literal L2
// projection of the designed 6x6 kernel onto the fetched span: the error
// is exactly -sigmaX (x) sigmaY, sigma[far] = w0, the three near rows of
// G*sigma vanish (G = fetch Gram; entries are 2*sin(nW)/n: g(0)=pi,
// g(+-1)=2, g(+-2)=0, g(+-3)=-2/3, g(+-4)=0, g(+-5)=2/5), and the merged
// position enters only through u = 1-2t. The loop's sumW becomes
// 1 - Sx*Sy (S = sigma(0) ~ 0.63*w0) and restores DC.
//   Measured (worst phase, both axes, sharp tier): RMS response error
//   2.9e-3 -> 3.2e-4; max 4.5e-3 (edge/DC ridge) -> 1.3e-3 (band corner
//   only -- removable solely by fetching the corners). The execution
//   residual now sits ~10x below the design's own irreducible residual.
//
// MEASURED (worst phase f = 1/2, band [0, pi/2], DC-exact, real taps;
// hand-solved -- re-derive from the formulas here before shipping claims):
//   9-tap:  +2.9% / -2.7%  (Wiener beta=3.2: +2.7 / -3.1; wall +-2.8)
//           phase-avg band MSE 5.6e-4 (floor 2.8e-4; DC_EXACT: 3.8e-4,
//           worst-phase edge -4.7% vs -2.7%)
//   21-tap: +0.6% max in-band, band edge exactly 1.0 at every phase,
//           phase-avg band MSE ~1.9e-5 (floor ~1.5e-5)
//   complex error at (f=1/4, w=pi/2): 2.0% (9-tap) vs 2.8% (old Wiener);
//   phase error 0.44 deg; centroid error <= 0.002 texel.
//
// INVARIANTS:
//   * Integer phases return the identity stencil to within one rounding
//     (at f = 0: alpha = -1, beta = 0, b = 0 exactly; a = 1 +- 1 ulp).
//   * The merged center pair is same-signed across the phase range
//     (|alpha| <= 1 <= a, equality only at the integer endpoint), so the
//     fused bilinear fetch is a true merge and t stays in [0, 1].
//   * Negative-lobe mass -- the clamp-inert threshold for
//     taaHistoryOvershoot -- peaks at ~0.19 (9-tap) and ~0.26 (21-tap).
//
// SERIES PRECISION: sin(x)/x through x^8 (dropped term x^10/11! < 2.3e-9
// at x = pi/4); cos(x) through x^8 (dropped term x^10/10! < 2.5e-8).
// Alternating series with monotonically decreasing terms: the error is
// bounded by the first dropped term. Gram conditioning: 9-tap even ~4,
// 9-tap odd ~70, 21-tap even (pinned form) denominator 1.008, 21-tap odd
// (pinned form) denominator 0.475, corner 3x3 ~15.
//
// RE-DERIVATION for another band W: the Gram entries are
//     E_jk = sin((j-k)W)/(j-k) + sin((j+k+1)W)/(j+k+1)
//     O_jk = sin((j-k)W)/(j-k) - sin((j+k+1)W)/(j+k+1)
// (diagonals W +- sin((2j+1)W)/(2j+1)); the six lags re-expand with the
// sin W, cos W, sin 2W, cos 2W, sin 3W literals reappearing, and the
// series domain becomes [0, W/2] (add terms if W > pi/2). W = pi/2 is
// what makes the literals 0/+-1 and the domain [0, pi/4]; it is also the
// band the TAA loop can sustain.
//
// FRAGMENT HEADER: compiled only inside taa.fx.hlsl. Requires host context:
// samplers (historyTex, sceneTex), cbuffer perDraw (taaHistoryOvershoot),
// postFx macros (tex2Dlod). Requires fragments included before:
// taaShared.h.hlsl, taaConstants.h.hlsl, taaColor.h.hlsl (ToSpace).
// ============================================================================
#ifndef TAA_RESAMPLE_H_HLSL
#define TAA_RESAMPLE_H_HLSL

static const float kPi     = 3.1415926535897932;   // 2W: fetch-Gram diagonal
static const float kHalfPi = 1.5707963267948966;

// 9-tap even-block DC pin:
//   0 = invSum rescale (default): exact optimum of "MMSE s.t. DC exact and
//       band edge >= E_r(f)" -- near-minimax, worst-phase edge ~0.973.
//   1 = exact DC-constrained LS: 1.5x lower band MSE (phase-avg 3.8e-4 vs
//       5.6e-4), worst-phase edge ~0.953. The soft direction.
#define TAA_WIENER4_DC_EXACT 0

// S = sin(pi/2 * f), C = cos(pi/2 * f), SoverF = sin(pi/2 * f)/f, for
// f in [0, 1/2] (the callers mirror). Series, not transcendentals; the
// bracket IS SoverF/(pi/2), so the f = 0 singularity never divides.
void WienerQuarterBand(float f, out float S, out float C, out float SoverF)
{
    float x  = f * kHalfPi;   // [0, pi/4]
    float x2 = x * x;

    // sin(x)/x through x^8: 1 - x^2/3! + x^4/5! - x^6/7! + x^8/9!.
    float sinc = 1.0 - x2 * (1.0 / 6.0
                   - x2 * (1.0 / 120.0
                   - x2 * (1.0 / 5040.0
                   - x2 * (1.0 / 362880.0))));
    // cos(x) through x^8: 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!.
    float cosx = 1.0 - x2 * (1.0 / 2.0
                   - x2 * (1.0 / 24.0
                   - x2 * (1.0 / 720.0
                   - x2 * (1.0 / 40320.0))));

    S      = x * sinc;
    C      = cosx;
    SoverF = kHalfPi * sinc;
}

// The odd (phase-carrying) half of the STANDARD Wiener solve (9-tap tier):
// the unconstrained LS fit of sin(w*(f-1/2)) by { sin(w/2), sin(3w/2) }.
// Gram: O00 = W-1, O11 = W+1/3, O01 = 1 (condition ~70). alpha = -1,
// beta = 0 EXACTLY at f = 0.
void WienerOddSolve(float S, float C, float Q, float f,
                    out float alpha, out float beta)
{
    float S0 = C / (1.0 - f) - Q;             // g(f-1) - g(f)
    float S1 = S / (2.0 - f) - C / (1.0 + f); // g(f-2) - g(f+1)

    float O00   = kHalfPi - 1.0;
    float O11   = kHalfPi + (1.0 / 3.0);
    float invD  = 1.0 / (O00 * O11 - 1.0);

    alpha = (O11 * S0 - S1) * invD;   // c1 - c0
    beta  = (O00 * S1 - S0) * invD;   // c2 - c(-1)
}

// The odd half of the SHARP (21-tap) tier: one LS degree of freedom,
// subject to B(pi/2) = sin(pi/2 * (f-1/2)) exactly -- alpha + beta = S - C
// (both basis sines evaluate to sqrt(2)/2 at the band edge).
void WienerOddSolveEdgePinned(float S, float C, float Q, float f,
                              out float alpha, out float beta)
{
    float S0 = C / (1.0 - f) - Q;
    float S1 = S / (2.0 - f) - C / (1.0 + f);

    float den = 2.0 * kHalfPi - (8.0 / 3.0);  // 0.47493
    float r11 = kHalfPi - (2.0 / 3.0);        // 0.90413
    alpha = ((S0 - S1) + (S - C) * r11) / den;
    beta  = (S - C) - alpha;
}

// Clamp a resampled color to the min/max of the taps it was built from, with a
// small overshoot margin. Shared by both history filters.
// NOTE: the Wiener kernels' negative-lobe mass (the provably inert margin)
// peaks at ~0.19 (9-tap) and ~0.26 (21-tap) at mid-phase.
float3 ClampToTapFootprint(float3 color, float3 tapMin, float3 tapMax)
{
    float3 tapRange  = max(tapMax - tapMin, kMinFootprintRange);
    float3 overshoot = taaHistoryOvershoot * tapRange;
    return clamp(color, tapMin - overshoot, tapMax + overshoot);
}

// Wiener-4 fused weights (MMSE tier): the exact 4-DOF solve, DC-pinned.
// Outputs: w = (tap base-1, merged pair, tap base+2), all DC-exact.
void GetWiener4FusedWeights(float f, float baseCoord, out float3 uv, out float3 w)
{
    // Mirror to [0, 1/2]: the LS problem reflects about the tap-set center.
    bool mirror = (f > 0.5);
    f = mirror ? (1.0 - f) : f;

    float S, C, Q;
    WienerQuarterBand(f, S, C, Q);

    // Even RHS: g(f-1) + g(f), g(f-2) + g(f+1).
    float R0 = C / (1.0 - f) + Q;
    float R1 = S / (2.0 - f) + C / (1.0 + f);

    // Even solve: LS fit of cos(w(f-1/2)) by { cos(w/2), cos(3w/2) }.
    // Gram: E00 = W + sin W, E11 = W + sin(3W)/3, E01 = sin W + sin(2W)/2.
    float E00  = kHalfPi + 1.0;
    float E11  = kHalfPi - (1.0 / 3.0);
    float invD = 1.0 / (E00 * E11 - 1.0);
    float a    = (E11 * R0 - R1) * invD;   // c0 + c1
    float b    = (E00 * R1 - R0) * invD;   // c(-1) + c2

#if TAA_WIENER4_DC_EXACT
    // Exact DC-constrained LS: Gram-metric projection onto a + b = 1
    // (E^-1 applied to the DC vector, normalized). 0.13132 / 0.86868.
    float dC   = 1.0 - (a + b);
    float denC = 2.0 * kHalfPi - (4.0 / 3.0);          // pi - 4/3
    a += dC * (kHalfPi - (4.0 / 3.0)) / denC;
    b += dC * (kHalfPi)               / denC;
#endif

    float alpha, beta;
    WienerOddSolve(S, C, Q, f, alpha, beta);

    float wB = (b - beta) * 0.5;          // tap base-1
    float wC = (b + beta) * 0.5;          // tap base+2
    float t  = 0.5 + 0.5 * (alpha / a);   // c1 / (c0 + c1)

    if (mirror)
    {
        float tmp = wB; wB = wC; wC = tmp;   // base-1 <-> base+2
        t = 1.0 - t;
    }

    // DC pin: the rescale path's load-bearing step; in the DC_EXACT path
    // a + b is structurally 1 and this is one-rounding insurance.
    float invDC = 1.0 / (a + b);
    w  = float3(wB, a, wC) * invDC;
    uv = float3(baseCoord - 1.0, baseCoord + t, baseCoord + 2.0);
}

float3 SampleHistoryColor_Wiener4_9Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uvX, weightX;
    float3 uvY, weightY;
    GetWiener4FusedWeights(fracOffset.x, baseTexel.x, uvX, weightX);
    GetWiener4FusedWeights(fracOffset.y, baseTexel.y, uvY, weightY);

    float3 tcX = clamp(uvX * vp.texelSize.x, minUV.x, maxUV.x);
    float3 tcY = clamp(uvY * vp.texelSize.y, minUV.y, maxUV.y);

    float3 color  = 0.0;
    float  sumW   = 0.0;
    float3 tapMin = float3( kLargeValue,  kLargeValue,  kLargeValue);
    float3 tapMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    [unroll]
    for (int y = 0; y < 3; ++y)
    {
        float wy = weightY[y];
        float vy = tcY[y];

        [unroll]
        for (int x = 0; x < 3; ++x)
        {
            float tc = tcX[x];
            float w  = weightX[x] * wy;

            float3 tap = max(tex2Dlod(historyTex, float4(tc, vy, 0.0, 0.0)).rgb, 0.0);
            color += tap * w;
            sumW  += w;

            tapMin = min(tapMin, tap);
            tapMax = max(tapMax, tap);
        }
    }

    color /= max(sumW, 1e-4);
    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

// Wiener-6 fused weights (sharp tier), plus the corner-redistribution
// vector sigma: the executed 21-fetch kernel is the L2 projection of the
// designed 6x6 kernel onto the fetched span (header). The far pair is
// purely even (gamma = 0: the third odd DOF is dropped -- its LS weight
// is bounded by ~4e-3 over the phase range, it costs <= 0.1% response at
// the extreme phases where total error is minimal, and keeping it pushes
// the odd solve's condition number from ~70 to ~1700 in fp32).
void GetWiener6FusedWeights(float frac, float baseCoord,
                            out float3 posPt, out float2 posBi,
                            out float w[5], out float sigma[5])
{
    float f = frac;
    bool mirror = (f > 0.5);
    f = mirror ? (1.0 - f) : f;

    float S, C, Q;
    WienerQuarterBand(f, S, C, Q);

    // Even RHS: g(f-2)+g(f+1), g(f-3)+g(f+2). (g(f-1)+g(f) is not needed:
    // the DC and edge pins replace it.)
    float R1 =  S / (2.0 - f) + C / (1.0 + f);
    float R2 = -(C / (3.0 - f) + S / (2.0 + f));

    // Even solve, DC + edge pinned: a + b + c = 1, a - b - c = C + S,
    // one LS degree of freedom (the b/c split), eliminated in closed form:
    //   b * (E11 - 2*E12 + E22) = (R1 - R2) - a*(E01 - E02) - bc*(E12 - E22)
    // with, at W = pi/2: E01 - E02 = 4/3, E12 - E22 = 0.8 - pi/2,
    // E11 - 2*E12 + E22 = pi - 32/15.
    float d  = C + S;
    float a6 = 0.5 * (1.0 + d);                      // c0 + c1
    float bc = 0.5 * (1.0 - d);                      // b + c
    float b6 = (R1 - R2 - a6 * (4.0 / 3.0)
                + bc * (kHalfPi - 0.8)) / (2.0 * kHalfPi - (32.0 / 15.0));
    float c6 = bc - b6;

    float alpha, beta;
    WienerOddSolveEdgePinned(S, C, Q, f, alpha, beta);

    float wB = (b6 - beta) * 0.5;              // tap base-1
    float wC = (b6 + beta) * 0.5;              // tap base+2
    float t  = 0.5 + 0.5 * (alpha / a6);       // c1 / (c0 + c1)

    if (mirror)
    {
        float tmp = wB; wB = wC; wC = tmp;     // base-1 <-> base+2
        t = 1.0 - t;                           // w[0]/w[4] equal: no swap
    }

    // DC is structural (a6 + b6 + c6 = 1); kept as one-rounding insurance.
    float invDC = 1.0 / (a6 + b6 + c6);
    w[0] = w[4] = c6 * 0.5 * invDC;
    w[1] = wB * invDC;
    w[2] = a6 * invDC;
    w[3] = wC * invDC;

    // --- corner redistribution -------------------------------------------
    // sigma[far] = w[0]; the three near rows of G*sigma vanish. The merged
    // position enters only through u = 1 - 2t (post-mirror t). Denominator
    // 1.7627 + 0.0457u^2: bounded well away from zero on u in [0,1].
    {
        float w0 = w[0];
        float u  = 1.0 - 2.0 * t;
        float aG = kPi - (2.0 / 3.0);
        float bG = kPi + (2.0 / 3.0);
        float d2 = (kHalfPi + 1.0) + (kHalfPi - 1.0) * u * u;
        float x2 = (2.0 / 3.0 + 4.0 / aG)
                 / (d2 - 2.0 / aG - 2.0 * u * u / bG);
        float sm = (-4.0 - 2.0 * x2) / aG;   // x1 + x3
        float df = -2.0 * u * x2 / bG;       // x1 - x3
        sigma[0] = sigma[4] = w0;
        sigma[1] = w0 * (sm + df) * 0.5;
        sigma[2] = w0 * x2;
        sigma[3] = w0 * (sm - df) * 0.5;
    }

    posPt.x = baseCoord - 2.0;
    posPt.y = baseCoord - 1.0;
    posPt.z = baseCoord + 2.0;
    posBi.x = baseCoord + t;
    posBi.y = baseCoord + 3.0;
}

float3 SampleHistoryColor_Wiener6_21Tap(
    float2 historyUV, ViewportParams vp, float2 minUV, float2 maxUV)
{
    float2 pixelPos   = historyUV * vp.sizePixels;
    float2 baseTexel  = floor(pixelPos - 0.5) + 0.5;
    float2 fracOffset = pixelPos - baseTexel;

    float3 uX_pt, uY_pt;
    float2 uX_bi, uY_bi;
    float wX[5], wY[5];
    float sigX[5], sigY[5];

    GetWiener6FusedWeights(fracOffset.x, baseTexel.x, uX_pt, uX_bi, wX, sigX);
    GetWiener6FusedWeights(fracOffset.y, baseTexel.y, uY_pt, uY_bi, wY, sigY);

    float coordsX[5] = { uX_pt.x, uX_pt.y, uX_bi.x, uX_pt.z, uX_bi.y };
    float coordsY[5] = { uY_pt.x, uY_pt.y, uY_bi.x, uY_pt.z, uY_bi.y };

    float tcX[5], tcY[5];
    [unroll]
    for (int c = 0; c < 5; ++c)
    {
        tcX[c] = clamp(coordsX[c] * vp.texelSize.x, minUV.x, maxUV.x);
        tcY[c] = clamp(coordsY[c] * vp.texelSize.y, minUV.y, maxUV.y);
    }

    float3 color  = 0.0;
    float  sumW   = 0.0;
    float3 tapMin = float3( kLargeValue,  kLargeValue,  kLargeValue);
    float3 tapMax = float3(-kLargeValue, -kLargeValue, -kLargeValue);

    [unroll]
    for (int y = 0; y < 5; ++y)
    {
        float vy = tcY[y];
        float wy = wY[y];
        float sy = sigY[y];

        [unroll]
        for (int x = 0; x < 5; ++x)
        {
            // Corners skipped (fetch budget); the rank-1 redistribution
            // replaces their block; sumW (now 1 - Sx*Sy) restores DC.
            if ((x == 0 || x == 4) && (y == 0 || y == 4))
                continue;

            float vx = tcX[x];
            float w  = wX[x] * wy - sigX[x] * sy;

            float3 tap = max(tex2Dlod(historyTex, float4(vx, vy, 0.0, 0.0)).rgb, 0.0);
            color += tap * w;
            sumW  += w;

            tapMin = min(tapMin, tap);
            tapMax = max(tapMax, tap);
        }
    }

    color /= max(sumW, 1e-4);
    color = ClampToTapFootprint(color, tapMin, tapMax);
    return ToSpace(max(color, 0.0));
}

// ============================================================================
// FALLBACK FXAA (strict viewport-clamped; center + preloaded corner taps)
// ============================================================================
// cornersRGB: [0]=NW(-1,-1), [1]=NE(+1,-1), [2]=SW(-1,+1), [3]=SE(+1,+1)
float3 ApplyFXAA(float2 centerUV, float2 texelSize, float3 centerRGB, float3 cornersRGB[4], float2 minUV, float2 maxUV)
{
    float lumaNW = LumaRGB(cornersRGB[0]);
    float lumaNE = LumaRGB(cornersRGB[1]);
    float lumaSW = LumaRGB(cornersRGB[2]);
    float lumaSE = LumaRGB(cornersRGB[3]);
    float lumaM  = LumaRGB(centerRGB);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * kFXAAReduceMul), kFXAAReduceMin);
    float rcpDirMin = 1.0 / (min(abs(lumaMax - lumaMin), max(lumaMax, 1.0)) + dirReduce);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    dir = clamp(dir * rcpDirMin, float2(-kFXAAMaxDir, -kFXAAMaxDir), float2(kFXAAMaxDir, kFXAAMaxDir)) * texelSize;

    float2 uv0 = clamp(centerUV + dir * (1.0 / 3.0 - 0.5), minUV, maxUV);
    float2 uv1 = clamp(centerUV + dir * (2.0 / 3.0 - 0.5), minUV, maxUV);
    float3 rgbA = 0.5 * (
        ScrubRawRGB(tex2Dlod(sceneTex, float4(uv0, 0.0, 0.0)).rgb) +
        ScrubRawRGB(tex2Dlod(sceneTex, float4(uv1, 0.0, 0.0)).rgb));

    float2 uv2 = clamp(centerUV + dir * (0.0 / 3.0 - 0.5), minUV, maxUV);
    float2 uv3 = clamp(centerUV + dir * (3.0 / 3.0 - 0.5), minUV, maxUV);
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        ScrubRawRGB(tex2Dlod(sceneTex, float4(uv2, 0.0, 0.0)).rgb) +
        ScrubRawRGB(tex2Dlod(sceneTex, float4(uv3, 0.0, 0.0)).rgb));

    float lumaB = LumaRGB(rgbB);
    return ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
}

float ComputeFXAAFilterWeight(float currentBlendWeight)
{
    float weight = saturate((currentBlendWeight - kFXAABlendKnee) / max(1.0 - kFXAABlendKnee, 0.05));
    return saturate(weight * kFXAABlendSharpness);
}

#endif // TAA_RESAMPLE_H_HLSL