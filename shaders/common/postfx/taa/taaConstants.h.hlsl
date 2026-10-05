// ============================================================================
// TAA resolve-pass constants
// ----------------------------------------------------------------------------
// Pure: no cbuffer, sampler or function dependencies. Centralized so every
// fragment sees the same tuning values and the knob inventory is one place.
//
// FRAGMENT HEADER: not a standalone shader. Compiled only inside taa.fx.hlsl
// (included there right after taaShared.h.hlsl).
// ============================================================================
#ifndef TAA_CONSTANTS_H_HLSL
#define TAA_CONSTANTS_H_HLSL

static const float kLargeValue               = 32000.0;
static const float kSqrt2                    = 1.41421356;
static const float kLog2E                    = 1.44269504;
static const float kMinMotionDirLengthPx     = 0.1;     // below this motion is "static"
static const float kMinMotionBlendDropSpeed  = 0.1;
static const float kMotionFullStrengthPx     = 8.0;     // motion that fully drops history feedback

static const float kMinSigma                 = 0.001;
static const float kMinSpatialContrast       = 0.001;
static const float kMinFootprintRange        = 1e-4;
static const float kFireflyClampEpsilon      = 0.001;
static const float kFlickerPadThreshold      = 0.001;   // jitter anti-flicker padding engages above this

// Fallback FXAA
static const float kFXAAReduceMul            = 1.0 / 128.0;
static const float kFXAAReduceMin            = 1.0 / 128.0;
static const float kFXAAMaxDir               = 8.0;
static const float kFXAABlendKnee            = 0.8;
static const float kFXAABlendSharpness       = 2.0;

// Luma drift correction
static const float kLumaDriftLumaFloor       = 0.05;
static const float kLumaDriftRelThreshold    = 0.30;
static const float kLumaDriftAbsThreshold    = 0.03;
static const float kLumaDriftChromaFadeWidth = 2.0;

// Current-frame pursuit confirmation tolerance
static const float kPursuitVelBaseTolerancePx = 0.30;   // px: base divergence tolerance
static const float kMinVelCoherenceRadiusPx   = 0.25;   // floor for the velocity-coherence radius

// Smooth-field (continuous parallax) advection handling: the advection error
// is PREDICTED and subtracted (first-order Jacobian correction), the residual
// budget covers second-order terms only.
static const float kVelDiscontinuityRatio    = 0.65;    // curvature/pair-gradient ratio that still counts as continuous
static const float kVelDiscontinuityAbsPx    = 0.25;    // px: velocity-curvature noise floor
static const float kVelJacobianResidualFrac  = 0.25;    // fraction of |J|*miss kept for 2nd-order terms
static const float kPursuitSnapPadPx         = 1.5;     // px: fixed part of the residual pad
static const float kPursuitMaxAdvectionPx    = 12.0;    // px: hard clamp on the RESIDUAL budget only

// Layer-aware velocity selection (depth-invisible boundaries).
static const float kQuadStepGradMul          = 4.0;     // quad step must exceed coherentGrad * this to commit
static const float kMissCapHeadroom          = 1.5;     // same-surface miss cap headroom (Jacobian underestimate insurance)

// Velocity quantization fallback floor for the extrapolation gates.
static const float kVelQuantFloorPx          = 0.0625;

// Dilation-revocation gate: landing within this fraction of the texel center
// counts as ON the center.
static const float kCenterLandingFracPx      = 0.1;

// Depth transport (see EstimateLayerForwardParallax): the T_y slope fit needs
// this much positional spread (px^2) to be informative, and when T_y is
// unmeasurable we assume |T_y|/w below this fraction.
static const float kMinParallaxInfoPx2       = 2.0;
static const float kTyUnmeasuredFrac         = 0.02;

// Exact-hull simplex clipper budget: unrolled simplex pivots, plus one final
// certificate pass on the last basis. Reaching the TRUE exit facet of a
// 9-point hull from the initial basis takes a handful of Dantzig pivots
// (each strictly raises the plane); 5 covers the generic case, and the
// noise-scaled certificate below means near-facet planes certify.
static const int   kHullSimplexPivots  = 5;
// Certificate tolerance floor, in exit-HEIGHT units; scaled at runtime by
// the neighborhood's z range (the data's own noise scale). A tap this far
// above the plane does not force another pivot.
static const float kHullSimplexCertEps = 1e-4;

// Sign-bit transport of the revocation flag through the alpha channel:
// negative = revoked.
static const float kRevokedAlphaEpsilon      = 1e-30;

// Debug views
static const float kDebugVelocityScale       = 0.1;
static const float kDebugLinearDepthRange    = 100.0;

static const float kStdWeights[9] = { 1.0, 0.36787944, 0.36787944, 0.36787944, 0.36787944, 0.13533528, 0.13533528, 0.13533528, 0.13533528 };
static const float kInvLength[9]  = { 0.0, 1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678 };

// Kaiser per-tier betas (see taaResample.h.hlsl).
static const float kKaiserBeta4Tap           = 3.2;
static const float kKaiserBeta6Tap           = 5.2;

#endif // TAA_CONSTANTS_H_HLSL