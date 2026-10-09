<template>
  <div class="taa-bng-options">
    <!-- Master Header -->
    <div class="options-header">
      <div class="header-title">Temporal Anti-Aliasing</div>
      <div class="header-toggle">
        <BngSwitch v-model="isActive" @update:modelValue="toggleTaa">
          Enable TAA
        </BngSwitch>
      </div>
    </div>

    <!-- Presets Bar -->
    <div class="options-presets">
      <BngButton class="preset-btn" :accent="ACCENTS.outlined" @click="applyPreset(presets.Performance)">Performance</BngButton>
      <BngButton class="preset-btn" :accent="ACCENTS.outlined" @click="applyPreset(presets.Balanced)">Balanced</BngButton>
      <BngButton class="preset-btn" :accent="ACCENTS.outlined" @click="applyPreset(presets.Smooth)">Smooth</BngButton>
      <BngButton class="preset-btn" :accent="ACCENTS.outlined" @click="applyPreset(presets.Clarity)">Clarity</BngButton>
    </div>

    <!-- Main Scrollable List -->
    <div class="options-list-scroll" :class="{ 'is-disabled': !isActive }">
      <details 
        v-for="(category, catIndex) in settingsSchema" 
        :key="category.name"
        class="category-block"
        :open="catIndex === 0"
      >
        <summary class="category-title">{{ category.name }}</summary>

        <div class="category-items">
          <div 
            v-for="item in category.items" 
            :key="item.id" 
            class="options-item-row"
            @mouseenter="hoveredItem = item"
            @mouseleave="hoveredItem = null"
          >
            <!-- Top Half: Label & Switches -->
            <div class="row-header">
              <div class="option-label">{{ displayItem(item).name }}</div>

              <div v-if="item.type === 'bool' || item.type === 'numBool'" class="inline-controls">
                <BngSwitch 
                  :modelValue="config[item.id] === 1 || config[item.id] === true"
                  @update:modelValue="val => onSwitchChange(item, val)"
                  :disabled="!isActive"
                />
                <div class="option-reset">
                  <BngButton 
                    :icon="icons.undo"
                    :accent="ACCENTS.outlined"
                    @click="resetSetting(item)"
                    class="bng-reset-btn"
                    :style="{ visibility: config[item.id] !== item.default ? 'visible' : 'hidden' }"
                    title="Reset to default"
                  />
                </div>
              </div>
            </div>

            <!-- Bottom Half: Native BngSlider -->
            <div v-if="item.type === 'float'" class="stacked-controls">
              <BngSlider
                class="native-slider"
                :min="item.min"
                :max="item.max"
                :step="item.step || 0.01"
                :modelValue="sliderModel(item)"
                @update:modelValue="val => onSliderChange(item, val)"
                :with-input="true"
                :disabled="!isActive"
              />
              <div class="option-reset">
                <BngButton 
                  :icon="icons.undo"
                  :accent="ACCENTS.outlined"
                  @click="resetSetting(item)"
                  class="bng-reset-btn"
                  :style="{ visibility: sliderModified(item) ? 'visible' : 'hidden' }"
                  title="Reset to default"
                />
              </div>
            </div>

            <!-- Bottom Half: Custom Dropdown Select -->
            <div v-if="item.type === 'select'" class="stacked-controls select-control">
              <div v-if="openDropdown === item.id" class="dropdown-backdrop" @click="openDropdown = null"></div>

              <div class="custom-dropdown-container" :class="{ 'is-open': openDropdown === item.id, 'is-disabled': !isActive }">
                <div class="dropdown-selected" @click="toggleDropdown(item.id)">
                  <span>{{ getOptionLabel(item, config[item.id]) }}</span>
                  <span class="dropdown-arrow">▼</span>
                </div>
                
                <div class="dropdown-list" v-if="openDropdown === item.id">
                  <div
                    v-for="opt in item.options"
                    :key="opt.value"
                    class="dropdown-option"
                    :class="{ 'is-active': config[item.id] === opt.value }"
                    @click="selectOption(item, opt.value)"
                  >
                    {{ opt.label }}
                  </div>
                </div>
              </div>

              <div class="option-reset">
                <BngButton 
                  :icon="icons.undo"
                  :accent="ACCENTS.outlined"
                  @click="resetSetting(item)"
                  class="bng-reset-btn"
                  :style="{ visibility: config[item.id] !== item.default ? 'visible' : 'hidden' }"
                  title="Reset to default"
                />
              </div>
            </div>

          </div>
        </div>
      </details>
    </div>

    <!-- Dynamic Info Panel -->
    <div class="options-info-panel">
      <template v-if="hoveredItem">
        <div class="info-header">
          <span class="info-title">{{ displayItem(hoveredItem).name }}</span>
          <span class="info-default">Default: {{ formatDefault(displayItem(hoveredItem)) }}</span>
        </div>
        <div class="info-desc">{{ displayItem(hoveredItem).desc }}</div>
      </template>
      <div v-else class="info-empty">
        Hover over a setting to see details.
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue"
import { BngButton, BngSwitch, BngSlider, icons, ACCENTS } from "@/common/components/base"

const isActive = ref(false)
const config = ref({})
const hoveredItem = ref(null)
const openDropdown = ref(null)

// ---- clip-coverage slider math -------------------------------------------
// The clipping slider operates in ACCEPTED-HISTORY PERCENTAGE space; the
// stored setting (varianceGamma) stays the chi radius so the shader
// contract, the settings file and the Lua Student-fit plumbing are
// untouched. acceptance = CDF_chi2_3(chiEff^2), chiEff = chi * (1 + clipOvershoot).
function erf(x) { // Abramowitz & Stegun 7.1.26 (|eps| <= 1.5e-7)
  const sign = x < 0 ? -1 : 1
  x = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * x)
  const poly = 0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
  return sign * (1 - poly * t * Math.exp(-x * x))
}
function chiSq3Cdf(x) {
  if (x <= 0) return 0
  return erf(Math.sqrt(x / 2)) - Math.sqrt((2 * x) / Math.PI) * Math.exp(-x / 2)
}
function chiToCoverage(chi, overshoot) {
  const eff = Math.max(parseFloat(chi) || 0, 0) * (1 + Math.max(parseFloat(overshoot) || 0, 0))
  return chiSq3Cdf(eff * eff) * 100
}
function coverageToChi(pct) {
  const p = Math.min(Math.max(pct, 0.1), 99.995) / 100
  let lo = 0, hi = 8
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    if (chiSq3Cdf(mid * mid) < p) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

// The sharpness slider changes meaning with the Auto Sharpening toggle:
// auto = parity target fraction, manual = fixed RCAS strength.
const sharpnessAuto = {
  id: 'sharpness', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.85,
  name: "Sharpness (Auto Parity Target)",
  desc: "Target fraction of the raw frame's local sharpness to restore (Auto Sharpening on). The sharpening amount is derived per pixel in closed form from the measured blur -- the ratio of local high-frequency (acutance) energy between the raw scene and the resolved image -- so heavily accumulated (blurred) areas get boosted while fresh, disoccluded and border pixels get almost none. Aliasing is not re-introduced: the target never exceeds the raw image's own energy, sub-perceptual detail is ignored via a noise floor, and RCAS's contrast limiter and noise suppression only ever reduce the boost further. 1.0 = full perceptual parity with the raw image, 0 disables sharpening."
}
const sharpnessManual = {
  id: 'sharpness', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.85,
  name: "Sharpness (Manual Strength)",
  desc: "Fixed FSR RCAS sharpening strength (Auto Sharpening off), applied uniformly to every pixel with no acutance measurement: 0 disables, 1.0 is the maximum lobe RCAS permits (very strong). RCAS's built-in contrast limiter and noise suppression remain active, so edges are protected from ringing and grain amplification. Values around 0.3-0.5 are a typical manual starting point."
}

const settingsSchema = [
{
  name: "General & Sharpening",
  items: [
    sharpnessAuto,
    { id: 'autoSharpen', type: 'numBool', default: 1.0, name: "Auto Sharpening", desc: "Selects how the Sharpness slider drives the final FSR RCAS pass. On: the slider is an auto-parity target -- each pixel's sharpening amount is derived from the measured acutance ratio between the raw and resolved images, restoring accumulated blur without ever exceeding the raw frame's sharpness. Off: the slider is a plain manual sharpening strength applied uniformly (also slightly cheaper -- the acutance measurement is skipped)." },
    { id: 'jitterScale', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 1.0, name: "Jitter Spread Scale", desc: "Scales the sub-pixel camera offset pattern. Above 1 samples a wider area within each pixel (more edge anti-aliasing, more temporal softening); below 1 tightens it." },
    { id: 'fallbackFXAA', type: 'numBool', default: 1.0, name: "Fallback Spatial AA (FXAA)", desc: "Applies FXAA edge smoothing to pixels whose temporal history was rejected, so disoccluded areas don't alias while the history rebuilds." }
  ]
},
  {
    name: "History & Blending",
    items: [
      { id: 'feedbackMax', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Static Scenes)", desc: "How much of the previous frame is reused for stationary pixels. Higher = smoother and cleaner but slower to react to changes; lower = more responsive but noisier. 0.97 corresponds to roughly a 33-frame accumulation window." },
      { id: 'feedbackMin', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Moving Pixels)", desc: "History reuse once pixel velocity exceeds the motion transition range below. Typically set at or below the static value so motion receives less smoothing. While the two weights are equal the motion reduction is inert (the shader clamps it into this range); the Sub-Pixel Alignment Drop below is exempt -- it applies outside the clamp, on planar surfaces only." },
      { id: 'driftCompensation', type: 'numBool', default: 1.0, name: "Lighting Drift Compensation", desc: "Accepts temporally persistent, spatially common level shifts (lighting changes, exposure, moving shadows) directly into the history instead of clipping them frame by frame. Each frame measures the local common-mode innovation against the same-depth-layer neighborhood (the same 5-tap spatial estimator the clip record uses) and applies it to the history with a noise-aware shrinkage: large clean shifts track in about one frame, slow crawls track proportionally to their measured certainty, so the temporal smoothing survives. Requires Depth-Dilated Motion Search (the same-layer mask). Ghost protection is unaffected: the correction is capped at the local content's own variability (a larger common shift is a reveal/disocclusion event and stays with the rejection paths), and a spatially common ghost is simply replaced by the current content -- which is the eviction action anyway. Off = the previous behavior (lighting steps ramp through the clip at a spatially varying rate)." },
      { id: 'driftMaxGain', type: 'float', min: 0.0, max: 0.9, step: 0.01, default: 0.2, name: "Lighting Tracking Speed", desc: "How fast the drift tracker accepts a confirmed lighting shift once it clears the estimator's 3-sigma noise threshold. The relationship to noise is structural, not tunable elsewhere: a correction applied at gain g injects about g times the estimator's noise into the history each tracking frame, versus the normal accumulation path's much smaller share -- speed and noise are the same dial. 0.2 (default) tracks in roughly 5 frames with a brief, mild noise bump; 0.5+ tracks in ~2 frames at a visibly noisier transient; 0 disables the correction (lighting then ramps through the normal blend). Large steps are unaffected -- they snap through the spike/reset path regardless." },
      { id: 'lumaDriftStrength', type: 'float', min: 0.0, max: 0.3, step: 0.01, default: 0.0, name: "Luma Drift Correction", desc: "Pulls history brightness toward the current image to clear ghost trails from moving shadows, exposure changes and vehicle lights, without reducing temporal smoothing. 0 = off. Only engages on large brightness mismatches on the same surface (see the chroma gate below), so it does not chase noise, jitter or ghosts. Higher values clear trails faster. (Legacy: superseded by Lighting Drift Compensation above -- leave off unless you need the old manual behavior.)" },
      { id: 'lumaDriftChromaTol', type: 'float', min: 0.0, max: 0.5, step: 0.01, default: 0.1, name: "Drift Same-Surface Tolerance", desc: "Color-match tolerance for the luma drift correction, comparing color-per-brightness so that pure lighting changes (shadows, exposure) pass while a different surface does not. History from a differently-colored object (ghosts, reveal edges) fails this test and is left to the normal rejection paths -- drift only ever corrects brightness, never disguises color mismatches. Raise if legitimate shadows aren't being corrected; lower if colored ghosts linger. 0 requires an exact match (drift effectively off)." },
      { id: 'motionBlendStart', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Motion Transition Start", desc: "Pixel velocity (in pixels per frame) at which blending starts transitioning from the static to the motion weight. No effect while the two History Blend weights are equal." },
      { id: 'motionBlendDropSpeed', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 1.0, name: "Motion Transition Width", desc: "Width, in pixels per frame, of the transition between the two History Blend weights: the motion weight is fully reached at (Motion Transition Start + this value). 1.0 = an abrupt 1 px/frame transition. No effect while the two History Blend weights are equal." },
      { id: 'alignmentFeedbackDrop', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.25, name: "Sub-Pixel Alignment Drop", desc: "Lowers history weight when the reprojected history sample lands between texels, where the resampling kernel is least accurate -- preserving texture sharpness in motion. Active only on planar surface interiors (the bilinear-motion path); edges and dilation zones are exempt because their current samples are aliased and dropping history there makes them flicker. Works independently of the History Blend weights. 0 disables. At 0.5, badly-phased pixels drop from a 33-frame to a ~2-frame accumulation window; the cost is a fixed texel-scale noise pattern on smooth noisy surfaces (sky gradients, shadow noise) in static scenes. Verify with debug mode 10 (red = applied drop, green = eligible); note that the Kaiser-6 resampler already preserves most sub-texel detail, so the effect is most visible with Kaiser-6 off or sharpening at 0." }
    ]
},
{
  name: "Jitter & Sampling",
  items: [
    { id: 'useJitter', type: 'bool', default: true, name: "Sub-Pixel Camera Jitter", desc: "Offsets the camera by a sub-pixel amount each frame so successive frames sample different positions within each pixel. This is the source of TAA's supersampling -- without it, TAA only stabilizes noise, it does not anti-alias." },
    { id: 'useR2Jitter', type: 'bool', default: true, name: "R2 Jitter Sequence", desc: "Uses the R2 low-discrepancy sequence instead of Halton(2,3) for the jitter pattern. R2 spreads samples more evenly across its 32-frame cycle." },
    { id: 'useKaiser6', type: 'numBool', default: 1.0, name: "Kaiser-6 History Resampling", desc: "21-tap Kaiser-windowed sinc resampling of the history buffer. Retains noticeably more detail in motion than the 9-tap Kaiser-4 fallback (the two differ only for moving pixels), at over twice the sampling cost." },
    { id: 'historyOvershoot', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 1.0, name: "History Resampling Overshoot Margin", desc: "Soft anti-ringing margin for history resampling, shared by both Kaiser paths and applied equally to brightness and color, as a fraction of the local color range. Band-limited overshoot -- the edge detail the filter reconstructs -- survives, while ringing and color fringing are compressed. Higher = sharper history, lower = fewer halos." },
    { id: 'useDepthDilation', type: 'numBool', default: 1.0, name: "Depth-Dilated Motion Search", desc: "Searches the 3x3 neighborhood for the closest surface and uses its motion vector, so silhouette edges reproject with the foreground's motion instead of the background's. The resolved per-pixel motion, depth and layer classification are stored in the motion field each frame and become the authoritative data the next frame's disocclusion and history-validation tests read at the reprojection landing. With Motion-Field Validation disabled, dilation runs unvalidated." }
  ]
},
{
  name: "History Clipping",
  items: [
    { id: 'useHullClipping', type: 'numBool', default: 1.0, name: "Temporal Clip Memory", desc: "Gives the history clip a per-pixel memory: an exponential record of how much the resolved image has actually been changing at this texel (the frame-to-frame innovation), stored in the output's alpha channel and transported with the content through reprojection. The clip gate's size is driven by this record instead of the current frame's neighborhood spread alone -- which is what lets fine sub-pixel detail (specular sparkle, thin highlights, dense textures) accumulate its true anti-aliased value instead of being clipped away every frame (the classic fine-detail shimmer). The record also learns the history resampler's own blur, so no extra clip margin is needed to compensate for the resampling kernel. Resets on disocclusion and re-warms within about 7 frames; costs one extra history-buffer read. Off: the gate sizes itself from the current neighborhood's statistics only (classic variance-clipping behavior -- fine detail shimmers again)." },
    { id: 'varianceGamma', type: 'float', unit: 'coverage', min: 20.0, max: 99.99, step: 0.01,
      // default is in STORED units (chi), matching the Lua defaultSettings:
      // defaultSettings feeds applyPreset() raw, so a display-unit default
      // here made every preset write varianceGamma = 95.06 (chi!) into the
      // config and the settings file. All display-unit conversions live in
      // the helpers below.
      default: 2.8,
      name: "Clip Coverage (Accepted History)",
      desc: "The percentage of statistically valid history the clip gate accepts untouched each frame (the coverage of the 3-degree-of-freedom confidence region). 95% is the principled default. The false clips that remain at any setting are marginal -- a one-frame few-percent pull toward the neighborhood mean, invisible -- so the reason to go higher is heavy-tailed content (specular sparkle, noise) rather than shimmer: 99.9% is a ~4.0-sigma gate, 99.99% is ~4.6 (the pre-normalization-fix gate at the old default behaved like 99.997%). Lower = tighter: sharper and less ghosting, but below ~82% legitimate sub-pixel detail starts getting clipped and shimmers. This is the EFFECTIVE coverage: the Clip Radius Margin below widens it beyond what this slider alone shows." },
    { id: 'colorSpaceOklab', type: 'numBool', default: 0.0, name: "Perceptual Working Color Space (Oklab)", desc: "Runs the history clipping statistics in Oklab, a perceptually uniform color space: the clip gate's error budget then weights color differences roughly the way human vision does, so clipping decisions favor what you'd actually notice. Off switches to the cheaper YCoCg space -- the resolve's single biggest ALU block is the nine-tap Oklab conversion (three cube roots per tap), so off is a measurable GPU saving, and is what the Performance and Balanced presets select. The cost: chroma error gets weighted like luma error, so colored-edge clipping decisions are slightly less perceptual; expect subtly different chroma-shimmer / colored-ghost behavior at equal slider values." },
    { id: 'chromaVarianceMod', type: 'float', min: 0.5, max: 2.0, step: 0.01, default: 1.0, name: "Chroma Bounds Scale", desc: "Independent multiplier for the color (non-brightness) axes of the clip gate. Above 1 gives chroma more room than brightness (helps colored fine detail accumulate); below 1 clamps chroma harder (tighter control of colored-edge ghosting at the cost of chroma shimmer)." },
        { id: 'softClip', type: 'numBool', default: 1.0, name: "Bayesian Soft Clip", desc: "Beyond the clip gate's boundary, replaces the hard snap to the ellipsoid surface with the exact posterior-mean action: the ghost alternative is the empirical spatial mixture of the nine neighborhood taps (not a fitted Gaussian), and the prior odds carry the accumulated sequential evidence from the ghost detector -- so a pixel with a long clean record barely shrinks while a pixel the detector is already suspicious of pulls decisively. Mild overshoots get a gentle correction graded by ghost probability; confirmed ghosts pull fully to the best-explaining content. The ghost posterior ALSO floors the temporal blend weight (so persistent low-contrast ghosts are evicted even below the clip gate's radius, by detector evidence rather than a gate excursion). Motion raises the ghost prior. The contrast limiter (the backstop) still guarantees the result stays within the gate's containment. Off = the pure hard ellipsoid (the A/B baseline)." },
    { id: 'clipOvershoot', type: 'float', min: 0.0, max: 0.5, step: 0.01, default: 0.0, name: "Clip Radius Margin", desc: "Multiplies the clip gate's radius by (1 + this) -- a pure comfort margin, included in the EFFECTIVE coverage shown in the Clip Coverage slider above (raising it widens the coverage). With Temporal Clip Memory on, the record learns the resampling footprint by itself, so 0 is correct; use a small value (0.05-0.1) only with the memory off if legitimate edge detail looks clipped." },
    { id: 'clipScopedMu', type: 'float', min: 0.0, max: 1.0, step: 0.05, default: 1.0, name: "Clip Variance Scoping (Advanced)", desc: "How exactly the clip gate models the neighborhood mean's own variance. 1.0 (default) = residual-scoped, the statistically exact form: edge ghosts get clipped from roughly 0.2x the local contrast upward. 0 = the conservative full payment (~0.5x, the pre-fix behavior apart from the per-channel normalization). Back this off toward 0, or raise the coverage, if debug mode 11 shows static fine detail taking visible shrinkage (blue). Intermediate values blend continuously." },    
    { id: 'clipGhostReset', type: 'select', default: 1.0, name: "Ghost Trail Detector",
      desc: "Fights ghost trails on background pixels next to foreground edges -- the class of ghost that lives AT the clip gate's own neighborhood mean and is therefore invisible to mean-based rejection. The detector watches the only uncontaminated signal (the signed frame-to-frame difference between the current render and the accumulated history) and accumulates PERSISTENCE evidence: random noise breaks the sign run and resets it, so only a sustained one-sided drift -- a ghost -- accumulates. Soft (default): as evidence builds, the clip gate's center migrates from the edge-contaminated neighborhood statistics to the same-depth-layer statistics, pulling the ghost to the clean surface; the posterior also floors the blend weight toward the current frame so the ghost is evicted even below the gate's radius. Valid edges are safe by construction: their jitter-corrected innovation averages zero (runs reset), single-frame transients are amplitude-capped, and the law's per-side negative drift under no-ghost is structural. Telemetry: debug mode 11's blue channel shows the accumulator -- a static scene MUST sit near black (sustained brightness there means the detector is false-accumulating and something is wrong). Hard Reset additionally forces a one-frame full re-anchor when the run saturates. Requires Depth-Dilated Motion Search on for the layer statistics (the detector itself is pure color-side and works with disocclusion detection off).",
      options: [
        { label: 'Off', value: 0.0 },
        { label: 'Soft (default)', value: 1.0 },
        { label: 'Telemetry Only (check mode 11)', value: 2.0 },
        { label: 'Soft + Hard Reset', value: 3.0 }
      ] },
]
  },
  {
    name: "Motion & Anti-flicker",
    items: [
      { id: 'jitterFlickerPadding', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.0, name: "Jitter Anti-Flicker Padding", desc: "Legacy comfort knob: inflates the clip gate's spatial floor by the color shift the current sub-pixel jitter is expected to cause. With Temporal Clip Memory on, the record measures the real phase variation directly and this only adds slack -- keep it at 0. Useful only with the memory off, as a manual stand-in for it (then raise until fine detail stops shimmering)." },
      { id: 'directionalVariance', type: 'numBool', default: 1.0, name: "Directional Padding", desc: "Expands the padding along the color direction of the expected jitter shift rather than uniformly. Only active when Jitter Anti-Flicker Padding is above zero (legacy path)." },
      { id: 'jitterFlickerFade', type: 'numBool', default: 0.0, name: "Fade Padding in Motion", desc: "Disables the jitter anti-flicker padding as pixel velocity rises, since reprojection error dominates jitter error in motion." },
      { id: 'lumaVariance', type: 'numBool', default: 0.0, name: "Luma-Weighted Statistics", desc: "Downweights bright samples when computing neighborhood statistics (Karas-style), keeping the gate's spatial floor from being stretched by specular fireflies." },
      { id: 'jitterAwareVariance', type: 'numBool', default: 1.0, name: "Jitter-Aware Statistics", desc: "Weights neighborhood samples by their distance from the jittered sampling position instead of the pixel center, placing the clip statistics on the current phase's mixture. Since the gate's test-vector correction (the kernel-centroid fix), either position is statistically sound -- the correction removes the phase wander from every test -- but On remains the default: it centers the statistics on the anti-aliased value by construction rather than by correction. Off is slightly cheaper (plain table weights) and is a reasonable Performance-mode selection." },
      { id: 'velocityAlignedVariance', type: 'numBool', default: 0.0, name: "Velocity-Aligned Statistics", desc: "Downweights neighborhood samples that lie behind the direction of motion, tightening the gate's spatial floor along motion trails." }
    ]
  },
  {
    name: "Advanced Rejection",
    items: [
      { id: 'useMotionField', type: 'numBool', default: 1.0, name: "Motion-Field Validation", desc: "Runs the extra motion-field pass (a fullscreen write storing each texel's resolved motion, depth and layer state) and uses last frame's field in the resolve to validate depth-dilated motion, gate history ownership, and run the depth and velocity disocclusion tests below. Disabling reduces the writer to a minimal two-fetch fullscreen write (the pass must keep running for chain stability) and skips all landing-side analysis in the resolve -- a large fraction of the total TAA cost -- at the price of disocclusion being detected only through color clipping (more ghosting behind moving objects and at reveals). The Performance preset disables this." },
      { id: 'depthRejection', type: 'float', min: 0.0, max: 1.0, step: 0.001, default: 0.05, name: "Disocclusion Sensitivity (Depth)", desc: "Threshold of the geometry-based depth disocclusion test: how much closer than the current surface transported one frame forward (exact under camera rotation; the layer's forward motion is fitted per surface from the neighborhood's measured parallax) the history depth must be before it is rejected as stale. Lower = more sensitive. 0 disables the depth test (the velocity test then runs standalone). Requires Motion-Field Validation." },
      { id: 'velRejection', type: 'float', min: 0.0, max: 10.0, step: 0.1, default: 1.5, name: "Disocclusion Threshold (Velocity)", desc: "Pixel threshold of the velocity disocclusion test. History whose recorded surface motion no longer matches the current pixel is flagged, then confirmed by pursuing that surface into the current frame -- only a confirmed divergence rejects, so motion-vector noise alone cannot. 0 disables. Lower catches subtler ghosts behind accelerating occluders; too low speckles static scenes. The comparison is exactly de-jittered and the landing-side velocity is the exact stored effective velocity, so the default 1.5 is already conservative -- values down to ~0.5 are viable. Tune with debug modes 2 (green should appear only on true reveals) and 7. Known limitation: on fast-moving or rotating foregrounds, pixels in the object's edge dilation zone can trigger occasional random rejections (point-sampled motion of a fast layer); if that bothers you, raise Velocity Noise Allowance or lower Pursuit Confirmation Strength. Requires Motion-Field Validation." },
      { id: 'velGradientScale', type: 'float', min: 0.0, max: 4.0, step: 0.05, default: 1.0, name: "Velocity Noise Allowance", desc: "Scales the velocity-coherent noise allowance of the disocclusion alert (how much neighboring motion-vector variation is treated as noise rather than signal). Raise if noisy velocity content -- vegetation, particles, alpha-tested edges, or fast-foreground edge dilation zones -- causes speckled alerts in debug mode 2; lower for a stricter alert." },
      { id: 'crossTestStrength', type: 'float', min: 0.0, max: 1.0, step: 0.05, default: 0.35, name: "Pursuit Confirmation Strength", desc: "How strongly the current-frame pursuit must confirm a flagged velocity mismatch before history is actually rejected (scales the measured divergence against its tolerance). Higher = more velocity rejections; lower makes the pursuit stricter about confirming, which also suppresses the dilation-zone false positives on fast foregrounds. 0.35 is conservative; 0.6-1.0 is reasonable once verified against debug modes 2 and 7." },
      { id: 'clipDistanceRejectionEnabled', type: 'numBool', default: 0.0, name: "Smear Rejection", desc: "Drops history weight where the clip gate had to shrink the history a long way (measured in units of the gate's own sigma, so sub-pixel detail does not trigger it) -- a ghosting indicator for content without motion vectors (animated textures, particles)." },
      { id: 'clipDistanceRejectionAmount', type: 'float', min: 0.0, max: 1.0, step: 0.001, default: 0.0, name: "Smear Rejection Strength", desc: "How quickly history weight falls once the clip distance exceeds the minimum error below -- the ramp width is 1/this: at 1.0 rejection is complete one sigma past the knee, at 0.25 it ramps over four sigma. Higher = rejects sooner." },
      { id: 'clipDistanceRejectionMinError', type: 'float', min: 0.001, max: 0.5, step: 0.001, default: 0.15, name: "Smear Rejection Min Error", desc: "Minimum clip distance before smear rejection begins to engage." }
    ]
  },
  {
    name: "Debug & Diagnostics",
    items: [
      { 
        id: 'debugMode', 
        type: 'select', 
        default: 0.0, 
        name: "Debug View Mode", 
        desc: "Visualizes internal buffers and rejection masks (the final sharpen pass is bypassed in every mode except 0).\n\n1: Frame Motion -- reprojection motion per pixel.\n2: Disocclusion Breakdown -- red = depth rejection, green = velocity rejection, yellow = both, faint blue = velocity error flagged but pursuit did NOT confirm (suppressed alert).\n3: Center Velocity -- current-frame motion vector magnitude.\n4: Linearized Depth -- normalized 0-100 m.\n5: History Color -- resampled history before clipping.\n6: History-Side Velocity -- effective previous-frame velocity at the history landing.\n7: Pursuit Divergence -- red intensity = divergence magnitude, green = confirmed rejection.\n8: Layer State -- orange = revoked dilation candidate, red = kept dilation zone, cyan = crest, magenta = depth-flat but velocity-straddled.\n9: Dilation Gate -- red = revoked candidate, green = kept (full = flag branch, half = depth branch), blue = depth-rejected.\n10: Alignment Drop -- red intensity = the sub-pixel feedback drop actually applied, green tint = planar (eligible) pixels, blue tint = edges / dilation zones (exempt).\n11: Clip Gate State -- green = temporal clip memory carried, dim green = record reset this frame, red = no record. Blue = how far the gate shrank the history toward the neighborhood mean (static content should sit near black -- that is the tightness win), or with the Ghost Trail Detector in Telemetry mode the accumulated detector evidence: near black on a static scene is the HEALTHY signature (any proper likelihood-ratio increment has negative mean under no-ghost); sustained brightness on static content means the detector is false-accumulating and something is wrong.\n12: Dejittered Residual -- jitter-cancel verification; must be near black on static scenes.",
        options: [
          { label: 'Off (Normal Rendering)', value: 0.0 },
          { label: '1 - Frame Motion', value: 1.0 },
          { label: '2 - Disocclusion Breakdown (R:depth G:velocity B:unconfirmed)', value: 2.0 },
          { label: '3 - Center Velocity', value: 3.0 },
          { label: '4 - Linearized Depth', value: 4.0 },
          { label: '5 - History Color', value: 5.0 },
          { label: '6 - History-Side Velocity', value: 6.0 },
          { label: '7 - Pursuit Divergence', value: 7.0 },
          { label: '8 - Layer State', value: 8.0 },
          { label: '9 - Dilation Gate', value: 9.0 },
          { label: '10 - Alignment Drop', value: 10.0 },
          { label: '11 - Clip Gate State', value: 11.0 },
          { label: '12 - Dejittered Residual', value: 12.0 }
        ]
      }
    ]
  }
]

const defaultSettings = {}
settingsSchema.forEach(cat => {
  cat.items.forEach(item => { defaultSettings[item.id] = item.default })
})

const presets = {
  Performance: {
    feedbackMax: 0.95, feedbackMin: 0.95,
    useKaiser6: 0, colorSpaceOklab: 0, useMotionField: 0,
    clipScopedMu: 0,
    // --- aggressive tier: uncomment for max performance (each has a visible
    // --- cost; see the preset-notes discussion):
    // useDepthDilation: 0,            // -16 fetches/px: the biggest gate; silhouettes lose dilated motion
    // autoSharpen: 0, sharpness: 0,   // the whole sharpening pipeline becomes free
    // useHullClipping: 0,             // fine-detail shimmer; the state fetch only vanishes if sharpening is off too
    // jitterAwareVariance: 0,         // ~50 ops/px; statistically sound since the C2 test-vector fix
    // fallbackFXAA: 0,                // 4 fetches on transient pixels; aliased reveals
  },
  Balanced: { colorSpaceOklab: 0 },
  Clarity: { feedbackMax: 0.95, feedbackMin: 0.95, },
  Smooth: { varianceGamma: 4.594 }
}

// Per-item display override: the sharpness row follows the auto-sharpen toggle.
function displayItem(item) {
  if (item && item.id === 'sharpness') {
    const auto = config.value.autoSharpen === 1 || config.value.autoSharpen === true
    return auto ? sharpnessAuto : sharpnessManual
  }
  return item
}

function toLuaValue(val) {
  return typeof val === 'boolean' ? (val ? 'true' : 'false') : val
}

function formatDefault(item) {
  if (item.type === 'bool' || item.type === 'numBool') {
    return (item.default === 1 || item.default === true) ? 'Enabled' : 'Disabled'
  }
  if (item.type === 'select') {
    const opt = item.options.find(o => o.value === item.default)
    return opt ? opt.label : item.default
  }
  return item.unit === 'coverage' ? `${chiToCoverage(item.default, 0).toFixed(2)} %` : item.default
}

function resetSetting(item) {
  if (item.unit === 'coverage') {
    // item.default is the default chi: restore the default EFFECTIVE
    // coverage given the current radius margin.
    const margin = 1 + Math.max(parseFloat(config.value.clipOvershoot) || 0, 0)
    config.value.varianceGamma = +(item.default / margin).toFixed(4)
  } else {
    config.value[item.id] = item.default
  }
  updateSetting(item.id)
}

function sliderModel(item) {
  if (item.unit === 'coverage')
    return Math.min(99.99, +chiToCoverage(config.value.varianceGamma, config.value.clipOvershoot).toFixed(2))
  return config.value[item.id]
}

function sliderModified(item) {
  if (item.unit === 'coverage') {
    // Compare display against display, rounded identically (both sides call
    // the same chiToCoverage, so the defaults compare exactly equal at any
    // erf precision). Semantic: "the EFFECTIVE coverage differs from the
    // default coverage" -- margin-independent by design.
    const defaultDisp = +chiToCoverage(item.default, 0).toFixed(2)
    return Math.abs(sliderModel(item) - defaultDisp) >= 0.005
  }
  return config.value[item.id] !== item.default
}

function onSliderChange(item, val) {
  if (item.unit === 'coverage') {
    // The slider is the EFFECTIVE coverage (radius margin included); back
    // out the base radius the shader and the Lua Student fit consume.
    const chi = coverageToChi(val)
    const margin = 1 + Math.max(parseFloat(config.value.clipOvershoot) || 0, 0)
    config.value.varianceGamma = +(chi / margin).toFixed(4)
  } else {
    config.value[item.id] = val
  }
  updateSetting(item.id)
}

function onSwitchChange(item, val) {
  config.value[item.id] = (item.type === 'numBool') ? (val ? 1 : 0) : val
  updateSetting(item.id)
}

function toggleDropdown(id) {
  if (!isActive.value) return
  openDropdown.value = openDropdown.value === id ? null : id
}

function selectOption(item, val) {
  config.value[item.id] = val
  updateSetting(item.id)
  openDropdown.value = null
}

function getOptionLabel(item, val) {
  const opt = item.options.find(o => o.value === val)
  return opt ? opt.label : val
}

function applyPreset(presetOverrides) {
  for (const key in defaultSettings) { config.value[key] = defaultSettings[key] }
  for (const key in presetOverrides) {
    if (!(key in defaultSettings)) continue   // guard against stale preset keys
    config.value[key] = presetOverrides[key]
  }

  if (!window.bngApi || !window.bngApi.engineLua) return
  // Single round-trip: the Lua side applies the whole table and saves once
  // (falls back to per-key calls if the bulk API is not present).
  const entries  = Object.entries(config.value).map(([k, v]) => `${k} = ${toLuaValue(v)}`).join(', ')
  const fallback = Object.entries(config.value).map(([k, v]) => `t.uiSetSetting("${k}", ${toLuaValue(v)})`).join(' ')
  window.bngApi.engineLua(`local t = taa or taa_taa; if t then if t.uiSetSettings then t.uiSetSettings({${entries}}) else ${fallback} end end`)
}

function updateSetting(key) {
  if (!window.bngApi || !window.bngApi.engineLua) return
  const v = toLuaValue(config.value[key])
  window.bngApi.engineLua(`if taa then taa.uiSetSetting("${key}", ${v}) elseif taa_taa then taa_taa.uiSetSetting("${key}", ${v}) end`)
}

onMounted(() => {
  config.value = { ...defaultSettings }
  if (window.bngApi && window.bngApi.engineLua) {
    const script = "(taa and taa.requestUIState()) or (taa_taa and taa_taa.requestUIState()) or nil"
    window.bngApi.engineLua(script, (state) => {
      if (state) {
        isActive.value = state.active !== false
        if (state.settings && Object.keys(state.settings).length > 0) {
          config.value = { ...defaultSettings, ...state.settings }
        }
      }
    })
  }
})

function toggleTaa(val) {
  isActive.value = val
  const stateStr = val ? 'true' : 'false'
  if (window.bngApi && window.bngApi.engineLua) {
    window.bngApi.engineLua(`if taa then taa.uiSetEnabled(${stateStr}) elseif taa_taa then taa_taa.uiSetEnabled(${stateStr}) end`)
  }
}
</script>

<style scoped lang="scss">
* { box-sizing: border-box; }
.taa-bng-options {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 65vh; 
  overflow: hidden; 
  font-family: 'Overpass', sans-serif;
  color: #fff;
  background-color: var(--bng-off-black, rgba(15, 15, 15, 0.95));
  border-radius: 6px;
}
.options-header {
  flex: 0 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5em 1em 0.8em;
  background-color: rgba(0, 0, 0, 0.2);
  border-bottom: 2px solid var(--bng-orange-550, #f60);
  .header-title { font-size: 1.3rem; font-weight: 600; }
  .header-toggle { display: flex; align-items: center; font-weight: 600; }
}
.options-presets {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  padding: 0.5em 1em;
  background-color: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  .preset-btn { flex: 1; margin: 0 0.25em; --bng-button-padding-y: 0.3em; font-size: 0.9em; }
}
.options-list-scroll {
  flex: 1 1 0; 
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  padding: 0.5em;
  transition: opacity 0.2s;
  &.is-disabled { opacity: 0.35; pointer-events: none; }
  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: transparent; }
  &::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 4px; }
  &::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.4); }
}
details.category-block {
  margin-bottom: 0.5em;
  width: 100%;
  background: rgba(0, 0, 0, 0.15);
  border-radius: 4px;
}
summary.category-title {
  display: flex;
  align-items: center;
  padding: 0.5em 0.75em;
  font-size: 1.05rem;
  font-weight: 600;
  color: #ddd;
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  list-style: none;
  user-select: none;
  border-radius: 4px;
  transition: background-color 0.1s ease;
  &::-webkit-details-marker { display: none; }
  &:hover { background: rgba(255, 255, 255, 0.08); }
  &::before {
    content: '▶';
    display: inline-block;
    margin-right: 0.5em;
    font-size: 0.7rem;
    color: var(--bng-orange-550, #f60);
    transition: transform 0.2s ease;
  }
}
details[open] > summary.category-title {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  &::before { transform: rotate(90deg); }
}
.category-items { padding: 0.25em 0; }
.options-item-row {
  display: flex;
  flex-direction: column;
  background-color: rgba(0, 0, 0, 0.2); 
  margin-bottom: 2px;
  padding: 0.6em 0.8em;
  width: 100%; 
  &:nth-child(even) { background-color: transparent; }
  .row-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    .option-label { flex: 1 1 auto; font-size: 0.95rem; padding-right: 0.5em; }
    .inline-controls { flex: 0 0 auto; display: flex; align-items: center; gap: 0.75em; }
  }
  .stacked-controls {
    display: flex;
    align-items: center;
    width: 100%;
    margin-top: 0.6em;
    gap: 0.75em;
    .native-slider { flex: 1 1 auto; min-width: 0; }
  }
}
.dropdown-backdrop { position: fixed; inset: 0; z-index: 9998; cursor: default; }
.custom-dropdown-container {
  position: relative;
  flex: 1 1 auto;
  font-size: 0.95rem;
  z-index: 1; 
  &.is-open { z-index: 9999; }
  &.is-disabled { opacity: 0.4; pointer-events: none; }
}
.dropdown-selected {
  background: rgba(0, 0, 0, 0.4);
  color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  padding: 0.45em 0.75em;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  transition: border-color 0.15s ease;
  user-select: none;
  &:hover { border-color: rgba(255, 255, 255, 0.4); }
  .dropdown-arrow { font-size: 0.7em; color: rgba(255, 255, 255, 0.5); margin-left: 0.5em; }
}
.custom-dropdown-container.is-open .dropdown-selected {
  border-color: var(--bng-orange-550, #f60);
  .dropdown-arrow { color: var(--bng-orange-550, #f60); }
}
.dropdown-list {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  background: #1a1a1a;
  border: 1px solid var(--bng-orange-550, #f60);
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}
.dropdown-option {
  padding: 0.5em 0.75em;
  cursor: pointer;
  color: #ddd;
  transition: background-color 0.1s;
  &:hover { background: rgba(255, 102, 0, 0.2); color: #fff; }
  &.is-active { background: var(--bng-orange-550, #f60); color: #fff; font-weight: 600; }
}
.option-reset {
  flex: 0 0 2.5em; 
  width: 2.5em;
  display: flex;
  justify-content: flex-end;
  padding-left: 0.5em;
  .bng-reset-btn { --bng-button-padding-x: 0.3em; --bng-button-padding-y: 0.15em; opacity: 0.8; &:hover { opacity: 1; } }
}
.options-info-panel {
  flex: 0 0 8.0em; 
  min-height: 8.0em;
  max-height: 8.0em;
  width: 100%;
  background-color: rgba(0, 0, 0, 0.6);
  border-top: 2px solid var(--bng-orange-550, #f60);
  padding: 0.6em 1.2em;
  overflow-y: auto;
  overflow-x: hidden;
  .info-empty { height: 100%; display: flex; align-items: center; justify-content: center; color: rgba(255, 255, 255, 0.4); font-size: 0.95em; font-style: italic; }
  .info-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.2em;
    .info-title { font-weight: 700; font-size: 1.05rem; color: #fff; }
    .info-default { font-family: monospace; font-size: 0.9em; color: rgba(255, 255, 255, 0.5); }
  }
  .info-desc { font-size: 0.9rem; color: #ccc; line-height: 1.35; white-space: pre-wrap; }
}
</style>