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
            <!-- Header: label, switch (bools), and the ONE reset button -->
            <div class="row-header">
              <div class="option-label">{{ item.name }}</div>
              <div class="header-controls">
                <BngSwitch
                  v-if="item.type === 'bool' || item.type === 'numBool'"
                  :modelValue="config[item.id] === 1 || config[item.id] === true"
                  @update:modelValue="val => onSwitchChange(item, val)"
                  :disabled="!isActive"
                />
                <BngButton
                  :icon="icons.undo"
                  :accent="ACCENTS.outlined"
                  @click="resetSetting(item)"
                  class="bng-reset-btn"
                  :style="{ visibility: isModified(item) ? 'visible' : 'hidden' }"
                  title="Reset to default"
                />
              </div>
            </div>

            <!-- Float slider -->
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
            </div>

            <!-- Dropdown select -->
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
            </div>

          </div>
        </div>
      </details>
    </div>

    <!-- Dynamic Info Panel -->
    <div class="options-info-panel">
      <template v-if="hoveredItem">
        <div class="info-header">
          <span class="info-title">{{ hoveredItem.name }}</span>
          <span class="info-default">Default: {{ formatDefault(hoveredItem) }}</span>
        </div>
        <div class="info-desc">{{ hoveredItem.desc }}</div>
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
// The host's Student fit is constructed to restore this nominal coverage at
// every record age, so the slider's meaning is age-independent.
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

const settingsSchema = [
{
  name: "General & Sharpening",
  items: [
    { id: 'sharpness', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.50, name: "Sharpness", desc: "Fixed FSR RCAS sharpening strength: 0 disables, 1.0 is the maximum lobe RCAS permits (very strong). RCAS's built-in contrast limiter and noise suppression remain active, so edges are protected from ringing and grain amplification. Values around 0.3-0.5 are a typical starting point." },
    { id: 'jitterScale', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 1.0, name: "Jitter Spread Scale", desc: "Scales the sub-pixel camera offset pattern. Above 1 samples a wider area within each pixel (more edge anti-aliasing, more temporal softening); below 1 tightens it." },
    { id: 'fallbackFXAA', type: 'numBool', default: 1.0, name: "Fallback Spatial AA (FXAA)", desc: "Applies FXAA edge smoothing to pixels whose temporal history was rejected, so disoccluded areas don't alias while the history rebuilds." }
  ]
},
{
  name: "History & Blending",
  items: [
    { id: 'feedbackMax', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Static Scenes)", desc: "How much of the previous frame is reused for stationary pixels. Higher = smoother and cleaner but slower to react to changes; lower = more responsive but noisier. 0.97 corresponds to roughly a 33-frame accumulation window." },
    { id: 'feedbackMin', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Moving Pixels)", desc: "History reuse once pixel velocity exceeds the motion transition range below. Typically set at or below the static value so motion receives less smoothing. While the two weights are equal the motion reduction is inert (the shader clamps it into this range); the Sub-Pixel Alignment Drop below is exempt -- it applies outside the clamp, on planar surfaces only." },
    { id: 'motionBlendStart', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Motion Transition Start", desc: "Pixel velocity (in pixels per frame) at which blending starts transitioning from the static to the motion weight. No effect while the two History Blend weights are equal." },
    { id: 'motionBlendDropSpeed', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 1.0, name: "Motion Transition Width", desc: "Width, in pixels per frame, of the transition between the two History Blend weights: the motion weight is fully reached at (Motion Transition Start + this value). 1.0 = an abrupt 1 px/frame transition. No effect while the two History Blend weights are equal." },
    { id: 'alignmentFeedbackDrop', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.15, name: "Sub-Pixel Alignment Drop", desc: "Lowers history weight when the reprojected history sample lands between texels, where the resampling kernel is least accurate -- preserving texture sharpness in motion. Active only on planar surface interiors (the bilinear-motion path); edges and dilation zones are exempt because their current samples are aliased and dropping history there makes them flicker. Works independently of the History Blend weights. 0 disables. At 0.5, badly-phased pixels drop from a 33-frame to a ~2-frame accumulation window; the cost is a fixed texel-scale noise pattern on smooth noisy surfaces (sky gradients, shadow noise) in static scenes. Verify with debug mode 10 (red = applied drop, green = eligible); note that the Wiener-6 resampler already preserves most sub-texel detail, so the effect is most visible with Wiener-6 off or sharpening at 0." },
  ]
},
{
  name: "Jitter & Sampling",
  items: [
    { id: 'useJitter', type: 'bool', default: true, name: "Sub-Pixel Camera Jitter", desc: "Offsets the camera by a sub-pixel amount each frame so successive frames sample different positions within each pixel. This is the source of TAA's supersampling -- without it, TAA only stabilizes noise, it does not anti-alias." },
    { id: 'useR2Jitter', type: 'bool', default: true, name: "R2 Jitter Sequence", desc: "Uses the R2 low-discrepancy sequence instead of Halton(2,3) for the jitter pattern. R2 spreads samples more evenly across its 32-frame cycle." },
    { id: 'useWiener6', type: 'numBool', default: 1.0, name: "Wiener-6 History Resampling", desc: "21-tap Wiener-windowed sinc resampling of the history buffer. Retains noticeably more detail in motion than the 9-tap Wiener-4 fallback (the two differ only for moving pixels), at over twice the sampling cost." },
    { id: 'historyOvershoot', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 1.0, name: "History Resampling Overshoot Margin", desc: "Soft anti-ringing margin for history resampling, shared by both Wiener paths and applied equally to brightness and color, as a fraction of the local color range. Band-limited overshoot -- the edge detail the filter reconstructs -- survives, while ringing and color fringing are compressed. Higher = sharper history, lower = fewer halos." },
    { id: 'useDepthDilation', type: 'numBool', default: 1.0, name: "Depth-Dilated Motion Search", desc: "Searches the 3x3 neighborhood for the closest surface and uses its motion vector, so silhouette edges reproject with the foreground's motion instead of the background's. The resolved per-pixel motion, depth and layer classification are stored in the motion field each frame and become the authoritative data the next frame's disocclusion and history-validation tests read at the reprojection landing. With Motion-Field Validation disabled, dilation runs unvalidated." }
  ]
},
{
  name: "History Clipping",
  items: [
    { id: 'useHullClipping', type: 'numBool', default: 1.0, name: "Temporal Clip Memory", desc: "Gives the history clip a per-pixel memory, transported through the output's alpha channel. It carries: an exponential record of the per-sample noise -- measured against the current frame's own neighborhood model, independent of the history itself, so neither persistent content changes nor the clipper's own past actions can contaminate it; the accumulated-noise share of the history read (12-bit precision); and the structure-split fraction -- how much of the neighborhood's spatial variation is stable texture rather than per-frame noise, learned over ~5 frames. The clip gate's size is driven by this state instead of the current frame's spread alone -- which is what lets fine sub-pixel detail (specular sparkle, thin highlights, dense textures) accumulate its true anti-aliased value instead of being clipped away every frame (the classic fine-detail shimmer). The record also learns the history resampler's own blur, so no extra clip margin is needed for the resampling kernel. A practical consequence of the noise-only record: persistent content changes (lighting steps, moving shadows) are no longer absorbed as 'noise' -- the gate sees them at full strength and the clamp itself evicts them within a few frames. Resets on disocclusion; re-warms in about 7 frames (the structure split in ~5). Costs one extra history-buffer read. Off: the gate sizes itself from the current neighborhood's statistics only (classic variance-clipping behavior -- fine detail shimmers again)." },
    { id: 'varianceGamma', type: 'float', unit: 'coverage', min: 20.0, max: 99.99, step: 0.01,
      // default is in STORED units (chi), matching the Lua defaultSettings:
      // defaultSettings feeds applyPreset() raw, so a display-unit default
      // here made every preset write varianceGamma = 95.06 (chi!) into the
      // config and the settings file. All display-unit conversions live in
      // the helpers below.
      default: 2.8,// 95%,
      name: "Clip Coverage (Accepted History)",
      desc: "The percentage of statistically valid history the clip gate accepts untouched each frame (the coverage of the 3-degree-of-freedom confidence region). 95% is the principled default. The false clips that remain at any setting are marginal -- a one-frame few-percent pull toward the neighborhood mean, invisible -- so the reason to go higher is heavy-tailed content (specular sparkle, noise) rather than shimmer: 99.9% is a ~4.0-sigma gate, 99.99% is ~4.6. Lower = tighter: sharper and less ghosting, but below ~82% legitimate sub-pixel detail starts getting clipped and shimmers. This is the EFFECTIVE coverage: the Clip Radius Margin below widens it beyond what this slider alone shows. Calibrate with debug mode 13: on a static scene the brightness histogram's 95th percentile should sit at mid-gray -- the nominal point is exact by construction, so a static scene should sit ON it. A systematic offset there is a modeling error, not a tuning choice." },
    { id: 'colorSpaceOklab', type: 'numBool', default: 0.0, name: "Perceptual Working Color Space (Oklab)", desc: "Runs the history clipping statistics in Oklab, a perceptually uniform color space: the clip gate's error budget then weights color differences roughly the way human vision does, so clipping decisions favor what you'd actually notice. Off switches to the cheaper YCoCg space -- the resolve's single biggest ALU block is the nine-tap Oklab conversion (three cube roots per tap), so off is a measurable GPU saving, and is what the Performance preset selects. The cost: chroma error gets weighted like luma error, so colored-edge clipping decisions are slightly less perceptual; expect subtly different chroma-shimmer / colored-ghost behavior at equal slider values." },
    { id: 'chromaVarianceMod', type: 'float', min: 0.5, max: 2.0, step: 0.01, default: 1.0, name: "Chroma Bounds Scale", desc: "Independent multiplier for the color (non-brightness) axes of the clip gate. Above 1 gives chroma more room than brightness (helps colored fine detail accumulate); below 1 clamps chroma harder (tighter control of colored-edge ghosting at the cost of chroma shimmer). Note: this is the one deliberately uncalibrated knob -- at any value other than 1 the gate's coverage percentage above is approximate, because scaling the metric after calibration turns the statistic into a weighted chi-square." },
    { id: 'softClip', type: 'numBool', default: 0.0, name: "Bayesian Soft Clip", desc: "Beyond the clip gate's boundary, replaces the hard snap to the ellipsoid surface with the exact posterior-mean action. The ghost alternative is the LANDING-MISS hypothesis: a proper mixture of the eight SURROUNDING taps' accumulated content (what a history that landed on the wrong texel holds) plus one temporal tail -- the center tap is deliberately excluded, since it is the no-ghost hypothesis's own evidence. The whole alternative is priced by a single MEASURED miss distribution: the transport machinery's own landing-scatter estimate generates the prior odds, the neighbor weights and the tail mass coherently -- there are no hand-set odds anywhere. Consequences: on clean static content the prior collapses and the soft clip is inert by design (no landing miss is possible; content changes are the hard clamp's business). On content with noisy motion vectors -- vegetation, particles -- it engages more decisively than any fixed prior could. Optional: the hard path is complete on its own (the exact smallest modification of the history consistent with the frame), and the difference is confined to marginal pixels and pull direction. Off = the pure hard ellipsoid (the A/B baseline)." },
    { id: 'clipOvershoot', type: 'float', min: 0.0, max: 0.5, step: 0.01, default: 0.0, name: "Clip Radius Margin", desc: "Multiplies the clip gate's radius by (1 + this) -- a pure comfort margin, included in the EFFECTIVE coverage shown in the Clip Coverage slider above (raising it widens the coverage). With Temporal Clip Memory on, the record learns the resampling footprint by itself, so 0 is correct; use a small value (0.05-0.1) only with the memory off if legitimate edge detail looks clipped." }
  ]
},
{
  name: "Statistics Weighting",
  items: [
    { id: 'lumaVariance', type: 'numBool', default: 0.0, name: "Luma-Weighted Statistics", desc: "Downweights bright samples when computing neighborhood statistics (Karas-style), keeping the plane fit and the gate's spatial floor from being stretched by specular fireflies." },
    { id: 'jitterAwareVariance', type: 'numBool', default: 1.0, name: "Jitter-Aware Statistics", desc: "Weights neighborhood samples by their distance from the jittered sampling position instead of the pixel center, placing the clip statistics on the current phase's mixture. Either position is statistically sound -- the test-vector correction removes the phase wander from every test -- but On remains the default: it centers the statistics on the anti-aliased value by construction rather than by correction. Off is slightly cheaper (plain table weights) and is a reasonable Performance-mode selection." },
    { id: 'velocityAlignedVariance', type: 'numBool', default: 0.0, name: "Velocity-Aligned Statistics", desc: "Downweights neighborhood samples that lie behind the direction of motion, tightening the gate's spatial floor along motion trails." }
  ]
},
{
  name: "Advanced Rejection",
  items: [
    { id: 'useMotionField', type: 'numBool', default: 1.0, name: "Motion-Field Validation", desc: "Runs the extra motion-field pass (a fullscreen write storing each texel's resolved motion, depth and layer state) and uses last frame's field in the resolve to validate depth-dilated motion, gate history ownership, and run the depth and velocity disocclusion tests below. Disabling reduces the writer to a minimal two-fetch fullscreen write (the pass must keep running for chain stability) and skips all landing-side analysis in the resolve -- a large fraction of the total TAA cost -- at the price of disocclusion being detected only through color clipping (more ghosting behind moving objects and at reveals). With Clip-Excess Blend Reduction enabled in History & Blending, that fallback is decisive on any reveal with a visible color change -- the residual gap is same-color geometry transitions and unvalidated depth-dilated edges. The Performance preset disables this and enables the blend reduction in its place." },
    { id: 'depthRejection', type: 'float', min: 0.0, max: 1.0, step: 0.001, default: 0.05, name: "Disocclusion Sensitivity (Depth)", desc: "Threshold of the geometry-based depth disocclusion test: how much closer than the current surface transported one frame forward (exact under camera rotation; the layer's forward motion is fitted per surface from the neighborhood's measured parallax) the history depth must be before it is rejected as stale. Lower = more sensitive. 0 disables the depth test (the velocity test then runs standalone). Requires Motion-Field Validation." },
    { id: 'velRejection', type: 'float', min: 0.0, max: 10.0, step: 0.1, default: 1.5, name: "Disocclusion Threshold (Velocity)", desc: "Pixel threshold of the velocity disocclusion test. History whose recorded surface motion no longer matches the current pixel is flagged, then confirmed by pursuing that surface into the current frame -- only a confirmed divergence rejects, so motion-vector noise alone cannot. 0 disables. Lower catches subtler ghosts behind accelerating occluders; too low speckles static scenes. The comparison is exactly de-jittered and the landing-side velocity is the exact stored effective velocity, so the default 1.5 is already conservative -- values down to ~0.5 are viable. Tune with debug modes 2 (green should appear only on true reveals) and 7. Known limitation: on fast-moving or rotating foregrounds, pixels in the object's edge dilation zone can trigger occasional random rejections (point-sampled motion of a fast layer); if that bothers you, raise Velocity Noise Allowance or lower Pursuit Confirmation Strength. Requires Motion-Field Validation." },
    { id: 'velGradientScale', type: 'float', min: 0.0, max: 4.0, step: 0.05, default: 1.0, name: "Velocity Noise Allowance", desc: "Scales the velocity-coherent noise allowance of the disocclusion alert (how much neighboring motion-vector variation is treated as noise rather than signal). Raise if noisy velocity content -- vegetation, particles, alpha-tested edges, or fast-foreground edge dilation zones -- causes speckled alerts in debug mode 2; lower for a stricter alert." },
    { id: 'crossTestStrength', type: 'float', min: 0.0, max: 1.0, step: 0.05, default: 0.35, name: "Pursuit Confirmation Strength", desc: "How strongly the current-frame pursuit must confirm a flagged velocity mismatch before history is actually rejected (scales the measured divergence against its tolerance). Higher = more velocity rejections; lower makes the pursuit stricter about confirming, which also suppresses the dilation-zone false positives on fast foregrounds. 0.35 is conservative; 0.6-1.0 is reasonable once verified against debug modes 2 and 7." },
    { id: 'clipBlendDrop', type: 'numBool', default: 0, name: "Clip-Excess Blend Reduction", desc: "Lowers the history blend when the reprojected history lands far OUTSIDE the clip gate -- the color-statistics ghost rejection path. The gate measures how many gate radii away the history landed (1.0 = exactly on the boundary); past the start margin below, the history blend falls linearly to zero over the drop width. The clamp and Bayesian soft clip repair the stored value; this decides how much of it survives the blend, so a badly-wrong history converges immediately instead of leaking through the feedback. This is the feature that makes Motion-Field Validation optional: on true reveals the history sits 3-10+ gate radii out, deep in the drop zone, while at the default margin only ~0.3% of statistically-valid pixels even touch the bottom of the ramp -- and current-frame fireflies cannot trip it, because the gate's robust weighting moves its center, not the history's residual. Honest residual gap vs. the depth/velocity detector: reveals where the old and new content have nearly the SAME color are invisible to any color test, and with the motion field off, depth-dilated edge motion runs unvalidated. Verify with debug mode 13: the drop engages where the grayscale brightness exceeds (1 + start margin) squared times the nominal mid-gray point -- at the defaults, roughly from 88% brightness up." },
    { id: 'clipBlendDropStart', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Clip-Excess Start Margin", desc: "How far past the clip gate's boundary, in gate radii, the history must land before the blend reduction begins. 0 starts reducing immediately at the boundary -- at 95% coverage that means 5% of statistically-valid pixels enter the ramp every frame (noise-side). 0.5 (default) starts at 1.5x the gate radius, where only ~0.3% of valid pixels enter, and only at the ramp's bottom; 1.0+ reserves the drop for gross violations only." },
    { id: 'clipBlendDropWidth', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Clip-Excess Drop Width", desc: "How much further past the start margin, in the same gate-radii units, until the history blend reaches zero -- the drop is linear in between. Small values (0.25-0.5) behave like a hard rejection threshold; large values (2+) make it a gradual confidence ramp. The default 1.0 reaches zero at 2.5x the gate radius (with the default margin): under the nominal law about 2-in-100-million valid pixels reach full drop per frame, while true reveals exceed it routinely." }
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
      desc: "Visualizes internal buffers and rejection masks (the final sharpen pass is bypassed in every mode except 0).\n\n1: Frame Motion -- reprojection motion per pixel.\n2: Disocclusion Breakdown -- red = depth rejection, green = velocity rejection, yellow = both, faint blue = velocity error flagged but pursuit did NOT confirm (suppressed alert).\n3: Center Velocity -- current-frame motion vector magnitude.\n4: Linearized Depth -- normalized 0-100 m.\n5: History Color -- the STORED history buffer exactly as this frame's blend read it.\n6: History-Side Velocity -- effective previous-frame velocity at the history landing.\n7: Pursuit Divergence -- red intensity = divergence magnitude, green = confirmed rejection.\n8: Layer State -- orange = revoked dilation candidate, red = kept dilation zone, cyan = crest, magenta = depth-flat but velocity-straddled.\n9: Dilation Gate -- red = revoked candidate, green = kept (full = flag branch, half = depth branch), blue = depth-rejected.\n10: Alignment Drop -- red intensity = the sub-pixel feedback drop actually applied, green tint = planar (eligible) pixels, blue tint = edges / dilation zones (exempt).\n11: Clip Gate State -- bright green = temporal clip memory carried, mid green = record reset this frame, red = no record. Blue = how far the gate shrank the history toward the neighborhood mean, including any soft-clip pull (static content should sit near black -- that is the tightness win; a persistent bright-B region on static content is a trail being evicted or a modeling error worth investigating).\n12: Dejittered Residual -- jitter-cancel verification; must be near black on static scenes.\n13: Gate Null Law (calibration) -- brightness encodes the realized clip-gate statistic; the nominal 95th percentile sits at mid-gray at the default 95% coverage. Park on a STATIC scene: the nominal point is exact by construction (the least-squares bookkeeping is exact), so a static scene should sit ON it -- systematically brighter = the gate runs hot (over-clipping, flicker-side), dimmer = conservative (ghost-side), and either is a modeling error, not a tuning choice. Blue tint marks pixels engaging the gate this frame. Also watch this view over ~30 seconds on NOISY static content: any upward creep of the brightness distribution over time indicates the record is drifting (a stability bug), not content. \n14: Deployment Canary -- three one-bit probes as color channels: R = the pushed host constants are live, G = the viewport uniforms are sane, B = the transported clip state decodes. White = the chain is alive end to end; a solid missing channel names the dead subsystem (R dead = stale/no-op'ed constants, G dead = viewport uniform failure, B dead = the store chain never ran). Settings-independent.",
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
        { label: '12 - Dejittered Residual', value: 12.0 },
        { label: '13 - Gate Null Law (calibration)', value: 13.0 },
        { label: '14 - Deployment Canary', value: 14.0 }
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
    useWiener6: 0, useMotionField: 0,
    clipBlendDrop: 1,       // the color-based fallback for the disabled motion field
    // --- aggressive tier (each has a visible cost):
    useDepthDilation: 0,    // -16 fetches/px, the biggest gate; silhouettes lose dilated motion
    // sharpness: 0,          // the RCAS pass becomes free
    useHullClipping: 0,    // fine-detail shimmer (the record, VarH and the split all ride this state)
    // jitterAwareVariance: 0, // ~50 ops/px; statistically sound since the test-vector phase fix
    // fallbackFXAA: 0,        // 4 fetches on transient pixels; aliased reveals
  },
  Balanced: {},                     // = defaults
  Clarity: { feedbackMax: 0.95, feedbackMin: 0.95 },
  Smooth: { varianceGamma: 4.5943 }  // stored chi: 99.99% effective coverage
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

// ONE modified test for every control type (the coverage slider compares
// display against display, margin-independent by design).
function isModified(item) {
  if (item.unit === 'coverage') {
    const defaultDisp = +chiToCoverage(item.default, 0).toFixed(2)
    return Math.abs(sliderModel(item) - defaultDisp) >= 0.005
  }
  return config.value[item.id] !== item.default
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

function onSliderChange(item, val) {
  if (item.unit === 'coverage') {
    // The slider is the EFFECTIVE coverage (radius margin included); back
    // out the base radius the shader and the Lua Student fit consume.
    const chi = coverageToChi(Number(val))
    const margin = 1 + Math.max(parseFloat(config.value.clipOvershoot) || 0, 0)
    config.value.varianceGamma = +(chi / margin).toFixed(4)
  } else {
    // Coerce: the slider's input box can emit strings, which would leak
    // into the Lua call and break the modified-state comparison.
    config.value[item.id] = Number(val)
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
          // Merge, then STRIP keys the schema no longer knows: settings
          // retired from the shader (the legacy flicker padding, the
          // variance scoping, the luma drift pair, the smear rejection
          // trio, the drift-correction pair, the ghost detector) stop
          // round-tripping through the config and the settings file
          // instead of resurrecting on every load.
          const merged = { ...defaultSettings, ...state.settings }
          config.value = Object.fromEntries(
            Object.entries(merged).filter(([k]) => k in defaultSettings)
          )
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

  .bng-reset-btn {
    --bng-button-padding-x: 0.3em;
    --bng-button-padding-y: 0.15em;
    opacity: 0.8;
    &:hover { opacity: 1; }
  }
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
    .header-controls { flex: 0 0 auto; display: flex; align-items: center; gap: 0.75em; }
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