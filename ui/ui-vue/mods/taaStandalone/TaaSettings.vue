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
              <div class="option-label">{{ item.name }}</div>

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
                :modelValue="config[item.id]"
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
                  :style="{ visibility: config[item.id] !== item.default ? 'visible' : 'hidden' }"
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

const settingsSchema = [
  {
    name: "General & Sharpening",
    items: [
      { id: 'sharpness', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.50, name: "Sharpening Strength (RCAS)", desc: "Contrast-adaptive sharpening applied to the final resolved image, clamped to the local pixel range so it cannot overshoot or ring. 0 disables." },
      { id: 'jitterScale', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 1.0, name: "Jitter Spread Scale", desc: "Scales the sub-pixel camera offset pattern. Above 1 samples a wider area within each pixel (more edge anti-aliasing, more temporal softening); below 1 tightens it." },
      { id: 'fallbackFXAA', type: 'numBool', default: 1.0, name: "Fallback Spatial AA (FXAA)", desc: "Applies FXAA edge smoothing to pixels whose temporal history was rejected, so disoccluded areas don't alias while the history rebuilds." }
    ]
  },
  {
    name: "History & Blending",
    items: [
      { id: 'feedbackMax', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Static Scenes)", desc: "How much of the previous frame is reused for stationary pixels. Higher = smoother and cleaner but slower to react to changes; lower = more responsive but noisier. 0.97 corresponds to roughly a 33-frame accumulation window." },
      { id: 'feedbackMin', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "History Blend (Moving Pixels)", desc: "History reuse once pixel velocity exceeds the motion transition range below. Typically set at or below the static value so motion receives less smoothing." },
      { id: 'lumaDriftStrength', type: 'float', min: 0.0, max: 0.3, step: 0.01, default: 0.0, name: "Luma Drift Correction", desc: "Pulls history brightness toward the current image to clear ghost trails from moving shadows, exposure changes and vehicle lights, without reducing temporal smoothing. 0 = off. Only engages on large brightness mismatches on the same surface (see the chroma gate below), so it does not chase noise, jitter or ghosts. Higher values clear trails faster." },
      { id: 'lumaDriftChromaTol', type: 'float', min: 0.0, max: 0.5, step: 0.01, default: 0.1, name: "Drift Same-Surface Tolerance", desc: "Color-match tolerance for the luma drift correction, comparing color-per-brightness so that pure lighting changes (shadows, exposure) pass while a different surface does not. History from a differently-colored object (ghosts, reveal edges) fails this test and is left to the normal rejection paths -- drift only ever corrects brightness, never disguises color mismatches. Raise if legitimate shadows aren't being corrected; lower if colored ghosts linger. 0 requires an exact match (drift effectively off)." },
      { id: 'motionBlendStart', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Motion Transition Start", desc: "Pixel velocity (in pixels per frame) at which blending starts transitioning from the static to the motion weight." },
      { id: 'motionBlendDropSpeed', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 1.0, name: "Motion Transition End", desc: "Pixel velocity at which the blend reaches the motion weight fully." },
      { id: 'alignmentFeedbackDrop', type: 'float', min: 0.5, max: 1.0, step: 0.01, default: 0.9, name: "Sub-Pixel Alignment Drop", desc: "Reduces history weight when the reprojected sample lands between pixels (sub-pixel misalignment), where the resampling kernel is least accurate. 1.0 disables the reduction." }
    ]
  },
  {
    name: "Jitter & Sampling",
    items: [
      { id: 'useJitter', type: 'bool', default: true, name: "Sub-Pixel Camera Jitter", desc: "Offsets the camera by a sub-pixel amount each frame so successive frames sample different positions within each pixel. This is the source of TAA's supersampling -- without it, TAA only stabilizes noise, it does not anti-alias." },
      { id: 'useR2Jitter', type: 'bool', default: true, name: "R2 Jitter Sequence", desc: "Uses the R2 low-discrepancy sequence instead of Halton(2,3) for the jitter pattern. R2 spreads samples more evenly across its 32-frame cycle." },
      { id: 'useLanczos3', type: 'numBool', default: 1.0, name: "Lanczos 3 History Resampling", desc: "36-tap Lanczos resampling of the history buffer. Retains noticeably more detail in motion than the 5-tap Catmull-Rom fallback (the two differ only for moving pixels), at roughly 7x the sampling cost." },
      { id: 'historyOvershoot', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 1.0, name: "History Resampling Overshoot Margin", desc: "Soft anti-ringing margin for history resampling, shared by both the Lanczos and Catmull-Rom paths and applied equally to brightness and color, as a fraction of the local color range. Band-limited overshoot -- the edge detail the filter reconstructs -- survives, while ringing and color fringing are compressed. Higher = sharper history, lower = fewer halos." },
      { id: 'useDepthDilation', type: 'numBool', default: 1.0, name: "Depth-Dilated Motion Search", desc: "Searches the 3x3 neighborhood for the closest surface and uses its motion vector, so silhouette edges reproject with the foreground's motion instead of the background's." }
    ]
  },
  {
    name: "Variance Clipping",
    items: [
      { id: 'useKDopClipping', type: 'numBool', default: 1.0, name: "k-DOP History Clipping", desc: "Clips history against a 16-direction convex hull of the neighborhood colors instead of a simple bounding box. A tighter bound on valid history at higher cost." },
      { id: 'kdopVarianceClipping', type: 'numBool', default: 0.0, name: "k-DOP Variance Extents", desc: "Builds the k-DOP bounds from statistical variance instead of the absolute min/max of the neighborhood -- more forgiving of single outlier samples." },
      { id: 'useCovarianceClipping', type: 'numBool', default: 0.0, name: "Covariance Clipping (Ellipsoid)", desc: "Clips history against an ellipsoid fit of the neighborhood color distribution. Only active when k-DOP clipping is disabled -- the k-DOP path supersedes it." },
      { id: 'colorSpaceOklab', type: 'numBool', default: 1.0, name: "Oklab Clipping Color Space", desc: "Performs history clipping in Oklab (perceptually uniform) instead of YCoCg. Slightly higher GPU cost." },
      { id: 'varianceGamma', type: 'float', min: 0.0, max: 3.0, step: 0.01, default: 1.50, name: "Variance Box Scale", desc: "Scales the size of the color bounds that clamp history. Higher = looser clamp (history survives more, with more smear potential); lower = tighter (sharper, but more history rejection)." },
      { id: 'chromaVarianceMod', type: 'float', min: 0.5, max: 2.0, step: 0.01, default: 1.0, name: "Chroma Bounds Scale", desc: "Independent multiplier for the color (non-brightness) axes of the variance bounds." },
      { id: 'softClip', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.0, name: "Soft Clip Strength", desc: "Eases history into the variance bounds instead of snapping hard, in near-static scenes. Reduces clipping 'popping' at the cost of a slight ghost linger; fades out with motion." },
      { id: 'clipOvershoot', type: 'float', min: 0.0, max: 0.5, step: 0.01, default: 0.0, name: "Clip Overshoot Margin", desc: "Lets accumulated history exceed the neighborhood color bounds by this fraction of the local color range. The resampling kernel's negative lobes reconstruct edges steeper than any single frame's samples; a strict per-frame bound flattens that reconstruction. With a margin, frame-consistent edge overshoot accumulates (sharper edges over multiple reprojections) while frame-inconsistent ringing averages away -- at the cost of some visible ringing on thin high-contrast geometry. On the k-DOP path this margin plus the hull is the entire color bound (there is no outer safety clamp). Raise if fine detail looks soft; lower if edges halo, crawl, or ghosts linger. If ghost trails lengthen, enable Smear Rejection under Advanced Rejection." }
    ]
  },
  {
    name: "Motion & Anti-Flicker",
    items: [
      { id: 'jitterFlickerPadding', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.0, name: "Jitter Anti-Flicker Padding", desc: "Expands the variance bounds in proportion to the current sub-pixel jitter offset, so valid history isn't clipped away purely because of jitter phase. Reduces shimmer in fine detail." },
      { id: 'directionalVariance', type: 'numBool', default: 1.0, name: "Directional Padding", desc: "Expands the bounds along the color direction of the expected jitter shift rather than uniformly. Only active when Jitter Anti-Flicker Padding is above zero." },
      { id: 'jitterFlickerFade', type: 'numBool', default: 0.0, name: "Fade Padding in Motion", desc: "Disables the jitter anti-flicker padding as pixel velocity rises, since reprojection error dominates jitter error in motion." },
      { id: 'lumaVariance', type: 'numBool', default: 0.0, name: "Luma-Weighted Statistics", desc: "Downweights bright samples when computing neighborhood statistics (Karis-style), keeping the bounds from being stretched by specular fireflies." },
      { id: 'jitterAwareVariance', type: 'numBool', default: 1.0, name: "Jitter-Aware Statistics", desc: "Weights neighborhood samples by their distance from the jittered sampling position instead of the pixel center." },
      { id: 'velocityAlignedVariance', type: 'numBool', default: 0.0, name: "Velocity-Aligned Statistics", desc: "Downweights neighborhood samples that lie behind the direction of motion, tightening the bounds along motion trails." }
    ]
  },
  {
    name: "Shadows & SSAO Mitigation",
    items: [
      { id: 'shadowMitigation', type: 'numBool', default: 0.0, name: "Shadow & SSAO Flicker Mitigation", desc: "Detects flickering shadows and ambient occlusion and reduces their history weight so they settle faster, at the cost of some temporal smoothing in dark areas. Note: Luma Drift Correction (History & Blending) clears stuck shadows without that trade -- try it first." },
      { id: 'shadowDarknessThreshold', type: 'float', min: 0.05, max: 0.8, step: 0.01, default: 0.25, name: "Shadow Detection Threshold", desc: "Brightness below this is treated as shadow for mitigation purposes." },
      { id: 'shadowBlendStrength', type: 'float', min: 0.5, max: 0.99, step: 0.01, default: 0.95, name: "Shadow History Weight", desc: "History weight applied inside detected unstable shadows. Lower = shadows respond faster to change but flicker more." },
      { id: 'shadowTemporalMult', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 10.0, name: "Shadow Temporal Risk Mult", desc: "Sensitivity of the temporal brightness-change test that flags unstable shadows." },
      { id: 'shadowSpatialMult', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 5.0, name: "Shadow Spatial Safety Mult", desc: "How strongly spatial texture suppresses shadow mitigation -- textured dark areas are left alone." },
      { id: 'shadowVarianceBase', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 0.2, name: "Shadow Variance Floor", desc: "Minimum variance scale forced inside unstable shadows, loosening the clamp so dark detail isn't crushed." }
    ]
  },
  {
    name: "Advanced Rejection",
    items: [
      { id: 'depthRejection', type: 'float', min: 0.0, max: 0.10, step: 0.001, default: 0.01, name: "Disocclusion Sensitivity (Depth)", desc: "Threshold of the geometry-based disocclusion test: how much closer than every surface in the 1-pixel dilation zone the history must be before it is rejected as stale. 0 disables depth rejection entirely." },
      { id: 'velRejection', type: 'float', min: 0.0, max: 10.0, step: 0.1, default: 1.5, name: "Disocclusion Threshold (Dilated Velocity)", desc: "Threshold (in pixels) for velocity consistency rejection using 3x3 dilated motion vectors (both current and historical). Rejects history if the historical dilated velocity differs from current motion, eliminating ghosts behind accelerating objects or rotating wheels. 0 disables." },
      { id: 'clipDistanceRejectionEnabled', type: 'numBool', default: 1.0, name: "Smear Rejection", desc: "Drops history weight where the clip hull had to move the history a long way -- a ghosting indicator for content without motion vectors (animated textures, particles)." },
      { id: 'clipDistanceRejectionAmount', type: 'float', min: 0.0, max: 1.0, step: 0.001, default: 0.0, name: "Smear Rejection Tolerance", desc: "How far the clip distance must exceed the minimum error before history is fully rejected." },
      { id: 'clipDistanceRejectionMinError', type: 'float', min: 0.001, max: 0.5, step: 0.001, default: 0.15, name: "Smear Rejection Min Error", desc: "Minimum clip distance before smear rejection begins to engage." },
      { id: 'fireflyClamp', type: 'float', min: 1.0, max: 10.0, step: 0.1, default: 4.0, name: "Firefly Clamp", desc: "Clamps the variance bounds against extreme bright outliers, in standard deviations of the neighborhood. Lower = tighter (fewer fireflies, more clipping of legitimate highlights)." }
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
        desc: "Visualizes internal buffers and rejection masks.\n\n0: Off (Normal)\n1: Pixel Motion Vectors\n2: Raw History Buffer\n3: Disocclusion Mask (Red=Depth, Cyan=Dilated Vel, Yellow=Both)\n4: Linear Depth (Normalized 100m)\n5: Shadow Risk Factor\n6: Historical Dilated Velocity (Diagnostics)\n7: Accumulation Blend Weight\n8: Velocity States (Red=Dilation, Cyan=Foreground Edge, Dark=Continuous)",
        options: [
          { label: 'Off (Normal Rendering)', value: 0.0 },
          { label: 'Motion Vectors', value: 1.0 },
          { label: 'History Buffer', value: 2.0 },
          { label: 'Disocclusion Mask (Red:Depth, Cyan:Vel, Yel:Both)', value: 3.0 },
          { label: 'Linearized Depth', value: 4.0 },
          { label: 'Shadow Risk Proxy', value: 5.0 },
          { label: 'Historical Dilated Velocity (Diagnostics)', value: 6.0 },
          { label: 'Final Blend Weight', value: 7.0 },
          { label: 'Velocity Classification (States / Dilation)', value: 8.0 }
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
  Performance: { feedbackMax: 0.95, feedbackMin: 0.95, useLanczos3: 0, useKDopClipping: 0, colorSpaceOklab: 0 },
  Balanced: { useKDopClipping: 0, colorSpaceOklab: 0}, 
  Clarity: { feedbackMax: 0.95, feedbackMin: 0.95, },
  Smooth: { }
}

function formatDefault(item) {
  if (item.type === 'bool' || item.type === 'numBool') {
    return (item.default === 1 || item.default === true) ? 'Enabled' : 'Disabled'
  }
  if (item.type === 'select') {
    const opt = item.options.find(o => o.value === item.default)
    return opt ? opt.label : item.default
  }
  return item.default
}

function resetSetting(item) {
  config.value[item.id] = item.default
  updateSetting(item.id)
}

function onSliderChange(item, val) {
  config.value[item.id] = val
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
  for (const key in presetOverrides) { config.value[key] = presetOverrides[key] }

  if (!window.bngApi || !window.bngApi.engineLua) return
  let script = "local t = taa or taa_taa; if t then\n"
  for (const key in config.value) {
    let val = config.value[key]
    let luaVal = typeof val === 'boolean' ? (val ? 'true' : 'false') : val
    script += `t.uiSetSetting("${key}", ${luaVal})\n`
  }
  script += "end"
  window.bngApi.engineLua(script)
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

function updateSetting(key) {
  let val = config.value[key]
  let luaVal = typeof val === 'boolean' ? (val ? 'true' : 'false') : val
  if (window.bngApi && window.bngApi.engineLua) {
    window.bngApi.engineLua(`if taa then taa.uiSetSetting("${key}", ${luaVal}) elseif taa_taa then taa_taa.uiSetSetting("${key}", ${luaVal}) end`)
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