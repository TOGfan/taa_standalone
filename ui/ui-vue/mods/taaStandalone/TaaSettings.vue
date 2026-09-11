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
      { id: 'sharpness', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.25, name: "RCAS Sharpening Base", desc: "Contrast-adaptive sharpening (RCAS) applied to the final resolved image." },
      { id: 'jitterScale', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 1.0, name: "Jitter Spread Scale", desc: "Multiplier for the sub-pixel camera offset." },
      { id: 'fallbackFXAA', type: 'numBool', default: 1.0, name: "Fallback Spatial AA (FXAA)", desc: "Applies FXAA to pixels where temporal history was rejected." }
    ]
  },
  {
    name: "History & Blending",
    items: [
      { id: 'feedbackMax', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "Static Blend Weight", desc: "Determines how much history is kept for stationary objects." },
      { id: 'feedbackMin', type: 'float', min: 0.0, max: 0.99, step: 0.01, default: 0.97, name: "Motion Blend Weight", desc: "Determines how much history is kept for moving objects." },
      { id: 'motionBlendStart', type: 'float', min: 0.0, max: 5.0, step: 0.01, default: 1.0, name: "Motion Blend Start Velocity", desc: "Minimum velocity threshold before starting to lower blend weight." },
      { id: 'motionBlendDropSpeed', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 1.0, name: "Motion Blend Drop Velocity", desc: "The pixel velocity magnitude required to transition fully from static to motion blend weights." },
      { id: 'alignmentFeedbackDrop', type: 'float', min: 0.5, max: 1.0, step: 0.01, default: 0.8, name: "Sub-Pixel Alignment Blend Drop", desc: "Multiplier applied to the temporal blend weight based on sub-pixel misalignment." }
    ]
  },
  {
    name: "Jitter & Sampling",
    items: [
      { id: 'useJitter', type: 'bool', default: true, name: "Enable Sub-Pixel Camera Jitter", desc: "Shifts the camera projection matrix by a sub-pixel offset each frame to sample missing geometry." },
      { id: 'useR2Jitter', type: 'bool', default: true, name: "R2 Jitter Sequence", desc: "Uses the R2 low-discrepancy sequence instead of the Halton sequence." },
      { id: 'useLanczos3', type: 'numBool', default: 1.0, name: "High-Quality Lanczos 3 Resampling", desc: "Uses high-quality Lanczos 3 resampling for history accumulation." },
      { id: 'useDepthDilation', type: 'numBool', default: 1.0, name: "Depth-Dilated Motion Search", desc: "Uses a depth-tested neighborhood search to find the closest foreground motion vector." }
    ]
  },
  {
    name: "Variance Clipping",
    items: [
      { id: 'useKDopClipping', type: 'numBool', default: 1.0, name: "k-DOP Neighborhood Clipping", desc: "Uses k-Discrete Oriented Polytopes (k-DOPs) for color neighborhood clipping." },
      { id: 'kdopVarianceClipping', type: 'numBool', default: 1.0, name: "Use k-DOP Variance Mode", desc: "Calculates k-DOP extents based on statistical variance rather than absolute min/max." },
      { id: 'useCovarianceClipping', type: 'numBool', default: 1.0, name: "Covariance Clipping (OBB)", desc: "Computes a 3x3 covariance matrix to orient the color bounding box." },
      { id: 'roundedAABB', type: 'numBool', default: 0.0, name: "Anti-Aliased Variance Bounds", desc: "Averages the bounds of the 5-pixel cross and 9-pixel box for variance clipping." },
      { id: 'colorSpaceOklab', type: 'numBool', default: 1.0, name: "Oklab Color Space", desc: "Converts samples to Oklab color space before clipping instead of YCoCg." },
      { id: 'varianceGamma', type: 'float', min: 0.0, max: 3.0, step: 0.01, default: 1.25, name: "Base Variance Gamma", desc: "Scaling factor for the variance bounding box size." },
      { id: 'chromaVarianceMod', type: 'float', min: 0.5, max: 2.0, step: 0.01, default: 1.0, name: "Chroma Variance Looseness", desc: "Multiplier applied to the chrominance axes of the variance bounding box." },
      { id: 'softClip', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.0, name: "Static Soft Clip Strength", desc: "Blend factor between the history color and the variance clip box during static scenes." }
    ]
  },
  {
    name: "Motion & Anti-Flicker",
    items: [
      { id: 'jitterFlickerPadding', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.0, name: "Jitter Anti-Flicker Padding", desc: "Expands the variance bounding box proportionally to the current sub-pixel jitter offset." },
      { id: 'directionalVariance', type: 'numBool', default: 1.0, name: "Color-Space Directional Padding", desc: "Expands the variance bounds exclusively along the color vector of the sub-pixel offset." },
      { id: 'jitterFlickerFade', type: 'numBool', default: 0.0, name: "Fade Jitter Padding in Motion", desc: "Attenuates the jitter anti-flicker padding when pixel velocity increases." },
      { id: 'adaptiveVariance', type: 'numBool', default: 0.0, name: "Adaptive Motion Ghosting Reduction", desc: "Contracts the variance bounding box dynamically based on pixel velocity." },
      { id: 'lumaVariance', type: 'numBool', default: 0.0, name: "Luma-Weighted Variance", desc: "Reduces the weight of samples based on their luminance (Karis Average)." },
      { id: 'jitterAwareVariance', type: 'numBool', default: 1.0, name: "Jitter-Aware Variance", desc: "Centers the variance bounding box around the sub-pixel camera jitter offset." },
      { id: 'velocityAlignedVariance', type: 'numBool', default: 0.0, name: "Velocity-Aligned Trailing Rejection", desc: "Discards neighborhood samples that fall behind the current motion vector when calculating variance." },
      { id: 'adaptiveVarStart', type: 'float', min: 0.1, max: 5.0, step: 0.01, default: 0.5, name: "Adaptive Var Motion Start", desc: "Pixel velocity magnitude at which adaptive variance begins to engage." },
      { id: 'adaptiveVarEnd', type: 'float', min: 1.0, max: 10.0, step: 0.01, default: 2.0, name: "Adaptive Var Motion End", desc: "Pixel velocity magnitude at which adaptive variance reaches its maximum effect." }
    ]
  },
  {
    name: "Shadows & SSAO Mitigation",
    items: [
      { id: 'shadowMitigation', type: 'numBool', default: 0.0, name: "Enable Shadow & SSAO Mitigation", desc: "Detects and blurs flickering in shadows and ambient occlusion." },
      { id: 'shadowDarknessThreshold', type: 'float', min: 0.05, max: 0.8, step: 0.01, default: 0.25, name: "Shadow Luma Threshold", desc: "The luma threshold below which a pixel is considered a shadow." },
      { id: 'shadowBlendStrength', type: 'float', min: 0.5, max: 0.99, step: 0.01, default: 0.95, name: "Shadow Smoothing Strength", desc: "The interpolation factor for the shadow smoothing pass." },
      { id: 'shadowTemporalMult', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 10.0, name: "Shadow Temporal Risk Mult", desc: "Multiplier for temporal variance when assessing shadow stability." },
      { id: 'shadowSpatialMult', type: 'float', min: 1.0, max: 20.0, step: 0.1, default: 5.0, name: "Shadow Spatial Safety Mult", desc: "Multiplier for spatial variance when applying shadow mitigation." },
      { id: 'shadowVarianceBase', type: 'float', min: 0.0, max: 2.0, step: 0.01, default: 0.2, name: "Shadow Variance Base", desc: "Minimum variance gamma forced inside shadow areas." }
    ]
  },
  {
    name: "Advanced Rejection",
    items: [
      { id: 'depthRejection', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 1.0, name: "Depth Mismatch Rejection", desc: "Relative depth discontinuity threshold beyond local geometric slope." },
      { id: 'clipDistanceRejectionEnabled', type: 'numBool', default: 0.0, name: "Clip-Distance Smear Rejection", desc: "Detects severe history clipping distances and drops history weight." },
      { id: 'clipDistanceRejectionAmount', type: 'float', min: 0.0, max: 1.0, step: 0.001, default: 0.0, name: "Smear Rejection Tolerance", desc: "Color clipping tolerance before history is fully rejected." },
      { id: 'clipDistanceRejectionMinError', type: 'float', min: 0.001, max: 0.5, step: 0.001, default: 0.05, name: "Clip Rejection Min Error", desc: "Minimum color divergence required before initiating smear rejection." },
      { id: 'fireflyClamp', type: 'float', min: 1.0, max: 10.0, step: 0.1, default: 4.0, name: "Firefly Clamp Threshold", desc: "Standard deviation threshold for clamping high-luminance outliers." },
      { id: 'collapseRatioMin', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.05, name: "Collapse Ratio Min", desc: "Lower threshold of neighborhood collapse ratio for adaptive contraction." },
      { id: 'collapseRatioMax', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.35, name: "Collapse Ratio Max", desc: "Upper threshold of neighborhood collapse ratio for adaptive contraction." }
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
        desc: "Visualizes internal buffers and rejection masks.\n\n0: Off (Normal)\n1: Pixel Motion Vectors\n2: Raw History Buffer\n3: Disocclusion Mask\n4: Linear Depth (Normalized 100m)\n5: Shadow Risk Factor\n6: Confidence Heatmap\n7: Accumulation Blend Weight",
        options: [
          { label: 'Off (Normal Rendering)', value: 0.0 },
          { label: 'Motion Vectors', value: 1.0 },
          { label: 'History Buffer', value: 2.0 },
          { label: 'Disocclusion Mask', value: 3.0 },
          { label: 'Linearized Depth', value: 4.0 },
          { label: 'Shadow Risk Proxy', value: 5.0 },
          { label: 'Confidence Heatmap', value: 6.0 },
          { label: 'Final Blend Weight', value: 7.0 }
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
  Performance: { useLanczos3: 0, useKDopClipping: 0, colorSpaceOklab: 0, kdopVarianceClipping: 0, useCovarianceClipping: 0 },
  Balanced: { useKDopClipping: 0, colorSpaceOklab: 0, kdopVarianceClipping: 0, useCovarianceClipping: 0 }, 
  Clarity: { },
  Smooth: { feedbackMax: 0.99, feedbackMin: 0.99, alignmentFeedbackDrop: 0.9, depthRejection: 0.0 }
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