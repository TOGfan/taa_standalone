import { lua } from "@/bridge"

export async function onLoad() {
  // Registers a button in the shared "Mods" tab of the Pause Menu
  await lua.extensions.ui_pause_actions.registerModButton({
    id: "taa-standalone-settings",
    tabId: "mods",
    label: "TAA Settings",
    icon: "video", // Uses built-in BeamNG icon
    componentName: "/ui/ui-vue/mods/taaStandalone/TaaSettings.vue",
  })
}

export async function onUnload() {
  await lua.extensions.ui_pause_actions.unregisterModButton("taa-standalone-settings")
}