-- lua/ge/client/postFx/taa.lua

local TAA_ChainPostFX = {}

local taaStateBlock = scenetree.findObject("TAA_StateBlock")
if not taaStateBlock then
  taaStateBlock = createObject("GFXStateBlockData")
  taaStateBlock.zDefined = true; taaStateBlock.zEnable = false; taaStateBlock.zWriteEnable = false
  taaStateBlock.blendDefined = true; taaStateBlock:setField("blendSrc", 0, "GFXBlendOne"); taaStateBlock:setField("blendDest", 0, "GFXBlendZero")
  taaStateBlock.cullDefined = true; taaStateBlock:setField("cullMode", 0, "GFXCullNone")
  taaStateBlock.samplersDefined = true
  taaStateBlock:setField("samplerStates", 0, "SamplerClampPoint"); taaStateBlock:setField("samplerStates", 1, "SamplerClampPoint")
  taaStateBlock:setField("samplerStates", 2, "SamplerClampLinear"); taaStateBlock:setField("samplerStates", 3, "SamplerClampPoint")
  taaStateBlock:registerObject("TAA_StateBlock")
end

local taaCopyStateBlock = scenetree.findObject("TAA_Copy_StateBlock")
if not taaCopyStateBlock then
  taaCopyStateBlock = createObject("GFXStateBlockData")
  taaCopyStateBlock.zDefined = true; taaCopyStateBlock.zEnable = false; taaCopyStateBlock.zWriteEnable = false
  taaCopyStateBlock.blendDefined = true; taaCopyStateBlock:setField("blendSrc", 0, "GFXBlendOne"); taaCopyStateBlock:setField("blendDest", 0, "GFXBlendZero")
  taaCopyStateBlock.cullDefined = true; taaCopyStateBlock:setField("cullMode", 0, "GFXCullNone")
  taaCopyStateBlock.samplersDefined = true; taaCopyStateBlock:setField("samplerStates", 0, "SamplerClampLinear")
  taaCopyStateBlock:registerObject("TAA_Copy_StateBlock")
end

-- Hardcoded standalone shader paths
local taaShaderResolve = scenetree.findObject("TAA_Resolve_ShaderData")
if not taaShaderResolve then 
  taaShaderResolve = createObject("ShaderData")
  taaShaderResolve.DXVertexShaderFile = "shaders/common/postFx/taa/taa.fx.hlsl"
  taaShaderResolve.DXPixelShaderFile  = "shaders/common/postFx/taa/taa.fx.hlsl"
  taaShaderResolve.pixVersion = 5.0
  taaShaderResolve:registerObject("TAA_Resolve_ShaderData") 
end

local taaShaderCopy = scenetree.findObject("TAA_Copy_ShaderData")
if not taaShaderCopy then 
  taaShaderCopy = createObject("ShaderData")
  taaShaderCopy.DXVertexShaderFile = "shaders/common/postFx/taa/taaCopy.fx.hlsl"
  taaShaderCopy.DXPixelShaderFile  = "shaders/common/postFx/taa/taaCopy.fx.hlsl"
  taaShaderCopy.pixVersion = 5.0
  taaShaderCopy:registerObject("TAA_Copy_ShaderData") 
end

local taaShaderFinal = scenetree.findObject("TAA_Final_ShaderData")
if not taaShaderFinal then 
  taaShaderFinal = createObject("ShaderData")
  taaShaderFinal.DXVertexShaderFile = "shaders/common/postFx/taa/taaFinal.fx.hlsl"
  taaShaderFinal.DXPixelShaderFile  = "shaders/common/postFx/taa/taaFinal.fx.hlsl"
  taaShaderFinal.pixVersion = 5.0
  taaShaderFinal:registerObject("TAA_Final_ShaderData") 
end

local taaPreFx = scenetree.findObject("TAA_PreFx")
if not taaPreFx then
  taaPreFx = createObject("PostEffect")
  taaPreFx.isEnabled = false; taaPreFx.allowReflectPass = false
  taaPreFx:setField("renderTime", 0, "PFXBeforeBin"); taaPreFx:setField("renderBin", 0, "EditorBin"); taaPreFx.renderPriority = 0.1
  taaPreFx:setField("shader", 0, "TAA_Resolve_ShaderData"); taaPreFx:setField("stateBlock", 0, "TAA_StateBlock")
  taaPreFx:setField("targetScale", 0, "1.0 1.0")

  taaPreFx:setField("texture", 0, "$backBuffer"); taaPreFx:setField("texture", 1, "#prepass[Depth]")
  taaPreFx:setField("texture", 2, "#TAA_History"); taaPreFx:setField("texture", 3, "#velocitybuffer")
  taaPreFx:setField("target", 0, "#TAA_Result"); taaPreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaPreFx:setField("targetClear", 0, "PFXTargetClear_OnDraw")

  local taaFinalFx = createObject("PostEffect")
  taaFinalFx:setField("shader", 0, "TAA_Final_ShaderData"); taaFinalFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
  taaFinalFx:setField("texture", 0, "#TAA_Result"); taaFinalFx:setField("texture", 1, "#velocitybuffer")
  taaFinalFx:setField("texture", 2, "#TAA_History"); taaFinalFx:setField("texture", 3, "#prepass[Depth]")
  taaFinalFx:setField("target", 0, "$backBuffer")
  taaFinalFx:registerObject("TAA_FinalFx"); taaPreFx:add(taaFinalFx)

  local taaStoreFx = createObject("PostEffect")
  taaStoreFx:setField("shader", 0, "TAA_Copy_ShaderData"); taaStoreFx:setField("stateBlock", 0, "TAA_Copy_StateBlock")
  taaStoreFx:setField("targetScale", 0, "1.0 1.0")
  taaStoreFx:setField("texture", 0, "#TAA_Result"); taaStoreFx:setField("target", 0, "#TAA_History")
  taaStoreFx:setField("targetFormat", 0, "GFXFormatR32G32B32A32F"); taaStoreFx:setField("targetClear", 0, "PFXTargetClear_None")
  taaStoreFx:registerObject("TAA_StoreFx"); taaPreFx:add(taaStoreFx)

  taaPreFx:registerObject("TAA_PreFx")
end

-- Default Settings (used immediately upon load)
TAA_ChainPostFX.useJitter                 = true
TAA_ChainPostFX.useR2Jitter               = true
TAA_ChainPostFX.jitterScale               = 1.0
TAA_ChainPostFX.feedbackMin               = 0.99
TAA_ChainPostFX.feedbackMax               = 0.99
TAA_ChainPostFX.shadowMitigation          = 0.0
TAA_ChainPostFX.shadowDarknessThreshold   = 0.25
TAA_ChainPostFX.shadowBlendStrength       = 0.95
TAA_ChainPostFX.varianceGamma             = 1.25
TAA_ChainPostFX.softClip                  = 0.0
TAA_ChainPostFX.chromaVarianceMod         = 1.0
TAA_ChainPostFX.jitterFlickerPadding      = 0.0
TAA_ChainPostFX.directionalVariance       = 1.0
TAA_ChainPostFX.jitterFlickerFade         = 0.0
TAA_ChainPostFX.depthRejection            = 0.0
TAA_ChainPostFX.velDisocclusion           = 0.3
TAA_ChainPostFX.sharpness                 = 0.25
TAA_ChainPostFX.adaptiveSharp             = 0.0
TAA_ChainPostFX.debugMode                 = 0.0
TAA_ChainPostFX.useDepthDilation          = 1.0
TAA_ChainPostFX.adaptiveVariance          = 0.0
TAA_ChainPostFX.lumaVariance              = 0.0
TAA_ChainPostFX.useKDopClipping           = 1.0
TAA_ChainPostFX.kdopVarianceClipping      = 1.0
TAA_ChainPostFX.useCovarianceClipping     = 1.0
TAA_ChainPostFX.colorSpaceOklab           = 1.0
TAA_ChainPostFX.jitterAwareVariance       = 1.0
TAA_ChainPostFX.velocityAlignedVariance   = 0.0
TAA_ChainPostFX.alignmentFeedbackDrop     = 0.9
TAA_ChainPostFX.alignmentRCASBoost        = 0.0
TAA_ChainPostFX.motionBlendDropSpeed      = 1.0
TAA_ChainPostFX.bilinearHistoryVel        = 1.0
TAA_ChainPostFX.roundedAABB               = 0.0
TAA_ChainPostFX.useLanczos3               = 1.0
TAA_ChainPostFX.fireflyClamp              = 4.0
TAA_ChainPostFX.adaptiveVarStart          = 0.5
TAA_ChainPostFX.adaptiveVarEnd            = 2.0
TAA_ChainPostFX.flickerSpatialMult        = 15.0
TAA_ChainPostFX.shadowTemporalMult        = 10.0
TAA_ChainPostFX.shadowSpatialMult         = 5.0
TAA_ChainPostFX.clipDistanceRejectionEnabled  = 0.0
TAA_ChainPostFX.clipDistanceRejectionAmount   = 0.0
TAA_ChainPostFX.clipDistanceRejectionMinError = 0.05
TAA_ChainPostFX.depthRejRelStatic         = 0.1
TAA_ChainPostFX.depthRejRelMoving         = 0.02
TAA_ChainPostFX.depthRejAbs               = 0.01
TAA_ChainPostFX.velRejBaseStatic          = 0.5
TAA_ChainPostFX.velRejBaseMoving          = 0.05
TAA_ChainPostFX.velRejMotionScale         = 0.5
TAA_ChainPostFX.motionBlendStart          = 1.0
TAA_ChainPostFX.shadowVarianceBase        = 0.2
TAA_ChainPostFX.collapseRatioMin          = 0.05
TAA_ChainPostFX.collapseRatioMax          = 0.35
TAA_ChainPostFX.fallbackFXAA              = 1.0

TAA_ChainPostFX.DEFAULTS = deepcopy(TAA_ChainPostFX)

local function shaderConstsActual()
  if scenetree.TAA_PreFx then
    scenetree.TAA_PreFx:setShaderConst("$taaFeedbackMin",             TAA_ChainPostFX.feedbackMin)
    scenetree.TAA_PreFx:setShaderConst("$taaFeedbackMax",             TAA_ChainPostFX.feedbackMax)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowMitigation",        TAA_ChainPostFX.shadowMitigation)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowDarknessThreshold", TAA_ChainPostFX.shadowDarknessThreshold)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowBlendStrength",     TAA_ChainPostFX.shadowBlendStrength)
    scenetree.TAA_PreFx:setShaderConst("$taaVarianceGamma",           TAA_ChainPostFX.varianceGamma)
    scenetree.TAA_PreFx:setShaderConst("$taaSoftClip",                TAA_ChainPostFX.softClip)
    scenetree.TAA_PreFx:setShaderConst("$taaChromaVarianceMod",       TAA_ChainPostFX.chromaVarianceMod)
    scenetree.TAA_PreFx:setShaderConst("$taaJitterFlickerPadding",    TAA_ChainPostFX.jitterFlickerPadding)
    scenetree.TAA_PreFx:setShaderConst("$taaDirectionalVariance",     TAA_ChainPostFX.directionalVariance)
    scenetree.TAA_PreFx:setShaderConst("$taaJitterFlickerFade",       TAA_ChainPostFX.jitterFlickerFade)
    scenetree.TAA_PreFx:setShaderConst("$taaDepthRejection",          TAA_ChainPostFX.depthRejection)
    scenetree.TAA_PreFx:setShaderConst("$taaVelDisocclusion",         TAA_ChainPostFX.velDisocclusion)
    scenetree.TAA_PreFx:setShaderConst("$taaDebugMode",               TAA_ChainPostFX.debugMode)
    scenetree.TAA_PreFx:setShaderConst("$taaUseDepthDilation",        TAA_ChainPostFX.useDepthDilation)
    scenetree.TAA_PreFx:setShaderConst("$taaAdaptiveVariance",        TAA_ChainPostFX.adaptiveVariance)
    scenetree.TAA_PreFx:setShaderConst("$taaLumaVariance",            TAA_ChainPostFX.lumaVariance)
    scenetree.TAA_PreFx:setShaderConst("$taaUseKDopClipping",         TAA_ChainPostFX.useKDopClipping)
    scenetree.TAA_PreFx:setShaderConst("$taaKDopVariance",            TAA_ChainPostFX.kdopVarianceClipping)
    scenetree.TAA_PreFx:setShaderConst("$taaUseCovarianceClipping",   TAA_ChainPostFX.useCovarianceClipping)
    scenetree.TAA_PreFx:setShaderConst("$taaColorSpaceOklab",         TAA_ChainPostFX.colorSpaceOklab)
    scenetree.TAA_PreFx:setShaderConst("$taaJitterAwareVariance",     TAA_ChainPostFX.jitterAwareVariance)
    scenetree.TAA_PreFx:setShaderConst("$taaVelocityAlignedVariance", TAA_ChainPostFX.velocityAlignedVariance)
    scenetree.TAA_PreFx:setShaderConst("$taaAlignmentFeedbackDrop",   TAA_ChainPostFX.alignmentFeedbackDrop)
    scenetree.TAA_PreFx:setShaderConst("$taaMotionBlendDropSpeed",    TAA_ChainPostFX.motionBlendDropSpeed)
    scenetree.TAA_PreFx:setShaderConst("$taaBilinearHistoryVel",      TAA_ChainPostFX.bilinearHistoryVel)
    scenetree.TAA_PreFx:setShaderConst("$taaRoundedAABB",             TAA_ChainPostFX.roundedAABB)
    scenetree.TAA_PreFx:setShaderConst("$taaUseLanczos3",             TAA_ChainPostFX.useLanczos3)
    scenetree.TAA_PreFx:setShaderConst("$taaFireflyClamp",            TAA_ChainPostFX.fireflyClamp)
    scenetree.TAA_PreFx:setShaderConst("$taaAdaptiveVarStart",        TAA_ChainPostFX.adaptiveVarStart)
    scenetree.TAA_PreFx:setShaderConst("$taaAdaptiveVarEnd",          TAA_ChainPostFX.adaptiveVarEnd)
    scenetree.TAA_PreFx:setShaderConst("$taaFlickerSpatialMult",      TAA_ChainPostFX.flickerSpatialMult)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowTemporalMult",      TAA_ChainPostFX.shadowTemporalMult)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowSpatialMult",       TAA_ChainPostFX.shadowSpatialMult)
    scenetree.TAA_PreFx:setShaderConst("$taaClipDistanceRejectionEnabled",  TAA_ChainPostFX.clipDistanceRejectionEnabled)
    scenetree.TAA_PreFx:setShaderConst("$taaClipDistanceRejectionAmount",   TAA_ChainPostFX.clipDistanceRejectionAmount)
    scenetree.TAA_PreFx:setShaderConst("$taaClipDistanceRejectionMinError", TAA_ChainPostFX.clipDistanceRejectionMinError)
    scenetree.TAA_PreFx:setShaderConst("$taaDepthRejRelStatic",       TAA_ChainPostFX.depthRejRelStatic)
    scenetree.TAA_PreFx:setShaderConst("$taaDepthRejRelMoving",       TAA_ChainPostFX.depthRejRelMoving)
    scenetree.TAA_PreFx:setShaderConst("$taaDepthRejAbs",             TAA_ChainPostFX.depthRejAbs)
    scenetree.TAA_PreFx:setShaderConst("$taaVelRejBaseStatic",        TAA_ChainPostFX.velRejBaseStatic)
    scenetree.TAA_PreFx:setShaderConst("$taaVelRejBaseMoving",        TAA_ChainPostFX.velRejBaseMoving)
    scenetree.TAA_PreFx:setShaderConst("$taaVelRejMotionScale",       TAA_ChainPostFX.velRejMotionScale)
    scenetree.TAA_PreFx:setShaderConst("$taaMotionBlendStart",        TAA_ChainPostFX.motionBlendStart)
    scenetree.TAA_PreFx:setShaderConst("$taaShadowVarianceBase",      TAA_ChainPostFX.shadowVarianceBase)
    scenetree.TAA_PreFx:setShaderConst("$taaCollapseRatioMin",        TAA_ChainPostFX.collapseRatioMin)
    scenetree.TAA_PreFx:setShaderConst("$taaCollapseRatioMax",        TAA_ChainPostFX.collapseRatioMax)
    scenetree.TAA_PreFx:setShaderConst("$taaFallbackFXAA",            TAA_ChainPostFX.fallbackFXAA)
  end

  if scenetree.TAA_FinalFx then
    scenetree.TAA_FinalFx:setShaderConst("$taaSharpness",             TAA_ChainPostFX.sharpness)
    scenetree.TAA_FinalFx:setShaderConst("$taaAdaptiveSharp",         TAA_ChainPostFX.adaptiveSharp)
    scenetree.TAA_FinalFx:setShaderConst("$taaDebugMode",             TAA_ChainPostFX.debugMode)
    scenetree.TAA_FinalFx:setShaderConst("$taaUseDepthDilation",      TAA_ChainPostFX.useDepthDilation)
    scenetree.TAA_FinalFx:setShaderConst("$taaShadowDarknessThreshold", TAA_ChainPostFX.shadowDarknessThreshold)
    scenetree.TAA_FinalFx:setShaderConst("$taaDepthRejection",        TAA_ChainPostFX.depthRejection)
    scenetree.TAA_FinalFx:setShaderConst("$taaVelDisocclusion",       TAA_ChainPostFX.velDisocclusion)
    scenetree.TAA_FinalFx:setShaderConst("$taaAlignmentRCASBoost",    TAA_ChainPostFX.alignmentRCASBoost)
    scenetree.TAA_FinalFx:setShaderConst("$taaShadowTemporalMult",    TAA_ChainPostFX.shadowTemporalMult)
    scenetree.TAA_FinalFx:setShaderConst("$taaShadowSpatialMult",     TAA_ChainPostFX.shadowSpatialMult)
  end
end

TAA_ChainPostFX.setShaderConsts = function(inputs)
  if inputs and type(inputs) == "table" then tableMerge(TAA_ChainPostFX, inputs) end
  shaderConstsActual()
end

TAA_ChainPostFX.setEnabled = function(enabled)
  if scenetree.TAA_PreFx then
    if enabled then 
        scenetree.TAA_PreFx:enable() 
    else 
        scenetree.TAA_PreFx:disable()
        TAA_ChainPostFX.setShaderConsts(TAA_ChainPostFX.DEFAULTS) 
    end
  end
end

TAA_ChainPostFX.setPriority = function(priority)
  if scenetree.TAA_PreFx then
    local prevEnabled = scenetree.TAA_PreFx:isEnabled()
    scenetree.TAA_PreFx:disable(); scenetree.TAA_PreFx.renderPriority = priority
    if prevEnabled then scenetree.TAA_PreFx:enable() end
  end
end

shaderConstsActual()
rawset(_G, "TAA_ChainPostFX", TAA_ChainPostFX)
return TAA_ChainPostFX