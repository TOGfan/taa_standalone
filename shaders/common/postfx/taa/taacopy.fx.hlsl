#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(tex, 0);

cbuffer perDraw
{
  POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#ifdef SHADER_STAGE_VS
  #define mainV main
#else
  #define mainP main
#endif

float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    return tex2D(tex, IN.uv0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }