#include "shaders/common/postFx/postFx.h.hlsl"
#include "shaders/common/hlsl.h"

uniform_sampler2D(taaResultTex, 0);

cbuffer perDraw {
    float taaSharpness;
    float taaDebugMode;

    float2 oneOverTargetSize;
    POSTFX_UNIFORMS
};

#include "shaders/common/postFx/postFx.hlsl"

#ifdef SHADER_STAGE_VS
#define mainV main
#else
#define mainP main
#endif

// ============================================================================
// AMD FIDELITYFX SUPER RESOLUTION 1 -- RCAS + SRTM (32-bit path)
// ----------------------------------------------------------------------------
// Embedded from ffx_fsr1.h v1.20210629 (MIT license, notice below). Only the
// parts this pass needs are included: FsrRcasCon, the non-packed 32-bit
// FsrRcasF, and the SRTM reversible tonemapper. EASU / LFGA / TEPD and all
// 16-bit variants are omitted.
//
// Deviations from upstream (all OUTSIDE the algorithm bodies, which are
// line-for-line):
//   * ffx_a.h's 32-bit HLSL macro layer is reproduced inline below (the
//     header itself is not part of this engine's shader tree).
//   * 'varAF2(name)=initAF2(a,b)' (a CPU-compilation helper) is written as a
//     plain constructor in FsrRcasCon.
//   * FSR_RCAS_DENOISE is enabled (matches this mod's previous behavior of
//     always applying the noise-limiting term).
//   * The calling-shader callbacks (FsrRcasLoadF / FsrRcasInputF) implement
//     this engine's point-sampled fetch and the HDR input transform: RCAS
//     expects {0 to 1} input, the resolve output is linear HDR, so the input
//     transform applies AMD's SRTM and mainP applies the inverse afterwards.
// ----------------------------------------------------------------------------
// FidelityFX Super Resolution Sample
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// ============================================================================

// ---- ffx_a.h: minimal 32-bit HLSL subset ----------------------------------
#define A_GPU 1
#define A_HLSL 1

#define AF1  float
#define AF2  float2
#define AF3  float3
#define AF4  float4
#define AU1  uint
#define AU2  uint2
#define AU4  uint4
#define ASU2 int2

#define AF1_(a) ((AF1)(a))
#define AF2_(a) AF2(a, a)
#define AF3_(a) AF3(a, a, a)

#define A_STATIC static
#define outAU4 out AU4

#define AExp2F1(x)      exp2(x)
#define ASatF1(x)       saturate(x)
#define ARcpF1(x)       rcp(x)
#define APrxMedRcpF1(x) rcp(x)   // ffx_a.h maps the medium approx to rcp() on the A_HLSL path
#define AMin3F1(x, y, z) min(x, min(y, z))
#define AMax3F1(x, y, z) max(x, max(y, z))
#define AU1_AF1(x) asuint(x)
#define AF1_AU1(x) asfloat(x)
// Pack two floats as f16 halves into one uint (ffx_a.h AU1_AH2_AF2). Only the
// 16-bit RCAS variants consume con[1]; kept so FsrRcasCon stays verbatim.
#define AU1_AH2_AF2(a) (f32tof16((a).x) | (f32tof16((a).y) << 16))

#define FSR_RCAS_F 1
#define FSR_RCAS_DENOISE 1

// ============================================================================
//                                                      CONSTANT SETUP
// ============================================================================
// This is set at the limit of providing unnatural results for sharpening.
#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                      CONSTANT SETUP
//==============================================================================================================================
// Call to setup required constant values (works on CPU or GPU).
A_STATIC void FsrRcasCon(
outAU4 con,
// The scale is {0.0 := maximum, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
AF1 sharpness){
 // Transform from stops to linear value.
 sharpness=AExp2F1(-sharpness);
 AF2 hSharp=AF2(sharpness,sharpness);
 con[0]=AU1_AF1(sharpness);
 con[1]=AU1_AH2_AF2(hSharp);
 con[2]=0;
 con[3]=0;}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                   NON-PACKED 32-BIT VERSION
//==============================================================================================================================
#if defined(A_GPU)&&defined(FSR_RCAS_F)
 // Input callback prototypes that need to be implemented by calling shader
 AF4 FsrRcasLoadF(ASU2 p);
 void FsrRcasInputF(inout AF1 r,inout AF1 g,inout AF1 b);
//------------------------------------------------------------------------------------------------------------------------------
 void FsrRcasF(
 out AF1 pixR, // Output values, non-vector so port between RcasFilter() and RcasFilterH() is easy.
 out AF1 pixG,
 out AF1 pixB,
 #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
  out AF1 pixA,
 #endif
 AU2 ip, // Integer pixel position in output.
 AU4 con){ // Constant generated by RcasSetup().
  // Algorithm uses minimal 3x3 pixel neighborhood.
  //    b 
  //  d e f
  //    h
  ASU2 sp=ASU2(ip);
  AF3 b=FsrRcasLoadF(sp+ASU2( 0,-1)).rgb;
  AF3 d=FsrRcasLoadF(sp+ASU2(-1, 0)).rgb;
  #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
   AF4 ee=FsrRcasLoadF(sp);
   AF3 e=ee.rgb;pixA=ee.a;
  #else
   AF3 e=FsrRcasLoadF(sp).rgb;
  #endif
  AF3 f=FsrRcasLoadF(sp+ASU2( 1, 0)).rgb;
  AF3 h=FsrRcasLoadF(sp+ASU2( 0, 1)).rgb;
  // Rename (32-bit) or regroup (16-bit).
  AF1 bR=b.r;
  AF1 bG=b.g;
  AF1 bB=b.b;
  AF1 dR=d.r;
  AF1 dG=d.g;
  AF1 dB=d.b;
  AF1 eR=e.r;
  AF1 eG=e.g;
  AF1 eB=e.b;
  AF1 fR=f.r;
  AF1 fG=f.g;
  AF1 fB=f.b;
  AF1 hR=h.r;
  AF1 hG=h.g;
  AF1 hB=h.b;
  // Run optional input transform.
  FsrRcasInputF(bR,bG,bB);
  FsrRcasInputF(dR,dG,dB);
  FsrRcasInputF(eR,eG,eB);
  FsrRcasInputF(fR,fG,fB);
  FsrRcasInputF(hR,hG,hB);
  // Luma times 2.
  AF1 bL=bB*AF1_(0.5)+(bR*AF1_(0.5)+bG);
  AF1 dL=dB*AF1_(0.5)+(dR*AF1_(0.5)+dG);
  AF1 eL=eB*AF1_(0.5)+(eR*AF1_(0.5)+eG);
  AF1 fL=fB*AF1_(0.5)+(fR*AF1_(0.5)+fG);
  AF1 hL=hB*AF1_(0.5)+(hR*AF1_(0.5)+hG);
  // Noise detection.
  AF1 nz=AF1_(0.25)*bL+AF1_(0.25)*dL+AF1_(0.25)*fL+AF1_(0.25)*hL-eL;
  nz=ASatF1(abs(nz)*APrxMedRcpF1(AMax3F1(AMax3F1(bL,dL,eL),fL,hL)-AMin3F1(AMin3F1(bL,dL,eL),fL,hL)));
  nz=AF1_(-0.5)*nz+AF1_(1.0);
  // Min and max of ring.
  AF1 mn4R=min(AMin3F1(bR,dR,fR),hR);
  AF1 mn4G=min(AMin3F1(bG,dG,fG),hG);
  AF1 mn4B=min(AMin3F1(bB,dB,fB),hB);
  AF1 mx4R=max(AMax3F1(bR,dR,fR),hR);
  AF1 mx4G=max(AMax3F1(bG,dG,fG),hG);
  AF1 mx4B=max(AMax3F1(bB,dB,fB),hB);
  // Immediate constants for peak range.
  AF2 peakC=AF2(1.0,-1.0*4.0);
  // Limiters, these need to be high precision RCPs.
  AF1 hitMinR=min(mn4R,eR)*ARcpF1(AF1_(4.0)*mx4R);
  AF1 hitMinG=min(mn4G,eG)*ARcpF1(AF1_(4.0)*mx4G);
  AF1 hitMinB=min(mn4B,eB)*ARcpF1(AF1_(4.0)*mx4B);
  AF1 hitMaxR=(peakC.x-max(mx4R,eR))*ARcpF1(AF1_(4.0)*mn4R+peakC.y);
  AF1 hitMaxG=(peakC.x-max(mx4G,eG))*ARcpF1(AF1_(4.0)*mn4G+peakC.y);
  AF1 hitMaxB=(peakC.x-max(mx4B,eB))*ARcpF1(AF1_(4.0)*mn4B+peakC.y);
  AF1 lobeR=max(-hitMinR,hitMaxR);
  AF1 lobeG=max(-hitMinG,hitMaxG);
  AF1 lobeB=max(-hitMinB,hitMaxB);
  AF1 lobe=max(AF1_(-FSR_RCAS_LIMIT),min(AMax3F1(lobeR,lobeG,lobeB),AF1_(0.0)))*AF1_AU1(con.x);
  // Apply noise removal.
  #ifdef FSR_RCAS_DENOISE
   lobe*=nz;
  #endif
  // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
  AF1 rcpL=APrxMedRcpF1(AF1_(4.0)*lobe+AF1_(1.0));
  pixR=(lobe*bR+lobe*dR+lobe*hR+lobe*fR+eR)*rcpL;
  pixG=(lobe*bG+lobe*dG+lobe*hG+lobe*fG+eG)*rcpL;
  pixB=(lobe*bB+lobe*dB+lobe*hB+lobe*fB+eB)*rcpL;
  return;} 
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                          FSR - [SRTM] SIMPLE REVERSIBLE TONE-MAPPER
//
//------------------------------------------------------------------------------------------------------------------------------
// This provides a way to take linear HDR color {0 to FP16_MAX} and convert it into a temporary {0 to 1} ranged post-tonemapped linear.
// The tonemapper preserves RGB ratio, which helps maintain HDR color bleed during filtering.
//------------------------------------------------------------------------------------------------------------------------------
// Reversible tonemapper usage,
//  FsrSrtm*(color); // {0 to FP16_MAX} converted to {0 to 1}.
//  FsrSrtmInv*(color); // {0 to 1} converted into {0 to 32768, output peak safe for FP16}.
//==============================================================================================================================
#if defined(A_GPU)
 void FsrSrtmF(inout AF3 c){c*=AF3_(ARcpF1(AMax3F1(c.r,c.g,c.b)+AF1_(1.0)));}
 // The extra max solves the c=1.0 case (which is a /0).
 void FsrSrtmInvF(inout AF3 c){c*=AF3_(ARcpF1(max(AF1_(1.0/32768.0),AF1_(1.0)-AMax3F1(c.r,c.g,c.b))));}
#endif

// ============================================================================
// CALLING-SHADER CALLBACKS (per ffx_fsr1.h: implemented by the calling shader)
// ============================================================================
AF4 FsrRcasLoadF(ASU2 p)
{
    // Integer texel fetch: the pass's slot-0 sampler is ClampPoint, so border
    // offsets (p = -1 / p = size) clamp to the edge texel, matching the
    // reference's integer-load semantics.
    AF2 uv = (AF2(p) + AF2_(0.5)) * oneOverTargetSize;
    AF4 c = tex2Dlod(taaResultTex, float4(uv, 0.0, 0.0));
    c.rgb = max(c.rgb, AF3_(0.0));
    return c;
}

void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b)
{
    // HDR accommodation: map linear HDR {0 to FP16_MAX} into the {0 to 1}
    // range RCAS expects, with AMD's reversible tonemapper (RGB ratios are
    // preserved). mainP applies FsrSrtmInvF after filtering.
    AF3 c = AF3(r, g, b);
    FsrSrtmF(c);
    r = c.r; g = c.g; b = c.b;
}

// ============================================================================
// MAIN PASS
// ============================================================================
float4 mainP(PFXVertToPix IN) : SV_TARGET0
{
    // Debug view: raw resolve output, sharpening bypassed.
    if (taaDebugMode > 0.5) {
        return float4(max(0.0, tex2Dlod(taaResultTex, float4(IN.uv0, 0.0, 0.0)).rgb), 1.0);
    }

    // 0 disables sharpening entirely.
    AF1 strength = saturate(taaSharpness);
    if (strength <= 0.001) {
        return float4(max(0.0, tex2Dlod(taaResultTex, float4(IN.uv0, 0.0, 0.0)).rgb), 1.0);
    }

    // UI strength {0..1} -> RCAS "stops" ({0.0 := maximum sharpness, N > 0 :=
    // N halvings of the lobe}). FsrRcasCon stores exp2(-stops), which equals
    // the UI strength, so the slider maps linearly onto the sharpening lobe
    // exactly as it did with the previous implementation.
    AF1 stops = -log2(max(strength, 1.0 / 1024.0));

    AU4 con;
    FsrRcasCon(con, stops);

    // Integer pixel position of this output pixel (uv0 sits at texel centers).
    AU2 ip = AU2(IN.uv0 / oneOverTargetSize);

    AF1 pixR, pixG, pixB;
    FsrRcasF(pixR, pixG, pixB, ip, con);

    // Undo the SRTM applied inside FsrRcasInputF: back to linear HDR.
    AF3 pix = AF3(pixR, pixG, pixB);
    FsrSrtmInvF(pix);

    return float4(max(pix, 0.0), 1.0);
}

PFXVertToPix mainV(PFXVert IN) { return processPostFxVert(IN); }