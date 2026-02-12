Shader "Unlit/RainbowScan"
{
    Properties
    {
        _BackgroundTex ("Background", 2D) = "black" {}
        _ForegroundTex ("Foreground (PNG)", 2D) = "clear" {}
        _ForegroundOpacity ("Foreground Opacity", Range(0, 1)) = 1
        _DepthMin ("Depth Min", Range(0, 1)) = 0
        _DepthMax ("Depth Max", Range(0, 1)) = 1
        _DepthFeather ("Depth Feather", Range(0, 0.1)) = 0.005
        _ShowDepthOutside ("Show Depth Outside Mask", Float) = 0
        _DebugView ("Debug View (0=Off 1=Depth 2=Mask 3=Range)", Range(0, 3)) = 0
        _UseMeshUV ("Use Mesh UV", Float) = 0
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v, out float4 outpos : SV_POSITION)
            {
                v2f o;
                outpos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            Texture2D<float4> WebCamTex;
            Texture2D<float4> DepthTex;
            Texture2D<float4> ColorRampTex;
            Texture2D<float4> BackgroundTex;
            Texture2D<float4> ForegroundTex;

            SamplerState LinearClampSampler;

            float4 ScreenCamResolution;
            int DepthOnly;
            float _DepthMin;
            float _DepthMax;
            float _DepthFeather;
            float _ShowDepthOutside;
            float _DebugView;
            float _ForegroundOpacity;
            float _UseMeshUV;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                float2 camUV;
                float2 depthUV;
                if (_UseMeshUV > 0.5)
                {
                    camUV = i.uv;
                    depthUV = float2(i.uv.x, 1 - i.uv.y);
                }
                else
                {
        	        #if defined (SHADER_API_MOBILE)
                    camUV = 1 - float2((screenPos.y / ScreenCamResolution.x), 1 - (screenPos.x / ScreenCamResolution.y));
                    depthUV = 1 - float2((screenPos.y / ScreenCamResolution.x), (screenPos.x / ScreenCamResolution.y));
                    #else
                    camUV = float2((screenPos.x / ScreenCamResolution.y), (screenPos.y / ScreenCamResolution.x));
                    depthUV = float2((screenPos.x / ScreenCamResolution.y), 1 - (screenPos.y / ScreenCamResolution.x));
                    #endif
                }
                float4 rgba = WebCamTex.SampleLevel(LinearClampSampler, camUV, 0);
                float4 bg = BackgroundTex.SampleLevel(LinearClampSampler, camUV, 0);
                float4 fg = ForegroundTex.SampleLevel(LinearClampSampler, camUV, 0);
                float depth = DepthTex.SampleLevel(LinearClampSampler, depthUV, 0).r;

                depth = saturate(depth);
                float depthMin = min(_DepthMin, _DepthMax);
                float depthMax = max(_DepthMin, _DepthMax);
                if (depthMax > 1.0)
                {
                    depthMin /= 1024.0;
                    depthMax /= 1024.0;
                }
                float minEdge = smoothstep(depthMin - _DepthFeather, depthMin + _DepthFeather, depth);
                float maxEdge = 1 - smoothstep(depthMax - _DepthFeather, depthMax + _DepthFeather, depth);
                float depthMask = saturate(minEdge * maxEdge);

                if (_DebugView > 0.5 && _DebugView < 1.5)
                    return float4(depth, depth, depth, 1);

                if (_DebugView >= 1.5 && _DebugView < 2.5)
                    return float4(depthMask, depthMask, depthMask, 1);

                if (_DebugView >= 2.5)
                {
                    float3 debugColor = depth < depthMin ? float3(1, 0, 0) : (depth > depthMax ? float3(0, 0, 1) : float3(0, 1, 0));
                    return float4(debugColor, 1);
                }

                float4 webcamOverBackground = lerp(bg, rgba, depthMask);

                float rampCoord = depth * 1023.0;
                float4 depthColor = ColorRampTex.Load(uint3((uint)rampCoord, 0, 0));
                float4 baseColor = _ShowDepthOutside > 0.5 ? lerp(depthColor, rgba, depthMask) : webcamOverBackground;
                float fgAlpha = saturate(fg.a * _ForegroundOpacity);
                return lerp(baseColor, fg, fgAlpha);
            }
            ENDCG
        }
    }
}
