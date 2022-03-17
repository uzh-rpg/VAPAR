// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/OpticalFlow"
{
	Properties
	{
		_Sensitivity("Sensitivity", Float) = 10
	}
	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		
{			CGPROGRAM
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
				float4 vertex : SV_POSITION;
			};

			float4 _CameraMotionVectorsTexture_ST;
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _CameraMotionVectorsTexture);
				return o;
			}
			
			sampler2D _CameraMotionVectorsTexture;

            float3 Hue(float H)
			{
				float R = abs(H * 6 - 3) - 1;
			    float G = 2 - abs(H * 6 - 2);
			    float B = 2 - abs(H * 6 - 4);
			    return saturate(float3(R,G,B));
			}

			float3 HSVtoRGB(float3 HSV)
			{
			    return float3(((Hue(HSV.x) - 1) * HSV.y + 1) * HSV.z);
			}

			float3 hsv_to_rgb(float3 HSV)
			{
				float3 RGB = HSV.z;

				float var_h = HSV.x * 6;
				float var_i = floor(var_h);   // Or ... var_i = floor( var_h )
				float var_1 = HSV.z * (1.0 - HSV.y);
				float var_2 = HSV.z * (1.0 - HSV.y * (var_h-var_i));
				float var_3 = HSV.z * (1.0 - HSV.y * (1-(var_h-var_i)));
				if      (var_i == 0) { RGB = float3(HSV.z, var_3, var_1); }
				else if (var_i == 1) { RGB = float3(var_2, HSV.z, var_1); }
				else if (var_i == 2) { RGB = float3(var_1, HSV.z, var_3); }
				else if (var_i == 3) { RGB = float3(var_1, var_2, HSV.z); }
				else if (var_i == 4) { RGB = float3(var_3, var_1, HSV.z); }
				else                 { RGB = float3(HSV.z, var_1, var_2); }

				return (RGB);
			}

			float _Sensitivity;
			float3 MotionVectorsToOpticalFlow(float2 motion)
			{
				// Currently is based on HSV encoding from:
				//			"Optical Flow in a Smart Sensor Based on Hybrid Analog-Digital Architecture" by P. Guzman et al
				//			http://www.mdpi.com/1424-8220/10/4/2975

				// Analogous to http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
				// but might need to swap or rotate axis!

				// @TODO: support other HSV encodings (using lookup texture)
				// https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/ofdatabase_iccv_07.pdf
				// https://people.csail.mit.edu/celiu/SIFTflow/
				// some MATLAB code: https://github.com/suhangpro/epicflow/blob/master/utils/flow-code-matlab/computeColor.m

				// float angle = atan2(-motion.y, -motion.x);
				// float hue = angle / (UNITY_PI * 2.0) + 0.5;		// convert motion angle to Hue
				// float value = length(motion) * _Sensitivity;  	// convert motion strength to Value
    			// return HSVtoRGB(float3(hue, 1, value));		// HSV -> RGB

    			float angle = atan2(-motion.y, -motion.x);
				float hue = angle / (UNITY_PI * 2.0) + 0.5;
				float saturation = length(motion) * _Sensitivity;
				// return HSVtoRGB(float3(hue, saturation, 1));
				// return HSVtoRGB(float3(0.0, 0.5, 1.0));
				return hsv_to_rgb(float3(0.0, 0.5, 1.0));
			}

			/*fixed4 frag (v2f i) : SV_Target
			{
				float2 motion = tex2D(_CameraMotionVectorsTexture, i.uv).rg;
				// float3 rgb = MotionVectorsToOpticalFlow(motion);
				// rgb.x = 1.0;
				// rgb.y = 0.5;
				// rgb.z = 0.5;
				// rgb = GammaToLinearSpace(rgb);
				// return float4(rgb, 1);
				// return motion;
				// return float2(0.12345, 0.6789);
				return float2(0.0, 0.0);
			}*/

			float2 frag (v2f i) : SV_Target
			{
				float2 motion = tex2D(_CameraMotionVectorsTexture, i.uv).rg;
				return motion;
			}

			ENDCG
		}
	}
}
