#version 450
#extension GL_ARB_separate_shader_objects : enable

// @[export]
layout (set = 0, binding = 0) uniform texture2D in_color;

// @[export]
layout (set = 0, binding = 1) uniform texture2D in_blur;

// @[immutable_samplers([
//         (
//             mag_filter: Nearest,
//             min_filter: Nearest,
//             address_mode_u: ClampToEdge,
//             address_mode_v: ClampToEdge,
//             address_mode_w: ClampToEdge,
//             anisotropy_enable: false,
//             max_anisotropy: 1.0,
//             border_color: FloatOpaqueWhite,
//             unnormalized_coordinates: false,
//             compare_enable: false,
//             compare_op: Always,
//             mipmap_mode: Linear,
//             mip_lod_bias: 0,
//             min_lod: 0,
//             max_lod: 1000
//         )
// ])]
layout (set = 0, binding = 2) uniform sampler smp;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 out_sdr;

void main()
{
    vec4 color = texture(sampler2D(in_color, smp), inUV) + texture(sampler2D(in_blur, smp), inUV);

    // tonemapping.. TODO: implement auto-exposure
    vec3 mapped = color.rgb / (color.rgb + vec3(1.0));
    out_sdr = vec4(mapped, color.a);
}
