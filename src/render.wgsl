@group(0) @binding(0)
var color: texture_2d<f32>;
@group(0) @binding(1)
var color_sampler: sampler;

struct Info {
    // Values in -1..1 in the vertex shader but in 0..screen_res in the fragment shader
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Info {
    var pos: vec2<f32>;
    if i == 0u {
        switch v {
            case 0u: { pos = vec2(-1.0, -1.0); }
            case 1u: { pos = vec2(1.0, -1.0); }
            default: { pos = vec2(-1.0, 1.0); }
        }
    } else {
        switch v {
            case 0u: { pos = vec2(1.0, -1.0); }
            case 1u: { pos = vec2(1.0, 1.0); }
            default: { pos = vec2(-1.0, 1.0); }
        }
    }
    let coord = (pos + 1.0) / 2.0;
    return Info(vec4(pos, 0.0, 1.0), coord);
}

@fragment
fn fragment(info: Info) -> @location(0) vec4<f32> {
    return textureSample(color, color_sampler, info.tex_coord);
}