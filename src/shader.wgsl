// Vertex shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
}

const quadVertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
);

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    // vertex_index will be 0-3
    // we want a quad coverting the entire screen

    var out: VertexOutput;

/*
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;

    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);

*/


    if (in_vertex_index == 0u) {
        let vert = quadVertices[0];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 1u) {
        let vert = quadVertices[1];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 2u) {
        let vert = quadVertices[2];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 3u) {
        let vert = quadVertices[3];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }

    out.vert_pos = out.clip_position.xyz;

    return out;
}

 

 


// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.5, 0.1, 1.0);
}

 

 