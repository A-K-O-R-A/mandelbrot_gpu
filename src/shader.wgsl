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
    // we want a quad coverting the entire screen

    var out: VertexOutput;

    // tri 1
    if (in_vertex_index == 0u) {
        let vert = quadVertices[0];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    } else  if (in_vertex_index == 1u) {
        let vert = quadVertices[1];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    } else if (in_vertex_index == 2u) {
        let vert = quadVertices[2];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    } else 

    // tri 2
    if (in_vertex_index == 3u) {
        let vert = quadVertices[2];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    } else if (in_vertex_index == 4u) {
        let vert = quadVertices[1];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    } else if (in_vertex_index == 5u) {
        let vert = quadVertices[3];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }

    out.vert_pos = out.clip_position.xyz;

    return out;
}

 

 


// Fragment shader
struct ViewUniform {
    dpi: f32,
    radius: f32,
    max_iterations: u32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
}

@group(0) @binding(0) // 1.
var<uniform> view: ViewUniform;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv: vec2<f32> = in.vert_pos.xy;

    let iter: u32 = mandelbrot(uv);

    let p: f32 = 1. / (pow(f32(iter), 0.7) + 1.);
    
    let col = vec3<f32>(p, .1 , .1);

    //let col = (in.vert_pos + vec3<f32>(1.0,1.0,1.0)) / 2.0;

    return vec4<f32>(col.xyz, 1.0);
}

 
// const RADIUS: f32 = 2.0;
// const MAX_ITERATION: u32 = 1000u;

fn project_cords(pos: vec2<f32>) -> vec2<f32> {
    // Ranges from 0-1
    let norm = (pos + vec2<f32>(1., 1.)) * 0.5;

    let x_size = view.x_max - view.x_min;
    let y_size = view.y_max - view.y_min;
    let size = vec2<f32>(x_size, y_size);
    
    let off = vec2<f32>(view.x_min, view.y_min);

    let proj = (norm * size) + off;

    return proj;
}

fn mandelbrot(pos: vec2<f32>) -> u32 {
    let p0: vec2<f32> = project_cords(pos);

    var p: vec2<f32> = vec2<f32>(0.0, 0.0);
    var iteration: u32 = 0u;

    let r2 = view.radius * view.radius;

    while (dot(p, p) <= r2 && (iteration < view.max_iterations)) {
        let xtemp: f32 = p.x * p.x - p.y * p.y + p0.x;
        p.y = 2. * p.x * p.y + p0.y;
        p.x = xtemp;
        iteration += 1u;
    }
        
    return iteration;
}