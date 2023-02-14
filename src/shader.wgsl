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
    }
    if (in_vertex_index == 1u) {
        let vert = quadVertices[1];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 2u) {
        let vert = quadVertices[2];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }

    // tri 2
    if (in_vertex_index == 3u) {
        let vert = quadVertices[2];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 4u) {
        let vert = quadVertices[1];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }
    if (in_vertex_index == 5u) {
        let vert = quadVertices[3];
        out.clip_position = vec4<f32>(vert.x, vert.y, 0.0, 1.0);
    }

    out.vert_pos = out.clip_position.xyz;

    return out;
}

 

 


// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv: vec2<f32> = in.vert_pos.xy;

    let iter: u32 = mandelbrot(uv);

    let p: f32 = 1. / (pow(f32(iter), 0.5) + 1.);
    
    let col = vec3<f32>(p, p,p);

    //let col = (in.vert_pos + vec3<f32>(1.0,1.0,1.0)) / 2.0;

    return vec4<f32>(col.xyz, 1.0);
}

 
const RADIUS: f32 = 2.0;
const MAX_ITERATION: u32 = 100u;

fn mandelbrot(pos: vec2<f32>) -> u32 {
       let p0: vec2<f32> = pos - vec2<f32>(0.5, 0.5);

       var p: vec2<f32> = vec2<f32>(0.0, 0.0);
       var iteration: u32 = 0u;

        while (dot(p, p) <= (RADIUS * RADIUS) && (iteration < MAX_ITERATION)) {
            let xtemp: f32 = p.x * p.x - p.y * p.y + p0.x;
            p.y = 2. * p.x * p.y + p0.y;
            p.x = xtemp;
            iteration += 1u;
        }
        
        return iteration;
}
 

 
 /* 
 
 const float RADIUS = 2.0;
const int MAX_ITERATION = 100;

// Source: https://en.wikipedia.org/wiki/Mandelbrot_set
int get_pixel(vec2 pos) {
       vec2 p0 = pos - vec2(0.5, 0.5);

       vec2 p = vec2(0.0, 0.0);
       int iteration = 0;

        while (dot(p, p) <= (RADIUS * RADIUS) && (iteration < MAX_ITERATION)) {
            float xtemp = p.x * p.x - p.y * p.y + p0.x;
            p.y = 2. * p.x * p.y + p0.y;
            p.x = xtemp;
            iteration += 1;
        }
        
        return iteration;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    
    uv = uv - (iMouse.xy/iResolution.xy);

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));
    
    
    int iter = get_pixel(uv);
    
    //float p = float(iter) / float(MAX_ITERATION);
    float p = 1. / float(pow(float(iter), 0.5) + 1.);
    
    

    // Output to screen
    //fragColor = vec4(col,1.0);
    fragColor = vec4(p, p,p,1.0);
}

       
 */