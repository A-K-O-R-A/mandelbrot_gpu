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
    color_factor: f32,
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
    // Logical coordinates on screen
    let uv: vec2<f32> = in.vert_pos.xy;

    // Coordinates for the mandelbrot set
    let p0   = project_cords(uv);
    // Iteration count of escape time algorithm
    let res = mandelbrot(p0);
    let p   = res.yz;
    var raw_iter = res.x;


    // Skip black areas entirely
    if (raw_iter == f32(view.max_iterations)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    //Smooth coloring - https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring
    let log_zn = log(dot(p, p)) / 2.0;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    // Rearranging the potential function.
    // Dividing log_zn by log(2) instead of log(N = 1<<8)
    // because we want the entire palette to range from the
    // center to radius 2, NOT our bailout radius.
    let iter = f32(raw_iter) + 1.0 - nu;

    //let p: f32 = 1. / (pow(f32(iter), 0.7) + 1.);
    let iter_u = u32(floor(iter));

    //let col1 = hsvToRgb(vec3<f32>(f32(iter_u) % 1.9, 1.0, 1.0));
    //let col2 = hsvToRgb(vec3<f32>(f32(iter_u + 1u) % 1.0, 1.0, 1.0));
    //let col2 = hsvToRgb(iter_u + 1u);
    //let col = lerp(col1, col2, iter % 1.0);
    let col = hsvToRgb(vec3<f32>(iter / view.color_factor % 1.0, 1.0, 1.0));
    
    return vec4<f32>(col.xyz, 1.0);    
}

// Linear interpolation
fn lerp(a: vec3<f32>, b: vec3<f32>, f: f32) -> vec3<f32> {
    let v = b - a; //Vec from a to b

    return a + (b * f);
}

fn exponential_cyclic_coloring(iter: u32) -> vec3<f32> {
    let N = 1.0;
    let s = f32(iter) / f32(view.max_iterations);


    let v = pow(
                pow(s,0.7),
                1.5
            ) % N ;

    let col = vec3<f32>(v,v,v);

    return col;
}

const PI: f32 = 3.14159;
fn LCH_coloring(iter: u32) -> vec3<f32> {
    let N = 360.0;
    let s = f32(iter) / f32(view.max_iterations);

    let v = 1.0 - pow(cos(PI * s), 2.0);
    let LCH = vec3<f32>(
        75.0 - (75.0 * v),
        28.0 + (75.0 - (75.0 * v)),
        pow(360.0 * s, 1.5) % 360.0
    );
    //let col = vec3<f32>(p, p, p);

    return //xyz_to_rgb(
        lab_to_xyz(
            lch_to_lab(LCH)
        )
    ;//);
}

// http://www.brucelindbloom.com/index.html?Eqn_LCH_to_Lab.html
fn lch_to_lab(lch: vec3<f32>) -> vec3<f32> {
    let l = lch.x;
    let a = lch.y * cos(radians(lch.z));
    let b = lch.y * sin(radians(lch.z));

    return vec3<f32>(l,a,b);
}

// http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
const REF_WHITE = vec3<f32>(1.0, 1.0, 1.0);
fn lab_to_xyz(lab: vec3<f32>) -> vec3<f32> {
    let l = lab.x;
    let a = lab.y;
    let b = lab.z;

    let e = 0.008856;
    let k = 903.3;

    // the f's
    let f_y = (l + 16.0) / 116.0;
    let f_x = (a / 500.0) + f_y;
    let f_z = f_y - (b / 200.0);

    var x_r: f32 = 0.0;
    if (pow(f_x, 3.0) > e) {
        x_r = pow(f_x, 3.0);
    } else {
        x_r = ((116.0 * f_x) - 16.0) / k;
    }

    var y_r: f32 = 0.0;
    if (l > k*e) {
        y_r = pow(((l + 16.0) - 116.0), 3.0);
    } else {
        y_r = l / k;
    }

    var z_r: f32 = 0.0;
    if (pow(f_z, 3.0) > e) {
        z_r = pow(f_x, 3.0);
    } else {
        z_r = ((116.0 * f_z) - 16.0) / k;
    }
    

    return vec3<f32>(x_r, y_r, z_r) * REF_WHITE;
}

// http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
const M =  mat3x3<f32>(
    vec3<f32>(3.2404542,   -1.5371385,  -0.4985314),
    vec3<f32>(-0.9692660,   1.8760108,   0.0415560),
    vec3<f32>(0.0556434,   -0.2040259,   1.0572252)
);
fn xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    let rgb = M * xyz;

    return rgb;
}

// https://stackoverflow.com/questions/51203917/math-behind-hsv-to-rgb-conversion-of-colors
fn hsvToRgb(hsv: vec3<f32>) -> vec3<f32> {
    var rgb = vec3<f32>(0.0, 0.0, 0.0);

    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;

    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    var r = 0.0;
    var g = 0.0;
    var b = 0.0;

    let m = u32(i % 6.0);
    
    if (m == 0u) {
        r = v;
        g = t;
        b = p;
    } else if (m == 1u) {
        r = q;
        g = v;
        b = p;
    } else if (m == 2u) {
        r = p;
        g = v;
        b = t;
    } else if (m == 3u) {
        r = p;
        g = q;
        b = v;
    } else if (m == 4u) {
        r = t;
        g = p;
        b = v;
    } else if (m == 5u) {
        r = v;
        g = p;
        b = q;
    }
    

    return vec3<f32>(r, g, b);
}

// Actual computation

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

// Returns (iter, x , y)
fn mandelbrot(p0: vec2<f32>) -> vec3<f32> {
    var p: vec2<f32> = vec2<f32>(0.0, 0.0);
    var iteration: u32 = 0u;

    let r2 = view.radius * view.radius;

    while (dot(p, p) <= r2 && (iteration < view.max_iterations)) {
        let xtemp: f32 = p.x * p.x - p.y * p.y + p0.x;
        p.y = 2. * p.x * p.y + p0.y;
        p.x = xtemp;
        iteration += 1u;
    }
        
    return vec3<f32>(f32(iteration), p);
}
 