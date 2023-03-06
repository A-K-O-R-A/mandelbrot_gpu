use log::info;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use winit::window::Window;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    view: View,
}

#[derive(Debug)]
struct View {
    mouse_pos: PhysicalPosition<f64>,
    mouse_delta: (f64, f64),
    mouse_pressed: bool,
    uniform: ViewUniform,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    color_factor: f32,
    radius: f32,
    max_iterations: u32,
    // Bottom left corner
    min_x: f32,
    min_y: f32,
    // Top right corner
    max_x: f32,
    max_y: f32,
}

type Pos2 = (f32, f32);

impl ViewUniform {
    pub fn new() -> Self {
        // X Range
        let x_r = [-2.00, 0.47];
        // Y Range
        let y_r = [-1.12, 1.12];

        Self {
            color_factor: 50.,
            radius: 2.,
            max_iterations: 500,
            min_x: x_r[0],
            max_x: x_r[1],
            min_y: y_r[0],
            max_y: y_r[1],
        }
    }

    pub fn update(&mut self, a: Pos2, b: Pos2) {
        self.min_x = a.0;
        self.min_y = a.1;
        self.max_x = b.0;
        self.max_y = b.1;
    }

    /// Returns ( (x_min, y_min), (x_max, y_max))
    pub fn corners(&self) -> (Pos2, Pos2) {
        ((self.min_x, self.min_y), (self.max_x, self.max_y))
    }

    pub fn translate(&mut self, vec: Pos2) {
        let new_a = (self.min_x + vec.0, self.min_y + vec.1);
        let new_b = (self.max_x + vec.0, self.max_y + vec.1);

        self.update(new_a, new_b);
    }

    pub fn transform(&mut self, scale: (f32, f32)) {
        let old_range_sizes = self.range_sizes();
        // Calculate new coordinate ranges
        // Scale coordinate ranges according to the resizing of the window
        let new_range_sizes = (old_range_sizes.0 * scale.0, old_range_sizes.1 * scale.1);
        // The removed or added amount of coordinates
        let range_diffs = (
            old_range_sizes.0 - new_range_sizes.0,
            old_range_sizes.1 - new_range_sizes.1,
        );

        let (old_a, old_b) = self.corners();
        // "range_diffs.0 / 2" to keep the figure centered
        let new_a = (old_a.0 + range_diffs.0 / 2., old_a.1 + range_diffs.1 / 2.);
        let new_b = (old_b.0 - range_diffs.0 / 2., old_b.1 - range_diffs.1 / 2.);

        self.update(new_a, new_b);
    }

    /// Returns (xRange Size, yRange Size)
    pub fn range_sizes(&self) -> (f32, f32) {
        (self.max_x - self.min_x, self.max_y - self.min_y)
    }
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        info!("Using adapter {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Render pipeline

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // View Unifrom
        let view_uniform = ViewUniform::new();
        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Uniform Buffer"),
            contents: bytemuck::cast_slice(&[view_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("view_bind_group_layout"),
            });

        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[],           // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            view: View {
                mouse_pos: PhysicalPosition::new(0., 0.),
                mouse_delta: (0., 0.),
                mouse_pressed: false,
                uniform: view_uniform,
                buffer: view_buffer,
                bind_group: view_bind_group,
            },
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width <= 0 || new_size.height <= 0 {
            return;
        }

        // Calculate relative scale like (0.9, 1.1)
        let scale = (
            new_size.width as f32 / self.size.width as f32,
            new_size.height as f32 / self.size.height as f32,
        );

        // Resize the viewing coordinates
        self.view.uniform.transform(scale);

        // Resize window
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                let Some(key) = input.virtual_keycode else {
                    return  false;
                };

                // The speed at which the user pans with the keys
                const SPEED: f32 = 0.02;
                let range_sizes = self.view.uniform.range_sizes();

                match key {
                    // Reset config
                    VirtualKeyCode::R => {
                        self.view.uniform = ViewUniform::new();
                    }
                    // Change iterations
                    VirtualKeyCode::Comma => {
                        if self.view.uniform.max_iterations > 100 {
                            self.view.uniform.max_iterations -= 100;
                        }
                    }
                    VirtualKeyCode::Period => {
                        if self.view.uniform.max_iterations < u32::MAX / 10 {
                            self.view.uniform.max_iterations += 100;
                        }
                    }
                    // Change radius
                    VirtualKeyCode::K => {
                        if self.view.uniform.radius > 0.1 {
                            self.view.uniform.radius -= 0.1;
                        }
                    }
                    VirtualKeyCode::L => {
                        if self.view.uniform.radius < 1000.0 {
                            self.view.uniform.radius += 0.1;
                        }
                    }
                    // Change coloring
                    VirtualKeyCode::O => {
                        if self.view.uniform.color_factor > 1.0 {
                            self.view.uniform.color_factor -= 1.0;
                        }
                    }
                    VirtualKeyCode::P => {
                        if self.view.uniform.color_factor < 1000000.0 {
                            self.view.uniform.color_factor += 1.0;
                        }
                    }
                    // Move
                    VirtualKeyCode::Left | VirtualKeyCode::A => {
                        self.view.uniform.translate((-range_sizes.0 * SPEED, 0.))
                    }
                    VirtualKeyCode::Right | VirtualKeyCode::D => {
                        self.view.uniform.translate((range_sizes.0 * SPEED, 0.))
                    }
                    VirtualKeyCode::Up | VirtualKeyCode::W => {
                        self.view.uniform.translate((0., range_sizes.1 * SPEED))
                    }
                    VirtualKeyCode::Down | VirtualKeyCode::S => {
                        self.view.uniform.translate((0., -range_sizes.1 * SPEED))
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = position.clone();
                let delta = (
                    position.x - self.view.mouse_pos.x,
                    position.y - self.view.mouse_pos.y,
                );
                self.view.mouse_delta = delta;
                self.view.mouse_pos = new_pos;

                // User drags the mouse
                if self.view.mouse_pressed {
                    let logical_delta = (
                        delta.0 as f32 / self.size.width as f32,
                        delta.1 as f32 / self.size.height as f32,
                    );

                    let range_sizes = self.view.uniform.range_sizes();

                    let scaled_delta = (
                        -logical_delta.0 * range_sizes.0, //Negative because otherwise it's inverted
                        logical_delta.1 * range_sizes.1,
                    );

                    self.view.uniform.translate(scaled_delta);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                // Drag canvas
                match button {
                    MouseButton::Left => match state {
                        // User started dragging
                        ElementState::Pressed => self.view.mouse_pressed = true,
                        // User stopped dragging
                        ElementState::Released => self.view.mouse_pressed = false,
                    },
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::PixelDelta(_delta) => {
                    // Dragging with touchpads is ignored
                }
                MouseScrollDelta::LineDelta(_x, y) => {
                    let fact = (-y * 0.1) + 1.;
                    self.view.uniform.transform((fact, fact));
                }
            },
            _ => {}
        }

        false
    }

    fn update(&mut self) {
        // Updates the uniform buffer to let the GPU know are updated values of the ViewUniform
        self.queue.write_buffer(
            &self.view.buffer,
            0,
            bytemuck::cast_slice(&[self.view.uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            // Create render pass with a background color
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.view.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    println!("The following Keybindings are available:");
    let help = r"
    R - Reset everything
    
    , - decrease Iterations (-100)        K - decrease Radius (-0.1)
    . - increase Iterations (+100)        L - increase Radius (+0.1)
    O - decrease coloring factor (-1)
    P - increase coloring factor (+1)

    You can move with W/A/S/D, the arrow keys or by dragging with the mouse.

    Use the mouse wheel to zoom into or out of the center of the canvas.
    ";
    println!("{help}");
    println!("For debug info set the RUST_LOG environment variable to info/warn/error");

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Mandelbrot renderer")
        //.with_inner_size((500.0, 300.0))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                // new_inner_size is &&mut so we have to dereference it twice
                state.resize(**new_inner_size);
            }
            _ => {
                let _ = state.input(event);
            }
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }
        _ => {}
    });
}
