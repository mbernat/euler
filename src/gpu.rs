use wgpu::*;
use winit::window::Window;

pub struct SharedState {
    device: Device,
    queue: Queue,
    surface: Surface,
}

impl SharedState {
    pub async fn new(window: &Window) -> SharedState
    {
        let instance = Instance::new(Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Primary backend & optional surface compatible adapter");
        let features = Features::TIMESTAMP_QUERY; // | Features::STORAGE_RESOURCE_BINDING_ARRAY | Features::BUFFER_BINDING_ARRAY;
        let limits = Limits::downlevel_defaults();
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Main device"),
                    features,
                    limits,
                },
                None,
            )
            .await
            .expect("Feature & downlevel limit compatible device");

        let size = window.inner_size();
        let format = surface.get_supported_formats(&adapter)[0];
        surface.configure(&device, &SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
        });

        SharedState {
            // Create resources, poll
            device,
            // Write buffers & textures, submit commands
            queue,
            surface,
        }
    }
}

pub struct ComputeState {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    storage_buffer: Buffer,
}

impl ComputeState {
    pub fn new(shared: &SharedState) -> ComputeState {
        let device = &shared.device;
        let source = ShaderSource::Wgsl(
            r"
@group(0) @binding(0)
var<storage, read_write> t: array<f32>;

struct Ids {
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
}

@compute @workgroup_size(16, 16)
fn entry(ids: Ids) {
    let row = ids.num_groups.x * 16u;
    let id = ids.global_id.y * row + ids.global_id.x;
    t[id] += 1.0;
}
"
            .into(),
        );
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute shader module"),
            source,
        });
        let entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute bind group layout"),
            entries: &[entry],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "entry",
        });

        let values = [0.0; 16 * 16];
        let storage_buffer = {
            use util::DeviceExt;
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Compute storage buffer"),
                contents: bytemuck::bytes_of(&values),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        };
        let entry = BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        };
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute bind group"),
            layout: &bind_group_layout,
            entries: &[entry],
        });

        ComputeState {
            pipeline,
            bind_group,
            storage_buffer,
        }
    }

    pub fn run(&self, shared: &SharedState) {
        // Create passes, copy between buffers and textures
        let mut encoder = shared
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute command encoder"),
            });

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute compute pass"),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // Each axis: grid size = 256, workgroup_size = 16, num groups = 16
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        let size = self.storage_buffer.size();
        let output_buffer = shared.device.create_buffer(&BufferDescriptor {
            label: Some("Compute output buffer"),
            size: size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.storage_buffer, 0, &output_buffer, 0, size);

        let buffer = encoder.finish();
        shared.queue.submit([buffer]);
        let (tx, rx) = std::sync::mpsc::channel();
        output_buffer.slice(..).map_async(MapMode::Read, move |_| {
            tx.send(()).expect("Receiver lives");
        });
        shared.device.poll(MaintainBase::Wait);
        rx.recv().expect("Sender lives");
        let view = output_buffer.slice(..).get_mapped_range();
        let size = 256 * 4;
        let chunk = view.chunks(size).next().unwrap();
        println!("{:?}", bytemuck::from_bytes::<[f32; 256]>(chunk));
    }
}



pub struct RenderState {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    storage_buffer: Buffer,
}

impl RenderState {
    pub fn new(shared: &SharedState) -> RenderState {
        let device = &shared.device;
        let source = ShaderSource::Wgsl(
            r"
@group(0) @binding(0)
var<storage, read_write> t: array<f32>;

struct Info {
    // Values in -1..1 in the vertex shader but in 0..screen_res in the fragment shader
    @builtin(position) pos: vec4<f32>,
    @location(0) fraction: vec4<f32>,
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
    let uni = (pos + 1.0) / 2.0;
    return Info(vec4(pos, 0.0, 1.0), vec4(uni, 0.5, 1.0));
}

@fragment
fn fragment(info: Info) -> @location(0) vec4<f32> {
    return info.fraction;
}
"
            .into(),
        );
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Render shader module"),
            source,
        });
        let entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Render bind group layout"),
            entries: &[entry],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render compute pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vertex",
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fragment",
                // TODO use supported format from surface
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::default(),
                })],
            }),
            multiview: None,
        });

        let values = [0.0; 16 * 16];
        let storage_buffer = {
            use util::DeviceExt;
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Render storage buffer"),
                contents: bytemuck::bytes_of(&values),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        };
        let entry = BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        };
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render bind group"),
            layout: &bind_group_layout,
            entries: &[entry],
        });

        RenderState {
            pipeline,
            bind_group,
            storage_buffer,
        }
    }

    pub fn run(&self, shared: &SharedState) {
        let surface_texture = shared.surface.get_current_texture().expect("Valid surface texture");
        let view = surface_texture.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = shared
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render command encoder"),
            });

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render render pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLUE),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..2);
        drop(pass);
        let command_buffer = encoder.finish();
        shared.queue.submit([command_buffer]);
        surface_texture.present();
    }
}
