use wgpu::*;
use winit::window::Window;

pub struct SharedState {
    device: Device,
    queue: Queue,
    surface: Surface,
}

impl SharedState {
    pub async fn new(window: &Window) -> SharedState {
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
        surface.configure(
            &device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: PresentMode::Fifo,
                alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
            },
        );

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
    integrate_pipeline: ComputePipeline,
    gather_pipeline: ComputePipeline,
    scatter_bl_pipeline: ComputePipeline,
    scatter_tr_pipeline: ComputePipeline,
    advect_u_pipeline: ComputePipeline,
    advect_v_pipeline: ComputePipeline,
    advect_density_pipeline: ComputePipeline,
    copy_pipeline: ComputePipeline,
    physics_bind_group: BindGroup,
    compute_output_bind_group: BindGroup,
    storage_buffer: Buffer,
    image_view: TextureView,
}

impl ComputeState {
    pub fn new(shared: &SharedState) -> ComputeState {
        let device = &shared.device;
        let source = ShaderSource::Wgsl(include_str!("compute.wgsl").into());
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
        let physics_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Physics bind group layout"),
                entries: &[entry],
            });
        let physics_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Physics pipeline layout"),
            bind_group_layouts: &[&physics_bind_group_layout],
            push_constant_ranges: &[],
        });

        let integrate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics integrate pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "integrate",
        });
        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics gather pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "gather",
        });
        let scatter_bl_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics scatter bottom left pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "scatter_bottom_left",
        });
        let scatter_tr_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics scatter top right pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "scatter_top_right",
        });
        let advect_u_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics advect u pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "advect_u",
        });
        let advect_v_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics advect v pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "advect_v",
        });
        let advect_density_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Physics advect density pipeline"),
            layout: Some(&physics_pipeline_layout),
            module: &shader_module,
            entry_point: "advect_density",
        });

        let entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        };
        let compute_output_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Compute output bind group layout"),
                entries: &[entry],
            });
        let copy_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Copy pipeline layout"),
            bind_group_layouts: &[
                &physics_bind_group_layout,
                &compute_output_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let copy_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Copy compute pipeline"),
            layout: Some(&copy_pipeline_layout),
            module: &shader_module,
            entry_point: "copy",
        });

        use bytemuck::Zeroable;
        use bytemuck::Pod;
        #[derive(Clone, Copy, Zeroable, Pod)]
        #[repr(C)]
        struct Point {
            u: f32,
            v: f32,
            s: f32,
            rho: f32,
            avg_div: f32,
            nu: f32,
            nv: f32,
            nrho: f32,
        }

        const SIZE: (usize, usize) = (256, 256);
        let init = Point{u: 0.0, v: 0.0, s: 1.0, rho: 0.0, avg_div: 0.0, nu: 0.0, nv: 0.0, nrho: 0.0};
        let mut values = vec![init; SIZE.0 * SIZE.1];

        // Walls
        for i in 0..SIZE.0 {
            values[i].s = 0.0;
            values[(SIZE.1 - 1) * SIZE.0 + i].s = 0.0;
        }
        for j in 0..SIZE.1 {
            values[j * SIZE.0].s = 0.0;
            values[(j + 1) * SIZE.0 - 1].s = 0.0;
        }

        let radius: i32 = 20;
        let pos = (50, 128);
        // Obstacle
        for i in 0..SIZE.0 {
            for j in 0..SIZE.1 {
                let sq_dist = (i as i32 - pos.0).pow(2) + (j as i32 - pos.1).pow(2);
                if sq_dist <= radius.pow(2) {
                    let index = i as usize + j as usize * SIZE.0;
                    values[index].s = 0.0;
                    values[index].rho = 1.0;
                }
            }
        }

        // Inflow from the center left
        values[256 * 128 + 1].u = 100.0;
        // Outflow into center right
        values[256 * 128 + 255].u = 100.0;

        let storage_buffer = {
            use util::DeviceExt;
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Physics storage buffer"),
                contents: bytemuck::cast_slice(&values[..]),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        };
        let physics_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Physics bind group"),
            layout: &physics_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
        });

        let img = device.create_texture(&TextureDescriptor {
            label: Some("Copy output texture"),
            size: Extent3d {
                width: SIZE.0 as u32,
                height: SIZE.1 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        });
        let img_view = img.create_view(&TextureViewDescriptor::default());
        let compute_output_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Copy bind group"),
            layout: &compute_output_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&img_view),
            }],
        });

        ComputeState {
            integrate_pipeline,
            gather_pipeline,
            scatter_bl_pipeline,
            scatter_tr_pipeline,
            advect_u_pipeline,
            advect_v_pipeline,
            advect_density_pipeline,
            copy_pipeline,
            physics_bind_group,
            compute_output_bind_group,
            storage_buffer,
            image_view: img_view,
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

        // Each axis: grid size = 256, workgroup_size = 16, num groups = 16
        let groups = 256 / 16;

        pass.set_pipeline(&self.integrate_pipeline);
        pass.set_bind_group(0, &self.physics_bind_group, &[]);
        pass.dispatch_workgroups(groups, groups, 1);

        for _ in 0..1000 {
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &self.physics_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);

            pass.set_pipeline(&self.scatter_bl_pipeline);
            pass.set_bind_group(0, &self.physics_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);

            pass.set_pipeline(&self.scatter_tr_pipeline);
            pass.set_bind_group(0, &self.physics_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        pass.set_pipeline(&self.advect_u_pipeline);
        pass.set_bind_group(0, &self.physics_bind_group, &[]);
        pass.dispatch_workgroups(groups, groups, 1);

        pass.set_pipeline(&self.advect_v_pipeline);
        pass.set_bind_group(0, &self.physics_bind_group, &[]);
        pass.dispatch_workgroups(groups, groups, 1);

        pass.set_pipeline(&self.advect_density_pipeline);
        pass.set_bind_group(0, &self.physics_bind_group, &[]);
        pass.dispatch_workgroups(groups, groups, 1);

        pass.set_pipeline(&self.copy_pipeline);
        pass.set_bind_group(0, &self.physics_bind_group, &[]);
        pass.set_bind_group(1, &self.compute_output_bind_group, &[]);
        pass.dispatch_workgroups(groups, groups, 1);

        drop(pass);
        shared.queue.submit(Some(encoder.finish()));
    }

    fn inspect(&self, shared: &SharedState) {
        let mut encoder = shared
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute copy encoder"),
            });
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
}

impl RenderState {
    pub fn new(shared: &SharedState, compute: &ComputeState) -> RenderState {
        let device = &shared.device;
        let source = ShaderSource::Wgsl(include_str!("render.wgsl").into());
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Render shader module"),
            source,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Render bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render render pipeline"),
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

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Compute output sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&compute.image_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        RenderState {
            pipeline,
            bind_group,
        }
    }

    pub fn run(&self, shared: &SharedState) {
        let surface_texture = shared
            .surface
            .get_current_texture()
            .expect("Valid surface texture");
        let view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

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
                    load: LoadOp::Clear(Color::BLACK),
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
