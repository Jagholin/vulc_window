#![allow(dead_code, unused_variables, clippy::redundant_clone)]

mod cs;
mod input_helpers;
mod logger;
mod renderer;
mod vertex_type;
mod mesh;
mod graphics_context;
mod fbx;

use graphics_context::GraphicsContext;
use vertex_type::VertexStruct;
use image::{ImageBuffer, Rgba};
use logger::Logger;
use renderer::{VertexBufferRenderer, IssueCommands};
use vulkano::pipeline::GraphicsPipeline;
use std::sync::Arc;
use vulkano::image::ImageViewAbstract;
use vulkano::shader::ShaderModule;
use mesh::Mesh;

use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, ImageDimensions, ImageUsage, StorageImage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{compute::ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::{input_assembly::InputAssemblyState, vertex_input::BuffersDefinition, viewport::{Viewport, ViewportState}};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo, SwapchainCreationError, self, AcquireError};
use vulkano::sync::{self, GpuFuture, FlushError};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

pub type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_integer_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[i32]>> {
    let data_iter = 0..65536;
    CpuAccessibleBuffer::from_iter(device, BufferUsage::all(), false, data_iter)
        .expect("Cant create test buffer")
}

fn execute_command_buffer<T>(cg: &GraphicsContext, buff: T)
where
    T: PrimaryCommandBuffer + 'static,
{
    let future = sync::now(cg.device.clone())
        .then_execute(cg.queue.clone(), buff)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}

fn create_image(gc: &GraphicsContext) -> Arc<StorageImage> {
    let result = StorageImage::new(
        gc.device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(gc.queue.family()),
    )
    .unwrap();
    println!("image created!");
    result
}

fn clear_image<T>(
    gc: &GraphicsContext,
    image: Arc<StorageImage>,
    command_builder: &mut AutoCommandBufferBuilder<T>,
) {
    // let mut builder = gc.create_command_builder();
    command_builder
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
        .unwrap();
}

fn export_image<T>(
    gc: &GraphicsContext,
    image: Arc<StorageImage>,
    command_builder: &mut AutoCommandBufferBuilder<T>,
) -> impl FnOnce(&GraphicsContext) {
    let buf = CpuAccessibleBuffer::from_iter(
        gc.device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 255u8),
    )
    .expect("cant create buffer to save the image");

    command_builder
        .copy_image_to_buffer(image, buf.clone())
        .unwrap();

    move |gc| {
        let content = buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &content[..]).unwrap();
        println!("image exported!");
        image.save("image.png").unwrap();
    }
}

fn bind_data_to(
    pipeline: Arc<ComputePipeline>,
    data_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .expect("cant create descriptor set")
}

fn bind_imageview_to(
    pipeline: Arc<ComputePipeline>,
    img_view: Arc<dyn ImageViewAbstract>,
) -> Arc<PersistentDescriptorSet> {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, img_view.clone())],
    )
    .expect("cant create descriptor set")
}

fn shader_pipeline_compute(
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader: Arc<ShaderModule>,
) -> (Arc<ShaderModule>, Arc<ComputePipeline>) {
    // let shader = cs::load(device.clone()).expect("failed to create shader module");
    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("cant create compute pipeline");

    (shader, pipeline)
}

fn shader_pipeline_graphics(
    gc: &GraphicsContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    format: Format,
    dimensions: [f32; 2]
) -> (Arc<RenderPass>, Arc<GraphicsPipeline>) {
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions,
        depth_range: 0.0..1.0
    };

    let render_pass = vulkano::single_pass_renderpass!(gc.device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexStruct>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(gc.device.clone())
        .unwrap();
    (render_pass, pipeline)
}

fn compute_context(device: Arc<Device>, queue: Arc<Queue>) -> GraphicsContext {
    GraphicsContext::new(device, queue)
}

fn graphics_context(device: Arc<Device>, queue: Arc<Queue>) -> GraphicsContext {
    GraphicsContext::new(device, queue)
}

// todo: extract VBOs somewhere else - they are not needed here

/// This function initiates framebuffers, renderbuffers and pipeline
fn prepare_graphics(device: Arc<Device>, 
    queue: Arc<Queue>, 
    mesh: & Mesh,
    //vbo: Arc<CpuAccessibleBuffer<[VertexStruct]>>, 
    swapchain: Arc<Swapchain<Window>>, 
    swapchain_images: &[Arc<SwapchainImage<Window>>], 
    dimensions: [f32; 2]) -> Vec<VertexBufferRenderer<VertexStruct>> 
{
    let gc = graphics_context(device.clone(), queue.clone());
    let vs = cs::load_vert_shader(device.clone()).unwrap();
    let fs = cs::load_frag_shader(device.clone()).unwrap();
    let (render_pass, pipeline) = shader_pipeline_graphics(&gc, vs, fs, swapchain.image_format(), dimensions);
    // creating a framebuffer
    let fbs = get_framebuffers(swapchain_images, render_pass.clone());

    VertexBufferRenderer::new(&fbs, pipeline, mesh)
}

fn perform_compute(device: Arc<Device>, queue: Arc<Queue>) {
    let gc = compute_context(device.clone(), queue.clone());
    let (shader, pipeline) = shader_pipeline_compute(
        device.clone(),
        queue.clone(),
        cs::load_mult_const(device.clone()).expect("failed to create shader module"),
    );
    let data_buffer = create_integer_buffer(device.clone());
    let set = bind_data_to(pipeline.clone(), data_buffer.clone());
    let mut command_builder = gc.create_command_builder();

    command_builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024, 1, 1])
        .unwrap();

    let command_buffer = command_builder.build().unwrap();

    execute_command_buffer(&gc, command_buffer);

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as i32 * 12);
    }
    println!("Successful compute run!");

    // testing image creation
    let (_, pipeline) = shader_pipeline_compute(
        gc.device.clone(),
        queue.clone(),
        cs::load_test_compute(device).unwrap(),
    );
    let image = create_image(&gc);
    let image_view = ImageView::new_default(image.clone()).unwrap();
    let set = bind_imageview_to(pipeline.clone(), image_view);
    let mut command_builder = gc.create_command_builder();
    clear_image(&gc, image.clone(), &mut command_builder);
    command_builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap();
    let finish_export = export_image(&gc, image, &mut command_builder);
    let command_buffer = command_builder.build().unwrap();
    execute_command_buffer(&gc, command_buffer);
    finish_export(&gc);
}

fn main() {
    let mut logger = Logger::new(logger::create_logfile());

    let fbxfile = input_helpers::ask::<String>("enter fbx file name").unwrap();
    let vertex_data = fbx::read_fbx_document(&fbxfile, &mut logger).unwrap();
    // return;

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .expect("cant create info");
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (device, queue, caps, image_format) = {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
            .filter(|i| i.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|d| {
                d.queue_families()
                    .find(|q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                    .map(|q| (d, q))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .expect("no physical device match the criteria");

        println!(
            "physical device found! {}",
            physical_device.properties().device_name
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: physical_device
                    .required_extensions()
                    .union(&device_extensions),
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )
        .expect("creating virtual device failed");

        let queue = queues.next().unwrap();
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("Cant get device surface caps");
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        (device, queue, caps, image_format)
    };

    let gc = Arc::new(graphics_context(device.clone(), queue.clone()));

    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format: Some(image_format),
            image_extent: dimensions.into(),
            image_usage: ImageUsage::color_attachment(),
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    let mut mesh = Mesh::from_vertex_vec(gc.clone(), vertex_data);
    // let vertex_buffer = mesh.vbo();
    let mut command_builders = prepare_graphics(device.clone(), queue.clone(), &mesh, swapchain.clone(), &images,
        [dimensions.width as f32, dimensions.height as f32]);

    //logger.log(format!("{:#?}\n", caps).as_str());
    logger.log("Cant see whats behind you\n");

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let mut window_minimized = false;

    let mut current_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        },
        Event::WindowEvent { 
            event: WindowEvent::Resized(size), ..
        } => {
            window_resized = true;
            window_minimized = size.height == 0 || size.width == 0;
        },
        Event::MainEventsCleared => {
            let branch_start_time = std::time::Instant::now();
            if window_minimized {
                return;
            }
            if recreate_swapchain || window_resized {
                recreate_swapchain = false;
                let dimensions = surface.window().inner_size();

                let (new_swapchain, images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    .. swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                        return;
                    },
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                swapchain = new_swapchain;

                if window_resized {
                    window_resized = false;
                    command_builders = prepare_graphics(device.clone(), queue.clone(), &mesh, swapchain.clone(), &images,
                        [dimensions.width as f32, dimensions.height as f32]);
                }
            }
            let (image_id, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    return;
                },
                Err(e) => panic!("Acquiring swapchain image resulted in panic: {}", e),
            };
            let comm_buffer = command_builders.get(image_id).unwrap();
            let mut comm_builder = gc.create_command_builder();
            comm_buffer.issue_commands(&mut comm_builder);
            let comm_buffer = comm_builder.build().unwrap();
            if suboptimal {
                recreate_swapchain = true;
            }
            let branch_duration_prefence = branch_start_time.elapsed().as_micros();
            let exec = sync::now(device.clone())
                .join(acquire_future)
                .then_execute(queue.clone(), comm_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_id)
                .then_signal_fence_and_flush();


            match exec {
                Ok(future) => {
                    future.wait(None).unwrap();
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                }
                Err(e) => panic!("Panic on flush: {}", e)
            }
            let time_elapsed = current_time.elapsed().as_micros();
            // println!("FPS: {}, branch time: {} µs, pre-fence init time: {} µs", 1000000 / time_elapsed, branch_start_time.elapsed().as_micros(), branch_duration_prefence);
            current_time = std::time::Instant::now();
        },
        _ => {},
    });
}
