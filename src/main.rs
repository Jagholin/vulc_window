#![allow(dead_code, unused_variables, clippy::redundant_clone)]

mod cs;
mod fbx;
mod graphics_context;
mod input_helpers;
mod logger;
mod mesh;
mod pipeline;
mod renderer;
mod uniforms;
mod vertex_type;

use graphics_context::GraphicsContext;
use logger::Logger;
use mesh::Mesh;
use pipeline::{FramebufferFrameSwapper, StandardVulcanPipeline};
use renderer::{IssueCommands, VertexBufferRenderer};
use std::sync::Arc;
use uniforms::UniformHolder;

use cgmath::prelude::*;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::ImageFormatInfo;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::swapchain::{
    self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

pub type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

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

fn graphics_context(device: Arc<Device>, queue: Arc<Queue>) -> GraphicsContext {
    GraphicsContext::new(device, queue)
}

/// This function initiates framebuffers, renderbuffers and pipeline
fn prepare_graphics(
    gc: &GraphicsContext,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: impl IntoIterator<Item = Arc<SwapchainImage<Window>>>,
    depth_format: Format,
    dimensions: [f32; 2],
) -> (StandardVulcanPipeline, UniformHolder) {
    let vs = cs::load_vert_shader(gc.device.clone()).unwrap();
    let fs = cs::load_frag_shader(gc.device.clone()).unwrap();

    let mut frame_holder = FramebufferFrameSwapper::new();
    let pipe = StandardVulcanPipeline::new()
        .fs(fs)
        .vs(vs)
        .depth_format(depth_format)
        .dimensions(dimensions)
        .format(swapchain.image_format())
        .images(swapchain_images)
        .build(&gc, &mut frame_holder);

    let uniholder = UniformHolder::new(gc, pipe.pipeline.clone());

    (pipe, uniholder)
}

fn matrix_from_time(passed: std::time::Duration) -> cgmath::Matrix4<f32> {
    const DISTANCE: f32 = 1.5;

    let time_factor = passed.as_millis() as f32 / 1000.0;
    let x_coord = time_factor.sin() * DISTANCE;
    let y_coord = time_factor.cos() * DISTANCE;

    cgmath::Matrix4::look_at_rh(
        [x_coord, y_coord, DISTANCE].into(),
        [0.0, 0.0, 0.7].into(),
        [0.0, 0.0, 1.0].into(),
    )
}

fn main() {
    let mut logger = Logger::new(logger::create_logfile());

    let fbxfile = input_helpers::ask::<String>("enter fbx file name").unwrap();
    let vertex_data = fbx::read_fbx_document(&fbxfile, &mut logger).unwrap();

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

    let (device, queue, caps, image_format, depth_format) = {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
            .filter(|i| i.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|d| {
                d.queue_families()
                    .find(|q| {
                        q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false)
                    })
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
        let image_formats = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap();
        let candidate_depth_formats = [
            Format::D32_SFLOAT,
            Format::D32_SFLOAT_S8_UINT,
            Format::D24_UNORM_S8_UINT,
            Format::D16_UNORM,
            Format::D16_UNORM_S8_UINT,
        ];
        let depth_format = candidate_depth_formats
            .into_iter()
            .find(|df| {
                let props = physical_device.image_format_properties(ImageFormatInfo {
                    format: Some(df.to_owned()),
                    usage: ImageUsage::depth_stencil_attachment(),
                    ..Default::default()
                });
                if let Ok(Some(value)) = props {
                    return true;
                }
                false
            })
            .unwrap();
        println!("Image formats for given surface found: {:?}", image_formats);
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        (device, queue, caps, image_format, depth_format)
    };
    let gc = Arc::new(graphics_context(device.clone(), queue.clone()));

    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let mut persp_matrix = cgmath::perspective(
        cgmath::Deg(60.0),
        dimensions.width as f32 / dimensions.height as f32,
        0.001,
        1000.0,
    );

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
    // ImageView::new(image, create_info)

    let mesh = Mesh::from_vertex_vec(gc.clone(), vertex_data);
    let gc = graphics_context(device.clone(), queue.clone());
    // let vertex_buffer = mesh.vbo();
    let (mut pipeline, mut uniform_holder) = prepare_graphics(
        &gc,
        swapchain.clone(),
        images,
        depth_format,
        [dimensions.width as f32, dimensions.height as f32],
    );
    let mesh_vbo = VertexBufferRenderer::new(&mesh);

    //logger.log(format!("{:#?}\n", caps).as_str());
    logger.log("Cant see whats behind you\n");

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let mut window_minimized = false;

    let current_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            window_resized = true;
            window_minimized = size.height == 0 || size.width == 0;
        }
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
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                        return;
                    }
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                swapchain = new_swapchain;
                persp_matrix = cgmath::perspective(
                    cgmath::Deg(60.0),
                    dimensions.width as f32 / dimensions.height as f32,
                    0.001,
                    1000.0,
                );

                if window_resized {
                    window_resized = false;
                    // command_builders = prepare_graphics(device.clone(), queue.clone(), &mesh, swapchain.clone(), images,
                    //     depth_format, [dimensions.width as f32, dimensions.height as f32]);
                    (pipeline, uniform_holder) = prepare_graphics(
                        &gc,
                        swapchain.clone(),
                        images,
                        depth_format,
                        [dimensions.width as f32, dimensions.height as f32],
                    );
                }
            }
            let (image_id, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Acquiring swapchain image resulted in panic: {}", e),
                };
            // Create command buffer
            let matrix = persp_matrix * matrix_from_time(current_time.elapsed());
            uniform_holder.set_view_matrix(matrix.into());
            let mut comm_builder = gc.create_command_builder();
            pipeline.begin_render(&mut comm_builder, &uniform_holder, image_id);
            // Loop over all renderables, when there are more than 1
            mesh_vbo.issue_commands(&mut comm_builder);
            pipeline.end_render(&mut comm_builder);

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
                Err(e) => panic!("Panic on flush: {}", e),
            }
            let time_elapsed = current_time.elapsed().as_micros();
            // println!("FPS: {}, branch time: {} µs, pre-fence init time: {} µs", 1000000 / time_elapsed, branch_start_time.elapsed().as_micros(), branch_duration_prefence);
            // current_time = std::time::Instant::now();
        }
        _ => {}
    });
}
