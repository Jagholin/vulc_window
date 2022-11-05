#![allow(clippy::redundant_clone)]

mod app;
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

use app::{App, AppCreateStruct};
use graphics_context::GraphicsContextBuilder;
use logger::FileLogger;
use mesh::Mesh;
use pipeline::{FramebufferFrameSwapper, StandardVulcanPipeline};
use renderer::{IssueCommands, VertexBufferStreaming};
use std::sync::Arc;
use uniforms::UniformHolder;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::ImageFormatInfo;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

/* fn graphics_context(device: Arc<Device>, queue: Arc<Queue>, surface: Arc<Surface<Window>>) -> GraphicsContext {
    GraphicsContext::new(device, queue, surface)
} */

/// This function initiates framebuffers, renderbuffers and pipeline
pub fn prepare_graphics(
    // gc: &GraphicsContext,
    device: Arc<Device>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: impl IntoIterator<Item = Arc<SwapchainImage<Window>>>,
    depth_format: Format,
    dimensions: [f32; 2],
) -> (StandardVulcanPipeline, UniformHolder) {
    let vs = cs::load_vert_shader(device.clone()).unwrap();
    let fs = cs::load_frag_shader(device.clone()).unwrap();

    let mut frame_holder = FramebufferFrameSwapper::new();
    let pipe = StandardVulcanPipeline::new()
        .fs(fs)
        .vs(vs)
        .depth_format(depth_format)
        .dimensions(dimensions)
        .format(swapchain.image_format())
        .images(swapchain_images)
        .build(device.clone(), &mut frame_holder);

    let uniholder = UniformHolder::new(device.clone(), pipe.pipeline.clone());

    (pipe, uniholder)
}

fn main() {
    let mut logger = Box::new(FileLogger::new(logger::create_logfile()));

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
                if let Ok(Some(_)) = props {
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
    // let gc = Arc::new(graphics_context(device.clone(), queue.clone()));

    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

    let (swapchain, images) = Swapchain::new(
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

    let mut gc = {
        let mut gcb = GraphicsContextBuilder::new();
        gcb.init_device(device.clone())
            .init_queue(queue.clone())
            .init_surface(surface.clone());
        gcb.init_swapchain(swapchain.clone())
            .init_images(images.clone());
        // let gc = graphics_context(device.clone(), queue.clone(), surface.clone());
        // let vertex_buffer = mesh.vbo();
        let (pipeline, uniform_holder) = prepare_graphics(
            device.clone(),
            swapchain.clone(),
            images.clone(),
            depth_format,
            [dimensions.width as f32, dimensions.height as f32],
        );
        gcb.init_pipeline(pipeline.clone())
            .init_uniforms(uniform_holder.clone());
        gcb.build().expect("error building graphics context")
    };

    let mesh = {
        // Load mesh data from fbx file
        let file_name = input_helpers::ask::<String>("enter fbx file name").unwrap();
        let vertex_data = fbx::read_fbx_document(&file_name, &mut *logger).unwrap();
        Mesh::from_vertex_vec(gc.clone(), vertex_data)
    };

    // let mut mesh_vbo = VertexBufferStreaming::new(mesh.vertex_buffer());
    gc.pipeline.renderers.push(Arc::new(mesh));

    //logger.log(format!("{:#?}\n", caps).as_str());
    let app = AppCreateStruct {
        depth_format,
        gc,
        logger: Box::new(*logger),
    };
    let mut app = App::new(app);
    // app.logger.log("Cant see whats behind you\n");

    event_loop.run(move |event, _, control_flow| app.event_loop(event, control_flow));
}
