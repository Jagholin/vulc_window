#![allow(clippy::redundant_clone)]

use crate::app::{App, AppCreateStruct};
use crate::graphics_context::{GraphicsContextBuilder, GraphicsContext};
use crate::logger::FileLogger;
use crate::mesh::Mesh;
// use crate::renderer::{IssueCommands, VertexBufferStreaming};
use std::sync::Arc;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::ImageFormatInfo;
use vulkano::image::{ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{GenericMemoryAllocatorCreateInfo, StandardMemoryAllocator};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

/* fn graphics_context(device: Arc<Device>, queue: Arc<Queue>, surface: Arc<Surface<Window>>) -> GraphicsContext {
    GraphicsContext::new(device, queue, surface)
} */

pub fn real_main() {
    let mut logger = Box::new(FileLogger::new(crate::logger::create_logfile()));

    let vk_library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&vk_library);
    let instance = Instance::new(
        vk_library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("cant create info");
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (device, queues, caps, image_format, depth_format) = {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_idx) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|i| i.supported_extensions().contains(&device_extensions))
            .filter_map(|d| {
                let families =
                    d.queue_family_properties()
                        .into_iter()
                        .enumerate()
                        .find(|(qid, q)| {
                            let graphics_support =
                                q.supports_stage(vulkano::sync::PipelineStage::AllGraphics);
                            // q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false)
                            let surface_support = d.surface_support(*qid as u32, &surface).unwrap();
                            graphics_support && surface_support
                        });
                let families = families.map(|q| (q.0, q.1.to_owned()));
                families.map(|q| (d, q.0))
                // .map(|q| (d, q.0))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no physical device match the criteria");

        println!(
            "physical device found! {}",
            physical_device.properties().device_name
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_idx as u32,
                    queues: vec![0.5, 0.3],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("creating virtual device failed");

        let queue = queues.next().unwrap();
        let queue2 = queues.next().unwrap();
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
                    usage: ImageUsage {
                        depth_stencil_attachment: true,
                        ..Default::default()
                    },
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
        (device, (queue, queue2), caps, image_format, depth_format)
    };
    // let gc = Arc::new(graphics_context(device.clone(), queue.clone()));
    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format: Some(image_format),
            image_extent: dimensions.into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..Default::default()
            },
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();
    // ImageView::new(image, create_info)
    let allocator = Arc::new(
        StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &[(0, 1024)],
                ..Default::default()
            },
        )
        .unwrap(),
    );

    let mut gc = {
        let gcb = GraphicsContextBuilder::new();
        let gcb = gcb
            .init_device(device.clone())
            .init_queue(queues.0.clone())
            .init_secondary_queue(queues.1.clone())
            .init_surface(surface.clone());
        let gcb = gcb
            .init_swapchain(swapchain.clone())
            .init_images(images.clone());
        // let gc = graphics_context(device.clone(), queue.clone(), surface.clone());
        // let vertex_buffer = mesh.vbo();
        let (pipeline, uniform_holder) = GraphicsContext::prepare_graphics(
            allocator.clone(),
            device.clone(),
            swapchain.clone(),
            images.clone(),
            depth_format,
            [dimensions.width as f32, dimensions.height as f32],
        );
        let gcb = gcb
            .init_pipeline(pipeline.clone())
            .init_uniforms(uniform_holder)
            .init_allocator(allocator)
            .init_depth_format(depth_format);
        gcb.build().expect("error building graphics context")
    };

    let mesh = {
        // Load mesh data from fbx file
        let file_name = crate::input_helpers::ask::<String>("enter fbx file name").unwrap();
        let vertex_data = crate::fbx::read_fbx_document(&file_name, &mut *logger).unwrap();
        Mesh::from_vertex_vec(vertex_data)
    };

    // let mut mesh_vbo = VertexBufferStreaming::new(mesh.vertex_buffer());
    gc.mut_pipeline().renderers.push(Arc::new(mesh));

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
