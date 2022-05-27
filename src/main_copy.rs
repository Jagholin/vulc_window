#![allow(dead_code, unused_variables, clippy::redundant_clone)]

mod cs;
mod input_helpers;
mod logger;

use bytemuck::{Pod, Zeroable};
use logger::Logger;
use vulkano::shader::ShaderModule;
use std::sync::Arc;
use std::collections::HashMap;
use image::{ImageBuffer, Rgba};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, BufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::{Format, ClearValue};
use vulkano::image::{view::ImageView, ImageUsage, SwapchainImage, ImageDimensions, StorageImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{compute::ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

struct GraphicsContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    cached_commands: HashMap<String, Arc<dyn PrimaryCommandBuffer>>
    // command_builder: StandardCommandBuilder,
    // pipeline: Arc<ComputePipeline>,
    // descriptor_set: Arc<PersistentDescriptorSet>,
}

impl GraphicsContext {
    fn new(
        device: Arc<Device>,
        queue: Arc<Queue>
        // pipeline: Arc<ComputePipeline>,
        // command_builder: StandardCommandBuilder
        /* descriptor_set: Arc<PersistentDescriptorSet>*/) -> GraphicsContext {
        GraphicsContext {
            device, 
            queue, 
            cached_commands: HashMap::new()
            // command_builder,
            // pipeline,
            // descriptor_set
        }
    }
    fn create_command_builder(&self) -> StandardCommandBuilder {
        AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }
    fn command_buffer_cached<F>(&mut self, name: String, initializer_fn: F) -> Arc<dyn PrimaryCommandBuffer>
    where
        F: FnOnce(&GraphicsContext) -> Arc<dyn PrimaryCommandBuffer> {
        match self.cached_commands.get(&name) {
            Some (buff) => buff.to_owned(),
            None => {
                let buff = initializer_fn(&self);
                self.cached_commands.insert(name, buff.clone());
                buff
            }
        }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Zeroable, Pod)]
struct VertexStruct {
    position: [f32; 2],
}

vulkano::impl_vertex!(VertexStruct);

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
    T: PrimaryCommandBuffer + 'static {
    let future = sync::now(cg.device.clone())
        .then_execute(cg.queue.clone(), buff)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}

fn create_image (gc: &GraphicsContext) -> Arc<StorageImage> {
    let result = StorageImage::new(
        gc.device.clone(),
        ImageDimensions::Dim2d { width: 1024, height: 1024, array_layers: 1 },
        Format::R8G8B8A8_UNORM,
        Some(gc.queue.family())).unwrap();
    println!("image created!");
    result
}

fn clear_image(gc: &GraphicsContext, image: Arc<StorageImage>) {
    
    let mut builder = gc.create_command_builder();
    builder.clear_color_image(image, ClearValue::Float([0.0, 0.0, 1.0, 1.0])).unwrap();
    let command_buffer = builder.build().unwrap();
    // execute command buffer and wait for 

    execute_command_buffer(&gc, command_buffer);
    println!("image cleared!");
}

fn export_image(gc: &GraphicsContext, image: Arc<StorageImage>) {
    let buf = CpuAccessibleBuffer::from_iter(
        gc.device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 1024*1024*4).map(|_| 255u8)
    ).expect("cant create buffer to save the image");
    let mut command_builder = gc.create_command_builder();
    command_builder
        .copy_image_to_buffer(image, buf.clone())
        .unwrap();
    let commands = command_builder.build().unwrap();
    execute_command_buffer(&gc, commands);

    let content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &content[..]).unwrap();
    println!("image exported!");
    image.save("image.png").unwrap();
}

fn bind_data_to(pipeline: Arc<ComputePipeline>, data_buffer: Arc<dyn BufferAccess>) -> Arc<PersistentDescriptorSet> {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .expect("cant create descriptor set")
}

fn shader_pipeline_compute(device: Arc<Device>, queue: Arc<Queue>, shader: Arc<ShaderModule>) -> (Arc<ShaderModule>, Arc<ComputePipeline>) {
    // let shader = cs::load(device.clone()).expect("failed to create shader module");
    let data_buffer = create_integer_buffer(device.clone());

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

fn compute_context(device: Arc<Device>, queue: Arc<Queue>) -> GraphicsContext {
    GraphicsContext::new(device, queue)
}

fn perform_compute(device: Arc<Device>, queue: Arc<Queue>) {
    let gc = compute_context(device.clone(), queue.clone());
    let (shader, pipeline) = shader_pipeline_compute(
        device.clone(), 
        queue.clone(), 
        cs::load_mult_const(device.clone()).expect("failed to create shader module"));
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
    let image = create_image(&gc);
    clear_image(&gc, image.clone());
    export_image(&gc, image);
}

fn main() {
    let mut logger = Logger::new(logger::create_logfile());

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

    perform_compute(device.clone(), queue.clone());
    // return;

    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;
    // println!("{:#?}", physical_device);

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface,
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

    let render_pass = vulkano::single_pass_renderpass!(device.clone(), attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let framebuffers = get_framebuffers(&images, render_pass);

    // log_writer.write_all(format!("{:#?}", physical_device).as_bytes()).expect("cant write into logfile");
    logger.log(format!("{:#?}\n", caps).as_str());
    logger.log("Cant see whats behind you\n");

    event_loop.run(|event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    });
}
