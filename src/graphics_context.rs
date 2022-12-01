use crate::pipeline::{FramebufferFrameSwapper, VulkanPipeline, StandardVulcanPipeline};
use crate::uniforms::UniformHolder;
use crate::StandardCommandBuilder;
use crate::image_library::{ImageLibrary, StandardImageBuffer};
use anyhow::anyhow;
use image::DynamicImage;
use vulkano::descriptor_set::{WriteDescriptorSet, PersistentDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::format::Format;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sampler::{Sampler, SamplerCreateInfo, Filter, SamplerAddressMode};
use winit::dpi::PhysicalSize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::image::SwapchainImage;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError, self, AcquireError, SwapchainPresentInfo};
use vulkano::sync::{self, GpuFuture, FenceSignalFuture, FlushError};
use winit::window::Window;

// enums describing communication between primary and secondary threads

type Finisher = Box<dyn FnOnce() -> () + Send>;

#[derive(Debug)]
pub enum GraphicsContextError {
    SwapchainError(AcquireError),
    FlushError(FlushError),
    SuboptimalImage,
}

impl Display for GraphicsContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapchainError(err) => f.write_fmt(format_args!("GC Error: {}", err)),
            Self::SuboptimalImage => f.write_str("Suboptimal Image"),
            Self::FlushError(err) => f.write_fmt(format_args!("GC Flush error: {}", err)),
        }
    }
}

impl From<AcquireError> for GraphicsContextError {
    fn from(e: AcquireError) -> Self {
        Self::SwapchainError(e)
    }
}

impl From<FlushError> for GraphicsContextError {
    fn from(e: FlushError) -> Self {
        Self::FlushError(e)
    }
}

impl std::error::Error for GraphicsContextError {}

enum SecondaryCommand {
    Stop,
    RunCommandBuffer(PrimaryAutoCommandBuffer, Finisher),
}

enum ShaderBinding {
    UniformBinding(u32, UniformHolder),
    TextureBinding(u32, Arc<Sampler>, String),
    StandardUniform(u32),
}

pub struct GCDeviceInternals {
    device: Arc<Device>,
    queue: Arc<Queue>,
    cached_commands: Arc<RefCell<HashMap<String, Arc<dyn PrimaryCommandBufferAbstract>>>>, // command_builder: StandardCommandBuilder,
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>,
    fences_in_flight: Vec<RefCell<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>>,
}

impl GCDeviceInternals {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }
    pub fn images(&self) -> &Vec<Arc<SwapchainImage>> {
        &self.images
    }

    pub fn recreate_swapchain(&mut self) {
        let dimensions = self.window_dimensions();
        let (new_swapchain, images) =
            match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: dimensions.into(),
                ..self.swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                    return;
                }
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };
        self.swapchain = new_swapchain;
        self.images = images;
    }

    pub fn window_dimensions(&self) -> PhysicalSize<u32> {
        let window = self
            .surface
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap();
        window.inner_size()
    }
}

pub struct GCPipelineInternals {
    pipeline: VulkanPipeline<FramebufferFrameSwapper>,
    uniform_holder: UniformHolder,
    texture_library: ImageLibrary,
    shader_bindings: Vec<ShaderBinding>,
    depth_format: Format,
}

impl GCPipelineInternals {
    pub fn pipeline(&self) -> &VulkanPipeline<FramebufferFrameSwapper> {
        &self.pipeline
    }
    pub fn mut_pipeline(&mut self) -> &mut VulkanPipeline<FramebufferFrameSwapper> {
        &mut self.pipeline
    }
}

pub struct GCAllocators {
    pub standard_allocator: Arc<StandardMemoryAllocator>,
    descriptor_allocator: StandardDescriptorSetAllocator,
    standard_cb_allocator: StandardCommandBufferAllocator,
}

pub struct GCWorkerThread {
    secondary_queue: Arc<Queue>,
    secondary_thread_handle: RefCell<Option<std::thread::JoinHandle<()>>>,
    secondary_sender: Sender<SecondaryCommand>,
    secondary_receiver: RefCell<Option<Receiver<SecondaryCommand>>>,
}

/// # GraphicsContext
/// Graphics context is an entry point for many graphical operations,
/// like rendering, attaching new renderables, or executing command buffers
/// on a secondary thread.
/// 
/// ## Elements
/// Individual parts are segregated into different substructures by 
/// *usage/function*. 
pub struct GraphicsContext {
    dev: GCDeviceInternals,
    pipe: GCPipelineInternals,
    alloc: GCAllocators,
    jobs: GCWorkerThread,
}

pub struct GCDevAlloc<'a> {
    dev: &'a GCDeviceInternals,
    alloc: &'a GCAllocators,
}

impl<'a> GCDevAlloc<'a> {
    pub fn create_command_builder(&self) -> StandardCommandBuilder {
        AutoCommandBufferBuilder::primary(
            &self.alloc.standard_cb_allocator,
            self.dev.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }
    pub fn alloc(&self) -> &GCAllocators {
        self.alloc
    }
}

pub struct GCDevJobs<'a> {
    dev: &'a GCDeviceInternals,
    jobs: &'a GCWorkerThread,
}

impl<'a> GCDevJobs<'a> {
    pub fn run_secondary_action(
        &self,
        cmb: PrimaryAutoCommandBuffer,
        finished_sig: Finisher,
    ) {
        // start the thread if it's not running
        let mut current_thr = self.jobs.secondary_thread_handle.borrow_mut();
        let maybe_rx = self.jobs.secondary_receiver.borrow_mut().take();
        current_thr.get_or_insert_with(|| {
            let qdev = self.dev.device.clone();
            let queue_thread = self.jobs.secondary_queue.clone();
            let rx = maybe_rx.unwrap();
            // let rx = self.secondary_receiver.take().unwrap();
            std::thread::spawn(move || {
                let mut running = true;
                while running {
                    let cmd = rx.recv().unwrap();
                    match cmd {
                        SecondaryCommand::Stop => {
                            running = false;
                        }
                        SecondaryCommand::RunCommandBuffer(cmb, finish_signal) => {
                            let f = sync::now(qdev.clone());
                            let j = f
                                .then_execute(queue_thread.clone(), cmb)
                                .unwrap()
                                .then_signal_fence_and_flush()
                                .unwrap();
                            j.wait(None)
                                .expect("waiting on task in secondary thread failed");
                            finish_signal();
                            //*lock = true;
                        }
                    }
                }
            })
        });
        self.jobs.secondary_sender
            .send(SecondaryCommand::RunCommandBuffer(cmb, finished_sig))
            .expect("Channel send failed?!");
    }
}

#[derive(Default)]
pub struct GraphicsContextBuilder {
    pub device: Option<Arc<Device>>,
    pub queue: Option<Arc<Queue>>,
    pub secondary_queue: Option<Arc<Queue>>,
    pub surface: Option<Arc<Surface>>,
    pub swapchain: Option<Arc<Swapchain>>,
    pub images: Option<Vec<Arc<SwapchainImage>>>,
    pub pipeline: Option<VulkanPipeline<FramebufferFrameSwapper>>,
    pub uniform_holder: Option<UniformHolder>,
    pub allocator: Option<Arc<StandardMemoryAllocator>>,
    pub depth_format: Option<Format>,
}

impl GraphicsContextBuilder {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn init_device(mut self, dev: Arc<Device>) -> Self {
        self.device = Some(dev);
        self
    }
    pub fn init_queue(mut self, q: Arc<Queue>) -> Self {
        self.queue = Some(q);
        self
    }
    pub fn init_secondary_queue(mut self, q: Arc<Queue>) -> Self {
        self.secondary_queue = Some(q);
        self
    }
    pub fn init_surface(mut self, s: Arc<Surface>) -> Self {
        self.surface = Some(s);
        self
    }
    pub fn init_swapchain(mut self, s: Arc<Swapchain>) -> Self {
        self.swapchain = Some(s);
        self
    }
    pub fn init_images(mut self, s: Vec<Arc<SwapchainImage>>) -> Self {
        self.images = Some(s);
        self
    }
    pub fn init_pipeline(mut self, s: VulkanPipeline<FramebufferFrameSwapper>) -> Self {
        self.pipeline = Some(s);
        self
    }
    pub fn init_uniforms(mut self, s: UniformHolder) -> Self {
        self.uniform_holder = Some(s);
        self
    }
    pub fn init_allocator(mut self, a: Arc<StandardMemoryAllocator>) -> Self {
        self.allocator = Some(a);
        self
    }
    pub fn init_depth_format(mut self, a: Format) -> Self {
        self.depth_format = Some(a);
        self
    }
    pub fn build(self) -> anyhow::Result<GraphicsContext> {
        let dev = self.device.ok_or(anyhow!("No device present"))?;
        let que = self.queue.ok_or(anyhow!("No queue present"))?;
        let que2 = self
            .secondary_queue
            .ok_or(anyhow!("No secondary queue present"))?;
        let surface = self.surface.ok_or(anyhow!("No surface present"))?;
        let swapchain = self.swapchain.ok_or(anyhow!("No swapchain present"))?;
        let images = self.images.ok_or(anyhow!("No images present"))?;
        let pipeline = self.pipeline.ok_or(anyhow!("No pipeline present"))?;
        let uniforms = self.uniform_holder.ok_or(anyhow!("No uniforms present"))?;
        let allocator = self.allocator.ok_or(anyhow!("No allocator present"))?;

        let (tx, rx) = channel::<SecondaryCommand>();

        // create image library here
        let cb_allocator = StandardCommandBufferAllocator::new(dev.clone(), Default::default());
        let cbb = AutoCommandBufferBuilder::primary(&cb_allocator, que.queue_family_index(), CommandBufferUsage::OneTimeSubmit);
        let texture_library = ImageLibrary::new(&allocator, cbb.unwrap(), Box::new(|cb_builder| {
            let cb = cb_builder.build().unwrap();
            let f = sync::now(dev.clone());
            let j = f
                .then_execute(que.clone(), cb)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();
            j.wait(None)
                .expect("waiting on task in secondary thread failed");
        }));

        let images_count = images.len();
        let mut fences_in_flight: Vec<RefCell<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>> =
            Vec::with_capacity(images_count);
        for _ in 0..images_count {
            fences_in_flight.push(RefCell::new(None));
        }

        let int_alloc = GCAllocators {
            standard_allocator: allocator,
            descriptor_allocator: StandardDescriptorSetAllocator::new(dev.clone()),
            standard_cb_allocator: cb_allocator,
        };

        let int_dev = GCDeviceInternals {
            device: dev,
            queue: que,
            cached_commands: Arc::new(RefCell::new(HashMap::new())),
            surface,
            swapchain,
            images,
            fences_in_flight,
        };

        let int_pipe = GCPipelineInternals {
            pipeline,
            uniform_holder: uniforms,
            texture_library,
            shader_bindings: vec![ShaderBinding::StandardUniform(0)],
            depth_format: self.depth_format.expect("No depth format provided"),
        };

        let int_thr = GCWorkerThread {
            secondary_queue: que2,
            secondary_thread_handle: Default::default(),
            secondary_sender: tx,
            secondary_receiver: RefCell::new(Some(rx)),
        };

        Ok(GraphicsContext {
            dev: int_dev,
            alloc: int_alloc,
            pipe: int_pipe,
            jobs: int_thr,
        })
    }
}

impl GraphicsContext {
    /// This function initiates framebuffers, renderbuffers and pipeline
    pub fn prepare_graphics(
        // gc: &GraphicsContext,
        allocator: Arc<StandardMemoryAllocator>,
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
        swapchain_images: impl IntoIterator<Item = Arc<SwapchainImage>>,
        depth_format: Format,
        dimensions: [f32; 2],
    ) -> (StandardVulcanPipeline, UniformHolder) {
        let vs = crate::cs::load_vert_shader(device.clone()).unwrap();
        let fs = crate::cs::load_frag_shader(device.clone()).unwrap();

        let mut frame_holder = FramebufferFrameSwapper::new();
        let pipe = StandardVulcanPipeline::new()
            .fs(fs)
            .vs(vs)
            .depth_format(depth_format)
            .dimensions(dimensions)
            .format(swapchain.image_format())
            .images(swapchain_images)
            .build(&*allocator, device.clone(), &mut frame_holder);

        let uniholder = UniformHolder::new(allocator);

        (pipe, uniholder)
    }

    pub fn reset_pipeline(
        &mut self,
        // swapchain_images: impl IntoIterator<Item = Arc<SwapchainImage>>,
        // depth_format: Format,
        // dimensions: [f32; 2],
        // old_pipeline: &mut crate::pipeline::StandardVulcanPipeline,
    ) {
        let (p, u) = Self::prepare_graphics(
            self.alloc.standard_allocator.clone(),
            self.dev.device.clone(),
            self.dev.swapchain.clone(),
            self.dev.images.iter().cloned(),
            self.pipe.depth_format,
            self.dev.window_dimensions().into(),
        );
        let old_pipe = std::mem::replace(&mut self.pipe.pipeline, p);
        self.pipe.pipeline.renderers = old_pipe.renderers;
        self.pipe.uniform_holder = u;
        // p.renderers = old_pipeline.renderers;
        // *old_pipeline = p;
        // u
    }
}

impl GraphicsContext {

    pub fn dev_alloc<'a>(dev: &'a GCDeviceInternals, alloc: &'a GCAllocators) -> GCDevAlloc<'a> {
        GCDevAlloc { dev, alloc }
    }

    pub fn dev_jobs<'a>(dev: &'a GCDeviceInternals, jobs: &'a GCWorkerThread) -> GCDevJobs<'a> {
        GCDevJobs { dev, jobs }
    }

    pub fn create_command_builder(&self) -> StandardCommandBuilder {
        /* AutoCommandBufferBuilder::primary(
            &self.alloc.standard_cb_allocator,
            self.dev.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap() */
        Self::dev_alloc(&self.dev, &self.alloc).create_command_builder()
    }

    #[allow(dead_code)]
    pub fn command_buffer_cached<F>(
        &mut self,
        name: String,
        initializer_fn: F,
    ) -> Arc<dyn PrimaryCommandBufferAbstract>
    where
        F: FnOnce(&GraphicsContext) -> Arc<dyn PrimaryCommandBufferAbstract>,
    {
        let mut cached_commands = self.dev.cached_commands.borrow_mut();
        match cached_commands.get(&name) {
            Some(buff) => buff.to_owned(),
            None => {
                let buff = initializer_fn(self);
                cached_commands.insert(name, buff.clone());
                buff
            }
        }
    }

    pub fn run_secondary_action(
        &self,
        cmb: PrimaryAutoCommandBuffer,
        finished_sig: Finisher,
    ) {
        Self::dev_jobs(&self.dev, &self.jobs).run_secondary_action(cmb, finished_sig)
    }

    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.pipe.pipeline.descriptor_layout()
    }

    pub fn generate_descriptor_set(&self) -> Vec<WriteDescriptorSet> {
        // let desc = self.uniform_holder.write_descriptor();
        // let desc = desc.expect("Failed to generate descriptor set: uniform not ready");
        // vec![desc]
        self.pipe.shader_bindings.iter().map(|a| {
            match a {
                ShaderBinding::UniformBinding(binding, uni) => {
                    uni.write_descriptor(*binding).expect("Failed to generate descriptor set: uniform not ready")
                }
                ShaderBinding::TextureBinding(binding, sampler, tex_key) => {
                    let sampler = sampler.clone();
                    let texture = self.pipe.texture_library.get_image(tex_key);
                    let texture = texture.map(|v| v.image_view()).flatten();
                    let texture = texture.unwrap_or(self.pipe.texture_library.fallback_texture.image_view().unwrap());
                    // generate write descriptor for the texture here
                    WriteDescriptorSet::image_view_sampler(*binding, texture, sampler)
                }
                ShaderBinding::StandardUniform(binding) => {
                    self.pipe.uniform_holder.write_descriptor(*binding).expect("failed to generate descriptor set: uniform not ready")
                }
            }
        }).collect()
    }

    pub fn render_loop(&self) -> Result<(), GraphicsContextError> {
        let (image_id, suboptimal, acquire_future) =
            swapchain::acquire_next_image(self.dev.swapchain.clone(), None)?;
        if suboptimal {
            return Err(GraphicsContextError::SuboptimalImage);
        }

        // get the future associated with the image id
        // can unwrap safely because image index is within this vector by construction
        let previous_future = self.dev.fences_in_flight.get(image_id as usize).unwrap();
        let mut previous_future = previous_future.borrow_mut();
        if let Some(previous_future) =
            previous_future.take()
        {
            // previous_future.cleanup_finished();
            // TODO: what do we do on flush error here if it ever happens?
            // for now we just panic
            previous_future.wait(None).unwrap();
        }
        drop(previous_future);

        // Create command buffer
        let mut comm_builder = self.create_command_builder();
        self.begin_render(&mut comm_builder, image_id as usize);
        self.render(&mut comm_builder);
        self.end_render(&mut comm_builder);

        let comm_buffer = comm_builder.build().unwrap();
        // let branch_duration_prefence = branch_start_time.elapsed().as_micros();
        let exec = sync::now(self.dev.device.clone())
            .join(acquire_future)
            .then_execute(self.dev.queue.clone(), comm_buffer)
            .unwrap()
            .then_swapchain_present(
                self.dev.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.dev.swapchain.clone(),
                    image_id,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush()?;
        // fences_in_flight.push(Some(future));
        let mut fence_ptr = self.dev.fences_in_flight[image_id as usize].borrow_mut();
        *fence_ptr = Some(exec);
        // self.fences_in_flight[image_id as usize] = Some(future);
        // future.wait(None).unwrap();
        Ok(())
    }

    pub fn begin_render(&self, command_builder: &mut StandardCommandBuilder, frameindex: usize) {
        // creating descriptor set for uniform block and for images
        let pds = PersistentDescriptorSet::new(&self.alloc.descriptor_allocator, 
            self.descriptor_set_layout(), 
            self.generate_descriptor_set()).unwrap();
        self.pipe.pipeline.begin_render(command_builder, frameindex);
        command_builder.bind_descriptor_sets(PipelineBindPoint::Graphics, 
            self.pipe.pipeline.pipeline_layout(), 0, pds);
    }

    pub fn render(&self, command_builder: &mut StandardCommandBuilder) {
        self.pipe.pipeline.render(&self, command_builder);
    }

    pub fn end_render(&self, command_builder: &mut StandardCommandBuilder) {
        self.pipe.pipeline.end_render(command_builder)
    }

    pub fn create_texture(&mut self, img: DynamicImage, key: impl ToString) {
        let img = img.flipv();
        let img = img.into_rgba8();
        self.pipe.texture_library.insert_image(key, 
            &img, 
            Self::dev_alloc(&self.dev, &self.alloc), 
            Self::dev_jobs(&self.dev, &self.jobs));
    }

    pub fn activate_texture(&mut self, key: impl ToString, binding: usize) -> anyhow::Result<()>{
        let res = self.pipe.shader_bindings.iter_mut().find_map(|val|
            if let ShaderBinding::TextureBinding(b, _, k) = val {
                if *b == binding as u32 {
                    *k = key.to_string();
                    Some((b, k))
                } else { None }
            } 
            else {None});
        if res.is_none() {
            // texture binding was not found, insert at position
            let my_sampler = Sampler::new(self.dev.device.clone(), SamplerCreateInfo {
                min_filter: Filter::Nearest,
                mag_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            })?;
            // let im_view = self.pipe.texture_library.get_image(key).unwrap();
            self.pipe.shader_bindings.push(ShaderBinding::TextureBinding(binding as u32, my_sampler, key.to_string()));
        }
        Ok(())
    }

    pub fn mut_pipeline(&mut self) -> &mut VulkanPipeline<FramebufferFrameSwapper> {
        &mut self.pipe.pipeline
    }

    pub fn dev(&self) -> &GCDeviceInternals {
        &self.dev
    }

    pub fn mut_dev(&mut self) -> &mut GCDeviceInternals {
        &mut self.dev
    }

    pub fn alloc(&self) -> &GCAllocators {
        &self.alloc
    }

    pub fn mut_uniforms(&mut self) -> &mut UniformHolder {
        &mut self.pipe.uniform_holder
    }
}

impl Drop for GraphicsContext {
    // unused_must_use: GraphicsContext is dropped only when the application finishes,
    // so it doesnt matter what join() returnes or why.
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        self.jobs.secondary_sender.send(SecondaryCommand::Stop);
        if let Some(thr) = self.jobs.secondary_thread_handle.take() {
            thr.join();
            println!("{}", "Joined secondary thread");
        }
    }
}
