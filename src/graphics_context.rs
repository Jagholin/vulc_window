use crate::pipeline::{FramebufferFrameSwapper, VulkanPipeline};
use crate::uniforms::UniformHolder;
use crate::StandardCommandBuilder;
use anyhow::anyhow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::image::SwapchainImage;
use vulkano::swapchain::{Surface, Swapchain};
use winit::window::Window;

#[derive(Clone)]
pub struct GraphicsContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    cached_commands: Arc<RefCell<HashMap<String, Arc<dyn PrimaryCommandBuffer>>>>, // command_builder: StandardCommandBuilder,
    // pipeline: Arc<ComputePipeline>,
    // descriptor_set: Arc<PersistentDescriptorSet>,
    pub surface: Arc<Surface<Window>>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub images: Vec<Arc<SwapchainImage<Window>>>,
    pub pipeline: VulkanPipeline<FramebufferFrameSwapper>,
    pub uniform_holder: UniformHolder,
}

#[derive(Default)]
pub struct GraphicsContextBuilder {
    pub device: Option<Arc<Device>>,
    pub queue: Option<Arc<Queue>>,
    pub surface: Option<Arc<Surface<Window>>>,
    pub swapchain: Option<Arc<Swapchain<Window>>>,
    pub images: Option<Vec<Arc<SwapchainImage<Window>>>>,
    pub pipeline: Option<VulkanPipeline<FramebufferFrameSwapper>>,
    pub uniform_holder: Option<UniformHolder>,
}

impl GraphicsContextBuilder {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn init_device(&mut self, dev: Arc<Device>) -> &mut Self {
        self.device = Some(dev);
        self
    }
    pub fn init_queue(&mut self, q: Arc<Queue>) -> &mut Self {
        self.queue = Some(q);
        self
    }
    pub fn init_surface(&mut self, s: Arc<Surface<Window>>) -> &mut Self {
        self.surface = Some(s);
        self
    }
    pub fn init_swapchain(&mut self, s: Arc<Swapchain<Window>>) -> &mut Self {
        self.swapchain = Some(s);
        self
    }
    pub fn init_images(&mut self, s: Vec<Arc<SwapchainImage<Window>>>) -> &mut Self {
        self.images = Some(s);
        self
    }
    pub fn init_pipeline(&mut self, s: VulkanPipeline<FramebufferFrameSwapper>) -> &mut Self {
        self.pipeline = Some(s);
        self
    }
    pub fn init_uniforms(&mut self, s: UniformHolder) -> &mut Self {
        self.uniform_holder = Some(s);
        self
    }
    pub fn build(self) -> anyhow::Result<GraphicsContext> {
        let dev = self.device.ok_or(anyhow!("No device present"))?;
        let que = self.queue.ok_or(anyhow!("No queue present"))?;
        let surface = self.surface.ok_or(anyhow!("No surface present"))?;
        let swapchain = self.swapchain.ok_or(anyhow!("No swapchain present"))?;
        let images = self.images.ok_or(anyhow!("No images present"))?;
        let pipeline = self.pipeline.ok_or(anyhow!("No pipeline present"))?;
        let uniforms = self.uniform_holder.ok_or(anyhow!("No uniforms present"))?;
        Ok(GraphicsContext {
            device: dev,
            queue: que,
            cached_commands: Arc::new(RefCell::new(HashMap::new())),
            surface,
            swapchain,
            images,
            pipeline,
            uniform_holder: uniforms,
        })
    }
}

impl GraphicsContext {
    pub fn create_command_builder(&self) -> StandardCommandBuilder {
        AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }
    pub fn command_buffer_cached<F>(
        &mut self,
        name: String,
        initializer_fn: F,
    ) -> Arc<dyn PrimaryCommandBuffer>
    where
        F: FnOnce(&GraphicsContext) -> Arc<dyn PrimaryCommandBuffer>,
    {
        let mut cached_commands = self.cached_commands.borrow_mut();
        match cached_commands.get(&name) {
            Some(buff) => buff.to_owned(),
            None => {
                let buff = initializer_fn(self);
                cached_commands.insert(name, buff.clone());
                buff
            }
        }
    }
}
