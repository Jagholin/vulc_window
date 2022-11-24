use crate::pipeline::{FramebufferFrameSwapper, VulkanPipeline};
use crate::uniforms::UniformHolder;
use crate::StandardCommandBuilder;
use anyhow::anyhow;
use vulkano::descriptor_set::{WriteDescriptorSet, PersistentDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::pipeline::PipelineBindPoint;
use std::cell::RefCell;
use std::collections::HashMap;
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
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::sync::{self, GpuFuture};

// enums describing communication between primary and secondary threads

type Finisher = Box<dyn FnOnce() -> () + Send>;

enum SecondaryCommand {
    Stop,
    RunCommandBuffer(PrimaryAutoCommandBuffer, Finisher),
}

// #[derive(Clone)]
pub struct GraphicsContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    secondary_queue: Arc<Queue>,
    cached_commands: Arc<RefCell<HashMap<String, Arc<dyn PrimaryCommandBufferAbstract>>>>, // command_builder: StandardCommandBuilder,
    // pipeline: Arc<ComputePipeline>,
    // descriptor_set: Arc<PersistentDescriptorSet>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<SwapchainImage>>,
    pub pipeline: VulkanPipeline<FramebufferFrameSwapper>,
    pub uniform_holder: UniformHolder,
    pub standard_allocator: Arc<StandardMemoryAllocator>,
    descriptor_allocator: StandardDescriptorSetAllocator,
    standard_cb_allocator: StandardCommandBufferAllocator,
    secondary_thread_handle: RefCell<Option<std::thread::JoinHandle<()>>>,
    secondary_sender: Sender<SecondaryCommand>,
    secondary_receiver: RefCell<Option<Receiver<SecondaryCommand>>>,
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
        /* let qdev = dev.clone();
        let queue_thread = que2.clone();
        let action_thread = std::thread::spawn(move || {
            let mut running = true;
            while running {
                let cmd = rx.recv().unwrap();
                match cmd {
                    SecondaryCommand::Stop => {
                        running = false;
                    }
                    SecondaryCommand::RunCommandBuffer(cmb, finish_signal) => {
                        let f = sync::now(qdev.clone());
                        let j = f.then_execute(queue_thread.clone(), cmb)
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap();
                        j.wait(None);
                        let mut lock = finish_signal.write().unwrap();
                        *lock = true;
                    }
                }
            }
        }); */

        Ok(GraphicsContext {
            device: dev.clone(),
            queue: que,
            secondary_queue: que2,
            cached_commands: Arc::new(RefCell::new(HashMap::new())),
            surface,
            swapchain,
            images,
            pipeline,
            uniform_holder: uniforms,
            standard_allocator: allocator,
            descriptor_allocator: StandardDescriptorSetAllocator::new(dev.clone()),
            standard_cb_allocator: StandardCommandBufferAllocator::new(dev, Default::default()),
            secondary_thread_handle: Default::default(),
            secondary_sender: tx,
            secondary_receiver: RefCell::new(Some(rx)),
        })
    }
}

impl GraphicsContext {
    pub fn create_command_builder(&self) -> StandardCommandBuilder {
        AutoCommandBufferBuilder::primary(
            &self.standard_cb_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
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

    pub fn run_secondary_action(
        &self,
        cmb: PrimaryAutoCommandBuffer,
        finished_sig: Finisher,
    ) {
        // start the thread if it's not running
        let mut current_thr = self.secondary_thread_handle.borrow_mut();
        let maybe_rx = self.secondary_receiver.borrow_mut().take();
        current_thr.get_or_insert_with(|| {
            let qdev = self.device.clone();
            let queue_thread = self.secondary_queue.clone();
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
        self.secondary_sender
            .send(SecondaryCommand::RunCommandBuffer(cmb, finished_sig))
            .expect("Channel send failed?!");
    }

    pub fn quit(self) {
        unimplemented!();
        // self.secondary_sender.send(SecondaryCommand::Stop);
        // self.secondary_thread_handle.join();
    }

    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.pipeline.descriptor_layout()
    }

    pub fn generate_descriptor_set(&self) -> Vec<WriteDescriptorSet> {
        let desc = self.uniform_holder.write_descriptor();
        let desc = desc.expect("Failed to generate descriptor set: uniform not ready");
        vec![desc]
    }

    pub fn begin_render(&self, command_builder: &mut StandardCommandBuilder, frameindex: usize) {
        // creating descriptor set for uniform block and for images
        let pds = PersistentDescriptorSet::new(&self.descriptor_allocator, 
            self.descriptor_set_layout(), 
            self.generate_descriptor_set()).unwrap();
        self.pipeline.begin_render(command_builder, frameindex);
        command_builder.bind_descriptor_sets(PipelineBindPoint::Graphics, 
            self.pipeline.pipeline_layout(), 0, pds);
    }

    pub fn render(&self, command_builder: &mut StandardCommandBuilder) {
        self.pipeline.render(&self, command_builder);
    }

    pub fn end_render(&self, command_builder: &mut StandardCommandBuilder) {
        self.pipeline.end_render(command_builder)
    }
}

impl Drop for GraphicsContext {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        self.secondary_sender.send(SecondaryCommand::Stop);
        if let Some(thr) = self.secondary_thread_handle.take() {
            thr.join();
            println!("{}", "Joined secondary thread");
        }
    }
}
