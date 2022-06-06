use crate::StandardCommandBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::device::Device;
use vulkano::device::Queue;
pub struct GraphicsContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    cached_commands: HashMap<String, Arc<dyn PrimaryCommandBuffer>>, // command_builder: StandardCommandBuilder,
                                                                     // pipeline: Arc<ComputePipeline>,
                                                                     // descriptor_set: Arc<PersistentDescriptorSet>,
}

impl GraphicsContext {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>, // pipeline: Arc<ComputePipeline>,
                           // command_builder: StandardCommandBuilder
                           /* descriptor_set: Arc<PersistentDescriptorSet>*/
    ) -> GraphicsContext {
        GraphicsContext {
            device,
            queue,
            cached_commands: HashMap::new(), // command_builder,
                                             // pipeline,
                                             // descriptor_set
        }
    }
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
        match self.cached_commands.get(&name) {
            Some(buff) => buff.to_owned(),
            None => {
                let buff = initializer_fn(self);
                self.cached_commands.insert(name, buff.clone());
                buff
            }
        }
    }
}
