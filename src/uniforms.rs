use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::PipelineLayout;

use crate::cs;
use crate::cs::ty::MatBlock;
use crate::graphics_context::GraphicsContext;

pub trait UniformStruct {
    fn apply_uniforms(&self, comm_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>);
}

pub struct UniformHolder {
    // pub view_matrix: cs::ty::MatBlock,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<MatBlock>>,
    pub pipe_layout: Arc<PipelineLayout>,
    pub desc_set: Arc<PersistentDescriptorSet>,
}

impl UniformHolder {
    pub fn new(gc: &GraphicsContext, pipeline: Arc<GraphicsPipeline>) -> Self {
        let initial_data = cs::ty::MatBlock {
            view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let buff = CpuAccessibleBuffer::from_data(
            gc.device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            initial_data,
        )
        .unwrap();
        let pipe_layout = pipeline.layout().clone();
        let desc_sets = pipe_layout.set_layouts();
        let desc_layout = desc_sets
            .into_iter()
            .find(|d| {
                if let Some(bindings) = d.bindings().get(&0) {
                    if bindings.descriptor_type == DescriptorType::UniformBuffer
                        && bindings.stages.vertex
                    {
                        return true;
                    }
                }
                false
            })
            .unwrap()
            .to_owned();
        let desc_set = PersistentDescriptorSet::new(
            desc_layout,
            [WriteDescriptorSet::buffer(0, buff.clone())],
        )
        .unwrap();
        Self {
            uniform_buffer: buff,
            pipe_layout,
            desc_set,
        }
    }

    pub fn set_view_matrix(&self, value: [[f32; 4]; 4]) {
        let mut write_lock = self.uniform_buffer.write().unwrap();
        write_lock.view_matrix = value;
        // drop()s writeLock
    }
}

impl UniformStruct for UniformHolder {
    fn apply_uniforms(
        &self,
        comm_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        comm_builder.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipe_layout.clone(),
            0,
            self.desc_set.clone(),
        );
    }
}
