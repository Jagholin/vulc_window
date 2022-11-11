use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::PipelineLayout;

use crate::cs::ty::MatBlock;

pub trait UniformStruct {
    fn apply_uniforms(&self, comm_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>);
}

// #[derive(Clone)]
pub struct UniformHolder {
    // pub view_matrix: cs::ty::MatBlock,
    pub uniform_buffer_pool: CpuBufferPool<MatBlock>,
    pub desc_layout: Arc<DescriptorSetLayout>,
    // pub desc_set: Arc<PersistentDescriptorSet>,
    pipe_layout: Arc<PipelineLayout>,
    allocator: StandardDescriptorSetAllocator,
}

pub struct UniformApplier(Arc<PersistentDescriptorSet>, Arc<PipelineLayout>);

impl UniformHolder {
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        device: Arc<Device>,
        pipeline: Arc<GraphicsPipeline>,
    ) -> Self {
        /* let initial_data = cs::ty::MatBlock {
            view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }; */
        let buff: CpuBufferPool<MatBlock> = CpuBufferPool::uniform_buffer(allocator);
        buff.reserve(64)
            .expect("Cant reserve space for uniform data");

        /* let buff = CpuAccessibleBuffer::from_data(
            gc.device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            initial_data,
        )
        .unwrap(); */
        let pipe_layout = pipeline.layout().clone();
        let desc_sets = pipe_layout.set_layouts();
        for ref ds in desc_sets {
            for b in ds.bindings() {
                println!("Binding {} is of type {:?}", b.0, b.1.descriptor_type);
            }
        }
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
        // let desc_set = PersistentDescriptorSet::new(
        //    desc_layout,
        //     [WriteDescriptorSet::buffer(0, buff.clone())],
        //)
        //.unwrap();
        Self {
            uniform_buffer_pool: buff,
            desc_layout,
            // desc_set,
            pipe_layout,
            allocator: StandardDescriptorSetAllocator::new(device),
        }
    }

    pub fn set_view_matrix(&self, value: [[f32; 4]; 4]) -> UniformApplier {
        // let mut write_lock = self.uniform_buffer_pool.write().unwrap();
        // write_lock.view_matrix = value;
        // drop()s writeLock
        let buffer = self
            .uniform_buffer_pool
            .try_next(MatBlock { view_matrix: value })
            .unwrap();
        let desc_set = PersistentDescriptorSet::new(
            &self.allocator,
            self.desc_layout.clone(),
            [WriteDescriptorSet::buffer(0, buffer)],
        )
        .unwrap();
        UniformApplier(desc_set, self.pipe_layout.clone())
    }
}

impl UniformStruct for UniformApplier {
    fn apply_uniforms(
        &self,
        comm_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        comm_builder.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.1.clone(),
            0,
            self.0.clone(),
        );
    }
}
