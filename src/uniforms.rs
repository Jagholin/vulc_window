use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::cs::ty::MatBlock;

/* pub trait UniformStruct {
    fn apply_uniforms(&self, comm_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>);
} */ 

// #[derive(Clone)]
pub struct UniformHolder {
    // pub view_matrix: cs::ty::MatBlock,
    uniform_buffer_pool: CpuBufferPool<MatBlock>,
    current_buffer: Option<Arc<CpuBufferPoolSubbuffer<MatBlock>>>,
    // pub desc_layout: Arc<DescriptorSetLayout>,
    // pub desc_set: Arc<PersistentDescriptorSet>,
    // pipe_layout: Arc<PipelineLayout>,
    // allocator: StandardDescriptorSetAllocator,
}

// pub struct UniformApplier(Arc<PersistentDescriptorSet>, Arc<PipelineLayout>);

impl UniformHolder {
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        // device: Arc<Device>,
        // pipeline: Arc<GraphicsPipeline>,
    ) -> Self {
        let buff: CpuBufferPool<MatBlock> = CpuBufferPool::uniform_buffer(allocator);
        buff.reserve(64)
            .expect("Cant reserve space for uniform data");

        /* let pipe_layout = pipeline.layout().clone();
        let desc_sets = pipe_layout.set_layouts();
        for ds in desc_sets {
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
            .to_owned(); */
        // let desc_set = PersistentDescriptorSet::new(
        //    desc_layout,
        //     [WriteDescriptorSet::buffer(0, buff.clone())],
        //)
        //.unwrap();
        Self {
            uniform_buffer_pool: buff,
            current_buffer: None,
            // desc_layout,
            // desc_set,
            // pipe_layout,
            // allocator: StandardDescriptorSetAllocator::new(device),
        }
    }

    pub fn set_view_matrix(&mut self, value: [[f32; 4]; 4]) {
        // let mut write_lock = self.uniform_buffer_pool.write().unwrap();
        // write_lock.view_matrix = value;
        // drop()s writeLock
        let buffer = self
            .uniform_buffer_pool
            .try_next(MatBlock { view_matrix: value })
            .unwrap();
        // let desc_set = PersistentDescriptorSet::new(
        //    &self.allocator,
        //    self.desc_layout.clone(),
        //    [WriteDescriptorSet::buffer(0, buffer)],
        //)
        //.unwrap();
        // UniformApplier(desc_set, self.pipe_layout.clone())
        // TODO: hard coded binding locations
        // WriteDescriptorSet::buffer(0, buffer)
        self.current_buffer = Some(buffer);
    }

    pub fn write_descriptor(&self) -> Option<WriteDescriptorSet> {
        self.current_buffer.clone().map(|b|
            WriteDescriptorSet::buffer(0, b))
    }
}

/* impl UniformStruct for UniformApplier {
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
} */
