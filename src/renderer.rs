use std::sync::Arc;
use vulkano::buffer::{BufferContents, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

pub trait VertexBufferProducer<T> {
    fn produce_vbo(&self, verts: &mut u32) -> Arc<T>;
}
pub trait IssueCommands {
    fn issue_commands(&self, command_builder: &mut StandardCommandBuilder);
}

/* struct VertexBufferRenderPreparation<T>
where [T]: BufferContents {
    // pipeline: Arc<GraphicsPipeline>,
    vbo: Arc<CpuAccessibleBuffer<[T]>>,
    verts_count: u32,
}*/ 

pub struct VertexBufferRenderer<T>
where [T]: BufferContents {
    // fb: Arc<Framebuffer>,
    // pipeline: Arc<GraphicsPipeline>,
    vbo: Arc<CpuAccessibleBuffer<[T]>>,
    verts_count: u32,
    // prep: Arc<VertexBufferRenderPreparation<T>>,
}

impl<T> VertexBufferRenderer<T>
where [T]: BufferContents {
    pub fn new (vbo_producer: &impl VertexBufferProducer<CpuAccessibleBuffer<[T]>>) 
    -> Self {
        let mut verts_count: u32 = 0;
        let vbo = vbo_producer.produce_vbo(&mut verts_count);
        Self { vbo, verts_count }
    }
}

impl<T> IssueCommands for VertexBufferRenderer<T>
where [T]: BufferContents {
    fn issue_commands(&self, command_builder: &mut StandardCommandBuilder) {
        command_builder// .begin_render_pass(self.fb.clone(), SubpassContents::Inline, vec![[0.0, 0.0, 0.0, 1.0].into(), ClearValue::Depth(1000.0)])
            // .unwrap()
            // .bind_pipeline_graphics(self.prep.pipeline.clone())
            .bind_vertex_buffers(0, self.vbo.clone())
            .draw(self.verts_count, 1, 0, 0)
            .unwrap()
            // .end_render_pass()
            //.unwrap();
            ;
        
    }
}
