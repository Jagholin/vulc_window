use std::sync::Arc;
use vulkano::{pipeline::GraphicsPipeline, render_pass::Framebuffer};
use vulkano::buffer::{BufferContents, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents};

type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;
pub trait IssueCommands {
    fn issue_commands(&self, command_builder: &mut StandardCommandBuilder);
}

struct VertexBufferRenderPreparation<T>
where [T]: BufferContents {
    pipeline: Arc<GraphicsPipeline>,
    vbo: Arc<CpuAccessibleBuffer<[T]>>
}

pub struct VertexBufferRenderer<T>
where [T]: BufferContents {
    fb: Arc<Framebuffer>,
    // pipeline: Arc<GraphicsPipeline>,
    // vbo: Arc<CpuAccessibleBuffer<[T]>>,
    prep: Arc<VertexBufferRenderPreparation<T>>,
}

impl<T> VertexBufferRenderer<T>
where [T]: BufferContents {
    pub fn new (fbs: &[Arc<Framebuffer>], pipeline: Arc<GraphicsPipeline>, vbo: Arc<CpuAccessibleBuffer<[T]>>) -> Vec<VertexBufferRenderer<T>> {
        let prep = Arc::new(VertexBufferRenderPreparation {
            pipeline, vbo
        });
        fbs.iter().map(|fb| VertexBufferRenderer {fb: fb.clone(), prep: prep.clone()}).collect()
    }
}

impl<T> IssueCommands for VertexBufferRenderer<T>
where [T]: BufferContents {
    fn issue_commands(&self, command_builder: &mut StandardCommandBuilder) {
        command_builder.begin_render_pass(self.fb.clone(), SubpassContents::Inline, vec![[0.0, 0.0, 0.0, 1.0].into()])
            .unwrap()
            .bind_pipeline_graphics(self.prep.pipeline.clone())
            .bind_vertex_buffers(0, self.prep.vbo.clone())
            .draw(3, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();
    }
}
