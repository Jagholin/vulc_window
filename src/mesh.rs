use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use crate::vertex_type::VertexStruct;
use crate::renderer::VertexBufferRenderer;
use crate::graphics_context::GraphicsContext;
use std::sync::Arc;

fn create_default_vertex_vec() -> Vec<VertexStruct>{
    let v1 = VertexStruct{position: [-0.5, -0.5]};
    let v2 = VertexStruct{position: [0.0, 0.5]};
    let v3 = VertexStruct{position: [0.5, -0.25]};
    vec![v1, v2, v3]
    // CpuAccessibleBuffer::from_iter(gc.device.clone(), BufferUsage::vertex_buffer(), false, [v1, v2, v3].into_iter()).unwrap()
}

pub struct Mesh<'a> {
    vertices: Vec<VertexStruct>,
    renderer: Option<VertexBufferRenderer<VertexStruct>>,
    vbo: Option<Arc<CpuAccessibleBuffer<[VertexStruct]>>>,
    gc: &'a GraphicsContext,
}

impl<'a> Mesh<'a> {
    pub fn new(gc: &'a GraphicsContext) -> Mesh<'a> {
        Mesh { vertices: create_default_vertex_vec(), renderer: None, vbo: None, gc: gc }
    }

    pub fn vbo(&mut self) -> Arc<CpuAccessibleBuffer<[VertexStruct]>> {
        //let verts = std::mem::take(self.vertices);
        let res = self.vbo.get_or_insert_with(|| {
            CpuAccessibleBuffer::from_iter(self.gc.device.clone(), BufferUsage::vertex_buffer(), false, self.vertices.iter().cloned()).unwrap()
        });
        res.clone()
    }
}
