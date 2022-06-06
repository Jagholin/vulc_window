use crate::graphics_context::GraphicsContext;
use crate::renderer::VertexBufferProducer;
use crate::vertex_type::VertexStruct;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};

fn create_default_vertex_vec() -> Vec<VertexStruct> {
    let v1 = VertexStruct {
        position: [-0.5, -0.5, 0.0],
        normal: [1.0, 0.0, 0.0],
    };
    let v2 = VertexStruct {
        position: [0.0, 0.5, 0.0],
        normal: [1.0, 0.0, 0.0],
    };
    let v3 = VertexStruct {
        position: [0.5, -0.25, 0.0],
        normal: [1.0, 0.0, 0.0],
    };
    vec![v1, v2, v3]
    // CpuAccessibleBuffer::from_iter(gc.device.clone(), BufferUsage::vertex_buffer(), false, [v1, v2, v3].into_iter()).unwrap()
}

pub struct Mesh {
    vertices: Vec<VertexStruct>,
    // renderer: Option<VertexBufferRenderer<VertexStruct>>,
    vbo: Arc<CpuAccessibleBuffer<[VertexStruct]>>,
    gc: Arc<GraphicsContext>,
}

fn vbo_from_iter<I>(gc: Arc<GraphicsContext>, data: I) -> Arc<CpuAccessibleBuffer<[VertexStruct]>>
where
    I: IntoIterator<Item = VertexStruct>,
    I::IntoIter: ExactSizeIterator,
{
    CpuAccessibleBuffer::from_iter(gc.device.clone(), BufferUsage::vertex_buffer(), false, data)
        .unwrap()
}

impl Mesh {
    pub fn new(gc: Arc<GraphicsContext>) -> Mesh {
        let vertices = create_default_vertex_vec();
        let vbo = vbo_from_iter(gc.clone(), vertices.iter().cloned());
        Mesh {
            vertices,
            // renderer: None,
            vbo,
            gc,
        }
    }

    pub fn from_vertex_vec(gc: Arc<GraphicsContext>, vertices: Vec<VertexStruct>) -> Mesh {
        let vbo = vbo_from_iter(gc.clone(), vertices.iter().cloned());
        Mesh {
            vertices,
            // renderer: None,
            vbo,
            gc,
        }
    }

    pub fn vbo(&self) -> Arc<CpuAccessibleBuffer<[VertexStruct]>> {
        //let verts = std::mem::take(self.vertices);
        self.vbo.clone()
    }
}

impl VertexBufferProducer<CpuAccessibleBuffer<[VertexStruct]>> for Mesh {
    fn produce_vbo(&self, verts: &mut u32) -> Arc<CpuAccessibleBuffer<[VertexStruct]>> {
        *verts = self.vertices.len() as u32;
        self.vbo()
    }
}

impl VertexBufferProducer<CpuAccessibleBuffer<[VertexStruct]>> for &Mesh {
    fn produce_vbo(&self, verts: &mut u32) -> Arc<CpuAccessibleBuffer<[VertexStruct]>> {
        *verts = self.vertices.len() as u32;
        self.vbo()
    }
}
