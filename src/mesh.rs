use crate::graphics_context::GraphicsContext;
use crate::pipeline::Renderer;
use crate::renderer::IssueCommands;
use crate::renderer::VertexBufferProducer;
use crate::renderer::VertexBufferStreaming;
use crate::vertex_type::VertexStruct;
use crate::StandardCommandBuilder;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer /* ImmutableBuffer */};

pub struct Mesh {
    pub vertices: Vec<VertexStruct>,
    vbo: RefCell<Option<MeshVertexBufferProducer>>,
}

#[derive(Clone)]
pub struct MeshVertexBufferProducer {
    vbo: Arc<DeviceLocalBuffer<[VertexStruct]>>,
    buffer_ready: Arc<RwLock<bool>>,
    vert_len: usize,
}

fn vbo_from_iter<I>(
    gc: &GraphicsContext,
    data: I,
) -> (Arc<DeviceLocalBuffer<[VertexStruct]>>, Arc<RwLock<bool>>)
where
    I: IntoIterator<Item = VertexStruct>,
    I::IntoIter: ExactSizeIterator,
{
    let mut cbb = gc.create_command_builder();
    let buff = DeviceLocalBuffer::from_iter(
        &gc.standard_allocator,
        data,
        BufferUsage {
            vertex_buffer: true,
            transfer_dst: true,
            ..Default::default()
        },
        &mut cbb,
    )
    .unwrap();
    let cb = cbb.build().unwrap();
    let finished_flag = Arc::new(RwLock::new(false));
    gc.run_secondary_action(cb, finished_flag.clone());
    (buff, finished_flag)
}

impl Mesh {

    pub fn from_vertex_vec(vertices: Vec<VertexStruct>) -> Self {
        Mesh {
            vertices,
            vbo: RefCell::new(None),
        }
    }

    pub fn vertex_buffer(&self, gc: &GraphicsContext) -> MeshVertexBufferProducer {
        let cache = self.vbo.borrow();
        if let Some(res) = &*cache {
            return res.clone();
        }
        drop(cache);

        let (vbo, buffer_ready) = vbo_from_iter(gc, self.vertices.clone());
        let res = MeshVertexBufferProducer {
            vbo,
            buffer_ready,
            vert_len: self.vertices.len(),
        };
        self.vbo.replace(Some(res.clone()));
        res
    }
}

impl VertexBufferProducer for MeshVertexBufferProducer {
    type Result = DeviceLocalBuffer<[VertexStruct]>;
    fn produce_vbo(&self, verts: &mut u32) -> Option<Arc<DeviceLocalBuffer<[VertexStruct]>>> {
        *verts = self.vert_len as u32;
        let is_ready = self.buffer_ready.read().unwrap();
        if !*is_ready {
            return None;
        }
        Some(self.vbo.clone())
    }
}

impl Renderer for Mesh {
    fn render(&self, gc: &GraphicsContext, cb: &mut StandardCommandBuilder) {
        let mut vbo = VertexBufferStreaming::new(self.vertex_buffer(gc));
        vbo.issue_commands(cb);
    }
}
