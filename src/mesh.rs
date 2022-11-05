use crate::graphics_context::GraphicsContext;
use crate::pipeline::Renderer;
use crate::renderer::VertexBufferProducer;
use crate::vertex_type::VertexStruct;
use crate::IssueCommands;
use crate::StandardCommandBuilder;
use crate::VertexBufferStreaming;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::sync::GpuFuture;

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
    pub vertices: Vec<VertexStruct>,
    // renderer: Option<VertexBufferRenderer<VertexStruct>>,
    // vbo: Arc<ImmutableBuffer<[VertexStruct]>>,
    // buffer_ready: Arc<Mutex<bool>>,
    gc: GraphicsContext,
    vbo: RefCell<Option<MeshVertexBufferProducer>>,
}

#[derive(Clone)]
pub struct MeshVertexBufferProducer {
    vbo: Arc<ImmutableBuffer<[VertexStruct]>>,
    buffer_ready: Arc<Mutex<bool>>,
    vert_len: usize,
    // mesh: &'a Mesh<'a>,
}

fn vbo_from_iter<I>(
    gc: GraphicsContext,
    data: I,
) -> (Arc<ImmutableBuffer<[VertexStruct]>>, Arc<Mutex<bool>>)
where
    I: IntoIterator<Item = VertexStruct>,
    I::IntoIter: ExactSizeIterator,
{
    // CpuAccessibleBuffer::from_iter(gc.device.clone(), BufferUsage::vertex_buffer(), false, data)
    //    .unwrap()
    let (buff, fut) =
        ImmutableBuffer::from_iter(data, BufferUsage::vertex_buffer(), gc.queue.clone()).unwrap();
    let signal = Arc::new(Mutex::new(false));
    let signal_moved = signal.clone();
    std::thread::spawn(move || {
        fut.then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let mut sig = signal_moved.lock().expect("how can this fail");
        *sig = true;
    });
    (buff, signal)
}

impl Mesh {
    pub fn new(gc: GraphicsContext) -> Mesh {
        let vertices = create_default_vertex_vec();
        // let (vbo, buffer_ready) = vbo_from_iter(gc.clone(), vertices.iter().cloned());
        Mesh {
            vertices,
            // renderer: None,
            // vbo,
            // buffer_ready,
            gc,
            vbo: RefCell::new(None),
        }
    }

    pub fn from_vertex_vec(gc: GraphicsContext, vertices: Vec<VertexStruct>) -> Mesh {
        // let (vbo, buffer_ready) = vbo_from_iter(gc.clone(), vertices.iter().cloned());
        Mesh {
            vertices,
            // renderer: None,
            // vbo,
            // buffer_ready,
            gc,
            vbo: RefCell::new(None),
        }
    }

    pub fn vertex_buffer(&self) -> MeshVertexBufferProducer {
        let cache = self.vbo.borrow();
        if let Some(res) = &*cache {
            return res.clone();
        }
        drop(cache);

        let (vbo, buffer_ready) = vbo_from_iter(self.gc.clone(), self.vertices.clone());
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
    type Result = ImmutableBuffer<[VertexStruct]>;
    fn produce_vbo(&self, verts: &mut u32) -> Option<Arc<ImmutableBuffer<[VertexStruct]>>> {
        *verts = self.vert_len as u32;
        let is_ready = self.buffer_ready.lock().unwrap();
        if !*is_ready {
            return None;
        }
        Some(self.vbo.clone())
    }
}

impl Renderer for Mesh {
    fn render(&self, cb: &mut StandardCommandBuilder) {
        let mut vbo = VertexBufferStreaming::new(self.vertex_buffer());
        vbo.issue_commands(cb);
    }
}

/* impl VertexBufferProducer for &Mesh {
    type Result = ImmutableBuffer<[VertexStruct]>;
    fn produce_vbo(&self, verts: &mut u32) -> Option<Arc<ImmutableBuffer<[VertexStruct]>>> {
        *verts = self.vertices.len() as u32;
        let is_ready = self.buffer_ready.lock().unwrap();
        if !*is_ready {
            return None;
        }
        Some(self.vbo())
    }
} */
