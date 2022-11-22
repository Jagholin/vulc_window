use crate::graphics_context::GraphicsContext;
use crate::pipeline::Renderer;
use crate::renderer::IssueCommands;
use crate::renderer::VertexBufferProducer;
use crate::renderer::VertexBufferStreaming;
use crate::vertex_type::VertexStruct;
use crate::streaming::Asset;
use crate::StandardCommandBuilder;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer /* ImmutableBuffer */};

type VertexBufferType = Arc<DeviceLocalBuffer<[VertexStruct]>>;
struct MeshStreamingData {
    buff: VertexBufferType,
    finished_flag: Arc<RwLock<bool>>,
}

pub struct Mesh {
    pub vertices: Vec<VertexStruct>,
    // vbo: RefCell<Option<VertexBufferType>>,

    // to implement Asset<...> we need this additional data
    asset_data: RefCell<Option<MeshStreamingData>>,
}

/* #[derive(Clone)]
pub struct MeshVertexBufferProducer {
    vbo: Arc<DeviceLocalBuffer<[VertexStruct]>>,
    buffer_ready: Arc<RwLock<bool>>,
    vert_len: usize,
} */

impl<I> Asset<I> for Mesh
where I: IntoIterator<Item=VertexStruct>,
      I::IntoIter: ExactSizeIterator {

    fn init_from_data(&self, data: I, gc: &GraphicsContext) {
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
        let finished_flag_signal = finished_flag.clone();
        gc.run_secondary_action(cb, Box::new(move || {
            let mut sig = finished_flag_signal.write().expect("Error while unwrapping RwLock");
            *sig = true;
        }));
        // (buff, finished_flag)
        let mut asset_data = self.asset_data.borrow_mut();
        if asset_data.is_some() {
            panic!("asset data is not empty ?!!");
        }
        *asset_data = Some(MeshStreamingData { buff, finished_flag });
    }

    fn asset_ready(&self) -> bool {
        let asset_data = self.asset_data.borrow();
        match &*asset_data {
            Some(data) => {
                let result = data.finished_flag.read();
                *result.unwrap()
            }
            _ => false
        }
    }
}

impl Mesh {

    pub fn from_vertex_vec(vertices: Vec<VertexStruct>) -> Self {
        Mesh {
            vertices,
            // vbo: RefCell::new(None),
            asset_data: RefCell::new(None)
        }
    }

    /* pub fn vertex_buffer(&self, gc: &GraphicsContext) -> MeshVertexBufferProducer {
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
    } */
}

impl VertexBufferProducer for Mesh {
    type Result = DeviceLocalBuffer<[VertexStruct]>;
    fn produce_vbo(&self, verts: &mut u32, gc: &GraphicsContext) -> Option<VertexBufferType> {
        *verts = self.vertices.len() as u32;
        let asset_data = self.asset_data.borrow();
        let mut start_transfer = false;
        // start creating an asset if we dont have one
        let ret_data = match &*asset_data {
            Some(ad) => {
                let res = *ad.finished_flag.read().unwrap();
                if res {
                    Some(ad.buff.clone())
                } else { None }
            },
            None => {
                // start streaming now
                start_transfer = true;
                None
            }
        };
        drop(asset_data);
        if start_transfer {
            self.init_from_data(self.vertices.iter().copied(), gc);
        }
        ret_data
    }
}

impl VertexBufferProducer for &Mesh {
    type Result = DeviceLocalBuffer<[VertexStruct]>;
    fn produce_vbo(&self, verts: &mut u32, gc: &GraphicsContext) -> Option<Arc<Self::Result>> {
        (**self).produce_vbo(verts, gc)
    }
}

impl Renderer for Mesh {
    fn render(&self, gc: &GraphicsContext, cb: &mut StandardCommandBuilder) {
        let mut vbo = VertexBufferStreaming::new(self);
        vbo.issue_commands(gc, cb);
    }
}
