// Provides streaming support for assets like images and mesh data

use std::{cell::RefCell, sync::Arc};

use crate::{graphics_context::GraphicsContext, mesh::Mesh, vertex_type::VertexStruct};

pub trait Asset<D> {
    // This function should initiate async operation to get 
    // T, probably by sending it to a different thread
    fn init_from_data(&self, data: D, gc: &GraphicsContext);
    fn asset_ready(&self) -> bool;
}

enum Eventually_State<T> {
    JustCreated,
    PendingData(Arc<RefCell<Option<T>>>),
    Ready(T),
}

// Basically, a Future.
// Dont want to deal with rust's future/async systems
// due to how immature they are
struct Eventually<T> {
    data: RefCell<Eventually_State<T>>,
}

impl<T> Eventually<T> {
    pub fn new() -> Self {
        Self {
            data: Eventually_State::JustCreated,
        }
    }
    
    pub fn try_retrieve(&self) -> Option<T> {
        todo!()
    }
}

impl Asset<I> for Mesh
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
        gc.run_secondary_action(cb, finished_flag.clone());
        (buff, finished_flag)
    }
}
