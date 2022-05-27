use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Default, Clone, Copy, Zeroable, Pod)]
pub struct VertexStruct {
    pub position: [f32; 2],
}

vulkano::impl_vertex!(VertexStruct, position);