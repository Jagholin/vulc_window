use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Default, Clone, Copy, Zeroable, Pod)]
pub struct VertexStruct {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

vulkano::impl_vertex!(VertexStruct, position, normal, uv);
