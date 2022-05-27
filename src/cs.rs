#![allow(clippy::needless_question_mark)]
vulkano_shaders::shader! {
    shaders: {
        mult_const: {
            ty: "compute",
            src: "#version 450

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            
            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;
            
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }"
        },
        test_compute: {
            ty: "compute",
            path: "shaders/test.compute.txt"
        },
        vert_shader: {
            ty: "vertex",
            path: "shaders/vertex.txt"
        },
        frag_shader: {
            ty: "fragment",
            path: "shaders/frag.txt"
        }
    }
}
