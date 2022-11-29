mod app;
mod cs;
mod fbx;
mod graphics_context;
mod input_helpers;
mod logger;
mod mesh;
mod pipeline;
mod real_main;
mod renderer;
mod uniforms;
mod vertex_type;
mod image_library;
mod streaming;
mod mixins;

//reexport some stuff from real_main
// pub use real_main::prepare_graphics;
pub use real_main::StandardCommandBuilder;

fn main() {
    real_main::real_main();
}
