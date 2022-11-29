use std::time::Instant;

use crate::graphics_context::{GraphicsContext, GraphicsContextError};
use crate::logger::Logger;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use cgmath::Matrix4;
use vulkano::format::Format;
use winit::window::Window;

fn matrix_from_time(passed: std::time::Duration) -> cgmath::Matrix4<f32> {
    const DISTANCE: f32 = 1.5;

    let time_factor = passed.as_millis() as f32 / 1000.0;
    let x_coord = time_factor.sin() * DISTANCE;
    let y_coord = time_factor.cos() * DISTANCE;

    cgmath::Matrix4::look_at_rh(
        [x_coord, y_coord, DISTANCE].into(),
        [0.0, 0.0, 0.7].into(),
        [0.0, 0.0, 1.0].into(),
    )
}

pub struct App {
    #[allow(unused)]
    logger: Box<dyn Logger>,
    // pub vertex_data: Vec<VertexStruct>,
    gc: GraphicsContext,
    window_resized: bool,
    window_minimized: bool,
    recreate_swapchain: bool,
    persp_matrix: Matrix4<f32>,
    // depth_format: Format,
    start_time: Instant,
}

pub struct AppCreateStruct {
    pub logger: Box<dyn Logger>,
    pub gc: GraphicsContext,
    pub depth_format: Format,
}

impl App {
    pub fn new(from: AppCreateStruct) -> Self {
        // self.window_minimized = false;
        // self.window_resized = false;
        // self.recreate_swapchain = false;
        let window = from
            .gc.dev()
            .surface()
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap();
        let dimensions = window.inner_size();
        let persp_matrix = cgmath::perspective(
            cgmath::Deg(60.0),
            dimensions.width as f32 / dimensions.height as f32,
            0.001,
            1000.0,
        );

        let start_time = std::time::Instant::now();

        App {
            logger: from.logger,
            gc: from.gc,
            window_resized: false,
            window_minimized: false,
            recreate_swapchain: false,
            persp_matrix,
            start_time,
        }
    }

    pub fn event_loop(&mut self, event: Event<()>, cf: &mut ControlFlow) {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *cf = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                self.window_resized = true;
                self.window_minimized = size.height == 0 || size.width == 0;
            }
            Event::MainEventsCleared => {
                // let branch_start_time = std::time::Instant::now();
                if self.window_minimized {
                    return;
                }
                if self.recreate_swapchain || self.window_resized {
                    self.recreate_swapchain = false;
                    let mut_dev = self.gc.mut_dev();
                    mut_dev.recreate_swapchain();
                    let dimensions = mut_dev.window_dimensions();
                    self.persp_matrix = cgmath::perspective(
                        cgmath::Deg(60.0),
                        dimensions.width as f32 / dimensions.height as f32,
                        0.001,
                        1000.0,
                    );

                    if self.window_resized {
                        self.window_resized = false;
                        // command_builders = prepare_graphics(device.clone(), queue.clone(), &mesh, swapchain.clone(), images,
                        //     depth_format, [dimensions.width as f32, dimensions.height as f32]);
                        self.gc.reset_pipeline();
                    }
                }

                // ================================================================================
                // todo!(); // update uniforms in graphics context
                // let matrix = self.persp_matrix * matrix_from_time(self.start_time.elapsed());
                // self.gc.uniform_holder.set_view_matrix(matrix.into());
                match self.gc.render_loop() {
                    Err(GraphicsContextError::SuboptimalImage | GraphicsContextError::SwapchainError(_)) => {
                        self.recreate_swapchain = true;
                    },
                    Err(e) => {
                        panic!("Unexpected error in render loop: {}", e);
                    },
                    _ => {

                    }
                }
                // =============================================================
                // let time_elapsed = current_time.elapsed().as_micros();
                // println!("FPS: {}, branch time: {} µs, pre-fence init time: {} µs", 1000000 / time_elapsed, branch_start_time.elapsed().as_micros(), branch_duration_prefence);
                // current_time = std::time::Instant::now();
            }
            _ => {}
        }
    }
}
