use std::sync::Arc;
use std::time::Instant;

use crate::graphics_context::GraphicsContext;
use crate::logger::Logger;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use cgmath::Matrix4;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::SwapchainImage;
use vulkano::swapchain::{
    self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::FenceSignalFuture;
use vulkano::sync::{self, FlushError, GpuFuture};
use winit::window::Window;

fn reset_graphics(
    // gc: &GraphicsContext,
    device: Arc<Device>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: impl IntoIterator<Item = Arc<SwapchainImage<Window>>>,
    depth_format: Format,
    dimensions: [f32; 2],
    old_pipeline: &mut crate::pipeline::StandardVulcanPipeline,
) -> crate::uniforms::UniformHolder {
    let (p, u) = crate::prepare_graphics(
        device,
        swapchain,
        swapchain_images,
        depth_format,
        dimensions,
    );
    let old_pipe = std::mem::replace(old_pipeline, p);
    old_pipeline.renderers = old_pipe.renderers;
    // p.renderers = old_pipeline.renderers;
    // *old_pipeline = p;
    u
}

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
    logger: Box<dyn Logger>,
    // pub vertex_data: Vec<VertexStruct>,
    gc: GraphicsContext,
    window_resized: bool,
    window_minimized: bool,
    recreate_swapchain: bool,
    persp_matrix: Matrix4<f32>,
    depth_format: Format,
    fences_in_flight: Vec<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>,
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
        let dimensions = from.gc.surface.window().inner_size();
        let persp_matrix = cgmath::perspective(
            cgmath::Deg(60.0),
            dimensions.width as f32 / dimensions.height as f32,
            0.001,
            1000.0,
        );

        let mut fences_in_flight: Vec<Option<FenceSignalFuture<Box<dyn GpuFuture>>>> =
            Vec::with_capacity(from.gc.images.len());
        for _ in 0..from.gc.images.len() {
            fences_in_flight.push(None);
        }

        let start_time = std::time::Instant::now();

        App {
            logger: from.logger,
            gc: from.gc,
            window_resized: false,
            window_minimized: false,
            recreate_swapchain: false,
            persp_matrix,
            depth_format: from.depth_format,
            fences_in_flight,
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
                    let dimensions = self.gc.surface.window().inner_size();

                    let (new_swapchain, images) =
                        match self.gc.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..self.gc.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                return;
                            }
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    self.gc.swapchain = new_swapchain;
                    self.gc.images = images;
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
                        self.gc.uniform_holder = reset_graphics(
                            self.gc.device.clone(),
                            self.gc.swapchain.clone(),
                            self.gc.images.clone(),
                            self.depth_format,
                            [dimensions.width as f32, dimensions.height as f32],
                            &mut self.gc.pipeline,
                        );
                    }
                }
                let (image_id, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(self.gc.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            self.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Acquiring swapchain image resulted in panic: {}", e),
                    };
                // get the future associated with the image id
                // can unwrap safely because image index is within this vector by construction
                if let Some(previous_future) =
                    std::mem::take(self.fences_in_flight.get_mut(image_id).unwrap())
                {
                    // previous_future.cleanup_finished();
                    // TODO: what do we do on flush error here if it ever happens?
                    // for now we just panic
                    previous_future.wait(None).unwrap();
                }

                // Create command buffer
                let matrix = self.persp_matrix * matrix_from_time(self.start_time.elapsed());
                let uniform_set = self.gc.uniform_holder.set_view_matrix(matrix.into());
                let mut comm_builder = self.gc.create_command_builder();
                self.gc
                    .pipeline
                    .begin_render(&mut comm_builder, &uniform_set, image_id);
                // Loop over all renderables, when there are more than 1
                // mesh_vbo.issue_commands(&mut comm_builder);
                self.gc.pipeline.render(&mut comm_builder);
                self.gc.pipeline.end_render(&mut comm_builder);

                let comm_buffer = comm_builder.build().unwrap();
                if suboptimal {
                    self.recreate_swapchain = true;
                }
                // let branch_duration_prefence = branch_start_time.elapsed().as_micros();
                let exec = sync::now(self.gc.device.clone())
                    .join(acquire_future)
                    .then_execute(self.gc.queue.clone(), comm_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.gc.queue.clone(),
                        self.gc.swapchain.clone(),
                        image_id,
                    )
                    .boxed()
                    .then_signal_fence_and_flush();

                match exec {
                    Ok(future) => {
                        // fences_in_flight.push(Some(future));
                        self.fences_in_flight[image_id] = Some(future);
                        // future.wait(None).unwrap();
                    }
                    Err(FlushError::OutOfDate) => {
                        self.recreate_swapchain = true;
                    }
                    Err(e) => panic!("Panic on flush: {}", e),
                }
                // let time_elapsed = current_time.elapsed().as_micros();
                // println!("FPS: {}, branch time: {} µs, pre-fence init time: {} µs", 1000000 / time_elapsed, branch_start_time.elapsed().as_micros(), branch_duration_prefence);
                // current_time = std::time::Instant::now();
            }
            _ => {}
        }
    }
}
