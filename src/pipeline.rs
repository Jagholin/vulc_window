use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::device::Device;
use vulkano::format::ClearValue;
use vulkano::format::Format;
use vulkano::image::ImageSubresourceRange;
use vulkano::image::{view::ImageView, view::ImageViewCreateInfo, AttachmentImage, SwapchainImage};
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::{FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
// use std::sync::RwLock;

use crate::graphics_context::GraphicsContext;
use crate::vertex_type::VertexStruct;
type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

/// render pass requires:
/// color and depth attachment formats
/// device
///
/// render pass is used in:
/// pipeline definition
/// framebuffers creation
///
///
/// graphics pipeline requires:
/// definition of vertex buffer(or any input buffers)
/// vertex shader object
/// fragment shader object
/// input assembly state (input geometry topology, default is single triangles)
///
/// graphics pipeline is used in:
/// issuing rendering commands
///

pub trait FramebufferGiver {
    fn give_framebuffer(&self, frameindex: usize) -> Arc<Framebuffer>;
}

pub trait FramebufferHolder {
    type Giver;

    fn framebuffer_created(&mut self, fb: Arc<Framebuffer>);
    fn framebuffer_giver(&self) -> Self::Giver;
    //
}

#[derive(Clone)]
pub struct FramebufferFrameSwapper {
    fbs: Rc<RefCell<Vec<Arc<Framebuffer>>>>,
}

impl FramebufferFrameSwapper {
    pub fn new() -> Self {
        FramebufferFrameSwapper {
            fbs: Default::default(),
        }
    }
}

impl FramebufferHolder for FramebufferFrameSwapper {
    type Giver = Self;
    fn framebuffer_created(&mut self, fb: Arc<Framebuffer>) {
        let mut inner = self.fbs.borrow_mut();
        inner.push(fb);
        // self.fbs.push(fb);
    }
    fn framebuffer_giver(&self) -> Self::Giver {
        Self {
            fbs: Rc::clone(&self.fbs),
        }
    }
}

impl FramebufferGiver for FramebufferFrameSwapper {
    fn give_framebuffer(&self, framenumber: usize) -> Arc<Framebuffer> {
        let inner = self.fbs.borrow();
        // let fb_count = inner.len();
        let result = inner.get(framenumber).unwrap();
        result.clone()
    }
}

fn shader_pipeline_graphics(
    // gc: &GraphicsContext,
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    format: Format,
    depth_format: Format,
    dimensions: [f32; 2],
) -> (Arc<RenderPass>, Arc<GraphicsPipeline>) {
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions,
        depth_range: 0.0..1.0,
    };

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: format,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: Store,
                format: depth_format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexStruct>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .build(device.clone())
        .unwrap();
    (render_pass, pipeline)
}

fn get_framebuffers(
    // gc: &GraphicsContext,
    allocator: &impl MemoryAllocator,
    images: impl IntoIterator<Item = Arc<SwapchainImage>>,
    render_pass: Arc<RenderPass>,
    depth_format: Format,
    dimensions: [u32; 2],
) -> Vec<Arc<Framebuffer>> {
    images
        .into_iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let depth_buffer =
                AttachmentImage::transient(allocator, dimensions, depth_format).unwrap();
            let depth_view = ImageView::new(
                depth_buffer.clone(),
                ImageViewCreateInfo {
                    format: Some(depth_format),
                    subresource_range: ImageSubresourceRange {
                        aspects: depth_format.aspects(),
                        mip_levels: 0..1,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

#[derive(Clone)]
pub struct VulkanPipeline<T> {
    pub pipeline: Arc<GraphicsPipeline>,
    #[allow(unused)]
    render_pass: Arc<RenderPass>,
    // framebuffers: Vec<Arc<Framebuffer>>,
    framebuffer_giver: T,
    pre_renderers: Vec<Arc<dyn PreRenderingSteps>>,
    pub renderers: Vec<Arc<dyn Renderer>>,
}

pub struct VulkanPipelineBuilder<T> {
    vs: Option<Arc<ShaderModule>>,
    fs: Option<Arc<ShaderModule>>,
    format: Option<Format>,
    images: Option<T>,
    depth_format: Option<Format>,
    dimensions: Option<[f32; 2]>,
}

pub trait Renderer {
    fn render(&self, gc: &GraphicsContext, cb: &mut StandardCommandBuilder);
}

pub trait PreRenderingSteps {
    fn pre_render(&self, pipe: Arc<GraphicsPipeline>, cb: &mut StandardCommandBuilder);
}

impl<T> VulkanPipelineBuilder<T>
where
    T: IntoIterator<Item = Arc<SwapchainImage>>,
{
    pub fn vs(self, sm: Arc<ShaderModule>) -> Self {
        Self {
            vs: Some(sm),
            ..self
        }
    }
    pub fn fs(self, sm: Arc<ShaderModule>) -> Self {
        Self {
            fs: Some(sm),
            ..self
        }
    }
    pub fn format(self, f: Format) -> Self {
        Self {
            format: Some(f),
            ..self
        }
    }
    pub fn images(self, img: T) -> Self {
        Self {
            images: Some(img),
            ..self
        }
    }
    pub fn depth_format(self, f: Format) -> Self {
        Self {
            depth_format: Some(f),
            ..self
        }
    }
    pub fn dimensions(self, dim: [f32; 2]) -> Self {
        Self {
            dimensions: Some(dim),
            ..self
        }
    }

    pub fn build<F: FramebufferHolder>(
        self,
        // gc: &GraphicsContext,
        allocator: &impl MemoryAllocator,
        device: Arc<Device>,
        fb_holder: &mut F,
    ) -> VulkanPipeline<F::Giver> {
        let vs = self.vs.unwrap();
        let fs = self.fs.unwrap();
        let format = self.format.unwrap();
        let images = self.images.unwrap();
        let depth_format = self.depth_format.unwrap();
        let dims = self.dimensions.unwrap();

        let (render_pass, pipeline) =
            shader_pipeline_graphics(device.clone(), vs, fs, format, depth_format, dims);
        let fbs = get_framebuffers(
            allocator,
            images,
            render_pass.clone(),
            depth_format,
            [dims[0] as u32, dims[1] as u32],
        );
        // let fbh = fb_holder.get_mut().unwrap();
        for f in fbs {
            fb_holder.framebuffer_created(f);
        }
        VulkanPipeline {
            pipeline,
            render_pass,
            framebuffer_giver: fb_holder.framebuffer_giver(),
            pre_renderers: vec![],
            renderers: vec![],
        }
    }
}

impl<T> VulkanPipeline<T>
where
    T: FramebufferGiver,
{
    pub fn new<Y>() -> VulkanPipelineBuilder<Y> {
        VulkanPipelineBuilder {
            vs: None,
            fs: None,
            format: None,
            images: None,
            depth_format: None,
            dimensions: None,
        }
    }

    pub fn begin_render(
        &self,
        command_builder: &mut StandardCommandBuilder,
        // unis: &impl UniformStruct,
        // descriptors: impl IntoIterator<Item = WriteDescriptorSet>,
        frameindex: usize,
    ) {
        let fb = self.framebuffer_giver.give_framebuffer(frameindex);
        let mut pass_begin = RenderPassBeginInfo::framebuffer(fb);
        pass_begin.clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some(ClearValue::Depth(1000.0)),
        ];
        command_builder
            .begin_render_pass(pass_begin, SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone());
        // applying uniforms and other descriptor sets is the task of the GraphicsContext
        // unis.apply_uniforms(command_builder);
        for prer in self.pre_renderers.iter() {
            prer.pre_render(self.pipeline.clone(), command_builder);
        }
    }

    pub fn end_render(&self, command_builder: &mut StandardCommandBuilder) {
        command_builder.end_render_pass().unwrap();
    }

    pub fn pipeline_descriptor(&self) -> &[Arc<DescriptorSetLayout>] {
        let l = self.pipeline.layout();
        l.set_layouts()
    }

    pub fn pipeline_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline.layout().clone()
    }

    pub fn add_prerender(&mut self, pr: Arc<dyn PreRenderingSteps>) {
        self.pre_renderers.push(pr);
    }

    pub fn render(&self, gc: &GraphicsContext, command_builder: &mut StandardCommandBuilder) {
        for renderer in &self.renderers {
            renderer.render(gc, command_builder);
        }
    }

    pub fn descriptor_layout(&self) -> Arc<DescriptorSetLayout> {
        let pipe_layout = self.pipeline.layout().set_layouts();
        if pipe_layout.len() != 1 {
            panic!("unexpected length of pipeline layouts: {}. Did something change?", pipe_layout.len());
        }
        let pipe_layout = pipe_layout.get(0).unwrap();
        pipe_layout.clone()
    }
}

pub type StandardVulcanPipeline = VulkanPipeline<FramebufferFrameSwapper>;
