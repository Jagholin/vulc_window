use std::sync::Arc;
use vulkano::buffer::{BufferContents, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

pub trait VertexBufferProducer {
    type Result;
    fn produce_vbo(&self, verts: &mut u32) -> Option<Arc<Self::Result>>;
}
pub trait IssueCommands {
    fn issue_commands(&mut self, command_builder: &mut StandardCommandBuilder);
}

pub struct VertexBufferRenderer<T>
where
    T: VertexBufferProducer,
{
    vbo: VertexBufferStreaming<T>,
    // verts_count: u32,
}

impl<T, U> VertexBufferRenderer<T>
where
    T: VertexBufferProducer<Result = ImmutableBuffer<[U]>>,
    [U]: BufferContents
{
    pub fn new(vbo_producer: T) -> Self {
        // let mut verts_count: u32 = 0;
        // let vbo = vbo_producer.produce_vbo(&mut verts_count);
        // let vbo = vbo_producer.provide();
        // vbo.map(|v| Self {vbo: VertexBufferStreaming::new(v.0)})
        Self {vbo: VertexBufferStreaming::new(vbo_producer)}
    }
    /* pub fn from_vbo(vbo: Arc<ImmutableBuffer<[U]>>, verts_count: u32) -> Self {
        Self {vbo, verts_count}
    } */
}

impl<T, U> IssueCommands for VertexBufferRenderer<T>
where
    T: VertexBufferProducer<Result = ImmutableBuffer<[U]>>,
    [U]: BufferContents
{
    fn issue_commands(&mut self, command_builder: &mut StandardCommandBuilder) {
        self.vbo.do_ifready(|(v, vcount)| {
            command_builder
                .bind_vertex_buffers(0, v.clone())
                .draw(*vcount, 1, 0, 0)
                .unwrap();
        })
    }
}

pub trait StreamingProvider {
    type Result;
    fn provide(&self) -> Option<Self::Result>;
}

enum VertexBufferStreaming<T> 
where T: StreamingProvider {
    Loading(T),
    Ready(T::Result)
}

impl<T> VertexBufferStreaming<T>
where T: StreamingProvider {
    fn new(provider: T) -> Self {
        Self::Loading(provider)
    }

    fn update(&mut self) {
        if let Self::Loading(ref d) = self {
            if let Some(res) = d.provide() {
                *self = Self::Ready(res);
            }
        }
    }

    fn do_ifready_mut<F>(&mut self, f: F) where F: FnOnce(&mut T::Result) {
        self.update();
        if let Self::Ready(ref mut d) = self {
            f(d);
        }
    }

    fn do_ifready<F>(&mut self, f: F) where F: FnOnce(&T::Result) {
        self.update();
        if let Self::Ready(ref d) = self {
            f(d);
        }
    }
}

impl<T> StreamingProvider for T
where
    T: VertexBufferProducer 
{
    type Result = (Arc<T::Result>, u32);
    fn provide(&self) -> Option<Self::Result> {
        let mut verts = 0;
        let result = self.produce_vbo(&mut verts);
        result.map(|r| (r, verts))
    }
}
