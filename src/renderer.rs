use std::sync::Arc;
use vulkano::buffer::{BufferContents, DeviceLocalBuffer /* ImmutableBuffer */};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::graphics_context::GraphicsContext;

type StandardCommandBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

pub trait VertexBufferProducer {
    type Result;
    fn produce_vbo(&self, verts: &mut u32, gc: &GraphicsContext) -> Option<Arc<Self::Result>>;
}
pub trait IssueCommands {
    fn issue_commands(&mut self, gc: &GraphicsContext, command_builder: &mut StandardCommandBuilder);
}

pub trait StreamingProvider {
    type Result;
    fn provide(&self, gc: &GraphicsContext) -> Option<Self::Result>;
}

pub enum VertexBufferStreaming<T>
where
    T: StreamingProvider,
{
    Loading(T),
    Ready(T::Result),
}

impl<T> VertexBufferStreaming<T>
where
    T: StreamingProvider,
{
    pub fn new(provider: T) -> Self {
        Self::Loading(provider)
    }

    fn update(&mut self, gc: &GraphicsContext) {
        if let Self::Loading(ref d) = self {
            if let Some(res) = d.provide(gc) {
                *self = Self::Ready(res);
            }
        }
    }

    #[allow(unused)]
    fn do_ifready_mut<F>(&mut self, gc: &GraphicsContext, f: F)
    where
        F: FnOnce(&mut T::Result),
    {
        self.update(gc);
        if let Self::Ready(ref mut d) = self {
            f(d);
        }
    }

    fn do_ifready<F>(&mut self, gc: &GraphicsContext, f: F)
    where
        F: FnOnce(&T::Result),
    {
        self.update(gc);
        if let Self::Ready(ref d) = self {
            f(d);
        }
    }
}

impl<T> StreamingProvider for T
where
    T: VertexBufferProducer,
{
    type Result = (Arc<T::Result>, u32);
    fn provide(&self, gc: &GraphicsContext) -> Option<Self::Result> {
        let mut verts = 0;
        let result = self.produce_vbo(&mut verts, gc);
        result.map(|r| (r, verts))
    }
}

impl<T, U> IssueCommands for VertexBufferStreaming<T>
where
    T: VertexBufferProducer<Result = DeviceLocalBuffer<[U]>>,
    [U]: BufferContents,
{
    fn issue_commands(&mut self, gc: &GraphicsContext, command_builder: &mut StandardCommandBuilder) {
        self.do_ifready(gc, |(v, vcount)| {
            command_builder
                .bind_vertex_buffers(0, v.clone())
                .draw(*vcount, 1, 0, 0)
                .unwrap();
        })
    }
}
