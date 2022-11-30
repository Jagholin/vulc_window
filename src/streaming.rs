// Provides streaming support for assets like images and mesh data

use crate::{graphics_context::GraphicsContext};

pub trait Asset<D> {
    // This function should initiate async operation to get 
    // T, probably by sending it to a different thread
    fn init_from_data(&self, data: D, gc: &GraphicsContext);
    fn asset_ready(&self) -> bool;
}
