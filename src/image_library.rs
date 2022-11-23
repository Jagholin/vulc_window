/* 
Holds loaded images and loads them as textures
*/

use image::{RgbImage, ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::{image::ImmutableImage, device::Device, format::Format};

use crate::graphics_context::GraphicsContext;

pub type StandardImageBuffer = ImageBuffer<Rgba<u8>, Vec<u8>>;

pub struct StoredImage {
    pub image: Arc<ImmutableImage>,
}

impl StoredImage {
    pub fn new_from(img: &StandardImageBuffer, gc: &GraphicsContext) -> Self {
        let img_samples = img.as_flat_samples();
        println!("Image loaded with sample layout {:?}, color hint {:?}", img_samples.layout, img_samples.color_hint);
        let img_samples = img_samples.samples;
        let mut cbb = gc.create_command_builder();
        let res = ImmutableImage::from_iter(
            gc.standard_allocator.as_ref(), 
            img_samples.into_iter().copied(), 
            vulkano::image::ImageDimensions::Dim2d { width: img.width(), height: img.height(), array_layers: 1 }, 
            vulkano::image::MipmapsCount::One, 
            Format::R8G8B8A8_UINT, 
            &mut cbb);
        
        let res = res.expect("Cant load image");
        // todo!()
        Self { image: res }
    }
}


