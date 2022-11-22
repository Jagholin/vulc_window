/* 
Holds loaded images and loads them as textures
*/

use image::RgbImage;
use std::sync::Arc;
use vulkano::{image::ImmutableImage, device::Device};

pub struct StoredImage {
    pub image: ImmutableImage,
}

impl StoredImage {
    fn new_from(img: &RgbImage, dev: Arc<Device>) -> Self {
        todo!()
    }
}
