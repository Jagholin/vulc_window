/* 
Holds loaded images and loads them as textures
*/

use image::{ImageBuffer, Rgba};
use std::{sync::{Arc, RwLock}, collections::HashMap};
use vulkano::{image::{ImmutableImage, view::ImageView}, format::Format, memory::allocator::MemoryAllocator};

use crate::{graphics_context::{GCDevAlloc, GCDevJobs}, StandardCommandBuilder};

pub type StandardImageBuffer = ImageBuffer<Rgba<u8>, Vec<u8>>;
type StandardImageView = Arc<ImageView<ImmutableImage>>;

pub struct StoredImage {
    image: Arc<ImmutableImage>,
    // loaded: Arc<RwLock<bool>>,
    image_view: Arc<RwLock<Option<StandardImageView>>>,
}

impl StoredImage {
    fn new_from(img: &StandardImageBuffer, /*gc: &GraphicsContext*/gc_dev: GCDevAlloc, gc_jobs: GCDevJobs) -> Self {
        let img_samples = img.as_flat_samples();
        println!("Image loaded with sample layout {:?}, color hint {:?}", img_samples.layout, img_samples.color_hint);
        let img_samples = img_samples.samples;
        let mut cbb = gc_dev.create_command_builder();
        let res = ImmutableImage::from_iter(
            gc_dev.alloc().standard_allocator.as_ref(), 
            img_samples.into_iter().copied(), 
            vulkano::image::ImageDimensions::Dim2d { width: img.width(), height: img.height(), array_layers: 1 }, 
            vulkano::image::MipmapsCount::One, 
            Format::R8G8B8A8_UINT, 
            &mut cbb);
        
        let res = res.expect("Cant load image");
        let im_view = Arc::new(RwLock::new(None));
        let im_view_threaded = im_view.clone();
        let img_threaded = res.clone();
        // let loaded = Arc::new(RwLock::new(false));
        // let loaded_sender = loaded.clone();
        gc_jobs.run_secondary_action(cbb.build().unwrap(), Box::new(move || {
            // let mut sender = loaded_sender.write().unwrap();
            // *sender = true;
            let mut im_view = im_view_threaded.write().unwrap();
            im_view.replace(ImageView::new_default(img_threaded).unwrap());
        }));
        
        Self { image: res, image_view: im_view }
    }

    fn new_fallback<'func>(allocator: &impl MemoryAllocator, mut cbb: StandardCommandBuilder, cbb_runner: Box<dyn FnOnce(StandardCommandBuilder) + 'func>) -> Self {
        // create an image filled with blue color
        let buffer = (0 .. 256*256*4).map(|i| if i%4 == 2  {255u8} else {0u8}).collect::<Vec<_>>();
        let res = ImmutableImage::from_iter(allocator,
            buffer,
            vulkano::image::ImageDimensions::Dim2d { width: 256, height: 256, array_layers: 1 },
            vulkano::image::MipmapsCount::One,
            Format::R8G8B8A8_UINT,
            &mut cbb).expect("Cant create a fallback texture");
        
        // gc(cbb.build().unwrap());
        cbb_runner(cbb);
        let im_view = ImageView::new_default(res.clone()).unwrap();
        let im_view = Arc::new(RwLock::new(Some(im_view)));
        Self { image: res, image_view: im_view }
    }

    pub fn image_view(&self) -> Option<StandardImageView> {
        let read = self.image_view.read().unwrap();
        read.as_ref().map(|val| val.clone())
    }
}

pub struct ImageLibrary {
    content: HashMap<String, StoredImage>,
    pub fallback_texture: StoredImage,
}

impl ImageLibrary {
    pub fn new<'func>(allocator: &impl MemoryAllocator, cbb: StandardCommandBuilder, cbb_runner: Box<dyn FnOnce(StandardCommandBuilder) + 'func>) -> Self {

        ImageLibrary { content: HashMap::new(), fallback_texture: StoredImage::new_fallback(allocator, cbb, cbb_runner) }
    }

    pub fn insert_image(&mut self, key: impl ToString, img: &'_ StandardImageBuffer, /* gc: &'_ GraphicsContext */gc_dev: GCDevAlloc, gc_jobs: GCDevJobs) -> &StoredImage {
        let key = key.to_string();
        let img = StoredImage::new_from(img, gc_dev, gc_jobs);
        self.content.insert(key.clone(), img);

        self.content.get(&key).unwrap()
    }

    pub fn get_image(&self, key: impl ToString) -> Option<&StoredImage> {
        self.content.get(& key.to_string())
    }
}
