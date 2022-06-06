
fn clear_image<T>(
    gc: &GraphicsContext,
    image: Arc<StorageImage>,
    command_builder: &mut AutoCommandBufferBuilder<T>,
) {
    // let mut builder = gc.create_command_builder();
    command_builder
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
        .unwrap();
}

fn create_integer_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[i32]>> {
    let data_iter = 0..65536;
    CpuAccessibleBuffer::from_iter(device, BufferUsage::all(), false, data_iter)
        .expect("Cant create test buffer")
}

fn create_image(gc: &GraphicsContext) -> Arc<StorageImage> {
    let result = StorageImage::new(
        gc.device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(gc.queue.family()),
    )
    .unwrap();
    println!("image created!");
    result
}

fn export_image<T>(
    gc: &GraphicsContext,
    image: Arc<StorageImage>,
    command_builder: &mut AutoCommandBufferBuilder<T>,
) -> impl FnOnce(&GraphicsContext) {
    let buf = CpuAccessibleBuffer::from_iter(
        gc.device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 255u8),
    )
    .expect("cant create buffer to save the image");

    command_builder
        .copy_image_to_buffer(image, buf.clone())
        .unwrap();

    move |gc| {
        let content = buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &content[..]).unwrap();
        println!("image exported!");
        image.save("image.png").unwrap();
    }
}

fn bind_data_to(
    pipeline: Arc<ComputePipeline>,
    data_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .expect("cant create descriptor set")
}

fn bind_imageview_to(
    pipeline: Arc<ComputePipeline>,
    img_view: Arc<dyn ImageViewAbstract>,
) -> Arc<PersistentDescriptorSet> {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, img_view.clone())],
    )
    .expect("cant create descriptor set")
}

fn shader_pipeline_compute(
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader: Arc<ShaderModule>,
) -> (Arc<ShaderModule>, Arc<ComputePipeline>) {
    // let shader = cs::load(device.clone()).expect("failed to create shader module");
    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("cant create compute pipeline");

    (shader, pipeline)
}

fn compute_context(device: Arc<Device>, queue: Arc<Queue>) -> GraphicsContext {
    GraphicsContext::new(device, queue)
}

fn perform_compute(device: Arc<Device>, queue: Arc<Queue>) {
    let gc = compute_context(device.clone(), queue.clone());
    let (shader, pipeline) = shader_pipeline_compute(
        device.clone(),
        queue.clone(),
        cs::load_mult_const(device.clone()).expect("failed to create shader module"),
    );
    let data_buffer = create_integer_buffer(device.clone());
    let set = bind_data_to(pipeline.clone(), data_buffer.clone());
    let mut command_builder = gc.create_command_builder();

    command_builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024, 1, 1])
        .unwrap();

    let command_buffer = command_builder.build().unwrap();

    execute_command_buffer(&gc, command_buffer);

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as i32 * 12);
    }
    println!("Successful compute run!");

    // testing image creation
    let (_, pipeline) = shader_pipeline_compute(
        gc.device.clone(),
        queue.clone(),
        cs::load_test_compute(device).unwrap(),
    );
    let image = create_image(&gc);
    let image_view = ImageView::new_default(image.clone()).unwrap();
    let set = bind_imageview_to(pipeline.clone(), image_view);
    let mut command_builder = gc.create_command_builder();
    clear_image(&gc, image.clone(), &mut command_builder);
    command_builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap();
    let finish_export = export_image(&gc, image, &mut command_builder);
    let command_buffer = command_builder.build().unwrap();
    execute_command_buffer(&gc, command_buffer);
    finish_export(&gc);
}
