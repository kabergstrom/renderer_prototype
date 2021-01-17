use log::LevelFilter;

use rafx_api::*;
use rafx_nodes::SubmitNode;
use rafx_resources::graph::{
    RenderGraphBuilder, RenderGraphExecutor, RenderGraphImageConstraint, RenderGraphImageExtents,
    RenderGraphImageSpecification, RenderGraphNodeCallbacks, RenderGraphQueue,
    SwapchainSurfaceInfo,
};
use rafx_resources::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, FixedFunctionState, MaterialPassVertexInput,
    ShaderModuleHash, ShaderModuleResourceDef, VertexDataLayout,
};
use std::sync::Arc;

const WINDOW_WIDTH: u32 = 900;
const WINDOW_HEIGHT: u32 = 600;

fn main() {
    env_logger::Builder::from_default_env()
        .default_format_timestamp_nanos(true)
        .filter_level(LevelFilter::Debug)
        .init();

    run().unwrap();
}

#[derive(Default, Clone, Copy)]
struct PositionColorVertex {
    position: [f32; 2],
    color: [f32; 3],
}

fn run() -> RafxResult<()> {
    //
    // Init SDL2
    //
    let sdl2_systems = sdl2_init();

    //
    // Create the api
    //
    let mut api = RafxApi::new_vulkan(
        &sdl2_systems.window,
        &RafxApiDef {
            validation_mode: RafxValidationMode::EnabledIfAvailable,
        },
        &Default::default(),
    )?;

    // Wrap all of this so that it gets dropped
    {
        let device_context = api.device_context();

        //
        // Create a swapchain
        //
        let (window_width, window_height) = sdl2_systems.window.drawable_size();
        let swapchain = device_context.create_swapchain(
            &sdl2_systems.window,
            &RafxSwapchainDef {
                width: window_width,
                height: window_height,
                enable_vsync: true,
            },
        )?;

        //
        // Wrap the swapchain in this helper to cut down on boilerplate. This helper is
        // multithreaded-rendering friendly! The PresentableFrame it returns can be sent to another
        // thread and presented from there, and any errors are returned back to the main thread
        // when the next image is acquired. The helper also ensures that the swapchain is rebuilt
        // as necessary.
        //
        let mut swapchain_helper = RafxSwapchainHelper::new(&device_context, swapchain, None)?;

        //
        // Allocate a graphics queue. By default, there is just one graphics queue and it is shared.
        // There currently is no API for customizing this but the code would be easy to adapt to act
        // differently. Most recommendations I've seen are to just use one graphics queue. (The
        // rendering hardware is shared among them)
        //
        let graphics_queue = device_context.create_queue(RafxQueueType::Graphics)?;

        //
        // Create a ResourceContext. The Resource
        //
        let render_registry = rafx_nodes::RenderRegistryBuilder::default()
            .register_render_phase::<OpaqueRenderPhase>("Opaque")
            .build();
        let mut resource_manager =
            rafx_resources::ResourceManager::new(&device_context, &render_registry);
        let resource_context = resource_manager.resource_context();

        //
        // Load a shader from source - this part is API-specific. vulkan will want SPV, metal wants
        // source code or even better a pre-compiled library. But their compile toolchain only works on
        // mac/windows and is a command line tool without programmatic access. In an engine, it
        // would be better to pack different formats depending on the platform being built. For this
        // example, we'll just do it manually.
        //
        // Accessing the underlying API is straightforward - all Rafx objects (i.e. RafxTexture,
        // RafxBuffer, RafxShaderModule) have an accessor for the API-specific implementation. For
        // example, device_context.vk_device_context() returns an Option<&RafxDeviceContextVulkan>
        // that can be unwrapped. This allows accessing API-specific details.
        //
        // The resulting shader modules represent a loaded shader blob that can be used to create the
        // shader. (They can be discarded once the graphics pipeline is built.)
        //

        // Load Vec<u8> from files
        let vert_source_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/triangle_graph/shader.vert.spv");
        let frag_source_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/triangle_graph/shader.frag.spv");

        let vert_bytes = std::fs::read(vert_source_path)?;
        let frag_bytes = std::fs::read(frag_source_path)?;

        // Load the data as shader modules in the resource manager
        let vert_shader_module_def =
            RafxShaderModuleDef::Vk(RafxShaderModuleDefVulkan::SpvBytes(&vert_bytes));
        let shader_module_resource_def = Arc::new(ShaderModuleResourceDef {
            shader_module_hash: ShaderModuleHash::new(&vert_shader_module_def),
            code: vert_bytes,
        });
        let vert_shader_module = resource_context
            .resources()
            .get_or_create_shader_module(&shader_module_resource_def)?;

        let frag_shader_module_def =
            RafxShaderModuleDef::Vk(RafxShaderModuleDefVulkan::SpvBytes(&frag_bytes));
        let frag_shader_module_resource_def = Arc::new(ShaderModuleResourceDef {
            shader_module_hash: ShaderModuleHash::new(&frag_shader_module_def),
            code: frag_bytes,
        });
        let frag_shader_module = resource_context
            .resources()
            .get_or_create_shader_module(&frag_shader_module_resource_def)?;

        //
        // Create the shader object by combining the stages
        //
        // Hardcode the reflecton data required to interact with the shaders. This can be generated
        // offline and loaded with the shader but this is not currently provided in rafx-resources
        // itself. (But see the shader pipeline in higher-level rafx crates for example usage,
        // generated from spirv_cross)
        //

        // For this example we'll show exposing this to both the vertex and fragment shader
        let shader_color_resource = RafxShaderResource {
            name: Some("color".to_string()),
            set_index: 0,
            binding: 0,
            resource_type: RafxResourceType::UNIFORM_BUFFER,
            ..Default::default()
        };

        let vert_shader_stage_def = RafxShaderStageDef {
            shader_stage: RafxShaderStageFlags::VERTEX,
            entry_point: "main".to_string(),
            shader_module: vert_shader_module.get_raw().shader_module.clone(),
            resources: vec![
                // Example binding
                shader_color_resource.clone(),
            ],
        };

        let frag_shader_stage_def = RafxShaderStageDef {
            shader_stage: RafxShaderStageFlags::FRAGMENT,
            entry_point: "main".to_string(),
            shader_module: frag_shader_module.get_raw().shader_module.clone(),
            resources: vec![
                // Example binding
                shader_color_resource.clone(),
            ],
        };

        //
        // Combine the shader stages into a single shader
        //
        let shader = resource_context.resources().get_or_create_shader(
            &[vert_shader_stage_def, frag_shader_stage_def],
            &[vert_shader_module, frag_shader_module],
        )?;

        //
        // Create the root signature object - it represents the pipeline layout and can be shared
        // among shaders. But one per shader is fine.
        //
        let root_signature = resource_context.resources().get_or_create_root_signature(
            &[shader.clone()],
            &[],
            &[],
        )?;

        //
        // Rafx resources provides a high-level wrapper around a descriptor set layout that adds
        // additional functionality. We can configure that here now (or generate it from reflection
        // or custom annotation within shaders). For this example we'll just hardcode it. The demo
        // reads annotations in shader data and produces reflection so that this is automatic.
        //
        let descriptor_set_layout = resource_context
            .resources()
            .get_or_create_descriptor_set_layout(
                &root_signature,
                0,
                &DescriptorSetLayout {
                    bindings: vec![DescriptorSetLayoutBinding {
                        resource: shader_color_resource.clone(),
                        immutable_samplers: None,
                        internal_buffer_per_descriptor_size: Some(16), // A single f32 vec4
                    }],
                },
            )?;

        //
        // Now set up the fixed function and vertex input state. LOTS of things can be configured
        // here, but aside from the vertex layout most of it can be left as default.
        //
        let fixed_function_state = Arc::new(FixedFunctionState {
            rasterizer_state: Default::default(),
            depth_state: Default::default(),
            blend_state: Default::default(),
        });

        // These names will need to match the vertex layout below
        let vertex_inputs = Arc::new(vec![
            MaterialPassVertexInput {
                semantic: "POSITION".to_string(),
                location: 0,
            },
            MaterialPassVertexInput {
                semantic: "COLOR".to_string(),
                location: 1,
            },
        ]);

        //
        // Create the material pass. A material pass encapsulates everything necessary to create a
        // graphics pipeline to render a single pass for a material
        //
        let material_pass = resource_context.resources().get_or_create_material_pass(
            shader,
            root_signature,
            vec![descriptor_set_layout],
            fixed_function_state,
            vertex_inputs,
        )?;

        //
        // The vertex format does not need to be specified up-front to create the material pass.
        // This allows a single material to be used with vertex data stored in any format. While we
        // don't need to create it just yet, we'll do it here once and put it in an arc so we can
        // easily use it later without having to reconstruct every frame.
        //
        let vertex_layout = Arc::new(
            VertexDataLayout::build_vertex_layout(
                &PositionColorVertex::default(),
                |builder, vertex| {
                    builder.add_member(&vertex.position, "POSITION", RafxFormat::R32G32_SFLOAT);
                    builder.add_member(&vertex.color, "COLOR", RafxFormat::R32G32B32_SFLOAT);
                },
            )
            .into_set(RafxPrimitiveTopology::TriangleList),
        );

        let start_time = std::time::Instant::now();

        //
        // SDL2 window pumping
        //
        log::info!("Starting window event loop");
        let mut event_pump = sdl2_systems
            .context
            .event_pump()
            .expect("Could not create sdl event pump");

        'running: loop {
            if !process_input(&mut event_pump) {
                break 'running;
            }

            let current_time = std::time::Instant::now();
            let seconds = (current_time - start_time).as_secs_f32();

            //
            // Acquire swapchain image
            //
            let (window_width, window_height) = sdl2_systems.window.vulkan_drawable_size();
            let presentable_frame =
                swapchain_helper.acquire_next_image(window_width, window_height, None)?;

            //
            // Mark the previous frame complete. This causes old resources that are no longer in
            // use to be dropped. It needs to go after the acquire image, because the acquire image
            // waits on the *gpu* to finish the frame.
            //
            resource_manager.on_frame_complete()?;

            //
            // Register the swapchain image as a resource - this allows us to treat it like any
            // other resource. However keep in mind the image belongs to the swapchain. So holding
            // references to it beyond a single frame is dangerous!
            //
            let swapchain_image = resource_context
                .resources()
                .insert_render_target(presentable_frame.render_target().clone());

            let swapchain_image_view = resource_context
                .resources()
                .get_or_create_image_view(&swapchain_image, None)?;

            //
            // Create a graph to describe how we will draw the frame. Here we just have a single
            // renderpass with a color attachment. See the demo for more complex example usage.
            //
            let mut graph_builder = RenderGraphBuilder::default();
            let mut graph_callbacks = RenderGraphNodeCallbacks::<()>::default();

            let node = graph_builder.add_node("opaque", RenderGraphQueue::DefaultGraphics);
            let color_attachment = graph_builder.create_color_attachment(
                node,
                0,
                Some(RafxColorClearValue([0.0, 0.0, 0.0, 0.0])),
                RenderGraphImageConstraint {
                    samples: Some(RafxSampleCount::SampleCount1),
                    format: Some(swapchain_helper.format()),
                    ..Default::default()
                },
                Default::default(),
            );
            graph_builder.set_image_name(color_attachment, "color");

            //
            // The callback will be run when the graph is executed. We clone a few things and
            // capture them in this closure. We could alternatively create an arbitrary struct and
            // pass it in as a "user context".
            //
            let captured_vertex_layout = vertex_layout.clone();
            let captured_material_pass = material_pass.clone();
            graph_callbacks.set_renderpass_callback(node, move |args, _user_context| {
                let vertex_layout = &captured_vertex_layout;
                let material_pass = &captured_material_pass;

                //
                // Some data we will draw
                //
                #[rustfmt::skip]
                let vertex_data = [
                    PositionColorVertex { position: [0.0, 0.5], color: [1.0, 0.0, 0.0] },
                    PositionColorVertex { position: [-0.5 + (seconds.cos() / 2. + 0.5), -0.5], color: [0.0, 1.0, 0.0] },
                    PositionColorVertex { position: [0.5 - (seconds.cos() / 2. + 0.5), -0.5], color: [0.0, 0.0, 1.0] },
                ];

                assert_eq!(20, std::mem::size_of::<PositionColorVertex>());

                let color = (seconds.cos() + 1.0) / 2.0;
                let uniform_data = [color, 0.0, 1.0 - color, 1.0];

                //
                // Here we create a vertex buffer. Since we only use it once we won't bother putting
                // it into dedicated GPU memory.
                //
                // The vertex_buffer is ref-counted and can be kept around as long as you like. The
                // resource manager will ensure it stays allocated until enough frames are presented
                // that it's safe to delete.
                //
                // The resource allocators should be used and dropped, not kept around. They are
                // pooled/re-used.
                //
                let resource_allocator = args.graph_context.resource_context().create_dyn_resource_allocator_set();
                let vertex_buffer = args.graph_context.device_context().create_buffer(
                    &RafxBufferDef::for_staging_vertex_buffer_data(&vertex_data)
                )?;

                vertex_buffer.copy_to_host_visible_buffer(&vertex_data)?;

                let vertex_buffer = resource_allocator.insert_buffer(vertex_buffer);

                //
                // Create a descriptor set. USUALLY - you can use the autogenerated code from the shader pipeline
                // in higher level rafx crates to make this more straightforward - this is shown in the demo.
                // Also, flush_changes is automatically called when dropped, we only have to call it
                // here because we immediately use the descriptor set.
                //
                // Once the descriptor set is created, it's ref-counted and you can keep it around
                // as long as you like. The resource manager will ensure it stays allocated
                // until enough frames are presented that it's safe to delete.
                //
                // The allocator should be used and dropped, not kept around. It is pooled/re-used.
                // flush_changes is automatically called on drop.
                //
                let descriptor_set_layout = material_pass
                    .get_raw()
                    .descriptor_set_layouts[0]
                    .clone();

                let mut descriptor_set_allocator = args.graph_context.resource_context().create_descriptor_set_allocator();
                let mut dyn_descriptor_set = descriptor_set_allocator.create_dyn_descriptor_set_uninitialized(&descriptor_set_layout)?;
                dyn_descriptor_set.set_buffer_data(0, &uniform_data);
                dyn_descriptor_set.flush(&mut descriptor_set_allocator)?;
                descriptor_set_allocator.flush_changes()?;

                // At this point if we don't intend to change the descriptor, we can grab the
                // descriptor set inside and use it as a ref-counted resource.
                let descriptor_set = dyn_descriptor_set.descriptor_set();

                //
                // Fetch the pipeline. If we have a pipeline for this material that's compatible with
                // the render target and vertex layout, we'll use it. Otherwise, we create it.
                //
                // The render phase is not really utilized to the full extent in this demo, but it
                // would normally help pair materials with render targets, ensuring newly loaded
                // materials can create pipelines ahead-of-time, off the render codepath.
                //
                let pipeline = args
                    .graph_context
                    .resource_context()
                    .graphics_pipeline_cache()
                    .get_or_create_graphics_pipeline(
                    OpaqueRenderPhase::render_phase_index(),
                    &material_pass,
                    &args.render_target_meta,
                    &vertex_layout
                )?;

                //
                // We have everything needed to draw now, write instruction to the command buffer
                //
                let cmd_buffer = args.command_buffer;
                cmd_buffer.cmd_bind_pipeline(&pipeline.get_raw().pipeline)?;
                cmd_buffer.cmd_bind_vertex_buffers(
                    0,
                    &[RafxVertexBufferBinding {
                        buffer: &vertex_buffer.get_raw().buffer,
                        offset: 0,
                    }],
                )?;
                descriptor_set.bind(&cmd_buffer)?;
                cmd_buffer.cmd_draw(3, 0)?;

                Ok(())
            });

            //
            // Flag the color attachment as needing to output to the swapchain image. This is not a
            // copy - the graph walks backwards from outputs so that it operates directly on the
            // intended output image where possible. It only creates additional resources if
            // necessary.
            //
            graph_builder.set_output_image(
                color_attachment,
                swapchain_image_view,
                RenderGraphImageSpecification {
                    samples: RafxSampleCount::SampleCount1,
                    format: swapchain_helper.format(),
                    resource_type: RafxResourceType::TEXTURE,
                    extents: RenderGraphImageExtents::MatchSurface,
                    layer_count: 1,
                    mip_count: 1,
                },
                Default::default(),
                RafxResourceState::PRESENT,
            );

            //
            // Prepare to run the graph. We create an executor to allocate resources and run through
            // the graph, dispatching callbacks as needed to record instructions to command buffers
            //
            let swapchain_def = swapchain_helper.swapchain_def();
            let swapchain_surface_info = SwapchainSurfaceInfo {
                format: swapchain_helper.format(),
                extents: RafxExtents2D {
                    width: swapchain_def.width,
                    height: swapchain_def.height,
                },
            };

            let executor = RenderGraphExecutor::new(
                &device_context,
                &resource_context,
                graph_builder,
                &swapchain_surface_info,
                graph_callbacks,
            )?;

            //
            // Execute the graph. This will write out command buffer(s)
            //
            let command_buffers = executor.execute_graph(&(), &graphics_queue)?;

            //
            // Submit the command buffers to the GPU
            //
            let refs: Vec<&RafxCommandBuffer> = command_buffers.iter().map(|x| &**x).collect();
            presentable_frame.present(&graphics_queue, &refs)?;
        }
    }

    // Optional, but calling this verifies that all rafx objects/device contexts have been
    // destroyed and where they were created. Good for finding unintended leaks!
    api.destroy()?;

    Ok(())
}

//
// A phase combines renderables that may come from different features. This example doesnt't use
// render nodes fully, but the pipeline cache uses it to define which renderpass/material pairs
//
use rafx_nodes::RenderPhase;
use rafx_nodes::RenderPhaseIndex;

rafx_nodes::declare_render_phase!(
    OpaqueRenderPhase,
    OPAQUE_RENDER_PHASE_INDEX,
    opaque_render_phase_sort_submit_nodes
);

#[profiling::function]
fn opaque_render_phase_sort_submit_nodes(mut submit_nodes: Vec<SubmitNode>) -> Vec<SubmitNode> {
    // Sort by feature
    log::trace!(
        "Sort phase {}",
        OpaqueRenderPhase::render_phase_debug_name()
    );
    submit_nodes.sort_unstable_by(|a, b| a.feature_index().cmp(&b.feature_index()));

    submit_nodes
}

//
// SDL2 helpers
//
pub struct Sdl2Systems {
    pub context: sdl2::Sdl,
    pub video_subsystem: sdl2::VideoSubsystem,
    pub window: sdl2::video::Window,
}

pub fn sdl2_init() -> Sdl2Systems {
    // Setup SDL
    let context = sdl2::init().expect("Failed to initialize sdl2");
    let video_subsystem = context
        .video()
        .expect("Failed to create sdl video subsystem");

    // Create the window
    let window = video_subsystem
        .window("Rafx Example", WINDOW_WIDTH, WINDOW_HEIGHT)
        .position_centered()
        .allow_highdpi()
        .resizable()
        .build()
        .expect("Failed to create window");

    Sdl2Systems {
        context,
        video_subsystem,
        window,
    }
}

fn process_input(event_pump: &mut sdl2::EventPump) -> bool {
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    for event in event_pump.poll_iter() {
        //log::trace!("{:?}", event);
        match event {
            //
            // Halt if the user requests to close the window
            //
            Event::Quit { .. } => return false,

            //
            // Close if the escape key is hit
            //
            Event::KeyDown {
                keycode: Some(keycode),
                keymod: _modifiers,
                ..
            } => {
                //log::trace!("Key Down {:?} {:?}", keycode, modifiers);
                if keycode == Keycode::Escape {
                    return false;
                }
            }

            _ => {}
        }
    }

    true
}