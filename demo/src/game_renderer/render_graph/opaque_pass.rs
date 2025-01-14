use crate::phases::OpaqueRenderPhase;
use crate::render_contexts::RenderJobWriteContext;
use ash::vk;
use rafx::graph::*;
use rafx::resources::vk_description as dsc;

use super::RenderGraphContext;
use super::ShadowMapImageResources;

pub(super) struct OpaquePass {
    pub(super) node: RenderGraphNodeId,
    pub(super) color: RenderGraphImageUsageId,
    pub(super) depth: RenderGraphImageUsageId,
    pub(super) shadow_maps: Vec<RenderGraphImageUsageId>,
}

pub(super) fn opaque_pass(
    context: &mut RenderGraphContext,
    shadow_map_passes: &[ShadowMapImageResources],
) -> OpaquePass {
    let node = context
        .graph
        .add_node("Opaque", RenderGraphQueue::DefaultGraphics);

    let color = context.graph.create_color_attachment(
        node,
        0,
        Some(vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        }),
        RenderGraphImageConstraint {
            samples: Some(context.graph_config.samples),
            format: Some(context.graph_config.color_format),
            ..Default::default()
        },
    );
    context.graph.set_image_name(color, "color");

    let depth = context.graph.create_depth_attachment(
        node,
        Some(vk::ClearDepthStencilValue {
            depth: 0.0,
            stencil: 0,
        }),
        RenderGraphImageConstraint {
            samples: Some(context.graph_config.samples),
            format: Some(context.graph_config.depth_format),
            ..Default::default()
        },
    );
    context.graph.set_image_name(depth, "depth");

    let mut shadow_maps = Vec::with_capacity(shadow_map_passes.len());
    for shadow_map_pass in shadow_map_passes {
        let sampled_image = match shadow_map_pass {
            ShadowMapImageResources::Single(image) => context.graph.sample_image(
                node,
                *image,
                Default::default(),
                RenderGraphImageSubresourceRange::AllMipsAllLayers,
                dsc::ImageViewType::Type2D,
            ),
            ShadowMapImageResources::Cube(cube_map_image) => context.graph.sample_image(
                node,
                *cube_map_image,
                Default::default(),
                RenderGraphImageSubresourceRange::AllMipsAllLayers,
                dsc::ImageViewType::Cube,
            ),
        };
        shadow_maps.push(sampled_image);
    }

    context
        .graph_callbacks
        .add_renderphase_dependency::<OpaqueRenderPhase>(node);

    let main_view = context.main_view.clone();
    context
        .graph_callbacks
        .set_renderpass_callback(node, move |args, user_context| {
            let mut write_context = RenderJobWriteContext::from_graph_visit_render_pass_args(&args);
            user_context
                .prepared_render_data
                .write_view_phase::<OpaqueRenderPhase>(&main_view, &mut write_context);
            Ok(())
        });

    OpaquePass {
        node,
        color,
        depth,
        shadow_maps,
    }
}
