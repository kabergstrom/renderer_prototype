use crate::features::debug3d::extract::Debug3dExtractJob;
use crate::render_contexts::{
    RenderJobExtractContext, RenderJobPrepareContext, RenderJobWriteContext,
};
use atelier_assets::loader::handle::Handle;
use rafx::assets::MaterialAsset;
use rafx::nodes::ExtractJob;
use rafx::nodes::RenderFeature;
use rafx::nodes::RenderFeatureIndex;
use rafx::resources::{VertexDataLayout, VertexDataSetLayout};
use std::convert::TryInto;

mod debug3d_resource;
mod extract;
mod prepare;
mod write;

pub use debug3d_resource::*;

pub fn create_debug3d_extract_job(
) -> Box<dyn ExtractJob<RenderJobExtractContext, RenderJobPrepareContext, RenderJobWriteContext>> {
    Box::new(Debug3dExtractJob::new())
}

pub type Debug3dUniformBufferObject = shaders::debug_vert::PerFrameUboUniform;

/// Vertex format for vertices sent to the GPU
#[derive(Clone, Debug, Copy, Default)]
#[repr(C)]
pub struct Debug3dVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

lazy_static::lazy_static! {
    pub static ref DEBUG_VERTEX_LAYOUT : VertexDataSetLayout = {
        use rafx::resources::vk_description::Format;

        VertexDataLayout::build_vertex_layout(&Debug3dVertex::default(), |builder, vertex| {
            builder.add_member(&vertex.pos, "POSITION", Format::R32G32B32_SFLOAT);
            builder.add_member(&vertex.color, "COLOR", Format::R32G32B32A32_SFLOAT);
        }).into_set()
    };
}

rafx::declare_render_feature!(Debug3dRenderFeature, DEBUG_3D_FEATURE_INDEX);

pub(self) struct ExtractedDebug3dData {
    line_lists: Vec<LineList3D>,
}

#[derive(Debug)]
struct Debug3dDrawCall {
    first_element: u32,
    count: u32,
}
