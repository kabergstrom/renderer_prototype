use crate::assets::gltf::MeshAssetData;
use atelier_assets::loader::Loader;
use rafx::{assets::AssetLookup, resources::{VertexDataLayout, VertexDataSetLayout}};
use rafx::assets::MaterialPass;
use rafx::resources::{BufferResource, DescriptorSetArc, ResourceArc};
use std::sync::Arc;
use type_uuid::*;

pub struct MeshAssetPart {
    pub opaque_pass: MaterialPass,
    pub opaque_material_descriptor_set: DescriptorSetArc,
    // These are optional because we might want to disable casting shadows
    pub shadow_map_pass: Option<MaterialPass>,
    pub vertex_layouts: VertexDataSetLayout, 
    pub vertex_binding_buffer_offsets: Vec<usize>, // Vertex buffer offset corresponding to binding number in VertexDataSetLayout
    pub index_buffer_offset: usize, 
    pub num_vertices: usize,
    pub num_indices: usize,
}

pub struct MeshAssetInner {
    pub mesh_parts: Vec<Option<MeshAssetPart>>,
    pub vertex_buffer: ResourceArc<BufferResource>,
    pub asset_data: MeshAssetData,
}

#[derive(TypeUuid, Clone)]
#[uuid = "689a0bf0-e320-41c0-b4e8-bdb2055a7a57"]
pub struct MeshAsset {
    pub inner: Arc<MeshAssetInner>,
}

#[derive(Debug)]
pub struct GameLoadedAssetMetrics {
    pub mesh_count: usize,
}

//
// Lookups by asset for loaded asset state
//
pub struct GameLoadedAssetLookupSet {
    pub meshes: AssetLookup<MeshAsset>,
}

impl GameLoadedAssetLookupSet {
    pub fn new(loader: &Loader) -> Self {
        GameLoadedAssetLookupSet {
            meshes: AssetLookup::new(loader),
        }
    }

    pub fn metrics(&self) -> GameLoadedAssetMetrics {
        GameLoadedAssetMetrics {
            mesh_count: self.meshes.len(),
        }
    }

    pub fn destroy(&mut self) {
        self.meshes.destroy();
    }
}
