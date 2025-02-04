use crate::assets::gltf::MeshAssetData;
use crate::game_asset_lookup::{
    GameLoadedAssetLookupSet, GameLoadedAssetMetrics, MeshAsset, MeshAssetInner, MeshAssetPart,
};
use crate::phases::{OpaqueRenderPhase, ShadowMapRenderPhase};
use ash::prelude::VkResult;
use atelier_assets::loader::handle::AssetHandle;
use atelier_assets::loader::handle::Handle;
use atelier_assets::loader::storage::AssetLoadOp;
use atelier_assets::loader::Loader;
use crossbeam_channel::Sender;
use rafx::{assets::{AssetLookup, AssetManager, GenericLoader, LoadQueues}, resources::VertexDataSetLayout};
use std::sync::Arc;

#[derive(Debug)]
pub struct GameAssetManagerMetrics {
    pub game_loaded_asset_metrics: GameLoadedAssetMetrics,
}

#[derive(Default)]
pub struct GameLoadQueueSet {
    pub meshes: LoadQueues<MeshAssetData, MeshAsset>,
}

pub struct GameAssetManager {
    loaded_assets: GameLoadedAssetLookupSet,
    load_queues: GameLoadQueueSet,
}

impl GameAssetManager {
    pub fn new(loader: &Loader) -> Self {
        GameAssetManager {
            loaded_assets: GameLoadedAssetLookupSet::new(loader),
            load_queues: Default::default(),
        }
    }

    pub fn create_mesh_loader(&self) -> GenericLoader<MeshAssetData, MeshAsset> {
        self.load_queues.meshes.create_loader()
    }

    pub fn mesh(
        &self,
        handle: &Handle<MeshAsset>,
    ) -> Option<&MeshAsset> {
        self.loaded_assets
            .meshes
            .get_committed(handle.load_handle())
    }

    // Call whenever you want to handle assets loading/unloading
    #[profiling::function]
    pub fn update_asset_loaders(
        &mut self,
        asset_manager: &AssetManager,
    ) -> VkResult<()> {
        self.process_mesh_load_requests(asset_manager);
        Ok(())
    }

    pub fn metrics(&self) -> GameAssetManagerMetrics {
        let game_loaded_asset_metrics = self.loaded_assets.metrics();

        GameAssetManagerMetrics {
            game_loaded_asset_metrics,
        }
    }

    #[profiling::function]
    fn process_mesh_load_requests(
        &mut self,
        asset_manager: &AssetManager,
    ) {
        for request in self.load_queues.meshes.take_load_requests() {
            log::trace!("Create mesh {:?}", request.load_handle);
            let loaded_asset = self.load_mesh(asset_manager, &request.asset);
            Self::handle_load_result(
                request.load_op,
                loaded_asset,
                &mut self.loaded_assets.meshes,
                request.result_tx,
            );
        }

        Self::handle_commit_requests(&mut self.load_queues.meshes, &mut self.loaded_assets.meshes);
        Self::handle_free_requests(&mut self.load_queues.meshes, &mut self.loaded_assets.meshes);
    }

    fn handle_load_result<AssetT: Clone>(
        load_op: AssetLoadOp,
        loaded_asset: VkResult<AssetT>,
        asset_lookup: &mut AssetLookup<AssetT>,
        result_tx: Sender<AssetT>,
    ) {
        match loaded_asset {
            Ok(loaded_asset) => {
                asset_lookup.set_uncommitted(load_op.load_handle(), loaded_asset.clone());
                result_tx.send(loaded_asset).unwrap();
                load_op.complete()
            }
            Err(err) => {
                load_op.error(err);
            }
        }
    }

    fn handle_commit_requests<AssetDataT, AssetT>(
        load_queues: &mut LoadQueues<AssetDataT, AssetT>,
        asset_lookup: &mut AssetLookup<AssetT>,
    ) {
        for request in load_queues.take_commit_requests() {
            log::trace!(
                "commit asset {:?} {}",
                request.load_handle,
                core::any::type_name::<AssetDataT>()
            );
            asset_lookup.commit(request.load_handle);
        }
    }

    fn handle_free_requests<AssetDataT, AssetT>(
        load_queues: &mut LoadQueues<AssetDataT, AssetT>,
        asset_lookup: &mut AssetLookup<AssetT>,
    ) {
        for request in load_queues.take_commit_requests() {
            asset_lookup.commit(request.load_handle);
        }
    }

    #[profiling::function]
    fn load_mesh(
        &mut self,
        asset_manager: &AssetManager,
        mesh_asset: &MeshAssetData,
    ) -> VkResult<MeshAsset> {
        let vertex_buffer = asset_manager
            .loaded_assets()
            .buffers
            .get_latest(mesh_asset.buffer.load_handle())
            .unwrap()
            .buffer
            .clone();

        let mesh_parts: Vec<_> = mesh_asset
            .mesh_parts
            .iter()
            .map(|mesh_part| {
                let material_instance = asset_manager
                    .loaded_assets()
                    .material_instances
                    .get_committed(mesh_part.material_instance.load_handle())
                    .unwrap();

                let opaque_pass_index = material_instance
                    .material
                    .find_pass_by_phase::<OpaqueRenderPhase>();
                if opaque_pass_index.is_none() {
                    log::error!(
                        "A mesh part with material {:?} has no opaque phase",
                        material_instance.material_handle
                    );
                    return None;
                }
                let opaque_pass_index = opaque_pass_index.unwrap();

                //NOTE: For now require this, but we might want to disable shadow casting, in which
                // case no material is necessary
                let shadow_map_pass_index = material_instance
                    .material
                    .find_pass_by_phase::<ShadowMapRenderPhase>();
                if shadow_map_pass_index.is_none() {
                    log::error!(
                        "A mesh part with material {:?} has no shadow map phase",
                        material_instance.material_handle
                    );
                    return None;
                }

                const PER_MATERIAL_DESCRIPTOR_SET_LAYOUT_INDEX: usize = 1;

                let mut layout_offsets = Vec::new();
                for (_, offset) in &mesh_part.vertex_layouts {
                    layout_offsets.push(*offset);
                }
                let layout_set = VertexDataSetLayout::new(mesh_part.vertex_layouts.iter().map(|(layout, _)| layout.clone()).collect());
                Some(MeshAssetPart {
                    opaque_pass: material_instance.material.passes[opaque_pass_index].clone(),
                    opaque_material_descriptor_set: material_instance.material_descriptor_sets
                        [opaque_pass_index][PER_MATERIAL_DESCRIPTOR_SET_LAYOUT_INDEX]
                        .as_ref()
                        .unwrap()
                        .clone(),
                    shadow_map_pass: shadow_map_pass_index
                        .map(|pass_index| material_instance.material.passes[pass_index].clone()),
                    vertex_layouts: layout_set,
                    vertex_binding_buffer_offsets: layout_offsets,
                    index_buffer_offset: mesh_part.index_layout.1,
                    num_vertices: mesh_part.num_vertices,
                    num_indices: mesh_part.num_indices,
                })
            })
            .collect();

        let inner = MeshAssetInner {
            vertex_buffer,
            asset_data: mesh_asset.clone(),
            mesh_parts,
        };

        Ok(MeshAsset {
            inner: Arc::new(inner),
        })
    }
}

impl Drop for GameAssetManager {
    fn drop(&mut self) {
        log::info!("Cleaning up game resource manager");
        log::trace!("Game Resource Manager Metrics:\n{:#?}", self.metrics());

        // Wipe out any loaded assets. This will potentially drop ref counts on resources
        self.loaded_assets.destroy();

        log::info!("Dropping game resource manager");
        log::trace!("Resource Game Manager Metrics:\n{:#?}", self.metrics());
    }
}
