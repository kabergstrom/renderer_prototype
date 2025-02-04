use crate::{
    BufferAsset, ComputePipelineAsset, GraphicsPipelineAsset, ImageAsset, MaterialAsset,
    MaterialInstanceAsset, RenderpassAsset, SamplerAsset, ShaderAsset,
};
use atelier_assets::loader::storage::IndirectionTable;
use atelier_assets::loader::LoadHandle;
use atelier_assets::loader::Loader;
use fnv::FnvHashMap;

//
// Represents a single asset which may simultaneously have committed and uncommitted loaded state
//
pub struct LoadedAssetState<AssetT> {
    pub committed: Option<AssetT>,
    pub uncommitted: Option<AssetT>,
}

impl<AssetT> Default for LoadedAssetState<AssetT> {
    fn default() -> Self {
        LoadedAssetState {
            committed: None,
            uncommitted: None,
        }
    }
}

fn resolve_load_handle(
    load_handle: LoadHandle,
    indirection_table: &IndirectionTable,
) -> Option<LoadHandle> {
    if load_handle.is_indirect() {
        indirection_table.resolve(load_handle)
    } else {
        Some(load_handle)
    }
}

pub struct AssetLookup<AssetT> {
    //TODO: Slab these for faster lookup?
    pub loaded_assets: FnvHashMap<LoadHandle, LoadedAssetState<AssetT>>,
    pub indirection_table: IndirectionTable,
}

impl<AssetT> AssetLookup<AssetT> {
    pub fn new(loader: &Loader) -> Self {
        AssetLookup {
            loaded_assets: Default::default(),
            indirection_table: loader.indirection_table(),
        }
    }

    pub fn set_uncommitted(
        &mut self,
        load_handle: LoadHandle,
        loaded_asset: AssetT,
    ) {
        log::trace!("set_uncommitted {:?}", load_handle);
        debug_assert!(!load_handle.is_indirect());
        self.loaded_assets
            .entry(load_handle)
            .or_default()
            .uncommitted = Some(loaded_asset);
    }

    pub fn commit(
        &mut self,
        load_handle: LoadHandle,
    ) {
        log::trace!("commit {:?}", load_handle);
        debug_assert!(!load_handle.is_indirect());
        let state = self.loaded_assets.get_mut(&load_handle).unwrap();
        state.committed = state.uncommitted.take();
    }

    pub fn free(
        &mut self,
        load_handle: LoadHandle,
    ) {
        log::trace!("free {:?}", load_handle);
        debug_assert!(!load_handle.is_indirect());
        let old = self.loaded_assets.remove(&load_handle);
        assert!(old.is_some());
    }

    pub fn get_latest(
        &self,
        load_handle: LoadHandle,
    ) -> Option<&AssetT> {
        let load_handle = resolve_load_handle(load_handle, &self.indirection_table)?;

        if let Some(loaded_assets) = self.loaded_assets.get(&load_handle) {
            if let Some(uncommitted) = &loaded_assets.uncommitted {
                Some(uncommitted)
            } else if let Some(committed) = &loaded_assets.committed {
                Some(committed)
            } else {
                // It's an error to reach here because of uncommitted and committed are none, there
                // shouldn't be an entry in loaded_assets
                unreachable!();
            }
        } else {
            None
        }
    }

    pub fn get_committed(
        &self,
        load_handle: LoadHandle,
    ) -> Option<&AssetT> {
        let load_handle = resolve_load_handle(load_handle, &self.indirection_table)?;

        if let Some(loaded_assets) = self.loaded_assets.get(&load_handle) {
            if let Some(committed) = &loaded_assets.committed {
                Some(committed)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.loaded_assets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.loaded_assets.is_empty()
    }

    pub fn destroy(&mut self) {
        self.loaded_assets.clear();
    }
}

#[derive(Debug)]
pub struct LoadedAssetMetrics {
    pub shader_module_count: usize,
    pub graphics_pipeline_count: usize,
    pub compute_pipeline_count: usize,
    pub renderpass_count: usize,
    pub material_count: usize,
    pub material_instance_count: usize,
    pub sampler_count: usize,
    pub image_count: usize,
    pub buffer_count: usize,
}

//
// Lookups by asset for loaded asset state
//
pub struct AssetLookupSet {
    pub shader_modules: AssetLookup<ShaderAsset>,
    pub graphics_pipelines: AssetLookup<GraphicsPipelineAsset>,
    pub compute_pipelines: AssetLookup<ComputePipelineAsset>,
    pub renderpasses: AssetLookup<RenderpassAsset>,
    pub materials: AssetLookup<MaterialAsset>,
    pub material_instances: AssetLookup<MaterialInstanceAsset>,
    pub samplers: AssetLookup<SamplerAsset>,
    pub images: AssetLookup<ImageAsset>,
    pub buffers: AssetLookup<BufferAsset>,
}

impl AssetLookupSet {
    pub fn new(loader: &Loader) -> AssetLookupSet {
        AssetLookupSet {
            shader_modules: AssetLookup::new(loader),
            graphics_pipelines: AssetLookup::new(loader),
            compute_pipelines: AssetLookup::new(loader),
            renderpasses: AssetLookup::new(loader),
            materials: AssetLookup::new(loader),
            material_instances: AssetLookup::new(loader),
            samplers: AssetLookup::new(loader),
            images: AssetLookup::new(loader),
            buffers: AssetLookup::new(loader),
        }
    }

    pub fn metrics(&self) -> LoadedAssetMetrics {
        LoadedAssetMetrics {
            shader_module_count: self.shader_modules.len(),
            graphics_pipeline_count: self.graphics_pipelines.len(),
            compute_pipeline_count: self.compute_pipelines.len(),
            renderpass_count: self.renderpasses.len(),
            material_count: self.materials.len(),
            material_instance_count: self.material_instances.len(),
            sampler_count: self.samplers.len(),
            image_count: self.images.len(),
            buffer_count: self.buffers.len(),
        }
    }

    pub fn destroy(&mut self) {
        self.shader_modules.destroy();
        self.graphics_pipelines.destroy();
        self.compute_pipelines.destroy();
        self.renderpasses.destroy();
        self.materials.destroy();
        self.material_instances.destroy();
        self.samplers.destroy();
        self.images.destroy();
        self.buffers.destroy();
    }
}
