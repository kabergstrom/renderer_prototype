use crate::assets::BufferAssetData;
use crate::assets::ImageAssetData;
use crate::assets::ShaderAssetData;
use crate::assets::{
    BufferAsset, GraphicsPipelineAsset, ImageAsset, MaterialAsset, MaterialInstanceAsset,
    RenderpassAsset, SamplerAsset, ShaderAsset,
};
use crate::assets::{
    GraphicsPipelineAssetData, MaterialAssetData, MaterialInstanceAssetData, RenderpassAssetData,
    SamplerAssetData,
};
use crate::resource_loader::ResourceLoadResult;
use crate::{ComputePipelineAsset, ComputePipelineAssetData, ResourceLoader};
use atelier_assets::loader::storage::AssetLoadOp;
use atelier_assets::loader::LoadHandle;
use crossbeam_channel::{Receiver, Sender};
use std::marker::PhantomData;
use type_uuid::TypeUuid;

//
// Message handling for asset load/commit/free events
//
pub struct LoadRequest<AssetDataT, AssetT> {
    pub load_handle: LoadHandle,
    pub load_op: AssetLoadOp,
    pub result_tx: Sender<AssetT>,
    pub asset: AssetDataT,
}

pub struct CommitRequest<T> {
    pub load_handle: LoadHandle,
    phantom_data: PhantomData<T>,
}

pub struct FreeRequest<T> {
    pub load_handle: LoadHandle,
    phantom_data: PhantomData<T>,
}

pub struct LoadQueuesTx<AssetDataT, AssetT> {
    load_request_tx: Sender<LoadRequest<AssetDataT, AssetT>>,
    commit_request_tx: Sender<CommitRequest<AssetDataT>>,
    free_request_tx: Sender<FreeRequest<AssetDataT>>,
}

impl<AssetDataT, AssetT> Clone for LoadQueuesTx<AssetDataT, AssetT> {
    fn clone(&self) -> Self {
        LoadQueuesTx {
            load_request_tx: self.load_request_tx.clone(),
            commit_request_tx: self.commit_request_tx.clone(),
            free_request_tx: self.free_request_tx.clone(),
        }
    }
}

pub struct LoadQueuesRx<AssetDataT, AssetT> {
    load_request_rx: Receiver<LoadRequest<AssetDataT, AssetT>>,
    commit_request_rx: Receiver<CommitRequest<AssetDataT>>,
    free_request_rx: Receiver<FreeRequest<AssetDataT>>,
}

pub struct LoadQueues<AssetDataT, AssetT> {
    tx: LoadQueuesTx<AssetDataT, AssetT>,
    rx: LoadQueuesRx<AssetDataT, AssetT>,
}

impl<AssetDataT, AssetT> LoadQueues<AssetDataT, AssetT> {
    pub fn take_load_requests(&mut self) -> Vec<LoadRequest<AssetDataT, AssetT>> {
        self.rx.load_request_rx.try_iter().collect()
    }

    pub fn take_commit_requests(&mut self) -> Vec<CommitRequest<AssetDataT>> {
        self.rx.commit_request_rx.try_iter().collect()
    }

    pub fn take_free_requests(&mut self) -> Vec<FreeRequest<AssetDataT>> {
        self.rx.free_request_rx.try_iter().collect()
    }
}

impl<AssetDataT, AssetT> LoadQueues<AssetDataT, AssetT>
where
    AssetDataT: for<'a> serde::Deserialize<'a> + 'static + Send + Clone,
    AssetT: TypeUuid + 'static + Send,
{
    pub fn create_loader(&self) -> GenericLoader<AssetDataT, AssetT> {
        GenericLoader {
            load_queues: self.tx.clone(),
        }
    }
}

impl<AssetDataT, AssetT> Default for LoadQueues<AssetDataT, AssetT> {
    fn default() -> Self {
        let (load_request_tx, load_request_rx) = crossbeam_channel::unbounded();
        let (commit_request_tx, commit_request_rx) = crossbeam_channel::unbounded();
        let (free_request_tx, free_request_rx) = crossbeam_channel::unbounded();

        let tx = LoadQueuesTx {
            load_request_tx,
            commit_request_tx,
            free_request_tx,
        };

        let rx = LoadQueuesRx {
            load_request_rx,
            commit_request_rx,
            free_request_rx,
        };

        LoadQueues { tx, rx }
    }
}

//
// A generic load handler that allows routing load/commit/free events
//
pub struct GenericLoader<AssetDataT, AssetT>
where
    AssetDataT: for<'a> serde::Deserialize<'a> + 'static + Send,
    AssetT: TypeUuid + 'static + Send,
{
    load_queues: LoadQueuesTx<AssetDataT, AssetT>,
}

impl<AssetDataT, AssetT> ResourceLoader<AssetDataT, AssetT> for GenericLoader<AssetDataT, AssetT>
where
    AssetDataT: for<'a> serde::Deserialize<'a> + 'static + Send,
    AssetT: TypeUuid + 'static + Send,
{
    fn update_asset(
        &mut self,
        load_handle: LoadHandle,
        load_op: AssetLoadOp,
        asset: AssetDataT,
    ) -> ResourceLoadResult<AssetT> {
        log::trace!(
            "GenericLoader update_asset {} {:?}",
            core::any::type_name::<AssetDataT>(),
            load_handle
        );

        let (result_tx, result_rx) = crossbeam_channel::bounded(1);

        let request = LoadRequest {
            load_handle,
            load_op,
            result_tx,
            asset,
        };

        self.load_queues.load_request_tx.send(request).unwrap();
        ResourceLoadResult::new(result_rx)
    }

    fn commit_asset_version(
        &mut self,
        load_handle: LoadHandle,
    ) {
        log::trace!(
            "GenericLoader commit_asset_version {} {:?}",
            core::any::type_name::<AssetDataT>(),
            load_handle
        );
        let request = CommitRequest {
            load_handle,
            phantom_data: Default::default(),
        };

        self.load_queues.commit_request_tx.send(request).unwrap();
    }

    fn free(
        &mut self,
        load_handle: LoadHandle,
    ) {
        log::trace!(
            "GenericLoader free {} {:?}",
            core::any::type_name::<AssetDataT>(),
            load_handle
        );
        let request = FreeRequest {
            load_handle,
            phantom_data: Default::default(),
        };

        self.load_queues.free_request_tx.send(request).unwrap();
    }
}

#[derive(Default)]
pub struct LoadQueueSet {
    pub shader_modules: LoadQueues<ShaderAssetData, ShaderAsset>,
    pub graphics_pipelines: LoadQueues<GraphicsPipelineAssetData, GraphicsPipelineAsset>,
    pub compute_pipelines: LoadQueues<ComputePipelineAssetData, ComputePipelineAsset>,
    pub renderpasses: LoadQueues<RenderpassAssetData, RenderpassAsset>,
    pub materials: LoadQueues<MaterialAssetData, MaterialAsset>,
    pub material_instances: LoadQueues<MaterialInstanceAssetData, MaterialInstanceAsset>,
    pub samplers: LoadQueues<SamplerAssetData, SamplerAsset>,
    pub images: LoadQueues<ImageAssetData, ImageAsset>,
    pub buffers: LoadQueues<BufferAssetData, BufferAsset>,
}
