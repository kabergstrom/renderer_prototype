use crate::asset_storage::{DynAssetLoader, UpdateAssetResult};
use atelier_assets::loader::handle::RefOp;
use atelier_assets::loader::handle::SerdeContext;
use atelier_assets::loader::storage::AssetLoadOp;
use atelier_assets::loader::storage::LoaderInfoProvider;
use atelier_assets::loader::LoadHandle;
use crossbeam_channel::Sender;
use rafx::assets::GenericLoader;
use rafx::assets::ResourceLoader;
use std::error::Error;
use type_uuid::TypeUuid;

pub struct ResourceAssetLoader<AssetDataT, AssetT>(pub GenericLoader<AssetDataT, AssetT>)
where
    AssetDataT: for<'a> serde::Deserialize<'a> + 'static + Send,
    AssetT: TypeUuid + 'static + Send;

impl<AssetDataT, AssetT> DynAssetLoader<AssetT> for ResourceAssetLoader<AssetDataT, AssetT>
where
    AssetDataT: for<'a> serde::Deserialize<'a> + 'static + Send,
    AssetT: TypeUuid + 'static + Send,
{
    fn update_asset(
        &mut self,
        refop_sender: &Sender<RefOp>,
        loader_info: &dyn LoaderInfoProvider,
        data: &[u8],
        load_handle: LoadHandle,
        load_op: AssetLoadOp,
        _version: u32,
    ) -> Result<UpdateAssetResult<AssetT>, Box<dyn Error + Send>> {
        // To enable automatic serde of Handle, we need to set up a SerdeContext with a RefOp sender
        let asset = futures_lite::future::block_on(SerdeContext::with(
            loader_info,
            refop_sender.clone(),
            async {
                bincode::deserialize::<AssetDataT>(data)
                    // Coerce into boxed error
                    .map_err(|x| -> Box<dyn Error + Send + 'static> { Box::new(x) })
            },
        ))?;

        let result = self.0.update_asset(load_handle, load_op, asset);
        Ok(UpdateAssetResult::AsyncResult(result.result_rx))
    }

    fn commit_asset_version(
        &mut self,
        handle: LoadHandle,
        _version: u32,
    ) {
        self.0.commit_asset_version(handle);
    }

    fn free(
        &mut self,
        handle: LoadHandle,
    ) {
        self.0.free(handle);
    }
}
