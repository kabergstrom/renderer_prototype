use super::load_queue::LoadRequest;
use super::BufferAssetData;
use super::ImageAssetData;
use super::{BufferAsset, ImageAsset};
use crate::{buffer_upload, image_upload, DecodedImage};
use ash::prelude::VkResult;
use ash::vk;
use atelier_assets::loader::{storage::AssetLoadOp, LoadHandle};
use crossbeam_channel::{Receiver, Sender};
use rafx_api_vulkan::{
    VkBuffer, VkDeviceContext, VkImage, VkTransferUpload, VkTransferUploadState, VkUploadError,
};
use std::mem::ManuallyDrop;

//
// Ghetto futures - UploadOp is used to signal completion and UploadOpAwaiter is used to check the result
//
pub enum UploadOpResult<ResourceT, AssetT> {
    UploadError(LoadHandle),
    UploadComplete(AssetLoadOp, Sender<AssetT>, ResourceT),
    UploadDrop(LoadHandle),
}

pub struct UploadOp<ResourceT, AssetT> {
    load_handle: LoadHandle,
    asset_sender: Option<Sender<AssetT>>, // This sends back to the asset storage, we just pass it along
    sender: Option<Sender<UploadOpResult<ResourceT, AssetT>>>, // This sends back to the resource manager to finalize the load
}

impl<ResourceT, AssetT> UploadOp<ResourceT, AssetT> {
    pub fn new(
        load_handle: LoadHandle,
        asset_sender: Sender<AssetT>,
        sender: Sender<UploadOpResult<ResourceT, AssetT>>,
    ) -> Self {
        Self {
            load_handle,
            asset_sender: Some(asset_sender),
            sender: Some(sender),
        }
    }

    pub fn complete(
        mut self,
        image: ResourceT,
        load_op: AssetLoadOp,
    ) {
        let _ = self
            .sender
            .as_ref()
            .unwrap()
            .send(UploadOpResult::UploadComplete(
                load_op,
                self.asset_sender.take().unwrap(),
                image,
            ));
        self.sender = None;
    }

    pub fn error(mut self) {
        let _ = self
            .sender
            .as_ref()
            .unwrap()
            .send(UploadOpResult::UploadError(self.load_handle));
        self.sender = None;
    }
}

impl<ResourceT, AssetT> Drop for UploadOp<ResourceT, AssetT> {
    fn drop(&mut self) {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(UploadOpResult::UploadDrop(self.load_handle));
        }
    }
}

pub type ImageUploadOpResult = UploadOpResult<VkImage, ImageAsset>;
pub type ImageUploadOp = UploadOp<VkImage, ImageAsset>;

pub type BufferUploadOpResult = UploadOpResult<VkBuffer, BufferAsset>;
pub type BufferUploadOp = UploadOp<VkBuffer, BufferAsset>;

//
// Represents a single request inserted into the upload queue that hasn't started yet
//
//TODO: Make a helper object that carries an Arc<Receiver> that can be called
pub struct PendingImageUpload {
    pub load_op: AssetLoadOp,
    pub upload_op: ImageUploadOp,
    pub texture: DecodedImage,
}

pub struct PendingBufferUpload {
    pub load_op: AssetLoadOp,
    pub upload_op: BufferUploadOp,
    pub data: Vec<u8>,
}

//
// Represents a single request that the upload queue has started
//
struct InFlightImageUpload {
    load_op: AssetLoadOp,
    upload_op: ImageUploadOp,
    image: ManuallyDrop<VkImage>,
}

pub struct InFlightBufferUpload {
    load_op: AssetLoadOp,
    upload_op: BufferUploadOp,
    buffer: ManuallyDrop<VkBuffer>,
}

//
// Represents a batch of requests that has been started, contains multiple InFlightImageUpload and
// InFlightBufferUploads
//

// The result from polling a single upload (which may contain multiple images in it)
pub enum InProgressUploadPollResult {
    Pending,
    Complete,
    Error,
    Destroyed,
}

// This is an inner of InProgressImageUpload - it is wrapped in a Option to avoid borrowing issues
// when polling by allowing us to temporarily take ownership of it and then put it back
struct InProgressUploadInner {
    image_uploads: Vec<InFlightImageUpload>,
    buffer_uploads: Vec<InFlightBufferUpload>,
    upload: VkTransferUpload,
}

struct InProgressUploadDebugInfo {
    upload_id: usize,
    start_time: std::time::Instant,
    size: u64,
    image_count: usize,
    buffer_count: usize,
}

// A single upload which may contain multiple images
struct InProgressUpload {
    // Only valid if the upload is actually in progress
    inner: Option<InProgressUploadInner>,
    debug_info: InProgressUploadDebugInfo,
}

impl InProgressUpload {
    pub fn new(
        image_uploads: Vec<InFlightImageUpload>,
        buffer_uploads: Vec<InFlightBufferUpload>,
        upload: VkTransferUpload,
        debug_info: InProgressUploadDebugInfo,
    ) -> Self {
        let inner = InProgressUploadInner {
            image_uploads,
            buffer_uploads,
            upload,
        };

        InProgressUpload {
            inner: Some(inner),
            debug_info,
        }
    }

    // The main state machine for an upload:
    // - Submits on the transfer queue and waits
    // - Submits on the graphics queue and waits
    //
    // Calls load_op.complete() or load_op.error() as appropriate
    pub fn poll_load(
        &mut self,
        device_context: &VkDeviceContext,
    ) -> InProgressUploadPollResult {
        loop {
            if let Some(mut inner) = self.take_inner() {
                match inner.upload.state() {
                    Ok(state) => match state {
                        VkTransferUploadState::Writable => {
                            //log::trace!("VkTransferUploadState::Writable");
                            inner
                                .upload
                                .submit_transfer(&device_context.queues().transfer_queue)
                                .unwrap();
                            self.inner = Some(inner);
                        }
                        VkTransferUploadState::SentToTransferQueue => {
                            //log::trace!("VkTransferUploadState::SentToTransferQueue");
                            self.inner = Some(inner);
                            break InProgressUploadPollResult::Pending;
                        }
                        VkTransferUploadState::PendingSubmitDstQueue => {
                            //log::trace!("VkTransferUploadState::PendingSubmitDstQueue");
                            inner
                                .upload
                                .submit_dst(&device_context.queues().graphics_queue)
                                .unwrap();
                            self.inner = Some(inner);
                        }
                        VkTransferUploadState::SentToDstQueue => {
                            //log::trace!("VkTransferUploadState::SentToDstQueue");
                            self.inner = Some(inner);
                            break InProgressUploadPollResult::Pending;
                        }
                        VkTransferUploadState::Complete => {
                            //log::trace!("VkTransferUploadState::Complete");
                            for mut upload in inner.image_uploads {
                                let image = unsafe { ManuallyDrop::take(&mut upload.image) };
                                upload.upload_op.complete(image, upload.load_op);
                            }

                            for mut upload in inner.buffer_uploads {
                                let buffer = unsafe { ManuallyDrop::take(&mut upload.buffer) };
                                upload.upload_op.complete(buffer, upload.load_op);
                            }

                            break InProgressUploadPollResult::Complete;
                        }
                    },
                    Err(err) => {
                        for mut upload in inner.image_uploads {
                            upload.load_op.error(err);
                            upload.upload_op.error();
                            unsafe {
                                ManuallyDrop::drop(&mut upload.image);
                            }
                        }

                        for mut upload in inner.buffer_uploads {
                            upload.load_op.error(err);
                            upload.upload_op.error();
                            unsafe {
                                ManuallyDrop::drop(&mut upload.buffer);
                            }
                        }

                        break InProgressUploadPollResult::Error;
                    }
                }
            } else {
                break InProgressUploadPollResult::Destroyed;
            }
        }
    }

    // Allows taking ownership of the inner object
    fn take_inner(&mut self) -> Option<InProgressUploadInner> {
        let mut inner = None;
        std::mem::swap(&mut self.inner, &mut inner);
        inner
    }
}

impl Drop for InProgressUpload {
    fn drop(&mut self) {
        if let Some(mut inner) = self.take_inner() {
            for image in &mut inner.image_uploads {
                unsafe {
                    ManuallyDrop::drop(&mut image.image);
                }
            }

            for buffer in &mut inner.buffer_uploads {
                unsafe {
                    ManuallyDrop::drop(&mut buffer.buffer);
                }
            }
        }
    }
}

pub struct UploadQueueConfig {
    pub max_bytes_per_upload: usize,
    pub max_concurrent_uploads: usize,
    pub max_new_uploads_in_single_frame: usize,
}

//
// Receives sets of images that need to be uploaded and kicks off the upload. Responsible for
// batching image updates together into uploads
//
pub struct UploadQueue {
    device_context: VkDeviceContext,
    config: UploadQueueConfig,

    // For enqueueing images to upload
    pending_image_tx: Sender<PendingImageUpload>,
    pending_image_rx: Receiver<PendingImageUpload>,

    // If we fail to upload due to size limitation, keep the failed upload here to retry later
    next_image_upload: Option<PendingImageUpload>,

    // For enqueueing buffers to upload
    pending_buffer_tx: Sender<PendingBufferUpload>,
    pending_buffer_rx: Receiver<PendingBufferUpload>,

    // If we fail to upload due to size limitation, keep the failed upload here to retry later
    next_buffer_upload: Option<PendingBufferUpload>,

    // These are uploads that are currently in progress
    uploads_in_progress: Vec<InProgressUpload>,

    next_upload_id: usize,
}

impl UploadQueue {
    pub fn new(
        device_context: &VkDeviceContext,
        config: UploadQueueConfig,
    ) -> Self {
        let (pending_image_tx, pending_image_rx) = crossbeam_channel::unbounded();
        let (pending_buffer_tx, pending_buffer_rx) = crossbeam_channel::unbounded();

        UploadQueue {
            device_context: device_context.clone(),
            config,
            pending_image_tx,
            pending_image_rx,
            next_image_upload: None,
            pending_buffer_tx,
            pending_buffer_rx,
            next_buffer_upload: None,
            uploads_in_progress: Default::default(),
            next_upload_id: 1,
        }
    }

    pub fn pending_image_tx(&self) -> &Sender<PendingImageUpload> {
        &self.pending_image_tx
    }

    pub fn pending_buffer_tx(&self) -> &Sender<PendingBufferUpload> {
        &self.pending_buffer_tx
    }

    // Ok(None) = upload enqueue
    // Ok(Some) = upload not enqueued because there was not enough room
    // Err = Vulkan error
    fn try_enqueue_image_upload(
        &mut self,
        upload: &mut VkTransferUpload,
        pending_image: PendingImageUpload,
        in_flight_uploads: &mut Vec<InFlightImageUpload>,
    ) -> VkResult<Option<PendingImageUpload>> {
        let result = image_upload::enqueue_load_image(
            &self.device_context,
            upload,
            self.device_context
                .queue_family_indices()
                .transfer_queue_family_index,
            self.device_context
                .queue_family_indices()
                .graphics_queue_family_index,
            &pending_image.texture,
        );

        match result {
            Ok(image) => {
                in_flight_uploads.push(InFlightImageUpload {
                    image,
                    load_op: pending_image.load_op,
                    upload_op: pending_image.upload_op,
                });
                Ok(None)
            }
            Err(VkUploadError::VkError(e)) => Err(e),
            Err(VkUploadError::BufferFull) => Ok(Some(pending_image)),
        }
    }

    fn start_new_image_uploads(
        &mut self,
        upload: &mut VkTransferUpload,
    ) -> VkResult<Vec<InFlightImageUpload>> {
        let mut in_flight_uploads = vec![];

        // If we had a pending image upload from before, try to upload it now
        self.next_image_upload = if let Some(next_image_upload) = self.next_image_upload.take() {
            self.try_enqueue_image_upload(upload, next_image_upload, &mut in_flight_uploads)?
        } else {
            None
        };

        // The first image we tried to upload failed. Log an error since we aren't making forward progress
        if let Some(next_image_upload) = &self.next_image_upload {
            log::error!(
                "Image of {} bytes has repeatedly exceeded the available room in the upload buffer. ({} of {} bytes free)",
                next_image_upload.texture.data.len(),
                upload.bytes_free(),
                upload.buffer_size()
            );
            return Ok(vec![]);
        }

        let rx = self.pending_image_rx.clone();
        for pending_upload in rx.try_iter() {
            self.next_image_upload =
                self.try_enqueue_image_upload(upload, pending_upload, &mut in_flight_uploads)?;

            if let Some(next_image_upload) = &self.next_image_upload {
                log::debug!(
                    "Image of {} bytes exceeds the available room in the upload buffer. ({} of {} bytes free)",
                    next_image_upload.texture.data.len(),
                    upload.bytes_free(),
                    upload.buffer_size(),
                );
                break;
            }
        }

        Ok(in_flight_uploads)
    }

    // Ok(None) = upload enqueue
    // Ok(Some) = upload not enqueued because there was not enough room
    // Err = Vulkan error
    fn try_enqueue_buffer_upload(
        &mut self,
        upload: &mut VkTransferUpload,
        pending_buffer: PendingBufferUpload,
        in_flight_uploads: &mut Vec<InFlightBufferUpload>,
    ) -> VkResult<Option<PendingBufferUpload>> {
        let result = buffer_upload::enqueue_load_buffer(
            &self.device_context,
            upload,
            self.device_context
                .queue_family_indices()
                .transfer_queue_family_index,
            self.device_context
                .queue_family_indices()
                .graphics_queue_family_index,
            &pending_buffer.data,
        );

        match result {
            Ok(buffer) => {
                in_flight_uploads.push(InFlightBufferUpload {
                    buffer,
                    load_op: pending_buffer.load_op,
                    upload_op: pending_buffer.upload_op,
                });
                Ok(None)
            }
            Err(VkUploadError::VkError(e)) => Err(e),
            Err(VkUploadError::BufferFull) => Ok(Some(pending_buffer)),
        }
    }

    fn start_new_buffer_uploads(
        &mut self,
        upload: &mut VkTransferUpload,
    ) -> VkResult<Vec<InFlightBufferUpload>> {
        let mut in_flight_uploads = vec![];

        // If we had a pending image upload from before, try to upload it now
        self.next_buffer_upload = if let Some(next_buffer_upload) = self.next_buffer_upload.take() {
            self.try_enqueue_buffer_upload(upload, next_buffer_upload, &mut in_flight_uploads)?
        } else {
            None
        };

        // The first buffer we tried to upload failed. Log an error since we aren't making forward progress
        if let Some(next_buffer_upload) = &self.next_buffer_upload {
            log::error!(
                "Buffer of {} bytes has repeatedly exceeded the available room in the upload buffer. ({} of {} bytes free)",
                next_buffer_upload.data.len(),
                upload.bytes_free(),
                upload.buffer_size()
            );
            return Ok(vec![]);
        }

        let rx = self.pending_buffer_rx.clone();
        for pending_upload in rx.try_iter() {
            self.next_buffer_upload =
                self.try_enqueue_buffer_upload(upload, pending_upload, &mut in_flight_uploads)?;

            if let Some(next_buffer_upload) = &self.next_buffer_upload {
                log::debug!(
                    "Buffer of {} bytes exceeds the available room in the upload buffer. ({} of {} bytes free)",
                    next_buffer_upload.data.len(),
                    upload.bytes_free(),
                    upload.buffer_size(),
                );
                break;
            }
        }

        Ok(in_flight_uploads)
    }

    fn start_new_uploads(&mut self) -> VkResult<()> {
        for _ in 0..self.config.max_new_uploads_in_single_frame {
            if self.pending_image_rx.is_empty()
                && self.next_image_upload.is_none()
                && self.pending_buffer_rx.is_empty()
                && self.next_buffer_upload.is_none()
            {
                return Ok(());
            }

            if self.uploads_in_progress.len() >= self.config.max_concurrent_uploads {
                log::trace!(
                    "Max number of uploads already in progress. Waiting to start a new one"
                );
                return Ok(());
            }

            if !self.start_new_upload()? {
                return Ok(());
            }
        }

        Ok(())
    }

    fn start_new_upload(&mut self) -> VkResult<bool> {
        let mut upload = VkTransferUpload::new(
            &self.device_context,
            self.device_context
                .queue_family_indices()
                .transfer_queue_family_index,
            self.device_context
                .queue_family_indices()
                .graphics_queue_family_index,
            self.config.max_bytes_per_upload as u64,
        )?;

        let in_flight_image_uploads = self.start_new_image_uploads(&mut upload)?;
        let in_flight_buffer_uploads = self.start_new_buffer_uploads(&mut upload)?;

        if !in_flight_image_uploads.is_empty() || !in_flight_buffer_uploads.is_empty() {
            let upload_id = self.next_upload_id;
            self.next_upload_id += 1;

            log::debug!(
                "Submitting {} byte upload with {} images and {} buffers, UploadId = {}",
                upload.bytes_written(),
                in_flight_image_uploads.len(),
                in_flight_buffer_uploads.len(),
                upload_id
            );

            upload.submit_transfer(&self.device_context.queues().transfer_queue)?;

            let debug_info = InProgressUploadDebugInfo {
                upload_id,
                buffer_count: in_flight_buffer_uploads.len(),
                image_count: in_flight_image_uploads.len(),
                size: upload.bytes_written(),
                start_time: std::time::Instant::now(),
            };

            self.uploads_in_progress.push(InProgressUpload::new(
                in_flight_image_uploads,
                in_flight_buffer_uploads,
                upload,
                debug_info,
            ));

            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn update_existing_uploads(&mut self) {
        // iterate backwards so we can use swap_remove
        for i in (0..self.uploads_in_progress.len()).rev() {
            let result = self.uploads_in_progress[i].poll_load(&self.device_context);
            match result {
                InProgressUploadPollResult::Pending => {
                    // do nothing
                }
                InProgressUploadPollResult::Complete => {
                    //load_op.complete() is called by poll_load

                    let debug_info = &self.uploads_in_progress[i].debug_info;
                    log::debug!(
                        "Completed {} byte upload with {} images and {} buffers in {} ms, UploadId = {}",
                        debug_info.size,
                        debug_info.image_count,
                        debug_info.buffer_count,
                        (std::time::Instant::now() - debug_info.start_time).as_secs_f32(),
                        debug_info.upload_id
                    );

                    self.uploads_in_progress.swap_remove(i);
                }
                InProgressUploadPollResult::Error => {
                    //load_op.error() is called by poll_load

                    let debug_info = &self.uploads_in_progress[i].debug_info;
                    log::error!(
                        "Failed {} byte upload with {} images and {} buffers in {} ms, UploadId = {}",
                        debug_info.size,
                        debug_info.image_count,
                        debug_info.buffer_count,
                        (std::time::Instant::now() - debug_info.start_time).as_secs_f32(),
                        debug_info.upload_id
                    );

                    self.uploads_in_progress.swap_remove(i);
                }
                InProgressUploadPollResult::Destroyed => {
                    // not expected - this only occurs if polling the upload when it is already in a complete or error state
                    unreachable!();
                }
            }
        }
    }

    pub fn update(&mut self) -> VkResult<()> {
        self.start_new_uploads()?;
        self.update_existing_uploads();
        Ok(())
    }
}

pub struct UploadManager {
    upload_queue: UploadQueue,

    pub image_upload_result_tx: Sender<ImageUploadOpResult>,
    pub image_upload_result_rx: Receiver<ImageUploadOpResult>,

    pub buffer_upload_result_tx: Sender<BufferUploadOpResult>,
    pub buffer_upload_result_rx: Receiver<BufferUploadOpResult>,
}

impl UploadManager {
    pub fn new(
        device_context: &VkDeviceContext,
        upload_queue_config: UploadQueueConfig,
    ) -> Self {
        let (image_upload_result_tx, image_upload_result_rx) = crossbeam_channel::unbounded();
        let (buffer_upload_result_tx, buffer_upload_result_rx) = crossbeam_channel::unbounded();

        UploadManager {
            upload_queue: UploadQueue::new(device_context, upload_queue_config),
            image_upload_result_rx,
            image_upload_result_tx,
            buffer_upload_result_rx,
            buffer_upload_result_tx,
        }
    }

    pub fn update(&mut self) -> VkResult<()> {
        self.upload_queue.update()
    }

    pub fn upload_image(
        &self,
        request: LoadRequest<ImageAssetData, ImageAsset>,
    ) -> VkResult<()> {
        let mips = DecodedImage::default_mip_settings_for_image_size(
            request.asset.width,
            request.asset.height,
        );

        let color_space = request.asset.color_space.into();

        let decoded_image = DecodedImage {
            width: request.asset.width,
            height: request.asset.height,
            mips,
            color_space,
            data: request.asset.data,
        };

        self.upload_queue
            .pending_image_tx()
            .send(PendingImageUpload {
                load_op: request.load_op,
                upload_op: UploadOp::new(
                    request.load_handle,
                    request.result_tx,
                    self.image_upload_result_tx.clone(),
                ),
                texture: decoded_image,
            })
            .map_err(|_err| {
                log::error!("Could not enqueue image upload");
                vk::Result::ERROR_UNKNOWN
            })
    }

    pub fn upload_buffer(
        &self,
        request: LoadRequest<BufferAssetData, BufferAsset>,
    ) -> VkResult<()> {
        self.upload_queue
            .pending_buffer_tx()
            .send(PendingBufferUpload {
                load_op: request.load_op,
                upload_op: UploadOp::new(
                    request.load_handle,
                    request.result_tx,
                    self.buffer_upload_result_tx.clone(),
                ),
                data: request.asset.data,
            })
            .map_err(|_err| {
                log::error!("Could not enqueue buffer upload");
                vk::Result::ERROR_UNKNOWN
            })
    }
}
