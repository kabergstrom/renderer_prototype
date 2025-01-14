use ash::prelude::VkResult;
use ash::vk;

use crate::{VkBuffer, VkDeviceContext};
use ash::version::DeviceV1_0;
use std::mem::ManuallyDrop;

use std::sync::{Arc, Mutex};

// Based on UploadHeap in cauldron
// (https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron/blob/5acc12602c55e469cc1f9181967dbcb122f8e6c7/src/VK/base/UploadHeap.h)

#[derive(Debug)]
pub enum VkUploadError {
    BufferFull,
    VkError(vk::Result),
}

impl core::fmt::Display for VkUploadError {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        match *self {
            VkUploadError::BufferFull => write!(fmt, "UploadBufferFull"),
            VkUploadError::VkError(ref e) => e.fmt(fmt),
        }
    }
}

impl From<vk::Result> for VkUploadError {
    fn from(result: vk::Result) -> Self {
        VkUploadError::VkError(result)
    }
}

impl Into<vk::Result> for VkUploadError {
    fn into(self) -> vk::Result {
        match self {
            VkUploadError::BufferFull => vk::Result::ERROR_UNKNOWN,
            VkUploadError::VkError(e) => e,
        }
    }
}

#[derive(PartialEq)]
pub enum VkUploadState {
    /// The upload is not submitted yet and data may be appended to it
    Writable,

    /// The buffer has been sent to the GPU and is no longer writable
    SentToGpu,

    /// The upload is finished and the resources may be used
    Complete,
}

/// This is a convenience class that allows accumulating writes into a staging buffer and commands
/// to execute on the staging buffer. This allows for batching uploading resources.
pub struct VkUpload {
    device_context: VkDeviceContext,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    buffer: ManuallyDrop<VkBuffer>,

    writable: bool,
    fence: vk::Fence,

    buffer_begin: *mut u8,
    buffer_end: *mut u8,
    buffer_write_pointer: *mut u8,
}

unsafe impl Send for VkUpload {}
unsafe impl Sync for VkUpload {}

impl VkUpload {
    pub fn new(
        device_context: &VkDeviceContext,
        queue_family_index: u32,
        buffer_size: u64,
    ) -> VkResult<Self> {
        //
        // Command Buffers
        //
        let command_pool = Self::create_command_pool(device_context.device(), queue_family_index)?;

        let command_buffer = Self::create_command_buffer(device_context.device(), &command_pool)?;
        Self::begin_command_buffer(device_context.device(), command_buffer)?;

        let buffer = ManuallyDrop::new(VkBuffer::new(
            &device_context,
            vk_mem::MemoryUsage::CpuOnly,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            buffer_size,
        )?);

        let (buffer_begin, buffer_end, buffer_write_pointer) = unsafe {
            //TODO: Better way of handling allocator errors
            let buffer_begin = device_context
                .allocator()
                .map_memory(&buffer.allocation())
                .map_err(|_| vk::Result::ERROR_MEMORY_MAP_FAILED)?
                as *mut u8;

            let buffer_end = buffer_begin.add(buffer.size() as usize);
            let buffer_write_pointer = buffer_begin;

            (buffer_begin, buffer_end, buffer_write_pointer)
        };

        let fence = Self::create_fence(device_context.device())?;

        let upload = VkUpload {
            device_context: device_context.clone(),
            command_pool,
            command_buffer,
            buffer,
            fence,
            writable: true,
            buffer_begin,
            buffer_end,
            buffer_write_pointer,
        };

        Ok(upload)
    }

    fn create_command_pool(
        logical_device: &ash::Device,
        queue_family_index: u32,
    ) -> VkResult<vk::CommandPool> {
        log::trace!(
            "Creating command pool with queue family index {}",
            queue_family_index
        );
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(queue_family_index);

        unsafe { logical_device.create_command_pool(&pool_create_info, None) }
    }

    fn create_command_buffer(
        logical_device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> VkResult<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe { Ok(logical_device.allocate_command_buffers(&command_buffer_allocate_info)?[0]) }
    }

    fn create_fence(logical_device: &ash::Device) -> VkResult<vk::Fence> {
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::empty());

        unsafe { Ok(logical_device.create_fence(&fence_create_info, None)?) }
    }

    fn begin_command_buffer(
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> VkResult<()> {
        let command_buffer_begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::empty());
        unsafe { logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
    }

    pub fn has_space_available(
        &self,
        bytes_to_write: usize,
        required_alignment: usize,
        number_of_writes: usize,
    ) -> bool {
        let mut write_end_ptr = self.buffer_write_pointer as usize;

        for _ in 0..number_of_writes {
            // Align the current write pointer
            let write_begin_ptr = ((write_end_ptr + required_alignment - 1) / required_alignment)
                * required_alignment;
            write_end_ptr = write_begin_ptr + bytes_to_write;
        }

        // See if we would walk past the end of the buffer
        write_end_ptr <= self.buffer_end as usize
    }

    pub fn push(
        &mut self,
        data: &[u8],
        required_alignment: usize,
    ) -> Result<vk::DeviceSize, VkUploadError> {
        log::trace!("Pushing {} bytes into upload", data.len());

        if self.writable {
            unsafe {
                // Figure out the span of memory we will write over
                let write_begin_ptr = (((self.buffer_write_pointer as usize + required_alignment
                    - 1)
                    / required_alignment)
                    * required_alignment) as *mut u8; // as const *u8;
                let write_end_ptr = write_begin_ptr.add(data.len());

                // If the span walks past the end of the buffer, fail
                if write_end_ptr > self.buffer_end {
                    Err(VkUploadError::BufferFull)?;
                }

                std::ptr::copy_nonoverlapping(data.as_ptr(), write_begin_ptr, data.len());
                self.buffer_write_pointer = write_end_ptr;

                Ok(write_begin_ptr as vk::DeviceSize - self.buffer_begin as vk::DeviceSize)
            }
        } else {
            log::error!("Upload buffer is not writable");
            Err(vk::Result::ERROR_UNKNOWN)?
        }
    }

    pub fn buffer_size(&self) -> u64 {
        self.buffer_end as u64 - self.buffer_begin as u64
    }

    pub fn bytes_written(&self) -> u64 {
        self.buffer_write_pointer as u64 - self.buffer_begin as u64
    }

    pub fn bytes_free(&self) -> u64 {
        self.buffer_end as u64 - self.buffer_write_pointer as u64
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub fn staging_buffer(&self) -> &VkBuffer {
        &self.buffer
    }

    pub fn submit(
        &mut self,
        queue: &Arc<Mutex<vk::Queue>>,
    ) -> VkResult<()> {
        if self.writable {
            unsafe {
                self.device_context
                    .device()
                    .end_command_buffer(self.command_buffer)?;
            }

            let submit = vk::SubmitInfo::builder()
                .command_buffers(&[self.command_buffer])
                .build();

            unsafe {
                {
                    let q = queue.lock().unwrap();
                    self.device_context
                        .device()
                        .queue_submit(*q, &[submit], self.fence)?;
                }
                self.writable = false;
            }
        }

        Ok(())
    }

    pub fn state(&self) -> VkResult<VkUploadState> {
        let state = if self.writable {
            VkUploadState::Writable
        } else {
            let submit_complete =
                unsafe { self.device_context.device().get_fence_status(self.fence)? };
            if submit_complete {
                VkUploadState::Complete
            } else {
                VkUploadState::SentToGpu
            }
        };

        Ok(state)
    }

    fn wait_for_idle(&self) -> VkResult<()> {
        unsafe {
            if !self.writable {
                self.device_context
                    .device()
                    .wait_for_fences(&[self.fence], true, core::u64::MAX)
            } else {
                Ok(())
            }
        }
    }
}

impl Drop for VkUpload {
    fn drop(&mut self) {
        log::trace!("destroying VkUpload");

        // If the transfer is in flight, wait for it to complete
        self.wait_for_idle().unwrap();

        unsafe {
            self.device_context
                .allocator()
                .unmap_memory(&self.buffer.allocation())
                .unwrap();
            ManuallyDrop::drop(&mut self.buffer);
            self.device_context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.device_context.device().destroy_fence(self.fence, None);
        }

        log::trace!("destroyed VkUpload");
    }
}

#[derive(PartialEq)]
pub enum VkTransferUploadState {
    /// The upload is not submitted yet and data may be appended to it
    Writable,

    /// The buffer has been sent to the GPU's transfer queue and is no longer writable
    SentToTransferQueue,

    /// The submit to the transfer queue finished. We are ready to submit to the graphics queue
    /// but we wait here until called explicitly because submitting to a queue is not thread-safe.
    /// Additionally, it's likely we will want to batch this submit with other command buffers going
    /// to the same queue
    PendingSubmitDstQueue,

    /// The buffer has been sent to the GPU's graphics queue but has not finished
    SentToDstQueue,

    /// The submit has finished on both queues and the uploaded resources are ready for use
    Complete,
}

/// A state machine and associated buffers/synchronization primitives to simplify uploading resources
/// to the GPU via a transfer queue, and then submitting a memory barrier to the graphics queue
pub struct VkTransferUpload {
    device_context: VkDeviceContext,
    upload: VkUpload,

    dst_command_pool: vk::CommandPool,
    dst_command_buffer: vk::CommandBuffer,

    dst_fence: vk::Fence,
    sent_to_dst_queue: bool,
}

impl VkTransferUpload {
    pub fn new(
        device_context: &VkDeviceContext,
        transfer_queue_family_index: u32,
        dst_queue_family_index: u32,
        size: u64,
    ) -> VkResult<Self> {
        //
        // Command Buffers
        //
        let dst_command_pool =
            Self::create_command_pool(device_context.device(), dst_queue_family_index)?;

        let dst_command_buffer =
            Self::create_command_buffer(device_context.device(), &dst_command_pool)?;
        Self::begin_command_buffer(device_context.device(), dst_command_buffer)?;

        let upload = VkUpload::new(device_context, transfer_queue_family_index, size)?;

        let dst_fence = Self::create_fence(device_context.device())?;

        Ok(VkTransferUpload {
            device_context: device_context.clone(),
            upload,
            dst_command_pool,
            dst_command_buffer,
            dst_fence,
            sent_to_dst_queue: false,
        })
    }

    pub fn has_space_available(
        &self,
        bytes_to_write: usize,
        required_alignment: usize,
        number_of_writes: usize,
    ) -> bool {
        self.upload
            .has_space_available(bytes_to_write, required_alignment, number_of_writes)
    }

    pub fn push(
        &mut self,
        data: &[u8],
        required_alignment: usize,
    ) -> Result<vk::DeviceSize, VkUploadError> {
        self.upload.push(data, required_alignment)
    }

    pub fn buffer_size(&self) -> u64 {
        self.upload.buffer_size()
    }

    pub fn bytes_written(&self) -> u64 {
        self.upload.bytes_written()
    }

    pub fn bytes_free(&self) -> u64 {
        self.upload.bytes_free()
    }

    pub fn staging_buffer(&self) -> &VkBuffer {
        &self.upload.staging_buffer()
    }

    pub fn transfer_command_buffer(&self) -> vk::CommandBuffer {
        self.upload.command_buffer()
    }

    pub fn dst_command_buffer(&self) -> vk::CommandBuffer {
        self.dst_command_buffer
    }

    fn create_command_pool(
        logical_device: &ash::Device,
        queue_family_index: u32,
    ) -> VkResult<vk::CommandPool> {
        log::trace!(
            "Creating command pool with queue family index {}",
            queue_family_index
        );
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(queue_family_index);

        unsafe { logical_device.create_command_pool(&pool_create_info, None) }
    }

    fn create_command_buffer(
        logical_device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> VkResult<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe { Ok(logical_device.allocate_command_buffers(&command_buffer_allocate_info)?[0]) }
    }

    fn create_fence(logical_device: &ash::Device) -> VkResult<vk::Fence> {
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::empty());

        unsafe { Ok(logical_device.create_fence(&fence_create_info, None)?) }
    }

    fn begin_command_buffer(
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> VkResult<()> {
        let command_buffer_begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::empty());
        unsafe { logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
    }

    pub fn submit_transfer(
        &mut self,
        transfer_queue: &Arc<Mutex<vk::Queue>>,
    ) -> VkResult<()> {
        self.upload.submit(transfer_queue)
    }

    pub fn submit_dst(
        &mut self,
        dst_queue: &Arc<Mutex<vk::Queue>>,
    ) -> VkResult<()> {
        if self.state()? == VkTransferUploadState::PendingSubmitDstQueue {
            unsafe {
                self.device_context
                    .device()
                    .end_command_buffer(self.dst_command_buffer)?;
            }

            let submit = vk::SubmitInfo::builder()
                .command_buffers(&[self.dst_command_buffer])
                .build();

            unsafe {
                let queue = dst_queue.lock().unwrap();
                self.device_context
                    .device()
                    .queue_submit(*queue, &[submit], self.dst_fence)?;
                self.sent_to_dst_queue = true;
            }
        }

        Ok(())
    }

    pub fn state(&self) -> VkResult<VkTransferUploadState> {
        let state = if self.sent_to_dst_queue {
            let submit_complete = unsafe {
                self.device_context
                    .device()
                    .get_fence_status(self.dst_fence)?
            };
            if submit_complete {
                VkTransferUploadState::Complete
            } else {
                VkTransferUploadState::SentToDstQueue
            }
        } else {
            match self.upload.state()? {
                VkUploadState::Writable => VkTransferUploadState::Writable,
                VkUploadState::SentToGpu => VkTransferUploadState::SentToTransferQueue,
                VkUploadState::Complete => VkTransferUploadState::PendingSubmitDstQueue,
            }
        };

        Ok(state)
    }

    fn wait_for_idle(&self) -> VkResult<()> {
        unsafe {
            if self.sent_to_dst_queue {
                self.device_context.device().wait_for_fences(
                    &[self.dst_fence],
                    true,
                    core::u64::MAX,
                )
            } else {
                Ok(())
            }
        }
    }

    pub fn block_until_upload_complete(
        &mut self,
        transfer_queue: &Arc<Mutex<vk::Queue>>,
        dst_queue: &Arc<Mutex<vk::Queue>>,
    ) -> VkResult<()> {
        self.submit_transfer(transfer_queue)?;
        loop {
            if self.state()? == VkTransferUploadState::PendingSubmitDstQueue {
                break;
            }
        }

        self.submit_dst(dst_queue)?;
        loop {
            if self.state()? == VkTransferUploadState::Complete {
                break;
            }
        }

        Ok(())
    }
}

impl Drop for VkTransferUpload {
    fn drop(&mut self) {
        log::trace!("destroying VkUpload");

        // If the transfer is in flight, wait for it to complete
        self.upload.wait_for_idle().unwrap();
        self.wait_for_idle().unwrap();

        unsafe {
            self.device_context
                .device()
                .destroy_command_pool(self.dst_command_pool, None);
            self.device_context
                .device()
                .destroy_fence(self.dst_fence, None);
        }

        log::trace!("destroyed VkUpload");
    }
}
