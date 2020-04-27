use std::io;
use ash::vk;
use ash::prelude::VkResult;
use ash::version::DeviceV1_0;

/// Loads a shader into a buffer
pub use ash::util::read_spv;

// Don't actually do this in shipping code
/*
/// Fires off a command buffer and then waits for the device to be idle
pub fn submit_single_use_command_buffer<F: Fn(vk::CommandBuffer)>(
    logical_device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    f: F,
) -> VkResult<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let command_buffer = unsafe {
        let command_buffer = logical_device.allocate_command_buffers(&alloc_info)?[0];

        logical_device.begin_command_buffer(command_buffer, &begin_info)?;

        f(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;

        command_buffer
    };

    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);

    unsafe {
        logical_device.queue_submit(queue, &[submit_info.build()], vk::Fence::null())?;
        logical_device.device_wait_idle()?;

        logical_device.free_command_buffers(command_pool, &command_buffers);
    }

    Ok(())
}
*/