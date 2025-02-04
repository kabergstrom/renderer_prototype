use ash::vk;
use std::mem::ManuallyDrop;

use ash::version::DeviceV1_0;

use rafx_api_vulkan::{VkDeviceContext, VkTransferUpload, VkUploadError};

use rafx_api_vulkan::VkImage;
use std::sync::{Arc, Mutex};

use crate::DecodedImage;
use crate::DecodedImageColorSpace;
use crate::DecodedImageMips;

#[derive(PartialEq)]
pub enum ImageMemoryBarrierType {
    PreUpload,
    PostUploadUnifiedQueues,
    PostUploadTransferQueue,
    PostUploadDstQueue,
}

pub fn cmd_image_memory_barrier(
    logical_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    images: &[vk::Image],
    barrier_type: ImageMemoryBarrierType,
    mut src_queue_family: u32,
    mut dst_queue_family: u32,
    subresource_range: &vk::ImageSubresourceRange,
) {
    if src_queue_family == dst_queue_family {
        src_queue_family = vk::QUEUE_FAMILY_IGNORED;
        dst_queue_family = vk::QUEUE_FAMILY_IGNORED;
    }

    struct SyncInfo {
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    }

    let sync_info = match barrier_type {
        ImageMemoryBarrierType::PreUpload => SyncInfo {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            src_layout: vk::ImageLayout::UNDEFINED,
            dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        },
        ImageMemoryBarrierType::PostUploadUnifiedQueues => SyncInfo {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        ImageMemoryBarrierType::PostUploadTransferQueue => SyncInfo {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::empty(),
            src_stage: vk::PipelineStageFlags::TRANSFER,
            dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE, // ignored, this is a release of resources to another queue
            src_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        ImageMemoryBarrierType::PostUploadDstQueue => SyncInfo {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE, // ignored, this is an acquire of resources from another queue
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
    };

    let barrier_infos: Vec<_> = images
        .iter()
        .map(|image| {
            vk::ImageMemoryBarrier::builder()
                .src_access_mask(sync_info.src_access_mask)
                .dst_access_mask(sync_info.dst_access_mask)
                .old_layout(sync_info.src_layout)
                .new_layout(sync_info.dst_layout)
                .src_queue_family_index(src_queue_family)
                .dst_queue_family_index(dst_queue_family)
                .image(*image)
                .subresource_range(*subresource_range)
                .build()
        })
        .collect();

    unsafe {
        logical_device.cmd_pipeline_barrier(
            command_buffer,
            sync_info.src_stage,
            sync_info.dst_stage,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &barrier_infos,
        );
    }
}

pub fn cmd_copy_buffer_to_image(
    logical_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    image: vk::Image,
    extent: &vk::Extent3D,
    image_subresource: &vk::ImageSubresourceLayers,
) {
    let image_copy = vk::BufferImageCopy::builder()
        .buffer_offset(offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(*image_subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(*extent);

    unsafe {
        logical_device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*image_copy],
        );
    }
}

fn generate_mips_for_image(
    device_context: &VkDeviceContext,
    upload: &mut VkTransferUpload,
    transfer_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: &ManuallyDrop<VkImage>,
    layer: u32,
    mip_level_count: u32,
) {
    let first_mip_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(layer)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1)
        .build();

    transition_for_mipmap(
        device_context.device(),
        upload.transfer_command_buffer(),
        image.image(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        transfer_queue_family_index,
        dst_queue_family_index,
        &first_mip_range,
    );

    transition_for_mipmap(
        device_context.device(),
        upload.dst_command_buffer(),
        image.image(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        transfer_queue_family_index,
        dst_queue_family_index,
        &first_mip_range,
    );

    do_generate_mips_for_image(
        device_context,
        upload.dst_command_buffer(),
        dst_queue_family_index,
        &image,
        layer,
        mip_level_count,
    );

    let all_mips_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(layer)
        .layer_count(1)
        .level_count(mip_level_count)
        .build();

    // Everything is in transfer read mode, transition it to our final layout
    transition_for_mipmap(
        device_context.device(),
        upload.dst_command_buffer(),
        image.image(),
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        dst_queue_family_index,
        dst_queue_family_index,
        &all_mips_range,
    );
}

fn do_generate_mips_for_image(
    device_context: &VkDeviceContext,
    command_buffer: vk::CommandBuffer,
    queue_family_index: u32, // queue family that will do mip generation
    image: &ManuallyDrop<VkImage>,
    layer: u32,
    mip_level_count: u32,
) {
    log::debug!("Generating mipmaps");

    // Walk through each mip level n:
    // - put level n+1 into write mode
    // - blit from n to n+1
    // - put level n+1 into read mode
    for dst_level in 1..mip_level_count {
        log::trace!("Generating mipmap level {}", dst_level);
        let src_level = dst_level - 1;

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(layer)
            .mip_level(src_level);

        let src_offsets = [
            vk::Offset3D::default(),
            vk::Offset3D::builder()
                .x((image.extent.width as i32 >> src_level as i32).max(1))
                .y((image.extent.height as i32 >> src_level as i32).max(1))
                .z(1)
                .build(),
        ];

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(layer)
            .mip_level(dst_level);

        let dst_offsets = [
            vk::Offset3D::default(),
            vk::Offset3D::builder()
                .x((image.extent.width as i32 >> dst_level as i32).max(1))
                .y((image.extent.height as i32 >> dst_level as i32).max(1))
                .z(1)
                .build(),
        ];

        log::trace!("src {:?}", src_offsets[1]);
        log::trace!("dst {:?}", dst_offsets[1]);

        let mip_subrange = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(dst_level)
            .level_count(1)
            .layer_count(1)
            .base_array_layer(layer);

        log::trace!("  transition to write");
        transition_for_mipmap(
            device_context.device(),
            command_buffer,
            image.image(),
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            queue_family_index,
            queue_family_index,
            &mip_subrange,
        );

        let image_blit = vk::ImageBlit::builder()
            .src_offsets(src_offsets)
            .src_subresource(*src_subresource)
            .dst_offsets(dst_offsets)
            .dst_subresource(*dst_subresource);

        log::trace!("  blit");
        unsafe {
            device_context.device().cmd_blit_image(
                command_buffer,
                image.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*image_blit],
                vk::Filter::LINEAR,
            );
        }

        log::trace!("  transition to read");
        transition_for_mipmap(
            device_context.device(),
            command_buffer,
            image.image(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            queue_family_index,
            queue_family_index,
            &mip_subrange,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn transition_for_mipmap(
    logical_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    src_layout: vk::ImageLayout,
    dst_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_queue_family: u32,
    dst_queue_family: u32,
    subresource_range: &vk::ImageSubresourceRange,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(src_layout)
        .new_layout(dst_layout)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .image(image)
        .subresource_range(*subresource_range)
        .build();

    unsafe {
        logical_device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );
    }
}

// This function is a little more complex to use than enqueue_load_images but can support cubemaps
// We create a layer for each layer_image_assignment, and copy from the decoded_image
// at the index matching the assignment
pub fn enqueue_load_layered_image_2d(
    device_context: &VkDeviceContext,
    upload: &mut VkTransferUpload,
    transfer_queue_family_index: u32,
    dst_queue_family_index: u32,
    decoded_images: &[DecodedImage],
    layer_image_assignments: &[usize],
    create_flags: vk::ImageCreateFlags,
) -> Result<ManuallyDrop<VkImage>, VkUploadError> {
    // All images must have identical mip level count
    #[cfg(debug_assertions)]
    {
        let first = &decoded_images[0];
        for decoded_image in decoded_images {
            assert_eq!(first.mips, decoded_image.mips);
            assert_eq!(first.width, decoded_image.width);
            assert_eq!(first.height, decoded_image.height);
            assert_eq!(first.color_space, decoded_image.color_space);
            assert_eq!(first.data.len(), decoded_image.data.len());
        }
    }

    // Arbitrary, not sure if there is any requirement
    const REQUIRED_ALIGNMENT: usize = 16;

    // Check ahead of time if there is space since we are uploading multiple images
    let has_space_available = upload.has_space_available(
        decoded_images[0].data.len(),
        REQUIRED_ALIGNMENT,
        decoded_images.len(),
    );
    if !has_space_available {
        Err(VkUploadError::BufferFull)?;
    }

    let extent = vk::Extent3D {
        width: decoded_images[0].width,
        height: decoded_images[0].height,
        depth: 1,
    };

    let (mip_level_count, generate_mips) = match decoded_images[0].mips {
        DecodedImageMips::None => (1, false),
        DecodedImageMips::Precomputed(_mip_count) => unimplemented!(), //(info.mip_level_count, false),
        DecodedImageMips::Runtime(mip_count) => (mip_count, true),
    };

    // Push all images into the staging buffer
    let mut layer_offsets = Vec::default();
    for decoded_image in decoded_images {
        layer_offsets.push(upload.push(&decoded_image.data, REQUIRED_ALIGNMENT)?);
    }

    let mut image_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
    if generate_mips {
        image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    };

    let format = match decoded_images[0].color_space {
        DecodedImageColorSpace::Linear => vk::Format::R8G8B8A8_UNORM,
        DecodedImageColorSpace::Srgb => vk::Format::R8G8B8A8_SRGB,
    };

    // Allocate an image
    let image = ManuallyDrop::new(VkImage::new(
        device_context,
        vk_mem::MemoryUsage::GpuOnly,
        create_flags,
        image_usage,
        extent,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::SampleCountFlags::TYPE_1,
        layer_image_assignments.len() as u32,
        mip_level_count,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?);

    //
    // Write into the transfer command buffer
    // - transition destination memory to receive the data
    // - copy the data
    // - transition the destination to the graphics queue
    //

    // Mip 0, all layers
    let subresource_range = vk::ImageSubresourceRange::builder()
        .base_array_layer(0)
        .layer_count(layer_image_assignments.len() as u32)
        .base_mip_level(0)
        .level_count(1)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .build();

    cmd_image_memory_barrier(
        device_context.device(),
        upload.transfer_command_buffer(),
        &[image.image()],
        ImageMemoryBarrierType::PreUpload,
        transfer_queue_family_index,
        transfer_queue_family_index,
        &subresource_range,
    );

    for (layer_index, image_index) in layer_image_assignments.iter().enumerate() {
        let layer_subresource_layers = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(layer_index as u32)
            .layer_count(1)
            .mip_level(0)
            .build();

        let layer_subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(layer_index as u32)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(1)
            .build();

        cmd_copy_buffer_to_image(
            device_context.device(),
            upload.transfer_command_buffer(),
            upload.staging_buffer().buffer(),
            layer_offsets[*image_index],
            image.image(),
            &image.extent,
            &layer_subresource_layers,
        );

        if generate_mips {
            // Generating mipmaps includes image barriers, so this function will handle writing the
            // image barriers required to pass from the transfer queue to the dst queue
            generate_mips_for_image(
                device_context,
                upload,
                transfer_queue_family_index,
                dst_queue_family_index,
                &image,
                layer_index as u32,
                mip_level_count,
            );
        } else {
            cmd_image_memory_barrier(
                device_context.device(),
                upload.transfer_command_buffer(),
                &[image.image()],
                ImageMemoryBarrierType::PostUploadTransferQueue,
                transfer_queue_family_index,
                dst_queue_family_index,
                &layer_subresource_range,
            );

            cmd_image_memory_barrier(
                device_context.device(),
                upload.dst_command_buffer(),
                &[image.image()],
                ImageMemoryBarrierType::PostUploadDstQueue,
                transfer_queue_family_index,
                dst_queue_family_index,
                &layer_subresource_range,
            );
        }
    }

    Ok(image)
}

pub fn enqueue_load_image(
    device_context: &VkDeviceContext,
    upload: &mut VkTransferUpload,
    transfer_queue_family_index: u32,
    dst_queue_family_index: u32,
    decoded_image: &DecodedImage,
) -> Result<ManuallyDrop<VkImage>, VkUploadError> {
    enqueue_load_layered_image_2d(
        device_context,
        upload,
        transfer_queue_family_index,
        dst_queue_family_index,
        std::slice::from_ref(decoded_image),
        &[0],
        vk::ImageCreateFlags::empty(),
    )
}

pub fn load_layered_image_2d_blocking(
    device_context: &VkDeviceContext,
    transfer_queue_family_index: u32,
    transfer_queue: &Arc<Mutex<vk::Queue>>,
    dst_queue_family_index: u32,
    dst_queue: &Arc<Mutex<vk::Queue>>,
    decoded_images: &[DecodedImage],
    layer_image_assignments: &[usize],
    create_flags: vk::ImageCreateFlags,
    upload_buffer_max_size: u64,
) -> Result<ManuallyDrop<VkImage>, VkUploadError> {
    let mut upload = VkTransferUpload::new(
        device_context,
        transfer_queue_family_index,
        dst_queue_family_index,
        upload_buffer_max_size,
    )?;

    let image = enqueue_load_layered_image_2d(
        device_context,
        &mut upload,
        transfer_queue_family_index,
        dst_queue_family_index,
        decoded_images,
        layer_image_assignments,
        create_flags,
    )?;

    upload.block_until_upload_complete(transfer_queue, dst_queue)?;

    Ok(image)
}

pub fn load_image_blocking(
    device_context: &VkDeviceContext,
    transfer_queue_family_index: u32,
    transfer_queue: &Arc<Mutex<vk::Queue>>,
    dst_queue_family_index: u32,
    dst_queue: &Arc<Mutex<vk::Queue>>,
    decoded_image: &DecodedImage,
    upload_buffer_max_size: u64,
) -> Result<ManuallyDrop<VkImage>, VkUploadError> {
    let mut upload = VkTransferUpload::new(
        device_context,
        transfer_queue_family_index,
        dst_queue_family_index,
        upload_buffer_max_size,
    )?;

    let image = enqueue_load_image(
        device_context,
        &mut upload,
        transfer_queue_family_index,
        dst_queue_family_index,
        decoded_image,
    )?;

    upload.block_until_upload_complete(transfer_queue, dst_queue)?;

    Ok(image)
}
