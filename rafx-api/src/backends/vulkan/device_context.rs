use super::internal::*;
use crate::*;
use ash::vk;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use crate::vulkan::{
    RafxBufferVulkan, RafxDescriptorSetArrayVulkan, RafxFenceVulkan, RafxPipelineVulkan,
    RafxQueueVulkan, RafxRootSignatureVulkan, RafxSamplerVulkan, RafxSemaphoreVulkan,
    RafxShaderModuleVulkan, RafxShaderVulkan, RafxSwapchainVulkan, RafxTextureVulkan,
};
use ash::extensions::khr;
use fnv::FnvHashMap;
use std::ffi::{CStr, CString};
#[cfg(debug_assertions)]
#[cfg(feature = "track-device-contexts")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};

/// Used to specify which type of physical device is preferred. It's recommended to read the Vulkan
/// spec to understand precisely what these types mean
///
/// Values here match VkPhysicalDeviceType, DiscreteGpu is the recommended default
#[derive(Copy, Clone, Debug)]
pub enum PhysicalDeviceType {
    /// Corresponds to `VK_PHYSICAL_DEVICE_TYPE_OTHER`
    Other = 0,

    /// Corresponds to `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`
    IntegratedGpu = 1,

    /// Corresponds to `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`
    DiscreteGpu = 2,

    /// Corresponds to `VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU`
    VirtualGpu = 3,

    /// Corresponds to `VK_PHYSICAL_DEVICE_TYPE_CPU`
    Cpu = 4,
}

impl PhysicalDeviceType {
    /// Convert to `vk::PhysicalDeviceType`
    pub fn to_vk(self) -> vk::PhysicalDeviceType {
        match self {
            PhysicalDeviceType::Other => vk::PhysicalDeviceType::OTHER,
            PhysicalDeviceType::IntegratedGpu => vk::PhysicalDeviceType::INTEGRATED_GPU,
            PhysicalDeviceType::DiscreteGpu => vk::PhysicalDeviceType::DISCRETE_GPU,
            PhysicalDeviceType::VirtualGpu => vk::PhysicalDeviceType::VIRTUAL_GPU,
            PhysicalDeviceType::Cpu => vk::PhysicalDeviceType::CPU,
        }
    }
}

#[derive(Clone)]
pub struct PhysicalDeviceInfo {
    pub score: i32,
    pub queue_family_indices: VkQueueFamilyIndices,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub extension_properties: Vec<ash::vk::ExtensionProperties>,
    pub all_queue_families: Vec<ash::vk::QueueFamilyProperties>,
}

#[derive(Default, Clone, Debug)]
pub struct VkQueueFamilyIndices {
    pub graphics_queue_family_index: u32,
    pub compute_queue_family_index: u32,
    pub transfer_queue_family_index: u32,
}

pub struct RafxDeviceContextVulkanInner {
    pub(crate) resource_cache: RafxDeviceVulkanResourceCache,
    pub(crate) descriptor_heap: RafxDescriptorHeapVulkan,
    pub(crate) device_info: RafxDeviceInfo,
    pub(crate) queue_allocator: VkQueueAllocatorSet,

    // If we need a dedicated present queue, we share a single queue across all swapchains. This
    // lock ensures that the present operations for those swapchains do not occur concurrently
    pub(crate) dedicated_present_queue_lock: Mutex<()>,

    // Device memories for externally-allocated images (bypassing gpu_allocator).
    // Keyed by VkImage handle so export_texture_handle can find the backing memory.
    pub(crate) external_device_memories: Mutex<FnvHashMap<vk::Image, vk::DeviceMemory>>,

    device: ash::Device,
    allocator: ManuallyDrop<Mutex<gpu_allocator::vulkan::Allocator>>,
    destroyed: AtomicBool,
    entry: Arc<VkEntry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    physical_device_info: PhysicalDeviceInfo,
    debug_reporter: Option<Arc<VkDebugReporter>>,

    #[cfg(debug_assertions)]
    #[cfg(feature = "track-device-contexts")]
    next_create_index: AtomicU64,

    #[cfg(debug_assertions)]
    #[cfg(feature = "track-device-contexts")]
    pub(crate) all_contexts: Mutex<fnv::FnvHashMap<u64, backtrace::Backtrace>>,
}

impl Drop for RafxDeviceContextVulkanInner {
    fn drop(&mut self) {
        if !self.destroyed.swap(true, Ordering::AcqRel) {
            unsafe {
                log::trace!("destroying device");

                // Free externally-managed device memories
                for (image, memory) in self.external_device_memories.lock().unwrap().drain() {
                    self.device.destroy_image(image, None);
                    self.device.free_memory(memory, None);
                }

                self.allocator
                    .lock()
                    .unwrap()
                    .report_memory_leaks(log::Level::Warn);
                ManuallyDrop::drop(&mut self.allocator);
                self.device.destroy_device(None);
                //self.surface_loader.destroy_surface(self.surface, None);
                log::trace!("destroyed device");
            }
        }
    }
}

impl RafxDeviceContextVulkanInner {
    pub fn new(
        instance: &VkInstance,
        vk_api_def: &RafxApiDefVulkan,
    ) -> RafxResult<Self> {
        let physical_device_type_priority = vec![
            PhysicalDeviceType::DiscreteGpu,
            PhysicalDeviceType::IntegratedGpu,
        ];

        // Pick a physical device
        let (physical_device, physical_device_info) =
            choose_physical_device(&instance.instance, &physical_device_type_priority)?;

        //TODO: Don't hardcode queue counts
        let queue_requirements = VkQueueRequirements::determine_required_queue_counts(
            physical_device_info.queue_family_indices.clone(),
            &physical_device_info.all_queue_families,
            VkQueueAllocationStrategy::ShareFirstQueueInFamily,
            VkQueueAllocationStrategy::ShareFirstQueueInFamily,
            VkQueueAllocationStrategy::ShareFirstQueueInFamily,
        );

        // Create a logical device
        let logical_device = create_logical_device(
            &instance.instance,
            physical_device,
            &physical_device_info,
            &queue_requirements,
            &vk_api_def.physical_device_features,
            &vk_api_def.additional_device_extensions,
        )?;

        let queue_allocator = VkQueueAllocatorSet::new(
            &logical_device,
            &physical_device_info.all_queue_families,
            queue_requirements,
        );

        let allocator_create_info = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            device: logical_device.clone(),
            instance: instance.instance.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false, // Should check BufferDeviceAddressFeatures first
        };

        let allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_info)?;

        let limits = &physical_device_info.properties.limits;

        // Debug names should be enable if explicitly opted-in and available (i.e. we created a
        // debug reporter)
        let debug_names_enabled =
            vk_api_def.enable_debug_names && instance.debug_reporter.is_some();

        let device_info = RafxDeviceInfo {
            supports_multithreaded_usage: true,
            debug_names_enabled,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as u32,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment as u32,
            upload_texture_alignment: limits.optimal_buffer_copy_offset_alignment as u32,
            upload_texture_row_alignment: limits.optimal_buffer_copy_row_pitch_alignment as u32,
            supports_clamp_to_border_color: true,
            max_vertex_attribute_count: limits.max_vertex_input_attributes,
        };

        let resource_cache = RafxDeviceVulkanResourceCache::default();
        let descriptor_heap = RafxDescriptorHeapVulkan::new(&logical_device)?;

        #[cfg(debug_assertions)]
        #[cfg(feature = "track-device-contexts")]
        let all_contexts = {
            let create_backtrace = backtrace::Backtrace::new_unresolved();
            let mut all_contexts = fnv::FnvHashMap::<u64, backtrace::Backtrace>::default();
            all_contexts.insert(0, create_backtrace);
            all_contexts
        };

        Ok(RafxDeviceContextVulkanInner {
            resource_cache,
            descriptor_heap,
            device_info,
            queue_allocator,
            dedicated_present_queue_lock: Mutex::default(),
            external_device_memories: Mutex::new(FnvHashMap::default()),
            entry: instance.entry.clone(),
            instance: instance.instance.clone(),
            physical_device,
            physical_device_info,
            device: logical_device,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            destroyed: AtomicBool::new(false),
            debug_reporter: instance.debug_reporter.clone(),

            #[cfg(debug_assertions)]
            #[cfg(feature = "track-device-contexts")]
            all_contexts: Mutex::new(all_contexts),

            #[cfg(debug_assertions)]
            #[cfg(feature = "track-device-contexts")]
            next_create_index: AtomicU64::new(1),
        })
    }
}

pub struct RafxDeviceContextVulkan {
    pub(crate) inner: Arc<RafxDeviceContextVulkanInner>,
    #[cfg(debug_assertions)]
    #[cfg(feature = "track-device-contexts")]
    pub(crate) create_index: u64,
}

impl std::fmt::Debug for RafxDeviceContextVulkan {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        f.debug_struct("RafxDeviceContextVulkan")
            .field("handle", &self.device().handle())
            .finish()
    }
}

impl Clone for RafxDeviceContextVulkan {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        #[cfg(feature = "track-device-contexts")]
        let create_index = {
            let create_index = self.inner.next_create_index.fetch_add(1, Ordering::Relaxed);

            #[cfg(feature = "track-device-contexts")]
            {
                let create_backtrace = backtrace::Backtrace::new_unresolved();
                self.inner
                    .as_ref()
                    .all_contexts
                    .lock()
                    .unwrap()
                    .insert(create_index, create_backtrace);
            }

            log::trace!(
                "Cloned RafxDeviceContextVulkan create_index {}",
                create_index
            );
            create_index
        };
        RafxDeviceContextVulkan {
            inner: self.inner.clone(),
            #[cfg(debug_assertions)]
            #[cfg(feature = "track-device-contexts")]
            create_index,
        }
    }
}

impl Drop for RafxDeviceContextVulkan {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        #[cfg(feature = "track-device-contexts")]
        {
            self.inner
                .all_contexts
                .lock()
                .unwrap()
                .remove(&self.create_index);
        }
    }
}

impl Into<RafxDeviceContext> for RafxDeviceContextVulkan {
    fn into(self) -> RafxDeviceContext {
        RafxDeviceContext::Vk(self)
    }
}

impl RafxDeviceContextVulkan {
    pub(crate) fn resource_cache(&self) -> &RafxDeviceVulkanResourceCache {
        &self.inner.resource_cache
    }

    pub(crate) fn descriptor_heap(&self) -> &RafxDescriptorHeapVulkan {
        &self.inner.descriptor_heap
    }

    pub fn device_info(&self) -> &RafxDeviceInfo {
        &self.inner.device_info
    }

    pub fn entry(&self) -> &VkEntry {
        &*self.inner.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(super) fn debug_reporter(&self) -> Option<&VkDebugReporter> {
        self.inner.debug_reporter.as_deref()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    pub fn physical_device_info(&self) -> &PhysicalDeviceInfo {
        &self.inner.physical_device_info
    }

    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.physical_device_info().properties.limits
    }

    pub fn allocator(&self) -> &Mutex<gpu_allocator::vulkan::Allocator> {
        &self.inner.allocator
    }

    pub fn queue_allocator(&self) -> &VkQueueAllocatorSet {
        &self.inner.queue_allocator
    }

    pub fn queue_family_indices(&self) -> &VkQueueFamilyIndices {
        &self.inner.physical_device_info.queue_family_indices
    }

    pub fn dedicated_present_queue_lock(&self) -> &Mutex<()> {
        &self.inner.dedicated_present_queue_lock
    }

    pub fn new(
        // instance: &VkInstance,
        // window: &dyn HasRawWindowHandle,
        inner: Arc<RafxDeviceContextVulkanInner>,
    ) -> RafxResult<Self> {
        //let inner = RafxDeviceContextVulkanInner::new(instance, window)?;

        Ok(RafxDeviceContextVulkan {
            inner,
            #[cfg(debug_assertions)]
            #[cfg(feature = "track-device-contexts")]
            create_index: 0,
        })
    }

    pub fn create_queue(
        &self,
        queue_type: RafxQueueType,
    ) -> RafxResult<RafxQueueVulkan> {
        RafxQueueVulkan::new(self, queue_type)
    }

    pub fn create_fence(&self) -> RafxResult<RafxFenceVulkan> {
        RafxFenceVulkan::new(self)
    }

    pub fn create_semaphore(&self) -> RafxResult<RafxSemaphoreVulkan> {
        RafxSemaphoreVulkan::new(self)
    }

    pub fn create_swapchain(
        &self,
        raw_display_handle: &dyn HasRawDisplayHandle,
        raw_window_handle: &dyn HasRawWindowHandle,
        _present_queue: &RafxQueueVulkan,
        swapchain_def: &RafxSwapchainDef,
    ) -> RafxResult<RafxSwapchainVulkan> {
        RafxSwapchainVulkan::new(self, raw_display_handle, raw_window_handle, swapchain_def)
    }

    pub fn wait_for_fences(
        &self,
        fences: &[&RafxFenceVulkan],
    ) -> RafxResult<()> {
        RafxFenceVulkan::wait_for_fences(self, fences)
    }

    pub fn create_sampler(
        &self,
        sampler_def: &RafxSamplerDef,
    ) -> RafxResult<RafxSamplerVulkan> {
        RafxSamplerVulkan::new(self, sampler_def)
    }

    pub fn create_texture(
        &self,
        texture_def: &RafxTextureDef,
    ) -> RafxResult<RafxTextureVulkan> {
        RafxTextureVulkan::new(self, texture_def)
    }

    pub fn create_buffer(
        &self,
        buffer_def: &RafxBufferDef,
    ) -> RafxResult<RafxBufferVulkan> {
        RafxBufferVulkan::new(self, buffer_def)
    }

    pub fn create_shader(
        &self,
        stages: Vec<RafxShaderStageDef>,
    ) -> RafxResult<RafxShaderVulkan> {
        RafxShaderVulkan::new(self, stages)
    }

    pub fn create_root_signature(
        &self,
        root_signature_def: &RafxRootSignatureDef,
    ) -> RafxResult<RafxRootSignatureVulkan> {
        RafxRootSignatureVulkan::new(self, root_signature_def)
    }

    pub fn create_descriptor_set_array(
        &self,
        descriptor_set_array_def: &RafxDescriptorSetArrayDef,
    ) -> RafxResult<RafxDescriptorSetArrayVulkan> {
        RafxDescriptorSetArrayVulkan::new(self, self.descriptor_heap(), descriptor_set_array_def)
    }

    pub fn create_graphics_pipeline(
        &self,
        graphics_pipeline_def: &RafxGraphicsPipelineDef,
    ) -> RafxResult<RafxPipelineVulkan> {
        RafxPipelineVulkan::new_graphics_pipeline(self, graphics_pipeline_def)
    }

    pub fn create_compute_pipeline(
        &self,
        compute_pipeline_def: &RafxComputePipelineDef,
    ) -> RafxResult<RafxPipelineVulkan> {
        RafxPipelineVulkan::new_compute_pipeline(self, compute_pipeline_def)
    }

    pub(crate) fn create_renderpass(
        &self,
        renderpass_def: &RafxRenderpassVulkanDef,
    ) -> RafxResult<RafxRenderpassVulkan> {
        RafxRenderpassVulkan::new(self, renderpass_def)
    }

    pub fn create_shader_module(
        &self,
        data: RafxShaderModuleDefVulkan,
    ) -> RafxResult<RafxShaderModuleVulkan> {
        RafxShaderModuleVulkan::new(self, data)
    }

    // // Just expects bytes with no particular alignment requirements, suitable for reading from a file
    // pub fn create_shader_module_from_bytes(
    //     &self,
    //     data: &[u8],
    // ) -> RafxResult<RafxShaderModuleVulkan> {
    //     RafxShaderModuleVulkan::new_from_bytes(self, data)
    // }
    //
    // // Expects properly aligned, correct endianness, valid SPV
    // pub fn create_shader_module_from_spv(
    //     &self,
    //     spv: &[u32],
    // ) -> RafxResult<RafxShaderModuleVulkan> {
    //     RafxShaderModuleVulkan::new_from_spv(self, spv)
    // }

    pub fn create_exportable_texture(
        &self,
        texture_def: &RafxTextureDef,
    ) -> RafxResult<RafxTextureVulkan> {
        texture_def.verify();

        let dimensions = texture_def
            .dimensions
            .determine_dimensions(texture_def.extents);
        let image_type = match dimensions {
            crate::RafxTextureDimensions::Dim1D => vk::ImageType::TYPE_1D,
            crate::RafxTextureDimensions::Dim2D => vk::ImageType::TYPE_2D,
            crate::RafxTextureDimensions::Dim3D => vk::ImageType::TYPE_3D,
            crate::RafxTextureDimensions::Auto => panic!("dimensions() should not return auto"),
        };

        let format_vk: vk::Format = texture_def.format.into();

        let mut usage_flags =
            super::util::resource_type_image_usage_flags(texture_def.resource_type);
        if texture_def
            .resource_type
            .intersects(RafxResourceType::RENDER_TARGET_COLOR)
        {
            usage_flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if usage_flags.intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE) {
            usage_flags |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        }

        let extent = vk::Extent3D {
            width: texture_def.extents.width,
            height: texture_def.extents.height,
            depth: texture_def.extents.depth,
        };

        #[cfg(target_os = "linux")]
        let (image, device_memory) = {
            let mut external_info = vk::ExternalMemoryImageCreateInfo::builder()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                .build();

            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(image_type)
                .extent(extent)
                .mip_levels(texture_def.mip_count)
                .array_layers(texture_def.array_length)
                .format(format_vk)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(texture_def.sample_count.into())
                .push_next(&mut external_info);

            let device = self.device();
            let image = unsafe { device.create_image(&image_create_info, None)? };

            let mem_reqs = unsafe { device.get_image_memory_requirements(image) };

            let mem_type_index = find_memory_type(
                self,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let mut export_info = vk::ExportMemoryAllocateInfo::builder()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                .build();
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                .image(image)
                .build();

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_info)
                .push_next(&mut dedicated_info);

            let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            unsafe { device.bind_image_memory(image, device_memory, 0)? };

            (image, device_memory)
        };

        #[cfg(target_os = "macos")]
        let (image, device_memory) = {
            eprintln!("  create_exportable_texture: macOS path — chaining ExportMetalObjectCreateInfoEXT");
            // Chain ExportMetalObjectCreateInfoEXT to request IOSurface-backed image
            let mut metal_export_info = vk::ExportMetalObjectCreateInfoEXT::builder()
                .export_object_type(vk::ExportMetalObjectTypeFlagsEXT::METAL_IOSURFACE)
                .build();

            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(image_type)
                .extent(extent)
                .mip_levels(texture_def.mip_count)
                .array_layers(texture_def.array_length)
                .format(format_vk)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(texture_def.sample_count.into())
                .push_next(&mut metal_export_info);

            let device = self.device();
            eprintln!("  create_exportable_texture: calling vkCreateImage...");
            let image = unsafe { device.create_image(&image_create_info, None)? };
            eprintln!("  create_exportable_texture: vkCreateImage OK");

            let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
            eprintln!("  create_exportable_texture: mem_reqs size={}, type_bits={:#x}", mem_reqs.size, mem_reqs.memory_type_bits);

            let mem_type_index = find_memory_type(
                self,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            eprintln!("  create_exportable_texture: mem_type_index={}", mem_type_index);

            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                .image(image)
                .build();

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut dedicated_info);

            eprintln!("  create_exportable_texture: calling vkAllocateMemory...");
            let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            eprintln!("  create_exportable_texture: calling vkBindImageMemory...");
            unsafe { device.bind_image_memory(image, device_memory, 0)? };
            eprintln!("  create_exportable_texture: image created and bound OK");

            (image, device_memory)
        };

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            // Suppress unused-variable warnings for platform-gated code above.
            let _ = image_type;
            let _ = format_vk;
            let _ = usage_flags;
            return Err(RafxError::StringError(
                "Exportable textures not supported on this platform".to_string(),
            ));
        }

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let raw_image = crate::vulkan::RafxRawImageVulkan {
                image,
                // allocation is None — we manage device_memory ourselves.
                // The image+memory are cleaned up by ExternalImageCleanup stored in the texture.
                allocation: None,
            };

            // Store the device memory so we can free it and export fd from it.
            // We attach it via a wrapper that we keep alive alongside the texture.
            let texture = RafxTextureVulkan::from_existing(self, Some(raw_image), texture_def)?;

            // Stash the device_memory on a side-table so export_texture_handle can find it
            self.inner
                .external_device_memories
                .lock()
                .unwrap()
                .insert(texture.vk_image(), device_memory);

            Ok(texture)
        }
    }

    pub fn export_texture_handle(
        &self,
        texture: &RafxTextureVulkan,
    ) -> RafxResult<crate::RafxExternalTextureHandle> {
        #[cfg(target_os = "linux")]
        {
            let device_memory = *self
                .inner
                .external_device_memories
                .lock()
                .unwrap()
                .get(&texture.vk_image())
                .ok_or_else(|| {
                    RafxError::StringError(
                        "Texture was not created with create_exportable_texture".to_string(),
                    )
                })?;

            let fd_loader =
                ash::extensions::khr::ExternalMemoryFd::new(self.instance(), self.device());
            let fd_info = vk::MemoryGetFdInfoKHR::builder()
                .memory(device_memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let fd = unsafe { fd_loader.get_memory_fd(&fd_info)? };
            Ok(crate::RafxExternalTextureHandle::Fd(fd))
        }

        #[cfg(target_os = "macos")]
        {
            let metal_objects_fn = vk::ExtMetalObjectsFn::load(|name| unsafe {
                std::mem::transmute(
                    self.instance()
                        .get_device_proc_addr(self.device().handle(), name.as_ptr()),
                )
            });

            let mut iosurface_info = vk::ExportMetalIOSurfaceInfoEXT::builder()
                .image(texture.vk_image())
                .build();
            let mut export_info = vk::ExportMetalObjectsInfoEXT::builder()
                .push_next(&mut iosurface_info)
                .build();

            unsafe {
                (metal_objects_fn.export_metal_objects_ext)(self.device().handle(), &mut export_info)
            };

            if iosurface_info.io_surface.is_null() {
                return Err(RafxError::StringError(
                    "Failed to export IOSurface from texture".to_string(),
                ));
            }

            let id = unsafe { IOSurfaceGetID(iosurface_info.io_surface) };
            Ok(crate::RafxExternalTextureHandle::IOSurfaceId(id))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Err(RafxError::StringError(
            "Texture export not supported on this platform".to_string(),
        ))
    }

    pub fn import_texture(
        &self,
        texture_def: &RafxTextureDef,
        handle: crate::RafxExternalTextureHandle,
    ) -> RafxResult<RafxTextureVulkan> {
        texture_def.verify();

        let dimensions = texture_def
            .dimensions
            .determine_dimensions(texture_def.extents);
        let image_type = match dimensions {
            crate::RafxTextureDimensions::Dim1D => vk::ImageType::TYPE_1D,
            crate::RafxTextureDimensions::Dim2D => vk::ImageType::TYPE_2D,
            crate::RafxTextureDimensions::Dim3D => vk::ImageType::TYPE_3D,
            crate::RafxTextureDimensions::Auto => panic!("dimensions() should not return auto"),
        };

        let format_vk: vk::Format = texture_def.format.into();

        let mut usage_flags =
            super::util::resource_type_image_usage_flags(texture_def.resource_type);
        if texture_def
            .resource_type
            .intersects(RafxResourceType::RENDER_TARGET_COLOR)
        {
            usage_flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if usage_flags.intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE) {
            usage_flags |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        }

        let extent = vk::Extent3D {
            width: texture_def.extents.width,
            height: texture_def.extents.height,
            depth: texture_def.extents.depth,
        };

        #[cfg(target_os = "linux")]
        {
            let crate::RafxExternalTextureHandle::Fd(fd) = handle else {
                return Err(RafxError::StringError(
                    "Expected Fd handle on Linux".to_string(),
                ));
            };

            let mut external_info = vk::ExternalMemoryImageCreateInfo::builder()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                .build();

            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(image_type)
                .extent(extent)
                .mip_levels(texture_def.mip_count)
                .array_layers(texture_def.array_length)
                .format(format_vk)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(texture_def.sample_count.into())
                .push_next(&mut external_info);

            let device = self.device();
            let image = unsafe { device.create_image(&image_create_info, None)? };

            let mem_reqs = unsafe { device.get_image_memory_requirements(image) };

            let mem_type_index = find_memory_type(
                self,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let mut import_fd_info = vk::ImportMemoryFdInfoKHR::builder()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                .fd(fd)
                .build();
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                .image(image)
                .build();

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut import_fd_info)
                .push_next(&mut dedicated_info);

            let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            unsafe { device.bind_image_memory(image, device_memory, 0)? };

            let raw_image = crate::vulkan::RafxRawImageVulkan {
                image,
                allocation: None,
            };

            self.inner
                .external_device_memories
                .lock()
                .unwrap()
                .insert(image, device_memory);

            RafxTextureVulkan::from_existing(self, Some(raw_image), texture_def)
        }

        #[cfg(target_os = "macos")]
        {
            let crate::RafxExternalTextureHandle::IOSurfaceId(surface_id) = handle else {
                return Err(RafxError::StringError(
                    "Expected IOSurfaceId handle on macOS".to_string(),
                ));
            };

            let io_surface = unsafe { IOSurfaceLookup(surface_id) };
            if io_surface.is_null() {
                return Err(RafxError::StringError(format!(
                    "IOSurfaceLookup({surface_id}) returned null"
                )));
            }

            let mut import_iosurface = vk::ImportMetalIOSurfaceInfoEXT::builder()
                .io_surface(io_surface)
                .build();

            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(image_type)
                .extent(extent)
                .mip_levels(texture_def.mip_count)
                .array_layers(texture_def.array_length)
                .format(format_vk)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(texture_def.sample_count.into())
                .push_next(&mut import_iosurface);

            let device = self.device();
            let image = unsafe { device.create_image(&image_create_info, None)? };

            let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
            let mem_type_index = find_memory_type(
                self,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                .image(image)
                .build();
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut dedicated_info);

            let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            unsafe { device.bind_image_memory(image, device_memory, 0)? };

            let raw_image = crate::vulkan::RafxRawImageVulkan {
                image,
                allocation: None,
            };

            self.inner
                .external_device_memories
                .lock()
                .unwrap()
                .insert(image, device_memory);

            RafxTextureVulkan::from_existing(self, Some(raw_image), texture_def)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Err(RafxError::StringError(
            "Texture import not supported on this platform".to_string(),
        ))
    }

    pub fn create_exportable_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> RafxResult<crate::vulkan::RafxTimelineSemaphoreVulkan> {
        #[cfg(target_os = "linux")]
        {
            let mut export_info = vk::ExportSemaphoreCreateInfo::builder()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                .build();

            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(initial_value)
                .build();

            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .push_next(&mut export_info)
                .build();

            let vk_semaphore =
                unsafe { self.device().create_semaphore(&create_info, None)? };

            Ok(unsafe {
                crate::vulkan::RafxTimelineSemaphoreVulkan::from_existing(self, vk_semaphore)
            })
        }

        #[cfg(target_os = "macos")]
        {
            // Create MTLSharedEvent directly via Metal API so we get a real
            // system object that supports machPort for cross-process sharing.
            // Get machPort NOW (before MoltenVK can wrap/replace the object),
            // then import the event into MoltenVK.
            let mtl_shared_event = create_mtl_shared_event()?;
            let mach_port = get_shared_event_mach_port(mtl_shared_event)?;

            let mut import_event = vk::ImportMetalSharedEventInfoEXT::builder()
                .mtl_shared_event(mtl_shared_event)
                .build();

            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(initial_value)
                .build();

            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .push_next(&mut import_event)
                .build();

            let vk_semaphore =
                unsafe { self.device().create_semaphore(&create_info, None)? };

            let mut sem = unsafe {
                crate::vulkan::RafxTimelineSemaphoreVulkan::from_existing(self, vk_semaphore)
            };
            sem.mach_port = Some(mach_port);
            Ok(sem)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Err(RafxError::StringError(
            "Exportable timeline semaphores not supported on this platform".to_string(),
        ))
    }

    pub fn export_timeline_semaphore_handle(
        &self,
        semaphore: &crate::vulkan::RafxTimelineSemaphoreVulkan,
    ) -> RafxResult<crate::RafxExternalSemaphoreHandle> {
        #[cfg(target_os = "linux")]
        {
            let fd_loader =
                ash::extensions::khr::ExternalSemaphoreFd::new(self.instance(), self.device());
            let fd_info = vk::SemaphoreGetFdInfoKHR::builder()
                .semaphore(semaphore.vk_semaphore())
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
            let fd = unsafe { fd_loader.get_semaphore_fd(&fd_info)? };
            Ok(crate::RafxExternalSemaphoreHandle::Fd(fd))
        }

        #[cfg(target_os = "macos")]
        {
            // machPort was obtained during create_exportable_timeline_semaphore
            // from the Metal-created MTLSharedEvent (before MoltenVK import).
            let mach_port = semaphore.mach_port.ok_or_else(|| {
                RafxError::StringError(
                    "Timeline semaphore has no stored machPort (was it created with create_exportable_timeline_semaphore?)".to_string(),
                )
            })?;
            Ok(crate::RafxExternalSemaphoreHandle::MachPort(mach_port))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Err(RafxError::StringError(
            "Timeline semaphore export not supported on this platform".to_string(),
        ))
    }

    pub fn import_timeline_semaphore(
        &self,
        handle: crate::RafxExternalSemaphoreHandle,
    ) -> RafxResult<crate::vulkan::RafxTimelineSemaphoreVulkan> {
        #[cfg(target_os = "linux")]
        {
            let crate::RafxExternalSemaphoreHandle::Fd(fd) = handle else {
                return Err(RafxError::StringError(
                    "Expected Fd handle on Linux".to_string(),
                ));
            };

            // Create a timeline semaphore and import the fd
            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0)
                .build();
            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .build();
            let vk_semaphore =
                unsafe { self.device().create_semaphore(&create_info, None)? };

            let fd_loader =
                ash::extensions::khr::ExternalSemaphoreFd::new(self.instance(), self.device());
            let import_info = vk::ImportSemaphoreFdInfoKHR::builder()
                .semaphore(vk_semaphore)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                .fd(fd);
            unsafe { fd_loader.import_semaphore_fd(&import_info)? };

            Ok(unsafe {
                crate::vulkan::RafxTimelineSemaphoreVulkan::from_existing(self, vk_semaphore)
            })
        }

        #[cfg(target_os = "macos")]
        {
            let crate::RafxExternalSemaphoreHandle::MachPort(mach_port) = handle else {
                return Err(RafxError::StringError(
                    "Expected MachPort handle on macOS".to_string(),
                ));
            };

            let mtl_shared_event = create_shared_event_from_mach_port(mach_port)?;

            let mut import_event = vk::ImportMetalSharedEventInfoEXT::builder()
                .mtl_shared_event(mtl_shared_event)
                .build();
            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0)
                .build();
            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .push_next(&mut import_event)
                .build();

            let vk_semaphore =
                unsafe { self.device().create_semaphore(&create_info, None)? };

            Ok(unsafe {
                crate::vulkan::RafxTimelineSemaphoreVulkan::from_existing(self, vk_semaphore)
            })
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Err(RafxError::StringError(
            "Timeline semaphore import not supported on this platform".to_string(),
        ))
    }

    pub fn find_supported_format(
        &self,
        candidates: &[RafxFormat],
        resource_type: RafxResourceType,
    ) -> Option<RafxFormat> {
        let mut features = vk::FormatFeatureFlags::empty();
        if resource_type.intersects(RafxResourceType::RENDER_TARGET_COLOR) {
            features |= vk::FormatFeatureFlags::COLOR_ATTACHMENT;
        }

        if resource_type.intersects(RafxResourceType::RENDER_TARGET_DEPTH_STENCIL) {
            features |= vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        }

        do_find_supported_format(
            &self.inner.instance,
            self.inner.physical_device,
            candidates,
            vk::ImageTiling::OPTIMAL,
            features,
        )
    }

    pub fn find_supported_sample_count(
        &self,
        candidates: &[RafxSampleCount],
    ) -> Option<RafxSampleCount> {
        do_find_supported_sample_count(self.limits(), candidates)
    }
}

pub fn do_find_supported_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    candidates: &[RafxFormat],
    image_tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<RafxFormat> {
    for &candidate in candidates {
        let props = unsafe {
            instance.get_physical_device_format_properties(physical_device, candidate.into())
        };

        let is_supported = match image_tiling {
            vk::ImageTiling::LINEAR => (props.linear_tiling_features & features) == features,
            vk::ImageTiling::OPTIMAL => (props.optimal_tiling_features & features) == features,
            _ => unimplemented!(),
        };

        if is_supported {
            return Some(candidate);
        }
    }

    None
}

fn do_find_supported_sample_count(
    limits: &vk::PhysicalDeviceLimits,
    sample_count_priority: &[RafxSampleCount],
) -> Option<RafxSampleCount> {
    for &sample_count in sample_count_priority {
        let vk_sample_count: vk::SampleCountFlags = sample_count.into();
        if (vk_sample_count.as_raw()
            & limits.framebuffer_depth_sample_counts.as_raw()
            & limits.framebuffer_color_sample_counts.as_raw())
            != 0
        {
            log::trace!("Sample count {:?} is supported", sample_count);
            return Some(sample_count);
        } else {
            log::trace!("Sample count {:?} is unsupported", sample_count);
        }
    }

    None
}

fn choose_physical_device(
    instance: &ash::Instance,
    physical_device_type_priority: &[PhysicalDeviceType],
) -> RafxResult<(ash::vk::PhysicalDevice, PhysicalDeviceInfo)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    if physical_devices.is_empty() {
        panic!("Could not find a physical device");
    }

    let mut best_physical_device = None;
    let mut best_physical_device_info = None;
    let mut best_physical_device_score = -1;

    // let mut best_physical_device_queue_family_indices = None;
    for physical_device in physical_devices {
        let result = query_physical_device_info(
            instance,
            physical_device,
            //surface_loader,
            //surface,
            physical_device_type_priority,
        );

        if let Some(physical_device_info) = result? {
            if physical_device_info.score > best_physical_device_score {
                best_physical_device = Some(physical_device);
                best_physical_device_score = physical_device_info.score;
                best_physical_device_info = Some(physical_device_info);
            }
        }
    }

    //TODO: Return an error
    let physical_device = best_physical_device.expect("Could not find suitable device");
    let physical_device_info = best_physical_device_info.unwrap();

    Ok((physical_device, physical_device_info))
}

fn vk_version_to_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

fn query_physical_device_info(
    instance: &ash::Instance,
    device: ash::vk::PhysicalDevice,
    //surface_loader: &ash::extensions::khr::Surface,
    //surface: ash::vk::SurfaceKHR,
    physical_device_type_priority: &[PhysicalDeviceType],
) -> RafxResult<Option<PhysicalDeviceInfo>> {
    log::info!(
        "Preferred device types: {:?}",
        physical_device_type_priority
    );

    let properties: ash::vk::PhysicalDeviceProperties =
        unsafe { instance.get_physical_device_properties(device) };
    let device_name = unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
            .to_str()
            .unwrap()
            .to_string()
    };

    //TODO: Check that the extensions we want to use are supported
    let extensions: Vec<ash::vk::ExtensionProperties> =
        unsafe { instance.enumerate_device_extension_properties(device)? };
    let features: vk::PhysicalDeviceFeatures =
        unsafe { instance.get_physical_device_features(device) };
    let all_queue_families: Vec<ash::vk::QueueFamilyProperties> =
        unsafe { instance.get_physical_device_queue_family_properties(device) };

    let queue_family_indices = find_queue_families(&all_queue_families)?;
    if let Some(queue_family_indices) = queue_family_indices {
        // Determine the index of the device_type within physical_device_type_priority
        let index = physical_device_type_priority
            .iter()
            .map(|x| x.to_vk())
            .position(|x| x == properties.device_type);

        // Convert it to a score
        let rank = if let Some(index) = index {
            // It's in the list, return a value between 1..n
            physical_device_type_priority.len() - index
        } else {
            // Not in the list, return a zero
            0
        } as i32;

        let mut score = 0;
        score += rank * 100;

        log::info!(
            "Found suitable device '{}' API: {} DriverVersion: {} Score = {}",
            device_name,
            vk_version_to_string(properties.api_version),
            vk_version_to_string(properties.driver_version),
            score
        );

        let result = PhysicalDeviceInfo {
            score,
            queue_family_indices,
            properties,
            extension_properties: extensions,
            features,
            all_queue_families,
        };

        log::trace!("{:#?}", properties);
        Ok(Some(result))
    } else {
        log::info!(
            "Found unsuitable device '{}' API: {} DriverVersion: {} could not find queue families",
            device_name,
            vk_version_to_string(properties.api_version),
            vk_version_to_string(properties.driver_version)
        );
        log::trace!("{:#?}", properties);
        Ok(None)
    }
}

//TODO: Could improve this by looking at vendor/device ID, VRAM size, supported feature set, etc.
fn find_queue_families(
    all_queue_families: &[ash::vk::QueueFamilyProperties]
) -> RafxResult<Option<VkQueueFamilyIndices>> {
    let mut graphics_queue_family_index = None;
    let mut compute_queue_family_index = None;
    let mut transfer_queue_family_index = None;

    log::info!("Available queue families:");
    for (queue_family_index, queue_family) in all_queue_families.iter().enumerate() {
        log::info!("Queue Family {}", queue_family_index);
        log::info!("{:#?}", queue_family);
    }

    //
    // Find the first queue family that supports graphics and use it for graphics
    //
    for (queue_family_index, queue_family) in all_queue_families.iter().enumerate() {
        let queue_family_index = queue_family_index as u32;
        let supports_graphics = queue_family.queue_flags & ash::vk::QueueFlags::GRAPHICS
            == ash::vk::QueueFlags::GRAPHICS;

        if supports_graphics {
            graphics_queue_family_index = Some(queue_family_index);
            break;
        }
    }

    //
    // Find a compute queue family in the following order of preference:
    // - Doesn't support graphics
    // - Supports graphics but hasn't already been claimed by graphics
    // - Fallback to using the graphics queue family as it's guaranteed to support compute
    //
    for (queue_family_index, queue_family) in all_queue_families.iter().enumerate() {
        let queue_family_index = queue_family_index as u32;
        let supports_graphics = queue_family.queue_flags & ash::vk::QueueFlags::GRAPHICS
            == ash::vk::QueueFlags::GRAPHICS;
        let supports_compute =
            queue_family.queue_flags & ash::vk::QueueFlags::COMPUTE == ash::vk::QueueFlags::COMPUTE;

        if !supports_graphics && supports_compute {
            // Ideally we want to find a dedicated compute queue (i.e. doesn't support graphics)
            compute_queue_family_index = Some(queue_family_index);
            break;
        } else if supports_compute
            && compute_queue_family_index.is_none()
            && Some(queue_family_index) != graphics_queue_family_index
        {
            // Otherwise accept the first queue that supports compute that is NOT the graphics queue
            compute_queue_family_index = Some(queue_family_index);
        }
    }

    // If we didn't find a compute queue family != graphics queue family, settle for using the
    // graphics queue family. It's guaranteed to support compute.
    if compute_queue_family_index.is_none() {
        compute_queue_family_index = graphics_queue_family_index;
    }

    //
    // Find a transfer queue family in the following order of preference:
    // - Doesn't support graphics or compute
    // - Supports graphics but hasn't already been claimed by compute or graphics
    // - Fallback to using the graphics queue family as it's guaranteed to support transfers
    //
    for (queue_family_index, queue_family) in all_queue_families.iter().enumerate() {
        let queue_family_index = queue_family_index as u32;
        let supports_graphics = queue_family.queue_flags & ash::vk::QueueFlags::GRAPHICS
            == ash::vk::QueueFlags::GRAPHICS;
        let supports_compute =
            queue_family.queue_flags & ash::vk::QueueFlags::COMPUTE == ash::vk::QueueFlags::COMPUTE;
        let supports_transfer = queue_family.queue_flags & ash::vk::QueueFlags::TRANSFER
            == ash::vk::QueueFlags::TRANSFER;

        if !supports_graphics && !supports_compute && supports_transfer {
            // Ideally we want to find a dedicated transfer queue
            transfer_queue_family_index = Some(queue_family_index);
            break;
        } else if supports_transfer
            && transfer_queue_family_index.is_none()
            && Some(queue_family_index) != graphics_queue_family_index
            && Some(queue_family_index) != compute_queue_family_index
        {
            // Otherwise accept the first queue that supports transfers that is NOT the graphics queue or compute queue
            transfer_queue_family_index = Some(queue_family_index);
        }
    }

    // If we didn't find a transfer queue family != graphics queue family, settle for using the
    // graphics queue family. It's guaranteed to support transfer.
    if transfer_queue_family_index.is_none() {
        transfer_queue_family_index = graphics_queue_family_index;
    }

    log::info!(
        "Graphics QF: {:?}  Compute QF: {:?}  Transfer QF: {:?}",
        graphics_queue_family_index,
        compute_queue_family_index,
        transfer_queue_family_index
    );

    if let (
        Some(graphics_queue_family_index),
        Some(compute_queue_family_index),
        Some(transfer_queue_family_index),
    ) = (
        graphics_queue_family_index,
        compute_queue_family_index,
        transfer_queue_family_index,
    ) {
        Ok(Some(VkQueueFamilyIndices {
            graphics_queue_family_index,
            compute_queue_family_index,
            transfer_queue_family_index,
        }))
    } else {
        Ok(None)
    }
}

fn find_memory_type(
    device_context: &RafxDeviceContextVulkan,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> RafxResult<u32> {
    let mem_properties = unsafe {
        device_context
            .instance()
            .get_physical_device_memory_properties(device_context.physical_device())
    };
    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    Err(RafxError::StringError(
        "Failed to find suitable memory type".to_string(),
    ))
}

// IOSurface FFI (macOS)
#[cfg(target_os = "macos")]
#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetID(surface: vk::IOSurfaceRef) -> u32;
    fn IOSurfaceLookup(surface_id: u32) -> vk::IOSurfaceRef;
}

/// Create an MTLSharedEvent directly via the Metal API.
/// This produces a real system MTLSharedEvent that supports machPort,
/// unlike the one MoltenVK creates internally via vkExportMetalObjectsEXT.
#[cfg(target_os = "macos")]
fn create_mtl_shared_event() -> RafxResult<vk::MTLSharedEvent_id> {
    use std::ffi::c_void;

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_msgSend();
        fn sel_registerName(name: *const u8) -> *mut c_void;
    }

    #[link(name = "Metal", kind = "framework")]
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }

    type MsgSendObj = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;

    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return Err(RafxError::StringError(
                "MTLCreateSystemDefaultDevice returned null".to_string(),
            ));
        }

        let send_obj: MsgSendObj =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let sel = sel_registerName(b"newSharedEvent\0".as_ptr());
        let event = send_obj(device, sel);
        if event.is_null() {
            return Err(RafxError::StringError(
                "MTLDevice.newSharedEvent returned null".to_string(),
            ));
        }
        Ok(event)
    }
}

// Cross-process MTLSharedEvent sharing via Mach port.
//
// The MTLSharedEvent protocol has no `machPort` getter. Instead we go through
// MTLSharedEventHandle: [event newSharedEventHandle] → [handle eventPort].
//
// IMPORTANT: On ARM64, objc_msgSend uses the standard (non-variadic) calling convention.
// We must use typed function pointers so arguments are passed in registers.
#[cfg(target_os = "macos")]
fn get_shared_event_mach_port(
    mtl_shared_event: vk::MTLSharedEvent_id,
) -> RafxResult<u32> {
    use std::ffi::c_void;

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_msgSend();
        fn sel_registerName(name: *const u8) -> *mut c_void;
    }

    type MsgSendObj = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    type MsgSendU32 = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;

    unsafe {
        let send_obj: MsgSendObj =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let send_u32: MsgSendU32 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        // [event newSharedEventHandle] → MTLSharedEventHandle*
        let sel_new_handle = sel_registerName(b"newSharedEventHandle\0".as_ptr());
        let handle = send_obj(mtl_shared_event, sel_new_handle);
        if handle.is_null() {
            return Err(RafxError::StringError(
                "MTLSharedEvent.newSharedEventHandle returned nil".to_string(),
            ));
        }

        // [handle eventPort] → uint32_t (Mach port send right)
        let sel_event_port = sel_registerName(b"eventPort\0".as_ptr());
        let port = send_u32(handle, sel_event_port);

        // Don't release the handle — it owns the Mach send right.
        // If we release it, the send right is deallocated, leaving a dead name.
        // The handle is leaked intentionally; it's a small object and the
        // semaphore lives for the lifetime of the process.

        if port == 0 {
            return Err(RafxError::StringError(
                "MTLSharedEventHandle.eventPort returned MACH_PORT_NULL".to_string(),
            ));
        }
        Ok(port)
    }
}

#[cfg(target_os = "macos")]
fn create_shared_event_from_mach_port(
    mach_port: u32,
) -> RafxResult<vk::MTLSharedEvent_id> {
    use std::ffi::c_void;

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_msgSend();
        fn sel_registerName(name: *const u8) -> *mut c_void;
    }

    #[link(name = "Metal", kind = "framework")]
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }

    // [device newSharedEventWithMachPort:] takes u32, returns id<MTLSharedEvent>
    type MsgSendPort = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> *mut c_void;

    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return Err(RafxError::StringError(
                "MTLCreateSystemDefaultDevice returned null".to_string(),
            ));
        }

        let send_port: MsgSendPort =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let sel = sel_registerName(b"newSharedEventWithMachPort:\0".as_ptr());
        let event = send_port(device, sel, mach_port);
        if event.is_null() {
            return Err(RafxError::StringError(format!(
                "newSharedEventWithMachPort({mach_port}) returned null"
            )));
        }
        Ok(event)
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    physical_device_info: &PhysicalDeviceInfo,
    queue_requirements: &VkQueueRequirements,
    physical_device_features: &Option<vk::PhysicalDeviceFeatures>,
    additional_device_extensions: &[CString],
) -> RafxResult<ash::Device> {
    //TODO: Ideally we would set up validation layers for the logical device too.

    #[allow(unused_mut)]
    let mut device_extension_names = vec![khr::Swapchain::name().as_ptr()];

    #[cfg(target_os = "macos")]
    {
        fn khr_portability_subset_extension_name() -> &'static CStr {
            CStr::from_bytes_with_nul(b"VK_KHR_portability_subset\0")
                .expect("Wrong extension string")
        }

        // Add VK_KHR_portability_subset if the extension exists (this is mandated by spec)
        let portability_subset_extension_name = khr_portability_subset_extension_name();
        for extension in &physical_device_info.extension_properties {
            let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };

            if extension_name == portability_subset_extension_name {
                device_extension_names.push(portability_subset_extension_name.as_ptr());
                break;
            }
        }
    }

    for ext in additional_device_extensions {
        device_extension_names.push(ext.as_ptr());
    }

    // If no features were specified, enable a few that are very widely supported features.
    let physical_device_features = physical_device_features.clone().unwrap_or_else(|| {
        vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .sample_rate_shading(true)
            // Used for debug drawing lines/points
            .fill_mode_non_solid(true)
            // We can trivially fake this if the feature isn't available, so we can have it on by default
            .multi_draw_indirect(physical_device_info.features.multi_draw_indirect != 0)
            .build()
    });

    let mut queue_families_to_create = FnvHashMap::default();
    for (&queue_family_index, &count) in &queue_requirements.queue_counts {
        queue_families_to_create.insert(queue_family_index, vec![1.0 as f32; count as usize]);
    }

    let queue_infos: Vec<_> = queue_families_to_create
        .iter()
        .map(|(&queue_family_index, priorities)| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(priorities)
                .build()
        })
        .collect();

    // Check if timeline semaphore extension was requested
    let timeline_semaphore_ext_name =
        CStr::from_bytes_with_nul(b"VK_KHR_timeline_semaphore\0").unwrap();
    let needs_timeline_semaphore = additional_device_extensions
        .iter()
        .any(|ext| ext.as_c_str() == timeline_semaphore_ext_name);

    let mut timeline_semaphore_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::builder()
        .timeline_semaphore(true)
        .build();

    let mut device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extension_names)
        .enabled_features(&physical_device_features);

    if needs_timeline_semaphore {
        device_create_info = device_create_info.push_next(&mut timeline_semaphore_features);
    }

    let device: ash::Device =
        unsafe { instance.create_device(physical_device, &device_create_info, None)? };

    Ok(device)
}
