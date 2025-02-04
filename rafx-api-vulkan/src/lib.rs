#[macro_use]
extern crate log;

pub use ash;
pub use vk_mem;

use ash::vk;

mod alignment;
use alignment::Align;

pub mod util;

pub mod cleanup;
pub use cleanup::VkResource;
pub use cleanup::VkResourceDropSink;

pub mod pool;
pub use pool::VkDescriptorPoolAllocator;
pub use pool::VkPoolAllocator;
pub use pool::VkPoolResourceImpl;

mod window_support;
pub use window_support::Window;

mod instance;
pub use instance::VkCreateInstanceError;
pub use instance::VkInstance;

mod device;
pub use device::VkCreateDeviceError;
pub use device::VkDevice;
pub use device::VkDeviceContext;
pub use device::VkQueueFamilyIndices;
pub use device::VkQueues;

mod swapchain;
pub use swapchain::SwapchainInfo;
pub use swapchain::VkSwapchain;
pub use swapchain::MAX_FRAMES_IN_FLIGHT;

mod buffer;
pub use buffer::VkBuffer;
pub use buffer::VkBufferRaw;

mod image;
pub use image::VkImage;
pub use image::VkImageRaw;

mod upload;
pub use upload::VkTransferUpload;
pub use upload::VkTransferUploadState;
pub use upload::VkUpload;
pub use upload::VkUploadError;
pub use upload::VkUploadState;

mod debug_reporter;
pub use debug_reporter::VkDebugReporter;

mod context;
pub use context::VkContext;
pub use context::VkContextBuilder;
pub use context::VkCreateContextError;

#[allow(clippy::module_inception)]
mod surface;
pub use surface::FrameInFlight;
pub use surface::VkSurface;
pub use surface::VkSurfaceSwapchainLifetimeListener;

mod coordinates;
use ash::vk::SampleCountFlags;
pub use coordinates::LogicalSize;
pub use coordinates::PhysicalSize;
pub use coordinates::Size;

mod entry;
#[cfg(feature = "static-vulkan")]
pub use entry::MoltenEntry;
pub use entry::VkEntry;

//mod fence;

// If using an SDR format, consider using the swapchain surface format!
pub const DEFAULT_COLOR_FORMATS_SDR: [vk::Format; 1] = [
    vk::Format::R8G8B8A8_SNORM, // 100% coverage with optimal
];

pub const DEFAULT_COLOR_FORMATS_HDR: [vk::Format; 1] = [
    vk::Format::R32G32B32A32_SFLOAT, // 100% coverage with optimal
];

pub const DEFAULT_DEPTH_FORMATS: [vk::Format; 3] = [
    vk::Format::D32_SFLOAT,         // 100% coverage with optimal
    vk::Format::D32_SFLOAT_S8_UINT, // 100% coverage with optimal
    vk::Format::D24_UNORM_S8_UINT,
];

#[derive(Copy, Clone, Debug)]
pub enum VulkanLinkMethod {
    /// Link vulkan dynamically (recommended and default)
    Dynamic,

    /// Mainly for platforms like iOS
    #[cfg(feature = "static-vulkan")]
    Static,
}

impl Default for VulkanLinkMethod {
    fn default() -> Self {
        VulkanLinkMethod::Dynamic
    }
}

/// 1x, 2x, 4x, etc. antialiasing
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MsaaLevel {
    Sample1,
    Sample2,
    Sample4,
    Sample8,
    Sample16,
    Sample32,
    Sample64,
}

impl Default for MsaaLevel {
    fn default() -> Self {
        MsaaLevel::Sample1
    }
}

impl Into<vk::SampleCountFlags> for MsaaLevel {
    fn into(self) -> SampleCountFlags {
        match self {
            MsaaLevel::Sample1 => vk::SampleCountFlags::TYPE_1,
            MsaaLevel::Sample2 => vk::SampleCountFlags::TYPE_2,
            MsaaLevel::Sample4 => vk::SampleCountFlags::TYPE_4,
            MsaaLevel::Sample8 => vk::SampleCountFlags::TYPE_8,
            MsaaLevel::Sample16 => vk::SampleCountFlags::TYPE_16,
            MsaaLevel::Sample32 => vk::SampleCountFlags::TYPE_32,
            MsaaLevel::Sample64 => vk::SampleCountFlags::TYPE_64,
        }
    }
}

/// Used to select which PresentMode is preferred. Some of this is hardware/platform dependent and
/// it's a good idea to read the Vulkan spec.
///
/// `Fifo` is always available on Vulkan devices that comply with the spec and is a good default for
/// many cases.
///
/// Values here match VkPresentModeKHR
#[derive(Copy, Clone, Debug)]
pub enum PresentMode {
    /// (`VK_PRESENT_MODE_IMMEDIATE_KHR`) - No internal buffering, and can result in screen
    /// tearin.
    Immediate = 0,

    /// (`VK_PRESENT_MODE_MAILBOX_KHR`) - This allows rendering as fast as the hardware will
    /// allow, but queues the rendered images in a way that avoids tearing. In other words, if the
    /// hardware renders 10 frames within a single vertical blanking period, the first 9 will be
    /// dropped. This is the best choice for lowest latency where power consumption is not a
    /// concern.
    Mailbox = 1,

    /// (`VK_PRESENT_MODE_FIFO_KHR`) - Default option, guaranteed to be available, and locks
    /// screen draw to vsync. This is a good default choice generally, and more power efficient
    /// than mailbox, but can have higher latency than mailbox.
    Fifo = 2,

    /// (`VK_PRESENT_MODE_FIFO_RELAXED_KHR`) - Similar to Fifo but if rendering is late,
    /// screen tearing can be observed.
    FifoRelaxed = 3,
}

impl PresentMode {
    /// Convert to `vk::PresentModeKHR`
    pub fn to_vk(self) -> vk::PresentModeKHR {
        match self {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        }
    }
}

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
