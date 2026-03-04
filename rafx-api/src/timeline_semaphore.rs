#[cfg(feature = "rafx-vulkan")]
use crate::vulkan::RafxTimelineSemaphoreVulkan;

use crate::RafxResult;

/// A timeline semaphore — a monotonically increasing GPU counter that can be waited on
/// and signaled from both CPU and GPU. Unlike binary semaphores, timeline semaphores carry
/// a u64 value and allow multiple waiters/signalers at different values.
///
/// Timeline semaphores must not be dropped if they are in use by the GPU.
pub enum RafxTimelineSemaphore {
    #[cfg(feature = "rafx-vulkan")]
    Vk(RafxTimelineSemaphoreVulkan),
    // Other backends would go here. DX12 fences are inherently timeline semaphores.
    // Metal uses MTLSharedEvent.
}

impl RafxTimelineSemaphore {
    /// Get the underlying vulkan API object.
    #[cfg(feature = "rafx-vulkan")]
    pub fn vk_timeline_semaphore(&self) -> Option<&RafxTimelineSemaphoreVulkan> {
        match self {
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => Some(inner),
        }
    }

    /// Query the current value of the timeline semaphore (CPU-side).
    pub fn value(&self) -> RafxResult<u64> {
        match self {
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.value(),
        }
    }

    /// CPU-side wait until the semaphore reaches at least `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RafxResult<()> {
        match self {
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.wait(value, timeout_ns),
        }
    }

    /// CPU-side signal: set the semaphore to `value`.
    pub fn signal(&self, value: u64) -> RafxResult<()> {
        match self {
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.signal(value),
        }
    }
}
