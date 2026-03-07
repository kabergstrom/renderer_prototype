#[cfg(feature = "rafx-dx12")]
use crate::dx12::RafxTimelineSemaphoreDx12;
#[cfg(any(
    feature = "rafx-empty",
    not(any(
        feature = "rafx-dx12",
        feature = "rafx-metal",
        feature = "rafx-vulkan",
        feature = "rafx-gles2",
        feature = "rafx-gles3"
    ))
))]
use crate::empty::RafxTimelineSemaphoreEmpty;
#[cfg(feature = "rafx-vulkan")]
use crate::vulkan::RafxTimelineSemaphoreVulkan;

use crate::RafxResult;

/// A timeline semaphore — a monotonically increasing GPU counter that can be waited on
/// and signaled from both CPU and GPU. Unlike binary semaphores, timeline semaphores carry
/// a u64 value and allow multiple waiters/signalers at different values.
///
/// Timeline semaphores must not be dropped if they are in use by the GPU.
pub enum RafxTimelineSemaphore {
    #[cfg(feature = "rafx-dx12")]
    Dx12(RafxTimelineSemaphoreDx12),
    #[cfg(feature = "rafx-vulkan")]
    Vk(RafxTimelineSemaphoreVulkan),
    #[cfg(any(
        feature = "rafx-empty",
        not(any(
            feature = "rafx-dx12",
            feature = "rafx-metal",
            feature = "rafx-vulkan",
            feature = "rafx-gles2",
            feature = "rafx-gles3"
        ))
    ))]
    Empty(RafxTimelineSemaphoreEmpty),
}

impl RafxTimelineSemaphore {
    /// Get the underlying DX12 API object.
    #[cfg(feature = "rafx-dx12")]
    pub fn dx12_timeline_semaphore(&self) -> Option<&RafxTimelineSemaphoreDx12> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxTimelineSemaphore::Dx12(inner) => Some(inner),
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(_) => None,
            #[cfg(any(
                feature = "rafx-empty",
                not(any(
                    feature = "rafx-dx12",
                    feature = "rafx-metal",
                    feature = "rafx-vulkan",
                    feature = "rafx-gles2",
                    feature = "rafx-gles3"
                ))
            ))]
            RafxTimelineSemaphore::Empty(_) => None,
        }
    }

    /// Get the underlying vulkan API object.
    #[cfg(feature = "rafx-vulkan")]
    pub fn vk_timeline_semaphore(&self) -> Option<&RafxTimelineSemaphoreVulkan> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxTimelineSemaphore::Dx12(_) => None,
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => Some(inner),
            #[cfg(any(
                feature = "rafx-empty",
                not(any(
                    feature = "rafx-dx12",
                    feature = "rafx-metal",
                    feature = "rafx-vulkan",
                    feature = "rafx-gles2",
                    feature = "rafx-gles3"
                ))
            ))]
            RafxTimelineSemaphore::Empty(_) => None,
        }
    }

    /// Query the current value of the timeline semaphore (CPU-side).
    pub fn value(&self) -> RafxResult<u64> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxTimelineSemaphore::Dx12(inner) => inner.value(),
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.value(),
            #[cfg(any(
                feature = "rafx-empty",
                not(any(
                    feature = "rafx-dx12",
                    feature = "rafx-metal",
                    feature = "rafx-vulkan",
                    feature = "rafx-gles2",
                    feature = "rafx-gles3"
                ))
            ))]
            RafxTimelineSemaphore::Empty(inner) => inner.value(),
        }
    }

    /// CPU-side wait until the semaphore reaches at least `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RafxResult<()> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxTimelineSemaphore::Dx12(inner) => inner.wait(value, timeout_ns),
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.wait(value, timeout_ns),
            #[cfg(any(
                feature = "rafx-empty",
                not(any(
                    feature = "rafx-dx12",
                    feature = "rafx-metal",
                    feature = "rafx-vulkan",
                    feature = "rafx-gles2",
                    feature = "rafx-gles3"
                ))
            ))]
            RafxTimelineSemaphore::Empty(inner) => inner.wait(value, timeout_ns),
        }
    }

    /// CPU-side signal: set the semaphore to `value`.
    pub fn signal(&self, value: u64) -> RafxResult<()> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxTimelineSemaphore::Dx12(inner) => inner.signal(value),
            #[cfg(feature = "rafx-vulkan")]
            RafxTimelineSemaphore::Vk(inner) => inner.signal(value),
            #[cfg(any(
                feature = "rafx-empty",
                not(any(
                    feature = "rafx-dx12",
                    feature = "rafx-metal",
                    feature = "rafx-vulkan",
                    feature = "rafx-gles2",
                    feature = "rafx-gles3"
                ))
            ))]
            RafxTimelineSemaphore::Empty(inner) => inner.signal(value),
        }
    }
}
