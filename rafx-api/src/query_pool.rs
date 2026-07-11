#[cfg(feature = "rafx-dx12")]
use crate::dx12::RafxQueryPoolDx12;
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
use crate::empty::RafxQueryPoolEmpty;
#[cfg(feature = "rafx-vulkan")]
use crate::vulkan::RafxQueryPoolVulkan;
use crate::RafxResult;

/// A pool of GPU timestamp queries.
///
/// Timestamps are written on the GPU timeline with
/// `RafxCommandBuffer::cmd_write_timestamp` and read back on the CPU with
/// `read_timestamps`. The intended per-frame cycle is:
///  * `cmd_reset_query_pool` once at the start of the command buffer
///    (outside any render pass)
///  * `cmd_write_timestamp` at each point of interest
///  * `cmd_resolve_query_pool` once after the last write (outside any render
///    pass — required on dx12, no-op on vulkan)
///  * `read_timestamps` on the CPU, but only after the submit has been proven
///    complete by a fence/timeline wait — reading queries the GPU has not
///    written yet is undefined (vulkan blocks, dx12 returns garbage)
/// Convert ticks to nanoseconds with `RafxQueue::timestamp_period_ns`.
///
/// Not supported on all backends — gate on
/// `RafxDeviceInfo::supports_gpu_timestamps`.
pub enum RafxQueryPool {
    #[cfg(feature = "rafx-dx12")]
    Dx12(RafxQueryPoolDx12),
    #[cfg(feature = "rafx-vulkan")]
    Vk(RafxQueryPoolVulkan),
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
    Empty(RafxQueryPoolEmpty),
}

impl RafxQueryPool {
    /// Number of query slots in the pool
    pub fn query_count(&self) -> u32 {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxQueryPool::Dx12(inner) => inner.query_count(),
            #[cfg(feature = "rafx-vulkan")]
            RafxQueryPool::Vk(inner) => inner.query_count(),
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
            RafxQueryPool::Empty(inner) => inner.query_count(),
        }
    }

    /// Read raw GPU tick values for queries
    /// `[first_query .. first_query + results.len())`. Only valid after the
    /// submit that wrote (and on dx12, resolved) them has completed.
    pub fn read_timestamps(
        &self,
        first_query: u32,
        results: &mut [u64],
    ) -> RafxResult<()> {
        match self {
            #[cfg(feature = "rafx-dx12")]
            RafxQueryPool::Dx12(inner) => inner.read_timestamps(first_query, results),
            #[cfg(feature = "rafx-vulkan")]
            RafxQueryPool::Vk(inner) => inner.read_timestamps(first_query, results),
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
            RafxQueryPool::Empty(inner) => inner.read_timestamps(first_query, results),
        }
    }

    /// Get the underlying dx12 API object. This provides access to any internally created
    /// dx12 objects.
    #[cfg(feature = "rafx-dx12")]
    pub fn dx12_query_pool(&self) -> Option<&RafxQueryPoolDx12> {
        #[allow(unreachable_patterns)]
        match self {
            RafxQueryPool::Dx12(inner) => Some(inner),
            _ => None,
        }
    }

    /// Get the underlying vulkan API object. This provides access to any internally created
    /// vulkan objects.
    #[cfg(feature = "rafx-vulkan")]
    pub fn vk_query_pool(&self) -> Option<&RafxQueryPoolVulkan> {
        #[allow(unreachable_patterns)]
        match self {
            RafxQueryPool::Vk(inner) => Some(inner),
            _ => None,
        }
    }

    /// Get the underlying empty API object. This provides access to any internally created
    /// empty objects.
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
    pub fn empty_query_pool(&self) -> Option<&RafxQueryPoolEmpty> {
        #[allow(unreachable_patterns)]
        match self {
            RafxQueryPool::Empty(inner) => Some(inner),
            _ => None,
        }
    }
}
