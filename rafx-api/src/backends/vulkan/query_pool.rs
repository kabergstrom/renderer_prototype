use crate::vulkan::RafxDeviceContextVulkan;
use crate::{RafxQueryPoolDef, RafxResult};
use ash::vk;

/// Vulkan implementation of RafxQueryPool (timestamp queries).
///
/// Results are fetched CPU-side via vkGetQueryPoolResults, so
/// cmd_resolve_query_pool is a no-op on this backend. Queries must be reset
/// on the GPU timeline (cmd_reset_query_pool, outside a render pass) before
/// each use — including the first.
pub struct RafxQueryPoolVulkan {
    device_context: RafxDeviceContextVulkan,
    vk_query_pool: vk::QueryPool,
    query_count: u32,
}

impl std::fmt::Debug for RafxQueryPoolVulkan {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        f.debug_struct("RafxQueryPoolVulkan")
            .field("vk_query_pool", &self.vk_query_pool)
            .field("query_count", &self.query_count)
            .finish()
    }
}

impl RafxQueryPoolVulkan {
    pub fn new(
        device_context: &RafxDeviceContextVulkan,
        query_pool_def: &RafxQueryPoolDef,
    ) -> RafxResult<Self> {
        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_pool_def.query_count);

        let vk_query_pool = unsafe {
            device_context
                .device()
                .create_query_pool(&create_info, None)?
        };

        Ok(RafxQueryPoolVulkan {
            device_context: device_context.clone(),
            vk_query_pool,
            query_count: query_pool_def.query_count,
        })
    }

    pub fn vk_query_pool(&self) -> vk::QueryPool {
        self.vk_query_pool
    }

    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// See `RafxQueryPool::read_timestamps`. WAIT is passed, but because the
    /// caller must only read queries whose submit already completed, it never
    /// actually blocks — reading a query that was reset but never written
    /// WOULD block forever, so don't.
    pub fn read_timestamps(
        &self,
        first_query: u32,
        results: &mut [u64],
    ) -> RafxResult<()> {
        unsafe {
            self.device_context.device().get_query_pool_results(
                self.vk_query_pool,
                first_query,
                results.len() as u32,
                results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(())
    }
}

impl Drop for RafxQueryPoolVulkan {
    fn drop(&mut self) {
        unsafe {
            self.device_context
                .device()
                .destroy_query_pool(self.vk_query_pool, None);
        }
    }
}
