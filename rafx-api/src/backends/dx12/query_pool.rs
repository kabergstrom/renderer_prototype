use super::d3d12;
use crate::dx12::{RafxBufferDx12, RafxDeviceContextDx12};
use crate::{
    RafxBufferDef, RafxMemoryUsage, RafxQueryPoolDef, RafxQueueType, RafxResourceType, RafxResult,
};

/// DX12 implementation of RafxQueryPool (timestamp queries).
///
/// Query data lives in an ID3D12QueryHeap which the CPU cannot read;
/// cmd_resolve_query_pool copies it into the pool's internal READBACK buffer
/// (created in COPY_DEST state, which ResolveQueryData requires) and MUST be
/// recorded after the last cmd_write_timestamp of the frame or
/// read_timestamps returns stale data. cmd_reset_query_pool is a no-op on
/// this backend — dx12 queries need no reset.
pub struct RafxQueryPoolDx12 {
    query_heap: d3d12::ID3D12QueryHeap,
    readback_buffer: RafxBufferDx12,
    query_count: u32,
}

impl std::fmt::Debug for RafxQueryPoolDx12 {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        f.debug_struct("RafxQueryPoolDx12")
            .field("query_count", &self.query_count)
            .finish()
    }
}

impl RafxQueryPoolDx12 {
    pub fn new(
        device_context: &RafxDeviceContextDx12,
        query_pool_def: &RafxQueryPoolDef,
    ) -> RafxResult<Self> {
        let heap_desc = d3d12::D3D12_QUERY_HEAP_DESC {
            Type: d3d12::D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: query_pool_def.query_count,
            NodeMask: 0,
        };

        let mut query_heap: Option<d3d12::ID3D12QueryHeap> = None;
        unsafe {
            device_context
                .d3d12_device()
                .CreateQueryHeap(&heap_desc, &mut query_heap)?;
        }
        let query_heap = query_heap.unwrap();

        let readback_buffer = RafxBufferDx12::new(
            device_context,
            &RafxBufferDef {
                size: query_pool_def.query_count as u64 * std::mem::size_of::<u64>() as u64,
                alignment: 0,
                memory_usage: RafxMemoryUsage::GpuToCpu,
                queue_type: RafxQueueType::Graphics,
                resource_type: RafxResourceType::BUFFER,
                ..Default::default()
            },
        )?;

        Ok(RafxQueryPoolDx12 {
            query_heap,
            readback_buffer,
            query_count: query_pool_def.query_count,
        })
    }

    pub fn query_heap(&self) -> &d3d12::ID3D12QueryHeap {
        &self.query_heap
    }

    pub fn readback_resource(&self) -> &d3d12::ID3D12Resource {
        self.readback_buffer.dx12_resource()
    }

    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// See `RafxQueryPool::read_timestamps`. Reads the resolved copy in the
    /// readback buffer — only meaningful after the submit containing
    /// cmd_resolve_query_pool has completed.
    pub fn read_timestamps(
        &self,
        first_query: u32,
        results: &mut [u64],
    ) -> RafxResult<()> {
        if first_query as usize + results.len() > self.query_count as usize {
            return Err("read_timestamps range exceeds query pool size")?;
        }

        let mapped = self.readback_buffer.map_buffer()?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.add(first_query as usize * std::mem::size_of::<u64>()) as *const u64,
                results.as_mut_ptr(),
                results.len(),
            );
        }
        self.readback_buffer.unmap_buffer()?;
        Ok(())
    }
}
