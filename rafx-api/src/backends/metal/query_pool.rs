use crate::metal::RafxDeviceContextMetal;
use crate::{RafxQueryPoolDef, RafxResult};
use metal_rs::foreign_types::ForeignType;
use metal_rs::{MTLCounterSamplingPoint, MTLResourceOptions, MTLStorageMode, NSRange};
use objc::{msg_send, sel, sel_impl};
use std::sync::{Arc, Mutex};

/// MTLCommonCounterSetTimestamp
const TIMESTAMP_COUNTER_SET: &str = "timestamp";

/// Whether the device can write timestamps the way this backend does: at
/// encoder (stage) boundaries, the only sampling point Apple GPUs offer.
pub(crate) fn query_pool_supports_gpu_timestamps(device: &metal_rs::DeviceRef) -> bool {
    device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary)
        && device
            .counter_sets()
            .iter()
            .any(|set| set.name() == TIMESTAMP_COUNTER_SET)
}

struct RafxQueryPoolMetalInner {
    sample_buffer: metal_rs::CounterSampleBuffer,
    // A word each timestamp's encoder fills, so the encoder has work and is not dropped.
    scratch: metal_rs::Buffer,
    query_count: u32,
    // The query each query's value is read from: itself, or for a timestamp written inside a
    // render pass (sampled once the pass's encoder has ended, RafxCommandBufferMetal), the one
    // sampled for it and the others written in the same pass.
    sources: Mutex<Vec<u32>>,
}

/// Metal implementation of RafxQueryPool (timestamp queries): an
/// MTLCounterSampleBuffer of the timestamp counter set. Apple GPUs sample
/// only at encoder boundaries, so each timestamp is a little blit encoder of
/// its own, sampled when it ends (RafxCommandBufferMetal::cmd_write_timestamp).
/// Reset and resolve need nothing on the GPU timeline.
#[derive(Clone)]
pub struct RafxQueryPoolMetal {
    inner: Arc<RafxQueryPoolMetalInner>,
}

// For the metal_rs objects
unsafe impl Send for RafxQueryPoolMetal {}
unsafe impl Sync for RafxQueryPoolMetal {}

impl std::fmt::Debug for RafxQueryPoolMetal {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        f.debug_struct("RafxQueryPoolMetal")
            .field("query_count", &self.inner.query_count)
            .finish()
    }
}

impl RafxQueryPoolMetal {
    pub fn new(
        device_context: &RafxDeviceContextMetal,
        query_pool_def: &RafxQueryPoolDef,
    ) -> RafxResult<Self> {
        let device = device_context.device();
        let counter_set = device
            .counter_sets()
            .into_iter()
            .find(|set| set.name() == TIMESTAMP_COUNTER_SET)
            .ok_or("the metal device has no timestamp counter set")?;
        let descriptor = metal_rs::CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&counter_set);
        descriptor.set_sample_count(query_pool_def.query_count as u64);
        // Shared: read_timestamps resolves on the CPU.
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let sample_buffer = device.new_counter_sample_buffer_with_descriptor(&descriptor)?;
        let scratch = device.new_buffer(4, MTLResourceOptions::StorageModePrivate);
        Ok(RafxQueryPoolMetal {
            inner: Arc::new(RafxQueryPoolMetalInner {
                sample_buffer,
                scratch,
                query_count: query_pool_def.query_count,
                sources: Mutex::new((0..query_pool_def.query_count).collect()),
            }),
        })
    }

    pub fn query_count(&self) -> u32 {
        self.inner.query_count
    }

    pub fn metal_counter_sample_buffer(&self) -> &metal_rs::CounterSampleBufferRef {
        &self.inner.sample_buffer
    }

    pub(crate) fn scratch(&self) -> &metal_rs::BufferRef {
        &self.inner.scratch
    }

    /// Read `query` from `source`'s sample from now on (until the next reset).
    pub(crate) fn set_source(
        &self,
        query: u32,
        source: u32,
    ) {
        self.inner.sources.lock().unwrap()[query as usize] = source;
    }

    pub(crate) fn reset(&self) {
        for (i, source) in self.inner.sources.lock().unwrap().iter_mut().enumerate() {
            *source = i as u32;
        }
    }

    /// See `RafxQueryPool::read_timestamps`. Ticks are the GPU's timestamp
    /// counter (RafxQueueMetal::timestamp_period_ns).
    pub fn read_timestamps(
        &self,
        first_query: u32,
        results: &mut [u64],
    ) -> RafxResult<()> {
        let sources = self.inner.sources.lock().unwrap();
        let wanted = &sources[first_query as usize..first_query as usize + results.len()];
        // The range of samples behind the wanted queries.
        let (Some(&lo), Some(&hi)) = (wanted.iter().min(), wanted.iter().max()) else {
            return Ok(());
        };
        let ticks = objc::rc::autoreleasepool(|| -> RafxResult<Vec<u64>> {
            unsafe {
                let data: *mut objc::runtime::Object = msg_send![
                    self.inner.sample_buffer.as_ptr(),
                    resolveCounterRange: NSRange::new(lo as u64, (hi - lo + 1) as u64)
                ];
                if data.is_null() {
                    Err("resolving the metal timestamp samples failed")?;
                }
                let bytes: *const u64 = msg_send![data, bytes];
                let length: u64 = msg_send![data, length];
                let len =
                    (length as usize / std::mem::size_of::<u64>()).min((hi - lo + 1) as usize);
                Ok(std::slice::from_raw_parts(bytes, len).to_vec())
            }
        })?;
        for (result, &source) in results.iter_mut().zip(wanted) {
            *result = ticks.get((source - lo) as usize).copied().unwrap_or(0);
        }
        Ok(())
    }
}
