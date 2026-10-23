use crate::metal::RafxDeviceContextMetal;
use crate::RafxResult;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct RafxSemaphoreMetal {
    _device_context: RafxDeviceContextMetal,

    // Set to true when an operation is scheduled to signal this semaphore
    // Cleared when an operation is scheduled to consume this semaphore
    signal_available: AtomicBool,

    metal_event: Box<MetalEvent>,
}

// metal-rs makes its object types with a proc macro (foreign-types). Held in a
// boxed struct of ours, they leave this type's layout readable to tools that lay
// out types from source without expanding proc macros (rust-analyzer without a
// proc-macro server).
struct MetalEvent(metal_rs::Event);

// for metal_rs::Event
unsafe impl Send for RafxSemaphoreMetal {}
unsafe impl Sync for RafxSemaphoreMetal {}

impl RafxSemaphoreMetal {
    pub fn new(device_context: &RafxDeviceContextMetal) -> RafxResult<RafxSemaphoreMetal> {
        //TODO: Need to add support for new_event() in metal crate

        let metal_event = device_context.device().new_event();

        Ok(RafxSemaphoreMetal {
            _device_context: device_context.clone(),
            metal_event: Box::new(MetalEvent(metal_event)),
            signal_available: AtomicBool::new(false),
        })
    }

    pub fn metal_event(&self) -> &metal_rs::EventRef {
        self.metal_event.0.as_ref()
    }

    pub(crate) fn signal_available(&self) -> bool {
        self.signal_available.load(Ordering::Relaxed)
    }

    pub(crate) fn set_signal_available(
        &self,
        available: bool,
    ) {
        self.signal_available.store(available, Ordering::Relaxed);
    }
}
