use crate::metal::RafxDeviceContextMetal;
use crate::RafxResult;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A timeline semaphore backed by a Metal `MTLSharedEvent`.
///
/// `MTLSharedEvent` carries a monotonically increasing u64 counter that can be
/// signaled and waited on from both CPU and GPU.
pub struct RafxTimelineSemaphoreMetal {
    _device_context: RafxDeviceContextMetal,
    shared_event: metal_rs::SharedEvent,
    /// Shared state for CPU-side wait/notify.
    wait_state: Arc<WaitState>,
    /// Listener must stay alive for the lifetime of registered notifications.
    _listener: metal_rs::SharedEventListener,
}

struct WaitState {
    mu: Mutex<u64>,
    cv: Condvar,
}

// metal_rs::SharedEvent is Send+Sync (Obj-C reference counted).
// SharedEventListener is Send+Sync.
unsafe impl Send for RafxTimelineSemaphoreMetal {}
unsafe impl Sync for RafxTimelineSemaphoreMetal {}

impl RafxTimelineSemaphoreMetal {
    pub fn new(
        device_context: &RafxDeviceContextMetal,
        initial_value: u64,
    ) -> RafxResult<Self> {
        let shared_event = device_context.device().new_shared_event();
        shared_event.set_signaled_value(initial_value);

        let listener = unsafe {
            metal_rs::SharedEventListener::from_queue_handle(
                dispatch::ffi::dispatch_get_main_queue() as *mut _
            )
        };
        let wait_state = Arc::new(WaitState {
            mu: Mutex::new(initial_value),
            cv: Condvar::new(),
        });

        Ok(Self {
            _device_context: device_context.clone(),
            shared_event,
            wait_state,
            _listener: listener,
        })
    }

    pub fn metal_shared_event(&self) -> &metal_rs::SharedEventRef {
        self.shared_event.as_ref()
    }

    /// Query the current value of the timeline semaphore (CPU-side).
    pub fn value(&self) -> RafxResult<u64> {
        Ok(self.shared_event.signaled_value())
    }

    /// CPU-side wait until the semaphore reaches at least `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RafxResult<()> {
        // Fast path: already reached.
        if self.shared_event.signaled_value() >= value {
            return Ok(());
        }

        // Register a notification so the GPU signals us when it reaches `value`.
        let ws = self.wait_state.clone();
        let block = block::ConcreteBlock::new(move |_event: &metal_rs::SharedEventRef, new_val: u64| {
            let mut guard = ws.mu.lock().unwrap();
            if new_val > *guard {
                *guard = new_val;
            }
            ws.cv.notify_all();
        });
        let block = block.copy();

        self.shared_event.notify(
            &self._listener,
            value,
            block,
        );

        // Wait on condvar with timeout.
        let timeout = if timeout_ns == u64::MAX {
            Duration::from_secs(3600) // cap at 1 hour to avoid overflow
        } else {
            Duration::from_nanos(timeout_ns)
        };

        let guard = self.wait_state.mu.lock().unwrap();
        let _guard = self
            .wait_state
            .cv
            .wait_timeout_while(guard, timeout, |current| *current < value)
            .unwrap();

        Ok(())
    }

    /// CPU-side signal: set the semaphore to `value`.
    pub fn signal(&self, value: u64) -> RafxResult<()> {
        self.shared_event.set_signaled_value(value);
        // Also update the condvar state so any CPU waiters wake up.
        let mut guard = self.wait_state.mu.lock().unwrap();
        if value > *guard {
            *guard = value;
        }
        self.wait_state.cv.notify_all();
        Ok(())
    }
}
