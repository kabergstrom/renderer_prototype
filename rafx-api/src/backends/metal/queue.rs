use crate::metal::{
    BarrierFlagsMetal, RafxCommandBufferMetal, RafxCommandPoolMetal, RafxDeviceContextMetal,
    RafxFenceMetal, RafxSemaphoreMetal, RafxSwapchainMetal, RafxTimelineSemaphoreMetal,
};
use crate::{RafxCommandPoolDef, RafxPresentSuccessResult, RafxQueueType, RafxResult};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

static NEXT_QUEUE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug)]
pub struct RafxQueueMetalInner {
    device_context: RafxDeviceContextMetal,
    queue_type: RafxQueueType,
    queue: metal_rs::CommandQueue,
    queue_id: u32,
    barrier_flags: AtomicU8,
    fence: metal_rs::Fence,
}

// for metal_rs::CommandQueue
unsafe impl Send for RafxQueueMetalInner {}
unsafe impl Sync for RafxQueueMetalInner {}

#[derive(Clone, Debug)]
pub struct RafxQueueMetal {
    inner: Arc<RafxQueueMetalInner>,
}

impl RafxQueueMetal {
    pub fn queue_id(&self) -> u32 {
        self.inner.queue_id
    }

    //
    // These barrier flag helpers are not meant to be threadsafe, the Rafx API assumes command
    // buffers and the queues they come from are not concurrently accessed. (Even if we were careful
    // about thread-safety here, underlying GPU APIs will often produce undefined behavior if this
    // occurs.)
    //
    pub fn barrier_flags(&self) -> BarrierFlagsMetal {
        BarrierFlagsMetal::from_bits(self.inner.barrier_flags.load(Ordering::Relaxed)).unwrap()
    }

    // Get the fence used internally by the currently recording command buffer
    pub fn metal_fence(&self) -> &metal_rs::FenceRef {
        self.inner.fence.as_ref()
    }

    pub fn add_barrier_flags(
        &self,
        flags: BarrierFlagsMetal,
    ) {
        self.inner
            .barrier_flags
            .fetch_or(flags.bits(), Ordering::Relaxed);
    }

    pub fn clear_barrier_flags(&self) {
        self.inner.barrier_flags.store(0, Ordering::Relaxed);
    }

    pub fn queue_type(&self) -> RafxQueueType {
        self.inner.queue_type
    }

    pub fn device_context(&self) -> &RafxDeviceContextMetal {
        &self.inner.device_context
    }

    pub fn metal_queue(&self) -> &metal_rs::CommandQueueRef {
        self.inner.queue.as_ref()
    }

    /// Nanoseconds per GPU timestamp tick (RafxQueryPoolMetal), measured once against the CPU
    /// clock over a few milliseconds: Metal reports no period, and the GPU's counter need not
    /// count nanoseconds.
    pub fn timestamp_period_ns(&self) -> RafxResult<f32> {
        static PERIOD_NS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let device = self.inner.device_context.device();
        let sample = || {
            let (mut cpu, mut gpu) = (0u64, 0u64);
            device.sample_timestamps(&mut cpu, &mut gpu);
            (std::time::Instant::now(), gpu)
        };
        let period = *PERIOD_NS.get_or_init(|| {
            let (t0, g0) = sample();
            std::thread::sleep(std::time::Duration::from_millis(20));
            let (t1, g1) = sample();
            let ticks = g1.saturating_sub(g0);
            if ticks == 0 {
                return 0.0;
            }
            (t1 - t0).as_nanos() as f64 as f32 / ticks as f32
        });
        if period > 0.0 {
            Ok(period)
        } else {
            Err("the metal GPU timestamp counter did not advance")?
        }
    }

    pub fn create_command_pool(
        &self,
        command_pool_def: &RafxCommandPoolDef,
    ) -> RafxResult<RafxCommandPoolMetal> {
        RafxCommandPoolMetal::new(&self, command_pool_def)
    }

    pub fn new(
        device_context: &RafxDeviceContextMetal,
        queue_type: RafxQueueType,
    ) -> RafxResult<RafxQueueMetal> {
        let queue = device_context.device().new_command_queue();
        let fence = device_context.device().new_fence();

        let queue_id = NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed);
        let inner = RafxQueueMetalInner {
            device_context: device_context.clone(),
            queue_type,
            queue,
            queue_id,
            barrier_flags: AtomicU8::default(),
            fence,
        };

        Ok(RafxQueueMetal {
            inner: Arc::new(inner),
        })
    }

    pub fn wait_for_queue_idle(&self) -> RafxResult<()> {
        let wait = self
            .inner
            .queue
            .new_command_buffer_with_unretained_references();
        wait.commit();
        wait.wait_until_completed();
        Ok(())
    }

    fn submit_semaphore_wait(
        &self,
        wait_semaphores: &[&RafxSemaphoreMetal],
        wait_timeline: &[(&RafxTimelineSemaphoreMetal, u64)],
    ) {
        let wait_command_buffer_required =
            !wait_timeline.is_empty() || wait_semaphores.iter().any(|x| x.signal_available());

        if wait_command_buffer_required {
            let wait_command_buffer = self
                .inner
                .queue
                .new_command_buffer_with_unretained_references();
            for wait_semaphore in wait_semaphores {
                if wait_semaphore.signal_available() {
                    wait_command_buffer.encode_wait_for_event(wait_semaphore.metal_event(), 1);
                    wait_semaphore.set_signal_available(false);
                }
            }
            for (semaphore, value) in wait_timeline {
                wait_command_buffer.encode_wait_for_event(semaphore.metal_shared_event(), *value);
            }

            wait_command_buffer.commit();
        }
    }

    pub fn submit(
        &self,
        command_buffers: &[&RafxCommandBufferMetal],
        wait_semaphores: &[&RafxSemaphoreMetal],
        signal_semaphores: &[&RafxSemaphoreMetal],
        signal_fence: Option<&RafxFenceMetal>,
    ) -> RafxResult<()> {
        assert!(!command_buffers.is_empty());
        self.submit_with_timeline(
            command_buffers,
            wait_semaphores,
            signal_semaphores,
            &[],
            &[],
            signal_fence,
        )
    }

    /// Submit with timeline semaphore (`MTLSharedEvent`) waits and signals.
    ///
    /// - Waits are encoded on a command buffer committed ahead of the submitted ones: an event
    ///   wait holds back the work the queue schedules after it.
    /// - Signals are encoded on the last command buffer once its encoders have ended (Metal
    ///   refuses event commands while an encoder is open). With no command buffers, an empty
    ///   one carries the signals and the fence.
    pub fn submit_with_timeline(
        &self,
        command_buffers: &[&RafxCommandBufferMetal],
        wait_semaphores: &[&RafxSemaphoreMetal],
        signal_semaphores: &[&RafxSemaphoreMetal],
        wait_timeline: &[(&RafxTimelineSemaphoreMetal, u64)],
        signal_timeline: &[(&RafxTimelineSemaphoreMetal, u64)],
        signal_fence: Option<&RafxFenceMetal>,
    ) -> RafxResult<()> {
        objc::rc::autoreleasepool(|| {
            self.submit_semaphore_wait(wait_semaphores, wait_timeline);

            for command_buffer in command_buffers {
                command_buffer.end_current_encoders(false)?;
            }

            let empty_command_buffer = if command_buffers.is_empty() {
                Some(
                    self.inner
                        .queue
                        .new_command_buffer_with_unretained_references()
                        .to_owned(),
                )
            } else {
                None
            };
            let metal_command_buffers: Vec<&metal_rs::CommandBufferRef> =
                match &empty_command_buffer {
                    Some(command_buffer) => vec![command_buffer.as_ref()],
                    None => command_buffers
                        .iter()
                        .map(|x| x.metal_command_buffer().unwrap())
                        .collect(),
                };
            let last = *metal_command_buffers.last().unwrap();

            // If a signal fence exists, mark it as submitted and add a closure to execute at the end
            // of each command buffer. The closure will have a shared atomic counter that is incremented
            // each time it is executed. When the counter reaches the number of command buffers, we know
            // that all command buffers are finished executing and will signal the semaphore
            if let Some(signal_fence) = signal_fence {
                signal_fence.set_submitted(true);

                let command_count = metal_command_buffers.len();
                let complete_count = Arc::new(AtomicUsize::new(0));
                let dispatch_semaphore = signal_fence.metal_dispatch_semaphore().clone();
                let block = block::ConcreteBlock::new(move |_command_buffer_ref| {
                    // Add 1 because fetch_add returns the value from before the add
                    let complete =
                        complete_count.fetch_add(1, Ordering::Relaxed) + 1 == command_count;
                    if complete {
                        dispatch_semaphore.signal();
                    }
                })
                .copy();

                for command_buffer in &metal_command_buffers {
                    command_buffer.add_completed_handler(&block);
                }
            }

            // A command buffer that fails (a GPU fault or timeout) skips the rest of its
            // commands, its event signals included, and nothing else reports it.
            let report_error =
                block::ConcreteBlock::new(|command_buffer: &metal_rs::CommandBufferRef| {
                    if command_buffer.status() == metal_rs::MTLCommandBufferStatus::Error {
                        log::error!(
                            "Metal command buffer failed: {}",
                            command_buffer_error(command_buffer)
                        );
                    }
                })
                .copy();
            for command_buffer in &metal_command_buffers {
                command_buffer.add_completed_handler(&report_error);
            }

            for signal_semaphore in signal_semaphores {
                last.encode_signal_event(signal_semaphore.metal_event(), 1);
                signal_semaphore.set_signal_available(true);
            }
            for (semaphore, value) in signal_timeline {
                last.encode_signal_event(semaphore.metal_shared_event(), *value);
            }

            for command_buffer in &metal_command_buffers {
                command_buffer.commit();
            }
            for command_buffer in command_buffers {
                command_buffer.clear_command_buffer();
            }

            Ok(())
        })
    }

    pub fn present(
        &self,
        swapchain: &RafxSwapchainMetal,
        wait_semaphores: &[&RafxSemaphoreMetal],
        _image_index: u32,
    ) -> RafxResult<RafxPresentSuccessResult> {
        objc::rc::autoreleasepool(|| {
            self.submit_semaphore_wait(wait_semaphores, &[]);

            let command_buffer = self.inner.queue.new_command_buffer();
            let drawable = swapchain.take_drawable().unwrap();
            command_buffer.present_drawable(drawable.as_ref());
            // Invalidate swapchain texture in some way?
            command_buffer.commit();

            Ok(RafxPresentSuccessResult::Success)
        })
    }
}

/// The `NSError` description of a failed command buffer.
fn command_buffer_error(command_buffer: &metal_rs::CommandBufferRef) -> String {
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};
    unsafe {
        let error: *mut Object = msg_send![command_buffer, error];
        if error.is_null() {
            return "no error object".to_string();
        }
        let description: *mut Object = msg_send![error, localizedDescription];
        let mut message = ns_string(description);
        // With MTLCommandBufferErrorOptionEncoderExecutionStatus: each encoder's label, debug
        // groups and state (0 unknown, 1 completed, 2 affected, 3 pending, 4 faulted).
        let user_info: *mut Object = msg_send![error, userInfo];
        if !user_info.is_null() {
            let key = ns_string_from_str("MTLCommandBufferEncoderInfoErrorKey");
            let infos: *mut Object = msg_send![user_info, objectForKey: key];
            if !infos.is_null() {
                let count: usize = msg_send![infos, count];
                for i in 0..count {
                    let info: *mut Object = msg_send![infos, objectAtIndex: i];
                    let state: isize = msg_send![info, errorState];
                    if state == 1 {
                        continue;
                    }
                    let label: *mut Object = msg_send![info, label];
                    let signposts: *mut Object = msg_send![info, debugSignposts];
                    let signpost_count: usize = msg_send![signposts, count];
                    let signposts: Vec<String> = (0..signpost_count)
                        .map(|j| {
                            let signpost: *mut Object = msg_send![signposts, objectAtIndex: j];
                            ns_string(signpost)
                        })
                        .collect();
                    message.push_str(&format!(
                        "\n  encoder {:?} state {} groups {:?}",
                        ns_string(label),
                        state,
                        signposts
                    ));
                }
            }
        }
        message
    }
}

unsafe fn ns_string(string: *mut objc::runtime::Object) -> String {
    use objc::{msg_send, sel, sel_impl};
    if string.is_null() {
        return String::new();
    }
    let text: *const std::os::raw::c_char = msg_send![string, UTF8String];
    if text.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(text)
        .to_string_lossy()
        .into_owned()
}

unsafe fn ns_string_from_str(string: &str) -> *mut objc::runtime::Object {
    use objc::{class, msg_send, sel, sel_impl};
    let text = std::ffi::CString::new(string).unwrap();
    msg_send![class!(NSString), stringWithUTF8String: text.as_ptr()]
}
