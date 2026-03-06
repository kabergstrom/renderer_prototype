use crate::vulkan::RafxDeviceContextVulkan;
use crate::RafxResult;
use ash::vk;

/// A timeline semaphore — monotonically increasing counter that can be waited on
/// and signaled from both CPU and GPU.
pub struct RafxTimelineSemaphoreVulkan {
    device_context: RafxDeviceContextVulkan,
    vk_semaphore: vk::Semaphore,
    /// On macOS, stores the Mach port for cross-process sharing.
    /// Set during create_exportable_timeline_semaphore, read by export_timeline_semaphore_handle.
    #[cfg(target_os = "macos")]
    pub(crate) mach_port: Option<u32>,
}

impl Drop for RafxTimelineSemaphoreVulkan {
    fn drop(&mut self) {
        unsafe {
            self.device_context
                .device()
                .destroy_semaphore(self.vk_semaphore, None);
        }
    }
}

impl RafxTimelineSemaphoreVulkan {
    /// Create a new timeline semaphore with the given initial value.
    pub fn new(
        device_context: &RafxDeviceContextVulkan,
        initial_value: u64,
    ) -> RafxResult<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value)
            .build();

        let create_info = vk::SemaphoreCreateInfo::builder()
            .push_next(&mut type_info)
            .build();

        let vk_semaphore = unsafe {
            device_context
                .device()
                .create_semaphore(&create_info, None)?
        };

        Ok(Self {
            device_context: device_context.clone(),
            vk_semaphore,
            #[cfg(target_os = "macos")]
            mach_port: None,
        })
    }

    /// Wrap an existing VkSemaphore that was created as a timeline semaphore
    /// (e.g. imported from another process).
    ///
    /// # Safety
    /// The caller must ensure `vk_semaphore` is a valid timeline semaphore
    /// that will outlive this wrapper, or that this wrapper takes ownership.
    pub unsafe fn from_existing(
        device_context: &RafxDeviceContextVulkan,
        vk_semaphore: vk::Semaphore,
    ) -> Self {
        Self {
            device_context: device_context.clone(),
            vk_semaphore,
            #[cfg(target_os = "macos")]
            mach_port: None,
        }
    }

    pub fn vk_semaphore(&self) -> vk::Semaphore {
        self.vk_semaphore
    }

    /// Query the current value of the timeline semaphore (CPU-side).
    pub fn value(&self) -> RafxResult<u64> {
        let value = unsafe {
            self.device_context
                .device()
                .get_semaphore_counter_value(self.vk_semaphore)?
        };
        Ok(value)
    }

    /// CPU-side wait until the semaphore reaches at least `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RafxResult<()> {
        let semaphores = [self.vk_semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            self.device_context
                .device()
                .wait_semaphores(&wait_info, timeout_ns)?;
        }
        Ok(())
    }

    /// CPU-side signal: set the semaphore to `value`.
    pub fn signal(&self, value: u64) -> RafxResult<()> {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.vk_semaphore)
            .value(value);
        unsafe {
            self.device_context
                .device()
                .signal_semaphore(&signal_info)?;
        }
        Ok(())
    }
}
