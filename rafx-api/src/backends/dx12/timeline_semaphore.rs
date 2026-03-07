use crate::dx12::RafxDeviceContextDx12;
use crate::RafxResult;

use super::d3d12;

/// A timeline semaphore backed by a DX12 fence.
///
/// DX12 `ID3D12Fence` is natively a timeline semaphore: a monotonically increasing
/// u64 counter that can be waited on and signaled from both CPU and GPU at arbitrary
/// values.
pub struct RafxTimelineSemaphoreDx12 {
    _device_context: RafxDeviceContextDx12,
    fence: d3d12::ID3D12Fence1,
    wait_event: windows::Win32::Foundation::HANDLE,
}

impl RafxTimelineSemaphoreDx12 {
    /// Create a new timeline semaphore (non-shared) with the given initial value.
    pub fn new(
        device_context: &RafxDeviceContextDx12,
        initial_value: u64,
    ) -> RafxResult<Self> {
        let fence = unsafe {
            device_context
                .d3d12_device()
                .CreateFence(initial_value, d3d12::D3D12_FENCE_FLAGS::default())?
        };
        let wait_event = unsafe {
            windows::Win32::System::Threading::CreateEventW(None, false, false, None)?
        };
        Ok(Self {
            _device_context: device_context.clone(),
            fence,
            wait_event,
        })
    }

    /// Create a timeline semaphore backed by a shared fence (for cross-process export).
    pub fn new_shared(
        device_context: &RafxDeviceContextDx12,
        initial_value: u64,
    ) -> RafxResult<Self> {
        let fence = unsafe {
            device_context
                .d3d12_device()
                .CreateFence(initial_value, d3d12::D3D12_FENCE_FLAG_SHARED)?
        };
        let wait_event = unsafe {
            windows::Win32::System::Threading::CreateEventW(None, false, false, None)?
        };
        Ok(Self {
            _device_context: device_context.clone(),
            fence,
            wait_event,
        })
    }

    /// Wrap an existing ID3D12Fence1 (e.g. imported from another process).
    pub fn from_existing(
        device_context: &RafxDeviceContextDx12,
        fence: d3d12::ID3D12Fence1,
    ) -> RafxResult<Self> {
        let wait_event = unsafe {
            windows::Win32::System::Threading::CreateEventW(None, false, false, None)?
        };
        Ok(Self {
            _device_context: device_context.clone(),
            fence,
            wait_event,
        })
    }

    pub fn dx12_fence(&self) -> &d3d12::ID3D12Fence1 {
        &self.fence
    }

    /// Query the current counter value (CPU-side).
    pub fn value(&self) -> RafxResult<u64> {
        Ok(unsafe { self.fence.GetCompletedValue() })
    }

    /// CPU-side wait until the semaphore reaches at least `value`.
    pub fn wait(
        &self,
        value: u64,
        timeout_ns: u64,
    ) -> RafxResult<()> {
        if unsafe { self.fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        unsafe {
            windows::Win32::System::Threading::ResetEvent(self.wait_event);
            self.fence
                .SetEventOnCompletion(value, self.wait_event)?;
            let timeout_ms = if timeout_ns == u64::MAX {
                u32::MAX // INFINITE
            } else {
                (timeout_ns / 1_000_000).min(u32::MAX as u64) as u32
            };
            windows::Win32::System::Threading::WaitForSingleObject(self.wait_event, timeout_ms);
        }
        Ok(())
    }

    /// CPU-side signal: set the semaphore to `value`.
    pub fn signal(
        &self,
        value: u64,
    ) -> RafxResult<()> {
        unsafe {
            self.fence.Signal(value)?;
        }
        Ok(())
    }
}

impl Drop for RafxTimelineSemaphoreDx12 {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.wait_event);
        }
    }
}
