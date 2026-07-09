#[cfg(feature = "rafx-vulkan")]
use ash::vk;
use std::sync::Arc;

pub type RafxResult<T> = Result<T, RafxError>;

/// Generic error that contains all the different kinds of errors that may occur when using the API
#[derive(Debug, Clone)]
pub enum RafxError {
    StringError(String),
    ValidationRequiredButUnavailable,
    IoError(Arc<std::io::Error>),
    #[cfg(feature = "rafx-dx12")]
    WindowsApiError(windows::core::Error),
    #[cfg(feature = "rafx-dx12")]
    HResult(windows::core::HRESULT),
    #[cfg(feature = "rafx-dx12")]
    HassleError(Arc<hassle_rs::HassleError>),
    #[cfg(feature = "rafx-vulkan")]
    VkError(vk::Result),
    #[cfg(feature = "rafx-vulkan")]
    VkLoadingError(Arc<ash::LoadingError>),
    #[cfg(any(feature = "rafx-dx12", feature = "rafx-vulkan",))]
    AllocationError(Arc<gpu_allocator::AllocationError>),
    #[cfg(any(feature = "rafx-gles2", feature = "rafx-gles3"))]
    GlError(u32),
}

impl RafxError {
    /// True when this error means the GPU device is lost/removed/reset —
    /// every GPU object is invalid and the device must be rebuilt. Callers
    /// use this to route into device-lost recovery instead of treating the
    /// error as a transient per-frame failure.
    pub fn is_device_lost(&self) -> bool {
        match self {
            #[cfg(feature = "rafx-vulkan")]
            RafxError::VkError(r) => *r == ash::vk::Result::ERROR_DEVICE_LOST,
            #[cfg(feature = "rafx-dx12")]
            RafxError::WindowsApiError(e) => Self::is_device_lost_hresult(e.code()),
            #[cfg(feature = "rafx-dx12")]
            RafxError::HResult(hr) => Self::is_device_lost_hresult(*hr),
            _ => false,
        }
    }

    #[cfg(feature = "rafx-dx12")]
    fn is_device_lost_hresult(hr: windows::core::HRESULT) -> bool {
        const DXGI_ERROR_DEVICE_HUNG: u32 = 0x887A0006;
        const DXGI_ERROR_DEVICE_REMOVED: u32 = 0x887A0005;
        const DXGI_ERROR_DEVICE_RESET: u32 = 0x887A0007;
        const DXGI_ERROR_DRIVER_INTERNAL_ERROR: u32 = 0x887A0020;
        matches!(
            hr.0 as u32,
            DXGI_ERROR_DEVICE_HUNG
                | DXGI_ERROR_DEVICE_REMOVED
                | DXGI_ERROR_DEVICE_RESET
                | DXGI_ERROR_DRIVER_INTERNAL_ERROR
        )
    }

    /// An error that classifies as device-lost on the active backend, for
    /// fault-injection testing. Uses the backend's genuine error type so the
    /// synthetic path flows through the same classification as a real loss.
    #[allow(unreachable_code)]
    pub fn synthetic_device_lost() -> Self {
        #[cfg(feature = "rafx-vulkan")]
        return RafxError::VkError(ash::vk::Result::ERROR_DEVICE_LOST);
        #[cfg(feature = "rafx-dx12")]
        return RafxError::HResult(windows::core::HRESULT(0x887A0005u32 as i32));
        RafxError::StringError("synthetic device-lost (backend has no device-lost error)".into())
    }
}

impl std::error::Error for RafxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            RafxError::StringError(_) => None,
            RafxError::ValidationRequiredButUnavailable => None,
            RafxError::IoError(ref e) => Some(&**e),

            #[cfg(feature = "rafx-dx12")]
            RafxError::WindowsApiError(ref e) => Some(e),
            #[cfg(feature = "rafx-dx12")]
            RafxError::HResult(ref e) => None,
            #[cfg(feature = "rafx-dx12")]
            RafxError::HassleError(ref e) => Some(e),
            #[cfg(feature = "rafx-vulkan")]
            RafxError::VkError(ref e) => Some(e),
            #[cfg(feature = "rafx-vulkan")]
            RafxError::VkLoadingError(ref e) => Some(&**e),
            #[cfg(any(feature = "rafx-dx12", feature = "rafx-vulkan",))]
            RafxError::AllocationError(ref e) => Some(&**e),
            #[cfg(any(feature = "rafx-gles2", feature = "rafx-gles3"))]
            RafxError::GlError(_) => None,
        }
    }
}

impl core::fmt::Display for RafxError {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        match *self {
            RafxError::StringError(ref e) => e.fmt(fmt),
            RafxError::ValidationRequiredButUnavailable => {
                "ValidationRequiredButUnavailable".fmt(fmt)
            }
            RafxError::IoError(ref e) => e.fmt(fmt),
            #[cfg(feature = "rafx-dx12")]
            RafxError::WindowsApiError(ref e) => e.fmt(fmt),
            #[cfg(feature = "rafx-dx12")]
            RafxError::HResult(ref e) => e.fmt(fmt),
            #[cfg(feature = "rafx-dx12")]
            RafxError::HassleError(ref e) => e.fmt(fmt),
            #[cfg(feature = "rafx-vulkan")]
            RafxError::VkError(ref e) => e.fmt(fmt),
            #[cfg(feature = "rafx-vulkan")]
            RafxError::VkLoadingError(ref e) => e.fmt(fmt),
            #[cfg(any(feature = "rafx-dx12", feature = "rafx-vulkan",))]
            RafxError::AllocationError(ref e) => e.fmt(fmt),
            #[cfg(any(feature = "rafx-gles2", feature = "rafx-gles3"))]
            RafxError::GlError(ref e) => e.fmt(fmt),
        }
    }
}

impl From<&str> for RafxError {
    fn from(str: &str) -> Self {
        RafxError::StringError(str.to_string())
    }
}

impl From<String> for RafxError {
    fn from(string: String) -> Self {
        RafxError::StringError(string)
    }
}

impl From<std::io::Error> for RafxError {
    fn from(error: std::io::Error) -> Self {
        RafxError::IoError(Arc::new(error))
    }
}

#[cfg(feature = "rafx-dx12")]
impl From<windows::core::Error> for RafxError {
    fn from(result: windows::core::Error) -> Self {
        RafxError::WindowsApiError(result)
    }
}

#[cfg(feature = "rafx-dx12")]
impl From<windows::core::HRESULT> for RafxError {
    fn from(result: windows::core::HRESULT) -> Self {
        RafxError::HResult(result)
    }
}

#[cfg(feature = "rafx-dx12")]
impl From<hassle_rs::HassleError> for RafxError {
    fn from(result: hassle_rs::HassleError) -> Self {
        RafxError::HassleError(Arc::new(result))
    }
}

#[cfg(feature = "rafx-vulkan")]
impl From<vk::Result> for RafxError {
    fn from(result: vk::Result) -> Self {
        RafxError::VkError(result)
    }
}

#[cfg(feature = "rafx-vulkan")]
impl From<ash::LoadingError> for RafxError {
    fn from(result: ash::LoadingError) -> Self {
        RafxError::VkLoadingError(Arc::new(result))
    }
}

#[cfg(any(feature = "rafx-dx12", feature = "rafx-vulkan",))]
impl From<gpu_allocator::AllocationError> for RafxError {
    fn from(error: gpu_allocator::AllocationError) -> Self {
        RafxError::AllocationError(Arc::new(error))
    }
}
