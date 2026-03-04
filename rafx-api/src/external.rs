/// Platform-specific handle for cross-process texture sharing.
#[derive(Debug)]
pub enum RafxExternalTextureHandle {
    /// Linux: opaque fd from VK_KHR_external_memory_fd
    Fd(i32),
    /// macOS: IOSurface global ID (from IOSurfaceGetID)
    IOSurfaceId(u32),
}

/// Platform-specific handle for cross-process timeline semaphore sharing.
#[derive(Debug)]
pub enum RafxExternalSemaphoreHandle {
    /// Linux: opaque fd from VK_KHR_external_semaphore_fd
    Fd(i32),
    /// macOS: Mach port from MTLSharedEvent.machPort
    MachPort(u32),
}
