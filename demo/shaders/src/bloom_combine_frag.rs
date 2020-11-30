// This code is auto-generated by the shader processor.

#[allow(unused_imports)]
use rafx_resources::ash::prelude::VkResult;

#[allow(unused_imports)]
use rafx_resources::{
    DescriptorSetAllocator, DescriptorSetArc, DescriptorSetInitializer, DynDescriptorSet,
    ImageViewResource, ResourceArc,
};

pub const IN_COLOR_DESCRIPTOR_SET_INDEX: usize = 0;
pub const IN_COLOR_DESCRIPTOR_BINDING_INDEX: usize = 0;
pub const IN_BLUR_DESCRIPTOR_SET_INDEX: usize = 0;
pub const IN_BLUR_DESCRIPTOR_BINDING_INDEX: usize = 1;
pub const SMP_DESCRIPTOR_SET_INDEX: usize = 0;
pub const SMP_DESCRIPTOR_BINDING_INDEX: usize = 2;

pub struct DescriptorSet0Args<'a> {
    pub in_color: &'a ResourceArc<ImageViewResource>,
    pub in_blur: &'a ResourceArc<ImageViewResource>,
}

impl<'a> DescriptorSetInitializer<'a> for DescriptorSet0Args<'a> {
    type Output = DescriptorSet0;

    fn create_dyn_descriptor_set(
        descriptor_set: DynDescriptorSet,
        args: Self,
    ) -> Self::Output {
        let mut descriptor = DescriptorSet0(descriptor_set);
        descriptor.set_args(args);
        descriptor
    }

    fn create_descriptor_set(
        descriptor_set_allocator: &mut DescriptorSetAllocator,
        descriptor_set: DynDescriptorSet,
        args: Self,
    ) -> VkResult<DescriptorSetArc> {
        let mut descriptor = Self::create_dyn_descriptor_set(descriptor_set, args);
        descriptor.0.flush(descriptor_set_allocator)?;
        Ok(descriptor.0.descriptor_set().clone())
    }
}

pub struct DescriptorSet0(pub DynDescriptorSet);

impl DescriptorSet0 {
    pub fn set_args_static(
        descriptor_set: &mut DynDescriptorSet,
        args: DescriptorSet0Args,
    ) {
        descriptor_set.set_image(IN_COLOR_DESCRIPTOR_BINDING_INDEX as u32, args.in_color);
        descriptor_set.set_image(IN_BLUR_DESCRIPTOR_BINDING_INDEX as u32, args.in_blur);
    }

    pub fn set_args(
        &mut self,
        args: DescriptorSet0Args,
    ) {
        self.set_in_color(args.in_color);
        self.set_in_blur(args.in_blur);
    }

    pub fn set_in_color(
        &mut self,
        in_color: &ResourceArc<ImageViewResource>,
    ) {
        self.0
            .set_image(IN_COLOR_DESCRIPTOR_BINDING_INDEX as u32, in_color);
    }

    pub fn set_in_blur(
        &mut self,
        in_blur: &ResourceArc<ImageViewResource>,
    ) {
        self.0
            .set_image(IN_BLUR_DESCRIPTOR_BINDING_INDEX as u32, in_blur);
    }

    pub fn flush(
        &mut self,
        descriptor_set_allocator: &mut DescriptorSetAllocator,
    ) -> VkResult<()> {
        self.0.flush(descriptor_set_allocator)
    }
}
