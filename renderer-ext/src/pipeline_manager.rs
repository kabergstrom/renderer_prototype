
use crate::pipeline_description as dsc;
use fnv::FnvHashMap;
use ash::vk;
use ash::version::DeviceV1_0;
use renderer_shell_vulkan::VkDeviceContext;
use ash::prelude::VkResult;
use std::collections::hash_map::Entry::Occupied;
use ash::vk::PipelineDynamicStateCreateInfo;

struct DescriptorSetLayoutState {
    vk_obj: vk::DescriptorSetLayout
}

struct PipelineLayoutState {
    vk_obj: vk::PipelineLayout
}

struct RenderPassState {
    vk_obj: vk::RenderPass
}

struct ShaderModuleState {
    vk_obj: vk::ShaderModule
}

struct GraphicsPipelineState {
    vk_obj: vk::Pipeline
}

struct PipelineManager {
    device_context: VkDeviceContext,
    descriptor_set_layouts: FnvHashMap<dsc::DescriptorSetLayout, DescriptorSetLayoutState>,
    pipeline_layouts: FnvHashMap<dsc::PipelineLayout, PipelineLayoutState>,
    renderpasses: FnvHashMap<dsc::RenderPass, RenderPassState>,
    shader_modules: FnvHashMap<dsc::ShaderModule, ShaderModuleState>,
    graphics_pipelines: FnvHashMap<dsc::GraphicsPipeline, GraphicsPipelineState>,
    swapchain_surface_info: dsc::SwapchainSurfaceInfo,
}

impl PipelineManager {
    fn new(device_context: &VkDeviceContext, swapchain_surface_info: dsc::SwapchainSurfaceInfo) -> Self {
        PipelineManager {
            device_context: device_context.clone(),
            descriptor_set_layouts: Default::default(),
            pipeline_layouts: Default::default(),
            renderpasses: Default::default(),
            shader_modules: Default::default(),
            graphics_pipelines: Default::default(),
            swapchain_surface_info
        }
    }

    pub fn get_or_create_descriptor_set_layout(
        &mut self,
        descriptor_set_layout: &dsc::DescriptorSetLayout
    ) -> VkResult<vk::DescriptorSetLayout> {
        let entry = self.descriptor_set_layouts
            .entry(descriptor_set_layout.clone());

        if let Occupied(entry) = entry {
            Ok(entry.get().vk_obj)
        } else {
            let bindings : Vec<_> = descriptor_set_layout.descriptor_set_layout_bindings.iter()
                .map(|binding| binding.as_builder().build())
                .collect();

            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings);

            let vk_obj = unsafe {
                self.device_context.device().create_descriptor_set_layout(&*create_info, None)?
            };

            entry.or_insert(DescriptorSetLayoutState {
                vk_obj
            });
            Ok(vk_obj)
        }
    }

    pub fn get_or_create_pipeline_layout(
        &mut self,
        pipeline_layout: &dsc::PipelineLayout
    ) -> VkResult<vk::PipelineLayout> {
        if let Some(pipeline_layout_state) = self.pipeline_layouts.get(pipeline_layout) {
            Ok(pipeline_layout_state.vk_obj)
        } else {
            let mut descriptor_set_layouts = Vec::with_capacity(pipeline_layout.descriptor_set_layouts.len());
            for descriptor_set_layout in &pipeline_layout.descriptor_set_layouts {
                descriptor_set_layouts.push(self.get_or_create_descriptor_set_layout(descriptor_set_layout)?);
            }

            let push_constant_ranges : Vec<_> = pipeline_layout.push_constant_ranges.iter()
                .map(|push_constant_range| push_constant_range.as_builder().build())
                .collect();

            let create_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(descriptor_set_layouts.as_slice())
                .push_constant_ranges(push_constant_ranges.as_slice());

            let vk_obj = unsafe {
                self.device_context.device().create_pipeline_layout(&*create_info, None)?
            };

            self.pipeline_layouts.insert(pipeline_layout.clone(), PipelineLayoutState {
                vk_obj
            });
            Ok(vk_obj)
        }
    }

    pub fn get_or_create_renderpass(
        &mut self,
        renderpass: &dsc::RenderPass,
    ) -> VkResult<vk::RenderPass> {
        if let Some(renderpass) = self.renderpasses.get(renderpass) {
            Ok(renderpass.vk_obj)
        } else {
            let attachments : Vec<_> = renderpass.attachments.iter()
                .map(|attachment| attachment.as_builder(&self.swapchain_surface_info).build())
                .collect();

            let mut color_attachments : Vec<Vec<vk::AttachmentReference>> = Vec::with_capacity(renderpass.subpasses.len());
            let mut input_attachments : Vec<Vec<vk::AttachmentReference>> = Vec::with_capacity(renderpass.subpasses.len());
            let mut resolve_attachments : Vec<Vec<vk::AttachmentReference>> = Vec::with_capacity(renderpass.subpasses.len());
            let mut depth_stencil_attachments : Vec<vk::AttachmentReference> = Vec::with_capacity(renderpass.subpasses.len());
            let mut subpasses : Vec<_> = Vec::with_capacity(renderpass.subpasses.len());

            for subpass in &renderpass.subpasses {
                color_attachments.push(subpass.color_attachments.iter().map(|attachment| attachment.as_builder().build()).collect());
                input_attachments.push(subpass.input_attachments.iter().map(|attachment| attachment.as_builder().build()).collect());
                resolve_attachments.push(subpass.resolve_attachments.iter().map(|attachment| attachment.as_builder().build()).collect());
                depth_stencil_attachments.push(subpass.depth_stencil_attachment.as_builder().build());

                let subpass_description = vk::SubpassDescription::builder()
                    .pipeline_bind_point(subpass.pipeline_bind_point.into())
                    .color_attachments(color_attachments.last().unwrap())
                    .input_attachments(input_attachments.last().unwrap())
                    .resolve_attachments(resolve_attachments.last().unwrap())
                    .depth_stencil_attachment(depth_stencil_attachments.last().unwrap())
                    .build();

                subpasses.push(subpass_description);
            }

            let dependencies : Vec<_> = renderpass.dependencies.iter()
                .map(|dependency| dependency.as_builder().build())
                .collect();

            let create_info = vk::RenderPassCreateInfo::builder()
                .attachments(attachments.as_slice())
                .subpasses(subpasses.as_slice())
                .dependencies(dependencies.as_slice());

            let vk_obj = unsafe {
                self.device_context.device().create_render_pass(&*create_info, None)?
            };

            self.renderpasses.insert(renderpass.clone(), RenderPassState {
                vk_obj
            });
            Ok(vk_obj)
        }
    }

    pub fn get_or_create_shader_module(
        &mut self,
        shader_module: &dsc::ShaderModule
    ) -> VkResult<vk::ShaderModule> {
        if let Some(shader_module) = self.shader_modules.get(shader_module) {
            Ok(shader_module.vk_obj)
        } else {
            let shader_info = vk::ShaderModuleCreateInfo::builder()
                .code(&shader_module.code);

            unsafe {
                self.device_context.device().create_shader_module(&shader_info, None)
            }
        }
    }

    pub fn get_or_create_graphics_pipeline(
        &mut self,
        graphics_pipeline: &dsc::GraphicsPipeline,
    ) -> VkResult<vk::Pipeline> {
        if let Some(pipeline) = self.graphics_pipelines.get(graphics_pipeline) {
            Ok(pipeline.vk_obj)
        } else {
            let pipeline_layout = self.get_or_create_pipeline_layout(&graphics_pipeline.pipeline_layout)?;
            let renderpass = self.get_or_create_renderpass(&graphics_pipeline.renderpass)?;
            let fixed_function_state = &graphics_pipeline.fixed_function_state;

            let input_assembly_state = fixed_function_state.input_assembly_state.as_builder().build();

            let mut vertex_input_attribute_descriptions : Vec<_> = fixed_function_state.vertex_input_state.attribute_descriptions.iter()
                .map(|attribute| attribute.as_builder(&self.swapchain_surface_info).build())
                .collect();

            let mut vertex_input_binding_descriptions : Vec<_> = fixed_function_state.vertex_input_state.binding_descriptions.iter()
                .map(|binding| binding.as_builder().build())
                .collect();

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(vertex_input_attribute_descriptions.as_slice())
                .vertex_binding_descriptions(&vertex_input_binding_descriptions);

            let scissors : Vec<_> = fixed_function_state.viewport_state.scissors.iter()
                .map(|scissors| scissors.to_rect2d(&self.swapchain_surface_info))
                .collect();

            let viewports : Vec<_> = fixed_function_state.viewport_state.viewports.iter()
                .map(|viewport| viewport.as_builder(&self.swapchain_surface_info).build())
                .collect();

            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .scissors(&scissors)
                .viewports(&viewports);

            let rasterization_state = fixed_function_state.rasterization_state.as_builder();

            let multisample_state = fixed_function_state.multisample_state.as_builder();

            let color_blend_attachments : Vec<_> = fixed_function_state.color_blend_state.attachments.iter().map(|attachment| attachment.as_builder().build()).collect();
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(fixed_function_state.color_blend_state.logic_op.into())
                .logic_op_enable(fixed_function_state.color_blend_state.logic_op_enable)
                .blend_constants(fixed_function_state.color_blend_state.blend_constants_as_f32())
                .attachments(&color_blend_attachments);

            let dynamic_states : Vec<vk::DynamicState> = fixed_function_state.dynamic_state.dynamic_states.iter().map(|dynamic_state| dynamic_state.clone().into()).collect();
            let dynamic_state = PipelineDynamicStateCreateInfo::builder()
                .dynamic_states(&dynamic_states);


            let mut stages = Vec::with_capacity(graphics_pipeline.pipeline_shader_stages.stages.len());
            for pipeline_shader_stage in &graphics_pipeline.pipeline_shader_stages.stages {
                let module = self.get_or_create_shader_module(&pipeline_shader_stage.shader_module)?;
                stages.push(vk::PipelineShaderStageCreateInfo::builder()
                    .stage(pipeline_shader_stage.stage.into())
                    .module(module)
                    .name(&pipeline_shader_stage.entry_name)
                    .build());
            }

            let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
                .input_assembly_state(&input_assembly_state)
                .vertex_input_state(&vertex_input_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .layout(pipeline_layout)
                .render_pass(renderpass)
                .stages(&stages)
                .build();

            let vk_obj = unsafe {
                match self.device_context.device().create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                ) {
                    Ok(result) => Ok(result[0]),
                    Err(e) => Err(e.1),
                }?
            };

            self.graphics_pipelines.insert(graphics_pipeline.clone(), GraphicsPipelineState {
                vk_obj
            });

            Ok(vk_obj)
        }
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        unsafe {
            for (dsc, state) in &self.graphics_pipelines {
                self.device_context.device().destroy_pipeline(state.vk_obj, None);
            }
            self.graphics_pipelines.clear();

            for (dsc, state) in &self.shader_modules {
                self.device_context.device().destroy_shader_module(state.vk_obj, None);
            }
            self.shader_modules.clear();

            for (dsc, state) in &self.renderpasses {
                self.device_context.device().destroy_render_pass(state.vk_obj, None);
            }
            self.renderpasses.clear();

            for (dsc, state) in &self.pipeline_layouts {
                self.device_context.device().destroy_pipeline_layout(state.vk_obj, None);
            }
            self.pipeline_layouts.clear();

            for (dsc, state) in &self.descriptor_set_layouts {
                self.device_context.device().destroy_descriptor_set_layout(state.vk_obj, None);
            }
            self.descriptor_set_layouts.clear();
        }
    }
}