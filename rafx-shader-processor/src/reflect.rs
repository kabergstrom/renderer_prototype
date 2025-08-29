use crate::shader_types::{
    element_count, generate_struct, MemoryLayout, TypeAlignmentInfo, UserType,
};
use fnv::FnvHashMap;
use rafx_api::{
    RafxAddressMode, RafxCompareOp, RafxFilterType, RafxGlUniformMember, RafxMipMapMode,
    RafxReflectedDescriptorSetLayout, RafxReflectedDescriptorSetLayoutBinding,
    RafxReflectedEntryPoint, RafxReflectedVertexInput, RafxResourceType, RafxResult,
    RafxSamplerDef, RafxShaderResource, RafxShaderStageFlags, RafxShaderStageReflection,
    MAX_DESCRIPTOR_SET_LAYOUTS,
};
use spirv_cross2::handle::{Handle, TypeId};
use spirv_cross2::reflect::DecorationValue;
use spirv_cross2::spirv::ExecutionModel;

fn get_descriptor_count_from_type(
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    ty: Handle<TypeId>,
) -> RafxResult<u32> {
    use spirv_cross2::reflect::TypeInner as Type;
    Ok(
        match artifact
            .type_description(ty)
            .map_err(|_x| "could not get type from reflection data")?
            .inner
        {
            Type::Array { dimensions, .. } => {
                let mut count = 1;
                for dim in &dimensions {
                    match dim {
                        spirv_cross2::reflect::ArrayDimension::Literal(size) => {
                            count *= size;
                        }
                        spirv_cross2::reflect::ArrayDimension::Constant(_handle) => {
                            Err(
                            "Cannot determine static descriptor count for binding with dynamic array size (specialization constant)."
                        )?;
                        }
                    }
                }
                count
            }
            _ => 1,
        },
    )
}

fn get_descriptor_size_from_resource_rafx(
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    resource: &spirv_cross2::reflect::Resource,
    resource_type: RafxResourceType,
) -> RafxResult<u32> {
    Ok(
        if resource_type.intersects(
            RafxResourceType::UNIFORM_BUFFER
                | RafxResourceType::BUFFER
                | RafxResourceType::BUFFER_READ_WRITE,
        ) {
            match artifact
                .type_description(resource.type_id)
                .map_err(|_x| "could not get type from reflection data")?
                .size_hint
            {
                spirv_cross2::reflect::TypeSizeHint::Static(size) => {
                    // The size is returned as usize, so we cast it to u32.
                    size as u32
                }

                spirv_cross2::reflect::TypeSizeHint::RuntimeArray(_)
                | spirv_cross2::reflect::TypeSizeHint::Matrix(_)
                | spirv_cross2::reflect::TypeSizeHint::UnknownArrayStride(_) => 0,
            }
            // (artifact
            //     .get_declared_struct_size(resource.type_id)
            //     .map_err(|_x| "could not get size from reflection data")?
            //     + 15,)
            //     / 16
            //     * 16
        } else {
            0
        },
    )
}

fn get_rafx_resource(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    declarations: &super::parse_declarations::ParseDeclarationsResult,
    resource: &spirv_cross2::reflect::Resource,
    resource_type: RafxResourceType,
    stage_flags: RafxShaderStageFlags,
) -> RafxResult<RafxShaderResource> {
    let set = artifact
        .decoration(resource.id, spirv_cross2::spirv::Decoration::DescriptorSet)
        .map_err(|_x| "could not get descriptor set index from reflection data")?;
    let Some(DecorationValue::Literal(set)) = set else {
        Err("could not get descriptor set index from reflection data")?
    };
    let binding = artifact
        .decoration(resource.id, spirv_cross2::spirv::Decoration::Binding)
        .map_err(|_x| "could not get descriptor binding index from reflection data")?;
    let Some(DecorationValue::Literal(binding)) = binding else {
        Err("could not get descriptor binding index from reflection data")?
    };
    let element_count = get_descriptor_count_from_type(artifact, resource.type_id)?;

    let parsed_binding = declarations.bindings.iter().find(|x| x.parsed.layout_parts.binding == Some(binding as usize) && x.parsed.layout_parts.set == Some(set as usize))
        .or_else(|| declarations.bindings.iter().find(|x| x.parsed.instance_name == *resource.name))
        .ok_or_else(|| format!("A resource named {} in spirv reflection data was not matched up to a resource scanned in source code.", resource.name))?;

    let slot_name = if let Some(annotation) = &parsed_binding.annotations.slot_name {
        Some(annotation.0.clone())
    } else {
        None
    };

    let mut gl_uniform_members = Vec::<RafxGlUniformMember>::default();
    if resource_type == RafxResourceType::UNIFORM_BUFFER {
        generate_gl_uniform_members(
            &builtin_types,
            &user_types,
            &parsed_binding.parsed.type_name,
            parsed_binding.parsed.type_name.clone(),
            0,
            &mut gl_uniform_members,
        )?;
    }

    let gles_name = if resource_type == RafxResourceType::UNIFORM_BUFFER {
        parsed_binding.parsed.type_name.clone()
    } else {
        parsed_binding.parsed.instance_name.clone()
    };

    let resource = RafxShaderResource {
        resource_type,
        set_index: set,
        binding,
        element_count,
        size_in_bytes: 0,
        used_in_shader_stages: stage_flags,
        name: Some(slot_name.unwrap_or_else(|| resource.name.to_string())),
        gles_name: Some(gles_name),
        gles_sampler_name: None, // This is set later if necessary when we cross compile GLES 2.0 src by set_gl_sampler_name
        gles2_uniform_members: gl_uniform_members,
        dx12_space: Some(set),
        dx12_reg: Some(binding),
    };

    resource.validate()?;

    Ok(resource)
}

fn get_reflected_binding(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    declarations: &super::parse_declarations::ParseDeclarationsResult,
    resource: &spirv_cross2::reflect::Resource,
    resource_type: RafxResourceType,
    stage_flags: RafxShaderStageFlags,
) -> RafxResult<RafxReflectedDescriptorSetLayoutBinding> {
    let name = &resource.name;
    let rafx_resource = get_rafx_resource(
        builtin_types,
        user_types,
        artifact,
        declarations,
        resource,
        resource_type,
        stage_flags,
    )?;
    let set = rafx_resource.set_index;
    let binding = rafx_resource.binding;

    let parsed_binding = declarations.bindings.iter().find(|x| x.parsed.layout_parts.binding == Some(binding as usize) && x.parsed.layout_parts.set == Some(set as usize))
        .or_else(|| declarations.bindings.iter().find(|x| x.parsed.instance_name == *name))
        .ok_or_else(|| format!("A resource named {} in spirv reflection data was not matched up to a resource scanned in source code.", resource.name))?;

    let size = get_descriptor_size_from_resource_rafx(artifact, resource, resource_type)
        .map_err(|_x| "could not get size from reflection data")?;

    let internal_buffer_per_descriptor_size =
        if parsed_binding.annotations.use_internal_buffer.is_some() {
            Some(size)
        } else {
            None
        };

    let immutable_samplers =
        if let Some(annotation) = &parsed_binding.annotations.immutable_samplers {
            Some(annotation.0.clone())
        } else {
            None
        };

    Ok(RafxReflectedDescriptorSetLayoutBinding {
        resource: rafx_resource,
        internal_buffer_per_descriptor_size,
        immutable_samplers,
    })
}

fn get_reflected_bindings(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    descriptors: &mut Vec<RafxReflectedDescriptorSetLayoutBinding>,
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    declarations: &super::parse_declarations::ParseDeclarationsResult,
    resources: &[spirv_cross2::reflect::Resource],
    resource_type: RafxResourceType,
    stage_flags: RafxShaderStageFlags,
) -> RafxResult<()> {
    for resource in resources {
        descriptors.push(get_reflected_binding(
            builtin_types,
            user_types,
            artifact,
            declarations,
            resource,
            resource_type,
            stage_flags,
        )?);
    }

    Ok(())
}

fn get_all_reflected_bindings(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    shader_resources: &spirv_cross2::reflect::ShaderResources,
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    declarations: &super::parse_declarations::ParseDeclarationsResult,
    stage_flags: RafxShaderStageFlags,
) -> RafxResult<Vec<RafxReflectedDescriptorSetLayoutBinding>> {
    let mut bindings = Vec::default();
    let uniform_buffers = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::UniformBuffer)
        .map_err(|_x| "could not uniform buffers from reflection data")?
        .collect::<Vec<_>>();
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &uniform_buffers,
        RafxResourceType::UNIFORM_BUFFER,
        stage_flags,
    )?;

    let all_storage_buffers = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::StorageBuffer)
        .map_err(|_x| "could not storage buffers from reflection data")?
        .collect::<Vec<_>>();
    let mut ro_storage_buffers = Vec::new();
    let mut rw_storage_buffers = Vec::new();
    for buf in all_storage_buffers {
        if let Some(DecorationValue::Present) = artifact
            .decoration(buf.id, spirv_cross2::spirv::Decoration::NonWritable)
            .map_err(|_x| "could not get decoration from reflection data")?
        {
            ro_storage_buffers.push(buf);
        } else {
            rw_storage_buffers.push(buf);
        }
    }

    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &rw_storage_buffers,
        RafxResourceType::BUFFER_READ_WRITE,
        stage_flags,
    )?;
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &ro_storage_buffers,
        RafxResourceType::BUFFER,
        stage_flags,
    )?;
    let storage_images = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::StorageImage)
        .map_err(|_x| "could not storage images from reflection data")?
        .collect::<Vec<_>>();
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &storage_images,
        RafxResourceType::TEXTURE_READ_WRITE,
        stage_flags,
    )?;
    let sampled_images = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::SampledImage)
        .map_err(|_x| "could not sampled images from reflection data")?
        .collect::<Vec<_>>();
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &sampled_images,
        RafxResourceType::COMBINED_IMAGE_SAMPLER,
        stage_flags,
    )?;
    let separate_images = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::SeparateImage)
        .map_err(|_x| "could not get separate images from reflection data")?
        .collect::<Vec<_>>();
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &separate_images,
        RafxResourceType::TEXTURE,
        stage_flags,
    )?;
    let separate_samplers = shader_resources
        .resources_for_type(spirv_cross2::reflect::ResourceType::SeparateSamplers)
        .map_err(|_x| "could not get separate samplers from reflection data")?
        .collect::<Vec<_>>();
    get_reflected_bindings(
        builtin_types,
        user_types,
        &mut bindings,
        artifact,
        declarations,
        &separate_samplers,
        RafxResourceType::SAMPLER,
        stage_flags,
    )?;

    Ok(bindings)
}

fn shader_stage_to_execution_model(
    stages: RafxShaderStageFlags
) -> Vec<spirv_cross2::spirv::ExecutionModel> {
    let mut out = vec![];
    if stages.intersects(RafxShaderStageFlags::VERTEX) {
        out.push(ExecutionModel::Vertex)
    }
    if stages.intersects(RafxShaderStageFlags::FRAGMENT) {
        out.push(ExecutionModel::Fragment)
    }
    if stages.intersects(RafxShaderStageFlags::COMPUTE) {
        out.push(ExecutionModel::GLCompute)
    }
    if stages.intersects(RafxShaderStageFlags::TESSELLATION_CONTROL) {
        out.push(ExecutionModel::TessellationControl)
    }
    if stages.intersects(RafxShaderStageFlags::TESSELLATION_EVALUATION) {
        out.push(ExecutionModel::TessellationEvaluation)
    }
    out
}

pub(crate) fn get_sorted_bindings_for_all_entry_points(
    entry_points: &[RafxReflectedEntryPoint]
) -> RafxResult<Vec<RafxShaderResource>> {
    let mut all_resources_lookup = FnvHashMap::<(u32, u32), RafxShaderResource>::default();
    for entry_point in entry_points {
        for resource in &entry_point.rafx_api_reflection.resources {
            let key = (resource.set_index, resource.binding);
            if let Some(old) = all_resources_lookup.get_mut(&key) {
                if resource.resource_type != old.resource_type {
                    Err(format!(
                        "Shaders with same set and binding {:?} have mismatching resource types {:?} and {:?}",
                        key,
                        resource.resource_type,
                        old.resource_type
                    ))?;
                }

                if resource.element_count_normalized() != old.element_count_normalized() {
                    Err(format!(
                        "Shaders with same set and binding {:?} have mismatching element counts {:?} and {:?}",
                        key,
                        resource.element_count_normalized(),
                        old.element_count_normalized()
                    ))?;
                }

                old.used_in_shader_stages |= resource.used_in_shader_stages;
            } else {
                all_resources_lookup.insert(key, resource.clone());
            }
        }
    }

    let mut resources: Vec<_> = all_resources_lookup.values().cloned().collect();
    resources.sort_by(|lhs, rhs| lhs.binding.cmp(&rhs.binding));

    Ok(resources)
}

pub(crate) fn get_hlsl_register_assignments(
    entry_points: &[RafxReflectedEntryPoint]
) -> RafxResult<Vec<HlslAssignment>> {
    let mut bindings = vec![];

    let resources = get_sorted_bindings_for_all_entry_points(entry_points)?;

    let mut max_space_index = -1;
    for resource in resources {
        let execution_models = shader_stage_to_execution_model(resource.used_in_shader_stages);
        if resource.resource_type != RafxResourceType::ROOT_CONSTANT {
            let space_register = spirv_cross2::compile::hlsl::RegisterBinding {
                space: resource.dx12_space.unwrap(),
                register: resource.dx12_reg.unwrap(),
            };
            max_space_index = max_space_index.max(space_register.space as i32);
            for execution_model in execution_models {
                bindings.push(HlslAssignment {
                    execution_model,
                    binding: spirv_cross2::compile::hlsl::ResourceBinding::Qualified {
                        set: resource.set_index,
                        binding: resource.binding,
                    },
                    bind_target: spirv_cross2::compile::hlsl::BindTarget {
                        cbv: Some(space_register),
                        uav: Some(space_register),
                        srv: Some(space_register),
                        sampler: Some(space_register),
                    },
                });
                /*spirv_cross2::hlsl::HlslResourceBinding {
                    desc_set: resource.set_index,
                    binding: resource.binding,
                    stage: execution_model,
                    cbv: space_register,
                    uav: space_register,
                    srv: space_register,
                    sampler: space_register,
                })*/
            }
        }
    }

    //
    // If we have a push constant, we need to add an assignment for the same binding to all relevant stages
    //
    let push_constant_space_index = (max_space_index + 1) as u32;
    let mut push_constant_stages = RafxShaderStageFlags::empty();
    for entry_point in entry_points {
        for resource in &entry_point.rafx_api_reflection.resources {
            if resource.resource_type == RafxResourceType::ROOT_CONSTANT {
                assert_eq!(resource.dx12_space, Some(push_constant_space_index));
                push_constant_stages |= resource.used_in_shader_stages;
            }
        }
    }

    if !push_constant_stages.is_empty() {
        let push_constant_execution_models = shader_stage_to_execution_model(push_constant_stages);
        let space_register = spirv_cross2::compile::hlsl::RegisterBinding {
            space: push_constant_space_index,
            register: 0,
        };
        for execution_model in push_constant_execution_models {
            // bindings.push(spirv_cross2::hlsl::HlslResourceBinding {
            //     desc_set: !0,
            //     binding: 0,
            //     stage: execution_model,
            //     cbv: space_register,
            //     uav: space_register,
            //     srv: space_register,
            //     sampler: space_register,
            // })
            todo!("this is sus, the space_register doesn't vary?");
            bindings.push(HlslAssignment {
                execution_model,
                binding: spirv_cross2::compile::hlsl::ResourceBinding::PushConstantBuffer,
                bind_target: spirv_cross2::compile::hlsl::BindTarget {
                    cbv: Some(space_register),
                    uav: Some(space_register),
                    srv: Some(space_register),
                    sampler: Some(space_register),
                },
            });
        }
    }

    Ok(bindings)
}

//NOTE: There are special set/binding pairs to control assignment of arg buffers themselves,
// push constant buffers, etc. For example:
//
//     ResourceBindingLocation { desc_set: 0, binding: !3u32, stage: ExecutionModel::Vertex}
//
// Will force descriptor set 0 to be [[buffer(n)]] where n is the value of ResourceBinding::buffer_id
//TODO: Exclude MSL constexpr samplers?
pub(crate) fn msl_assign_argument_buffer_ids(
    entry_points: &[RafxReflectedEntryPoint]
) -> RafxResult<Vec<MslAssignment>> {
    let resources = get_sorted_bindings_for_all_entry_points(entry_points)?;

    let mut argument_buffer_assignments = Vec::<MslAssignment>::default();

    // If we update this constant, update the arrays in this function
    assert_eq!(MAX_DESCRIPTOR_SET_LAYOUTS, 4);

    //
    // Assign unique buffer indexes for each resource in the set, sequentially and taking into account
    // that some entries take multiple "slots."
    //
    // Recently changed to re-use the dx12 assignment logic as it's essentially the same
    //

    let mut next_msl_buffer_id = 0;
    let mut next_msl_texture_id = 0;
    let mut next_msl_sampler_id = 0;

    let mut max_set_index = -1;
    for resource in resources {
        if resource.resource_type == RafxResourceType::ROOT_CONSTANT {
            continue;
        }

        // --- THIS ASSERTION IS REMOVED as it's no longer valid ---
        // The MSL argument ID and DX12 register are now generated by different,
        // independent, and correct logic paths. They are not expected to be equal.
        // assert_eq!(Some(msl_argument_buffer_id), resource.dx12_reg);

        max_set_index = max_set_index.max(resource.set_index as i32);

        // Create a default, empty bind target.
        let mut bind_target = spirv_cross2::compile::msl::BindTarget {
            buffer: 0,
            texture: 0,
            sampler: 0,
            count: std::num::NonZero::new(resource.element_count_normalized()),
        };

        // Based on the resource type, assign from the correct counter and
        // populate ONLY the relevant field in the bind target.
        let element_count = resource.element_count_normalized();
        if resource.resource_type.intersects(
            RafxResourceType::UNIFORM_BUFFER
                | RafxResourceType::BUFFER
                | RafxResourceType::BUFFER_READ_WRITE
                | RafxResourceType::TEXEL_BUFFER
                | RafxResourceType::TEXEL_BUFFER_READ_WRITE,
        ) {
            bind_target.buffer = next_msl_buffer_id;
            next_msl_buffer_id += element_count;
        } else if resource
            .resource_type
            .intersects(RafxResourceType::TEXTURE | RafxResourceType::TEXTURE_READ_WRITE)
        {
            bind_target.texture = next_msl_texture_id;
            next_msl_texture_id += element_count;
        } else if resource.resource_type.intersects(RafxResourceType::SAMPLER) {
            bind_target.sampler = next_msl_sampler_id;
            next_msl_sampler_id += element_count;
        }

        let execution_models = shader_stage_to_execution_model(resource.used_in_shader_stages);
        for execution_model in execution_models {
            argument_buffer_assignments.push(MslAssignment {
                execution_model,
                binding: spirv_cross2::compile::msl::ResourceBinding::Qualified {
                    set: resource.set_index,
                    binding: resource.binding,
                },
                bind_target: bind_target.clone(), // Use the precisely constructed bind target
            });
        }
    }
    //
    // If we have a push constant, we need to add an assignment for the same binding to all relevant stages
    //
    let push_constant_set_index = (max_set_index + 1) as u32;
    let mut push_constant_stages = RafxShaderStageFlags::empty();
    for entry_point in entry_points {
        for resource in &entry_point.rafx_api_reflection.resources {
            if resource.resource_type == RafxResourceType::ROOT_CONSTANT {
                assert_eq!(Some(push_constant_set_index), resource.dx12_space);
                push_constant_stages |= resource.used_in_shader_stages;
            }
        }
    }

    if !push_constant_stages.is_empty() {
        let push_constant_execution_models = shader_stage_to_execution_model(push_constant_stages);
        for execution_model in push_constant_execution_models {
            argument_buffer_assignments.push(MslAssignment {
                execution_model,
                binding: spirv_cross2::compile::msl::ResourceBinding::PushConstantBuffer,
                bind_target: spirv_cross2::compile::msl::BindTarget {
                    count: std::num::NonZero::new(1),
                    buffer: push_constant_set_index,
                    sampler: push_constant_set_index,
                    texture: push_constant_set_index,
                },
            });
        }
    }

    Ok(argument_buffer_assignments)
}

fn msl_create_sampler_data(sampler_def: &RafxSamplerDef) -> RafxResult<MslConstSampler> {
    let lod_clamp_min = sampler_def.mip_lod_bias;
    let lod_clamp_max = if sampler_def.mip_map_mode == RafxMipMapMode::Linear {
        sampler_def.mip_lod_bias
    } else {
        0.0
    };

    fn convert_filter(filter: RafxFilterType) -> SamplerFilter {
        match filter {
            RafxFilterType::Nearest => SamplerFilter::Nearest,
            RafxFilterType::Linear => SamplerFilter::Linear,
        }
    }

    fn convert_mip_map_mode(mip_map_mode: RafxMipMapMode) -> SamplerMipFilter {
        match mip_map_mode {
            RafxMipMapMode::Nearest => SamplerMipFilter::Nearest,
            RafxMipMapMode::Linear => SamplerMipFilter::Linear,
        }
    }

    fn convert_address_mode(address_mode: RafxAddressMode) -> SamplerAddress {
        match address_mode {
            RafxAddressMode::Mirror => SamplerAddress::MirroredRepeat,
            RafxAddressMode::Repeat => SamplerAddress::Repeat,
            RafxAddressMode::ClampToEdge => SamplerAddress::ClampToEdge,
            RafxAddressMode::ClampToBorder => SamplerAddress::ClampToBorder,
        }
    }

    fn convert_compare_op(compare_op: RafxCompareOp) -> SamplerCompareFunc {
        match compare_op {
            RafxCompareOp::Never => SamplerCompareFunc::Never,
            RafxCompareOp::Less => SamplerCompareFunc::Less,
            RafxCompareOp::Equal => SamplerCompareFunc::Equal,
            RafxCompareOp::LessOrEqual => SamplerCompareFunc::LessEqual,
            RafxCompareOp::Greater => SamplerCompareFunc::Greater,
            RafxCompareOp::NotEqual => SamplerCompareFunc::NotEqual,
            RafxCompareOp::GreaterOrEqual => SamplerCompareFunc::GreaterEqual,
            RafxCompareOp::Always => SamplerCompareFunc::Always,
        }
    }

    let max_anisotropy = if sampler_def.max_anisotropy == 0.0 {
        1
    } else {
        sampler_def.max_anisotropy as i32
    };

    use spirv_cross2::compile::msl::*;
    let _conversion = SamplerYcbcrConversion {
        // Sampler YCbCr conversion parameters
        planes: 0,
        resolution: YcbcrFormatResolution::FormatResolution444,
        chroma_filter: SamplerFilter::Nearest,
        x_chroma_offset: YcbcrChromaLocation::CositedEven,
        y_chroma_offset: YcbcrChromaLocation::CositedEven,
        swizzle: [
            YcbcrComponentSwizzle::Identity,
            YcbcrComponentSwizzle::Identity,
            YcbcrComponentSwizzle::Identity,
            YcbcrComponentSwizzle::Identity,
        ],
        ycbcr_model: YcbcrTargetFormat::RgbIdentity,
        ycbcr_range: YcbcrConversionRange::ItuFull,
        bpc: 8,
    };
    Ok(MslConstSampler {
        sampler: ConstexprSampler {
            coord: SamplerCoord::Normalized,
            min_filter: convert_filter(sampler_def.min_filter),
            mag_filter: convert_filter(sampler_def.mag_filter),
            mip_filter: convert_mip_map_mode(sampler_def.mip_map_mode),
            s_address: convert_address_mode(sampler_def.address_mode_u),
            t_address: convert_address_mode(sampler_def.address_mode_v),
            r_address: convert_address_mode(sampler_def.address_mode_w),
            compare_func: convert_compare_op(sampler_def.compare_op),
            border_color: SamplerBorderColor::TransparentBlack,
            lod_clamp_min,
            lod_clamp_max,
            max_anisotropy,
            compare_enable: sampler_def.compare_op != RafxCompareOp::Never,
            lod_clamp_enable: sampler_def.mip_lod_bias != 0.0,
            anisotropy_enable: sampler_def.max_anisotropy > 0.0,
        },
        conversion: None,
    })
}

pub struct MslConstSampler {
    pub sampler: spirv_cross2::compile::msl::ConstexprSampler,
    pub conversion: Option<spirv_cross2::compile::msl::SamplerYcbcrConversion>,
}

pub(crate) fn msl_const_samplers(
    entry_points: &[RafxReflectedEntryPoint]
) -> RafxResult<FnvHashMap<spirv_cross2::compile::msl::ResourceBinding, MslConstSampler>> {
    let mut immutable_samplers =
        FnvHashMap::<spirv_cross2::compile::msl::ResourceBinding, MslConstSampler>::default();

    for entry_point in entry_points {
        for layout in &entry_point.descriptor_set_layouts {
            if let Some(layout) = layout {
                for binding in &layout.bindings {
                    if let Some(immutable_sampler) = &binding.immutable_samplers {
                        let location = spirv_cross2::compile::msl::ResourceBinding::Qualified {
                            set: binding.resource.set_index,
                            binding: binding.resource.binding,
                        };

                        if immutable_sampler.len() > 1 {
                            Err(format!("Multiple immutable samplers in a single binding ({:?}) not supported in MSL", location))?;
                        }
                        let immutable_sampler = immutable_sampler.first().unwrap();

                        let sampler_data = msl_create_sampler_data(&immutable_sampler)?;

                        if let Some(old) = immutable_samplers.get(&location) {
                            // if *old != sampler_data {
                            //     Err(format!("Samplers in different entry points but same location ({:?}) do not match: \n{:#?}\n{:#?}", location, old, sampler_data))?;
                            // }
                        } else {
                            immutable_samplers.insert(location, sampler_data);
                        }
                    }
                }
            }
        }
    }

    Ok(immutable_samplers)
}

fn generate_gl_uniform_members(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    type_name: &str,
    prefix: String,
    offset: usize,
    gl_uniform_members: &mut Vec<RafxGlUniformMember>,
) -> RafxResult<()> {
    if builtin_types.contains_key(type_name) {
        //println!("{} at {}: {}", prefix, offset, tye_name);
        gl_uniform_members.push(RafxGlUniformMember {
            name: prefix,
            offset: offset as u32,
        })
    } else {
        let user_type = user_types.get(type_name).ok_or_else(|| {
            format!(
                "Could not find type named {} in generate_gl_uniform_members",
                type_name
            )
        })?;

        let generated_struct = generate_struct(
            builtin_types,
            user_types,
            &user_type.type_name,
            user_type,
            MemoryLayout::Std140,
        )?;

        for field in &*user_type.fields {
            let struct_member = generated_struct
                .members
                .iter()
                .find(|x| x.name == field.field_name)
                .ok_or_else(|| {
                    format!(
                        "Could not find member {} within generated struct {}",
                        field.field_name, generated_struct.name
                    )
                })?;

            if field.array_sizes.is_empty() {
                let member_full_name = format!("{}.{}", prefix, field.field_name);
                let field_offset = offset + struct_member.offset;
                generate_gl_uniform_members(
                    builtin_types,
                    user_types,
                    &field.type_name,
                    member_full_name,
                    field_offset,
                    gl_uniform_members,
                )?;
            } else {
                let element_count = element_count(&field.array_sizes);
                if element_count == 0 {
                    return Err("Variable array encountered in generate_gl_uniform_members, not supported in GL ES")?;
                }

                for i in 0..element_count {
                    let member_full_name = format!("{}.{}[{}]", prefix, field.field_name, i);
                    let field_offset =
                        offset + struct_member.offset + (i * struct_member.size / element_count);
                    generate_gl_uniform_members(
                        builtin_types,
                        user_types,
                        &field.type_name,
                        member_full_name,
                        field_offset,
                        gl_uniform_members,
                    )?;
                }
            }
        }
    }

    Ok(())
}

pub struct HlslAssignment {
    pub execution_model: spirv_cross2::spirv::ExecutionModel,
    pub binding: spirv_cross2::compile::hlsl::ResourceBinding,
    pub bind_target: spirv_cross2::compile::hlsl::BindTarget,
}
pub struct MslAssignment {
    pub execution_model: spirv_cross2::spirv::ExecutionModel,
    pub binding: spirv_cross2::compile::msl::ResourceBinding,
    pub bind_target: spirv_cross2::compile::msl::BindTarget,
}

pub struct ShaderProcessorRefectionData {
    pub reflection: Vec<RafxReflectedEntryPoint>,
    pub hlsl_register_assignments: Vec<HlslAssignment>,
    pub hlsl_vertex_attribute_remaps: Vec<RafxReflectedVertexInput>,
    pub msl_argument_buffer_assignments: Vec<MslAssignment>,
    pub msl_const_samplers:
        FnvHashMap<spirv_cross2::compile::msl::ResourceBinding, MslConstSampler>,
}

impl ShaderProcessorRefectionData {
    // GL ES 2.0 attaches sampler state to textures. So every texture must be associated with a
    // single sampler. This function is called when cross-compiling to GL ES 2.0 to set
    // gl_sampler_name on all texture resources.
    pub fn set_gl_sampler_name(
        &mut self,
        texture_gl_name: &str,
        sampler_gl_name: &str,
    ) {
        for entry_point in &mut self.reflection {
            for resource in &mut entry_point.rafx_api_reflection.resources {
                if resource.gles_name.as_ref().unwrap().as_str() == texture_gl_name {
                    assert!(resource.resource_type.intersects(
                        RafxResourceType::TEXTURE | RafxResourceType::TEXTURE_READ_WRITE
                    ));
                    resource.gles_sampler_name = Some(sampler_gl_name.to_string());
                }
            }

            for layout in &mut entry_point.descriptor_set_layouts {
                if let Some(layout) = layout {
                    for resource in &mut layout.bindings {
                        if resource.resource.gles_name.as_ref().unwrap().as_str() == texture_gl_name
                        {
                            assert!(resource.resource.resource_type.intersects(
                                RafxResourceType::TEXTURE | RafxResourceType::TEXTURE_READ_WRITE
                            ));
                            resource.resource.gles_sampler_name = Some(sampler_gl_name.to_string());
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn reflect_data(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    artifact: &spirv_cross2::compile::CompiledArtifact<spirv_cross2::targets::Glsl>,
    declarations: &super::parse_declarations::ParseDeclarationsResult,
    require_semantics: bool,
) -> RafxResult<ShaderProcessorRefectionData> {
    let mut reflected_entry_points = Vec::default();
    for entry_point in artifact
        .entry_points()
        .map_err(|_x| "could not get entry point from reflection data")?
    {
        let entry_point_name = entry_point.name;
        println!(
            "processing entry point {entry_point_name} of execution model {:?}",
            entry_point.execution_model
        );
        let stage_flags = map_shader_stage_flags(entry_point.execution_model)?;

        let shader_resources = artifact
            .shader_resources()
            .map_err(|_x| "could not get resources from reflection data")?;

        let mut dsc_bindings = get_all_reflected_bindings(
            builtin_types,
            user_types,
            &shader_resources,
            artifact,
            declarations,
            stage_flags,
        )?;

        //TODO: Assign dx12 values?
        //TODO: Assign MSL values?
        //TODO: This might not work because we need to merge resources between stages

        dsc_bindings.sort_by(|lhs, rhs| {
            if lhs.resource.set_index != rhs.resource.set_index {
                lhs.resource.set_index.cmp(&rhs.resource.set_index)
            } else {
                lhs.resource.binding.cmp(&rhs.resource.binding)
            }
        });

        // If we update this constant, update the arrays in this function
        assert_eq!(MAX_DESCRIPTOR_SET_LAYOUTS, 4);

        // Create separate, global counters for each DX12 register type.
        let mut next_cbv_register = 0;
        let mut next_srv_register = 0;
        let mut next_uav_register = 0;
        let mut next_sampler_register = 0;

        let mut max_set_index = -1;
        for binding in &mut dsc_bindings {
            if binding
                .resource
                .resource_type
                .intersects(RafxResourceType::ROOT_CONSTANT)
            {
                continue;
            }

            max_set_index = max_set_index.max(binding.resource.set_index as i32);

            binding.resource.dx12_space = Some(binding.resource.set_index);

            // Assign the register index from the correct resource type's counter.
            let element_count = binding.resource.element_count_normalized();
            let register = if binding
                .resource
                .resource_type
                .intersects(RafxResourceType::UNIFORM_BUFFER)
            {
                let reg = next_cbv_register;
                next_cbv_register += element_count;
                println!("uniform buffer: {:?}", binding.resource);
                reg
            } else if binding.resource.resource_type.intersects(
                RafxResourceType::TEXTURE
                    | RafxResourceType::BUFFER
                    | RafxResourceType::TEXEL_BUFFER,
            ) {
                let reg = next_srv_register;
                next_srv_register += element_count;
                println!("srv: {:?}", binding.resource);
                reg
            } else if binding.resource.resource_type.intersects(
                RafxResourceType::TEXTURE_READ_WRITE
                    | RafxResourceType::BUFFER_READ_WRITE
                    | RafxResourceType::TEXEL_BUFFER_READ_WRITE,
            ) {
                let reg = next_uav_register;
                next_uav_register += element_count;
                println!("uav: {:?}", binding.resource);
                reg
            } else if binding
                .resource
                .resource_type
                .intersects(RafxResourceType::SAMPLER)
            {
                let reg = next_sampler_register;
                next_sampler_register += element_count;
                println!("sampler: {:?}", binding.resource);
                reg
            } else {
                // Should not happen for descriptor-bound resources.
                // Default to 0 if it's an unknown type to avoid a compile error.
                println!("unknown: {:?}", binding.resource);
                0
            };

            binding.resource.dx12_reg = Some(register);
        }

        let push_constant_dx12_space = max_set_index + 1;

        // stage inputs
        // stage outputs
        // subpass inputs
        // atomic counters
        // push constant buffers

        let mut descriptor_set_layouts: Vec<Option<RafxReflectedDescriptorSetLayout>> = vec![];
        let mut rafx_bindings = Vec::default();
        for binding in dsc_bindings {
            rafx_bindings.push(binding.resource.clone());

            while descriptor_set_layouts.len() <= binding.resource.set_index as usize {
                descriptor_set_layouts.push(None);
            }

            match &mut descriptor_set_layouts[binding.resource.set_index as usize] {
                Some(x) => x.bindings.push(binding),
                x @ None => {
                    *x = Some(RafxReflectedDescriptorSetLayout {
                        bindings: vec![binding],
                    })
                }
            }
        }
        let all_resources = &shader_resources
            .all_resources()
            .map_err(|_e| "failed getting all resources from reflection data")?;
        //TODO: This is using a list of push constants but GLSL disallows multiple within
        // the same file
        for push_constant in &all_resources.push_constant_buffers {
            // let declared_size = artifact
            //     .get_declared_struct_size(push_constant.type_id)
            //     .unwrap();
            let push_constant_type = artifact.type_description(push_constant.type_id).unwrap();
            let spirv_cross2::reflect::TypeSizeHint::Static(declared_size) =
                push_constant_type.size_hint
            else {
                Err(format!(
                    "push constant {:?} has size_hint {:?}, expected static",
                    push_constant.name, push_constant_type.size_hint
                ))?
            };

            let resource = RafxShaderResource {
                resource_type: RafxResourceType::ROOT_CONSTANT,
                size_in_bytes: declared_size as u32,
                used_in_shader_stages: stage_flags,
                name: Some(push_constant.name.to_string()),
                set_index: u32::MAX,
                binding: u32::MAX,
                dx12_space: Some(push_constant_dx12_space as u32),
                dx12_reg: Some(0),
                ..Default::default()
            };
            resource.validate()?;

            rafx_bindings.push(resource);
        }

        //TODO: Store the type and verify that the format associated in the game i.e. R32G32B32 is
        // something reasonable (like vec3).
        let mut dsc_vertex_inputs = Vec::default();
        if entry_point.execution_model == spirv_cross2::spirv::ExecutionModel::Vertex {
            for resource in &all_resources.stage_inputs {
                let name = &resource.name;
                let location = artifact
                    .decoration(resource.id, spirv_cross2::spirv::Decoration::Location)
                    .map_err(|_x| "could not get descriptor binding index from reflection data")?;
                let Some(DecorationValue::Literal(location)) = location else {
                    Err("could not get descriptor binding index from reflection data")?
                };

                let parsed_binding = declarations.bindings.iter().find(|x| x.parsed.layout_parts.location == Some(location as usize))
                    .or_else(|| declarations.bindings.iter().find(|x| x.parsed.instance_name == *name))
                    .ok_or_else(|| format!("A resource named {} in spirv reflection data was not matched up to a resource scanned in source code.", resource.name))?;

                let semantic = &parsed_binding
                    .annotations
                    .semantic
                    .as_ref()
                    .map(|x| x.0.clone());

                let semantic = if require_semantics {
                    semantic.clone().ok_or_else(|| format!("No semantic annotation for vertex input '{}'. All vertex inputs must have a semantic annotation if generating rust code, HLSL, and/or cooked shaders.", name))?
                } else {
                    "".to_string()
                };

                // TODO(dvd): Might need other special type handling here.
                if parsed_binding.parsed.type_name == "mat4" {
                    for index in 0..4 {
                        dsc_vertex_inputs.push(RafxReflectedVertexInput {
                            name: name.to_string(),
                            semantic: format!("{}{}", semantic, index),
                            location: location + index,
                        });
                    }
                } else {
                    dsc_vertex_inputs.push(RafxReflectedVertexInput {
                        name: name.to_string(),
                        semantic,
                        location,
                    });
                }
            }
        }

        // if let Some(group_size) = &declarations.group_size {
        //     assert_eq!(entry_point.work_group_size.x, group_size.x);
        //     assert_eq!(entry_point.work_group_size.y, group_size.y);
        //     assert_eq!(entry_point.work_group_size.z, group_size.z);
        // }

        let rafx_reflection = RafxShaderStageReflection {
            shader_stage: stage_flags,
            resources: rafx_bindings,
            entry_point_name: entry_point_name.to_string(),
            compute_threads_per_group: declarations
                .group_size
                .as_ref()
                .map(|group_size| [group_size.x, group_size.y, group_size.z]),
        };

        reflected_entry_points.push(RafxReflectedEntryPoint {
            descriptor_set_layouts,
            vertex_inputs: dsc_vertex_inputs,
            rafx_api_reflection: rafx_reflection,
        });
    }

    let hlsl_register_assignments = get_hlsl_register_assignments(&reflected_entry_points)?;

    let msl_argument_buffer_assignments = msl_assign_argument_buffer_ids(&reflected_entry_points)?;

    let msl_const_samplers = msl_const_samplers(&reflected_entry_points)?;

    let mut hlsl_vertex_attribute_remaps = Vec::default();
    for entry_point in &reflected_entry_points {
        for vi in &entry_point.vertex_inputs {
            hlsl_vertex_attribute_remaps.push(vi.clone());
        }
    }

    Ok(ShaderProcessorRefectionData {
        reflection: reflected_entry_points,
        hlsl_register_assignments,
        hlsl_vertex_attribute_remaps,
        msl_argument_buffer_assignments,
        msl_const_samplers,
    })
}

fn map_shader_stage_flags(
    shader_stage: spirv_cross2::spirv::ExecutionModel
) -> RafxResult<RafxShaderStageFlags> {
    Ok(match shader_stage {
        ExecutionModel::Vertex => RafxShaderStageFlags::VERTEX,
        ExecutionModel::TessellationControl => RafxShaderStageFlags::TESSELLATION_CONTROL,
        ExecutionModel::TessellationEvaluation => RafxShaderStageFlags::TESSELLATION_EVALUATION,
        ExecutionModel::Geometry => RafxShaderStageFlags::GEOMETRY,
        ExecutionModel::Fragment => RafxShaderStageFlags::FRAGMENT,
        ExecutionModel::GLCompute => RafxShaderStageFlags::COMPUTE,
        ExecutionModel::Kernel => RafxShaderStageFlags::COMPUTE,
        ExecutionModel::TaskNV | ExecutionModel::TaskEXT => RafxShaderStageFlags::AMPLIFICATION,
        ExecutionModel::MeshEXT | ExecutionModel::MeshNV => RafxShaderStageFlags::MESH,
        ExecutionModel::RayGenerationNV => todo!(),
        ExecutionModel::IntersectionNV => todo!(),
        ExecutionModel::AnyHitNV => todo!(),
        ExecutionModel::ClosestHitNV => todo!(),
        ExecutionModel::MissNV => todo!(),
        ExecutionModel::CallableNV => todo!(),
    })
}
