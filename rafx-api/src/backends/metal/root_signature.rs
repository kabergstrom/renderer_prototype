use crate::metal::RafxDeviceContextMetal;
use crate::{
    RafxDescriptorIndex, RafxPipelineType, RafxResourceType, RafxResult, RafxRootSignatureDef,
    RafxShaderStageFlags, ALL_SHADER_STAGE_FLAGS, MAX_DESCRIPTOR_SET_LAYOUTS,
};
use cocoa_foundation::foundation::NSUInteger;
use fnv::FnvHashMap;
use metal_rs::{MTLResourceUsage, MTLTextureType};
use std::sync::Arc;

//TODO: Could compact this down quite a bit
#[derive(Clone, Debug)]
pub(crate) struct DescriptorInfo {
    pub(crate) name: Option<String>,
    pub(crate) resource_type: RafxResourceType,
    // Only valid for textures
    //pub(crate) texture_dimensions: Option<RafxTextureDimension>,

    // Also the set layout
    pub(crate) set_index: u32,
    // Binding within the set
    pub(crate) binding: u32,
    // Used for arrays of textures, samplers, etc.
    pub(crate) element_count: u32,

    pub(crate) push_constant_size: u32,
    pub(crate) used_in_shader_stages: RafxShaderStageFlags,

    // Index into DescriptorSetLayoutInfo::descriptors list
    // NOT THE BINDING INDEX!!!
    //pub(crate) descriptor_index: RafxDescriptorIndex,

    // --- metal-specific ---
    //pub(crate) usage: metal::MTLResourceUsage,
    // Push constants: the buffer index. Otherwise its index over the set's descriptors of
    // every stage, for residency (ArgumentBufferData::resource_pointers); the shaders read it
    // at its StageRegion ids.
    pub(crate) argument_buffer_id: NSUInteger,
}

#[derive(Default, Debug)]
pub(crate) struct DescriptorSetLayoutInfo {
    // Settable descriptors, immutable samplers are omitted
    pub(crate) descriptors: Vec<RafxDescriptorIndex>,
    // Indexes binding index to the descriptors list
    pub(crate) binding_to_descriptor_index: FnvHashMap<u32, RafxDescriptorIndex>,

    // --- metal-specific ---
    // All argument buffer IDs must be within 0..argument_buffer_id_range
    pub(crate) argument_buffer_id_range: u32,
    // The set's argument buffer layouts as the shader stages see them (StageRegion).
    pub(crate) regions: Vec<StageRegion>,
    // pub(crate) sampler_count: u32,
    // pub(crate) texture_count: u32,
    // pub(crate) buffer_count: u32,
}

/// One layout of a set's argument buffer, shared by the stages in `stages`. The shader
/// processor compiles each stage alone (rafx-shader-processor msl_assign_argument_buffer_ids),
/// numbering the set's resources that stage declares in binding order, so a stage that lacks
/// one of a lower binding than another stage's numbers the rest differently. Each distinct
/// numbering gets a region of its own in the set's argument buffer, encoded with the same
/// resources at its ids, and its stages bind the buffer at that region.
#[derive(Debug)]
pub(crate) struct StageRegion {
    pub(crate) stages: RafxShaderStageFlags,
    pub(crate) argument_descriptors: Vec<ArgumentDescriptor>,
    // The id in this region of each of the set's descriptors these stages declare.
    pub(crate) ids: FnvHashMap<RafxDescriptorIndex, u32>,
}

#[derive(Debug)]
pub(crate) struct ArgumentDescriptor {
    pub(crate) data_type: metal_rs::MTLDataType,
    pub(crate) index: u64,
    pub(crate) access: metal_rs::MTLArgumentAccess,
    pub(crate) array_length: u64,
    pub(crate) texture_type: MTLTextureType,
}

impl Into<metal_rs::ArgumentDescriptor> for &ArgumentDescriptor {
    fn into(self) -> metal_rs::ArgumentDescriptor {
        let argument_descriptor = metal_rs::ArgumentDescriptor::new().to_owned();
        argument_descriptor.set_access(self.access);
        argument_descriptor.set_array_length(self.array_length as _);
        argument_descriptor.set_data_type(self.data_type);
        argument_descriptor.set_index(self.index as _);
        argument_descriptor.set_texture_type(self.texture_type);
        argument_descriptor
    }
}

#[derive(Debug)]
pub(crate) struct RafxRootSignatureMetalInner {
    pub(crate) device_context: RafxDeviceContextMetal,
    pub(crate) pipeline_type: RafxPipelineType,
    pub(crate) layouts: [DescriptorSetLayoutInfo; MAX_DESCRIPTOR_SET_LAYOUTS],
    pub(crate) descriptors: Vec<DescriptorInfo>,
    pub(crate) name_to_descriptor_index: FnvHashMap<String, RafxDescriptorIndex>,
    pub(crate) push_constant_descriptors:
        [Option<RafxDescriptorIndex>; ALL_SHADER_STAGE_FLAGS.len()],

    // --- metal-specific ---
    pub(crate) argument_buffer_resource_usages:
        [Arc<Vec<MTLResourceUsage>>; MAX_DESCRIPTOR_SET_LAYOUTS],
    // Dynamic offsets buffer slots (crate::msl_dynamic_offset_index) of every buffer in each
    // set: a bind of the set rewrites them (0, or the bind's dynamic offset).
    pub(crate) buffer_dynamic_offset_slots: [Vec<u32>; MAX_DESCRIPTOR_SET_LAYOUTS],
    // Slots of the set's RafxRootSignatureDef::dynamic_buffer_bindings in binding order: the
    // order of a bind's dynamic offsets.
    pub(crate) dynamic_binding_slots: [Vec<u32>; MAX_DESCRIPTOR_SET_LAYOUTS],
    // Slots the shaders read: one past the highest buffer slot (0: no buffers).
    pub(crate) dynamic_offsets_len: u32,
}

// for metal_rs::ArgumentDescriptor
unsafe impl Send for RafxRootSignatureMetalInner {}
unsafe impl Sync for RafxRootSignatureMetalInner {}

#[derive(Clone, Debug)]
pub struct RafxRootSignatureMetal {
    pub(crate) inner: Arc<RafxRootSignatureMetalInner>,
}
impl PartialEq for RafxRootSignatureMetal {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        std::sync::Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl RafxRootSignatureMetal {
    pub fn device_context(&self) -> &RafxDeviceContextMetal {
        &self.inner.device_context
    }

    pub fn pipeline_type(&self) -> RafxPipelineType {
        self.inner.pipeline_type
    }

    pub fn find_descriptor_by_name(
        &self,
        name: &str,
    ) -> Option<RafxDescriptorIndex> {
        self.inner.name_to_descriptor_index.get(name).copied()
    }

    pub fn find_descriptor_by_binding(
        &self,
        set_index: u32,
        binding: u32,
    ) -> Option<RafxDescriptorIndex> {
        self.inner
            .layouts
            .get(set_index as usize)
            .and_then(|x| x.binding_to_descriptor_index.get(&binding))
            .copied()
    }

    pub fn find_push_constant_descriptor(
        &self,
        stage: RafxShaderStageFlags,
    ) -> Option<RafxDescriptorIndex> {
        let mut found_descriptor = None;
        for (stage_index, s) in ALL_SHADER_STAGE_FLAGS.iter().enumerate() {
            if s.intersects(stage) {
                let s_descriptor_index = self.inner.push_constant_descriptors[stage_index];
                if s_descriptor_index.is_some() {
                    if let Some(found_descriptor) = found_descriptor {
                        if found_descriptor != s_descriptor_index {
                            // The caller passed multiple stages and they do not use the same push constant descriptor
                            return None;
                        }
                    } else {
                        found_descriptor = Some(s_descriptor_index);
                    }
                }
            }
        }

        return found_descriptor.flatten();
    }

    pub(crate) fn descriptor(
        &self,
        descriptor_index: RafxDescriptorIndex,
    ) -> Option<&DescriptorInfo> {
        self.inner.descriptors.get(descriptor_index.0 as usize)
    }

    pub fn new(
        device_context: &RafxDeviceContextMetal,
        root_signature_def: &RafxRootSignatureDef,
    ) -> RafxResult<Self> {
        log::trace!("Create RafxRootSignatureMetal");

        // If we update this constant, update the arrays in this function
        assert_eq!(MAX_DESCRIPTOR_SET_LAYOUTS, 4);

        // let mut immutable_samplers = vec![];
        // for sampler_list in root_signature_def.immutable_samplers {
        //     for sampler in sampler_list.samplers {
        //         immutable_samplers.push(sampler.clone());
        //     }
        // }

        // Make sure all shaders are compatible/build lookup of shared data from them
        let (pipeline_type, mut merged_resources, _merged_resources_name_index_map) =
            crate::internal_shared::merge_resources(root_signature_def)?;

        merged_resources.sort_by(|lhs, rhs| lhs.binding.cmp(&rhs.binding));

        let mut layouts = [
            DescriptorSetLayoutInfo::default(),
            DescriptorSetLayoutInfo::default(),
            DescriptorSetLayoutInfo::default(),
            DescriptorSetLayoutInfo::default(),
        ];

        let mut resource_usages = [vec![], vec![], vec![], vec![]];
        let mut push_constant_descriptors = [None; ALL_SHADER_STAGE_FLAGS.len()];

        let mut next_argument_buffer_id = [0, 0, 0, 0];

        let mut descriptors = Vec::with_capacity(merged_resources.len());
        let mut name_to_descriptor_index = FnvHashMap::default();

        // Used to determine the arg buffer index for push constants
        let mut max_set_index = -1;
        for resource in &merged_resources {
            if resource.resource_type != RafxResourceType::ROOT_CONSTANT {
                max_set_index = max_set_index.max(resource.set_index as i32)
            }
        }

        for resource in &merged_resources {
            resource.validate()?;

            // Verify set index is valid

            if resource.resource_type != RafxResourceType::ROOT_CONSTANT {
                let immutable_sampler = crate::internal_shared::find_immutable_sampler_index(
                    root_signature_def.immutable_samplers,
                    &resource.name,
                    resource.set_index,
                    resource.binding,
                );

                // Check that if an immutable sampler is set, the array size matches the resource element count
                if let Some(immutable_sampler_index) = immutable_sampler {
                    if resource.element_count_normalized() as usize
                        != root_signature_def.immutable_samplers[immutable_sampler_index]
                            .samplers
                            .len()
                    {
                        Err(format!(
                            "Descriptor (set={:?} binding={:?}) named {:?} specifies {} elements but the count of provided immutable samplers ({}) did not match",
                            resource.set_index,
                            resource.binding,
                            resource.name,
                            resource.element_count_normalized(),
                            root_signature_def.immutable_samplers[immutable_sampler_index].samplers.len()
                        ))?;
                    }
                }

                let layout: &mut DescriptorSetLayoutInfo =
                    &mut layouts[resource.set_index as usize];

                let descriptor_index = RafxDescriptorIndex(descriptors.len() as u32);

                // An immutable sampler is a constexpr sampler in the shader: it takes no
                // argument buffer id (rafx-shader-processor msl_assign_argument_buffer_ids).
                let argument_buffer_id = next_argument_buffer_id[resource.set_index as usize];
                if immutable_sampler.is_none() {
                    next_argument_buffer_id[resource.set_index as usize] +=
                        resource.element_count_normalized();
                }

                //let update_data_offset_in_set = Some(layout.update_data_count_per_set);

                if let Some(_immutable_sampler_index) = immutable_sampler {
                    // This is now embedded by spirv_cross in the shader
                    // let samplers = root_signature_def
                    //     .immutable_samplers[immutable_sampler_index]
                    //     .samplers
                    //     .iter()
                    //     .map(|x| x.metal_sampler().unwrap().clone())
                    //     .collect();
                    //
                    // layout.immutable_samplers.push(ImmutableSampler {
                    //     //binding: resource.binding,
                    //     samplers,
                    //     argument_buffer_id: argument_buffer_id as _
                    // });
                } else {
                    // Add it to the descriptor list
                    descriptors.push(DescriptorInfo {
                        name: resource.name.clone(),
                        resource_type: resource.resource_type,
                        //texture_dimensions: resource.texture_dimensions,
                        set_index: resource.set_index,
                        binding: resource.binding,
                        element_count: resource.element_count_normalized(),
                        push_constant_size: 0,
                        used_in_shader_stages: resource.used_in_shader_stages,
                        //descriptor_index,
                        //immutable_sampler: immutable_sampler.map(|x| immutable_samplers[x].clone()),
                        //update_data_offset_in_set,
                        //usage
                        argument_buffer_id: argument_buffer_id as _,
                    });

                    if let Some(name) = resource.name.as_ref() {
                        name_to_descriptor_index.insert(name.clone(), descriptor_index);
                    }

                    layout.descriptors.push(descriptor_index);
                    layout
                        .binding_to_descriptor_index
                        .insert(resource.binding, descriptor_index);
                    layout.argument_buffer_id_range =
                        next_argument_buffer_id[resource.set_index as usize];

                    // Build out the MTLResourceUsage usages - it's used when we bind descriptor sets
                    let layout_resource_usages = &mut resource_usages[resource.set_index as usize];
                    layout_resource_usages.resize(
                        layout.argument_buffer_id_range as usize,
                        MTLResourceUsage::empty(),
                    );
                    let usage =
                        super::util::resource_type_mtl_resource_usage(resource.resource_type);
                    for i in argument_buffer_id..layout.argument_buffer_id_range {
                        layout_resource_usages[i as usize] = usage;
                    }

                    debug_assert_ne!(layout.argument_buffer_id_range, 0);
                }
            } else {
                let descriptor_index = RafxDescriptorIndex(descriptors.len() as u32);
                descriptors.push(DescriptorInfo {
                    name: resource.name.clone(),
                    resource_type: resource.resource_type,
                    //texture_dimensions: resource.texture_dimensions,
                    set_index: u32::MAX,
                    binding: u32::MAX,
                    element_count: 0,
                    push_constant_size: resource.size_in_bytes,
                    used_in_shader_stages: resource.used_in_shader_stages,
                    //descriptor_index,
                    //immutable_sampler: immutable_sampler.map(|x| immutable_samplers[x].clone()),
                    //update_data_offset_in_set,
                    //usage
                    argument_buffer_id: (max_set_index + 1) as _,
                });

                if let Some(name) = resource.name.as_ref() {
                    name_to_descriptor_index.insert(name.clone(), descriptor_index);
                }

                for (i, stage) in ALL_SHADER_STAGE_FLAGS.iter().enumerate() {
                    if stage.intersects(resource.used_in_shader_stages) {
                        push_constant_descriptors[i] = Some(descriptor_index);
                    }
                }
            }
        }

        // The stages' argument buffer layouts (StageRegion), the descriptors in binding order
        // as layouts[i].descriptors holds them (merged_resources is sorted by binding).
        for layout in &mut layouts {
            for stage in [
                RafxShaderStageFlags::VERTEX,
                RafxShaderStageFlags::FRAGMENT,
                RafxShaderStageFlags::COMPUTE,
            ] {
                let declared: Vec<RafxDescriptorIndex> = layout
                    .descriptors
                    .iter()
                    .copied()
                    .filter(|&d| {
                        descriptors[d.0 as usize]
                            .used_in_shader_stages
                            .intersects(stage)
                    })
                    .collect();
                if declared.is_empty() {
                    continue;
                }
                let same = layout.regions.iter_mut().find(|r| {
                    r.ids.len() == declared.len() && declared.iter().all(|d| r.ids.contains_key(d))
                });
                if let Some(region) = same {
                    region.stages |= stage;
                    continue;
                }
                let mut region = StageRegion {
                    stages: stage,
                    argument_descriptors: Vec::with_capacity(declared.len()),
                    ids: FnvHashMap::default(),
                };
                let mut next_id = 0;
                for d in declared {
                    let descriptor = &descriptors[d.0 as usize];
                    region.argument_descriptors.push(ArgumentDescriptor {
                        access: super::util::resource_type_mtl_argument_access(
                            descriptor.resource_type,
                        ),
                        array_length: descriptor.element_count as _,
                        data_type: super::util::resource_type_mtl_data_type(
                            descriptor.resource_type,
                        )
                        .unwrap(),
                        index: next_id as _,
                        //TODO: Temp, not sure if it's this gets changed when bound
                        texture_type: MTLTextureType::D2,
                    });
                    region.ids.insert(d, next_id);
                    next_id += descriptor.element_count;
                }
                layout.regions.push(region);
            }
        }

        let mut buffer_dynamic_offset_slots = [vec![], vec![], vec![], vec![]];
        let mut dynamic_binding_slots = [vec![], vec![], vec![], vec![]];
        let mut dynamic_offsets_len = 0;
        // descriptors are in binding order (merged_resources is sorted by binding)
        for descriptor in &descriptors {
            let is_buffer = descriptor.resource_type.intersects(
                RafxResourceType::UNIFORM_BUFFER
                    | RafxResourceType::BUFFER
                    | RafxResourceType::BUFFER_READ_WRITE,
            );
            if !is_buffer || descriptor.element_count != 1 {
                continue;
            }
            let Some(slot) =
                crate::msl_dynamic_offset_index(descriptor.set_index, descriptor.binding)
            else {
                continue;
            };
            let set = descriptor.set_index as usize;
            buffer_dynamic_offset_slots[set].push(slot);
            dynamic_offsets_len = dynamic_offsets_len.max(slot + 1);
            let is_dynamic = root_signature_def
                .dynamic_buffer_bindings
                .iter()
                .any(|k| k.set == descriptor.set_index && k.binding == descriptor.binding);
            if is_dynamic {
                dynamic_binding_slots[set].push(slot);
            }
        }
        for key in root_signature_def.dynamic_buffer_bindings {
            let found = descriptors
                .iter()
                .any(|d| d.set_index == key.set && d.binding == key.binding);
            let slotted =
                crate::msl_dynamic_offset_index(key.set, key.binding).map_or(false, |slot| {
                    dynamic_binding_slots
                        .get(key.set as usize)
                        .map_or(false, |slots| slots.contains(&slot))
                });
            if found && !slotted {
                Err(format!(
                    "dynamic_buffer_bindings entry (set={} binding={}) must be a non-array buffer with binding < {}",
                    key.set,
                    key.binding,
                    crate::MSL_DYNAMIC_OFFSET_BINDINGS_PER_SET
                ))?;
            }
        }

        let argument_buffer_resource_usages = [
            Arc::new(std::mem::take(&mut resource_usages[0])),
            Arc::new(std::mem::take(&mut resource_usages[1])),
            Arc::new(std::mem::take(&mut resource_usages[2])),
            Arc::new(std::mem::take(&mut resource_usages[3])),
        ];

        let inner = RafxRootSignatureMetalInner {
            device_context: device_context.clone(),
            pipeline_type,
            layouts,
            descriptors,
            name_to_descriptor_index,
            push_constant_descriptors,
            argument_buffer_resource_usages,
            buffer_dynamic_offset_slots,
            dynamic_binding_slots,
            dynamic_offsets_len,
        };

        Ok(RafxRootSignatureMetal {
            inner: Arc::new(inner),
        })
    }
}
