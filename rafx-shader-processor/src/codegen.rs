use crate::parse_declarations::{
    BindingType, ParseDeclarationsResult, ParseFieldResult, ParsedBindingWithAnnotations,
};
use crate::reflect::ShaderProcessorReflectionData;
use crate::{
    shader_types::{self, *},
    CompileResult,
};
use fnv::{FnvHashMap, FnvHashSet};
use rafx_api::{RafxReflectedEntryPoint, RafxResourceType};
use std::collections::BTreeMap;

// Structs can be used in one of these three ways. The usage will determine the memory layout
#[derive(Copy, Clone, Debug)]
enum StructBindingType {
    Uniform,
    Buffer,
    PushConstant,
}

// Determine the binding type of a struct based on parsed code
fn determine_binding_type(b: &ParsedBindingWithAnnotations) -> Result<StructBindingType, String> {
    if b.parsed.layout_parts.push_constant {
        Ok(StructBindingType::PushConstant)
    } else if b.parsed.binding_type == BindingType::Uniform {
        Ok(StructBindingType::Uniform)
    } else if b.parsed.binding_type == BindingType::Buffer {
        Ok(StructBindingType::Buffer)
    } else {
        Err("Unknown binding type".to_string())
    }
}

// Binding type determines memory layout that gets used
fn determine_memory_layout(binding_struct_type: StructBindingType) -> MemoryLayout {
    match binding_struct_type {
        StructBindingType::Uniform => MemoryLayout::Std140,
        StructBindingType::Buffer => MemoryLayout::Std430,
        StructBindingType::PushConstant => MemoryLayout::Std430,
    }
}

/// Map GLSL field names in a Vertex struct to channel bits.
/// Returns Err for unrecognized non-padding fields.
fn field_name_to_channel_bit(name: &str) -> Result<Option<u32>, String> {
    match name {
        "pos" | "position" => Ok(Some(1 << 0)), // POSITION
        "normal" => Ok(Some(1 << 1)),           // NORMAL
        "tangent" => Ok(Some(1 << 2)),          // TANGENT
        "uv" | "uv0" | "texcoord" | "texcoord0" => Ok(Some(1 << 3)), // UV0
        "uv2" | "uv1" | "texcoord1" => Ok(Some(1 << 4)),             // UV1
        "color" | "colour" => Ok(Some(1 << 5)), // COLOR
        _ if name.starts_with("_padding") => Ok(None),
        _ => Err(format!(
            "Unrecognized vertex field '{}' in VertexBuffer struct. \
             Recognized names: pos, normal, tangent, uv, uv2, color",
            name
        )),
    }
}

fn compute_channels_from_fields(fields: &[ParseFieldResult]) -> Result<u32, String> {
    let mut bitmask = 0u32;
    for field in fields {
        if let Some(bit) = field_name_to_channel_bit(&field.field_name)? {
            bitmask |= bit;
        }
    }
    Ok(bitmask)
}

/// Compute vertex channel bitmask by scanning all compile results for a
/// `buffer readonly VertexBuffer` binding (instance_name == "vertex_buffer").
///
/// The SSBO binding typically looks like:
///   layout(...) buffer readonly VertexBuffer { Vertex vertices[]; } vertex_buffer;
/// The binding's inline fields contain a single array field whose type_name
/// is the actual Vertex struct. We look up that struct's fields for channels.
fn compute_vertex_channels(compile_results: &[CompileResult]) -> Result<Option<u32>, String> {
    for cr in compile_results {
        for binding in &cr.parsed_declarations.bindings {
            if binding.parsed.instance_name != "vertex_buffer" {
                continue;
            }
            if binding.parsed.binding_type != BindingType::Buffer {
                continue;
            }

            // The binding has inline fields (e.g. `Vertex vertices[]`).
            // Find the array element's struct type and look up its fields.
            if let Some(inline_fields) = &binding.parsed.fields {
                for field in inline_fields.iter() {
                    // Look up the struct referenced by this field's type
                    if let Some(s) = cr
                        .parsed_declarations
                        .structs
                        .iter()
                        .find(|s| s.parsed.type_name == field.type_name)
                    {
                        return Ok(Some(compute_channels_from_fields(&s.parsed.fields)?));
                    }
                }
                // If no struct was found, try the fields directly (unlikely for SSBOs)
                return Ok(Some(compute_channels_from_fields(inline_fields)?));
            }

            // No inline fields — try the binding's type_name as a struct
            if let Some(s) = cr
                .parsed_declarations
                .structs
                .iter()
                .find(|s| s.parsed.type_name == binding.parsed.type_name)
            {
                return Ok(Some(compute_channels_from_fields(&s.parsed.fields)?));
            }
        }
    }
    Ok(None)
}

/// Public entry point for lib.rs when rust codegen is disabled.
pub(crate) fn compute_vertex_channels_from_results(
    compile_results: &[CompileResult],
) -> Option<u32> {
    match compute_vertex_channels(compile_results) {
        Ok(channels) => channels,
        Err(e) => {
            log::error!("Failed to compute vertex channels: {}", e);
            None
        }
    }
}

/// Returns (generated_rust_code, vertex_channels_bitmask).
pub(crate) fn generate_rust_code(
    pipeline_name: String,
    compile_results: &[CompileResult],
    reflection_data: &ShaderProcessorReflectionData,
    for_rafx_framework_crate: bool,
) -> Result<(String, Option<u32>), String> {
    // builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    let mut all_structs = FnvHashMap::default();
    for s in compile_results
        .iter()
        .flat_map(|c| &c.parsed_declarations.structs)
    {
        let _existing_struct = all_structs
            .entry(s.parsed.type_name.clone())
            .or_insert_with(|| s.clone());
    }
    let mut all_bindings = FnvHashMap::default();
    for b in compile_results
        .iter()
        .flat_map(|c| &c.parsed_declarations.bindings)
    {
        let existing_binding = all_bindings
            .entry(b.parsed.instance_name.clone())
            .or_insert_with(|| b.clone());

        match existing_binding.parsed.binding_type {
            BindingType::Uniform | BindingType::Buffer => {
                if existing_binding.parsed.layout_parts != b.parsed.layout_parts {
                    Err(format!("Binding {} in pipeline {} has different layout parts in different files: {:?} vs {:?}", existing_binding.parsed.instance_name, pipeline_name, existing_binding.parsed.layout_parts, b.parsed.layout_parts ))?;
                }
            }
            _ => {}
        }
    }

    let first_group_size = compile_results
        .iter()
        .filter_map(|c| c.parsed_declarations.group_size.clone())
        .next();
    let builtin_types = shader_types::create_builtin_type_lookup();
    let declarations = ParseDeclarationsResult {
        structs: all_structs.into_values().collect(),
        bindings: all_bindings.into_values().collect(),
        group_size: first_group_size,
    };
    let mut user_types = shader_types::create_user_type_lookup(&declarations)?;
    // parsed_declarations: &ParseDeclarationsResult,

    for compile_result in compile_results {
        // Any struct that's explicitly exported will produce all layouts
        for s in &compile_result.parsed_declarations.structs {
            if s.annotations.export.is_some() {
                recursive_modify_user_type(&mut user_types, &s.parsed.type_name, &|udt| {
                    let already_marked = udt.export_uniform_layout
                        && udt.export_push_constant_layout
                        && udt.export_buffer_layout;
                    udt.export_uniform_layout = true;
                    udt.export_push_constant_layout = true;
                    udt.export_buffer_layout = true;
                    !already_marked
                });
            }
        }

        //
        // Bindings can either be std140 (uniform) or std430 (push constant/buffer). Depending on the
        // binding, enable export for just the type that we need
        //
        for b in &compile_result.parsed_declarations.bindings {
            if b.annotations.export.is_some() {
                match determine_binding_type(b)? {
                    StructBindingType::PushConstant => {
                        recursive_modify_user_type(&mut user_types, &b.parsed.type_name, &|udt| {
                            let already_marked = udt.export_push_constant_layout;
                            udt.export_push_constant_layout = true;
                            !already_marked
                        });
                    }
                    StructBindingType::Uniform => {
                        recursive_modify_user_type(&mut user_types, &b.parsed.type_name, &|udt| {
                            let already_marked = udt.export_uniform_layout;
                            udt.export_uniform_layout = true;
                            !already_marked
                        });
                    }
                    StructBindingType::Buffer => {
                        recursive_modify_user_type(&mut user_types, &b.parsed.type_name, &|udt| {
                            let already_marked = udt.export_buffer_layout;
                            udt.export_buffer_layout = true;
                            !already_marked
                        });
                    }
                }
            }
        }
    }

    let vertex_channels = compute_vertex_channels(compile_results)?;
    let code = generate_rust_file(&declarations, &builtin_types, &user_types, vertex_channels)?;
    Ok((code, vertex_channels))
}

fn generate_rust_file(
    parsed_declarations: &ParseDeclarationsResult,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    vertex_channels: Option<u32>,
) -> Result<String, String> {
    let mut rust_code = Vec::<String>::default();

    //rust_header(&mut rust_code, for_rafx_framework_crate);

    let structs = rust_structs(&mut rust_code, builtin_types, user_types)?;

    rust_binding_constants(&mut rust_code, &parsed_declarations);

    /*rust_binding_wrappers(
        &mut rust_code,
        builtin_types,
        user_types,
        &parsed_declarations,
        reflected_entry_point,
    )?;*/

    //rust_tests(&mut rust_code, &structs);

    let mut rust_code_str = String::default();
    for s in rust_code {
        rust_code_str += &s;
    }

    Ok(rust_code_str)
}

fn rust_header(
    rust_code: &mut Vec<String>,
    for_rafx_framework_crate: bool,
) {
    rust_code.push("// This code is auto-generated by the shader processor.\n\n".to_string());

    if for_rafx_framework_crate {
        rust_code.push("#[allow(unused_imports)]\n".to_string());
        rust_code.push("use rafx_api::RafxResult;\n\n".to_string());
        rust_code.push("#[allow(unused_imports)]\n".to_string());
        rust_code.push("use crate::{ResourceArc, ImageViewResource, DynDescriptorSet, DescriptorSetAllocator, DescriptorSetInitializer, DescriptorSetArc, DescriptorSetWriter, DescriptorSetWriterContext, DescriptorSetBindings};\n\n".to_string());
    } else {
        rust_code.push("#[allow(unused_imports)]\n".to_string());
        rust_code.push("use rafx::RafxResult;\n\n".to_string());
        rust_code.push("#[allow(unused_imports)]\n".to_string());
        rust_code.push("use rafx::framework::{ResourceArc, ImageViewResource, DynDescriptorSet, DescriptorSetAllocator, DescriptorSetInitializer, DescriptorSetArc, DescriptorSetWriter, DescriptorSetWriterContext, DescriptorSetBindings};\n\n".to_string());
    }
}

fn rust_structs(
    rust_code: &mut Vec<String>,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
) -> Result<Vec<GenerateStructResult>, String> {
    let mut structs = Vec::default();
    for (type_name, user_type) in user_types {
        if user_type.export_uniform_layout {
            let s = generate_struct(
                &builtin_types,
                &user_types,
                type_name,
                user_type,
                MemoryLayout::Std140,
            )?;
            rust_code.push(generate_struct_code(&s));
            rust_code.push(generate_struct_default_code(&s));
            structs.push(s);
        }

        if user_type.export_uniform_layout {
            rust_code.push(format!(
                "pub type {} = {};\n\n",
                get_rust_type_name_alias(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    &[],
                    StructBindingType::Uniform
                )?,
                get_rust_type_name(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    MemoryLayout::Std140,
                    &[]
                )?
            ));
        }

        if user_type.export_push_constant_layout || user_type.export_buffer_layout {
            let s = generate_struct(
                &builtin_types,
                &user_types,
                type_name,
                user_type,
                MemoryLayout::Std430,
            )?;
            rust_code.push(generate_struct_code(&s));
            structs.push(s);
        }

        if user_type.export_push_constant_layout {
            rust_code.push(format!(
                "pub type {} = {};\n\n",
                get_rust_type_name_alias(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    &[],
                    StructBindingType::PushConstant
                )?,
                get_rust_type_name(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    MemoryLayout::Std430,
                    &[]
                )?
            ));
        }
        if user_type.export_buffer_layout {
            rust_code.push(format!(
                "pub type {} = {};\n\n",
                get_rust_type_name_alias(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    &[],
                    StructBindingType::Buffer
                )?,
                get_rust_type_name(
                    builtin_types,
                    user_types,
                    &user_type.type_name,
                    MemoryLayout::Std430,
                    &[]
                )?
            ));
        }
    }

    Ok(structs)
}

fn rust_binding_wrappers(
    rust_code: &mut Vec<String>,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    parsed_declarations: &ParseDeclarationsResult,
    reflected_entry_point: &RafxReflectedEntryPoint,
) -> Result<(), String> {
    let mut bindings_by_set =
        BTreeMap::<usize, BTreeMap<usize, &ParsedBindingWithAnnotations>>::default();
    for binding in &parsed_declarations.bindings {
        if let Some(set) = binding.parsed.layout_parts.set {
            if let Some(binding_index) = binding.parsed.layout_parts.binding {
                bindings_by_set
                    .entry(set)
                    .or_default()
                    .insert(binding_index, binding);
            }
        }
    }

    for (set_index, bindings) in bindings_by_set {
        let mut binding_wrapper_items = Vec::default();
        let mut binding_wrapper_struct_lifetimes = Vec::default();
        for (binding_index, binding) in bindings {
            create_binding_wrapper_binding_item(
                &mut binding_wrapper_items,
                &mut binding_wrapper_struct_lifetimes,
                user_types,
                builtin_types,
                reflected_entry_point,
                set_index,
                binding_index,
                binding,
            )?;
        }

        if binding_wrapper_items.is_empty() {
            continue;
        }

        let wrapper_struct_name = format!("DescriptorSet{}", set_index);

        let args_struct_name = format!("DescriptorSet{}Args", set_index);
        let wrapper_args_generic_params = if binding_wrapper_struct_lifetimes.is_empty() {
            String::default()
        } else {
            let mut set = FnvHashSet::default();
            let mut unique_lifetimes = Vec::default();

            for lifetime in binding_wrapper_struct_lifetimes {
                if set.insert(lifetime.clone()) {
                    unique_lifetimes.push(lifetime);
                }
            }

            format!("<{}>", unique_lifetimes.join(", "))
        };

        //
        // Args to create the descriptor set
        //
        rust_code.push(format!(
            "pub struct {}{} {{\n",
            args_struct_name, wrapper_args_generic_params
        ));
        for item in &binding_wrapper_items {
            rust_code.push(format!(
                "    pub {}: {},\n",
                item.binding_name, item.args_struct_member_type
            ));
        }
        rust_code.push("}\n\n".to_string());

        //
        // DescriptorSetInitializer trait impl
        //

        rust_code.push(format!(
            "impl<'a> DescriptorSetInitializer<'a> for {}{} {{\n",
            args_struct_name, wrapper_args_generic_params
        ));
        rust_code.push(format!("    type Output = {};\n\n", wrapper_struct_name));

        // create_dyn_descriptor_set
        rust_code.push("    fn create_dyn_descriptor_set(descriptor_set: DynDescriptorSet, args: Self) -> Self::Output {\n".to_string());
        rust_code.push(format!(
            "        let mut descriptor = {}(descriptor_set);\n",
            wrapper_struct_name
        ));
        rust_code.push("        descriptor.set_args(args);\n".to_string());
        rust_code.push("        descriptor\n".to_string());
        rust_code.push("    }\n\n".to_string());

        // create_descriptor_set
        rust_code.push("    fn create_descriptor_set(descriptor_set_allocator: &mut DescriptorSetAllocator, descriptor_set: DynDescriptorSet, args: Self) -> RafxResult<DescriptorSetArc> {\n".to_string());
        rust_code.push(
            "        let mut descriptor = Self::create_dyn_descriptor_set(descriptor_set, args);\n"
                .to_string(),
        );
        rust_code.push("        descriptor.0.flush(descriptor_set_allocator)?;\n".to_string());
        rust_code.push("        Ok(descriptor.0.descriptor_set().clone())\n".to_string());
        rust_code.push("    }\n".to_string());

        rust_code.push("}\n\n".to_string());

        //
        // DescriptorSetWriter trait impl
        //

        rust_code.push(format!(
            "impl<'a> DescriptorSetWriter<'a> for {}{} {{\n",
            args_struct_name, wrapper_args_generic_params
        ));

        // write_to
        rust_code.push(
            "    fn write_to(descriptor_set: &mut DescriptorSetWriterContext, args: Self) {\n"
                .to_string(),
        );
        for item in &binding_wrapper_items {
            if item.descriptor_count == 1 {
                rust_code.push(format!(
                    "        descriptor_set.{}({}, args.{});\n",
                    item.setter_fn_name_single, item.binding_index_string, item.binding_name
                ));
            } else {
                rust_code.push(format!(
                    "        descriptor_set.{}({}, args.{});\n",
                    item.setter_fn_name_multi, item.binding_index_string, item.binding_name
                ));
            }
        }
        rust_code.push("    }\n".to_string());
        rust_code.push("}\n\n".to_string());

        //
        // Wrapper struct
        //
        rust_code.push(format!(
            "pub struct {}(pub DynDescriptorSet);\n\n",
            wrapper_struct_name
        ));

        rust_code.push(format!("impl {} {{\n", wrapper_struct_name));

        //
        // set_args_static()
        //
        rust_code.push(format!(
            "    pub fn set_args_static(descriptor_set: &mut DynDescriptorSet, args: {}) {{\n",
            args_struct_name
        ));
        for item in &binding_wrapper_items {
            if item.descriptor_count == 1 {
                rust_code.push(format!(
                    "        descriptor_set.{}({}, args.{});\n",
                    item.setter_fn_name_single, item.binding_index_string, item.binding_name
                ));
            } else {
                rust_code.push(format!(
                    "        descriptor_set.{}({}, args.{});\n",
                    item.setter_fn_name_multi, item.binding_index_string, item.binding_name
                ));
            }
        }
        rust_code.push("    }\n\n".to_string());

        //
        // set_args()
        //
        rust_code.push(format!(
            "    pub fn set_args(&mut self, args: {}) {{\n",
            args_struct_name
        ));
        for item in &binding_wrapper_items {
            rust_code.push(format!(
                "        self.set_{}(args.{});\n",
                item.binding_name, item.binding_name
            ));
        }
        rust_code.push("    }\n\n".to_string());

        //
        // setters for individual bindings
        //
        //TODO: Make this support arrays
        for item in &binding_wrapper_items {
            if item.descriptor_count == 1 {
                //
                // Set the value
                //
                rust_code.push(format!(
                    "    pub fn set_{}(&mut self, {}: {}) {{\n",
                    item.binding_name, item.binding_name, item.set_element_param_type_single
                ));
                rust_code.push(format!(
                    "        self.0.{}({}, {});\n",
                    item.setter_fn_name_single, item.binding_index_string, item.binding_name
                ));
                rust_code.push("    }\n\n".to_string());
            } else if item.descriptor_count > 1 {
                //
                // Set all the values
                //
                rust_code.push(format!(
                    "    pub fn set_{}(&mut self, {}: {}) {{\n",
                    item.binding_name, item.binding_name, item.set_element_param_type_multi
                ));
                rust_code.push(format!(
                    "        self.0.{}({}, {});\n",
                    item.setter_fn_name_multi, item.binding_index_string, item.binding_name
                ));
                rust_code.push("    }\n\n".to_string());

                //
                // Set one of the values
                //
                rust_code.push(format!(
                    "    pub fn set_{}_element(&mut self, index: usize, element: {}) {{\n",
                    item.binding_name, item.set_element_param_type_single
                ));
                rust_code.push(format!(
                    "        self.0.{}({}, index, element);\n",
                    item.setter_fn_name_single, item.binding_index_string
                ));
                rust_code.push("    }\n\n".to_string());
            }
        }

        //
        // flush
        //
        rust_code.push("    pub fn flush(&mut self, descriptor_set_allocator: &mut DescriptorSetAllocator) -> RafxResult<()> {\n".to_string());
        rust_code.push("        self.0.flush(descriptor_set_allocator)\n".to_string());
        rust_code.push("    }\n".to_string());

        rust_code.push("}\n\n".to_string());
    }

    Ok(())
}

struct BindingWrapperItem {
    binding_name: String,
    setter_fn_name_single: String,
    setter_fn_name_multi: String,
    args_struct_member_type: String,
    set_element_param_type_single: String,
    set_element_param_type_multi: String,
    binding_index_string: String,
    descriptor_count: u32,
}

fn create_binding_wrapper_binding_item(
    binding_wrapper_items: &mut Vec<BindingWrapperItem>,
    binding_wrapper_struct_lifetimes: &mut Vec<String>,
    user_types: &FnvHashMap<String, UserType>,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    reflected_entry_point: &RafxReflectedEntryPoint,
    set_index: usize,
    binding_index: usize,
    binding: &ParsedBindingWithAnnotations,
) -> Result<(), String> {
    // Don't generate member function for this binding if the type wasn't exported
    if !binding.annotations.export.is_some() {
        return Ok(());
    }

    // Find the binding in the reflection data
    let e = reflected_entry_point
        .descriptor_set_layouts
        .get(set_index)
        .ok_or_else(|| {
            format!(
                "Could not find descriptor set index {} in reflection data",
                set_index
            )
        })?
        .as_ref()
        .ok_or_else(|| {
            format!(
                "Could not find descriptor set index {} in reflection data",
                set_index
            )
        })?
        .bindings
        .iter()
        .find(|x| x.resource.binding == binding_index as u32)
        .ok_or_else(|| {
            format!(
                "Could not find descriptor binding index {} in reflection data",
                binding_index
            )
        })?;

    use heck::SnakeCase;
    let binding_name = binding.parsed.instance_name.to_snake_case();
    let binding_index_string =
        format!("{}.binding", descriptor_constant_name(binding));

    if e.immutable_samplers.is_none()
        && e.resource.resource_type == RafxResourceType::COMBINED_IMAGE_SAMPLER
    {
        Err("Combined image samplers only supported with immutable samplers")?;
    }

    match e.resource.resource_type {
        RafxResourceType::SAMPLER => {
            if e.immutable_samplers.is_none() {
                // TODO: Generate a setter for samplers
            }
        }
        RafxResourceType::TEXTURE
        | RafxResourceType::TEXTURE_READ_WRITE
        | RafxResourceType::COMBINED_IMAGE_SAMPLER => {
            if e.resource.element_count_normalized() > 1 {
                binding_wrapper_items.push(BindingWrapperItem {
                    binding_name,
                    setter_fn_name_single: "set_image_at_index".to_string(),
                    setter_fn_name_multi: "set_images".to_string(),
                    args_struct_member_type: format!(
                        "&'a [Option<&'a ResourceArc<ImageViewResource>>; {}]",
                        e.resource.element_count_normalized()
                    )
                    .to_string(),
                    set_element_param_type_single: "&ResourceArc<ImageViewResource>".to_string(),
                    set_element_param_type_multi: format!(
                        "&[Option<& ResourceArc<ImageViewResource>>; {}]",
                        e.resource.element_count_normalized()
                    )
                    .to_string(),
                    binding_index_string,
                    descriptor_count: e.resource.element_count_normalized(),
                });
            } else {
                binding_wrapper_items.push(BindingWrapperItem {
                    binding_name,
                    setter_fn_name_single: "set_image".to_string(),
                    setter_fn_name_multi: "set_images".to_string(),
                    args_struct_member_type: "&'a ResourceArc<ImageViewResource>".to_string(),
                    set_element_param_type_single: "&ResourceArc<ImageViewResource>".to_string(),
                    set_element_param_type_multi: "&[& ResourceArc<ImageViewResource>]".to_string(),
                    binding_index_string,
                    descriptor_count: e.resource.element_count_normalized(),
                });
            }
            binding_wrapper_struct_lifetimes.push("'a".to_string());
        }
        RafxResourceType::UNIFORM_BUFFER
        | RafxResourceType::BUFFER
        | RafxResourceType::BUFFER_READ_WRITE => {
            assert_eq!(e.resource.element_count_normalized(), 1);
            let type_name = get_rust_type_name_alias(
                builtin_types,
                user_types,
                &binding.parsed.type_name,
                &binding.parsed.array_sizes,
                determine_binding_type(binding)?,
            )?;
            binding_wrapper_items.push(BindingWrapperItem {
                binding_name,
                setter_fn_name_single: "set_buffer_data".to_string(),
                setter_fn_name_multi: "set_buffer_data".to_string(),
                args_struct_member_type: format!("&'a {}", type_name),
                set_element_param_type_single: format!("&{}", type_name),
                set_element_param_type_multi: format!("&'[{}]", type_name),
                binding_index_string,
                descriptor_count: e.resource.element_count_normalized(),
            });
            binding_wrapper_struct_lifetimes.push("'a".to_string());
        }
        // No support for these yet
        // RafxResourceType::UniformBufferDynamic => {}
        // RafxResourceType::StorageBufferDynamic => {}
        // RafxResourceType::UniformTexelBuffer => {}
        // RafxResourceType::StorageTexelBuffer => {}
        // RafxResourceType::InputAttachment => {}
        _ => {
            Err(format!(
                "Unsupported resource type {:?}",
                e.resource.resource_type
            ))?;
        }
    };

    Ok(())
}

fn descriptor_constant_name(binding: &ParsedBindingWithAnnotations) -> String {
    use heck::ShoutySnakeCase;
    binding.parsed.instance_name.to_shouty_snake_case()
}

fn rust_binding_constants(
    rust_code: &mut Vec<String>,
    parsed_declarations: &ParseDeclarationsResult,
) {
    for binding in &parsed_declarations.bindings {
        if let (Some(set_index), Some(binding_index)) = (
            binding.parsed.layout_parts.set,
            binding.parsed.layout_parts.binding,
        ) {
            rust_code.push(format!(
                "pub const {}: crate::ShaderResourceBindingKey = crate::ShaderResourceBindingKey {{ set: {}, binding: {} }};\n",
                descriptor_constant_name(binding),
                set_index,
                binding_index,
            ));
        }
    }

    rust_code.push("\n".to_string());
}

fn rust_tests(
    rust_code: &mut Vec<String>,
    structs: &[GenerateStructResult],
) {
    if !structs.is_empty() {
        rust_code.push("#[cfg(test)]\nmod test {\n    use super::*;\n".to_string());
        for s in structs {
            rust_code.push(generate_struct_test_code(&s));
        }
        rust_code.push("}\n".to_string());
    }
}

fn get_rust_type_name(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    name: &str,
    layout: MemoryLayout,
    array_sizes: &[usize],
) -> Result<String, String> {
    let type_name = get_rust_type_name_non_array(builtin_types, user_types, name, layout)?;

    Ok(wrap_in_array(&type_name, array_sizes))
}

fn get_rust_type_name_alias(
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
    name: &str,
    array_sizes: &[usize],
    binding_struct_type: StructBindingType,
) -> Result<String, String> {
    let layout = determine_memory_layout(binding_struct_type);
    let alias_name = format!("{:?}", binding_struct_type);

    if builtin_types.contains_key(name) {
        get_rust_type_name(builtin_types, user_types, name, layout, array_sizes)
    } else if let Some(user_type) = user_types.get(name) {
        Ok(format!(
            "{}{}{}",
            user_type.type_name.clone(),
            alias_name,
            format_array_sizes(array_sizes)
        ))
    } else {
        Err(format!("Could not find type {}. Is this a built in type that needs to be added to create_builtin_type_lookup()?", name))
    }
}

fn generate_struct_code(st: &GenerateStructResult) -> String {
    let mut result_string = String::default();
    result_string += &format!(
        "#[derive(Copy, Clone, Debug)]\n#[repr(C)]\npub struct {} {{\n",
        st.name
    );
    for m in &st.members {
        result_string += &format_member(&m.name, &m.ty, m.offset, m.size);
    }
    result_string += &format!("}} // {} bytes\n\n", st.size);
    result_string
}

fn generate_struct_default_code(st: &GenerateStructResult) -> String {
    let mut result_string = String::default();
    result_string += &format!("impl Default for {} {{\n", st.name);
    result_string += &format!("    fn default() -> Self {{\n");
    result_string += &format!("        {} {{\n", st.name);
    for m in &st.members {
        //result_string += &format!("            {}: {}::default(),\n", &m.name, &m.ty);
        result_string += &format!("            {}: {},\n", &m.name, m.default_value);
    }
    result_string += &format!("        }}\n");
    result_string += &format!("    }}\n");
    result_string += &format!("}}\n\n");
    result_string
}

fn generate_struct_test_code(st: &GenerateStructResult) -> String {
    use heck::SnakeCase;
    let mut result_string = String::default();
    result_string += &format!(
        "\n    #[test]\n    fn test_struct_{}() {{\n",
        st.name.to_snake_case()
    );
    result_string += &format!(
        "        assert_eq!(std::mem::size_of::<{}>(), {});\n",
        st.name, st.size
    );
    for m in &st.members {
        result_string += &format!(
            "        assert_eq!(std::mem::size_of::<{}>(), {});\n",
            m.ty, m.size
        );
        result_string += &format!(
            "        assert_eq!(std::mem::align_of::<{}>(), {});\n",
            m.ty, m.align
        );

        // Very large structs may be larger than can fit on the stack, which doesn't work with memoffset::offset_of!()
        if st.size < (1024 * 1024) {
            result_string += &format!(
                "        assert_eq!(memoffset::offset_of!({}, {}), {});\n",
                st.name, m.name, m.offset
            );
        }
    }
    result_string += &format!("    }}\n");
    result_string
}

fn format_member(
    name: &str,
    ty: &str,
    offset: usize,
    size: usize,
) -> String {
    let mut str = format!("    pub {}: {}, ", name, ty);
    let whitespace = 40_usize.saturating_sub(str.len());
    str += " ".repeat(whitespace).as_str();
    str += &format!("// +{} (size: {})\n", offset, size);
    str
}
