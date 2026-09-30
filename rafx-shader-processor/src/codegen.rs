use crate::parse_declarations::{
    BindingType, ParseDeclarationsResult, ParseFieldResult, ParsedBindingWithAnnotations,
};
use crate::{
    shader_types::{self, *},
    CompileResult,
};
use fnv::FnvHashMap;

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
        "uv2" | "uv1" | "texcoord1" => Ok(Some(1 << 4)), // UV1
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
    compile_results: &[CompileResult]
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
    let code = generate_rust_file(&declarations, &builtin_types, &user_types)?;
    Ok((code, vertex_channels))
}

fn generate_rust_file(
    parsed_declarations: &ParseDeclarationsResult,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
) -> Result<String, String> {
    let mut rust_code = Vec::<String>::default();

    rust_structs(&mut rust_code, builtin_types, user_types)?;

    rust_binding_constants(&mut rust_code, &parsed_declarations);

    let mut rust_code_str = String::default();
    for s in rust_code {
        rust_code_str += &s;
    }

    Ok(rust_code_str)
}

fn rust_structs(
    rust_code: &mut Vec<String>,
    builtin_types: &FnvHashMap<String, TypeAlignmentInfo>,
    user_types: &FnvHashMap<String, UserType>,
) -> Result<(), String> {
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
