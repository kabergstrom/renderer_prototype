use spirv_cross2::compile::CompiledArtifact;
use spirv_cross2::targets::{Hlsl, Msl};
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

mod parse_source;
use parse_source::AnnotationText;
use parse_source::DeclarationText;

mod parse_declarations;

mod include;
use crate::parse_declarations::ParseDeclarationsResult;
use crate::parse_source::PreprocessorState;
use crate::reflect::{map_shader_stage_flags, ShaderProcessorReflectionData};
use fnv::{FnvHashMap, FnvHashSet};
use include::include_impl;
use include::IncludeType;
use rafx_api::{
    RafxHashedShaderPackage, RafxPipelinePackage, RafxPipelineVariant, RafxShaderPackage,
    RafxShaderPackageDx12, RafxShaderPackageGles2, RafxShaderPackageGles3,
    RafxShaderPackageMetal, RafxShaderPackageVulkan,
};
use shaderc::{CompilationArtifact, Compiler, ShaderKind};
use spirv_cross2::reflect::ShaderResources;

mod codegen;

mod reflect;

mod shader_types;

const PREPROCESSOR_DEF_PLATFORM_RUST_CODEGEN: &'static str = "PLATFORM_RUST_CODEGEN";
const PREPROCESSOR_DEF_PLATFORM_DX12: &'static str = "PLATFORM_DX12";
const PREPROCESSOR_DEF_PLATFORM_VULKAN: &'static str = "PLATFORM_VULKAN";
const PREPROCESSOR_DEF_PLATFORM_METAL: &'static str = "PLATFORM_METAL";
const PREPROCESSOR_DEF_PLATFORM_GLES2: &'static str = "PLATFORM_GLES2";
const PREPROCESSOR_DEF_PLATFORM_GLES3: &'static str = "PLATFORM_GLES3";

#[derive(Clone, Copy, Debug)]
enum RsFileType {
    Lib,
    Mod,
}

#[derive(Debug)]
struct RsFileOption {
    path: PathBuf,
    file_type: RsFileType,
}

#[derive(StructOpt, Debug)]
pub struct ShaderProcessorArgs {
    //
    // For one file at a time
    //
    #[structopt(name = "glsl-file", long, parse(from_os_str))]
    pub glsl_file: Option<PathBuf>,
    #[structopt(name = "spv-file", long, parse(from_os_str))]
    pub spv_file: Option<PathBuf>,
    #[structopt(name = "rs-file", long, parse(from_os_str))]
    pub rs_file: Option<PathBuf>,
    #[structopt(name = "dx12-generated-src-file", long, parse(from_os_str))]
    pub dx12_generated_src_file: Option<PathBuf>,
    #[structopt(name = "metal-generated-src-file", long, parse(from_os_str))]
    pub metal_generated_src_file: Option<PathBuf>,
    #[structopt(name = "gles2-generated-src-file", long, parse(from_os_str))]
    pub gles2_generated_src_file: Option<PathBuf>,
    #[structopt(name = "gles3-generated-src-file", long, parse(from_os_str))]
    pub gles3_generated_src_file: Option<PathBuf>,
    #[structopt(name = "cooked-shader-file", long, parse(from_os_str))]
    pub cooked_shader_file: Option<PathBuf>,

    //
    // For batch processing a folder
    //
    #[structopt(name = "glsl-path", long, parse(from_os_str))]
    pub glsl_files: Option<PathBuf>,
    #[structopt(name = "spv-path", long, parse(from_os_str))]
    pub spv_path: Option<PathBuf>,
    #[structopt(name = "rs-lib-path", long, parse(from_os_str))]
    pub rs_lib_path: Option<PathBuf>,
    #[structopt(name = "rs-mod-path", long, parse(from_os_str))]
    pub rs_mod_path: Option<PathBuf>,
    #[structopt(name = "dx12-generated-src-path", long, parse(from_os_str))]
    pub dx12_generated_src_path: Option<PathBuf>,
    #[structopt(name = "metal-generated-src-path", long, parse(from_os_str))]
    pub metal_generated_src_path: Option<PathBuf>,
    #[structopt(name = "gles2-generated-src-path", long, parse(from_os_str))]
    pub gles2_generated_src_path: Option<PathBuf>,
    #[structopt(name = "gles3-generated-src-path", long, parse(from_os_str))]
    pub gles3_generated_src_path: Option<PathBuf>,
    #[structopt(name = "cooked-shaders-path", long, parse(from_os_str))]
    pub cooked_shaders_path: Option<PathBuf>,

    #[structopt(name = "shader-kind", long)]
    pub shader_kind: Option<String>,

    #[structopt(name = "trace", long)]
    pub trace: bool,

    #[structopt(name = "optimize-shaders", long)]
    pub optimize_shaders: bool,

    #[structopt(name = "package-vk", long)]
    pub package_vk: bool,
    #[structopt(name = "package-dx12", long)]
    pub package_dx12: bool,
    #[structopt(name = "package-metal", long)]
    pub package_metal: bool,
    #[structopt(name = "package-gles2", long)]
    pub package_gles2: bool,
    #[structopt(name = "package-gles3", long)]
    pub package_gles3: bool,
    #[structopt(name = "package-all", long)]
    pub package_all: bool,

    #[structopt(name = "for-rafx-framework-crate", long)]
    pub for_rafx_framework_crate: bool,
}

pub fn run(args: &ShaderProcessorArgs) -> Result<(), Box<dyn Error>> {
    log::trace!("Shader processor args: {:#?}", args);
    if args.rs_lib_path.is_some() && args.rs_mod_path.is_some() {
        Err("Both --rs-lib-path and --rs-mod-path were provided, using both at the same time is not supported.")?;
    }

    let rs_file_option = if let Some(path) = &args.rs_lib_path {
        Some(RsFileOption {
            path: path.clone(),
            file_type: RsFileType::Lib,
        })
    } else if let Some(path) = &args.rs_mod_path {
        Some(RsFileOption {
            path: path.clone(),
            file_type: RsFileType::Mod,
        })
    } else {
        None
    };

    if let Some(glsl_file) = &args.glsl_file {
        //
        // Handle a single file given via --glsl_file. In this mode, the output files are explicit
        //
        log::info!("Processing file {:?}", glsl_file);

        //
        // Try to determine what kind of shader this is from the file name
        //
        // let shader_kind = shader_kind_from_args(args)
        //     .or_else(|| deduce_default_shader_kind_from_path(glsl_file))
        //     .unwrap_or(shaderc::ShaderKind::InferFromSource);

        todo!();
        // //
        // // Process this shader and write to output files
        // //
        // process_glsl_shader(
        //     glsl_file,
        //     args.spv_file.as_ref(),
        //     &rs_file_option,
        //     args.dx12_generated_src_path.as_ref(),
        //     args.metal_generated_src_file.as_ref(),
        //     args.gles2_generated_src_file.as_ref(),
        //     args.gles3_generated_src_file.as_ref(),
        //     args.cooked_shader_file.as_ref(),
        //     shader_kind,
        //     &args,
        // )
        // .map_err(|x| format!("{}: {}", glsl_file.to_string_lossy(), x.to_string()))?;

        // Ok(())
    } else if let Some(glsl_files) = &args.glsl_files {
        log::trace!("glsl files {:?}", args.glsl_files);
        process_directory(glsl_files, &args, &rs_file_option)
    } else {
        Ok(())
    }
}

//
// Handle a batch of file patterns (such as *.frag) via --glsl_files. Infer output files
// based on other args given in the form of output directories
//
fn process_directory(
    glsl_files: &PathBuf,
    args: &ShaderProcessorArgs,
    rs_file_option: &Option<RsFileOption>,
) -> Result<(), Box<dyn Error>> {
    // This will accumulate rust module names so we can produce a lib.rs if needed
    let mut module_names = FnvHashMap::<PathBuf, FnvHashSet<String>>::default();

    log::trace!("GLSL Root Dir: {:?}", glsl_files);

    let glob_walker = globwalk::GlobWalkerBuilder::from_patterns(
        glsl_files.to_str().unwrap(),
        &["*.{vert,frag,comp}"],
    )
    .file_type(globwalk::FileType::FILE)
    .build()?;

    let mut pipelines = HashMap::new();

    for glob in glob_walker {
        //
        // Determine the files we will write out
        //
        let glsl_file = glob?;

        if let Some(file_name_without_ext) = glsl_file.path().file_stem() {
            let path_without_ext = if let Some(parent) = glsl_file.path().parent() {
                parent.join(file_name_without_ext)
            } else {
                glsl_file.path().to_path_buf()
            };
            let pipeline_files = pipelines.entry(path_without_ext).or_insert(vec![]);
            pipeline_files.push(glsl_file.path().to_path_buf());
        }
    }
    for (stem, files) in pipelines {
        let empty_path = PathBuf::new();
        let outfile_prefix = stem
            .strip_prefix(glsl_files)?
            .parent()
            .unwrap_or(&empty_path);
        let Some(file_name) = stem.file_name() else {
            continue;
        };
        let file_name = file_name.to_string_lossy();

        let rs_module_name = file_name.to_string().to_lowercase().replace(".", "_");
        let rs_name = format!("{}.rs", rs_module_name);
        let rs_file = rs_file_option.as_ref().map(|x| RsFileOption {
            path: x.path.join(outfile_prefix).join(rs_name),
            file_type: x.file_type,
        });

        let package_vk = args.package_all || args.package_vk;
        let package_dx12 = args.package_all || args.package_dx12;
        let package_metal = args.package_all || args.package_metal;
        let package_gles2 = args.package_all || args.package_gles2;
        let package_gles3 = args.package_all || args.package_gles3;

        log::trace!(
            "package VK: {} dx12: {} Metal: {} GLES2: {} GLES3: {}",
            package_vk,
            package_dx12,
            package_metal,
            package_gles2,
            package_gles3
        );
        // ---- Pre-scan for @[vertex_formats] annotation ----
        let mut source_codes = Vec::new();
        let mut cross_compile_params = Vec::new();
        for glsl_file in &files {
            let code = std::fs::read_to_string(glsl_file)?;
            source_codes.push((glsl_file.clone(), code));
        }
        let vertex_formats = pre_scan_vertex_formats(
            &source_codes.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>(),
        );
        let has_vertex_formats = !vertex_formats.is_empty();

        // Build extra defines for the primary format (first in the list, if any)
        let primary_format_define;
        let primary_extra: Vec<(&str, &str)> = if has_vertex_formats {
            primary_format_define = format!("VERTEX_FORMAT_{}", vertex_formats[0]);
            vec![(&primary_format_define, "1")]
        } else {
            Vec::new()
        };

        // ---- Initial codegen compile ----
        let mut compile_results = Vec::new();
        for (glsl_file, code) in &source_codes {
            let shader_kind = shader_kind_from_args(args)
                .or_else(|| deduce_default_shader_kind_from_path(glsl_file))
                .unwrap_or(shaderc::ShaderKind::InferFromSource);

            if !(package_vk || package_dx12 || package_metal || package_gles2 || package_gles3) {
                Err("A cooked shader file or path was specified but no shader types are specified to package. Pass --package-vk, --package-dx12, --package-metal, --package-gles2, --package-gles3, or --package-all")?;
            }

            let compiler = shaderc::Compiler::new().unwrap();
            let compile_parameters = CompileParameters {
                glsl_file: glsl_file.clone(),
                shader_kind,
                code: code.clone(),
                entry_point_name: "main".to_string(),
                compiler,
            };

            log::info!("Compiling file {:?}", glsl_file);
            let mut defines: Vec<(&str, &str)> = vec![(PREPROCESSOR_DEF_PLATFORM_RUST_CODEGEN, "1")];
            defines.extend_from_slice(&primary_extra);
            let compile_result = compile_glsl(&compile_parameters, &defines)
                .map_err(|x| format!("{}: {}", glsl_file.to_string_lossy(), x.to_string()))?;

            compile_results.push(compile_result);
            cross_compile_params.push((glsl_file.clone(), compile_parameters));
        }
        let cross_compile_params_ref = cross_compile_params
            .iter()
            .map(|(p, cp)| (p.as_path(), cp))
            .collect::<Vec<_>>();
        let builtin_types = shader_types::create_builtin_type_lookup();
        log::info!("{:?}: reflect data", stem);
        let reflection_data = reflect::reflect_data(&builtin_types, &compile_results, true)
            .map_err(|x| format!("reflect_data: {}", x.to_string()))?;
        let glsl_file = stem.join(&*file_name);
        let (rust_code, vertex_channels) = if rs_file.is_some() {
            log::info!("{:?}: generate rust code", stem);
            let (code, channels) = codegen::generate_rust_code(
                stem.to_string_lossy().into(),
                &compile_results,
                &reflection_data,
                args.for_rafx_framework_crate,
            )?;
            (Some(code), channels)
        } else {
            let channels = codegen::compute_vertex_channels_from_results(&compile_results);
            (None, channels)
        };

        if let Some(rs_file) = &rs_file {
            write_output_file(&rs_file.path, rust_code.unwrap())?;
        }

        // ---- Cross-compile primary format ----
        log::info!("{:?}: cross compile primary", stem);
        let (vk_output, dx12_output, metal_output, gles2_output, gles3_output) =
            cross_compile_variant(&cross_compile_params_ref, args, &primary_extra, package_dx12, package_metal, package_gles2, package_gles3)?;

        if let Some(spv_file) = &args.spv_file {
            for shader in &vk_output.shader_results {
                let spv_file = spv_file
                    .join(outfile_prefix)
                    .join(shader.glsl_file.file_name().unwrap())
                    .with_extension("spv");
                write_output_file(&spv_file, &shader.new_src)?;
            }
        }

        // ---- Cook shader package ----
        if let Some(cooked_shader_path) = &args.cooked_shaders_path {
            let cooked_shader_path = cooked_shader_path
                .join(outfile_prefix)
                .join(glsl_file.file_name().unwrap())
                .with_extension("cookedshaderpackage");

            let glsl_paths: Vec<&Path> = cross_compile_params_ref.iter().map(|(p, _)| *p).collect();
            let packages = package_shader_stages(
                &compile_results, &glsl_paths, args,
                &vk_output, dx12_output.as_ref(), metal_output.as_ref(),
                gles2_output.as_ref(), gles3_output.as_ref(),
            )?;

            // ---- Additional vertex format variants ----
            let mut pipeline_variants = Vec::new();
            for format_name in vertex_formats.iter().skip(1) {
                log::info!("{:?}: cross compile variant {}", stem, format_name);
                let variant_define = format!("VERTEX_FORMAT_{}", format_name);
                let extra: Vec<(&str, &str)> = vec![(&variant_define, "1")];

                // Codegen compile just to get vertex channels for this variant
                let mut variant_compile_results = Vec::new();
                for (_, parameters) in &cross_compile_params {
                    let mut defines: Vec<(&str, &str)> = vec![(PREPROCESSOR_DEF_PLATFORM_RUST_CODEGEN, "1")];
                    defines.extend_from_slice(&extra);
                    let cr = compile_glsl(parameters, &defines)?;
                    variant_compile_results.push(cr);
                }
                let variant_channels = codegen::compute_vertex_channels_from_results(&variant_compile_results);

                // Cross-compile this variant
                let (v_vk, v_dx12, v_metal, v_gles2, v_gles3) =
                    cross_compile_variant(&cross_compile_params_ref, args, &extra, package_dx12, package_metal, package_gles2, package_gles3)?;

                let variant_packages = package_shader_stages(
                    &variant_compile_results, &glsl_paths, args,
                    &v_vk, v_dx12.as_ref(), v_metal.as_ref(),
                    v_gles2.as_ref(), v_gles3.as_ref(),
                )?;

                pipeline_variants.push(RafxPipelineVariant {
                    shaders: variant_packages,
                    vertex_channels: variant_channels.unwrap_or(0),
                });
            }

            let mut packaged_pipeline =
                RafxPipelinePackage::new(packages).with_vertex_channels(vertex_channels);
            packaged_pipeline.variants = pipeline_variants;
            let serialized = bincode::serialize(&packaged_pipeline)
                .map_err(|x| format!("Failed to serialize cooked pipeline: {}", x))?;
            write_output_file(&cooked_shader_path, serialized)?;
        }
        //
        if rs_file_option.is_some() {
            let module_names = module_names
                .entry(outfile_prefix.to_path_buf())
                .or_default();
            module_names.insert(rs_module_name.clone());
        }
    }
    //
    // Generate lib.rs or mod.rs files that includes all the compiled shaders
    //
    if let Some(rs_path) = &rs_file_option {
        // First ensure that for any nested submodules, they are declared in lib.rs/mod.rs files in
        // the parent dirs
        let outfile_prefixes: Vec<_> = module_names.keys().cloned().collect();
        for mut outfile_prefix in outfile_prefixes {
            while let Some(parent) = outfile_prefix.parent() {
                let new_module_name = outfile_prefix
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                log::trace!("add module {:?} to {:?}", new_module_name, parent);

                let module_names = module_names.entry(parent.to_path_buf()).or_default();
                module_names.insert(new_module_name);

                outfile_prefix = parent.to_path_buf();
            }
        }

        // Generate all lib.rs/mod.rs files
        for (outfile_prefix, module_names) in module_names {
            let module_filename = match rs_path.file_type {
                RsFileType::Lib => "lib.rs",
                RsFileType::Mod => "mod.rs",
            };
            let lib_file_path = rs_path.path.join(outfile_prefix).join(module_filename);
            log::trace!("Write lib/mod file {:?} {:?}", lib_file_path, module_names);

            let mut lib_file_string = String::default();
            lib_file_string += "// This code is auto-generated by the shader processor.\n\n";
            lib_file_string += "#![allow(dead_code)]\n\n";

            for module_name in module_names {
                lib_file_string += &format!("pub mod {};\n", module_name);
            }

            write_output_file(&lib_file_path, lib_file_string)?;
        }
    }

    Ok(())
}

struct CompileParameters {
    glsl_file: PathBuf,
    shader_kind: ShaderKind,
    code: String,
    entry_point_name: String,
    compiler: Compiler,
}

/// Pre-scan GLSL source text for `@[vertex_formats(["PC", "PU"])]` annotation.
/// Returns the format list, or empty vec if not found.
fn pre_scan_vertex_formats(sources: &[String]) -> Vec<String> {
    for source in sources {
        let needle = "@[vertex_formats(";
        if let Some(start) = source.find(needle) {
            // Extract from the opening ( to the matching )]
            let data_start = start + "@[vertex_formats".len();
            if let Some(end) = source[data_start..].find(")]") {
                let ron_data = &source[data_start..data_start + end + 1]; // include closing )
                if let Ok(ann) = ron::de::from_str::<crate::parse_declarations::VertexFormatsAnnotation>(ron_data) {
                    return ann.0;
                }
            }
        }
    }
    Vec::new()
}

/// Package shader stages from cross-compile outputs into `Vec<RafxHashedShaderPackage>`.
fn package_shader_stages(
    compile_results: &[CompileResult],
    glsl_files: &[&Path],
    args: &ShaderProcessorArgs,
    vk_output: &PipelineCrossCompileResult,
    dx12_output: Option<&PipelineCrossCompileResult>,
    metal_output: Option<&PipelineCrossCompileResult>,
    gles2_output: Option<&PipelineCrossCompileResult>,
    gles3_output: Option<&PipelineCrossCompileResult>,
) -> Result<Vec<RafxHashedShaderPackage>, Box<dyn Error>> {
    let mut packages = Vec::new();
    for (compile, glsl_file) in compile_results.iter().zip(glsl_files) {
        let entry_point = compile.artifact.entry_points()?.next().unwrap();
        let shader_stage = map_shader_stage_flags(entry_point.execution_model)?;
        let mut shader_package = RafxShaderPackage::default();

        if args.package_all || args.package_vk {
            let vk_shader = vk_output.shader_results.iter()
                .find(|s| &s.glsl_file == glsl_file).unwrap();
            shader_package.vk = Some(RafxShaderPackageVulkan::SpvBytes(vk_shader.new_src.clone()));
            shader_package.vk_reflection = vk_output.reflection_data.reflection.iter()
                .find(|r| r.rafx_api_reflection.shader_stage == shader_stage).cloned();
        }

        if let Some(dx12_output) = dx12_output {
            let dx12_shader = dx12_output.shader_results.iter()
                .find(|s| &s.glsl_file == glsl_file).unwrap();
            shader_package.dx12 = Some(RafxShaderPackageDx12::Src(
                std::str::from_utf8(&dx12_shader.new_src).unwrap().into(),
            ));
            shader_package.dx12_reflection = dx12_output.reflection_data.reflection.iter()
                .find(|r| r.rafx_api_reflection.shader_stage == shader_stage).cloned();
        }

        if let Some(metal_output) = metal_output {
            let metal_shader = metal_output.shader_results.iter()
                .find(|s| &s.glsl_file == glsl_file).unwrap();
            shader_package.metal = Some(RafxShaderPackageMetal::Src(
                std::str::from_utf8(&metal_shader.new_src).unwrap().into(),
            ));
            shader_package.metal_reflection = metal_output.reflection_data.reflection.iter()
                .find(|r| r.rafx_api_reflection.shader_stage == shader_stage).cloned();
        }

        if let Some(gles2) = gles2_output {
            let gles2_shader = gles2.shader_results.iter()
                .find(|s| &s.glsl_file == glsl_file).unwrap();
            shader_package.gles2 = Some(RafxShaderPackageGles2::Src(
                std::str::from_utf8(&gles2_shader.new_src).unwrap().into(),
            ));
            shader_package.gles2_reflection = gles2.reflection_data.reflection.iter()
                .find(|r| r.rafx_api_reflection.shader_stage == shader_stage).cloned();
        }

        if let Some(gles3) = gles3_output {
            let gles3_shader = gles3.shader_results.iter()
                .find(|s| &s.glsl_file == glsl_file).unwrap();
            shader_package.gles3 = Some(RafxShaderPackageGles3::Src(
                std::str::from_utf8(&gles3_shader.new_src).unwrap().into(),
            ));
            shader_package.gles3_reflection = gles3.reflection_data.reflection.iter()
                .find(|r| r.rafx_api_reflection.shader_stage == shader_stage).cloned();
        }

        shader_package.debug_name = Some(glsl_file.file_name().unwrap().to_string_lossy().to_string());
        log::info!("packaging {:?} as {:?}", glsl_file, shader_package.vk_reflection);
        packages.push(RafxHashedShaderPackage::new(shader_package));
    }
    Ok(packages)
}

/// Cross-compile a full set of shader stages for one vertex-format variant and
/// return (packages, vertex_channels).
fn cross_compile_variant(
    cross_compile_params: &[(&Path, &CompileParameters)],
    args: &ShaderProcessorArgs,
    extra_defines: &[(&str, &str)],
    package_dx12: bool,
    package_metal: bool,
    package_gles2: bool,
    package_gles3: bool,
) -> Result<(
    PipelineCrossCompileResult,  // vk
    Option<PipelineCrossCompileResult>,  // dx12
    Option<PipelineCrossCompileResult>,  // metal
    Option<PipelineCrossCompileResult>,  // gles2
    Option<PipelineCrossCompileResult>,  // gles3
), Box<dyn Error>> {
    let vk_output = cross_compile_to_vulkan(cross_compile_params, args, extra_defines)?;
    let dx12_output = if package_dx12 {
        Some(cross_compile_to_dx12(cross_compile_params, extra_defines)?)
    } else { None };
    let metal_output = if package_metal {
        Some(cross_compile_to_metal(cross_compile_params, extra_defines)?)
    } else { None };
    let gles2_output = if package_gles2 {
        Some(cross_compile_to_gles2(cross_compile_params, extra_defines)?)
    } else { None };
    let gles3_output = if package_gles3 {
        Some(cross_compile_to_gles3(cross_compile_params, extra_defines)?)
    } else { None };
    Ok((vk_output, dx12_output, metal_output, gles2_output, gles3_output))
}

struct ShaderCrossCompileResult {
    glsl_file: PathBuf,
    new_src: Vec<u8>,
}
struct PipelineCrossCompileResult {
    shader_results: Vec<ShaderCrossCompileResult>,
    reflection_data: ShaderProcessorReflectionData,
}
struct CompileResult {
    unoptimized_spv: CompilationArtifact,
    parsed_declarations: ParseDeclarationsResult,
    artifact: CompiledArtifact<spirv_cross2::targets::Glsl>,
    shader_kind: shaderc::ShaderKind,
}

fn try_load_override_src(
    original_path: &Path,
    extension: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let mut override_path = original_path.as_os_str().to_os_string();
    override_path.push(extension);
    let override_path = PathBuf::from(override_path);
    if override_path.exists() {
        log::info!(
            "  Override shader {:?} with {:?}",
            original_path,
            override_path.to_string_lossy()
        );

        let override_src = std::fs::read_to_string(&override_path)?;

        // We want to inline all the #includes because we are packaging the source for compilation
        // on target hardware and it won't be able to #include dependencies.
        let preprocessed_src =
            parse_source::inline_includes_in_override_src(&override_path, &override_src)?;

        Ok(Some(preprocessed_src))
    } else {
        Ok(None)
    }
}

fn compile_glsl(
    parameters: &CompileParameters,
    defines: &[(&str, &str)],
) -> Result<CompileResult, Box<dyn Error>> {
    log::trace!("{:?}: compile unoptimized", parameters.glsl_file);
    let (unoptimized_spv, parsed_source) = {
        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        compile_options.set_include_callback(include::shaderc_include_callback);
        compile_options.set_generate_debug_info();
        for (name, value) in defines {
            compile_options.add_macro_definition(name, Some(value));
        }

        log::trace!("compile to spriv for defines {:?}", defines);

        let unoptimized_spv = parameters.compiler.compile_into_spirv(
            &parameters.code,
            parameters.shader_kind,
            parameters.glsl_file.to_str().unwrap(),
            &parameters.entry_point_name,
            Some(&compile_options),
        )?;

        log::trace!("{:?}: parse glsl", parameters.glsl_file);

        let mut preprocessor_state = PreprocessorState::default();
        for (name, value) in defines {
            preprocessor_state.add_define(name.to_string(), value.to_string());
        }
        let parsed_source = parse_source::parse_glsl_src(
            &parameters.glsl_file,
            &parameters.code,
            &mut preprocessor_state,
        )?;

        (unoptimized_spv, parsed_source)
    };

    //
    // Read the unoptimized spv into spirv_cross so that we can grab reflection data
    //
    log::trace!("{:?}: read spirv_cross module", parameters.glsl_file);
    let spirv_cross_module = spirv_cross2::Module::from_words(unoptimized_spv.as_binary());

    //
    // Parse the declarations that were extracted from the source file
    //
    log::trace!("{:?}: parse declarations", parameters.glsl_file);
    let parsed_declarations = parse_declarations::parse_declarations(&parsed_source.declarations)?;
    let is_compute_shader = normalize_shader_kind(parameters.shader_kind) == ShaderKind::Compute;
    if parsed_declarations.group_size.is_some() && !is_compute_shader {
        Err("The shader is not a compute shader but a group size was specified")?;
    } else if parsed_declarations.group_size.is_none() && is_compute_shader {
        Err("The shader is a compute shader but a group size was not specified. Expected to find something like `layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;` in the shader")?;
    }

    log::trace!("{:?}: generate spirv_cross ast", parameters.glsl_file);
    let mut spirv_cross_glsl_options = spirv_cross2::compile::glsl::CompilerOptions::default();
    spirv_cross_glsl_options.vulkan_semantics = true;
    let compiler = spirv_cross2::Compiler::new(spirv_cross_module)?;
    let artifact = compiler.compile(&spirv_cross_glsl_options)?;

    log::trace!("{:?}: generate shader types", parameters.glsl_file);
    Ok(CompileResult {
        unoptimized_spv,
        parsed_declarations,
        artifact,
        shader_kind: parameters.shader_kind,
    })
}

fn cross_compile_to_vulkan(
    glsl_files: &[(&Path, &CompileParameters)],
    args: &ShaderProcessorArgs,
    extra_defines: &[(&str, &str)],
) -> Result<PipelineCrossCompileResult, Box<dyn Error>> {
    let mut compile_results = Vec::new();
    for (glsl_file, parameters) in glsl_files {
        log::trace!("{:?}: create vulkan", glsl_file);
        let mut defines = vec![(PREPROCESSOR_DEF_PLATFORM_VULKAN, "1")];
        defines.extend_from_slice(extra_defines);
        compile_results.push((
            glsl_file,
            compile_glsl(parameters, &defines)?,
            parameters,
        ));
    }
    let builtin_types = shader_types::create_builtin_type_lookup();
    let reflection_data = reflect::reflect_data(
        &builtin_types,
        compile_results
            .iter()
            .map(|(_, c, _)| c)
            .collect::<Vec<_>>(),
        true,
    )?;

    let mut shaders = Vec::new();
    for (glsl_file, compile_result, parameters) in compile_results {
        let vk_spv = if args.optimize_shaders {
            let mut compile_options = shaderc::CompileOptions::new().unwrap();
            compile_options.set_include_callback(include::shaderc_include_callback);
            compile_options.set_optimization_level(shaderc::OptimizationLevel::Performance);
            //NOTE: Could also use shaderc::OptimizationLevel::Size

            parameters
                .compiler
                .compile_into_spirv(
                    &parameters.code,
                    parameters.shader_kind,
                    glsl_file.to_str().unwrap(),
                    &parameters.entry_point_name,
                    Some(&compile_options),
                )?
                .as_binary_u8()
                .to_vec()
        } else {
            compile_result.unoptimized_spv.as_binary_u8().to_vec()
        };
        shaders.push(ShaderCrossCompileResult {
            glsl_file: glsl_file.to_path_buf(),
            new_src: vk_spv,
        });
    }

    Ok(PipelineCrossCompileResult {
        shader_results: shaders,
        reflection_data,
    })
}

fn cross_compile_to_dx12(
    glsl_files: &[(&Path, &CompileParameters)],
    extra_defines: &[(&str, &str)],
) -> Result<PipelineCrossCompileResult, Box<dyn Error>> {
    let mut compile_results = Vec::new();
    for (glsl_file, parameters) in glsl_files {
        log::trace!("{:?}: create dx12", glsl_file);
        let mut defines = vec![(PREPROCESSOR_DEF_PLATFORM_DX12, "1")];
        defines.extend_from_slice(extra_defines);
        compile_results.push((
            glsl_file,
            compile_glsl(parameters, &defines)?,
        ));
    }
    let builtin_types = shader_types::create_builtin_type_lookup();
    let reflection_data = reflect::reflect_data(
        &builtin_types,
        compile_results.iter().map(|(_, c)| c).collect::<Vec<_>>(),
        true,
    )?;

    let mut output = Vec::new();
    for (glsl_file, compile_result) in compile_results {
        let dx12_src = if let Some(src) = try_load_override_src(glsl_file, ".hlsl")? {
            src
        } else {
            let spirv_cross_module =
                spirv_cross2::Module::from_words(compile_result.unoptimized_spv.as_binary());

            let mut spirv_cross_hlsl_options =
                spirv_cross2::compile::hlsl::CompilerOptions::default();
            spirv_cross_hlsl_options.shader_model =
                spirv_cross2::compile::hlsl::HlslShaderModel::ShaderModel6_0;
            spirv_cross_hlsl_options.flatten_matrix_vertex_input_semantics = true;
            //DX12TODO: We want something more fine-grained than this
            // spirv_cross_hlsl_options.force_storage_buffer_as_uav = true;
            let mut compiler: spirv_cross2::Compiler<Hlsl> =
                spirv_cross2::Compiler::new(spirv_cross_module)?;

            for assignment in &reflection_data.hlsl_register_assignments {
                compiler.add_resource_binding(
                    assignment.execution_model,
                    assignment.binding,
                    &assignment.bind_target,
                )?;
            }

            for remap in &reflection_data.hlsl_vertex_attribute_remaps {
                // We require semantics to produce HLSL, an error should be thrown earlier if they are missing
                assert!(!remap.semantic.is_empty());
                if !remap.semantic.is_empty() {
                    compiler.remap_vertex_attribute(remap.location, remap.semantic.as_str())?;
                }
            }

            compiler.compile(&spirv_cross_hlsl_options)?.to_string()
        };
        output.push(ShaderCrossCompileResult {
            glsl_file: glsl_file.to_path_buf(),
            new_src: dx12_src.as_bytes().to_vec(),
        })
    }

    Ok(PipelineCrossCompileResult {
        shader_results: output,
        reflection_data,
    })
}

fn cross_compile_to_metal(
    glsl_files: &[(&Path, &CompileParameters)],
    extra_defines: &[(&str, &str)],
) -> Result<PipelineCrossCompileResult, Box<dyn Error>> {
    let mut compile_results = Vec::new();
    for (glsl_file, parameters) in glsl_files {
        log::trace!("{:?}: create msl", glsl_file);
        let mut defines = vec![(PREPROCESSOR_DEF_PLATFORM_METAL, "1")];
        defines.extend_from_slice(extra_defines);
        compile_results.push((
            glsl_file,
            compile_glsl(parameters, &defines)?,
        ));
    }
    let builtin_types = shader_types::create_builtin_type_lookup();
    let reflection_data = reflect::reflect_data(
        &builtin_types,
        compile_results.iter().map(|(_, c)| c).collect::<Vec<_>>(),
        true,
    )?;

    let mut shaders = Vec::new();
    for (glsl_file, compile_result) in compile_results {
        let metal_src = if let Some(src) = try_load_override_src(glsl_file, ".metal")? {
            src
        } else {
            let spirv_cross_module =
                spirv_cross2::Module::from_words(compile_result.unoptimized_spv.as_binary());

            let mut compiler: spirv_cross2::Compiler<Msl> =
                spirv_cross2::Compiler::new(spirv_cross_module)?;
            let mut spirv_cross_msl_options =
                spirv_cross2::compile::msl::CompilerOptions::default();
            spirv_cross_msl_options.version = spirv_cross2::compile::msl::MslVersion::new(2, 1, 0);
            spirv_cross_msl_options.argument_buffers = true;
            spirv_cross_msl_options.force_active_argument_buffer_resources = true;
            //TODO: Add equivalent to --msl-no-clip-distance-user-varying

            //TODO: Set this up
            // spirv_cross_msl_options.resource_binding_overrides = compile_result
            //     .reflection_data
            //     .as_ref()
            //     .unwrap()
            //     .msl_argument_buffer_assignments
            //     .clone();
            for assignment in &reflection_data.msl_argument_buffer_assignments {
                compiler
                    .add_resource_binding(
                        assignment.execution_model,
                        assignment.binding,
                        &assignment.bind_target,
                    )
                    .unwrap();
            }
            //println!(" binding overrides {:?}", spirv_cross_msl_options.resource_binding_overrides);
            //spirv_cross_msl_options.vertex_attribute_overrides
            for (binding, const_sampler) in &reflection_data.msl_const_samplers {
                if let spirv_cross2::compile::msl::ResourceBinding::Qualified { set, binding } =
                    binding
                {
                    compiler.remap_constexpr_sampler_by_binding(
                        *set,
                        *binding,
                        &const_sampler.sampler,
                        const_sampler.conversion.as_ref(),
                    )?;
                }
            }

            compiler.compile(&spirv_cross_msl_options)?.to_string()
        };
        shaders.push(ShaderCrossCompileResult {
            glsl_file: glsl_file.to_path_buf(),
            new_src: metal_src.as_bytes().to_vec(),
        })
    }

    Ok(PipelineCrossCompileResult {
        shader_results: shaders,
        reflection_data,
    })
}

fn cross_compile_to_gles3(
    glsl_files: &[(&Path, &CompileParameters)],
    extra_defines: &[(&str, &str)],
) -> Result<PipelineCrossCompileResult, Box<dyn Error>> {
    let mut compile_results = Vec::new();
    for (glsl_file, parameters) in glsl_files {
        log::trace!("{:?}: create gles3", glsl_file);
        let mut defines = vec![(PREPROCESSOR_DEF_PLATFORM_GLES3, "1")];
        defines.extend_from_slice(extra_defines);
        compile_results.push((
            glsl_file,
            compile_glsl(parameters, &defines)?,
        ));
    }

    let builtin_types = shader_types::create_builtin_type_lookup();
    let mut reflection_data = reflect::reflect_data(
        &builtin_types,
        compile_results.iter().map(|(_, c)| c).collect::<Vec<_>>(),
        true,
    )?;

    let mut shaders = Vec::new();
    for (glsl_file, compile_result) in compile_results {
        let gles3_src = if let Some(src) = try_load_override_src(glsl_file, ".gles3")? {
            src
        } else {
            let spirv_cross_module =
                spirv_cross2::Module::from_words(compile_result.unoptimized_spv.as_binary());

            let mut spirv_cross_gles3_options =
                spirv_cross2::compile::glsl::CompilerOptions::default();
            spirv_cross_gles3_options.version = spirv_cross2::compile::glsl::GlslVersion::Glsl300Es;
            spirv_cross_gles3_options.vulkan_semantics = false;
            spirv_cross_gles3_options.common.fixup_clipspace = true;
            spirv_cross_gles3_options.common.flip_vertex_y = true;

            let shader_resources = compile_result.artifact.shader_resources()?;
            let mut compiler = spirv_cross2::Compiler::new(spirv_cross_module)?;

            rename_gl_samplers(&mut reflection_data, &mut compiler)?;
            rename_gl_in_out_attributes(
                compile_result.shader_kind,
                &mut compiler,
                &shader_resources,
            )?;

            compiler.compile(&spirv_cross_gles3_options)?.to_string()
        };
        shaders.push(ShaderCrossCompileResult {
            glsl_file: glsl_file.to_path_buf(),
            new_src: gles3_src.as_bytes().to_vec(),
        })
    }

    Ok(PipelineCrossCompileResult {
        shader_results: shaders,
        reflection_data,
    })
}

fn cross_compile_to_gles2(
    glsl_files: &[(&Path, &CompileParameters)],
    extra_defines: &[(&str, &str)],
) -> Result<PipelineCrossCompileResult, Box<dyn Error>> {
    let mut compile_results = Vec::new();
    for (glsl_file, parameters) in glsl_files {
        log::trace!("{:?}: create gles2", glsl_file);
        let mut defines = vec![(PREPROCESSOR_DEF_PLATFORM_GLES2, "1")];
        defines.extend_from_slice(extra_defines);
        compile_results.push((
            glsl_file,
            compile_glsl(parameters, &defines)?,
        ));
    }
    let builtin_types = shader_types::create_builtin_type_lookup();
    let mut reflection_data = reflect::reflect_data(
        &builtin_types,
        compile_results.iter().map(|(_, c)| c).collect::<Vec<_>>(),
        true,
    )?;

    let mut shaders = Vec::new();
    for (glsl_file, compile_result) in compile_results {
        let gles2_src = if let Some(src) = try_load_override_src(glsl_file, ".gles2")? {
            src
        } else {
            let spirv_cross_module =
                spirv_cross2::Module::from_words(compile_result.unoptimized_spv.as_binary());

            let mut spirv_cross_gles2_options =
                spirv_cross2::compile::glsl::CompilerOptions::default();
            spirv_cross_gles2_options.version = spirv_cross2::compile::glsl::GlslVersion::Glsl100Es;
            spirv_cross_gles2_options.vulkan_semantics = false;
            spirv_cross_gles2_options.common.fixup_clipspace = true;
            spirv_cross_gles2_options.common.flip_vertex_y = true;
            let mut compiler = spirv_cross2::Compiler::new(spirv_cross_module)?;

            let shader_resources = compile_result.artifact.shader_resources()?;
            let uniform_buffers = shader_resources
                .resources_for_type(spirv_cross2::reflect::ResourceType::UniformBuffer)?;

            // Rename uniform blocks to be consistent with how they would appear in GL ES 3.0. This way
            // we can consistently use the same GL name across both backends
            for resource in uniform_buffers {
                let block_name_orig = compiler.name(resource.base_type_id)?;
                if let Some(block_name_orig) = block_name_orig {
                    let block_name = block_name_orig.to_string();
                    drop(block_name_orig);
                    compiler.set_name(
                        resource.base_type_id,
                        format!("{}_UniformBlock", block_name),
                    )?;
                    compiler.set_name(resource.id, block_name)?;
                }
            }

            rename_gl_samplers(&mut reflection_data, &mut compiler)?;
            rename_gl_in_out_attributes(
                compile_result.shader_kind,
                &mut compiler,
                &shader_resources,
            )?;

            compiler.compile(&spirv_cross_gles2_options)?.to_string()
        };
        shaders.push(ShaderCrossCompileResult {
            glsl_file: glsl_file.to_path_buf(),
            new_src: gles2_src.as_bytes().to_vec(),
        })
    }

    Ok(PipelineCrossCompileResult {
        reflection_data,
        shader_results: shaders,
    })
}

fn write_output_file<C: AsRef<[u8]>>(
    path: &PathBuf,
    contents: C,
) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, contents)
}

fn rename_gl_samplers(
    reflected_data: &mut ShaderProcessorReflectionData,
    compiler: &mut spirv_cross2::Compiler<spirv_cross2::targets::Glsl>,
) -> Result<(), Box<dyn Error>> {
    let dummy_sampler = compiler.create_dummy_sampler_for_combined_images()?;
    compiler.build_combined_image_samplers(dummy_sampler)?;

    let mut all_combined_textures = FnvHashSet::default();
    for remap in compiler.combined_image_samplers()? {
        let texture_name = compiler.name(remap.image_id)?;
        let sampler_name = compiler.name(remap.sampler_id)?;

        let already_sampled = !all_combined_textures.insert(remap.image_id.id());
        if already_sampled {
            Err(format!("The texture {:?} is being read by multiple samplers. This is not supported in GL ES 2.0", texture_name))?;
        }

        if let Some(texture_name) = texture_name {
            let texture_name = texture_name.to_string();
            if let Some(sampler_name) = sampler_name {
                reflected_data.set_gl_sampler_name(&texture_name, &sampler_name);
            }

            compiler.set_name(remap.combined_id, texture_name)?
        }
    }

    Ok(())
}

fn rename_gl_in_out_attributes(
    shader_kind: ShaderKind,
    compiler: &mut spirv_cross2::Compiler<spirv_cross2::targets::Glsl>,
    shader_resources: &ShaderResources,
) -> Result<(), Box<dyn Error>> {
    if normalize_shader_kind(shader_kind) == ShaderKind::Vertex {
        let stage_outputs = shader_resources
            .resources_for_type(spirv_cross2::reflect::ResourceType::StageOutput)?;
        for resource in stage_outputs {
            let location =
                compiler.decoration(resource.id, spirv_cross2::spirv::Decoration::Location)?;
            if let Some(spirv_cross2::reflect::DecorationValue::Literal(_location)) = location {
                todo!();
                // compiler.rename_interface_variable(
                //     &[resource.clone()],
                //     location,
                //     &format!("interface_var_{}", location),
                // )?;
            }
        }
    } else if normalize_shader_kind(shader_kind) == ShaderKind::Fragment {
        let stage_outputs = shader_resources
            .resources_for_type(spirv_cross2::reflect::ResourceType::StageOutput)?;
        for resource in stage_outputs {
            let location =
                compiler.decoration(resource.id, spirv_cross2::spirv::Decoration::Location)?;
            if let Some(spirv_cross2::reflect::DecorationValue::Literal(_location)) = location {
                todo!();
                // compiler.rename_interface_variable(
                //     &[resource.clone()],
                //     location,
                //     &format!("interface_var_{}", location),
                // )?;
            }
        }
    }

    Ok(())
}

fn shader_kind_from_args(args: &ShaderProcessorArgs) -> Option<shaderc::ShaderKind> {
    let extensions = [
        ("vert", shaderc::ShaderKind::Vertex),
        ("frag", shaderc::ShaderKind::Fragment),
        ("tesc", shaderc::ShaderKind::TessControl),
        ("tese", shaderc::ShaderKind::TessEvaluation),
        ("geom", shaderc::ShaderKind::Geometry),
        ("comp", shaderc::ShaderKind::Compute),
        //("spvasm", shaderc::ShaderKind::Vertex), // we don't parse spvasm
        ("rgen", shaderc::ShaderKind::RayGeneration),
        ("rahit", shaderc::ShaderKind::AnyHit),
        ("rchit", shaderc::ShaderKind::ClosestHit),
        ("rmiss", shaderc::ShaderKind::Miss),
        ("rint", shaderc::ShaderKind::Intersection),
        ("rcall", shaderc::ShaderKind::Callable),
        ("task", shaderc::ShaderKind::Task),
        ("mesh", shaderc::ShaderKind::Mesh),
    ];

    if let Some(shader_kind) = &args.shader_kind {
        for &(extension, kind) in &extensions {
            if shader_kind == extension {
                return Some(kind);
            }
        }
    }

    None
}

// based on https://github.com/google/shaderc/blob/caa519ca532a6a3a0279509fce2ceb791c4f4651/glslc/src/shader_stage.cc#L69
fn deduce_default_shader_kind_from_path(path: &Path) -> Option<shaderc::ShaderKind> {
    let extensions = [
        ("vert", shaderc::ShaderKind::DefaultVertex),
        ("frag", shaderc::ShaderKind::DefaultFragment),
        ("tesc", shaderc::ShaderKind::DefaultTessControl),
        ("tese", shaderc::ShaderKind::DefaultTessEvaluation),
        ("geom", shaderc::ShaderKind::DefaultGeometry),
        ("comp", shaderc::ShaderKind::DefaultCompute),
        //("spvasm", shaderc::ShaderKind::Vertex), // we don't parse spvasm
        ("rgen", shaderc::ShaderKind::DefaultRayGeneration),
        ("rahit", shaderc::ShaderKind::DefaultAnyHit),
        ("rchit", shaderc::ShaderKind::DefaultClosestHit),
        ("rmiss", shaderc::ShaderKind::DefaultMiss),
        ("rint", shaderc::ShaderKind::DefaultIntersection),
        ("rcall", shaderc::ShaderKind::DefaultCallable),
        ("task", shaderc::ShaderKind::DefaultTask),
        ("mesh", shaderc::ShaderKind::DefaultMesh),
    ];

    if let Some(extension) = path.extension() {
        let as_str = extension.to_string_lossy();

        for &(extension, kind) in &extensions {
            if as_str.contains(extension) {
                return Some(kind);
            }
        }
    }

    None
}

fn normalize_shader_kind(shader_kind: ShaderKind) -> ShaderKind {
    match shader_kind {
        ShaderKind::Vertex | ShaderKind::DefaultVertex => ShaderKind::Vertex,
        ShaderKind::Fragment | ShaderKind::DefaultFragment => ShaderKind::Fragment,
        ShaderKind::Compute | ShaderKind::DefaultCompute => ShaderKind::Compute,
        ShaderKind::Geometry | ShaderKind::DefaultGeometry => ShaderKind::Geometry,
        ShaderKind::TessControl | ShaderKind::DefaultTessControl => ShaderKind::TessControl,
        ShaderKind::TessEvaluation | ShaderKind::DefaultTessEvaluation => {
            ShaderKind::TessEvaluation
        }
        ShaderKind::RayGeneration | ShaderKind::DefaultRayGeneration => ShaderKind::RayGeneration,
        ShaderKind::AnyHit | ShaderKind::DefaultAnyHit => ShaderKind::AnyHit,
        ShaderKind::ClosestHit | ShaderKind::DefaultClosestHit => ShaderKind::ClosestHit,
        ShaderKind::Miss | ShaderKind::DefaultMiss => ShaderKind::Miss,
        ShaderKind::Intersection | ShaderKind::DefaultIntersection => ShaderKind::Intersection,
        ShaderKind::Callable | ShaderKind::DefaultCallable => ShaderKind::Callable,
        ShaderKind::Task | ShaderKind::DefaultTask => ShaderKind::Task,
        ShaderKind::Mesh | ShaderKind::DefaultMesh => ShaderKind::Mesh,
        ShaderKind::InferFromSource => ShaderKind::InferFromSource,
        ShaderKind::SpirvAssembly => ShaderKind::SpirvAssembly,
    }
}
