use super::{
    assets::{
        AnimationAssetData, GltfMaterialAsset, GltfMaterialDataShaderParam, MeshAssetData,
        MeshPartAssetData, SkeletonAssetData,
    },
    AnimTrack, AnimTrackData, InterpolationMode, JointTrackCollection, SkeletonJoint,
};
use atelier_assets::loader::handle::SerdeContext;
use atelier_assets::{
    core::{AssetRef, AssetUuid},
    importer::ImportOp,
};
use atelier_assets::{
    importer::{Error, ImportedAsset, Importer, ImporterValue},
    make_handle,
};
use atelier_assets::{loader::handle::Handle, make_handle_from_str};
use fnv::FnvHashMap;
use gltf::{animation::util::ReadOutputs, image::Data as GltfImageData};
use gltf::{animation::Interpolation, buffer::Data as GltfBufferData};
use itertools::Itertools;
use rafx::{assets::assets::BufferAssetData, resources::{VertexData, VertexDataLayout, VertexDataSet, VertexMember, vk_description::{Format, size_of_vertex_format}}};
use rafx::assets::assets::{ImageAssetColorSpace, ImageAssetData};
use rafx::assets::assets::{MaterialInstanceAssetData, MaterialInstanceSlotAssignment};
use rafx::assets::push_buffer::PushBuffer;
use rafx::assets::BufferAsset;
use rafx::assets::ImageAsset;
use rafx::assets::MaterialAsset;
use rafx::assets::MaterialInstanceAsset;
use serde::export::Formatter;
use serde::{Deserialize, Serialize};
use std::{alloc::Layout, convert::TryInto};
use std::io::Read;
use std::str::FromStr;
use type_uuid::*;

#[derive(Debug)]
struct GltfImportError {
    error_message: String,
}

impl GltfImportError {
    pub fn new(error_message: &str) -> Self {
        GltfImportError {
            error_message: error_message.to_string(),
        }
    }
}

impl std::error::Error for GltfImportError {}

impl std::fmt::Display for GltfImportError {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.error_message)
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum GltfObjectId {
    Name(String),
    Index(usize),
}

struct ImageToImport {
    id: GltfObjectId,
    asset: ImageAssetData,
}

struct MaterialToImport {
    id: GltfObjectId,
    asset: GltfMaterialAsset,
}

struct MeshToImport {
    id: GltfObjectId,
    asset: MeshAssetData,
}

struct AnimationToImport {
    id: GltfObjectId,
    asset: AnimationAssetData,
}

struct SkeletonToImport {
    id: GltfObjectId,
    asset: SkeletonAssetData,
}

struct BufferToImport {
    id: GltfObjectId,
    asset: BufferAssetData,
}

// fn get_or_create_uuid(option_uuid: &mut Option<AssetUuid>) -> AssetUuid {
//     let uuid = option_uuid.unwrap_or_else(|| AssetUuid(*uuid::Uuid::new_v4().as_bytes()));
//
//     *option_uuid = Some(uuid);
//     uuid
// }

// The asset state is stored in this format using Vecs
#[derive(TypeUuid, Serialize, Deserialize, Default, Clone)]
#[uuid = "807c83b3-c24c-4123-9580-5f9c426260b4"]
#[serde(default)]
pub struct GltfImporterStateStable {
    // Asset UUIDs for imported image by name. We use vecs here so we can sort by UUID for
    // deterministic output
    buffer_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    image_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    material_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    material_instance_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    mesh_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    animation_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
    skeleton_asset_uuids: Vec<(GltfObjectId, AssetUuid)>,
}

impl From<GltfImporterStateUnstable> for GltfImporterStateStable {
    fn from(other: GltfImporterStateUnstable) -> Self {
        let mut stable = GltfImporterStateStable::default();
        stable.buffer_asset_uuids = other
            .buffer_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.image_asset_uuids = other
            .image_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.material_asset_uuids = other
            .material_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.material_instance_asset_uuids = other
            .material_instance_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.mesh_asset_uuids = other
            .mesh_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.animation_asset_uuids = other
            .animation_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable.skeleton_asset_uuids = other
            .skeleton_asset_uuids
            .into_iter()
            .sorted_by_key(|(id, _uuid)| id.clone())
            .collect();
        stable
    }
}

#[derive(Default)]
pub struct GltfImporterStateUnstable {
    //asset_uuid: Option<AssetUuid>,

    // Asset UUIDs for imported image by name
    buffer_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    image_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    material_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    material_instance_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    mesh_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    animation_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
    skeleton_asset_uuids: FnvHashMap<GltfObjectId, AssetUuid>,
}

impl From<GltfImporterStateStable> for GltfImporterStateUnstable {
    fn from(other: GltfImporterStateStable) -> Self {
        let mut unstable = GltfImporterStateUnstable::default();
        unstable.buffer_asset_uuids = other.buffer_asset_uuids.into_iter().collect();
        unstable.image_asset_uuids = other.image_asset_uuids.into_iter().collect();
        unstable.material_asset_uuids = other.material_asset_uuids.into_iter().collect();
        unstable.material_instance_asset_uuids =
            other.material_instance_asset_uuids.into_iter().collect();
        unstable.mesh_asset_uuids = other.mesh_asset_uuids.into_iter().collect();
        unstable.animation_asset_uuids = other.animation_asset_uuids.into_iter().collect();
        unstable.skeleton_asset_uuids = other.skeleton_asset_uuids.into_iter().collect();
        unstable
    }
}

#[derive(TypeUuid)]
#[uuid = "fc9ae812-110d-4daf-9223-e87b40966c6b"]
pub struct GltfImporter;
impl Importer for GltfImporter {
    fn version_static() -> u32
    where
        Self: Sized,
    {
        29
    }

    fn version(&self) -> u32 {
        Self::version_static()
    }

    type Options = ();

    type State = GltfImporterStateStable;

    /// Reads the given bytes and produces assets.
    fn import(
        &self,
        op: &mut ImportOp,
        source: &mut dyn Read,
        _options: &Self::Options,
        stable_state: &mut Self::State,
    ) -> atelier_assets::importer::Result<ImporterValue> {
        let mut unstable_state: GltfImporterStateUnstable = stable_state.clone().into();

        //
        // Load the GLTF file
        //
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes)?;
        let result = gltf::import_slice(&bytes);
        if let Err(err) = result {
            log::error!("GLTF Import error: {:?}", err);
            return Err(Error::Boxed(Box::new(err)));
        }

        let (doc, buffers, images) = result.unwrap();

        // Accumulate everything we will import in this list
        let mut imported_assets = Vec::new();

        let image_color_space_assignments =
            build_image_color_space_assignments_from_materials(&doc);

        //
        // Images
        //
        let images_to_import =
            extract_images_to_import(&doc, &buffers, &images, &image_color_space_assignments);
        let mut image_index_to_handle = vec![];
        for image_to_import in images_to_import {
            // Find the UUID associated with this image or create a new one
            let image_uuid = *unstable_state
                .image_asset_uuids
                .entry(image_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());

            let image_handle = make_handle(image_uuid);
            // Push the UUID into the list so that we have an O(1) lookup for image index to UUID
            image_index_to_handle.push(image_handle);

            let mut search_tags: Vec<(String, Option<String>)> = vec![];
            if let GltfObjectId::Name(name) = &image_to_import.id {
                search_tags.push(("image_name".to_string(), Some(name.clone())));
            }

            log::debug!("Importing image uuid {:?}", image_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: image_uuid,
                search_tags,
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(image_to_import.asset),
            });
        }

        //
        // GLTF Material (which we may not end up needing)
        //
        let materials_to_import =
            extract_materials_to_import(&doc, &buffers, &images, &image_index_to_handle);
        let mut material_index_to_handle = vec![];
        for material_to_import in &materials_to_import {
            // Find the UUID associated with this image or create a new one
            let material_uuid = *unstable_state
                .material_asset_uuids
                .entry(material_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());

            let material_handle = make_handle(material_uuid);

            // Push the UUID into the list so that we have an O(1) lookup for image index to UUID
            material_index_to_handle.push(material_handle);

            let mut search_tags: Vec<(String, Option<String>)> = vec![];
            if let GltfObjectId::Name(name) = &material_to_import.id {
                search_tags.push(("material_name".to_string(), Some(name.clone())));
            }

            // let mut load_deps = vec![];
            // if let Some(image) = &material_to_import.asset.base_color_texture {
            //     let image_uuid = SerdeContext::with_active(|x, _| {
            //         x.get_asset_id(image.load_handle())
            //     }).unwrap();
            //
            //     load_deps.push(AssetRef::Uuid(image_uuid));
            // }

            log::debug!("Importing material uuid {:?}", material_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: material_uuid,
                search_tags,
                build_deps: vec![],
                //load_deps,
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(material_to_import.asset.clone()),
            });
        }

        let material_handle = make_handle_from_str("92a98639-de0d-40cf-a222-354f616346c3")?;

        let null_image_handle = make_handle_from_str("fc937369-cad2-4a00-bf42-5968f1210784")?;

        //
        // Material instance
        //
        let mut material_instance_index_to_handle = vec![];
        for material_to_import in &materials_to_import {
            let material_instance_uuid = *unstable_state
                .material_instance_asset_uuids
                .entry(material_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());

            let material_instance_handle = make_handle(material_instance_uuid);

            // Push the UUID into the list so that we have an O(1) lookup for image index to UUID
            material_instance_index_to_handle.push(material_instance_handle);

            let mut search_tags: Vec<(String, Option<String>)> = vec![];
            if let GltfObjectId::Name(name) = &material_to_import.id {
                search_tags.push(("material_name".to_string(), Some(name.clone())));
            }

            let mut slot_assignments = vec![];

            let material_data_shader_param: GltfMaterialDataShaderParam =
                material_to_import.asset.material_data.clone().into();
            slot_assignments.push(MaterialInstanceSlotAssignment {
                slot_name: "per_material_data".to_string(),
                image: None,
                sampler: None,
                buffer_data: Some(
                    rafx::api_vulkan::util::any_as_bytes(&material_data_shader_param).into(),
                ),
            });

            fn push_image_slot_assignment(
                slot_name: &str,
                slot_assignments: &mut Vec<MaterialInstanceSlotAssignment>,
                image: &Option<Handle<ImageAsset>>,
                default_image: &Handle<ImageAsset>,
            ) {
                slot_assignments.push(MaterialInstanceSlotAssignment {
                    slot_name: slot_name.to_string(),
                    image: Some(image.as_ref().map_or(default_image, |x| x).clone()),
                    sampler: None,
                    buffer_data: None,
                });
            }

            push_image_slot_assignment(
                "base_color_texture",
                &mut slot_assignments,
                &material_to_import.asset.base_color_texture,
                &null_image_handle,
            );
            push_image_slot_assignment(
                "metallic_roughness_texture",
                &mut slot_assignments,
                &material_to_import.asset.metallic_roughness_texture,
                &null_image_handle,
            );
            push_image_slot_assignment(
                "normal_texture",
                &mut slot_assignments,
                &material_to_import.asset.normal_texture,
                &null_image_handle,
            );
            push_image_slot_assignment(
                "occlusion_texture",
                &mut slot_assignments,
                &material_to_import.asset.occlusion_texture,
                &null_image_handle,
            );
            push_image_slot_assignment(
                "emissive_texture",
                &mut slot_assignments,
                &material_to_import.asset.emissive_texture,
                &null_image_handle,
            );

            let material_instance_asset = MaterialInstanceAssetData {
                material: material_handle.clone(),
                slot_assignments,
            };

            log::debug!(
                "Importing material instance uuid {:?}",
                material_instance_uuid
            );

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: material_instance_uuid,
                search_tags,
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(material_instance_asset),
            });
        }

        let animations_to_import =
            extract_animations_to_import(&mut unstable_state, &doc, &buffers)?;

        for anim_to_import in animations_to_import {
            let anim_uuid = *unstable_state
                .animation_asset_uuids
                .entry(anim_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());
            log::debug!("Importing animation uuid {:?}", anim_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: anim_uuid,
                search_tags: vec![],
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(anim_to_import.asset),
            });
        }

        let skeletons_to_import = extract_skeletons_to_import(&mut unstable_state, &doc, &buffers)?;
        for skeleton_to_import in skeletons_to_import {
            let skeleton_uuid = *unstable_state
                .skeleton_asset_uuids
                .entry(skeleton_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());
            log::debug!("Importing skeleton uuid {:?}", skeleton_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: skeleton_uuid,
                search_tags: vec![],
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(skeleton_to_import.asset),
            });
        }

        // let mut vertices = PushBuffer::new(16384);
        // let mut indices = PushBuffer::new(16384);

        //
        // Meshes
        //
        let (meshes_to_import, buffers_to_import) = extract_meshes_to_import(
            op,
            &mut unstable_state,
            &doc,
            &buffers,
            //&images,
            &material_index_to_handle,
            &material_instance_index_to_handle,
        )?;

        let mut buffer_index_to_handle = vec![];
        for buffer_to_import in buffers_to_import {
            // Find the UUID associated with this image or create a new one
            let buffer_uuid = *unstable_state
                .buffer_asset_uuids
                .entry(buffer_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());

            let buffer_handle = make_handle::<BufferAssetData>(buffer_uuid);

            // Push the UUID into the list so that we have an O(1) lookup for image index to UUID
            buffer_index_to_handle.push(buffer_handle);

            log::debug!("Importing buffer uuid {:?}", buffer_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: buffer_uuid,
                search_tags: vec![],
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(buffer_to_import.asset),
            });
        }

        //let mut mesh_index_to_uuid_lookup = vec![];
        for mesh_to_import in meshes_to_import {
            // Find the UUID associated with this image or create a new one
            let mesh_uuid = *unstable_state
                .mesh_asset_uuids
                .entry(mesh_to_import.id.clone())
                .or_insert_with(|| op.new_asset_uuid());

            // Push the UUID into the list so that we have an O(1) lookup for image index to UUID
            //mesh_index_to_uuid_lookup.push(mesh_uuid.clone());

            let mut search_tags: Vec<(String, Option<String>)> = vec![];
            if let GltfObjectId::Name(name) = &mesh_to_import.id {
                search_tags.push(("mesh_name".to_string(), Some(name.clone())));
            }

            // let mut load_deps = vec![];
            //
            // // Vertex buffer dependency
            // let vertex_buffer_uuid = SerdeContext::with_active(|x, _| {
            //     x.get_asset_id(mesh_to_import.asset.vertex_buffer.load_handle())
            // }).unwrap();
            // load_deps.push(AssetRef::Uuid(vertex_buffer_uuid));
            //
            // // Index buffer dependency
            // let index_buffer_uuid = SerdeContext::with_active(|x, _| {
            //     x.get_asset_id(mesh_to_import.asset.index_buffer.load_handle())
            // }).unwrap();
            // load_deps.push(AssetRef::Uuid(index_buffer_uuid));
            //
            // // Materials dependencies
            // for mesh_part in &mesh_to_import.asset.mesh_parts {
            //     if let Some(material) = &mesh_part.material {
            //         let material_uuid = SerdeContext::with_active(|x, _| {
            //             x.get_asset_id(material.load_handle())
            //         }).unwrap();
            //         load_deps.push(AssetRef::Uuid(material_uuid));
            //     }
            // }

            log::debug!("Importing mesh uuid {:?}", mesh_uuid);

            // Create the asset
            imported_assets.push(ImportedAsset {
                id: mesh_uuid,
                search_tags,
                build_deps: vec![],
                load_deps: vec![],
                build_pipeline: None,
                asset_data: Box::new(mesh_to_import.asset),
            });
        }

        *stable_state = unstable_state.into();

        Ok(ImporterValue {
            assets: imported_assets,
        })
    }
}

fn extract_skeletons_to_import(
    state: &mut GltfImporterStateUnstable,
    doc: &gltf::Document,
    buffers: &[GltfBufferData],
) -> atelier_assets::importer::Result<Vec<SkeletonToImport>> {
    let mut skeletons = Vec::new();
    for skin in doc.skins() {
        let mut joints = Vec::new();
        let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|x| &**x));
        if let Some(matrices) = reader.read_inverse_bind_matrices() {
            for (joint, matrix_array) in skin.joints().zip(matrices) {
                let matrix = glam::mat4(
                    matrix_array[0].into(),
                    matrix_array[1].into(),
                    matrix_array[2].into(),
                    matrix_array[3].into(),
                );
                joints.push(SkeletonJoint {
                    name: joint.name().unwrap_or("").to_string(),
                    self_index: joint.index(),
                    parent: None,
                    children: joint.children().map(|node| node.index()).collect(),
                    inverse_bind_matrix: matrix,
                });
            }
        }

        // fix up parent references
        for joint_idx in 0..joints.len() {
            let joint = &joints[joint_idx];
            for child in joint.children.clone() {
                let child_mut = &mut joints[child];
                child_mut.parent = Some(joint_idx);
            }
        }
        skeletons.push(SkeletonToImport {
            id: skin
                .name()
                .map(|name| GltfObjectId::Name(name.to_string()))
                .unwrap_or(GltfObjectId::Index(skin.index())),
            asset: SkeletonAssetData { joints: joints },
        });
    }
    Ok(skeletons)
}
fn extract_animations_to_import(
    state: &mut GltfImporterStateUnstable,
    doc: &gltf::Document,
    buffers: &[GltfBufferData],
) -> atelier_assets::importer::Result<Vec<AnimationToImport>> {
    let mut animations = FnvHashMap::default();
    for anim in doc.animations() {
        // println!("anim {:?}", anim.name());
        for channel in anim.channels() {
            // println!(
            //     "channel {:?} target node {:?} property {:?} interpolation {:?} skin {:?}",
            //     idx,
            //     channel.target().node().name(),
            //     channel.target().property(),
            //     channel.sampler().interpolation(),
            //     channel.target().node().skin()
            // );
            let interpolation_mode = match channel.sampler().interpolation() {
                Interpolation::Linear => InterpolationMode::Linear,
                Interpolation::Step => InterpolationMode::Constant,
                mode => panic!("Unsupported interpolation mode {:?}", mode),
            };
            let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
            let inputs = reader.read_inputs().unwrap();
            let outputs = reader.read_outputs().unwrap();

            let track = match outputs {
                ReadOutputs::Translations(translations) => {
                    let mut translation_data = Vec::new();
                    for (translation, time) in translations.zip(inputs) {
                        // println!("{}: {:?}", time, translation);
                        translation_data.push((
                            time,
                            glam::Vec3::new(translation[0], translation[1], translation[2]),
                        ));
                    }
                    AnimTrack {
                        max_time: translation_data
                            .iter()
                            .map(|v| v.0)
                            .max_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        min_time: translation_data
                            .iter()
                            .map(|v| v.0)
                            .min_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        data: AnimTrackData::Translation(translation_data),
                        interpolation_mode,
                    }
                }
                ReadOutputs::Rotations(rotations) => {
                    let mut rotation_data = Vec::new();
                    for (rotation, time) in rotations.into_f32().zip(inputs) {
                        rotation_data.push((
                            time,
                            glam::Quat::from_xyzw(
                                rotation[0],
                                rotation[1],
                                rotation[2],
                                rotation[3],
                            ),
                        ));
                    }
                    AnimTrack {
                        max_time: rotation_data
                            .iter()
                            .map(|v| v.0)
                            .max_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        min_time: rotation_data
                            .iter()
                            .map(|v| v.0)
                            .min_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        data: AnimTrackData::Rotation(rotation_data),
                        interpolation_mode,
                    }
                }
                ReadOutputs::Scales(scales) => {
                    let mut scale_data = Vec::new();
                    for (scale, time) in scales.zip(inputs) {
                        scale_data.push((time, glam::Vec3::new(scale[0], scale[1], scale[2])));
                    }
                    AnimTrack {
                        max_time: scale_data
                            .iter()
                            .map(|v| v.0)
                            .max_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        min_time: scale_data
                            .iter()
                            .map(|v| v.0)
                            .min_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap_or(0.0),
                        data: AnimTrackData::Scale(scale_data),
                        interpolation_mode,
                    }
                }
                _ => panic!("unhandled channel type {:?}", channel.target().property()),
            };
            // println!("track {:?}", track);
            if let Some(anim_name) = anim.name() {
                // println!(
                //     "channel {:?} with target {:?} property {:?} for animation  {:?}",
                //     idx,
                //     channel.target().node().name(),
                //     channel.target().property(),
                //     anim_name
                // );
                let animation = animations
                    .entry(GltfObjectId::Name(anim_name.to_string()))
                    .or_insert_with(|| AnimationAssetData {
                        name: anim.name().expect("Animation without name").to_string(),
                        joint_tracks: Vec::new(),
                    });
                let joint_name = channel
                    .target()
                    .node()
                    .name()
                    .expect("channel target has no name");

                let joint_track = if let Some(joint_track) = animation
                    .joint_tracks
                    .iter_mut()
                    .find(|a| a.target_joint == joint_name)
                {
                    joint_track
                } else {
                    animation.joint_tracks.push(JointTrackCollection {
                        target_joint: joint_name.to_string(),
                        tracks: Vec::new(),
                    });
                    animation.joint_tracks.last_mut().unwrap()
                };
                joint_track.tracks.push(track);
            } else {
                log::error!("channel's animation has no name");
            }
        }
        // for sampler in anim.samplers() {
        //     println!("sampler {:?}", sampler);
        // }
    }
    let mut animations_to_import = Vec::new();
    for (id, anim) in animations {
        animations_to_import.push(AnimationToImport { id, asset: anim })
    }
    Ok(animations_to_import)
}

fn extract_images_to_import(
    doc: &gltf::Document,
    _buffers: &[GltfBufferData],
    images: &[GltfImageData],
    image_color_space_assignments: &FnvHashMap<usize, ImageAssetColorSpace>,
) -> Vec<ImageToImport> {
    let mut images_to_import = Vec::with_capacity(images.len());
    for image in doc.images() {
        let image_data = &images[image.index()];

        // Convert it to standard RGBA format
        use gltf::image::Format;
        use image::buffer::ConvertBuffer;
        let converted_image: image::RgbaImage = match image_data.format {
            Format::R8 => image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::R8G8 => image::ImageBuffer::<image::LumaA<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::R8G8B8 => image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::R8G8B8A8 => image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::B8G8R8 => image::ImageBuffer::<image::Bgr<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::B8G8R8A8 => image::ImageBuffer::<image::Bgra<u8>, Vec<u8>>::from_vec(
                image_data.width,
                image_data.height,
                image_data.pixels.clone(),
            )
            .unwrap()
            .convert(),
            Format::R16 => {
                unimplemented!();
            }
            Format::R16G16 => {
                unimplemented!();
            }
            Format::R16G16B16 => {
                unimplemented!();
            }
            Format::R16G16B16A16 => {
                unimplemented!();
            }
        };

        let color_space = *image_color_space_assignments
            .get(&image.index())
            .unwrap_or(&ImageAssetColorSpace::Linear);
        log::info!(
            "Choosing color space {:?} for image index {}",
            color_space,
            image.index()
        );

        let asset = ImageAssetData {
            data: converted_image.to_vec(),
            width: image_data.width,
            height: image_data.height,
            color_space,
        };
        let id = image
            .name()
            .map(|s| GltfObjectId::Name(s.to_string()))
            .unwrap_or_else(|| GltfObjectId::Index(image.index()));

        let image_to_import = ImageToImport { id, asset };

        // Verify that we iterate images in order so that our resulting assets are in order
        assert!(image.index() == images_to_import.len());
        log::debug!(
            "Importing Texture name: {:?} index: {} width: {} height: {} bytes: {}",
            image.name(),
            image.index(),
            image_to_import.asset.width,
            image_to_import.asset.height,
            image_to_import.asset.data.len()
        );

        images_to_import.push(image_to_import);
    }

    images_to_import
}

fn build_image_color_space_assignments_from_materials(
    doc: &gltf::Document
) -> FnvHashMap<usize, ImageAssetColorSpace> {
    let mut image_color_space_assignments = FnvHashMap::default();

    for material in doc.materials() {
        let pbr_metallic_roughness = &material.pbr_metallic_roughness();

        if let Some(texture) = pbr_metallic_roughness.base_color_texture() {
            image_color_space_assignments.insert(
                texture.texture().source().index(),
                ImageAssetColorSpace::Srgb,
            );
        }

        if let Some(texture) = pbr_metallic_roughness.metallic_roughness_texture() {
            image_color_space_assignments.insert(
                texture.texture().source().index(),
                ImageAssetColorSpace::Linear,
            );
        }

        if let Some(texture) = material.normal_texture() {
            image_color_space_assignments.insert(
                texture.texture().source().index(),
                ImageAssetColorSpace::Linear,
            );
        }

        if let Some(texture) = material.occlusion_texture() {
            image_color_space_assignments.insert(
                texture.texture().source().index(),
                ImageAssetColorSpace::Srgb,
            );
        }

        if let Some(texture) = material.emissive_texture() {
            image_color_space_assignments.insert(
                texture.texture().source().index(),
                ImageAssetColorSpace::Srgb,
            );
        }
    }

    image_color_space_assignments
}

fn extract_materials_to_import(
    doc: &gltf::Document,
    _buffers: &[GltfBufferData],
    _images: &[GltfImageData],
    image_index_to_handle: &[Handle<ImageAsset>],
) -> Vec<MaterialToImport> {
    let mut materials_to_import = Vec::with_capacity(doc.materials().len());

    for material in doc.materials() {
        /*
                let mut material_data = GltfMaterialData {
                    base_color_factor: [f32; 4], // default: 1,1,1,1
                    emissive_factor: [f32; 3],
                    metallic_factor: f32, //default: 1,
                    roughness_factor: f32, // default: 1,
                    normal_texture_scale: f32, // default: 1
                    occlusion_texture_strength: f32, // default 1
                    alpha_cutoff: f32, // default 0.5
                }

                let material_asset = GltfMaterialAsset {
                    material_data,
                    base_color_factor: base_color,
                    base_color_texture: base_color_texture.clone(),
                    metallic_roughness_texture: None,
                    normal_texture: None,
                    occlusion_texture: None,
                    emissive_texture: None,
                };
        */
        let mut material_asset = GltfMaterialAsset::default();

        let pbr_metallic_roughness = &material.pbr_metallic_roughness();
        material_asset.material_data.base_color_factor = pbr_metallic_roughness.base_color_factor();
        material_asset.material_data.emissive_factor = material.emissive_factor();
        material_asset.material_data.metallic_factor = pbr_metallic_roughness.metallic_factor();
        material_asset.material_data.roughness_factor = pbr_metallic_roughness.roughness_factor();
        material_asset.material_data.normal_texture_scale =
            material.normal_texture().map_or(1.0, |x| x.scale());
        material_asset.material_data.occlusion_texture_strength =
            material.occlusion_texture().map_or(1.0, |x| x.strength());
        material_asset.material_data.alpha_cutoff = material.alpha_cutoff();

        material_asset.base_color_texture = pbr_metallic_roughness
            .base_color_texture()
            .map(|texture| image_index_to_handle[texture.texture().source().index()].clone());
        material_asset.metallic_roughness_texture = pbr_metallic_roughness
            .metallic_roughness_texture()
            .map(|texture| image_index_to_handle[texture.texture().source().index()].clone());
        material_asset.normal_texture = material
            .normal_texture()
            .map(|texture| image_index_to_handle[texture.texture().source().index()].clone());
        material_asset.occlusion_texture = material
            .occlusion_texture()
            .map(|texture| image_index_to_handle[texture.texture().source().index()].clone());
        material_asset.emissive_texture = material
            .emissive_texture()
            .map(|texture| image_index_to_handle[texture.texture().source().index()].clone());

        material_asset.material_data.has_base_color_texture =
            material_asset.base_color_texture.is_some();
        material_asset.material_data.has_metallic_roughness_texture =
            material_asset.metallic_roughness_texture.is_some();
        material_asset.material_data.has_normal_texture = material_asset.normal_texture.is_some();
        material_asset.material_data.has_occlusion_texture =
            material_asset.occlusion_texture.is_some();
        material_asset.material_data.has_emissive_texture =
            material_asset.emissive_texture.is_some();

        // pub base_color_texture: Option<Handle<ImageAsset>>,
        // // metalness in B, roughness in G
        // pub metallic_roughness_texture: Option<Handle<ImageAsset>>,
        // pub normal_texture: Option<Handle<ImageAsset>>,
        // pub occlusion_texture: Option<Handle<ImageAsset>>,
        // pub emissive_texture: Option<Handle<ImageAsset>>,

        let id = material
            .name()
            .map(|s| GltfObjectId::Name(s.to_string()))
            .unwrap_or_else(|| GltfObjectId::Index(material.index().unwrap()));

        let material_to_import = MaterialToImport {
            id,
            asset: material_asset,
        };

        // Verify that we iterate images in order so that our resulting assets are in order
        assert!(material.index().unwrap() == materials_to_import.len());
        log::debug!(
            "Importing Material name: {:?} index: {}",
            material.name(),
            material.index().unwrap(),
        );

        materials_to_import.push(material_to_import);
    }

    materials_to_import
}

//TODO: This feels kind of dumb..
fn convert_to_u16_indices(
    read_indices: gltf::mesh::util::ReadIndices
) -> Result<Vec<u16>, std::num::TryFromIntError> {
    let indices_u32: Vec<u32> = read_indices.into_u32().collect();
    let mut indices_u16: Vec<u16> = Vec::with_capacity(indices_u32.len());
    for index in indices_u32 {
        indices_u16.push(index.try_into()?);
    }

    Ok(indices_u16)
}

fn extract_meshes_to_import(
    op: &mut ImportOp,
    state: &mut GltfImporterStateUnstable,
    doc: &gltf::Document,
    buffers: &[GltfBufferData],
    //images: &Vec<GltfImageData>,
    material_index_to_handle: &[Handle<GltfMaterialAsset>],
    material_instance_index_to_handle: &[Handle<MaterialInstanceAsset>],
) -> atelier_assets::importer::Result<(Vec<MeshToImport>, Vec<BufferToImport>)> {
    let mut meshes_to_import = Vec::with_capacity(doc.meshes().len());
    let mut buffers_to_import = Vec::with_capacity(doc.meshes().len() * 2);

    for mesh in doc.meshes() {

        let mut skin_joint_names = Vec::new();
        for node in doc.nodes() {
            if let Some(node_mesh) = node.mesh() {
                if node_mesh.index() != mesh.index() {
                    continue;
                }
                if let Some(node_skin) = node.skin() {
                    for joint in node_skin.joints() {
                        skin_joint_names
                            .push(joint.name().expect("joint without name").to_string());
                    }
                    break;
                }
            }
        }

        let mut buffer_data = Vec::new(); 
        let mut buffer_data_layout = Layout::new::<()>();
        let mut mesh_parts: Vec<MeshPartAssetData> = Vec::with_capacity(mesh.primitives().len());
        //
        // Iterate all mesh parts, building a single vertex and index buffer. Each MeshPart will
        // hold offsets/lengths to their sections in the vertex/index buffers
        //
        for primitive in mesh.primitives() {
            let mesh_part = {
                let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|x| &**x));

                // Positions are stored in a separate buffer
                let positions = if let Some(positions) = reader.read_positions() {
                    let layout = VertexDataLayout::build_vertex_layout(&[0f32;3], |builder, vertex| {
                        builder.add_member(vertex, "POSITION", Format::R32G32B32_SFLOAT);
                    });
                    let position_data = positions.collect::<Vec<_>>();
                    Some(VertexData::new_from_slice(&layout, &position_data))
                } else {
                    None
                };

                let mut attribute_data = Vec::new();
                if let Some(normals) = reader.read_normals() {
                    let layout = VertexDataLayout::build_vertex_layout(&[0f32;3], |builder, vertex| {
                        builder.add_member(vertex, "NORMAL", Format::R32G32B32_SFLOAT);
                    });
                    let data = normals.collect::<Vec<_>>();
                    attribute_data.push(VertexData::new_from_slice(&layout, &data));
                }
                if let Some(tangents) = reader.read_tangents() {
                    let layout = VertexDataLayout::build_vertex_layout(&[0f32;4], |builder, vertex| {
                        builder.add_member(vertex, "TANGENT", Format::R32G32B32A32_SFLOAT);
                    });
                    let data = tangents.collect::<Vec<_>>();
                    attribute_data.push(VertexData::new_from_slice(&layout, &data));
                }
                // get up to 4 color attributes
                for i in 0..4 {
                    if let Some(colors) = reader.read_colors(i) {
                        let layout = VertexDataLayout::build_vertex_layout(&[0f32;4], |builder, vertex| {
                            builder.add_member(vertex, format!("COLOR{}", i), Format::R32G32B32A32_SFLOAT);
                        });
                        let data = colors.into_rgba_f32().collect::<Vec<_>>();
                        attribute_data.push(VertexData::new_from_slice(&layout, &data));
                    }
                }
                // get up to 4 texcoord attributes
                for i in 0..4 {
                    if let Some(tex_coords) = reader.read_tex_coords(0) {
                        let layout = VertexDataLayout::build_vertex_layout(&[0f32;2], |builder, vertex| {
                            builder.add_member(vertex, format!("TEXCOORD{}", i), Format::R32G32_SFLOAT);
                        });
                        let data = tex_coords.into_f32().collect::<Vec<_>>();
                        attribute_data.push(VertexData::new_from_slice(&layout, &data));
                    }
                }
                // Indices are stored in a separate buffer
                let indices = if let Some(indices) = reader.read_indices() {
                    let layout = VertexDataLayout::build_vertex_layout(&0u16, |builder, vertex| {
                        builder.add_member(vertex, "INDEX", Format::R16_UINT);
                    });
                    if let Ok(data) = convert_to_u16_indices(indices) {
                        Some(VertexData::new_from_slice(&layout, &data))
                    } else {
                        log::error!("indices must fit in u16");
                        return Err(atelier_assets::importer::Error::Boxed(Box::new(
                            GltfImportError::new("indices must fit in u16"),
                        )));
                    }
                } else {
                    None
                };

                match (indices, positions) {
                    (
                        Some(indices),
                        Some(positions),
                    ) => {
                        //TODO: Consider computing binormal (bitangent) here

                        let (material, material_instance) =
                            if let Some(material_index) = primitive.material().index() {
                                (
                                    material_index_to_handle[material_index].clone(),
                                    material_instance_index_to_handle[material_index].clone(),
                                )
                            } else {
                                return Err(atelier_assets::importer::Error::Boxed(Box::new(
                                    GltfImportError::new(
                                        "A mesh primitive did not have a material",
                                    ),
                                )));
                            };

                        let mut vertex_layouts = Vec::new();
                        // Copy positions into buffer
                        {
                            let positions_layout = Layout::from_size_align(positions.layout.vertex_size() * positions.vertex_count, 16).unwrap();
                            let (new_buffer_data_layout, positions_offset) = buffer_data_layout.extend(positions_layout).unwrap();
                            buffer_data_layout = new_buffer_data_layout;
                            buffer_data.resize(new_buffer_data_layout.size(), 0);
                            positions.copy_to_byte_slice(&positions.layout, &mut buffer_data[positions_offset..]).expect("failed to copy vertex data");
                            vertex_layouts.push((positions.layout, positions_offset));
                        }

                        // Copy indices into buffer
                        let index_layout = {
                            let indices_layout = Layout::from_size_align(indices.layout.vertex_size() * indices.vertex_count, 16).unwrap();
                            let (new_buffer_data_layout, indices_offset) = buffer_data_layout.extend(indices_layout).unwrap();
                            buffer_data_layout = new_buffer_data_layout;
                            buffer_data.resize(new_buffer_data_layout.size(), 0);
                            indices.copy_to_byte_slice(&indices.layout, &mut buffer_data[indices_offset..]).expect("failed to copy vertex data");
                            (indices.layout, indices_offset)
                        };

                        let num_vertices = positions.vertex_count;
                        if !attribute_data.is_empty() {
                            // Combine attributes into a single packed layout
                            let attribute_members = attribute_data.iter().flat_map(|a| a.layout.members()).collect::<Vec<_>>();
                            let mut combined_attribute_size = 0; 
                            let mut members = Vec::new();
                            for (semantic, meta) in attribute_members {
                                members.push(VertexMember { 
                                    semantic: semantic.clone(),
                                    offset: meta.offset + combined_attribute_size,
                                    format: meta.format,
                                });
                                combined_attribute_size += size_of_vertex_format(meta.format).expect("invalid vertex format");
                            }
                            let combined_attribute_layout = VertexDataLayout::new(combined_attribute_size, &members);
                            // Copy combined vertex attributes
                            let attribute_segment_layout = Layout::from_size_align(combined_attribute_layout.vertex_size() * num_vertices, 16).unwrap();
                            let (new_buffer_data_layout, attribute_offset) = buffer_data_layout.extend(attribute_segment_layout).unwrap();
                            buffer_data_layout = new_buffer_data_layout;
                            buffer_data.resize(new_buffer_data_layout.size(), 0);
                            for vertex_data in attribute_data {
                                vertex_data.copy_to_byte_slice(&combined_attribute_layout, &mut buffer_data[attribute_offset..]).expect("failed to copy vertex data");
                            }
                            vertex_layouts.push((combined_attribute_layout, attribute_offset));
                        }

                        Some(MeshPartAssetData {
                            material,
                            material_instance,
                            vertex_layouts,
                            index_layout,
                            num_vertices,
                            num_indices: indices.vertex_count,
                            skin_joint_names: if !skin_joint_names.is_empty() {
                                Some(skin_joint_names.clone())
                            } else {
                                None
                            },
                        })
                    }
                    (indices, positions) => {
                        let mut missing_primitives = Vec::new();
                        if indices.is_none() {
                            missing_primitives.push("indices");
                        }
                        if positions.is_none() {
                            missing_primitives.push("positions");
                        }
                        log::error!(
                            "Mesh primitives must specify indices and positions. {}.{} is missing {}",
                            mesh.name().unwrap_or(""),
                            primitive.index(),
                            itertools::join(missing_primitives, ", "),
                        );

                        return Err(atelier_assets::importer::Error::Boxed(Box::new(
                        GltfImportError::new("Mesh primitives must specify indices, positions"),
                    )));
                    }
                }
            };

            if let Some(mesh_part) = mesh_part {
                mesh_parts.push(mesh_part);
            }
        }

 
        //
        // Vertex Buffer
        //
        let vertex_buffer_asset = BufferAssetData {
            data: buffer_data,
        };

        let vertex_buffer_id = GltfObjectId::Index(mesh.index());
        let vertex_buffer_to_import = BufferToImport {
            asset: vertex_buffer_asset,
            id: vertex_buffer_id.clone(),
        };

        let vertex_buffer_uuid = *state
            .buffer_asset_uuids
            .entry(vertex_buffer_id)
            .or_insert_with(|| op.new_asset_uuid());

        buffers_to_import.push(vertex_buffer_to_import);

        let vertex_buffer_handle = make_handle(vertex_buffer_uuid);
        let asset = MeshAssetData {
            mesh_parts,
            buffer: vertex_buffer_handle,
        };

        let mesh_id = mesh
            .name()
            .map(|s| GltfObjectId::Name(s.to_string()))
            .unwrap_or_else(|| GltfObjectId::Index(mesh.index()));

        let mesh_to_import = MeshToImport { id: mesh_id, asset };

        // Verify that we iterate meshes in order so that our resulting assets are in order
        assert!(mesh.index() == meshes_to_import.len());
        log::debug!(
            "Importing Mesh name: {:?} index: {} mesh_parts count: {}",
            mesh.name(),
            mesh.index(),
            mesh_to_import.asset.mesh_parts.len()
        );

        meshes_to_import.push(mesh_to_import);
    }

    Ok((meshes_to_import, buffers_to_import))
}
