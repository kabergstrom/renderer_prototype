use atelier_assets::loader::handle::Handle;
use glam::{Mat4, Quat, Vec3};
use lazy_static;
use rafx::assets::ImageAsset;
use rafx::assets::MaterialInstanceAsset;
use rafx::resources::{VertexDataLayout, VertexDataSetLayout};
use rafx::{assets::BufferAsset, resources::VertexDataSet};
use serde::{Deserialize, Serialize};
use shaders::mesh_frag::MaterialDataStd140;
use type_uuid::*;

//TODO: These are extensions that might be interesting to try supporting. In particular, lights,
// LOD, and clearcoat
// Good explanations of upcoming extensions here: https://medium.com/@babylonjs/gltf-extensions-in-babylon-js-b3fa56de5483
//KHR_materials_clearcoat: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md
//KHR_materials_pbrSpecularGlossiness: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/README.md
//KHR_materials_unlit: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_unlit/README.md
//KHR_lights_punctual (directional, point, spot): https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md
//EXT_lights_image_based: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Vendor/EXT_lights_image_based/README.md
//MSFT_lod: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Vendor/MSFT_lod/README.md
//MSFT_packing_normalRoughnessMetallic: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Vendor/MSFT_packing_normalRoughnessMetallic/README.md
// Normal: NG, Roughness: B, Metallic: A
//MSFT_packing_occlusionRoughnessMetallic: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Vendor/MSFT_packing_occlusionRoughnessMetallic/README.md

// This is non-texture data associated with the material. Must convert to
// GltfMaterialDataShaderParam to bind to a shader uniform
#[derive(Serialize, Deserialize, Clone)]
#[repr(C)]
pub struct GltfMaterialData {
    // Using f32 arrays for serde support
    pub base_color_factor: [f32; 4],     // default: 1,1,1,1
    pub emissive_factor: [f32; 3],       // default: 0,0,0
    pub metallic_factor: f32,            //default: 1,
    pub roughness_factor: f32,           // default: 1,
    pub normal_texture_scale: f32,       // default: 1
    pub occlusion_texture_strength: f32, // default 1
    pub alpha_cutoff: f32,               // default 0.5

    pub has_base_color_texture: bool,
    pub has_metallic_roughness_texture: bool,
    pub has_normal_texture: bool,
    pub has_occlusion_texture: bool,
    pub has_emissive_texture: bool,
}

impl Default for GltfMaterialData {
    fn default() -> Self {
        GltfMaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_texture_scale: 1.0,
            occlusion_texture_strength: 1.0,
            alpha_cutoff: 0.5,
            has_base_color_texture: false,
            has_metallic_roughness_texture: false,
            has_normal_texture: false,
            has_occlusion_texture: false,
            has_emissive_texture: false,
        }
    }
}

pub type GltfMaterialDataShaderParam = MaterialDataStd140;

impl Into<MaterialDataStd140> for GltfMaterialData {
    fn into(self) -> MaterialDataStd140 {
        MaterialDataStd140 {
            base_color_factor: self.base_color_factor.into(),
            emissive_factor: self.emissive_factor.into(),
            metallic_factor: self.metallic_factor,
            roughness_factor: self.roughness_factor,
            normal_texture_scale: self.normal_texture_scale,
            occlusion_texture_strength: self.occlusion_texture_strength,
            alpha_cutoff: self.alpha_cutoff,
            has_base_color_texture: if self.has_base_color_texture { 1 } else { 0 },
            has_metallic_roughness_texture: if self.has_metallic_roughness_texture {
                1
            } else {
                0
            },
            has_normal_texture: if self.has_normal_texture { 1 } else { 0 },
            has_occlusion_texture: if self.has_occlusion_texture { 1 } else { 0 },
            has_emissive_texture: if self.has_emissive_texture { 1 } else { 0 },
            ..Default::default()
        }
    }
}

#[derive(TypeUuid, Serialize, Deserialize, Default, Clone)]
#[uuid = "130a91a8-ba80-4cad-9bce-848326b234c7"]
pub struct GltfMaterialAsset {
    //pub name: Option<String>,
    pub material_data: GltfMaterialData,

    pub base_color_texture: Option<Handle<ImageAsset>>,
    // metalness in B, roughness in G
    pub metallic_roughness_texture: Option<Handle<ImageAsset>>,
    pub normal_texture: Option<Handle<ImageAsset>>,
    pub occlusion_texture: Option<Handle<ImageAsset>>,
    pub emissive_texture: Option<Handle<ImageAsset>>,
    // We would need to change the pipeline for these
    // double_sided: bool, // defult false
    // alpha_mode: String, // OPAQUE, MASK, BLEND
    // support for points/lines?
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MeshPartAssetData {
    pub vertex_layouts: Vec<(VertexDataLayout, usize)>, // Vertex data layout, buffer offset
    pub index_layout: (VertexDataLayout, usize), // Indices data layout, buffer offset
    pub num_vertices: usize,
    pub num_indices: usize,
    pub material: Handle<GltfMaterialAsset>,
    pub material_instance: Handle<MaterialInstanceAsset>,
    pub skin_joint_names: Option<Vec<String>>, // index corresponds to vertex joint index
}

#[derive(TypeUuid, Serialize, Deserialize, Clone)]
#[uuid = "cf232526-3757-4d94-98d1-c2f7e27c979f"]
pub struct MeshAssetData {
    pub mesh_parts: Vec<MeshPartAssetData>,
    pub buffer: Handle<BufferAsset>, 
}

#[derive(TypeUuid, Serialize, Deserialize, Clone, Debug)]
#[uuid = "59866591-a9df-4fe4-a30c-818dbda9931c"]
pub struct SkeletonJoint {
    pub name: String,
    pub self_index: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub inverse_bind_matrix: Mat4,
}

#[derive(TypeUuid, Serialize, Deserialize, Clone, Debug)]
#[uuid = "47496390-9422-433b-ac18-6b86d275374b"]
pub struct SkeletonAssetData {
    pub joints: Vec<SkeletonJoint>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AnimTrackData {
    Translation(Vec<(f32, Vec3)>),
    Rotation(Vec<(f32, Quat)>),
    Scale(Vec<(f32, Vec3)>),
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum InterpolationMode {
    Linear,
    Constant,
}

#[derive(TypeUuid, Serialize, Deserialize, Clone, Debug)]
#[uuid = "8c1efce7-cc3f-4ff6-82a9-42100fdcbfc6"]
pub struct AnimTrack {
    pub min_time: f32,
    pub max_time: f32,
    pub interpolation_mode: InterpolationMode,
    pub data: AnimTrackData,
}

impl AnimTrack {
    pub fn sample(
        &self,
        time: f32,
        weight: f32,
        pos_out: &mut glam::Vec3,
        rot_out: &mut glam::Quat,
        scale_out: &mut glam::Vec3,
    ) {
        match &self.data {
            AnimTrackData::Translation(keyframes) => {
                let pos = {
                    if keyframes.len() == 0 {
                        glam::Vec3::default()
                    } else if keyframes.len() == 1 {
                        keyframes.get(0).unwrap().1
                    } else {
                        match &self.interpolation_mode {
                            InterpolationMode::Linear => {
                                let mut value = None;
                                for window in keyframes.windows(2) {
                                    let first = &window[0];
                                    let second = &window[1];
                                    if second.0 <= time {
                                        let duration = second.0 - first.0;
                                        let t = (first.0 - time) / duration;
                                        value = Some(first.1.lerp(second.1, t));
                                    }
                                }
                                value.unwrap_or_else(|| keyframes.last().unwrap().1)
                            }
                            mode => panic!("unexpected InterpolationMode: {:?}", mode),
                        }
                    }
                };
                *pos_out += pos * weight;
            }
            AnimTrackData::Rotation(keyframes) => {
                let rot = {
                    if keyframes.len() == 0 {
                        glam::Quat::default()
                    } else if keyframes.len() == 1 {
                        keyframes.get(0).unwrap().1
                    } else {
                        match &self.interpolation_mode {
                            InterpolationMode::Linear => {
                                let mut value = None;
                                for window in keyframes.windows(2) {
                                    let first = &window[0];
                                    let second = &window[1];
                                    if second.0 <= time {
                                        let duration = second.0 - first.0;
                                        let t = (first.0 - time) / duration;
                                        value = Some(first.1.slerp(second.1, t));
                                    }
                                }
                                value.unwrap_or_else(|| keyframes.last().unwrap().1)
                            }
                            mode => panic!("unexpected InterpolationMode: {:?}", mode),
                        }
                    }
                };
                *rot_out *= Quat::identity().slerp(rot, weight);
            }
            AnimTrackData::Scale(keyframes) => {
                let scale = {
                    if keyframes.len() == 0 {
                        glam::Vec3::one()
                    } else if keyframes.len() == 1 {
                        keyframes.get(0).unwrap().1
                    } else {
                        match &self.interpolation_mode {
                            InterpolationMode::Linear => {
                                let mut value = None;
                                for window in keyframes.windows(2) {
                                    let first = &window[0];
                                    let second = &window[1];
                                    if second.0 <= time {
                                        let duration = second.0 - first.0;
                                        let t = (first.0 - time) / duration;
                                        value = Some(first.1.lerp(second.1, t));
                                    }
                                }
                                value.unwrap_or_else(|| keyframes.last().unwrap().1)
                            }
                            mode => panic!("unexpected InterpolationMode: {:?}", mode),
                        }
                    }
                };
                *scale_out += scale * weight;
            }
        }
    }
}

#[derive(TypeUuid, Serialize, Deserialize, Clone, Debug)]
#[uuid = "59866591-a9df-4fe4-a30c-818dbda9931c"]
pub struct JointTrackCollection {
    pub tracks: Vec<AnimTrack>,
    pub target_joint: String,
}

#[derive(TypeUuid, Serialize, Deserialize, Clone, Debug)]
#[uuid = "5863a939-6325-4cae-9aba-58d7d7b885e4"]
pub struct AnimationAssetData {
    pub name: String,
    pub joint_tracks: Vec<JointTrackCollection>,
}
