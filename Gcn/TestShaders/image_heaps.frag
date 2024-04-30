#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require 

layout (set = 0, binding = 0) uniform sampler samp;

// VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
layout (set = 1, binding = 0) uniform texture1D TextureHeap1D[];
layout (set = 1, binding = 0) uniform texture2D TextureHeap2D[];
layout (set = 1, binding = 0) uniform texture3D TextureHeap3D[];

#define IMAGE_DIM_1D 0
#define IMAGE_DIM_2D 1
#define IMAGE_DIM_3D 2
#define IMAGE_DIM_CUBE 3
#define IMAGE_DIM_BUFFER 5

struct ImageMetadata {
    uint dims;
};

// Should be updated along with the
layout (set = 2, binding = 0) buffer TexMetadataArrayBlock {
    ImageMetadata[] textureMetadata;
};

// VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
layout (set = 3, binding = 0) uniform image1D ImageHeap1D[];
layout (set = 3, binding = 0) uniform image2D ImageHeap2D[];
layout (set = 3, binding = 0) uniform image3D ImageHeap3D[];

// Should be updated along with the
layout (set = 4, binding = 0) buffer ImageMetadataArrayBlock {
    ImageMetadata[] imageMetadata;
};


// Problems if image is stored to and sampled in same shader? (almost definitely)
// just assume that doesn't happen

layout (push_constant, std430) uniform PcBlock {
    uint imageIdx;
};

void main() {
    uint dims = imageMetadata[imageIdx].dims;

#if 0
    switch (dims) {
        case IMAGE_DIM_1D:
            texture(sampler1D(TextureHeap1D[imageIdx], samp), 0.0);
            break;
        case IMAGE_DIM_2D:
            texture(sampler2D(TextureHeap2D[imageIdx], samp), vec2(0.0));
            break;
        case IMAGE_DIM_3D:
            texture(sampler3D(TextureHeap3D[imageIdx], samp), vec3(0.0));
            break;
        default:
            break;
    }
#endif

#if 1
    switch (dims) {
        case IMAGE_DIM_1D:
            imageLoad(ImageHeap1D[imageIdx], 0);
            break;
        case IMAGE_DIM_2D:
            imageLoad(ImageHeap2D[imageIdx], ivec2(0));
            break;
        case IMAGE_DIM_3D:
            imageLoad(ImageHeap3D[imageIdx], ivec3(0));
            break;
        default:
            break;
    }
#endif
}