// TODO: use header file across hlsl/cpp that has heap enums and descriptor set numbers

// Assume texel type of image descriptors is a single channel (Vulkan)
// If GCN instruction is asking for multiple channels, emulate the read/write by doing a read/write for
// each channel.
// TODO: won't this break if the underlying image data is stored in a format where channels have unequal
// number of bits? Do we need to convert the image on the CPU side? (To R8G8B8A8 for example?)

// For now, assume image format:
// - has channels with equal width
// - R,G,B,A appear in that order

// Assume emu cpu code converts the images manually
// TODO: fully emulate formats on GPU with original image data

// coord calculation: 
// 3D coord: x*channels + y*width*channels + z*width*height*channels

// Just do 1D? Perf hit? bad locality? Also need to read width, height, depth from metadata

// Assume storage images have N texels per 1 texel in Ps4 (GNM) image. N is the number of channels
// in the GNM image's format

[[vk::binding(0, 2)]]
[[vk::image_format("unknown")]]
RWTexture1D<uint> Image1DHeap[];

[[vk::binding(0, 2)]]
[[vk::image_format("unknown")]]
RWTexture2D<uint> Image2DHeap[];

[[vk::binding(0, 2)]]
[[vk::image_format("unknown")]]
RWTexture3D<uint> Image3DHeap[];

enum class Dim : uint32_t {
  Dim1D = 0,
  Dim2D = 1,
  Dim3D = 2,
  Cube = 3,
  Rect = 4,
  Buffer = 5,
  SubpassData = 6,
};

struct ImageMetadataRecord {
    uint32_t dims : 4;     // 1D, 2D, etc
    uint32_t numChannels : 3; // 1 - 4 rgba 
};

[[vk::binding(0, 3)]]
RWStructuredBuffer<ImageMetadataRecord> ImageMetadataHeap;

export uint32_t lookupImageNumChannels(uint heapIdx) {
    return ImageMetadataHeap[heapIdx].numChannels;
}

template <typename ImageTy, typename CoordTy>
[[vk::ext_capability(55 /*StorageImageReadWithoutFormat*/)]]
[[vk::ext_instruction(98 /*OpImageRead*/)]]
uint4 imageRead(ImageTy image, CoordTy coords);

template <typename ImageTy, typename CoordTy>
uint4 readTexel(ImageTy image, CoordTy coords, uint numChannels) {
    uint4 texel;
    #pragma unroll
    CoordTy coordsWithChannel = coords;

    // if num channels is 3 (RGB)
    // Read from
    // image(x, y, z)
    // image(x + 1, y, z)
    // image(x + 2, y, z)
    // etc
    for (uint i = 0; i < numChannels; i++) {
        coords.x += 1;
        texel[i] = imageRead(image, coords).x;
    }
    return texel;
}

// 1D specialization (coords needs to be different)
template <>
uint4 readTexel(RWTexture1D<uint> image, uint coord, uint numChannels) {
    uint4 texel;
    #pragma unroll
    for (uint i = 0; i < numChannels; i++) {
        texel[i] = imageRead(image, coord + i).x;
    }
    return texel;
}

// TODO:
// replace shaders with optimized versions at runtime based on images actually used
export uint4 imageLoadHelper(uint heapIdx, uint3 coords) {
    ImageMetadataRecord metadata = ImageMetadataHeap[heapIdx];
    Dim dims = (Dim) metadata.dims;
    uint numChannels = metadata.numChannels;

    uint4 texel;
    switch (dims) {
        case Dim::Dim1D:
            texel = readTexel(Image1DHeap[heapIdx], coords.x, numChannels);
            break;
        case Dim::Dim2D:
            texel = readTexel(Image2DHeap[heapIdx], coords.xy, numChannels);
            break;
        case Dim::Dim3D:
            texel = readTexel(Image3DHeap[heapIdx], coords.xyz, numChannels);
            break;
        default:
            // unhandled
            break;
    }
    return texel;
};


template <typename ImageTy, typename CoordTy>
[[vk::ext_capability(56 /*StorageImageWriteWithoutFormat*/)]]
[[vk::ext_instruction(99 /*OpImageWrite*/)]]
void imageWrite(ImageTy image, CoordTy coords, float texel);