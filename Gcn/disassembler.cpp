#include "GcnDialect/GcnDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <tuple>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <sys/types.h>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <llvm/ADT/ArrayRef.h>


#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"

#include "GcnBinary.h"
#include "utils.h"

#define GET_INSTRINFO_ENUM 1
#include "AMDGPUGenInstrInfo_INSTRINFO.inc"
#define GET_REGINFO_ENUM 1
#include "AMDGPUGenRegisterInfo_REGINFO.inc"
#include "SIDefines.h"

#include "spirv-tools/linker.hpp"

namespace SpirvHelperModules {
#include "SpirvLibs/libMiscHelpers.spv.h"
};

using namespace llvm;
using namespace mlir::spirv;

// For hiding llvm command line options
cl::OptionCategory MyCategory("Compiler Options", "Options for controlling the compilation process.");

cl::opt<std::string> GcnBinaryDumpPath("dump-gcn", cl::desc("dump gcn binaries in the input file to N numbered files starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<std::string> SpirvDumpPath("dump-spirv", cl::desc("dump generated spirv binaries to N numbered files starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<std::string> SpirvAsmDumpPath("dump-spirvasm", cl::desc("TODO - dump disassembled spirv to N numbered files starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<std::string> MlirDumpPath("dump-mlir", cl::desc("dump generated mlir modules to N numbered files starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<std::string> DumpAllPath("dump", cl::desc("dump all info (spirv, spirv-asm, gcn, mlir) to a prefix starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<bool> PrintGcn("p", cl::desc("print gcn bytecode to stdout"));
cl::opt<bool> Quiet("q", cl::desc("don't print warnings"));
cl::opt<bool> PrintSlots("s", cl::desc("dump info about input usage slots"));

cl::opt<std::string> InputGcnPath(cl::Positional, cl::desc("<input file>"), cl::Required, cl::value_desc("filename"), cl::cat(MyCategory));

std::optional<fs::path> getGcnBinaryDumpPath() { return !GcnBinaryDumpPath.empty() ? GcnBinaryDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getSpirvDumpPath() { return !SpirvDumpPath.empty() ? SpirvDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getSpirvAsmDumpPath() { return !SpirvAsmDumpPath.empty() ? SpirvAsmDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getMlirDumpPath() { return !MlirDumpPath.empty() ? MlirDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }

static int FileNum = 0;

static bool readFileBytes(fs::path filepath, std::vector<uint8_t> &bytes) {
    bool success = true;
    FILE *f = fopen(filepath.c_str(), "r");
    if (!f) {
        printf("Couldn't open %s for reading\n", filepath.c_str());
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    bytes.resize(len);
    fseek(f, 0, SEEK_SET);
    if (1 != fread(bytes.data(), len, 1, f)) {
        printf("Error reading from %s: %s\n", filepath.c_str(), strerror(errno));
        success = false;
    }
    fclose(f);
    return success;
}

std::unique_ptr<raw_fd_ostream> createDumpFile(fs::path prefix, fs::path extension, int number) {
    std::unique_ptr<raw_fd_ostream> ptr;
    fs::path fullPath = prefix;
    if (number > -1) {
        fullPath += std::to_string(number);
    } else if ( !prefix.has_filename()) {
        fullPath += "output";
    }
    fs::path dirpath;
    if (prefix.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(prefix.parent_path(), ec);
        if (ec) {
            printf("Couldn't create dump file directory: %s\n", ec.message().c_str());
            return ptr;
        }
    }

    fullPath += extension;

    std::error_code ec;
    ptr = std::make_unique<raw_fd_ostream>(fullPath.c_str(), ec);
    if (ec) {
        printf("Couldn't open dump file %s: %s\n", fullPath.c_str(), ec.message().c_str());
    }
    return ptr;
}

// Input vertex attributes (interpolated in frag stage)
typedef GlobalVariableOp InputAttribute;
// Export targets. Includes render targets for frag shaders, position and user-defined attributes
// otherwise.
typedef GlobalVariableOp ExportTarget;

bool isSgpr(uint regno) {
    return regno >= AMDGPU::SGPR0 && regno <= AMDGPU::SGPR105;
}

bool isSgprPair(uint regno) {
    return regno >= AMDGPU::SGPR0_SGPR1 && regno <= AMDGPU::SGPR104_SGPR105;
}

bool isVgpr(uint regno) {
    return regno >= AMDGPU::VGPR0 && regno <= AMDGPU::VGPR255;
}

bool isVgprPair(uint regno) {
    return regno >= AMDGPU::VGPR0_VGPR1 && regno <= AMDGPU::VGPR254_VGPR255;
}

uint sgprOffset(uint regno) {
    assert(isSgpr(regno));
    return regno - AMDGPU::SGPR0;
}

uint sgprPairOffset(uint regno) {
    assert(isSgprPair(regno));
    return regno - AMDGPU::SGPR0_SGPR1;
}

uint vgprOffset(uint regno) {
    assert(isVgpr(regno));
    return regno - AMDGPU::VGPR0;
}

uint vgprPairOffset(uint regno) {
    assert(isVgprPair(regno));
    return regno - AMDGPU::VGPR0_VGPR1;
}

uint sgprPairToSgprBase(uint regno) {
    assert(isSgprPair(regno));
    return AMDGPU::SGPR0 + 2 * sgprPairOffset(regno);
}

uint vgprPairToVgprBase(uint regno) {
    assert(isVgprPair(regno));
    return AMDGPU::VGPR0 + vgprPairOffset(regno);
}

uint sgprGroupToSgprBase(uint regno) {
    if (isSgpr(regno)) {
        return regno;
    } else if (isSgprPair(regno)) {
        return sgprPairToSgprBase(regno);
    } else if (regno >= AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3
                && regno <= AMDGPU::SGPR100_SGPR101_SGPR102_SGPR103) {
        return AMDGPU::SGPR0 + 4 * (regno - AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3);
    } else if (regno >= AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7
                && regno <= AMDGPU::SGPR96_SGPR97_SGPR98_SGPR99_SGPR100_SGPR101_SGPR102_SGPR103) {
        return AMDGPU::SGPR0 + 4 * (regno - AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7);
    } else if (regno >= AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7_SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15
                && regno <= AMDGPU::SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95_SGPR96_SGPR97_SGPR98_SGPR99_SGPR100_SGPR101_SGPR102_SGPR103) {
        return AMDGPU::SGPR0 + 4 * (regno - AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7_SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15);
    }
    assert(false);
}

uint vgprGroupToVgprBase(uint regno) {
    if (isVgpr(regno)) {
        return regno;
    } else if (isVgprPair(regno)) {
        return vgprPairToVgprBase(regno);
    } else if (regno >= AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3
                && regno <= AMDGPU::VGPR252_VGPR253_VGPR254_VGPR255) {
        return AMDGPU::VGPR0 + regno - AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3;
    }
    assert(false);
}

bool isRenderTarget(uint target) {
    return target >= AMDGPU::Exp::ET_MRT0 && target <= AMDGPU::Exp::ET_NULL;
}

bool isPositionTarget(uint target) {
    return target >= AMDGPU::Exp::ET_POS0 && target <= AMDGPU::Exp::ET_POS3;
}

bool isUserAttributeTarget(uint target) {
    return target >= AMDGPU::Exp::ET_PARAM0 && target <= AMDGPU::Exp::ET_PARAM31;
}

uint userAttributeOffset(uint target) {
    assert(isUserAttributeTarget(target));
    return target - AMDGPU::Exp::ET_PARAM0;
}

// Sanity check this.
// Add more special regs that *should* be allowed as 32 bit operands.
// Until we see a convoluted situation where condition codes and
// "uniforms" are mixed up and ambiguous
bool isSpecialUniformReg(uint regno) {
    switch (regno) {
        case AMDGPU::EXEC_LO:
        case AMDGPU::EXEC_HI:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
        case AMDGPU::M0_gfxpre11:
            return true;
        default:
            return false;
    }
}

bool isSpecialRegPair(uint regno) {
    // check if supported yet
    switch (regno) {
        case AMDGPU::EXEC:
        case AMDGPU::VCC:
            return true;
        default:
            return false;
    }
}

uint isScalarCcReg(uint regno) {
    switch (regno) {
        case AMDGPU::SCC:
            return true;
        default:
            return false;
    }
}

llvm::SmallString<8> getUniformRegName(uint regno) {
    llvm::SmallString<8> name;
    if (isSgpr(regno)) {
        llvm::Twine twine("s_");
        uint sgprno = sgprOffset(regno);
        twine.concat(Twine(sgprno)).toStringRef(name);
        return name;
    }
    switch (regno) {
        case AMDGPU::EXEC_LO:
            name = "exec_lo";
            break;
        case AMDGPU::EXEC_HI:
            name = "exec_hi";
            break;
        case AMDGPU::VCC_LO:
            name = "vcc_lo";
            break;
        case AMDGPU::VCC_HI:
            name = "vcc_hi";
            break;
        case AMDGPU::M0_gfxpre11:
            name = "m0";
            break;
        default:
            assert(false && "unhandled");
            break;
    }
    return name;
}

llvm::SmallString<16> getCCRegName(uint regno) {
    llvm::SmallString<16> name;
    if (isSgprPair(regno)) {
        llvm::Twine twine("s[");
        uint pairOffset = sgprPairOffset(regno);
        uint sgprlo = 2 * pairOffset;
        uint sgprhi = sgprlo + 1;
        twine.concat(Twine(sgprlo)).
                concat(":").concat(Twine(sgprhi)).concat("]").toStringRef(name);
        return name;
    }
    switch (regno) {
        case AMDGPU::EXEC:
            name = "exec";
            break;
        case AMDGPU::VCC:
            name = "vcc";
            break;
        case AMDGPU::SCC:
            name = "scc";
            break;
        default:
            assert(false && "unhandled");
            break;
    }
    return name;
}

enum GcnRegType {
    SGpr,
    SgprPair,
    VGpr,
    VGprPair,
    Special // m0, VCC, etc
};

GcnRegType regnoToType(uint regno) {
    if (isSgpr(regno)) {
        return GcnRegType::SGpr;
    } else if (isSgprPair(regno)) {
        return GcnRegType::SgprPair;
    } else if (isVgpr(regno)) {
        return GcnRegType::VGpr;
    } else if (isVgprPair(regno)) {
        return GcnRegType::VGprPair;
    }
    switch (regno) {
        case AMDGPU::EXEC_LO:
        case AMDGPU::EXEC_HI:
        case AMDGPU::EXEC:
        case AMDGPU::VCC:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
        case AMDGPU::M0_gfxpre11:
            return GcnRegType::Special;
        default:
            assert(false && "unhandled");
    }
    return GcnRegType::Special;
}

typedef std::vector<uint32_t> SpirvBytecodeVector;

class SpirvConvertResult {
public:
    enum ConvertStatus {
        success,
        fail,
        empty,
    };

    operator bool() { return status == ConvertStatus::success; }

    SpirvConvertResult(SpirvBytecodeVector &&spirvModule, ConvertStatus status):
        module(std::move(spirvModule)), status(status) 
        {}

     SpirvBytecodeVector takeSpirvModule() {
        assert(status == success);
        return std::move(module);
    }

private:
    SpirvBytecodeVector module;
    ConvertStatus status;
};

#if defined(_DEBUG)
// For printing from GDB
thread_local MCInstPrinter *ThreadIP = nullptr;
thread_local const MCSubtargetInfo *ThreadSTI = nullptr;
extern "C" void  printInst(MCInst &MI) {
    llvm::outs().flush();
    ThreadIP->printInst(&MI, 0, "", *ThreadSTI, llvm::outs());
}
#endif

// Different register types in Gcn.
// Because Gcn is a hardware ISA, it has a notion of SIMD.
// We are modeling it in SPIR-V, which is basically SIMT and hides that other threads exist.
// (besides cross-workgroup ops, etc).
// Because of this gap, it's hard to model SGPRs (scalar general puporse registers) from Gcn in spir-v.
// In a Gcn program, two consecutive SGPRs (2 x 32 bits) could be written to by a comparison op to say,
// for each bit, a[lane_id] < b[lane_id]
// Then you could source the first of the two SGPRs as the value of a loop limit.
// i.e., for (int i = 0; i < SGPR[N]; i++) { ... }
// If you wanted to model that perfectly in spir-v, you'd need to have each thread write their comparison bit
// to some kind of shared memory, in the correct position based on lane-id. Then, if the program sourced it as
// a 32-bit operand (loop counter here), you'd take the 32 bits as is. If it was sourced as a bool instead (as 
// a branch cond, for example), you'd AND it with (1 << lane_id) to get your thread's bit.

// I don't know if that's realistic in spir-v. I think we can assume the ps4 compiler never does stupid stuff like
// that example, because that would imply some kind of data sharing between threads that shouldn't exist in
// vertex and other graphics stages. Maybe there are some situations in pixel shader quads
// (groups of 4 threads), or tesselation patches, idk. Figure that out later

// So in most cases, I will represent condition codes as a single bool per thread in the SIMT model.
// Comparison instructions might write to this "CC" register. And conditions might read it.

// I will assume 32 bit scalar operands source from the "Uniform" register.
// MOV should possibly move both the UniformReg and CCReg for a given Gcn Scalar register.
// Bitwise instructions AND, OR, idk.
// Figure these out on a case by case basis

// Notes:
// VOP3a operand modifiers are present as MCInst operands, 1 per relevant operand.
// I've seen them like (dst, src0Mods, src0, src1Mods, src1, ...)
// They are only there for float args
// I haven't seen result mods yet. I've only seen extra Imm operands, 0 so far.

// """
// EXEC can be read from, and written to, through scalar instructions; it also can be
// written as a result of a vector-ALU compare. This mask affects vector-ALU,
// vector-memory, LDS, and export instructions. It does not affect scalar execution
// or branches.
// """

// Strategy for buffer (V#), image (T#), sampler (S#) resources
// CI (sea islands) shaders can be programmed in immediate or flat table mode
// immediate: SGprs are initialized with SRDs (shader resource descriptors?) on program start
// Flat table: there's a few registers that hold a pointer to a table of SRDs. V/T/S#'s are fetched from there
// before being used in mem instructions.
// Strategy is to keep one descriptor set (0) for the immediates or table.
// We will represent any SRD with a uint32 index. Members of the Flat table will be these indexes.
// They will index into a descriptor heap that is created on the CPU/Vulkan side depending on what resources are allocated
// by the ps4 graphics API. For any resource allocated by the host API, we'll allocate a vulkan resource, and append the handle in
// a heap (one for buffers, one for textures maybe, etc), and pass around an index into the heap to refer to that resource.
// For each heap, there will be a descriptor set available to shaders that can be indexed by fake SRD indexes.
// These heaps should contain all resources allocated, so they won't need to be switched in/out often. (Just to add new resources,
// garbage collect dead resources, etc)
//
// This way, we can translate how Gcn shaders work with descriptors, with arbitrary movement to/from Sgprs. Instead we pass around
// our own handles in the variables created for those SGprs.
//
// When a memory instruction wants to sample, load, store, etc, create SPIRV that indexes into
// the corresponding descriptor array (depending on the instruction's meaning) using the index stored in the sgpr variable, 
// to fetch the right buffer address, texture, sampler, etc. Then do the SPIRV sample/load/store.
// THis is overlooking the fact that a shader *could* generate a resource descriptor itself instead of fetching one
// that was allocated through the graphics API on the CPU. But Im guessing that's really rare. Will worry about it when
// I see it happen.

// Holds a single 32 bit value per 64-thread wavefront.
// Could hold a loop counter for example
// VariableOp holds an i32 type
typedef VariableOp UniformReg;
// Holds a bool per thread.
// Could hold the result of a comparison, e.g. a < b
// VariableOp holds an i1 type
typedef VariableOp CCReg;
// Holds a 32 bit value per thread
// VariableOp holds an i32 type
typedef VariableOp VectorReg;

enum DescriptorSetKind {
    SrdFlatTable,// = 0,
    SrdImmediates,// = 0, // Mutually exclusive with SrdFlatTable
    SsboHeap,
    Image1DHeap,
    Image2DHeap,
    Image3DHeap,
    ImageMetadataHeap,
    Texture1DHeap,
    Texture2DHeap,
    Texture3DHeap,
    TextureMetadataHeap,
    SamplerHeap,
    CombinedSamplerHeap,
    NUM_KINDS
};

// Some descriptor sets/bindings alias
static int getDescriptorSetNumber(DescriptorSetKind dsKind) {
    static int descriptorSetKindToNumber[DescriptorSetKind::NUM_KINDS] = {
        0, // SrdFlatTable
        0, // SrdImmediates
        1, // SsboHeap
        2, // Image1DHeap
        2, // Image2DHeap
        2, // Image3DHeap
        3, // ImageMetadataHeap
        4, // Texture1DHeap,
        4, // Texture2DHeap,
        4, // Texture3DHeap,
        5, // TextureMetadataHeap,
        6, // SamplerHeap
        7, // CombinedSamplerHeap
    };

    return descriptorSetKindToNumber[dsKind];
}

enum SrdMode {
    Immediate,
    FlatTable,
};

class GcnToSpirvConverter {
public:
    GcnToSpirvConverter(const std::vector<MCInst> &machineCode, ExecutionModel execModel, SrdMode srdMode, std::vector<InputUsageSlot> srdSlots, const MCSubtargetInfo *STI, MCInstPrinter *IP):
        mlirContext(), 
        builder(mlir::UnknownLoc::get(&mlirContext), &mlirContext),
        functionBuilder(mlir::UnknownLoc::get(&mlirContext), &mlirContext),
        execModel(execModel),
        machineCode(machineCode),
        status(SpirvConvertResult::empty),
        srdMode(srdMode),
        srdSlots(srdSlots),
        mainEntryBlock(nullptr),
        moduleOp(nullptr),
        STI(STI),
        IP(IP)
    {
        mlirContext.getOrLoadDialect<mlir::gcn::GcnDialect>();
        mlirContext.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
        
        floatTy = builder.getF32Type();
        f64Ty = builder.getF64Type();
        float16Ty = builder.getF16Type();
        v2float32Ty = mlir::VectorType::get({2}, floatTy);
        v2float16Ty = mlir::VectorType::get({2}, float16Ty);

        intTy = builder.getI32Type();
        sintTy = builder.getIntegerType(32, true);
        sint64Ty = builder.getIntegerType(64, true);
        int64Ty = builder.getI64Type();
        int16Ty = builder.getI16Type();
        int8Ty = builder.getI8Type();
        boolTy = builder.getI1Type();
        v2intTy = mlir::VectorType::get({2}, intTy);
        v3intTy = mlir::VectorType::get({3}, intTy);

        arithCarryTy = StructType::get({ intTy, intTy });

        auto dwordArrayType = RuntimeArrayType::get(intTy, 4);
        ssboBlockDwordPtrType = PointerType::get(dwordArrayType, StorageClass::PhysicalStorageBuffer);

        initSpirvLibraries();

#if defined(_DEBUG)
        ThreadIP = IP;
        ThreadSTI = STI;
#endif
    }

    SpirvConvertResult convert();

private:
    bool processSrds();
    bool prepass();
    bool convertGcnOp(const MCInst &MI);
    bool buildSpirvDialect();
    bool finalizeModule();
    bool generateSpirvBytecode();

    bool isVertexShader() { return execModel == ExecutionModel::Vertex; }
    bool isFragShader() { return execModel == ExecutionModel::Fragment; }
    bool isFlatTableSrdMode() { return srdMode == SrdMode::FlatTable; }
    bool isImmediateSrdMode() { return srdMode == SrdMode::Immediate; }

    ConstantOp getZero(mlir::Type ty) { return ConstantOp::getZero(ty, builder.getLoc(), builder); }
    ConstantOp getOne(mlir::Type ty) { return ConstantOp::getOne(ty, builder.getLoc(), builder); }
    ConstantOp getF32Const(float f) { return builder.create<ConstantOp>(floatTy, builder.getF32FloatAttr(f)); }
    ConstantOp getF64Const(double d) { return builder.create<ConstantOp>(f64Ty, builder.getF64FloatAttr(d)); }
    ConstantOp getI32Const(int n) { return builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(n)); }
    ConstantOp getI64Const(int n) { return builder.create<ConstantOp>(int64Ty, builder.getI64IntegerAttr(n)); }
    ConstantOp getI8Const(int n) { return builder.create<ConstantOp>(int8Ty, builder.getI8IntegerAttr(n)); }

    enum SpirvLibrary {
        MISC_HELPER_FUNCTIONS,
        NUM_SPIRV_LIBRARIES
    };

    std::array<ArrayRef<uint32_t>, NUM_SPIRV_LIBRARIES> spirvLibraries;
    std::array<bool, NUM_SPIRV_LIBRARIES> usedSpirvLibraries {};

    void initSpirvLibraries() {
        spirvLibraries[MISC_HELPER_FUNCTIONS] = ArrayRef(reinterpret_cast<const uint32_t *>(SpirvHelperModules::g_miscHelpersBytes)
                , sizeof(SpirvHelperModules::g_miscHelpersBytes) / 4);
    }

    ArrayRef<uint32_t> loadSpirvLibrary(SpirvLibrary library) {
        usedSpirvLibraries[library] = true;
        return spirvLibraries[library];
    }

    // For now assume no overloading - all names unique
    // TODO: enumerate all library functions? Make table w/ names and types?
    llvm::StringMap<FuncOp> libraryFunctionDeclarations;

    FuncOp getOrInsertLibraryCallDecl(StringRef name, SmallVector<mlir::Type, 4> paramValueTypes, mlir::Type retTy) {
        auto it = libraryFunctionDeclarations.find(name);
        if (it != libraryFunctionDeclarations.end()) {
            return it->second;
        }

        llvm::SmallVector<mlir::Type, 4> paramRefTypes;
        for (mlir::Type valueType: paramValueTypes) {
            // For now convert all params to pointer type.
            // TODO match the spir-v generator (only do vectors and primitives?)
            mlir::Type refType = PointerType::get(valueType, StorageClass::Function);
            paramRefTypes.push_back(refType);
        }

        mlir::FunctionType fTy = mlir::FunctionType::get(&mlirContext, paramRefTypes, retTy);
        auto fn = functionBuilder.create<FuncOp>(name, fTy);
        fn.setLinkageAttributesAttr(LinkageAttributesAttr::get(&mlirContext, functionBuilder.getStringAttr(name), LinkageTypeAttr::get(&mlirContext, LinkageType::Import)));

        libraryFunctionDeclarations[name] = fn;
        return fn;
    }

    FunctionCallOp createLibraryCall(FuncOp fn, ArrayRef<mlir::Value> args) {
        mlir::FunctionType fTy = fn.getFunctionType();
        auto ref = mlir::FlatSymbolRefAttr::get(fn);

        // Create OpVariables for each arg and store the args to those.
        // Pass the OpVariable to the call.
        // The OpFunctionParameters will be setup as pointers,
        // since that's what glslang and dxc generate.
        SmallVector<mlir::Value, 4> spirvParams;
        for (uint i = 0; i < args.size(); i++) {
            auto arg = args[i];
            SmallString<24> paramName;
            Twine twine;
            twine.concat(fn->getName().getStringRef())
                    .concat(".param")
                    .concat(Twine(i))
                    .toStringRef(paramName);
            VariableOp param = buildTempReg(paramName, arg.getType());

            builder.create<StoreOp>(param, arg);
            spirvParams.push_back(param);
        }

        return builder.create<FunctionCallOp>(fTy.getResult(0), ref, spirvParams);
    }

    bool linkSpirvModules(SpirvBytecodeVector mainModule, const std::vector<ArrayRef<uint32_t>> &libraries, SpirvBytecodeVector &linkedModule) {
        spvtools::Context spvtoolsCtx(spv_target_env::SPV_ENV_VULKAN_1_3);
        spvtoolsCtx.SetMessageConsumer([](spv_message_level_t /* level */, const char* /* source */,
                const spv_position_t& /* position */, const char* message) {
            llvm::errs() << message << "\n";
        });

        std::vector<const uint32_t *> allInputModules;
        std::vector<size_t> moduleSizes;

        allInputModules.push_back(mainModule.data());
        moduleSizes.push_back(mainModule.size());

        for (auto lib: libraries) {
            allInputModules.push_back(lib.data());
            moduleSizes.push_back(lib.size());
        }

        spvtools::LinkerOptions linkOptions;
        linkOptions.SetVerifyIds(true);
        linkOptions.SetCreateLibrary(false);
        
        //spvtoolsCtx.SetMessageConsumer(MessageConsumer consumer)
        auto res = spvtools::Link(spvtoolsCtx, allInputModules.data(), moduleSizes.data(), allInputModules.size(), &linkedModule, linkOptions);
        return res == SPV_SUCCESS;
    }

    // build a VariableOp for the CC (condition code) register, and attach a name
    CCReg buildCCReg(llvm::StringRef name) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto boolPtrTy = PointerType::get(boolTy, StorageClass::Function);
        auto var = initBuilder.create<VariableOp>(boolPtrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), nullptr);
#if defined(_DEBUG)
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        return var;
    }
    // build and name a vgpr or uniform register.
    // The register holds either a 32 value per thread (vgpr) or a single 32 bit value per 64-thread wavefront.
    VariableOp build32BitReg(llvm::StringRef name) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto intPtrTy = PointerType::get(intTy, StorageClass::Function);
        auto var = initBuilder.create<VariableOp>(intPtrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), nullptr);
#if defined(_DEBUG)
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        return var;
    }

    VariableOp buildTempReg(llvm::StringRef name, mlir::Type dataType) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto ptrTy = PointerType::get(dataType, StorageClass::Function);
        auto var = initBuilder.create<VariableOp>(ptrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), nullptr);
#if defined(_DEBUG)
        // TODO handle dup names. Don't matter though for function, just reading
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        return var;
    }

    VectorReg initVgpr(uint regno) {
        auto it = usedVgprs.find(regno);
        if (it == usedVgprs.end()) {
            llvm::SmallString<8> name;
            uint vgprNameOffset = vgprOffset(regno);
            llvm::Twine twine("v_");
            twine.concat(Twine(vgprNameOffset)).toStringRef(name);
            VariableOp vgpr = build32BitReg(name);
            auto it = usedVgprs.insert(std::make_pair(regno, vgpr));
            return it.first->second;
        } else {
            return it->second;
        }
    }
    CCReg initCCReg(uint regno) {
        auto it = usedCCRegs.find(regno);
        if (it == usedCCRegs.end()) {
            auto name = getCCRegName(regno);
            CCReg cc = buildCCReg(name);
            auto it = usedCCRegs.insert(std::make_pair(regno, cc));
            return it.first->second;
        } else {
            return it->second;
        }
    }
    VariableOp initUniformReg(uint regno) {
        auto it = usedUniformRegs.find(regno);
        if (it == usedUniformRegs.end()) {
            auto name = getUniformRegName(regno);
            UniformReg uniform = build32BitReg(name);
            auto it = usedUniformRegs.insert(std::make_pair(regno, uniform));
            return it.first->second;
        } else {
            return it->second;
        }
    }

    GlobalVariableOp buildInputAttribute(llvm::StringRef name, uint attrno, mlir::Type attrType) {
        mlir::Region &region = moduleOp.getBodyRegion();
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(&(*region.begin()));
        auto attrPtrTy = PointerType::get(attrType, StorageClass::Input);
        auto var = initBuilder.create<GlobalVariableOp>(attrPtrTy, name, nullptr);
#if defined(_DEBUG)
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        var->setAttr("Location", builder.getI32IntegerAttr(attrno));
        return var;
    }

    GlobalVariableOp buildExportTarget(llvm::StringRef name, uint target, mlir::Type attrType) {
        mlir::Region &region = moduleOp.getBodyRegion();
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(&(*region.begin()));
        auto attrPtrTy = PointerType::get(attrType, StorageClass::Output);
        auto var = initBuilder.create<GlobalVariableOp>(attrPtrTy, name, nullptr);
#if defined(_DEBUG)
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        int vkLocation = -1;
        if (isUserAttributeTarget(target)) {
            vkLocation = userAttributeOffset(target);
        } else if (isRenderTarget(target)) {
            assert(fragExportTargetToVkLocation.contains(target));
            vkLocation = fragExportTargetToVkLocation[target];
        }

        if (vkLocation >= 0) {
            var->setAttr("Location", builder.getI32IntegerAttr(vkLocation));
        }

        return var;
    }

    GlobalVariableOp initInputAttribute(uint attrno, mlir::Type attrType) {
        auto it = inputAttributes.find(attrno);
        if (it == inputAttributes.end()) {
            llvm::SmallString<16> name;
            llvm::Twine twine("inAttr_");
            twine.concat(Twine(attrno)).toStringRef(name);
            InputAttribute attr = buildInputAttribute(name, attrno, attrType);
            auto it = inputAttributes.insert(std::make_pair(attrno, attr));
            return it.first->second;
        } else {
            return it->second;
        }
    }

    llvm::SmallString<16> nameExportTarget(uint target) {
        llvm::SmallString<16> name;
        llvm::Twine twine;
        switch (execModel) {
            case ExecutionModel::Fragment:
            {
                assert(isRenderTarget(target));
                assert(target != AMDGPU::Exp::ET_NULL);
                if (target == AMDGPU::Exp::ET_MRTZ) {
                    name = "mrtZ";
                } else {
                    uint mrtIdx = target - AMDGPU::Exp::ET_MRT0;
                    twine.concat("mrt").concat(Twine(mrtIdx)).toStringRef(name);
                }
                break;
            }
            case ExecutionModel::Vertex:
            {
                if (isPositionTarget(target)) {
                    // When would multiple position exports happen?
                    // multi view?
                    assert(target == AMDGPU::Exp::ET_POS0 && "unhandled position export in vert shader");
                    name = "position";
                } else  {
                    assert(isUserAttributeTarget(target));
                    uint userAttribIdx = userAttributeOffset(target);
                    twine.concat("outAttr_").concat(Twine(userAttribIdx)).toStringRef(name);
                }
                break;
            }
            default:
                assert(false && "unhandled");
        }
        return name;
    }

    GlobalVariableOp initExportTarget(uint target, mlir::Type expTy) {
        auto it = exportTargets.find(target);
        if (it == exportTargets.end()) {
            auto name = nameExportTarget(target);
            ExportTarget attr = buildExportTarget(name, target, expTy);
            auto it = exportTargets.insert(std::make_pair(target, attr));
            return it.first->second;
        } else {
            return it->second;
        }
    }

    mlir::Value sourceCCReg(uint regno) {
        assert(isSgprPair(regno) || isSpecialRegPair(regno) || isScalarCcReg(regno));
        VariableOp cc = initCCReg(regno);
        return builder.create<LoadOp>(cc);
    }

    mlir::Value sourceUniformReg(uint regno) {
        assert(isSpecialUniformReg(regno) || isSgpr(regno));
        VariableOp uni = initUniformReg(regno);
        return builder.create<LoadOp>(uni);
    }

    mlir::Value sourceVectorGpr(uint regno) {
        assert(isVgpr(regno));
        VariableOp vgpr = initVgpr(regno);
        return builder.create<LoadOp>(vgpr);
    }

    mlir::spirv::ConstantOp sourceImmediate(MCOperand &operand, mlir::Type type) {
        // Literal constant – a 32-bit value in the instruction stream. When a literal
        // constant is used with a 64bit instruction, the literal is expanded to 64 bits by:
        // padding the LSBs with zeros for floats, padding the MSBs with zeros for
        // unsigned ints, and by sign-extending signed ints.

        // TODO veify that signdness doesn't matter, and getImm() an already sign extended value when
        // it matters
        if (auto intTy = dyn_cast<mlir::IntegerType>(type)) {
            switch (intTy.getWidth()) {
                case 32:
                {
                    int32_t imm = static_cast<uint32_t>(operand.getImm() & 0xffffffff);
                    return getI32Const(imm);
                }
                case 64:
                {
                    // Not sure if operand comes sign extended or not, do it in case.
                    int64_t imm = operand.getImm();
                    return getI64Const(imm);
                }
                default:
                    assert(false && "unhandled int width");
                    break;
            }
        } else if (auto floatTy = dyn_cast<mlir::FloatType>(type)) {
            switch (floatTy.getWidth()) {
                case 32:
                {
                    uint32_t imm = operand.getImm() & 0xffffffff;
                    float fImm = llvm::bit_cast<float>(imm);
                    return getF32Const(fImm);
                }
                case 64:
                {
                    uint64_t imm = operand.getImm();
                    // extend float by padding LSBs, just in case. Same if already padded or not.
                    double dImm = llvm::bit_cast<double>(imm | (imm << 32));
                    return getF64Const(dImm);
                }
                default:
                    assert(false && "unhandled float width");
            }
        }
        assert(false && "unhandled immediate");
        return nullptr;
    }

    mlir::Value sourceScalarOperandUniform(MCOperand &operand, mlir::Type type) {
        mlir::Value rv;
        if (operand.isImm()) {
            ConstantOp immConst = sourceImmediate(operand, type);
            return immConst;
        }
        assert(operand.isReg());
        uint regno = operand.getReg();
    
        if (regnoToType(regno) == GcnRegType::Special) {
            assert(isSpecialUniformReg(regno));
        }
        // SGpr or special reg (VCC_LO, m0, etc)
        rv = sourceUniformReg(regno);

        if (type != rv.getType()) {
            rv = builder.create<BitcastOp>(type, rv);
        }
        return rv;
    }

    mlir::Value sourceCCOperand(MCOperand &operand) {
        mlir::Value rv;
        if (operand.isImm()) {
            // Only support this in compute (TODO)
            // return bool(laneID & imm)
            assert(false && "Unhandled: condition code source can't be an immediate");
        }

        assert(operand.isReg());
        uint regno = operand.getReg();

        if (regnoToType(regno) == GcnRegType::Special) {
            assert(isSpecialRegPair(regno));
        }

        rv = sourceCCReg(regno);

        return rv;
    }

    mlir::Value sourceVectorOperand(MCOperand &operand, mlir::Type type) {
        assert(type.getIntOrFloatBitWidth() == 32);
        mlir::Value rv;
        if (operand.isImm()) {
            ConstantOp immConst = sourceImmediate(operand, type);
            return immConst;
        } 
        
        assert(operand.isReg());
        uint regno = operand.getReg();
        switch (regnoToType(regno)) {
            case Special:
                assert(isSpecialUniformReg(regno));
                LLVM_FALLTHROUGH;
            case SGpr:
                rv =  sourceUniformReg(regno);
                break;
            case VGpr:
                rv = sourceVectorGpr(regno);
                break;
            default:
                assert(false && "unhandled");
                return nullptr;
        }
        if (type != rv.getType()) {
            rv = builder.create<BitcastOp>(type, rv);
        }
        return rv;
    }

    mlir::Value create64BitValue(mlir::Value lo, mlir::Value hi) {
        assert(lo.getType() == intTy && hi.getType() == intTy);
        hi = builder.create<UConvertOp>(int64Ty, hi);
        hi = builder.create<ShiftLeftLogicalOp>(hi, getI32Const(32));
        lo = builder.create<UConvertOp>(int64Ty, lo);
        return builder.create<BitwiseOrOp>(hi, lo);
    }

    mlir::Value sourceScalarOperand64(uint regno, mlir::Type type) {
        assert(type.getIntOrFloatBitWidth() == 64);
        mlir::Value lo, hi;

        mlir::Value rv;
        switch(regnoToType(regno)) {
            case Special:
            {
                assert(isSpecialRegPair(regno));
                uint loSrcNo, hiSrcNo;
                // TODO factor this out into like "specialRegPairToBase"
                switch (regno) {
                    case AMDGPU::VCC:
                        loSrcNo = AMDGPU::VCC_LO;
                        hiSrcNo = AMDGPU::VCC_HI;
                        break;
                    case AMDGPU::EXEC:
                        loSrcNo = AMDGPU::EXEC_LO;
                        hiSrcNo = AMDGPU::EXEC_HI;
                        break;
                    default:
                        loSrcNo = -1;
                        hiSrcNo = -1;
                        assert(false && "unhandled");
                        break;
                }
                lo = sourceUniformReg(loSrcNo);
                hi = sourceUniformReg(hiSrcNo);
                rv = create64BitValue(lo, hi);
                break;
            }
            case SgprPair:
            {
                uint base = sgprPairToSgprBase(regno);
                lo = sourceUniformReg(base);
                hi = sourceUniformReg(base + 1);
                rv = create64BitValue(lo, hi);
                break;
            }
            default:
                assert(false && "unhandled");
                rv = nullptr;
                break;
        }

        if (rv.getType() != type) {
            rv = builder.create<BitcastOp>(type, rv);
        }
        return rv;
    }

    mlir::Value sourceScalarOperand64(MCOperand &operand, mlir::Type type) {
        assert(type.getIntOrFloatBitWidth() == 64);
        if (operand.isImm()) {
            ConstantOp immConst = sourceImmediate(operand, type);
            return immConst;
        }

        assert(operand.isReg());
        uint regno = operand.getReg();
        return sourceScalarOperand64(regno, type);
    }

    mlir::Value sourceVectorOperand64(MCOperand &operand, mlir::Type type) {
        assert(type.getIntOrFloatBitWidth() == 64);
        if (operand.isImm()) {
            ConstantOp immConst = sourceImmediate(operand, type);
            return immConst;
        }

        assert(operand.isReg());
        uint regno = operand.getReg();
        switch(regnoToType(regno)) {
            case Special:
            case SgprPair:
                return sourceScalarOperand64(regno, type);
            case VGpr:
            {
                mlir::Value lo, hi;
                uint base = vgprPairToVgprBase(regno);
                lo = sourceVectorGpr(base);
                hi = sourceVectorGpr(base + 1);
                return create64BitValue(lo, hi);
            }
            default:
                assert(false && "unhandled");
                return nullptr;
        }
    }

    template<typename... Idxs>
    const std::array<MCOperand, sizeof...(Idxs)> unwrapMI(const MCInst &MI, Idxs... args) {
        return { (MI.getOperand(args))... };
    }

#define R0_2 0, 1, 2
#define R0_3 R0_2, 3
#define R0_4 R0_3, 4
#define R0_5 R0_4, 5
#define R0_6 R0_5, 6
#define R0_7 R0_6, 7
#define R0_8 R0_7, 8
#define R0_9 R0_8, 9
#define R0_10 R0_9, 10

    void storeScalarResult32(uint regno, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if (result.getType() != intTy) {
            result = builder.create<BitcastOp>(intTy, result);
        }

        assert(isSgpr(regno) || isSpecialUniformReg(regno));

        UniformReg uniform = initUniformReg(regno);
        builder.create<StoreOp>(uniform, result);
    }

    void storeScalarResult32(MCOperand &dst, mlir::Value result) {
        assert(dst.isReg());
        storeScalarResult32(dst.getReg(), result);
    }

    void storeScalarResult64(uint regno, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 64);
        if (result.getType() != int64Ty) {
            result = builder.create<BitcastOp>(int64Ty, result);
        }

        uint loDstNo, hiDstNo;
        if (regnoToType(regno) == GcnRegType::Special) {
            // TODO factor this
            switch (regno) {
                case AMDGPU::VCC:
                    loDstNo = AMDGPU::VCC_LO;
                    hiDstNo = AMDGPU::VCC_HI;
                    break;
                case AMDGPU::EXEC:
                    loDstNo = AMDGPU::EXEC_LO;
                    hiDstNo = AMDGPU::EXEC_HI;
                    break;
                default:
                    loDstNo = -1;
                    hiDstNo = -1;
                    break;
            }
        } else {
            assert(isSgprPair(regno));
            loDstNo = sgprPairToSgprBase(regno);
            hiDstNo = loDstNo + 1;
        }

        mlir::Value lo, hi;
        // val, offset, count
        ConstantOp fullmask = getI32Const(0xffffffff);
        lo = builder.create<BitFieldUExtractOp>(result, getZero(intTy), fullmask);
        lo = builder.create<UConvertOp>(intTy, lo);
        hi = builder.create<BitFieldUExtractOp>(result, getI32Const(32), fullmask);
        hi = builder.create<UConvertOp>(intTy, hi);
        storeScalarResult32(loDstNo, lo);
        storeScalarResult32(hiDstNo, hi);
    }

    void storeScalarResult64(MCOperand &dst, mlir::Value result) {
        uint regno = dst.getReg();
        assert(isSgprPair(regno) || isSpecialRegPair(regno));
        return storeScalarResult64(regno, result);
    }

    StoreOp storeResultCC(uint regno, mlir::Value result) {
        assert(isSgprPair(regno) || isSpecialRegPair(regno) || isScalarCcReg(regno));
        CCReg cc = initCCReg(regno);
        return builder.create<StoreOp>(cc, result);
    }

    StoreOp storeResultCC(MCOperand &dst, mlir::Value result) {
        assert(result.getType() == boolTy);
        uint regno = dst.getReg();
        return storeResultCC(regno, result);
    }

    void tagPredicated(mlir::Operation *op) {
        op->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getUnitAttr());
    }

    void storeVectorResult32(uint regno, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if (result.getType() != intTy) {
            result = builder.create<BitcastOp>(intTy, result);
        }
        VectorReg vgpr = initVgpr(regno);
        auto store = builder.create<StoreOp>(vgpr, result);
        // Don't write back the result if thread is masked off by EXEC
        // Will need to generate control flow, TODO
        tagPredicated(store);
    }

    void storeVectorResult32(MCOperand &dst, mlir::Value result) {
        assert(dst.isReg() && isVgpr(dst.getReg()));
        uint regno = dst.getReg();
        return storeVectorResult32(regno, result);
    }

    void storeVectorResult64(uint regno, mlir::Value result) {
        assert(isVgprPair(regno));
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 64);
        if (result.getType() != int64Ty) {
            result = builder.create<BitcastOp>(intTy, result);
        }

        mlir::Value lo, hi;
        // val, offset, count
        ConstantOp fullmask = getI32Const(0xffffffff);
        lo = builder.create<BitFieldUExtractOp>(result, getZero(intTy), fullmask);
        lo = builder.create<UConvertOp>(intTy, lo);
        hi = builder.create<BitFieldUExtractOp>(result, getI32Const(32), fullmask);
        hi = builder.create<UConvertOp>(intTy, hi);
        uint base = vgprPairToVgprBase(regno);
        storeVectorResult32(base, lo);
        storeVectorResult32(base + 1, hi);
    }

    // For VOP3a
    mlir::Value applyOperandModifiers(mlir::Value src, MCOperand &operandMods) {
        uint mods = operandMods.getImm();
        mlir::Value rv = src;
        if (mods & SISrcMods::ABS) {
            assert(src.getType().isa<mlir::FloatType>());
            rv = builder.create<GLFAbsOp>(rv);
        }
        if (mods & SISrcMods::NEG) {
            assert(src.getType().isa<mlir::FloatType>());
            rv = builder.create<FNegateOp>(rv);
        }
        return rv;
    }

    // For VOP3a
    // Don't know where clamp comes from yet, will need to see it printed
    mlir::Value applyResultModifiers(mlir::Value result, int clamp, int omod) {
        mlir::Value rv = result;
        mlir::Type resultTy = result.getType();
        if (clamp) {
            assert(!(clamp & ~1));
            assert(resultTy.isa<mlir::FloatType>());
            rv = builder.create<GLFClampOp>(resultTy, rv, getZero(resultTy), getOne(resultTy));
        }
        assert(!omod || resultTy.isa<mlir::FloatType>());
        switch(omod) {
            case 0:
                break;
            case SIOutMods::MUL2:
                rv = builder.create<FMulOp>(rv, builder.create<ConstantOp>(resultTy, builder.getFloatAttr(resultTy, 2.0)));
                break;
            case SIOutMods::MUL4:
                rv = builder.create<FMulOp>(rv, builder.create<ConstantOp>(resultTy, builder.getFloatAttr(resultTy, 4.0)));
                break;
            case SIOutMods::DIV2:
                rv = builder.create<FDivOp>(rv, builder.create<ConstantOp>(resultTy, builder.getFloatAttr(resultTy, 2.0)));
                break;
            default:
                assert(false && "Unhandled OMOD");
                break;
        }
        return rv;
    }

    GlobalVariableOp initDescriptorHeap(DescriptorSetKind dsKind, mlir::Type type, llvm::StringRef name) {
        uint setNumber = getDescriptorSetNumber(dsKind);
        auto it = descriptorSets.find(dsKind);
        if (it == descriptorSets.end()) {
            mlir::ImplicitLocOpBuilder initBuilder(builder);
            initBuilder.setInsertionPointToStart(mainEntryBlock);
            PointerType ptrTy = PointerType::get(type, StorageClass::Uniform);
            auto var = initBuilder.create<GlobalVariableOp>(ptrTy, name, nullptr);
            var->setAttr("DescriptorSet", initBuilder.getI32IntegerAttr(setNumber));
            var->setAttr("Binding", initBuilder.getI32IntegerAttr(0));
            descriptorSets.insert(std::make_pair(dsKind, var));
            return var;
        } else {
            return it->second;
        }
    }

    GlobalVariableOp initImmediateDataUBO(mlir::Type immDataBlockTy) {
        // Should only be called once
        DescriptorSetKind dsKind = DescriptorSetKind::SrdImmediates; 
        uint setNumber = getDescriptorSetNumber(dsKind);
        PointerType ptrTy = PointerType::get(immDataBlockTy, StorageClass::Uniform);
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto var = initBuilder.create<GlobalVariableOp>(ptrTy, "ImmediateUserdata");
        var->setAttr("DescriptorSet", initBuilder.getI32IntegerAttr(setNumber));
        var->setAttr("Binding", initBuilder.getI32IntegerAttr(0));
        descriptorSets.insert(std::make_pair(dsKind, var));
        return var;
    }

    GlobalVariableOp initSsboHeap() {
        // { address : 64, stride : 32, num_records : 32 }
        mlir::Type entryType = StructType::get({int64Ty, intTy, intTy}, {0, 8, 12});
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        // separate UBO and SSBO heap?
        // probably not because given SGPR(s) in a memory inst, we don't know which heap to index if they're separate
        return initDescriptorHeap(DescriptorSetKind::SsboHeap, heapTy, "SsboHeap");
    }

    GlobalVariableOp initImage1DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim1D);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Image1DHeap, heapTy, "Image1DHeap");
    }

    GlobalVariableOp initImage2DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim2D);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Image2DHeap, heapTy, "Image2DHeap");
    }

    GlobalVariableOp initImage3DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim3D);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Image3DHeap, heapTy, "Image3DHeap");
    }

    GlobalVariableOp initTexture1DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim1D, ImageDepthInfo::DepthUnknown, ImageArrayedInfo::NonArrayed,
                ImageSamplingInfo::SingleSampled, ImageSamplerUseInfo::NeedSampler);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Texture1DHeap, heapTy, "Texture1DHeap");
    }

    GlobalVariableOp initTexture2DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim2D, ImageDepthInfo::DepthUnknown, ImageArrayedInfo::NonArrayed,
                ImageSamplingInfo::SingleSampled, ImageSamplerUseInfo::NeedSampler);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Texture2DHeap, heapTy, "Texture2DHeap");
    }

    GlobalVariableOp initTexture3DHeap() {
        mlir::Type entryType = ImageType::get(floatTy, Dim::Dim3D, ImageDepthInfo::DepthUnknown, ImageArrayedInfo::NonArrayed,
                ImageSamplingInfo::SingleSampled, ImageSamplerUseInfo::NeedSampler);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::Texture3DHeap, heapTy, "Texture3DHeap");
    }

    mlir::Value extractImageDimType(mlir::Value imageMetadataRecord) {
        return builder.create<CompositeExtractOp>(imageMetadataRecord, std::array{0});
    }

    GlobalVariableOp initImageMetadataHeap() {
        // { int Dims : 8 }
        mlir::Type entryType = StructType::get({int8Ty}, {0});
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::ImageMetadataHeap, heapTy, "ImageMetadataHeap");
    }

    GlobalVariableOp initTextureMetadataHeap() {
        // { int Dims : 8 }
        mlir::Type entryType = StructType::get({int8Ty}, {0});
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::TextureMetadataHeap, heapTy, "TextureMetadataHeap");
    }

    // TODO:
    // mlir dialect doesn't support separate sampler and images, only combined samplers
    GlobalVariableOp initCombinedSamplerHeap() {
        // %28 = OpTypeImage %float 2D 2 0 0 0 Unknown
        // %float: sampled type, 2D : dim, 2 : depth, 0: arrayed, 0 : MS (multi-sampled), 0 (not statically known) : sampled, Unknown : Image format
        ImageType imageTy = ImageType::get(floatTy, Dim::Dim2D);
        mlir::Type entryType = SampledImageType::get(imageTy);
        mlir::Type heapTy = RuntimeArrayType::get(entryType);
        return initDescriptorHeap(DescriptorSetKind::CombinedSamplerHeap, heapTy, "CombinedSamplerHeap");
    }

    // sgprBase: regno for an sgpr holding an index into the combined sampler heap.
    //
    // The Sgpr should match the first register of a 4 or 8 sgpr group which would normally hold a T#
    // resource constant in a real gcn shader.
    // In the emulated shaders, we will pass around indexes into descriptor heaps, that way for any gcn code which
    // moves a T# from one block of registers into another group, the emulated shader will move the index that points
    // to the image/sampler descriptor into the virtual sgpr (Spirv OpVariable)
    //
    // This function returns the Spirv OpSampledImage from the combined sampler heap.
    mlir::Value lookupCombinedSamplerDescriptor(mlir::Value imageIdx) {
        GlobalVariableOp combinedSamplerHeap = initCombinedSamplerHeap();
        auto combinedSamplerHeapRef = builder.create<AddressOfOp>(combinedSamplerHeap);
        auto heapChain = builder.create<AccessChainOp>(combinedSamplerHeapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupImageDescriptor1D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initImage1DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupImageDescriptor2D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initImage2DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupImageDescriptor3D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initImage3DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupTextureDescriptor1D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initTexture1DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupTextureDescriptor2D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initTexture2DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupTextureDescriptor3D(mlir::Value imageIdx) {
        GlobalVariableOp heap = initTexture3DHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupImageMetaData(mlir::Value imageIdx) {
        GlobalVariableOp imageMetadataHeap = initImageMetadataHeap();
        auto imageMetadataHeapRef = builder.create<AddressOfOp>(imageMetadataHeap);
        auto heapChain = builder.create<AccessChainOp>(imageMetadataHeapRef, mlir::ValueRange({ imageIdx }));
        // record
        return builder.create<LoadOp>(heapChain);
    }

    mlir::Value lookupTextureMetaData(mlir::Value imageIdx) {
        GlobalVariableOp heap = initTextureMetadataHeap();
        auto heapRef = builder.create<AddressOfOp>(heap);
        auto heapChain = builder.create<AccessChainOp>(heapRef, mlir::ValueRange({ imageIdx }));
        // record
        return builder.create<LoadOp>(heapChain);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Factor out some common stuff, e.g. instructions that have a normal and VOP3a variant
    ///////////////////////////////////////////////////////////////////////////////////////////

    void handleVAddCU32(MCOperand &dst, MCOperand &src0, MCOperand &src1, uint carryInRegno, uint carryOutRegno) {
        mlir::Value carryIn, carryOut1, carryOut2;
        mlir::Value sumAndCarry1, sumAndCarry2;
        mlir::Value sum1, sum2;

        mlir::Value a = sourceVectorOperand(src0, intTy);
        mlir::Value b = sourceVectorOperand(src1, intTy);

        carryIn = sourceCCReg(carryInRegno);
        carryIn = builder.create<SelectOp>(carryIn, getZero(intTy), getOne(intTy));
        sumAndCarry1 = builder.create<IAddCarryOp>(a, b);
        sum1 = builder.create<CompositeExtractOp>(sumAndCarry1, std::array{0});
        carryOut1 = builder.create<CompositeExtractOp>(sumAndCarry1, std::array{1});
        carryOut1 = builder.create<INotEqualOp>(getZero(carryOut1.getType()), carryOut1);
        sumAndCarry2 = builder.create<IAddCarryOp>(sum1, carryIn);
        sum2 = builder.create<CompositeExtractOp>(sumAndCarry2, std::array{0});
        carryOut2 = builder.create<CompositeExtractOp>(sumAndCarry2, std::array{1});
        carryOut2 = builder.create<INotEqualOp>(getZero(carryOut2.getType()), carryOut2);

        auto store = storeResultCC(carryOutRegno, carryOut2);
        tagPredicated(store);
        storeVectorResult32(dst.getReg(), sum2);
    }

    void handleVCndMaskB32(MCOperand &dst, MCOperand &src0, MCOperand &src1, uint selRegno) {
        mlir::Value a = sourceVectorOperand(src0, intTy);
        mlir::Value b = sourceVectorOperand(src1, intTy);
        mlir::Value sel = sourceCCReg(selRegno);
        // VCC ? src1 : src0
        mlir::Value result = builder.create<SelectOp>(sel, b, a);
        storeVectorResult32(dst, result);
    }

    mlir::Value handleVMadF32(mlir::Value src0, mlir::Value src1, mlir::Value src2) {
        mlir::Value mad = builder.create<FMulOp>(src0, src1);
        return builder.create<FAddOp>(mad, src2);
    }

    mlir::Value handleVCvtPkrtzF16F32(mlir::Value src0, mlir::Value src1) {
        mlir::Value pack = builder.create<CompositeConstructOp>(v2float32Ty, std::array{src0, src1});
        // Needs llvm/mlir patch to add this opcode
        pack = builder.create<GLPackHalf2x16Op>(intTy, pack);
        // TODO: spirv dialect doesn't support this? Need to change llvm?
        // bF16->setAttr("FPRoundingMode", builder.getI32IntegerAttr(/* RTZ */ 1));
        return pack;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // GcnToSpirvConverter members
    ///////////////////////////////////////////////////////////////////////////////////////////

    mlir::MLIRContext mlirContext;
    mlir::ImplicitLocOpBuilder builder;
    mlir::ImplicitLocOpBuilder functionBuilder;
    ExecutionModel execModel;

    std::vector<MCInst> machineCode;
    SpirvBytecodeVector spirvModule;
    SpirvConvertResult::ConvertStatus status;

    llvm::DenseMap<uint, VectorReg> usedVgprs;
    // Includes Sgprs (CC), EXEC, VCC, etc
    llvm::DenseMap<uint, CCReg> usedCCRegs;
    // Includes Sgprs (uniforms - 32 bits), EXEC_LO, EXEC_HI, m0, VCC_LO, etc
    llvm::DenseMap<uint, UniformReg> usedUniformRegs;

    // Maps attribute number (GCN) to GlobalVariableOp for 'in' attributes
    llvm::DenseMap<uint, InputAttribute> inputAttributes;
    // Maps export number (GCN) to GlobalVariableOp for 'out' attributes.
    // Keys are from the Exp enum, which includes MRT render targets 0-7, MRTZ (assuming frag-only)
    // vertex position, and user-defined attributes from 32-63 (PARAM0-PARAM31) (assuming vert-only)
    llvm::DenseMap<uint, ExportTarget> exportTargets;

    // Used to compact the render targets MRT0...MRT7, MRTZ, etc to locations 0..N
    llvm::DenseMap<uint, uint> fragExportTargetToVkLocation;

    SrdMode srdMode;
    std::vector<InputUsageSlot> srdSlots;
    // Assume there is one binding per descriptor set - either a UBO or an array of descriptors of the same type.
    // Either way, an OpVariable of struct or array type
    llvm::DenseMap<uint /*DescriptorSetKind*/, GlobalVariableOp> descriptorSets;
    
    FuncOp mainFn;
    mlir::Block *mainEntryBlock;
    ModuleOp moduleOp;

    // spirv-val complains about duplicate int types, so mlir must be caching or handling signedness weird.
    // init once and reuse these instead
    mlir::FloatType floatTy;
    mlir::FloatType f64Ty;
    mlir::FloatType float16Ty;
    mlir::VectorType v2float32Ty;
    mlir::VectorType v2float16Ty;
    
    mlir::IntegerType intTy;
    mlir::IntegerType sintTy;
    mlir::IntegerType sint64Ty;
    mlir::IntegerType int64Ty;
    mlir::IntegerType int16Ty;
    mlir::IntegerType int8Ty;
    mlir::IntegerType boolTy;
    mlir::VectorType v2intTy;
    mlir::VectorType v3intTy;

    // struct has 2 u32 members.
    // This is used for a lot of carry procducing int ops, i.e. V_ADD_I32 -> OpIAddCarry
    StructType arithCarryTy;

    PointerType ssboBlockDwordPtrType;

    const MCSubtargetInfo *STI;
    MCInstPrinter *IP;
};

bool GcnToSpirvConverter::processSrds() {
    // Future: These could use push constants
    if (isImmediateSrdMode()) {
        // SI programming guide:
        // USER_DATA. Up to 16 SGPRs can be loaded with data from the
        // SPI_SHADER_USER_DATA_xx_0-15 hardware registers written by the driver with PM4 SET
        // commands. User data is used for communicating several details to the shader, most importantly
        // the memory address of resource constants in video memory.
        
        //std::vector<>
        // For immediate resources (buffers, images, samplers) create members in a UBO (DS 0, binding 0) for indexes into
        // descriptor heap arrays.
        // These will be created by driver and passed via UBO
        std::vector<std::string> names;
        std::vector<mlir::Type> types;
        std::vector<StructType::OffsetInfo> offsets;
        uint curOff = 0;
        for (uint i = 0; i < srdSlots.size(); i++) {
            const InputUsageSlot &slot = srdSlots[i];
            switch (slot.m_usageType) {
                case kShaderInputUsageImmResource:
                case kShaderInputUsageImmRwResource:
                    if (slot.m_resourceType == IMM_RESOURCE_BUFFER) {
                        names.emplace_back(std::string("bufferImmIdx") + std::to_string(i));
                    } else { // IMM_RESOURCE_IMAGE
                        names.emplace_back(std::string("imageImmIdx") + std::to_string(i));
                    }
                    types.push_back(intTy);
                    offsets.push_back(curOff);
                    curOff += 4;
                    break;
                case kShaderInputUsageImmSampler:
                    // Ignore for now until figure out separate samplers
                    break;
                case kShaderInputUsageImmConstBuffer:
                    // Consider making another descriptor heap for UBOs
                    names.emplace_back(std::string("constBufferImmIdx") + std::to_string(i));
                    types.push_back(intTy);
                    offsets.push_back(curOff);
                    curOff += 4;
                    break;
                default:
                    break;
            }
        }
        if (!names.empty()) {
            StructType immDataUBOBlockTy;
#if defined(_DEBUG)
            std::vector<StructType::MemberDecorationInfo> decs;
            for (uint i = 0; i < names.size(); i++) {
                // TODO names for members
#endif
            }
            immDataUBOBlockTy = StructType::get(types, offsets);
            GlobalVariableOp immDataUBO = initImmediateDataUBO(immDataUBOBlockTy);
            auto immDataUBORef = builder.create<AddressOfOp>(immDataUBO);

            descriptorSets[(uint) DescriptorSetKind::SrdImmediates] = immDataUBO;

            // Insert at prologue.
            // Load descriptor heap indexes out of the immediate data UBO and into scalar gprs based on
            // slot->m_startRegister
            uint immDataMemberNo = 0;
            for (uint i = 0; i < srdSlots.size(); i++) {
                const InputUsageSlot &slot = srdSlots[i];
                switch (slot.m_usageType) {
                    case kShaderInputUsageImmResource:
                    case kShaderInputUsageImmRwResource:
                    case kShaderInputUsageImmConstBuffer:
                    {
                        auto chain = builder.create<AccessChainOp>(immDataUBORef, mlir::ValueRange({ getI32Const(immDataMemberNo) }));
                        mlir::Value resourceIdx = builder.create<LoadOp>(chain);
                        storeScalarResult32(AMDGPU::SGPR0 + slot.m_startRegister, resourceIdx);
                        ++immDataMemberNo;
                        break;
                    }
                    case kShaderInputUsageImmSampler:
                        // Ignore for now until figure out separate samplers
                        break;
                    default:
                        break;
                }
            }
        }
    } else {
        //assert(false && "Unhandled flat table mode");
        // TODO
    }
    
    return true;
}

bool GcnToSpirvConverter::prepass() {
    // TODO : may need to declare input attributes with specific type
    // so interpolation happens correctly in frag stage. Also need to match previous
    // stage's output types with frag stage's input types
    struct InputAttributeInfos {
        mlir::Type eltTy;
        uint maxComponent;
    };

    llvm::DenseMap<uint, InputAttributeInfos> inputAttrInfos;

    struct ExportInfo {
        uint componentWidth;
        uint validMask : 4;
    };

    llvm::DenseMap<uint, ExportInfo> exportInfos;

    // Do some things:
    // Figure out how many components each in/out attribute has (1 - 4)

    // Use to compact render targets (frag shader outputs)
    llvm::BitVector usedExportTargets(AMDGPU::Exp::Target::ET_PARAM31 + 1);

    bool success = true;
    for (MCInst &MI: machineCode) {
        switch (MI.getOpcode()) {
            case AMDGPU::V_INTERP_P2_F32_si:
            {
                assert(isFragShader());
                if ( !isFragShader()) {
                    return false;
                }
                uint attrno, component;
                attrno = MI.getOperand(3).getImm();
                component = MI.getOperand(4).getImm();
                auto it = inputAttrInfos.find(attrno);
                if (it != inputAttrInfos.end()) {
                    uint maxComponent = std::max(it->second.maxComponent, component);
                    it->getSecond().maxComponent = maxComponent;
                } else {
                    // assume 32 bit float type for now - TODO
                    inputAttrInfos[attrno] = { floatTy, component };
                }
                break;
            }

            case AMDGPU::EXP_DONE_si:
            case AMDGPU::EXP_si:
            {
                uint target, compr, validMask;
                target = MI.getOperand(0).getImm();
                compr = MI.getOperand(6).getImm();
                validMask = MI.getOperand(7).getImm();

                // TODO - when vm flag is 0, are all components written
                // I'm only guessing about what validMask is

                if (target != AMDGPU::Exp::Target::ET_NULL) {
                    if (isFragShader()) {
                        assert(isRenderTarget(target));
                        usedExportTargets.set(target);
                    }

                    // Remember validMask to later figure out how many components
                    // compr == 1 -> 16 bit components
                    // Assume either of these types for render targets for now
                    uint componentWidth = compr ? 16 : 32;
                    auto it = exportInfos.find(target);
                    if (it != exportInfos.end()) {
                        ExportInfo info = it->getSecond();
                        assert(info.componentWidth == componentWidth);
                        info.validMask |= validMask;
                    } else {
                        exportInfos[target] = { componentWidth, validMask };
                    }
                }
                break;
            }
        }
    }

    // Initialize input attributes
    for (const auto &kv: inputAttrInfos) {
        uint attrno = kv.first;
        uint numComponents = kv.second.maxComponent + 1;
        mlir::Type attrType;
        if (numComponents > 1) {
            attrType = mlir::VectorType::get({numComponents}, floatTy);
        } else {
            attrType = floatTy;
        }

        initInputAttribute(attrno, attrType);
    }

    // Compact render targets to output locations 0...N
    if (isFragShader()) {
        uint location = 0;
        for (uint target = 0; target < usedExportTargets.size(); target++) {
            if (usedExportTargets.test(target)) {
                fragExportTargetToVkLocation[target] = location;
                ++location;
            }
        }
    }

    // Assign types, initialize variables for exports
    for (const auto &[target, expInfo]: exportInfos) {
        uint width = expInfo.componentWidth;
        assert(width == 16 || width == 32);
        mlir::Type eltTy = width == 16 ? float16Ty : floatTy;
        uint numComponents;
        // Basically find the index of the MSB of validMask
        if (expInfo.validMask & 8) {
            numComponents = 4;
        } else if (expInfo.validMask & 4) {
            numComponents = 3;
        } else if (expInfo.validMask & 2) {
            numComponents = 2;
        } else if (expInfo.validMask & 1) {
            numComponents = 1;
        } else {
            // is never written, we should skip
            continue;
        }

        mlir::Type expTy;
        if (numComponents > 1) {
            expTy = mlir::VectorType::get({numComponents}, eltTy);
        } else {
            expTy = eltTy;
        }
        initExportTarget(target, expTy);
    }

    return success;
}

bool GcnToSpirvConverter::convertGcnOp(const MCInst &MI) {
    unsigned int opcode = MI.getOpcode();
    switch (opcode) {
        // SOP1
        case AMDGPU::S_MOV_B32_gfx6_gfx7:
        {
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            assert(dst.isReg());
            mlir::Value result = sourceScalarOperandUniform(src, intTy);
            // TODO try to move CC bits too somehow.
            storeScalarResult32(dst, result);
            break;
        }
        case AMDGPU::S_MOV_B64_gfx6_gfx7:
        {
            // Try to move uniform register and cc register values
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            assert(dst.isReg());
            mlir::Value uniValLo, uniValHi;
            mlir::Value ccVal;

            if (src.isImm()) {
                ConstantOp immConst = sourceImmediate(src, int64Ty);
                auto constval0  = getZero(intTy);
                auto constval32 = getI32Const(32);

                // val, offset, count
                uniValLo = builder.create<BitFieldUExtractOp>(immConst, constval0, constval32);                
                uniValLo = builder.create<BitFieldUExtractOp>(immConst, constval32, constval32);

                // Don't move cc if the src is immediate.
                // TODO - support this in compute
            } else {
                assert(src.isReg());
                uint srcno = src.getReg();
                uint loSrcNo;
                uint hiSrcNo;

                if (regnoToType(srcno) == GcnRegType::Special) {
                    switch (srcno) {
                        case AMDGPU::VCC:
                            loSrcNo = AMDGPU::VCC_LO;
                            hiSrcNo = AMDGPU::VCC_HI;
                            break;
                        case AMDGPU::EXEC:
                            loSrcNo = AMDGPU::EXEC_LO;
                            hiSrcNo = AMDGPU::EXEC_HI;
                            break;
                        default:
                            loSrcNo = -1;
                            hiSrcNo = -1;
                            break;
                    }
                } else {
                    assert(isSgprPair(srcno));
                    loSrcNo = sgprPairToSgprBase(srcno);
                    hiSrcNo = loSrcNo + 1;
                }
                uniValLo = sourceUniformReg(loSrcNo);
                uniValHi = sourceUniformReg(hiSrcNo);

                // Move cc
                assert(isSgprPair(srcno) || isSpecialRegPair(srcno));
                ccVal = sourceCCReg(srcno);
                storeResultCC(dst, ccVal);
            }
            // Move uniform register pair
            assert(dst.isReg());
            uint dstno = dst.getReg();
            uint loDstNo;
            uint hiDstNo;
            if (regnoToType(dstno) == GcnRegType::Special) {
                switch (dstno) {
                    case AMDGPU::VCC:
                        loDstNo = AMDGPU::VCC_LO;
                        hiDstNo = AMDGPU::VCC_HI;
                        break;
                    case AMDGPU::EXEC:
                        loDstNo = AMDGPU::EXEC_LO;
                        hiDstNo = AMDGPU::EXEC_HI;
                        break;
                    default:
                        loDstNo = -1;
                        hiDstNo = -1;
                        break;
                }
            } else {
                assert(isSgprPair(dstno));
                loDstNo = sgprPairToSgprBase(dstno);
                hiDstNo = loDstNo + 1;
            }
            storeScalarResult32(loDstNo, uniValLo);
            storeScalarResult32(hiDstNo, uniValHi);
            break;
        }

        case AMDGPU::S_WQM_B64_gfx6_gfx7:
        {
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            // What I've seen so far
            // Probably just implementation detail for pixel shaders and can ignore.
            // Probably gets helper quad invocations running on AMD hardware
            // for derivatives and stuff
            assert(isFragShader());
            assert(dst.isReg() && dst.getReg() == AMDGPU::EXEC);
            assert(src.isReg() && src.getReg() == AMDGPU::EXEC);
            break;
        }

        case AMDGPU::S_SWAPPC_B64_gfx6_gfx7:
            assert(isVertexShader());
            break;

        // SOP2
        case AMDGPU::S_MUL_I32_gfx6_gfx7:
        {
            // D = S1 * S2. Low 32 bits of result.
            auto [ dst, ssrc0, ssrc1 ] = unwrapMI(MI, R0_2);
            mlir::Value a, b;

            a = sourceScalarOperandUniform(ssrc0, intTy);
            b = sourceScalarOperandUniform(ssrc1, intTy);
            auto result = builder.create<IMulOp>(a, b);
            storeScalarResult32(dst.getReg(), result);
            break;
        }

        case AMDGPU::S_AND_B32_gfx6_gfx7:
        case AMDGPU::S_ANDN2_B32_gfx6_gfx7:
        case AMDGPU::S_OR_B32_gfx6_gfx7:
        case AMDGPU::S_ORN2_B32_gfx6_gfx7:
        case AMDGPU::S_XOR_B32_gfx6_gfx7:
        {
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2);
            mlir::Value a, b;

            a = sourceScalarOperandUniform(src0, intTy);
            b = sourceScalarOperandUniform(src1, intTy);

            switch (opcode) {
                case AMDGPU::S_ANDN2_B64_gfx6_gfx7:
                case AMDGPU::S_ORN2_B64_gfx6_gfx7:
                    b = builder.create<NotOp>(b);
                    break;
                default:
                    break;
            }

            mlir::Value result;
            switch (opcode) {
                case AMDGPU::S_AND_B32_gfx6_gfx7:
                    result = builder.create<BitwiseAndOp>(a, b);
                    break;
                case AMDGPU::S_OR_B32_gfx6_gfx7:
                    result = builder.create<BitwiseOrOp>(a, b);
                    break;
                case AMDGPU::S_XOR_B32_gfx6_gfx7:
                    result = builder.create<BitwiseXorOp>(a, b);
                    break;
            }
            auto isNonZero = builder.create<INotEqualOp>(a, getZero(result.getType()));
            storeResultCC(AMDGPU::SCC, isNonZero);
            storeScalarResult32(dst, result);
            break;
        }

        case AMDGPU::S_AND_B64_gfx6_gfx7:
        case AMDGPU::S_ANDN2_B64_gfx6_gfx7:
        case AMDGPU::S_OR_B64_gfx6_gfx7:
        case AMDGPU::S_ORN2_B64_gfx6_gfx7:
        case AMDGPU::S_XOR_B64_gfx6_gfx7:
        {
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2);
            mlir::Value aUni, bUni, aCC, bCC;

            aUni = sourceScalarOperand64(src0, int64Ty);
            bUni = sourceScalarOperand64(src1, int64Ty);

            bool applyToCC = false;
            if ((src0.isReg() || src0.getImm() == 0 || src0.getImm() == ~0LL) &&
                    (src1.isReg() || src1.getImm() == 0 || src1.getImm() == ~0LL))
            {
                applyToCC = true;
                // If we can, do the operation on the separate condition code register (1 bool per thread).
                // We can only do this if, for both sources, the source is a register or the source is all
                // ones or all zeroes
                auto getCcArgument = [&](MCOperand &source) -> mlir::Value {
                    if (source.isReg()) {
                        return sourceCCOperand(src0);
                    } else {
                        int64_t imm = source.getImm();
                        assert(imm == 0 || imm == ~0LL);
                        if (imm == 0) {
                            return getZero(boolTy);
                        } else {
                            return getOne(boolTy);
                        }
                    }
                };
                aCC = getCcArgument(src0);
                bCC = getCcArgument(src1);
            }

            switch (opcode) {
                case AMDGPU::S_ANDN2_B64_gfx6_gfx7:
                case AMDGPU::S_ORN2_B64_gfx6_gfx7:
                    bUni = builder.create<NotOp>(bUni);
                    if (applyToCC) {
                        bCC = builder.create<LogicalNotOp>(bCC);
                    }
                    break;
                default:
                    break;
            }

            mlir::Value uniResult;
            mlir::Value ccResult;
            switch (opcode) {
                case AMDGPU::S_AND_B64_gfx6_gfx7:
                case AMDGPU::S_ANDN2_B64_gfx6_gfx7:
                    uniResult = builder.create<BitwiseAndOp>(aUni, bUni);
                    if (applyToCC) {
                        ccResult = builder.create<LogicalAndOp>(aCC, bCC);
                    }
                    break;
                case AMDGPU::S_OR_B64_gfx6_gfx7:
                case AMDGPU::S_ORN2_B64_gfx6_gfx7:
                    uniResult = builder.create<BitwiseOrOp>(aUni, bUni);
                    if (applyToCC) {
                        ccResult = builder.create<LogicalOrOp>(aCC, bCC);
                    }
                    break;
                case AMDGPU::S_XOR_B64_gfx6_gfx7:
                    uniResult = builder.create<BitwiseXorOp>(aUni, bUni);
                    if (applyToCC) {
                        ccResult = builder.create<LogicalNotEqualOp>(aCC, bCC);
                    }
                    break;
            }
            auto isNonZero = builder.create<INotEqualOp>(uniResult, getZero(uniResult.getType()));
            storeResultCC(AMDGPU::SCC, isNonZero);
            storeScalarResult64(dst, uniResult);

            if (applyToCC) {
                assert(ccResult);
                storeResultCC(dst, ccResult);
            }
            break;
        }

        // SOPP
        case AMDGPU::S_WAITCNT_gfx6_gfx7:
            // Used by AMD hardware to wait for long-latency instructions.
            // Should be able to ignore
            break;
        case AMDGPU::S_ENDPGM_gfx6_gfx7:
            // Do return here?
            // TEMP
            initSsboHeap();
            initCombinedSamplerHeap();
            break;

        // SMRD
        case AMDGPU::S_LOAD_DWORD_IMM_si:
        case AMDGPU::S_LOAD_DWORDX2_IMM_si:
        case AMDGPU::S_LOAD_DWORDX4_IMM_si:
        case AMDGPU::S_LOAD_DWORDX8_IMM_si:
        case AMDGPU::S_LOAD_DWORDX16_IMM_si:
        case AMDGPU::S_BUFFER_LOAD_DWORD_IMM_si:
        case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM_si:
        case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM_si:
        case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM_si:
        case AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM_si:
        {
            auto [ dst, buffer, offset, _] = unwrapMI(MI, R0_3);
            assert(_.getImm() == 0);

            uint numDwords;
            switch (opcode) {
                case AMDGPU::S_LOAD_DWORD_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORD_IMM_si:
                    numDwords = 1;
                    break;
                case AMDGPU::S_LOAD_DWORDX2_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM_si:
                    numDwords = 2;
                    break;
                case AMDGPU::S_LOAD_DWORDX4_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM_si:
                    numDwords = 4;
                    break;
                case AMDGPU::S_LOAD_DWORDX8_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM_si:
                    numDwords = 8;
                    break;
                case AMDGPU::S_LOAD_DWORDX16_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM_si:
                    numDwords = 16;
                    break;
                default:
                    assert(false);
                    break;
            }

            bool useStride = false;
            switch (opcode) {
                case AMDGPU::S_BUFFER_LOAD_DWORD_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM_si:
                case AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM_si:
                    useStride = true;
                    break;
                default:
                    break;
            }
            // Have seen S_LOAD* used with kShaderInputUsagePtrExtendedUserData resources.
            // These should be created by the API similarly to buffers. Treat them the same (index to ssbo heap)

            mlir::Value stride, numRecords;

            uint dstRegno = sgprGroupToSgprBase(dst.getReg());
            uint bufferConstRegno = sgprGroupToSgprBase(buffer.getReg());
            // we stash idx into ssbo descriptor heap in the base register
            mlir::Value ssboIdx = sourceUniformReg(bufferConstRegno);
            GlobalVariableOp ssboHeap = initSsboHeap();
            auto ssboHeapRef = builder.create<AddressOfOp>(ssboHeap);
            auto heapChain = builder.create<AccessChainOp>(ssboHeapRef, mlir::ValueRange({ ssboIdx }));
            mlir::Value ssboDescriptor = builder.create<LoadOp>(heapChain);
            mlir::Value ssboAddr = builder.create<CompositeExtractOp>(ssboDescriptor, std::array{0});
            // TODO clamp based on stride
            if (useStride) {
                //mlir::Value ssboStride = builder.create<CompositeExtractOp>(ssboDescriptor, std::array{1});
            }
            mlir::Value ssboBasePtr = builder.create<BitcastOp>(ssboBlockDwordPtrType, ssboAddr);
            mlir::Value offsetVal = getI32Const(offset.getImm());
            // For immediate offset, offset is given in dwords, not bytes
            for (uint i = 0; i < numDwords; i++) {
                auto src = builder.create<AccessChainOp>(ssboBasePtr, mlir::ValueRange({offsetVal}));
                auto dword = builder.create<LoadOp>(src, MemoryAccessAttr::get(&mlirContext, MemoryAccess::Aligned), builder.getI32IntegerAttr(4));
                storeScalarResult32(dstRegno + i, dword);
                if (i < numDwords - 1) {
                    offsetVal = builder.create<IAddOp>(offsetVal, getOne(intTy));
                }
            }
            break;
        }

        // VOP1
        case AMDGPU::V_FLOOR_F32_e32_gfx6_gfx7:
        case AMDGPU::V_FRACT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_RCP_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MOV_B32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_U32_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_I32_F32_e32_gfx6_gfx7:

        case AMDGPU::V_EXP_F32_e32_gfx6_gfx7:
        {
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            mlir::Value srcVal = sourceVectorOperand(src, floatTy);
            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_FLOOR_F32_e32_gfx6_gfx7:
                    result = builder.create<GLFloorOp>(srcVal);
                    break;
                case AMDGPU::V_FRACT_F32_e32_gfx6_gfx7:
                {
                    // V_FRACT_F32: D.f = S0.f - FLOOR(S0.f).
                    auto floor = builder.create<GLFloorOp>(srcVal);
                    result = builder.create<FSubOp>(srcVal, floor);
                    break;
                }
                case AMDGPU::V_RCP_F32_e32_gfx6_gfx7:
                    result = builder.create<FDivOp>(getOne(floatTy), srcVal);
                    break;
                case AMDGPU::V_MOV_B32_e32_gfx6_gfx7:
                    result = srcVal;
                    break;
                case AMDGPU::V_CVT_U32_F32_e32_gfx6_gfx7:
                    // Input is converted to an unsigned integer value using truncation. Positive float magnitudes
                    // too great to be represented by an unsigned integer float (unbiased exponent > 31) saturate
                    // to max_uint.
                    // Special number handling:
                    // -inf & NaN & 0 & -0 → 0
                    // Inf → max_uint
                    // D.u = (unsigned)S0.f

                    // TODO special value handling
                    result = builder.create<ConvertFToUOp>(intTy, srcVal);
                    break;
                case AMDGPU::V_CVT_I32_F32_e32_gfx6_gfx7:
                    result = builder.create<ConvertFToSOp>(intTy, srcVal);
                    break;
                case AMDGPU::V_EXP_F32_e32_gfx6_gfx7:
                    result = builder.create<GLExp2Op>(srcVal);
                    break;
            }
            storeVectorResult32(dst, result);
            break;
        }
        case AMDGPU::V_CVT_OFF_F32_I4_e32_gfx6_gfx7:
        {
            // Map 4 bit int to range of floats (increment by 0.0625)
            // 1000 -0.5f
            // 1001 -0.4375f
            // 1010 -0.375f
            // 1011 -0.3125f
            // 1100 -0.25f
            // 1101 -0.1875f
            // 1110 -0.125f
            // 1111 -0.0625f
            // 0000 0.0f
            // 0001 0.0625f
            // 0010 0.125f
            // 0011 0.1875f
            // 0100 0.25f
            // 0101 0.3125f
            // 0110 0.375f
            // 0111 0.4375f
            
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            mlir::Value srcVal = sourceVectorOperand(src, intTy);
            mlir::Value result;
            // For now, just do multiplication. 0.0625 * float(sext(src[3:0]))
            // Maybe put all constants + switch stmt in shader later
            auto scale = getF32Const(0.0625f);
            auto bfOff = getZero(intTy);
            auto bfCount = getI32Const(4);
            auto srcSExt = builder.create<BitFieldSExtractOp>(srcVal, bfOff, bfCount);
            result = builder.create<ConvertSToFOp>(floatTy, srcSExt);
            result = builder.create<FMulOp>(result, scale);
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_CVT_F32_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_F32_U32_e32_gfx6_gfx7:
        {
            auto [ dst, src ] = unwrapMI(MI, 0, 1);

            mlir::Value srcVal = sourceVectorOperand(src, intTy);
            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_CVT_F32_I32_e32_gfx6_gfx7:
                    result = builder.create<ConvertSToFOp>(floatTy, srcVal);
                    break;
                case AMDGPU::V_CVT_F32_U32_e32_gfx6_gfx7:
                    result = builder.create<ConvertUToFOp>(floatTy, srcVal);
                    break;
            }

            storeVectorResult32(dst, result);
            break;
        }

        // VOP2
        case AMDGPU::V_ADD_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MUL_F32_e32_gfx6_gfx7:
        case AMDGPU::V_SUB_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MIN_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MAX_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MAC_F32_e32_gfx6_gfx7:
        case AMDGPU::V_AND_B32_e32_gfx6_gfx7:
        case AMDGPU::V_OR_B32_e32_gfx6_gfx7:
        case AMDGPU::V_XOR_B32_e32_gfx6_gfx7:
        {
            auto [ dst, src0, vsrc1 ] = unwrapMI(MI, R0_2);

            mlir::Value a = sourceVectorOperand(src0, floatTy);
            mlir::Value b = sourceVectorOperand(vsrc1, floatTy);
            // overflow, side effects?
            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_ADD_F32_e32_gfx6_gfx7:
                    result = builder.create<FAddOp>(a, b);
                    break;
                case AMDGPU::V_MUL_F32_e32_gfx6_gfx7:
                    result = builder.create<FMulOp>(a, b);
                    break;
                case AMDGPU::V_SUB_F32_e32_gfx6_gfx7:
                    result = builder.create<FSubOp>(a, b);
                    break;
                case AMDGPU::V_MIN_F32_e32_gfx6_gfx7:
                    // TODO handle NANs
                    result = builder.create<GLFMinOp>(a, b);
                    break;
                case AMDGPU::V_MAX_F32_e32_gfx6_gfx7:
                    // TODO handle NANs
                    result = builder.create<GLFMaxOp>(a, b);
                    break;

                case AMDGPU::V_MAC_F32_e32_gfx6_gfx7:
                {
                    // Multiply a and b. Accumulate into dst
                    assert(dst.isReg() && dst.getReg() == MI.getOperand(3).getReg());
                    mlir::Value dstVal = sourceVectorOperand(dst, floatTy);
                    result = builder.create<FMulOp>(a, b);
                    result = builder.create<FAddOp>(dstVal, result);
                    break;
                }
                case AMDGPU::V_AND_B32_e32_gfx6_gfx7:
                    result = builder.create<BitwiseAndOp>(a, b);
                    break;
                case AMDGPU::V_OR_B32_e32_gfx6_gfx7:
                    result = builder.create<BitwiseOrOp>(a, b);
                    break;
                case AMDGPU::V_XOR_B32_e32_gfx6_gfx7:
                    result = builder.create<BitwiseXorOp>(a, b);
                    break;

            }
            storeVectorResult32(dst, result);            
            break;
        }

        case AMDGPU::V_CVT_PKRTZ_F16_F32_e32_gfx6_gfx7:
        {
            // decorate with FPRoundingMode?
            // OpFConvert?
            // OpQuantizeToF16?
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2) ;
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            auto pack = handleVCvtPkrtzF16F32(a, b);
            storeVectorResult32(dst, pack);
            break;
        }

        case AMDGPU::V_MADAK_F32_gfx6_gfx7:
        case AMDGPU::V_MADMK_F32_gfx6_gfx7:
        {
            // MADAK: addend is constant
            // MADMK: one of multipliers is constant
            // makes no diff for this
            auto [ dst, src0, src1, src2 ] = unwrapMI(MI, R0_3);
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            mlir::Value c = sourceVectorOperand(src2, floatTy);
            auto mad = handleVMadF32(a, b, c);
            storeVectorResult32(dst, mad);
            break;
        }

        case AMDGPU::V_ADD_I32_e32_gfx6_gfx7:
        case AMDGPU::V_SUB_I32_e32_gfx6_gfx7:
        {
            auto [ dst, src0, vsrc1 ] = unwrapMI(MI, R0_2);

            //MCOperand src
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(vsrc1, intTy);
            // overflow, side effects?
            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_ADD_I32_e32_gfx6_gfx7:
                {
                    // "Unsigned integer add based on signed or unsigned integer components. Produces an
                    // unsigned carry out in VCC or a scalar register" <- VOP2 is always VCC
                    mlir::Value carryOut;
                    auto sumAndCarry = builder.create<IAddCarryOp>(a, b);
                    result = builder.create<CompositeExtractOp>(sumAndCarry, std::array{0});
                    carryOut = builder.create<CompositeExtractOp>(sumAndCarry, std::array{1});
                    // Can't cast i32 -> i1 with OpUConvert because mlir forces i1 -> OpTypeBool
                    carryOut = builder.create<INotEqualOp>(getZero(carryOut.getType()), carryOut);
                    storeVectorResult32(dst, result);
                    auto store = storeResultCC(AMDGPU::VCC, carryOut);
                    tagPredicated(store);
                    break;
                }
                case AMDGPU::V_SUB_I32_e32_gfx6_gfx7:
                {
                    mlir::Value borrowOut;
                    auto subAndBorrow = builder.create<ISubBorrowOp>(a, b);
                    result = builder.create<CompositeExtractOp>(subAndBorrow, std::array{0});
                    borrowOut = builder.create<CompositeExtractOp>(subAndBorrow, std::array{1});
                    // Can't cast i32 -> i1 with OpUConvert because mlir forces i1 -> OpTypeBool
                    borrowOut = builder.create<INotEqualOp>(getZero(borrowOut.getType()), borrowOut);
                    storeVectorResult32(dst, result);
                    auto store = storeResultCC(AMDGPU::VCC, borrowOut);
                    tagPredicated(store);
                    break;  
                }


            }
            storeVectorResult32(dst, result);            
            break;
        }

        case AMDGPU::V_MUL_HI_U32_gfx6_gfx7:
        case AMDGPU::V_MUL_LO_U32_gfx6_gfx7:
        case AMDGPU::V_MUL_HI_I32_gfx6_gfx7:
        case AMDGPU::V_MUL_LO_I32_gfx6_gfx7:
        {
            // TODO check NEG, OMOD, CLAMP, ABS
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2);
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(src1, intTy);
            mlir::Value product;
            mlir::Value result;

            switch(opcode) {
                case AMDGPU::V_MUL_HI_U32_gfx6_gfx7:
                case AMDGPU::V_MUL_LO_U32_gfx6_gfx7:
                    product = builder.create<UMulExtendedOp>(a, b);
                    break;
                case AMDGPU::V_MUL_HI_I32_gfx6_gfx7:
                case AMDGPU::V_MUL_LO_I32_gfx6_gfx7:
                    product = builder.create<SMulExtendedOp>(a, b);
                    break;
            }

            switch(opcode) {
                case AMDGPU::V_MUL_HI_U32_gfx6_gfx7:
                case AMDGPU::V_MUL_HI_I32_gfx6_gfx7:
                    result = builder.create<CompositeExtractOp>(product, std::array{1});
                    break;
                case AMDGPU::V_MUL_LO_U32_gfx6_gfx7:
                case AMDGPU::V_MUL_LO_I32_gfx6_gfx7:
                    result = builder.create<CompositeExtractOp>(product, std::array{0});
                    break;
            }
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_ADDC_U32_e32_gfx6_gfx7:
        {
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2);
            handleVAddCU32(dst, src0, src1, AMDGPU::VCC, AMDGPU::VCC);
            break;
        }

        case AMDGPU::V_CNDMASK_B32_e32_gfx6_gfx7:
        {
            auto [ dst, src0, src1 ] = unwrapMI(MI, R0_2);
            handleVCndMaskB32(dst, src0, src1, AMDGPU::VCC);
            break;
        }

        // VCMP
        case AMDGPU::V_CMP_F_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_O_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_U_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NGE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NGT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NEQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_TRU_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_O_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_U_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NEQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_TRU_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_O_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_U_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NGE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NGT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NEQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NLT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_TRU_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_O_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_U_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NEQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_TRU_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_F_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_EQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_GT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_GE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_O_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_U_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NEQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_TRU_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_F_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_EQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_O_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_U_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLG_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLE_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NEQ_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_TRU_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_F_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_EQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_GT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_LG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_GE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_O_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_U_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NEQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPS_TRU_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_F_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_EQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_O_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_U_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLG_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLE_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NEQ_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLT_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPSX_TRU_F64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_I32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_I64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_U32_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_U64_e32_gfx6_gfx7:
        case AMDGPU::V_CMP_F_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_O_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_U_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NGE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NGT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NEQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_TRU_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_O_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_U_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NEQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_TRU_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_F_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_O_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_U_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NGE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NGT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NEQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NLT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_TRU_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_O_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_U_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NGT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NEQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NLT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_TRU_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_F_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_EQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_GT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_GE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_O_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_U_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NEQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_TRU_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_F_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_EQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_O_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_U_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLG_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLE_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NEQ_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLT_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_TRU_F32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_F_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_EQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_GT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_LG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_GE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_O_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_U_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NGT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NEQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_NLT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPS_TRU_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_F_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_EQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_LG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_GE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_O_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_U_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLG_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NGT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLE_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NEQ_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_NLT_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPSX_TRU_F64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_F_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_I32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_F_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_I64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_F_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_U32_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_F_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LT_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_EQ_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_NE_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_LE_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GT_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMP_GE_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_F_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LT_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_EQ_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_NE_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_LE_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GT_U64_e64_gfx6_gfx7:
        case AMDGPU::V_CMPX_GE_U64_e64_gfx6_gfx7:
        {
            // Ignore NAN signalling - maybe TODO
            GcnCmp::Op cmpOp = compareOpcodeToOperation(opcode);
            bool writeExec = compareWritesExec(opcode);
            bool isVop3 = compareIsVop3(opcode);
            GcnCmp::Type cmpOperandType = compareOpcodeToOperandType(opcode);

            bool isFloat = false;
            bool isSigned = false;
            bool is64Bit = false;
            mlir::Type argTy;
            switch (cmpOperandType) {
                case GcnCmp::F32:
                    isFloat = true;
                    argTy = floatTy;
                    break;
                case GcnCmp::F64:
                    isFloat = true;
                    is64Bit = true;
                    argTy = f64Ty;
                    break;
                case GcnCmp::I32:
                    isSigned = true;
                    LLVM_FALLTHROUGH;
                case GcnCmp::U32:
                    argTy = intTy;
                    break;
                case GcnCmp::I64:
                    isSigned = true;
                    LLVM_FALLTHROUGH;
                case GcnCmp::U64:
                    is64Bit = true;
                    argTy = int64Ty;
                    break;
            }

            MCOperand src0, src1, src0Mods, src1Mods;
            uint dst = AMDGPU::VCC;
            mlir::Value a, b, ccOut;

            if (isVop3) {
                dst = MI.getOperand(0).getReg();
                if (isFloat) {
                    src0Mods = MI.getOperand(1);
                    src0 = MI.getOperand(2);
                    src1Mods = MI.getOperand(3);
                    src1 = MI.getOperand(4);
                } else {
                    src0 = MI.getOperand(1);
                    src1 = MI.getOperand(2);
                }
            } else {
                src0 = MI.getOperand(0);
                src1 = MI.getOperand(1);
            }

            if (is64Bit) {
                // TODO should probably just handle 64 bit vector operands in sourceVectorOperand() itself
                a = sourceVectorOperand64(src0, argTy);
                b = sourceVectorOperand64(src1, argTy);           
            } else {
                a = sourceVectorOperand(src0, argTy);
                b = sourceVectorOperand(src1, argTy);
            }

            if (isVop3 && isFloat) {
                a = applyOperandModifiers(a, src0Mods);
                b = applyOperandModifiers(b, src1Mods);
            }

            //// ALL SPIRV OPS
            // OpIEqual
            // OpINotEqual
            // OpUGreaterThan
            // OpSGreaterThan
            // OpUGreaterThanEqual
            // OpSGreaterThanEqual
            // OpULessThan
            // OpSLessThan
            // OpULessThanEqual
            // OpSLessThanEqual

            // OpFOrdEqual
            // OpFUnordEqual
            // OpFOrdNotEqual
            // OpFUnordNotEqual
            // OpFOrdLessThan
            // OpFUnordLessThan
            // OpFOrdGreaterThan
            // OpFUnordGreaterThan
            // OpFOrdLessThanEqual
            // OpFUnordLessThanEqual
            // OpFOrdGreaterThanEqual
            // OpFUnordGreaterThanEqual

            bool invert;
            switch (cmpOp) {
                case GcnCmp::NGE:
                case GcnCmp::LG: // handle like NEQ
                case GcnCmp::NGT:
                case GcnCmp::NLE:
                case GcnCmp::NEQ:
                case GcnCmp::NLT:
                    invert = true;
                    break;
                default:
                    invert = false;
            }

            if (isFloat) {
                // Dont know about ordered/unordered here
                switch (cmpOp) {
                    case GcnCmp::F:
                        ccOut = getZero(boolTy);
                        break;
                    case GcnCmp::LT:
                    case GcnCmp::NLT:
                        ccOut = builder.create<FOrdLessThanOp>(a, b);
                        break;
                    case GcnCmp::EQ:
                    case GcnCmp::NEQ:
                    // Don't know what the point of these is
                    // Maybe they should tolerate small diffs? And (N)EQ is bitwise equality?
                    case GcnCmp::LG:
                    case GcnCmp::NLG:
                        ccOut = builder.create<FOrdEqualOp>(a, b);
                        break;
                    case GcnCmp::LE:
                    case GcnCmp::NLE:
                        ccOut = builder.create<FOrdLessThanEqualOp>(a, b);
                        break;
                    case GcnCmp::GT:
                    case GcnCmp::NGT:
                        ccOut = builder.create<FOrdGreaterThanOp>(a, b);
                        break;
                    case GcnCmp::GE:
                    case GcnCmp::NGE:
                        ccOut = builder.create<FOrdGreaterThanEqualOp>(a, b);
                        break;
                    case GcnCmp::O_F:
                        ccOut = builder.create<OrderedOp>(a, b);
                        break;
                    case GcnCmp::U_F:
                    {
                        ccOut = builder.create<OrderedOp>(a, a);
                        mlir::Value tmp = builder.create<OrderedOp>(b, b);
                        ccOut = builder.create<LogicalOrOp>(ccOut, tmp);
                        break;
                    }
                    case GcnCmp::TRU:
                        ccOut = getOne(boolTy);
                        break;
                }
            } else {
                assert(argTy.isa<mlir::IntegerType>());
                switch (cmpOp) {
                    case GcnCmp::F:
                        ccOut = getZero(boolTy);
                        break;
                    case GcnCmp::LT:
                        if (isSigned) {
                            ccOut = builder.create<SLessThanOp>(a, b);
                        } else {
                            ccOut = builder.create<ULessThanOp>(a, b);
                        }
                        break;
                    case GcnCmp::EQ:
                    case GcnCmp::NEQ:
                    case GcnCmp::LG:
                        ccOut = builder.create<IEqualOp>(a, b);
                        break;
                    case GcnCmp::LE:
                        if (isSigned) {
                            ccOut = builder.create<SLessThanEqualOp>(a, b);
                        } else {
                            ccOut = builder.create<ULessThanEqualOp>(a, b);
                        }
                        break;
                    case GcnCmp::GT:
                        if (isSigned) {
                            ccOut = builder.create<SGreaterThanOp>(a, b);
                        } else {
                            ccOut = builder.create<UGreaterThanOp>(a, b);
                        }
                        break;
                    case GcnCmp::GE:
                        if (isSigned) {
                            ccOut = builder.create<SGreaterThanEqualOp>(a, b);
                        } else {
                            ccOut = builder.create<UGreaterThanEqualOp>(a, b);                            
                        }
                        break;
                    case GcnCmp::TRU:
                        ccOut = getOne(boolTy);
                        break;
                    default:
                        assert(false && "unhandled");
                }
            }

            if (invert) {
                ccOut = builder.create<LogicalNotOp>(ccOut);
            }

            auto storeVCC = storeResultCC(dst, ccOut);
            tagPredicated(storeVCC);

            if (writeExec) {
                auto storeExec = storeResultCC(AMDGPU::EXEC, ccOut);
                tagPredicated(storeExec);
            }

            break;
        }

        case AMDGPU::S_CMP_EQ_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_LG_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_GT_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_GE_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_LE_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_LT_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_EQ_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LG_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_GT_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_GE_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LE_I32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LT_I32_gfx6_gfx7:
        case AMDGPU::S_CMP_EQ_U32_gfx6_gfx7:
        case AMDGPU::S_CMP_LG_U32_gfx6_gfx7:
        case AMDGPU::S_CMP_GT_U32_gfx6_gfx7:
        case AMDGPU::S_CMP_GE_U32_gfx6_gfx7:
        case AMDGPU::S_CMP_LE_U32_gfx6_gfx7:
        case AMDGPU::S_CMP_LT_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_EQ_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LG_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_GT_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_GE_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LE_U32_gfx6_gfx7:
        case AMDGPU::S_CMPK_LT_U32_gfx6_gfx7:
        {
            auto [ src0, src1 ] = unwrapMI(MI, 0, 1);
            mlir::Value a, b, ccOut;

            GcnCmp::Op cmpOp = compareOpcodeToOperation(opcode);
            GcnCmp::Type cmpOperandType = compareOpcodeToOperandType(opcode);

            assert(cmpOperandType == GcnCmp::I32 || cmpOperandType == GcnCmp::U32);
            bool isSigned = cmpOperandType == GcnCmp::I32;

            a = sourceScalarOperandUniform(src0, intTy);
            b = sourceScalarOperandUniform(src1, intTy);

            switch (cmpOp) {
                case GcnCmp::F:
                    ccOut = getZero(boolTy);
                    break;

                case GcnCmp::EQ:
                    ccOut = builder.create<IEqualOp>(a, b);
                    break;
                case GcnCmp::LG:
                    ccOut = builder.create<INotEqualOp>(a, b);
                    break;
                case GcnCmp::GT:
                    if (isSigned) {
                        ccOut = builder.create<SGreaterThanOp>(a, b);
                    } else {
                        ccOut = builder.create<UGreaterThanOp>(a, b);
                    }
                    break;
                case GcnCmp::GE:
                    if (isSigned) {
                        ccOut = builder.create<SGreaterThanEqualOp>(a, b);
                    } else {
                        ccOut = builder.create<UGreaterThanEqualOp>(a, b);                            
                    }
                    break;
                case GcnCmp::LE:
                    if (isSigned) {
                        ccOut = builder.create<SLessThanEqualOp>(a, b);
                    } else {
                        ccOut = builder.create<ULessThanEqualOp>(a, b);
                    }
                    break;
                case GcnCmp::LT:
                    if (isSigned) {
                        ccOut = builder.create<SLessThanOp>(a, b);
                    } else {
                        ccOut = builder.create<ULessThanOp>(a, b);
                    }
                    break;
                default:
                    assert(false && "unhandled");
            }

            storeResultCC(AMDGPU::SCC, ccOut);

            break;
        }

        // VOP3a

        case AMDGPU::V_RCP_F32_e64_gfx6_gfx7:
        {
            // VOP3a versions of VOP1 insts
            auto [ dst, srcMods, src, clamp, omod ] = unwrapMI(MI, R0_4);
            mlir::Value srcVal;

            srcVal = sourceVectorOperand(src, floatTy);
            srcVal = applyOperandModifiers(srcVal, srcMods);

            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_RCP_F32_e32_gfx6_gfx7:
                    result = builder.create<FDivOp>(getOne(floatTy), srcVal);
                    break;
            }

            result = applyResultModifiers(result, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_ADD_F32_e64_gfx6_gfx7:
        case AMDGPU::V_MUL_F32_e64_gfx6_gfx7:
        case AMDGPU::V_MAX_F32_e64_gfx6_gfx7:
        {
            // VOP3a versions of VOP2 insts
            auto [ dst, src0Mods, src0, src1Mods, src1, clamp, omod ] = unwrapMI(MI, R0_6);
            mlir::Value a, b;

            a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);

            mlir::Value result;
            switch (opcode) {
                case AMDGPU::V_ADD_F32_e64_gfx6_gfx7:
                    result = builder.create<FAddOp>(a, b);
                    break;
                case AMDGPU::V_MUL_F32_e64_gfx6_gfx7:
                    result = builder.create<FMulOp>(a, b);
                    break;
                case AMDGPU::V_MAX_F32_e64_gfx6_gfx7:
                    result = builder.create<GLFMaxOp>(a, b);
                    break;
            }

            result = applyResultModifiers(result, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_MED3_F32_gfx6_gfx7:
        {
            // TODO OMOD, CLAMP
            auto [ dst, src0Mods, src0, src1Mods, src1, src2Mods, src2, clamp, omod ] = unwrapMI(MI, R0_8);
            mlir::Value a, b, c;

            a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            c = sourceVectorOperand(src2, floatTy);
            c = applyOperandModifiers(c, src2Mods);
            mlir::Value result;
            //If (isNan(S0.f) || isNan(S1.f) || isNan(S2.f))
            //  D.f = MIN3(S0.f, S1.f, S2.f)
            //Else if (MAX3(S0.f,S1.f,S2.f) == S0.f)
            //  D.f = MAX(S1.f, S2.f)
            //Else if (MAX3(S0.f,S1.f,S2.f) == S1.f)
            //  D.f = MAX(S0.f, S2.f)
            //Else
            //  D.f = MAX(S0.f, S1.f)

            // ignore NaNs for now
            mlir::Value cmp, max_a_b, max3, max_b_c, max_a_c;
            max_a_b = builder.create<GLFMaxOp>(a, b);
            max_a_c = builder.create<GLFMaxOp>(a, c);
            max_b_c = builder.create<GLFMaxOp>(b, c);
            max3 = builder.create<GLFMaxOp>(max_a_b, max_a_c);

            cmp = builder.create<FOrdEqualOp>(max3, a);
            result = builder.create<SelectOp>(cmp, max_b_c, max_a_b);
            cmp = builder.create<FOrdEqualOp>(max3, b);
            result = builder.create<SelectOp>(cmp, max_a_c, result);

            result = applyResultModifiers(result, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_SAD_U32_gfx6_gfx7:
        {
            auto [ dst, src0, src1, src2 ] = unwrapMI(MI, R0_3);
            // Dunno what operand 4 is
            assert(MI.getOperand(4).getImm() == 0);
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(src1, intTy);
            mlir::Value c = sourceVectorOperand(src2, intTy);
            mlir::Value result;
            // ABS_DIFF (A,B) = (A>B) ? (A-B) : (B-A)
            // D.u = ABS_DIFF (S0.u,S1.u) + S2.u
            mlir::Value absDiff, aSubB, bSubA;
            aSubB = builder.create<ISubOp>(a, b);
            bSubA = builder.create<ISubOp>(b, a);
            auto cmp = builder.create<UGreaterThanOp>(a, b);
            absDiff = builder.create<SelectOp>(cmp, aSubB, bSubA);
            result = builder.create<IAddOp>(absDiff, c);
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_MAD_U64_U32_gfx7:
        {
            auto [ dst, cc, src0, src1, src2 ] = unwrapMI(MI, R0_4);
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(src1, intTy);
            // Multiply add using the product of two 32-bit unsigned integers, then added to a 64-bit integer.
            // {vcc_out,D.u64} = S0.u32 * S1.u32 + S2.u64

            // product : { lo, hi : 32 }
            mlir::Value product = builder.create<UMulExtendedOp>(a, b);
            mlir::Value lo = builder.create<CompositeExtractOp>(product, std::array{0});
            mlir::Value hi = builder.create<CompositeExtractOp>(product, std::array{1});
            mlir::Value prod64 = create64BitValue(lo, hi);
            mlir::Value addend = sourceVectorOperand64(src2, int64Ty);
            mlir::Value madAndCarry = builder.create<IAddCarryOp>(prod64, addend);
            mlir::Value mad = builder.create<CompositeExtractOp>(madAndCarry, std::array{0});
            mlir::Value carry = builder.create<CompositeExtractOp>(madAndCarry, std::array{1});
            carry = builder.create<INotEqualOp>(getZero(carry.getType()), carry);
            auto store = storeResultCC(cc, carry);
            tagPredicated(store);
            storeVectorResult64(dst.getReg(), mad);
            break;
        }

        case AMDGPU::V_MAD_U32_U24_gfx6_gfx7:
        {
            auto [ dst, src0, src1, src2 ] = unwrapMI(MI, R0_3);
            mlir::Value a, b, c, mad;
            a = sourceVectorOperand(src0, intTy);
            b = sourceVectorOperand(src1, intTy);
            c = sourceVectorOperand(src2, intTy);

            // a and b are treated as 24 bit uints
            auto mask = getI32Const((1 << 24) - 1);
            a = builder.create<BitwiseAndOp>(a, mask);
            b = builder.create<BitwiseAndOp>(b, mask);
            mad = builder.create<IMulOp>(a, b);
            mad = builder.create<IAddOp>(mad, c);
            storeVectorResult32(dst, mad);
            break;
        }

        case AMDGPU::V_ADDC_U32_e64_gfx6_gfx7:
        {
            auto [ dst, ccIn, src0, src1, ccOut ] = unwrapMI(MI, R0_4);
            handleVAddCU32(dst, src0, src1, ccIn.getReg(), ccOut.getReg());
            break;
        }

        case AMDGPU::V_MAD_F32_gfx6_gfx7:
        {
            // TODO OMOD, CLAMP
            auto [ dst, src0Mods, src0, src1Mods, src1, src2Mods, src2, clamp, omod ] = unwrapMI(MI, R0_8);
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            mlir::Value c = sourceVectorOperand(src2, floatTy);
            c = applyOperandModifiers(c, src2Mods);

            auto mad = handleVMadF32(a, b, c);
            mad = applyResultModifiers(mad, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, mad);
            break;
        }

        case AMDGPU::V_FMA_F32_gfx6_gfx7:
        {
            // Fused single-precision multiply-add. Only for double-precision parts
            // D.f = S0.f * S1.f + S2.f
            // Don't know why this ^ says double-precision parts TODO
            auto [ dst, src0Mods, src0, src1Mods, src1, src2Mods, src2, clamp, omod ] = unwrapMI(MI, R0_8);
            mlir::Value a, b, c;
            a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            c = sourceVectorOperand(src2, floatTy);
            c = applyOperandModifiers(c, src2Mods);
            mlir::Value fma = builder.create<FMulOp>(a, b);
            fma = builder.create<FAddOp>(fma, c);
            fma = applyResultModifiers(fma, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, fma);
            break;
        }

        case AMDGPU::V_CVT_PKRTZ_F16_F32_e64_gfx6_gfx7:
        {
            auto [ dst, src0Mods, src0, src1Mods, src1, clamp, omod ] = unwrapMI(MI, R0_6);
            mlir::Value a, b;
            a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            auto pack = handleVCvtPkrtzF16F32(a, b);
            pack = applyResultModifiers(pack, clamp.getImm(), omod.getImm());
            storeVectorResult32(dst, pack);
            break;
        }

        case AMDGPU::V_CNDMASK_B32_e64_gfx6_gfx7:
        {
            auto [ dst, src0Mods, src0, src1Mods, src1, srcSel ] = unwrapMI(MI, R0_5);
            assert(src0Mods.getImm() == 0 && src1Mods.getImm() == 0);
            uint selNo = srcSel.getReg();
            handleVCndMaskB32(dst, src0, src1, selNo);
            break;
        }

        // VINTRP
        case AMDGPU::V_INTERP_P1_F32_si:
        {
            // ignore for now
            break;
        }
        case AMDGPU::V_INTERP_P2_F32_si:
        {
            // Simple for now
            // Assume previous V_INTERP_P1 with 1st barycentric coord.
            // Assume this one uses 2nd bary coord
            auto [ dst, _1, _2, attrOprnd, componentOprnd ] = unwrapMI(MI, R0_4);

            int attrno, component;
            // src = MI.getOperand(2); // holds barycentric coord (I think): ignore and let Vulkan driver handle
            attrno = attrOprnd.getImm();
            component = componentOprnd.getImm();

            //assert(inp)
            auto it = inputAttributes.find(attrno);
            assert(it != inputAttributes.end());
            InputAttribute attr = it->second;
            PointerType attrPtrType = cast<PointerType>(attr.getType());
            mlir::Type attrType = attrPtrType.getPointeeType();

            auto gvRef = builder.create<AddressOfOp>(attr);
            mlir::Value attrVal = builder.create<LoadOp>(gvRef);
            if (auto fvecTy = dyn_cast<mlir::VectorType>(attrType)) {
                attrVal = builder.create<CompositeExtractOp>(attrVal, std::array{component});
            }
            assert(dyn_cast<mlir::VectorType>(attrType) || component == 0);
            storeVectorResult32(dst, attrVal);
            break;
        }

        // TODO: one descriptor heap per image dims? (1D, 2D, 3D, ...)
        // Can you always tell what dimensions an image is given an instruction?
        //     -> don't think so
        // Need some kind of indirection for separate heaps?
        // Option 1:
        // Two level heap for images: First level heap (all images) stores record (enum for next heap, index)
        // Option 2:
        // One heap for all images, another with matching array for metadata.
        // Can you put 1D, 2D, etc image descriptors in one runtime array?
        // Should be able to: in vulkan, they should share the same 

        // MIMG
        case AMDGPU::IMAGE_LOAD_V1_V1:
        case AMDGPU::IMAGE_LOAD_V4_V1:

        case AMDGPU::IMAGE_LOAD_MIP_V4_V1:
        {
            // TODO: Hack AF: should figure out how to write helper functions in glsl/opencl, compile to spirv, link with the MLIR module

            // V4_V1: 4 components of image, 1D address?
            // How else would we determine #components in address?
            auto [dst, addr, imageConst, dmask, unorm, _5, _6, _7, _8, _9, _10] = unwrapMI(MI, R0_10);
            assert(_5.getImm() == 0 && _6.getImm() == 0 && _7.getImm() == 0 && _8.getImm() == 0 && _9.getImm() == 0 && _10.getImm() == 0);

            uint dstRegno = vgprGroupToVgprBase(dst.getReg());
            uint imageConstBaseSgpr = sgprGroupToSgprBase(imageConst.getReg());
            mlir::Value imageIdx = sourceUniformReg(imageConstBaseSgpr);

            bool useMip;
            bool useTexDesc;
            bool doSample;
            switch (opcode) {
                case AMDGPU::IMAGE_LOAD_MIP_V4_V1:
                    useMip = true;
                    useTexDesc = true;
                    break;
                default:
                    useMip = false;
                    useTexDesc = true;
                    doSample = false;
                    break;
            }

            mlir::Value imageMetadataRecord = useTexDesc ? lookupTextureMetaData(imageIdx) : lookupImageMetaData(imageIdx);
            mlir::Value dimTypeVal = extractImageDimType(imageMetadataRecord);

            // V4_V1 : V4(#channels)_V1(???)
            // -> don't know what 2nd (V1) means. I've seen situations like
	        //      v_sad_u32 v4, vcc_lo, 0, v0
	        //      v_sad_u32 v5, s16, 0, v1
	        //      image_load_mip v[0:3], v4, s[0:7] dmask:0xf
            // where v4 is the only coord listed but v5 is written, when it would be otherwise dead after
            // the load. 

            uint addrRegBase = vgprGroupToVgprBase(dst.getReg());

            // "For image opcodes with no sampler, all VGPR address values are taken as uint"
            // Look at table 8.8 for addr

            // Use OpImageFetch or OpImageRead?
            // Difference is "Sampled" opeand of OpTypeImage
            // Spirv spec:
            // 0 indicates this is only known at run time, not at compile time
            // 1 indicates an image compatible with sampling operations
            // 2 indicates an image compatible with read/write operations (a storage or subpass data image).

            // "Reads: DMASK specifies which elements of the resource are returned to"
            // "consecutive VGPRs. The texture system reads data from memory and based"
            // "on the data format expands it to a canonical RGBA form, filling in zero or"
            // "one for missing components. Then, DMASK is applied, and only those"
            // "components selected are returned to the shader."

            // "Writes: When writing an image object, it is only possible to write an entire"
            // "element (all components), not just individual components. The components"
            // "come from consecutive VGPRs, and the texture system fills in the value zero"
            // "for any missing components of the image’s data format; it ignores any values"
            // "that are not part of the stored data format. For example, if the DMASK=1001,"
            // "the shader sends Red from VGPR_N, and Alpha from VGPR_N+1, to the"
            // "texture unit. If the image object is RGB, the texel is overwritten with Red from"
            // "the VGPR_N, Green and Blue set to zero, and Alpha from the shader"
            // "ignored."

            uint dmaskImm = dmask.getImm();
            int texelNumComponents = 0;
            // Basically find the index of the MSB of validMask
            if (dmaskImm & 8) {
                texelNumComponents = 4;
            } else if (dmaskImm & 4) {
                texelNumComponents = 3;
            } else if (dmaskImm & 2) {
                texelNumComponents = 2;
            } else if (dmaskImm & 1) {
                texelNumComponents = 1;
            }

            assert(texelNumComponents != 0);

            // "Result Type must be a scalar or vector of floating-point type or integer type."
            // "It must be a scalar or vector with component type the same as Sampled Type of the OpTypeImage (unless that Sampled Type is OpTypeVoid)."
            // TODO: probably do runtime switch based on format
            // Not sure if "sampled type" in OpTypeImage matters though, or if it's just a bitcast
            // on the read/fetch result.
            mlir::Type eltTy = floatTy;

            mlir::Type texelTy;
            if (texelNumComponents > 1) {
                texelTy = mlir::VectorType::get({texelNumComponents}, eltTy);
            } else {
                texelTy = eltTy;
            }

            mlir::Value texelTempReg = buildTempReg("imageLoadTexelTemp", texelTy);
            {
                mlir::Block *topLevelBlock = builder.getBlock();

                // TODO generate functions to factor this stuff out
                // TODO OpSwitch not supported in MLIR yet
                std::array<mlir::spirv::Dim, 3> dimTypes = { Dim::Dim1D, Dim::Dim2D, Dim::Dim3D };
                for (uint i = 0; i < dimTypes.size(); i++) {
                    mlir::spirv::Dim dim = dimTypes[i];
                    mlir::Block *headerBlock, *thenBlock, *elseBlock, *mergeBlock;

                    auto selection = builder.create<SelectionOp>(SelectionControl::None);
                    mlir::Region &selectionRegion = selection.getRegion();

                    headerBlock = &selectionRegion.emplaceBlock();
                    thenBlock = &selectionRegion.emplaceBlock();
                    mergeBlock = &selectionRegion.emplaceBlock();
                    if (i == dimTypes.size() - 1) {
                        elseBlock = mergeBlock;
                    } else {
                        elseBlock = new mlir::Block();
                        elseBlock->insertBefore(mergeBlock);
                    }

                    builder.setInsertionPointToEnd(headerBlock);
                    mlir::Value cmp = builder.create<IEqualOp>(dimTypeVal, getI8Const(static_cast<uint>(dim)));
                    builder.create<BranchConditionalOp>(cmp, thenBlock, mlir::ValueRange(), elseBlock, mlir::ValueRange(), std::nullopt);

                    builder.setInsertionPointToEnd(thenBlock);
                    {
                        mlir::Value imageDescriptor;
                        int numAddrComponents = 0;
                        switch (dim) {
                            case Dim::Dim1D:
                                imageDescriptor = useTexDesc ? lookupTextureDescriptor1D(imageIdx) : lookupImageDescriptor1D(imageIdx);
                                numAddrComponents = 1;
                                break;
                            case Dim::Dim2D:
                                imageDescriptor = useTexDesc ? lookupTextureDescriptor2D(imageIdx) : lookupImageDescriptor2D(imageIdx);
                                numAddrComponents = 2;
                                break;
                            case Dim::Dim3D:
                                imageDescriptor = useTexDesc ? lookupTextureDescriptor3D(imageIdx) : lookupImageDescriptor3D(imageIdx);
                                numAddrComponents = 3;
                                break;
                            default:
                                assert(false);
                                break;
                        }
                        mlir::Type addrType = intTy;
                        if (numAddrComponents > 1) {
                            addrType = mlir::VectorType::get({numAddrComponents}, addrType);
                        }
                        mlir::Value addrVal = getZero(addrType);
                        for (int i = 0 ; i < numAddrComponents; i++) {
                            mlir::Value component = sourceVectorGpr(addrRegBase + i);
                            if (numAddrComponents > 1) {
                                addrVal = builder.create<CompositeInsertOp>(component, addrVal, std::array{i});
                            } else {
                                addrVal = component;
                            }
                        }


                        mlir::Value texel;
                        ImageOperands imageOperandsAttr = ImageOperands::None;
                        llvm::SmallVector<mlir::Value, 2> imageOperands;
                        if (useMip) {
                            // TODO: need to use image with sampled bit and OpImageFetch.
                            // Lod image oprnd not allowed with OpImageRead
                            imageOperandsAttr = imageOperandsAttr | ImageOperands::Lod;
                            mlir::Value mipLevelVal = sourceVectorGpr(addrRegBase + numAddrComponents);
                            imageOperands.push_back(mipLevelVal);
                            texel = builder.create<ImageFetchOp>(texelTy, imageDescriptor, addrVal, 
                                    ImageOperandsAttr::get(&mlirContext, imageOperandsAttr), imageOperands);
                        } else if (doSample) {
                        
                        } else {
                            texel = builder.create<ImageReadOp>(texelTy, imageDescriptor, addrVal, 
                                    ImageOperandsAttr::get(&mlirContext, imageOperandsAttr), imageOperands);
                        }
                        builder.create<StoreOp>(texelTempReg, texel);
                    }
                    builder.create<BranchOp>(mergeBlock);

                    builder.setInsertionPointToEnd(mergeBlock);
                    builder.create<MergeOp>();

                    builder.setInsertionPointToEnd(elseBlock);
                    if (i != dimTypes.size() - 1) {
                        auto term = builder.create<BranchOp>(mergeBlock);
                        builder.setInsertionPoint(term);
                    }
                }

                builder.setInsertionPointToEnd(topLevelBlock);
            }

            auto texel = builder.create<LoadOp>(texelTempReg);
            for (int i = 0; i < texelNumComponents; i++) {
                if (dmaskImm & (1 << i)) {
                    mlir::Value component = texel;
                    if (isa<mlir::VectorType>(texelTy)) {
                        component = builder.create<CompositeExtractOp>(component, std::array{i});
                    }
                    // TODO: do we tightly pack the returned components based on dmask (Like data provided for writes)?
                    // If not, then store to dstRegno + i, don't bump dstRegno
                    storeVectorResult32(dstRegno, component);
                    ++dstRegno;
                }
            }
            break;
        }

        case AMDGPU::IMAGE_GET_RESINFO_V2_V1:
        {
            auto [dst, mipLevel, imageConst, dmask, _4, _5, _6, _7, _8, _9] = unwrapMI(MI, R0_9);
            assert(_4.getImm() == 0 && _5.getImm() == 0 && _6.getImm() == 0 && _7.getImm() == 0 && _8.getImm() == 0 && _9.getImm() == 0);

            int dmaskImm = dmask.getImm();
            // spec:
            // No sampler. Returns resource info into four VGPRs for the speci-
            // fied MIP level. These are 32-bit integer values:
            // Vdata3-0 = { #mipLevels, depth, height, width }
            // For cubemaps, depth = 6 * Number_of_array_slices.

            // TODO: need to be tagging things like image reads as predicated, for V_ instructions, because they might cause exceptions

            uint dstRegno = vgprGroupToVgprBase(dst.getReg());
            uint imageConstBaseSgpr = sgprGroupToSgprBase(imageConst.getReg());
            mlir::Value imageIdx = sourceUniformReg(imageConstBaseSgpr);
            mlir::Value imageDescriptor = lookupImageDescriptor2D(imageIdx);
            // For now, extract the contained image from the combined image sampler.
            // The type of the image arg to ImageQuery* must be OpTypeImage

            mlir::Value mipLevelVal = sourceVectorOperand(mipLevel, intTy);

            uint numDimComponents = 0;
            if (dmaskImm & 4) {
                // depth
                numDimComponents = 3;
            } else if (dmaskImm & 2) {
                // height
                numDimComponents = 2;
            } else if (dmaskImm & 1) {
                // width
                numDimComponents = 1;
            }
            
            mlir::Type dimsType = intTy;
            if (numDimComponents > 1) {
                dimsType = mlir::VectorType::get({numDimComponents}, dimsType);
            }

            if (numDimComponents > 0) {
                auto sizeQuery = builder.create<ImageQuerySizeLodOp>(dimsType, imageDescriptor, mipLevelVal);

                for (int i = 0; i < 3; i++) {
                    if (dmaskImm & (1 << i)) {
                        mlir::Value component = sizeQuery;
                        if (isa<mlir::VectorType>(dimsType)) {
                            component = builder.create<CompositeExtractOp>(component, std::array{i});
                        }
                        storeVectorResult32(dstRegno + i, component);
                    }
                }
            }

            if (dmaskImm & 8) {
                // query num levels
                mlir::Value numLevels = builder.create<ImageQueryLevelsOp>(intTy, imageDescriptor);
                storeVectorResult32(dstRegno + 3, numLevels);
            }

            break;
        }

        // EXP
        case AMDGPU::EXP_DONE_si:
        case AMDGPU::EXP_si:
        {
            uint target, validMask;
            auto [ targetOp, vsrc0, vsrc1, vsrc2, vsrc3, _done, _compr, validOp ] = unwrapMI(MI, R0_7);

            target = targetOp.getImm();
            validMask = validOp.getImm();

            assert( !(isFragShader() && !isRenderTarget(target)));
            if (target != AMDGPU::Exp::Target::ET_NULL) {
                // dunno what validMask actually is.
                // assuming is a writemask, but description of EXP instructions makes that sound wrong
                // Just treat it like a writemask for now
                assert(exportTargets.contains(target));
                ExportTarget exp = exportTargets[target];
                auto gvRef = builder.create<AddressOfOp>(exp);
                mlir::Type dstType = cast<PointerType>(gvRef.getType()).getPointeeType();

                int dstNumComponents;
                mlir::Type eltTy;
                if (auto vecTy = dyn_cast<mlir::VectorType>(dstType)) {
                    dstNumComponents = vecTy.getDimSize(0);
                    eltTy = vecTy.getElementType();
                } else {
                    dstNumComponents = 1;
                    eltTy = dstType;
                }
                assert(cast<mlir::FloatType>(eltTy));

                mlir::Value src;
                uint fullMask = (1 << dstNumComponents) - 1;
                if (fullMask != validMask) {
                    // partial write
                    // read from output variable, update it partially, then write the result back
                    src = builder.create<LoadOp>(gvRef);
                    printf("LOG: export is partial write\n");
                } else {
                    src = getZero(dstType);
                }

                uint width = eltTy.getIntOrFloatBitWidth();
                // index arg to composite insert/extract. Reuse
                switch (width) {
                    case 16:
                    {
                        // exports are in 16 bit format.
                        // vsrc0 and vsrc1 each contain 2 f16's
                        assert(_compr.getImm());
                        mlir::Value src_0_1, src_2_3;

                        if (validMask & 0x3) {
                            src_0_1 = sourceVectorGpr(vsrc0.getReg());
                            // Can we just bitfield extract and bitcast it? TODO
                            // Instead of caring about float conversions?
                            // uint -> vec2
                            src_0_1 = builder.create<GLUnpackHalf2x16Op>(v2float32Ty, src_0_1);
                        }
                        if (validMask & 0xc) {
                            src_2_3 = sourceVectorGpr(vsrc1.getReg());
                            // uint -> vec2
                            src_2_3 = builder.create<GLUnpackHalf2x16Op>(v2float32Ty, src_2_3);
                        }
                        for (int i = 0; i < dstNumComponents; i++) {
                            if (validMask & (1 << i)) {
                                auto half = i <= 1 ? src_0_1 : src_2_3;
                                // vec2 -> float
                                mlir::Value component = builder.create<CompositeExtractOp>(half, std::array{i&1});
                                // float -> float16
                                component = builder.create<FConvertOp>(float16Ty, component);
                                if (dstNumComponents > 1) {
                                    src = builder.create<CompositeInsertOp>(component, src, std::array{i});
                                } else {
                                    src = component;
                                }
                            }
                        }
                        break;
                    }
                    case 32:
                    {
                        llvm::SmallVector<MCOperand, 4> srcRegs = { vsrc0, vsrc1, vsrc2, vsrc3 };
                        for (int i = 0; i < dstNumComponents; i++) {
                            if (validMask & (1 << i)) {
                                mlir::Value component = sourceVectorOperand(srcRegs[i], eltTy);
                                if (dstNumComponents > 1) {
                                    src = builder.create<CompositeInsertOp>(component, src, std::array{i});
                                } else {
                                    src = component;
                                }
                            }
                        }                        
                        break;
                    }
                    default:
                        assert(false && "unhandled");
                        break;
                }
                auto store = builder.create<StoreOp>(gvRef, src);
                // exports shouldn't be written if the thread is masked off by EXEC
                // tag with "predicated" to later generate if stmt
                tagPredicated(store);
            }
            break;
        }

        default:
        {
            static bool flag = false;
            if ( !flag) {
                // Test
                loadSpirvLibrary(SpirvLibrary::MISC_HELPER_FUNCTIONS);

                FuncOp fn = getOrInsertLibraryCallDecl("mySum", {floatTy, floatTy}, floatTy);
                createLibraryCall(fn, {getF32Const(7.7), getF32Const(8.8)});
            }
            if (!Quiet) {
                llvm::errs() << "Unhandled instruction: \n";
                IP->printInst(&MI, 0, "", *STI, llvm::errs());
                llvm::errs() << "\n";
            }
            return false;
        }
    }
    return true;
}

bool GcnToSpirvConverter::buildSpirvDialect() {
    bool success = true;
    moduleOp = functionBuilder.create<ModuleOp>(AddressingModel::PhysicalStorageBuffer64, MemoryModel::GLSL450);
    mlir::Region &region = moduleOp.getBodyRegion();
    assert(region.hasOneBlock());
    functionBuilder.setInsertionPointToEnd(&(*region.begin()));

    mlir::FunctionType mainFTy = mlir::FunctionType::get(&mlirContext, {}, functionBuilder.getNoneType());
    mainFn = functionBuilder.create<FuncOp>("main", mainFTy);

    assert(mainFn->getNumRegions() == 1);
    mlir::Region &mainRegion = mainFn->getRegion(0);
    mainEntryBlock = &mainRegion.emplaceBlock();
    builder.setInsertionPointToStart(mainEntryBlock);
    mlir::Block &mainBody = mainRegion.emplaceBlock();
    // Keep OpVariables, etc in separate block
    builder.create<BranchOp>(&mainBody);
    builder.setInsertionPointToStart(&mainBody);

    if ( !processSrds()) {
        return false;
    }

    if ( !prepass()) {
        return false;
    }

    // TODO : load driver generated values, ie vertex index, based on shader stage/context, into initial vgpr/sgpr

    for (const MCInst &MI: machineCode) {
        success &= convertGcnOp(MI);
    }
    builder.create<ReturnOp>();
    
    return true;
    // return err; // TODO
}

bool GcnToSpirvConverter::finalizeModule() {
    mlir::Region &region = moduleOp.getBodyRegion();
    builder.setInsertionPointToEnd(&(*region.begin()));
    llvm::SmallVector<mlir::Attribute, 8> interfaceVars;

    for (const auto &[location, attr] : inputAttributes) {
        interfaceVars.push_back(mlir::SymbolRefAttr::get(attr));
    }

    for (const auto &[location, attr] : exportTargets) {
        interfaceVars.push_back(mlir::SymbolRefAttr::get(attr));
    }

    // For now, consider a descriptor set to be a single block (1 binding)
    for (const auto &[dsno, descriptorSetBlock] : descriptorSets) {
        interfaceVars.push_back(mlir::SymbolRefAttr::get(descriptorSetBlock));
    }

    builder.create<EntryPointOp>(execModel, mainFn, interfaceVars);
    if (isFragShader()) {
        llvm::SmallVector<int, 1> lits;
        builder.create<ExecutionModeOp>(mainFn, ExecutionMode::OriginUpperLeft, lits);
    }

    // TODO be conservative with Capabilities/extensions
    llvm::SmallVector<Capability, 8> caps = { Capability::Shader, Capability::Float16, Capability::Int16, Capability::Int64, Capability::PhysicalStorageBufferAddresses, Capability::ImageQuery,
            Capability::StorageImageReadWithoutFormat, Capability::StorageImageWriteWithoutFormat, Capability::Sampled1D, Capability::Int8, Capability::Linkage };
    llvm::SmallVector<Extension, 4> exts = { Extension::SPV_EXT_descriptor_indexing, Extension::SPV_KHR_physical_storage_buffer };
    auto vceTriple = VerCapExtAttr::get(Version::V_1_6, caps, exts, &mlirContext);
    moduleOp->setAttr("vce_triple", vceTriple);

    if (auto dumpPath = getMlirDumpPath()) {
        auto outfile = createDumpFile(dumpPath.value(), ".mlir", FileNum);
        if (outfile) {
            moduleOp->print(*outfile);
        } else {
            printf("Couldn't dump mlir to file %s\n", dumpPath.value().c_str());
            return 1;
        }
    }

    return true;
}

bool GcnToSpirvConverter::generateSpirvBytecode() {
    // TODO - temp
    // strip off added attributes
    moduleOp.walk([&](mlir::Operation *op) {
        for (mlir::NamedAttribute attr: op->getDiscardableAttrs()) {
            if (attr.getName().strref().starts_with("gcn.")) {
                op->removeAttr(attr.getName());
            }
        }
    });

    {
        // Stupid vector copy
        llvm::SmallVector<uint32_t, 0> tempSpirvModule;
        auto res = mlir::spirv::serialize(moduleOp, tempSpirvModule);
        if (res.failed()) {
            return false;
        }
        spirvModule.resize(tempSpirvModule.size());
        std::copy(tempSpirvModule.begin(), tempSpirvModule.end(), spirvModule.begin());
    }

    std::vector<ArrayRef<uint32_t>> inputLibs;
    for (uint i = 0 ; i < spirvLibraries.size(); i++) {
        if (usedSpirvLibraries[i]) {
            inputLibs.push_back(spirvLibraries[i]);
        }
    }
    bool needsLink = !inputLibs.empty();
    if (needsLink) {
        if (auto dumpPath = getSpirvDumpPath()) {
            auto outfile = createDumpFile(dumpPath.value(), "-before-link.spv", FileNum);
            if (outfile) {
                outfile->write(reinterpret_cast<const char *>(spirvModule.data()), spirvModule.size() * sizeof(uint32_t));
                if (outfile->has_error()) {
                    printf("Couldn't dump spirv bytecode to file %s\n", dumpPath.value().c_str());
                    return 1;
                }
            }
        }

        std::vector<uint32_t> linkedModule;
        if ( !linkSpirvModules(spirvModule, inputLibs, linkedModule)) {
            return false;
        }
        spirvModule = std::move(linkedModule);
    }

    if (auto dumpPath = getSpirvDumpPath()) {
        std::string extension = needsLink ? "-after-link.spv" : ".spv";
        auto outfile = createDumpFile(dumpPath.value(), "-after-link.spv", FileNum);
        if (outfile) {
            outfile->write(reinterpret_cast<const char *>(spirvModule.data()), spirvModule.size() * sizeof(uint32_t));
            if (outfile->has_error()) {
                printf("Couldn't dump spirv bytecode to file %s\n", dumpPath.value().c_str());
                return 1;
            }
        }
    }

    return true;
}

SpirvConvertResult GcnToSpirvConverter::convert() {
    if ( !buildSpirvDialect()) {
        return SpirvConvertResult({}, SpirvConvertResult::fail);
    }
    if ( !finalizeModule()) {
        return SpirvConvertResult({}, SpirvConvertResult::fail);
    }
    if ( !generateSpirvBytecode()) {
        return SpirvConvertResult({}, SpirvConvertResult::fail);
    }
    return SpirvConvertResult(std::move(spirvModule), SpirvConvertResult::success);
}

bool decodeAndConvertGcnCode(const std::vector<unsigned char> &gcnBytecode, ExecutionModel execModel, SrdMode srdMode, std::vector<InputUsageSlot> srdSlots) {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();

    const char *tripleName = "amdgcn-unknown-linux-gnu";
    //const char *mcpu = "fiji";
    const char *mcpu = "bonaire";
    const char *features = "";

    // Get the target specific parser.
    std::string Error;
    Triple TheTriple(Triple::normalize(tripleName));  
    const Target *TheTarget = TargetRegistry::lookupTarget("amdgcn", TheTriple,
                                                            Error);

    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(tripleName));
    assert(MRI && "Unable to create target register info!");

    MCTargetOptions mcOptions;
    std::unique_ptr<MCAsmInfo> MAI(
        TheTarget->createMCAsmInfo(*MRI, tripleName, mcOptions));
    assert(MAI && "Unable to create target asm info!");  

    std::unique_ptr<MCSubtargetInfo> STI(
        TheTarget->createMCSubtargetInfo(tripleName, mcpu, features));
    assert(STI && "Unable to create subtarget info!");

    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    assert(MCII && "Unable to create instruction info!");

    MCInstPrinter *IP = TheTarget->createMCInstPrinter(Triple(tripleName), 0, *MAI, *MCII, *MRI);

    std::unique_ptr<MCAsmInfo> AsmInfo(
        TheTarget->createMCAsmInfo(*MRI, TheTriple.getTriple(), MCTargetOptions()));

    std::unique_ptr<MCContext> Ctx(
        new MCContext(TheTriple, AsmInfo.get(), MRI.get(), STI.get()));

    std::unique_ptr<MCDisassembler> DisAsm(
        TheTarget->createMCDisassembler(*STI, *Ctx));

    uint64_t size;
    uint64_t addr = 0;

    ArrayRef byteView(gcnBytecode.data(), gcnBytecode.size());

    llvm::outs() << "\n";
    std::vector<MCInst> machineCode;
    while (addr < byteView.size()) {
        MCInst MI;
        MCDisassembler::DecodeStatus Res;
        Res = DisAsm->getInstruction(MI, size, byteView.slice(addr), addr, llvm::nulls());

        switch (Res) {
            case MCDisassembler::DecodeStatus::Success:
                break;
            case MCDisassembler::DecodeStatus::Fail:
            case MCDisassembler::DecodeStatus::SoftFail:
                printf("Failed decoding instruction\n");
                return false;
        }
        if (PrintGcn) {
            IP->printInst(&MI, 0, "", *STI, llvm::outs());
            llvm::outs() << "\n";
        }

        machineCode.push_back(std::move(MI));

        addr += size;
    }

    GcnToSpirvConverter converter(machineCode, execModel, srdMode, srdSlots, STI.get(), IP);
    SpirvConvertResult result = converter.convert();
    SpirvBytecodeVector spirvModule = result.takeSpirvModule();

    llvm_shutdown();
    return result;
}

int main(int argc, char **argv) {
    int err = 0;
    std::vector<uint8_t> buf;

    cl::HideUnrelatedOptions(MyCategory);
    cl::ParseCommandLineOptions(argc, argv);

    if (!readFileBytes(InputGcnPath.getValue(), buf)) {
        return 1;
    }
    size_t len = buf.size();

    const char *magic = "OrbShdr";
    const uint32_t fileHeaderMagic = 0x72646853; // "Shdr"

    // location "OrbShdr" string
    llvm::SmallVector<size_t, 4> programOffsets;
    // location of "Shdr" string
    llvm::SmallVector<size_t, 4> shdrOffsets;
    // Headers that follow binary
    llvm::SmallVector<size_t, 4> binaryInfoOffsets;

    for (size_t i = 0; i <= len - strlen(magic);) {
        if ( !memcmp(&buf[i], magic, strlen(magic))) {
            assert(i + sizeof(ShaderBinaryInfo) <= len);
            if (programOffsets.size() == binaryInfoOffsets.size()) {
                printf("Found BinaryInfo header without previous first instruction (mov vcc_hi 26). Skipping\n");
                err = 1;
                continue;
            }
            binaryInfoOffsets.push_back(i);
            i += strlen(magic);
        } else if (*reinterpret_cast<uint32_t *>(&buf[i]) == fileHeaderMagic) {
            shdrOffsets.push_back(i);
            i += sizeof(fileHeaderMagic);
        } else if ( *reinterpret_cast<uint32_t *>(&buf[i]) == 0xbeeb03ff) {
            if (programOffsets.size() > binaryInfoOffsets.size()) {
                printf("Found 2 first instructions (mov vcc_hi 26) in a row. Dropping the first one\n");
                err = 1;
                programOffsets.pop_back();
            }
            programOffsets.push_back(i);
            i+=4;
        } else {
            i++;
        }
    }

    size_t numBinaries = programOffsets.size();
    assert(numBinaries == binaryInfoOffsets.size() && numBinaries == shdrOffsets.size());

    if (numBinaries == 0) {
        printf("couldn't find first instruction\n");
        return 1;
    }

    for (uint i = 0; i < numBinaries; i++) {
        std::vector<uint8_t> gcnBytecode;
        size_t codeOff = programOffsets[i];
        size_t infoOff = binaryInfoOffsets[i];
        size_t shdrMagicOff = shdrOffsets[i];

        //void *afterShdrMagic = &buf[shdrMagicOff + strlen(shdrMagic)];
        //auto *header = (Header *) buf.data();
        //auto *fileHeader = (ShaderFileHeader*) buf.data(); 
        //auto *common = (ShaderCommonData*) buf.data();

        const ShaderBinaryHeader *binaryHeader = reinterpret_cast<const ShaderBinaryHeader*>(&buf[shdrMagicOff]);
        const ShaderFileHeader *fileHeader = reinterpret_cast<const ShaderFileHeader *>(binaryHeader) - 1;
        const ShaderCommonData *shaderCommon = reinterpret_cast<const ShaderCommonData*>(binaryHeader + 1);

        ShaderBinaryInfo *binaryInfo = (ShaderBinaryInfo *) &buf[infoOff];
        gcnBytecode.resize(binaryInfo->m_length);
        memcpy(gcnBytecode.data(), &buf[codeOff], gcnBytecode.size());

        const uint32_t *usageMasks = reinterpret_cast<uint32_t *>(binaryInfo) - binaryInfo->m_chunkUsageBaseOffsetInDW;
        const InputUsageSlot *usageSlots = reinterpret_cast<const InputUsageSlot *>(usageMasks) - binaryInfo->m_numInputUsageSlots;

        ExecutionModel execModel;
        printf("Binary %i: \"Shdr\" at offset %lu, code at offset %lu, size %i, \"OrbShdr\" at offset %lu\n", i, shdrMagicOff, codeOff, binaryInfo->m_length, infoOff);
        switch (static_cast<ShaderBinaryStageType>(binaryInfo->m_type)) {
	        case ShaderBinaryStageType::kPixelShader:
                execModel = ExecutionModel::Fragment;
                break;
	        case ShaderBinaryStageType::kVertexShader:
                execModel = ExecutionModel::Vertex;
                break;
	        case ShaderBinaryStageType::kComputeShader:
                execModel = ExecutionModel::GLCompute;
                break;
	        case ShaderBinaryStageType::kGeometryShader:
                execModel = ExecutionModel::Geometry;
                break;
	        case ShaderBinaryStageType::kHullShader:
                execModel = ExecutionModel::TessellationControl;
                break;
	        case ShaderBinaryStageType::kDomainShader:
                execModel = ExecutionModel::TessellationEvaluation;
                break;
            default:
                err = true;
                continue;
        }

        printf("stage type: %s\n", stringifyExecutionModel(execModel).str().c_str());
        printf("SRD mode: %s\n", fileHeader->m_usesShaderResourceTable ? "flat table" : "immediate");

        switch (execModel) {
            case ExecutionModel::Vertex:
            {
                auto *vsShader = reinterpret_cast<const VsShader *>(shaderCommon);
                const VertexInputSemantic *inputSemantics = vsShader->getInputSemanticTable();
                const VertexExportSemantic *exportSemantics = vsShader->getExportSemanticTable();
                break;
            }
            case ExecutionModel::Fragment:
            {
                auto *psShader = reinterpret_cast<const PsShader *>(shaderCommon);
                const PixelInputSemantic *inputSemantics = psShader->getPixelInputSemanticTable();
                break;
            }
            default:
                break;
        }
        
        //assert(!!binaryInfo->m_isSrt == !!fileHeader->m_usesShaderResourceTable);
        std::vector<InputUsageSlot> srdSlots;

        if (PrintSlots) {
            printf("//////////////////////////////////////////////////////////////////////////\n");
            printf("Usage Slots\n");
            printf("count: %i\n", binaryInfo->m_numInputUsageSlots);
            printf("\n");
            for (uint i = 0; i < binaryInfo->m_numInputUsageSlots; i++) {
                const InputUsageSlot *slot = &usageSlots[i];
                srdSlots.push_back(*slot);
                ShaderInputUsageType usageType = static_cast<ShaderInputUsageType>(slot->m_usageType);
                printf("/---------------------------------------------------\n");
                printf("| slot %i\n", i);
                printf("| type: %s\n", shaderInputUsageTypeToName(usageType));
                printf("| m_apiSlot: %i\n", slot->m_apiSlot);
                printf("| m_startRegister: %i\n", slot->m_startRegister);
                printf("| m_registerCount: %i\n", slot->m_registerCount);
                printf("| m_resourceType: %i\n", slot->m_resourceType);
                printf("| m_reserved: %i\n", slot->m_reserved);
                printf("| m_chunkMask: %i\n", slot->m_chunkMask);
                printf("| m_srtSizeInDWordMinusOne: %i\n", slot->m_srtSizeInDWordMinusOne);
                printf("| \n");
                int resourceSize = shaderInputUsageTypeToSize(usageType);
                if (resourceSize > 0) {
                    printf("| size: %i Bytes\n", resourceSize);
                    printf("| registers: [%i:%i]\n", slot->m_startRegister, slot->m_startRegister + resourceSize/4 - 1);
                }
                switch (ShaderInputUsageType(slot->m_usageType)) {
                    case kShaderInputUsageImmShaderResourceTable:
                        break;

                    case kShaderInputUsageImmResource:
                    case kShaderInputUsageImmRwResource:
                        // If 0, resource type <c>V#</c>; if 1, resource type <c>T#</c>, in case of a Gnm::kShaderInputUsageImmResource
                        if (slot->m_resourceType == IMM_RESOURCE_BUFFER) {
                            printf("|   *** Resource is a V# - buffer resource constant - 128 bits \n");
                        } else { // IMM_RESOURCE_IMAGE
                            // T# image constants can be 128 or 256 bits. 
                            //
                            // All image resource view descriptors (T#’s) are written by the driver as 256 bits.
                            // It is permissible to use only the first 128 bits when a simple 1D or 2D (not an
                            // array) is bound. This is specified in the MIMG R128 instruction field.
                            // The MIMG-format instructions have a DeclareArray (DA) bit that reflects whether
                            // the shader was expecting an array-texture or simple texture to be bound. When
                            // DA is zero, the hardware does not send an array index to the texture cache. If
                            // the texture map was indexed, the hardware supplies an index value of zero.
                            // Indices sent for non-indexed texture maps are ignored.
                            printf("|   *** Resource is a T# - image resource constant - 128/256 bits\n");                        
                        }
                        break;
                    case kShaderInputUsageImmSampler:
                    case kShaderInputUsageImmConstBuffer:
                    case kShaderInputUsageImmVertexBuffer:
                    case kShaderInputUsageImmAluFloatConst:
                    case kShaderInputUsageImmAluBool32Const:
                    case kShaderInputUsageImmGdsCounterRange:
                    case kShaderInputUsageImmGdsMemoryRange:
                    case kShaderInputUsageImmGwsBase:
                    case kShaderInputUsageImmLdsEsGsSize:
                    case kShaderInputUsageSubPtrFetchShader:
                    case kShaderInputUsagePtrResourceTable:
                    case kShaderInputUsagePtrInternalResourceTable:
                    case kShaderInputUsagePtrSamplerTable:
                    case kShaderInputUsagePtrConstBufferTable:
                    case kShaderInputUsagePtrVertexBufferTable:
                    case kShaderInputUsagePtrSoBufferTable:
                    case kShaderInputUsagePtrRwResourceTable:
                    case kShaderInputUsagePtrInternalGlobalTable:
                    case kShaderInputUsagePtrExtendedUserData:
                    case kShaderInputUsagePtrIndirectResourceTable:
                    case kShaderInputUsagePtrIndirectInternalResourceTable:
                    case kShaderInputUsagePtrIndirectRwResourceTable:
                        break;

                    default:
                        printf("Warning: unhandled usage type in input slots");
                        break;
                }
                printf("\\---------------------------------------------------\n");

            }
            printf("//////////////////////////////////////////////////////////////////////////\n");
        }

        if (auto dumpPath = getGcnBinaryDumpPath()) {
            auto outfile = createDumpFile(dumpPath.value(), ".sb", FileNum);
            if (outfile) {
                outfile->write(reinterpret_cast<const char *>(fileHeader), fileHeader->m_codeSize + sizeof(ShaderFileHeader));
                if (outfile->has_error()) {
                    printf("Couldn't dump gcn bytecode to file %s\n", dumpPath.value().c_str());
                    return 1;
                }
            }
        }

        decodeAndConvertGcnCode(gcnBytecode, execModel, fileHeader->m_usesShaderResourceTable ? SrdMode::FlatTable : SrdMode::Immediate, srdSlots);
        ++FileNum;
    }

    return err;
}