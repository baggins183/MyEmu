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
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <fstream>
#include <tuple>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <sstream>
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
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/MC/MCDisassembler/MCDisassembler.h"

#include "GcnBinary.h"

#include "AMDGPUGenInstrInfo_INSTRINFO.inc"
#include "AMDGPUGenRegisterInfo_REGINFO.inc"
#include "SIDefines.h"


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

cl::opt<std::string> InputGcnPath(cl::Positional, cl::desc("<input file>"), cl::Required, cl::value_desc("filename"), cl::cat(MyCategory));

std::optional<fs::path> getGcnBinaryDumpPath() { return !GcnBinaryDumpPath.empty() ? GcnBinaryDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getSpirvDumpPath() { return !SpirvDumpPath.empty() ? SpirvDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getSpirvAsmDumpPath() { return !SpirvAsmDumpPath.empty() ? SpirvAsmDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }
std::optional<fs::path> getMlirDumpPath() { return !MlirDumpPath.empty() ? MlirDumpPath.getValue() : (!DumpAllPath.empty() ? DumpAllPath.getValue() : std::optional<fs::path>() ); }

static int FileNum = 0;

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

typedef llvm::SmallVector<uint32_t, 0> SpirvBytecodeVector;

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
// We are modeling it in SPIR-V, which is basically SIMT and mostly hides that other threads exist.
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

class GcnToSpirvConverter {
public:
    GcnToSpirvConverter(const std::vector<MCInst> &machineCode, ExecutionModel execModel, const MCSubtargetInfo *STI, MCInstPrinter *IP):
        mlirContext(), 
        builder(mlir::UnknownLoc::get(&mlirContext), &mlirContext),
        execModel(execModel),
        machineCode(machineCode),
        status(SpirvConvertResult::empty),
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
        int64Ty = builder.getI64Type();
        int16Ty = builder.getI16Type();
        boolTy = builder.getI1Type();

        arithCarryTy = StructType::get({ intTy, intTy });

#if defined(_DEBUG)
        ThreadIP = IP;
        ThreadSTI = STI;
#endif
    }

    SpirvConvertResult convert();

private:
    bool prepass();
    bool convertGcnOp(const MCInst &MI);
    bool buildSpirvDialect();
    bool finalizeModule();
    bool generateSpirvBytecode();

    bool isVertexShader() { return execModel == ExecutionModel::Vertex; }
    bool isFragShader() { return execModel == ExecutionModel::Fragment; }

    ConstantOp getZero(mlir::Type ty) { return ConstantOp::getZero(ty, builder.getLoc(), builder); }
    ConstantOp getOne(mlir::Type ty) { return ConstantOp::getOne(ty, builder.getLoc(), builder); }
    ConstantOp getF32Const(float f) { return builder.create<ConstantOp>(floatTy, builder.getF32FloatAttr(f)); }
    ConstantOp getF64Const(double d) { return builder.create<ConstantOp>(f64Ty, builder.getF64FloatAttr(d)); }
    ConstantOp getI32Const(int n) { return builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(n)); }
    ConstantOp getI64Const(int n) { return builder.create<ConstantOp>(int64Ty, builder.getI64IntegerAttr(n)); }

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
        VariableOp cc = initCCReg(regno);
        return builder.create<LoadOp>(cc);
    }

    mlir::Value sourceUniformReg(uint regno) {
        VariableOp uni = initUniformReg(regno);
        return builder.create<LoadOp>(uni);
    }

    mlir::Value sourceVectorGpr(uint regno) {
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
                return create64BitValue(lo, hi);
            }
            case SgprPair:
            {
                uint base = sgprPairToSgprBase(regno);
                lo = sourceUniformReg(base);
                hi = sourceUniformReg(base + 1);
                return create64BitValue(lo, hi);
            }
            default:
                assert(false && "unhandled");
                return nullptr;
        }
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

    void storeResultCC(uint regno, mlir::Value result) {
        assert(isSgprPair(regno) || isSpecialRegPair(regno));
        CCReg cc = initCCReg(regno);
        builder.create<StoreOp>(cc, result);
    }

    void storeResultCC(MCOperand &dst, mlir::Value result) {
        assert(result.getType() == boolTy);
        uint regno = dst.getReg();
        storeResultCC(regno, result);
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
        store->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getUnitAttr());
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
                break;
        }
        return rv;
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

    mlir::MLIRContext mlirContext;
    mlir::ImplicitLocOpBuilder builder;
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
    mlir::IntegerType int64Ty;
    mlir::IntegerType int16Ty;
    mlir::IntegerType boolTy;

    // struct has 2 u32 members.
    // This is used for a lot of carry procducing int ops, i.e. V_ADD_I32 -> OpIAddCarry
    StructType arithCarryTy;

    const MCSubtargetInfo *STI;
    MCInstPrinter *IP;
};

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
    //llvm::SmallVector<mlir::NamedAttribute, 2> attrs 
            //= { mlir::NamedAttribute(builder.getStringAttr("gcn.predicated"), builder.getBoolAttr(true)) };
    //llvm::SmallVector<mlir::Value, 2> args;
    switch (MI.getOpcode()) {
        // SOP1
        case AMDGPU::S_MOV_B32_gfx6_gfx7:
        {
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);

            assert(dst.isReg());
            mlir::Value result = sourceScalarOperandUniform(src, intTy);
            // TODO try to move CC bits too somehow.
            storeScalarResult32(dst, result);
            break;
        }
        case AMDGPU::S_MOV_B64_gfx6_gfx7:
        {
            // Try to move uniform register and cc register values
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);

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
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);

            // What I've seen so far
            // Probably just implementation detail for pixel shaders and can ignore.
            // Probably gets helper quad invocations running on AMD hardware
            // for derivatives and stuff
            assert(dst.isReg() && dst.getReg() == AMDGPU::EXEC);
            assert(src.isReg() && src.getReg() == AMDGPU::EXEC);
            break;
        }

        // SOP2
        case AMDGPU::S_MUL_I32_gfx6_gfx7:
        {
            // D = S1 * S2. Low 32 bits of result.
            MCOperand dst, ssrc0, ssrc1;
            mlir::Value a, b;
            dst = MI.getOperand(0);
            ssrc0 = MI.getOperand(1);
            ssrc1 = MI.getOperand(2);
            a = sourceScalarOperandUniform(ssrc0, intTy);
            b = sourceScalarOperandUniform(ssrc1, intTy);
            auto result = builder.create<IMulOp>(a, b);
            storeScalarResult32(dst.getReg(), result);
            break;
        }

        // SOPP
        case AMDGPU::S_WAITCNT_gfx6_gfx7:
            // Used by AMD hardware to wait for long-latency instructions.
            // Should be able to ignore
            break;
        case AMDGPU::S_ENDPGM_gfx6_gfx7:
            // Do return here?
            break;

        // VOP1
        case AMDGPU::V_FLOOR_F32_e32_gfx6_gfx7:
        case AMDGPU::V_FRACT_F32_e32_gfx6_gfx7:
        case AMDGPU::V_RCP_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MOV_B32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_U32_F32_e32_gfx6_gfx7:
        {
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);
            mlir::Value srcVal = sourceVectorOperand(src, floatTy);
            mlir::Value result;
            switch (MI.getOpcode()) {
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
                {
                    auto numerator = getOne(floatTy);
                    result = builder.create<FDivOp>(numerator, srcVal);
                    break;
                }
                case AMDGPU::V_MOV_B32_e32_gfx6_gfx7:
                    result = srcVal;
                    break;
                case AMDGPU::V_CVT_U32_F32_e32_gfx6_gfx7:
                {
                    // Input is converted to an unsigned integer value using truncation. Positive float magnitudes
                    // too great to be represented by an unsigned integer float (unbiased exponent > 31) saturate
                    // to max_uint.
                    // Special number handling:
                    // -inf & NaN & 0 & -0 → 0
                    // Inf → max_uint
                    // D.u = (unsigned)S0.f
                    result = builder.create<ConvertFToUOp>(intTy, srcVal);
                    break;
                }
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
            
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);
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

        // VOP2
        case AMDGPU::V_ADD_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MUL_F32_e32_gfx6_gfx7:
        case AMDGPU::V_SUB_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MIN_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MAX_F32_e32_gfx6_gfx7:
        case AMDGPU::V_MAC_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_PKRTZ_F16_F32_e32_gfx6_gfx7:
        {
            //assert(MI.getNumOperands() == 3);
            MCOperand dst, src0, vsrc1;
            dst = MI.getOperand(0);
            src0 = MI.getOperand(1);
            vsrc1 = MI.getOperand(2);
            //MCOperand src
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            mlir::Value b = sourceVectorOperand(vsrc1, floatTy);
            // overflow, side effects?
            mlir::Value result;
            switch (MI.getOpcode()) {
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
                case AMDGPU::V_CVT_PKRTZ_F16_F32_e32_gfx6_gfx7:
                {
                    // decorate with FPRoundingMode?
                    // OpFConvert?
                    // OpQuantizeToF16?
                    result = builder.create<CompositeConstructOp>(v2float32Ty, std::array{a, b});
                    // Needs llvm/mlir patch to add this opcode
                    result = builder.create<GLPackHalf2x16Op>(intTy, result);
                    // TODO: spirv dialect doesn't support this? Need to change llvm?
                    // bF16->setAttr("FPRoundingMode", builder.getI32IntegerAttr(1));
                    break;
                }

            }
            storeVectorResult32(dst, result);            
            break;
        }

        case AMDGPU::V_ADD_I32_e32_gfx6_gfx7:
        case AMDGPU::V_SUB_I32_e32_gfx6_gfx7:
        {
            //assert(MI.getNumOperands() == 3);
            MCOperand dst, src0, vsrc1;
            dst = MI.getOperand(0);
            src0 = MI.getOperand(1);
            vsrc1 = MI.getOperand(2);
            //MCOperand src
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(vsrc1, intTy);
            // overflow, side effects?
            mlir::Value result;
            switch (MI.getOpcode()) {
                case AMDGPU::V_ADD_I32_e32_gfx6_gfx7:
                {
                    // "Unsigned integer add based on signed or unsigned integer components. Produces an
                    // unsigned carry out in VCC or a scalar register" <- VOP2 is always VCC
                    mlir::Value carryOut;
                    assert(MI.getNumOperands() == 3);
                    auto sumAndCarry = builder.create<IAddCarryOp>(a, b);
                    result = builder.create<CompositeExtractOp>(sumAndCarry, std::array{0});
                    carryOut = builder.create<CompositeExtractOp>(sumAndCarry, std::array{1});
                    // Can't cast i32 -> i1 with OpUConvert because mlir forces i1 -> OpTypeBool
                    carryOut = builder.create<INotEqualOp>(getZero(carryOut.getType()), carryOut);
                    storeVectorResult32(dst, result);
                    storeResultCC(AMDGPU::VCC, carryOut);
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
                    storeResultCC(AMDGPU::VCC, borrowOut);
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

            switch(MI.getOpcode()) {
                case AMDGPU::V_MUL_HI_U32_gfx6_gfx7:
                case AMDGPU::V_MUL_LO_U32_gfx6_gfx7:
                    product = builder.create<UMulExtendedOp>(a, b);
                    break;
                case AMDGPU::V_MUL_HI_I32_gfx6_gfx7:
                case AMDGPU::V_MUL_LO_I32_gfx6_gfx7:
                    product = builder.create<SMulExtendedOp>(a, b);
                    break;
            }

            switch(MI.getOpcode()) {
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


        // VOP3a
        case AMDGPU::V_MED3_F32_gfx6_gfx7:
        {
            // TODO OMOD, CLAMP
            auto [ dst, src0Mods, src0, src1Mods, src1, src2Mods, src2 ] = unwrapMI(MI, R0_6);
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            mlir::Value c = sourceVectorOperand(src2, floatTy);
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
            //cmp = builder.create<FOrdGreaterThanEqualOp>(a, b);
            max_a_b = builder.create<GLFMaxOp>(a, b);
            max_a_c = builder.create<GLFMaxOp>(a, c);
            max_b_c = builder.create<GLFMaxOp>(b, c);
            max3 = builder.create<GLFMaxOp>(max_a_b, max_a_c);

            cmp = builder.create<FOrdEqualOp>(max3, a);
            result = builder.create<SelectOp>(cmp, max_b_c, max_a_b);
            cmp = builder.create<FOrdEqualOp>(max3, b);
            result = builder.create<SelectOp>(cmp, max_a_c, result);

            break;
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
            break;
            storeVectorResult32(dst, result);
            break;
        }

        case AMDGPU::V_MAD_U64_U32_gfx7:
        {
            auto [ dst, cc, src0, src1, src2 ] = unwrapMI(MI, R0_4);
            assert(MI.getNumOperands() == 5);
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
            storeResultCC(cc, carry);
            storeVectorResult64(dst.getReg(), mad);
            break;
        }

        case AMDGPU::V_ADDC_U32_e64_gfx6_gfx7:
        {
            auto [ dst, cc, src0, src1, src2 ] = unwrapMI(MI, R0_4);
            mlir::Value a = sourceVectorOperand(src0, intTy);
            mlir::Value b = sourceVectorOperand(src1, intTy);
            mlir::Value carryIn, carryOut1, carryOut2;
            mlir::Value sumAndCarry1, sumAndCarry2;
            mlir::Value sum1, sum2;
            carryIn = sourceCCOperand(src2);
            carryIn = builder.create<SelectOp>(carryIn, getZero(intTy), getOne(intTy));
            sumAndCarry1 = builder.create<IAddCarryOp>(a, b);
            sum1 = builder.create<CompositeExtractOp>(sumAndCarry1, std::array{0});
            carryOut1 = builder.create<CompositeExtractOp>(sumAndCarry1, std::array{1});
            carryOut1 = builder.create<INotEqualOp>(getZero(carryOut1.getType()), carryOut1);
            sumAndCarry2 = builder.create<IAddCarryOp>(sum1, carryIn);
            sum2 = builder.create<CompositeExtractOp>(sumAndCarry2, std::array{0});
            carryOut2 = builder.create<CompositeExtractOp>(sumAndCarry2, std::array{1});
            carryOut2 = builder.create<INotEqualOp>(getZero(carryOut2.getType()), carryOut2);       
            storeResultCC(cc, carryOut2);
            storeVectorResult32(dst.getReg(), sum2);
            break;
        }

        case AMDGPU::V_MAD_F32_gfx6_gfx7:
        {
            // TODO OMOD, CLAMP
            auto [ dst, src0Mods, src0, src1Mods, src1, src2Mods, src2 ] = unwrapMI(MI, R0_6);
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);
            mlir::Value c = sourceVectorOperand(src2, floatTy);
            c = applyOperandModifiers(c, src2Mods);

            mlir::Value mad = builder.create<FMulOp>(a, b);
            mad = builder.create<FAddOp>(mad, c);
            storeVectorResult32(dst, mad);
            break;
        }

        case AMDGPU::V_MUL_F32_e64_gfx6_gfx7:
        {
            // TODO OMOD, CLAMP
            auto [ dst, src0Mods, src0, src1Mods, src1 ] = unwrapMI(MI, R0_4);
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            a = applyOperandModifiers(a, src0Mods);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            b = applyOperandModifiers(b, src1Mods);

            mlir::Value mul = builder.create<FMulOp>(a, b);
            // if ??? clamp mul
            // TODO
            mul = applyResultModifiers(mul, 0, 0);
            storeVectorResult32(dst, mul);
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
            MCOperand dst, src;
            int attrno, component;
            dst = MI.getOperand(0); // also operand 1 (I think)
            // src = MI.getOperand(2); // holds barycentric coord (I think): ignore and let Vulkan driver handle
            attrno = MI.getOperand(3).getImm();
            // Dunno if this needs to be an immediate
            component = MI.getOperand(4).getImm();

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

            // TODO: check if this store should be predicated by EXEC
            // Pretty sure it is
            storeVectorResult32(dst, attrVal);
            break;
        }


        case AMDGPU::EXP_DONE_si:
        case AMDGPU::EXP_si:
        {
            // The EXEC mask is applied to all exports.
            // Only pixels with the corresponding EXEC bit set to 1 export data to the output buffer.

            // exp mrtz v2, off, off, off
            // exp mrt0 v0, v0, v1, v1 done compr vm
            MCOperand vsrc0, vsrc1, vsrc2, vsrc3;
            uint target, done, compr, validMask;

            target = MI.getOperand(0).getImm();
            vsrc0 = MI.getOperand(1);
            vsrc1 = MI.getOperand(2);
            vsrc2 = MI.getOperand(3);
            vsrc3 = MI.getOperand(4);
            done = MI.getOperand(5).getImm();
            compr = MI.getOperand(6).getImm();
            validMask = MI.getOperand(7).getImm();

            assert( !(isFragShader() && !isRenderTarget(target)));
            if (target != AMDGPU::Exp::Target::ET_NULL) {
                // dunno what validMask actually is.
                // assuming is a writemask, but description of EXP instructions makes that sound wrong
                // Just treat it like a writemask for now
                assert(exportTargets.contains(target));
                ExportTarget exp = exportTargets[target];
                // exports are in 16 bit format.
                // vsrc0 and vsrc1 each contain 2 f16's
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
                                src = builder.create<CompositeInsertOp>(component, src, std::array{i});
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
                                src = builder.create<CompositeInsertOp>(component, src, std::array{i});
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
                store->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getUnitAttr());
            }
            break;
        }

        default:
            if (!Quiet) {
                llvm::errs() << "Unhandled instruction: \n";
                IP->printInst(&MI, 0, "", *STI, llvm::errs());
                llvm::errs() << "\n";
            }
            return false;
    }
    return true;
}

bool GcnToSpirvConverter::buildSpirvDialect() {
    bool success = true;
    moduleOp = builder.create<ModuleOp>(AddressingModel::Logical, MemoryModel::GLSL450);
    mlir::Region &region = moduleOp.getBodyRegion();
    //mlir::Block &entryBlock = region.emplaceBlock();
    assert(region.hasOneBlock());
    builder.setInsertionPointToEnd(&(*region.begin()));

    mlir::FunctionType mainFTy = mlir::FunctionType::get(&mlirContext, {}, builder.getNoneType());
    mainFn = builder.create<FuncOp>("main", mainFTy);

    assert(mainFn->getNumRegions() == 1);
    mlir::Region &mainRegion = mainFn->getRegion(0);
    mainEntryBlock = &mainRegion.emplaceBlock();
    builder.setInsertionPointToStart(mainEntryBlock);
    mlir::Block &mainBody = mainRegion.emplaceBlock();
    // Keep OpVariables, etc in separate block
    builder.create<BranchOp>(&mainBody);
    builder.setInsertionPointToStart(&mainBody);

    if ( !prepass()) {
        return false;
    }
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

    builder.create<EntryPointOp>(execModel, mainFn, interfaceVars);
    if (isFragShader()) {
        llvm::SmallVector<int, 1> lits;
        builder.create<ExecutionModeOp>(mainFn, ExecutionMode::OriginUpperLeft, lits);
    }

    llvm::SmallVector<Capability, 8> caps = { Capability::Shader, Capability::Float16, Capability::Int16, Capability::Int64 };
    llvm::SmallVector<Extension, 4> exts;
    auto vceTriple = VerCapExtAttr::get(Version::V_1_5, caps, exts, &mlirContext);
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

    auto res = mlir::spirv::serialize(moduleOp, spirvModule);
    return res.succeeded();
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

bool decodeAndConvertGcnCode(const std::vector<unsigned char> &gcnBytecode, ExecutionModel execModel) {
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

    GcnToSpirvConverter converter(machineCode, execModel, STI.get(), IP);
    SpirvConvertResult result = converter.convert();
    SpirvBytecodeVector spirvModule = result.takeSpirvModule();

    if (auto dumpPath = getSpirvDumpPath()) {
        auto outfile = createDumpFile(dumpPath.value(), ".spv", FileNum);
        if (outfile) {
            outfile->write(reinterpret_cast<const char *>(spirvModule.data()), spirvModule.size_in_bytes());
            if (outfile->has_error()) {
                printf("Couldn't dump spirv bytecode to file %s\n", dumpPath.value().c_str());
                return 1;
            }
        }
    }

    llvm_shutdown();
    return result;
}

int main(int argc, char **argv) {
    int err = 0;
    std::vector<uint8_t> buf;

    cl::HideUnrelatedOptions(MyCategory);
    cl::ParseCommandLineOptions(argc, argv);

    FILE *gcnFile = fopen(InputGcnPath.c_str(), "r");
    if (!gcnFile) {
        printf("Couldn't open %s for reading\n", InputGcnPath.c_str());
        return 1;
    }
    fseek(gcnFile, 0, SEEK_END);
    size_t len = ftell(gcnFile);
    buf.resize(len);
    fseek(gcnFile, 0, SEEK_SET);
    if (1 != fread(buf.data(), len, 1, gcnFile)) {
        printf("Error reading from %s: %s\n", InputGcnPath.c_str(), strerror(errno));
        return 1;
    }

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

        const ShaderFileHeader *fileHeader = reinterpret_cast<const ShaderFileHeader*>(&buf[shdrMagicOff]);
        const Header *binaryHeader = reinterpret_cast<const Header *>(fileHeader) - 1;
        const ShaderCommonData *shaderCommon = reinterpret_cast<const ShaderCommonData*>(fileHeader + 1);

        ShaderBinaryInfo *binaryInfo = (ShaderBinaryInfo *) &buf[infoOff];
        gcnBytecode.resize(binaryInfo->m_length);
        memcpy(gcnBytecode.data(), &buf[codeOff], gcnBytecode.size());

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

        if (auto dumpPath = getGcnBinaryDumpPath()) {
            auto outfile = createDumpFile(dumpPath.value(), ".sb", FileNum);
            if (outfile) {
                outfile->write(reinterpret_cast<const char *>(fileHeader), binaryHeader->m_codeSize);
                if (outfile->has_error()) {
                    printf("Couldn't dump gcn bytecode to file %s\n", dumpPath.value().c_str());
                    return 1;
                }
            }
        }

        decodeAndConvertGcnCode(gcnBytecode, execModel);
        ++FileNum;
    }

    return err;
}

