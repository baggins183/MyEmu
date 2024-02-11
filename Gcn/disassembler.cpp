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
#include <iostream>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <sys/types.h>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <byteswap.h>
#include <arpa/inet.h>
#include <llvm/ADT/ArrayRef.h>

#include "llvm/Support/ManagedStatic.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"

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

bool isSgpr(uint regno) {
    return regno >= AMDGPU::SGPR0 && regno <= AMDGPU::SGPR105;
}

bool isSgprPair(uint regno) {
    return regno >= AMDGPU::SGPR0_SGPR1 && regno <= AMDGPU::SGPR104_SGPR105;
}

bool isVgpr(uint regno) {
    return regno >= AMDGPU::VGPR0 && regno <= AMDGPU::VGPR255;
}

inline uint sgprOffset(uint regno) {
    assert(isSgpr(regno));
    return regno - AMDGPU::SGPR0;
}

inline uint sgprPairOffset(uint regno) {
    assert(isSgprPair(regno));
    return regno - AMDGPU::SGPR0_SGPR1;
}

inline uint vgprOffset(uint regno) {
    assert(isVgpr(regno));
    return regno - AMDGPU::VGPR0;
}

inline uint sgprPairToSgprBase(uint regno) {
    assert(isSgprPair(regno));
    return AMDGPU::SGPR0 + 2 * sgprPairOffset(regno);
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

bool isSpecialCCReg(uint regno) {
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
    VGpr,
    SgprPair,
    Special // m0, VCC, etc
};

GcnRegType regnoToType(uint regno) {
    if (isSgpr(regno)) {
        return GcnRegType::SGpr;
    } else if (isSgprPair(regno)) {
        return GcnRegType::SgprPair;
    } else if (isVgpr(regno)) {
        return GcnRegType::VGpr;
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
MCInstPrinter *ProcIP = nullptr;
const MCSubtargetInfo *ProcSTI = nullptr;
extern "C" void  printInst(MCInst &MI) {
    llvm::outs().flush();
    ProcIP->printInst(&MI, 0, "", *ProcSTI, llvm::outs());
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
        float16Ty = builder.getF16Type();
        intTy = builder.getI32Type();
        int64Ty = builder.getI64Type();
        int16Ty = builder.getI16Type();
        boolTy = builder.getI1Type();

#if defined(_DEBUG)
        ProcIP = IP;
        ProcSTI = STI;
#endif
    }

    SpirvConvertResult convert();

private:
    bool prepass();
    bool convertGcnOp(const MCInst &MI);
    bool buildSpirvDialect();
    bool finalizeModule();
    bool generateSpirvBytecode();

    // build a VariableOp for the CC (condition code) register, and attach a name
    CCReg buildAndNameCCReg(llvm::StringRef name) {
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
    VariableOp buildAndName32BitReg(llvm::StringRef name) {
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
            VariableOp vgpr = buildAndName32BitReg(name);
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
            CCReg cc = buildAndNameCCReg(name);
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
            UniformReg uniform = buildAndName32BitReg(name);
            auto it = usedUniformRegs.insert(std::make_pair(regno, uniform));
            return it.first->second;
        } else {
            return it->second;
        }
    }

    VariableOp buildAndNameInOutAttribute(llvm::StringRef name, uint attrno, mlir::Type attrType, StorageClass storage) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto attrPtrTy = PointerType::get(attrType, storage);
        auto var = initBuilder.create<VariableOp>(attrPtrTy, StorageClassAttr::get(&mlirContext, storage), nullptr);
#if defined(_DEBUG)
        var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
        var->setAttr("Location", builder.getI32IntegerAttr(attrno));
        return var;
    }

    VariableOp initInputAttribute(uint attrno, mlir::Type attrType) {
        auto it = inAttributes.find(attrno);
        if (it == inAttributes.end()) {
            llvm::SmallString<16> name;
            llvm::Twine twine("inAttr_");
            twine.concat(Twine(attrno)).toStringRef(name);
            InputAttribute attr = buildAndNameInOutAttribute(name, attrno, attrType, StorageClass::Input);
            auto it = inAttributes.insert(std::make_pair(attrno, attr));
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
                    return builder.create<ConstantOp>(type, builder.getI32IntegerAttr(imm));     
                }
                case 64:
                {
                    // Not sure if operand comes sign extended or not, do it in case.
                    int64_t imm = operand.getImm();
                    return builder.create<ConstantOp>(type, builder.getI64IntegerAttr(imm));
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
                    return builder.create<ConstantOp>(type, builder.getF32FloatAttr(fImm));
                }
                case 64:
                {
                    uint64_t imm = operand.getImm();
                    // extend float by padding LSBs, just in case. Same if already padded or not.
                    double dImm = llvm::bit_cast<double>(imm | (imm << 32));
                    return builder.create<ConstantOp>(type, builder.getF64FloatAttr(dImm));
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
            assert(isSpecialCCReg(regno));
        }

        rv = sourceCCReg(regno);

        return rv;
    }

    mlir::Value sourceVectorOperand(MCOperand &operand, mlir::Type type) {
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

    void storeResultCC(MCOperand &dst, mlir::Value result) {
        assert(result.getType() == boolTy);

        uint regno = dst.getReg();
        assert(isSgprPair(regno) || isSpecialCCReg(regno));
        CCReg cc = initCCReg(regno);
        builder.create<StoreOp>(cc, result);
    }

    void storeVectorResult32(MCOperand &dst, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if (result.getType() != intTy) {
            result = builder.create<BitcastOp>(intTy, result);
        }

        assert(dst.isReg() && isVgpr(dst.getReg()));
        uint regno = dst.getReg();
        VectorReg vgpr = initVgpr(regno);
        auto store = builder.create<StoreOp>(vgpr, result);
        // Don't write back the result if thread is masked off by EXEC
        // Will need to generate control flow, TODO
        store->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getUnitAttr());
    }

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

    // In/Out attributes
    typedef VariableOp InputAttribute;
    // Maps attribute number (Gcn) to VariableOp for 'in' vertex attributes
    llvm::DenseMap<uint, InputAttribute> inAttributes;
    
    FuncOp mainFn;
    mlir::Block *mainEntryBlock;
    ModuleOp moduleOp;

    // spirv-val complains about duplicate int types, so mlir must be caching or handling signedness weird.
    // init once and reuse these instead
    mlir::FloatType floatTy;
    mlir::FloatType float16Ty;
    mlir::IntegerType intTy;
    mlir::IntegerType int64Ty;
    mlir::IntegerType int16Ty;
    mlir::IntegerType boolTy;

    const MCSubtargetInfo *STI;
    MCInstPrinter *IP;
};

bool GcnToSpirvConverter::prepass() {
    llvm::DenseMap<uint, uint> attrToNumComponents;

    // Do some things:
    // Figure out how many components each in/out attribute has (1 - 4)

    bool success = true;
    for (MCInst &MI: machineCode) {
        switch (MI.getOpcode()) {
            case AMDGPU::V_INTERP_P2_F32_si:
            {
                assert(execModel == ExecutionModel::Fragment);
                if (execModel != ExecutionModel::Fragment) {
                    return false;
                }
                uint attrno, component;
                attrno = MI.getOperand(3).getImm();
                component = MI.getOperand(4).getImm();
                auto it = attrToNumComponents.find(attrno);
                if (it != attrToNumComponents.end()) {
                    uint maxComponent = std::max(it->second, component);
                    it->getSecond() = maxComponent;
                } else {
                    attrToNumComponents[attrno] = component;
                }
                break;
            }
        }
    }

    // Initialize input attributes
    for (const auto &kv: attrToNumComponents) {
        uint attrno = kv.first;
        uint numComponents = kv.second + 1;
        mlir::Type attrType;
        if (numComponents > 1) {
            attrType = mlir::VectorType::get({numComponents}, floatTy);
        } else {
            attrType = floatTy;
        }

        initInputAttribute(attrno, attrType);
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
                auto constval0  = builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(0));
                auto constval32 = builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(32));

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
                assert(isSgprPair(srcno) || isSpecialCCReg(srcno));
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
        {
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);
            mlir::Type floatTy = builder.getF32Type();
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
                    auto numerator = ConstantOp::getOne(floatTy, builder.getLoc(), builder);
                    result = builder.create<FDivOp>(numerator, srcVal);
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
            auto scale = builder.create<ConstantOp>(floatTy, builder.getF32FloatAttr(0.0625f));
            auto bfOff = builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(0));
            auto bfCount = builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(4));
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
        case AMDGPU::V_MAC_F32_e32_gfx6_gfx7:
        case AMDGPU::V_CVT_PKRTZ_F16_F32_e32_gfx6_gfx7:
        {
            //assert(MI.getNumOperands() == 3);
            MCOperand dst, src0, vsrc1;
            dst = MI.getOperand(0);
            src0 = MI.getOperand(1);
            vsrc1 = MI.getOperand(2);
            //MCOperand src
            mlir::Type floatTy = builder.getF32Type();
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
                    // TODO use glsl.450 ext with packHalf2x16
                    auto bF16 = builder.create<FConvertOp>(float16Ty, b);
                    auto bI16 = builder.create<BitcastOp>(int16Ty, bF16);
                    result = builder.create<UConvertOp>(intTy, bI16);
                    auto shiftBy = builder.create<ConstantOp>(intTy, builder.getI32IntegerAttr(16));
                    result = builder.create<ShiftLeftLogicalOp>(result, shiftBy);
                    auto aF16 = builder.create<FConvertOp>(float16Ty, a);
                    auto aI16 = builder.create<BitcastOp>(int16Ty, aF16);
                    auto aI32 = builder.create<UConvertOp>(intTy, aI16);
                    result = builder.create<BitwiseOrOp>(result, aI32);

                    // TODO: spirv dialect doesn't support this? Need to add to binary?
                    // bF16->setAttr("FPRoundingMode", builder.getI32IntegerAttr(1));
                }

            }
            storeVectorResult32(dst, result);            
            break;
        }

        // VOP3a
        case AMDGPU::V_MED3_F32_gfx6_gfx7:
        {
            // TODO check NEG, OMOD, CLAMP, ABS
            MCOperand dst, src0, src1, src2;
            dst = MI.getOperand(0);
            src0 = MI.getOperand(1);
            src1 = MI.getOperand(2);   
            src2 = MI.getOperand(3);
            mlir::Type floatTy = builder.getF32Type();
            mlir::Value a = sourceVectorOperand(src0, floatTy);
            mlir::Value b = sourceVectorOperand(src1, floatTy);
            mlir::Value c = sourceVectorOperand(src2, floatTy);
            mlir::Value result;
            switch(MI.getOpcode()) {
                case AMDGPU::V_MED3_F32_gfx6_gfx7:
                {
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
                }
            }
            storeVectorResult32(dst, result);
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
            auto it = inAttributes.find(attrno);
            assert(it != inAttributes.end());
            InputAttribute attr = it->second;
            PointerType attrPtrType = cast<PointerType>(attr.getType());
            mlir::Type attrType = attrPtrType.getPointeeType();

            mlir::Value attrVal = builder.create<LoadOp>(attr);
            if (auto fvecTy = dyn_cast<mlir::VectorType>(attrType)) {
                int idx[] = { component };
                attrVal = builder.create<CompositeExtractOp>(attrVal, idx);
            }
            assert(dyn_cast<mlir::VectorType>(attrType) || component == 0);

            // TODO: check if this store should be predicated by EXEC
            // Pretty sure it is
            storeVectorResult32(dst, attrVal);
            break;
        }

        // The EXEC mask is applied to all exports.
        // Only pixels with the corresponding EXEC bit set to 1 export data to the output buffer.

        default:
            llvm::outs() << "Unhandled instruction: \n";
            IP->printInst(&MI, 0, "", *STI, llvm::outs());
            llvm::outs() << "\n";
            return false;
            break;
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
    llvm::SmallVector<mlir::Attribute, 4> interfaceVars;
    builder.create<EntryPointOp>(execModel, mainFn, interfaceVars);

    llvm::SmallVector<Capability, 8> caps = { Capability::Shader, Capability::Float16, Capability::Int16 };
    llvm::SmallVector<Extension, 4> exts;
    auto vceTriple = VerCapExtAttr::get(Version::V_1_5, caps, exts, &mlirContext);
    moduleOp->setAttr("vce_triple", vceTriple);

    //moduleOp.dump();
    std::error_code ec;
    fs::path path = "output.mlir";
    raw_fd_ostream mlirFile(path.c_str(), ec);
    if (ec) {
        llvm::outs() << "Couldn't open " << path << ": " << ec.message() << "\n";
        return false;
    }
    moduleOp->print(mlirFile);

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

// For hiding llvm command line options
cl::OptionCategory MyCategory("Compiler Options", "Options for controlling the compilation process.");

cl::opt<std::string> GcnBinaryDumpFilepath("dump-gcn-inputs", cl::desc("dump gcn binaries in the input file to N numbered files starting with given name"), cl::value_desc("filename"), cl::cat(MyCategory));
cl::opt<std::string> InputGcnFilepath(cl::Positional, cl::desc("<input file>"), cl::Required, cl::value_desc("filename"), cl::cat(MyCategory));

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
        IP->printInst(&MI, 0, "", *STI, llvm::outs());
        llvm::outs() << "\n";

        machineCode.push_back(std::move(MI));

        addr += size;
    }

    GcnToSpirvConverter converter(machineCode, execModel, STI.get(), IP);
    SpirvConvertResult result = converter.convert();
    SpirvBytecodeVector spirvModule = result.takeSpirvModule();

    fs::path spirvPath = fs::path(InputGcnFilepath.getValue()).filename().stem();
    spirvPath += ".spv";
    std::error_code ec;
    raw_fd_ostream spirvFile(spirvPath.c_str(), ec);
    if (ec) {
        llvm::outs() << "Couldn't open " << spirvPath << ": " << ec.message() << "\n";
        return 1;
    }
    spirvFile.write(reinterpret_cast<char *>(spirvModule.data()), spirvModule.size_in_bytes());

    llvm_shutdown();
    return result;
}

int main(int argc, char **argv) {
    int err = 0;
    std::vector<uint8_t> buf;

    cl::HideUnrelatedOptions(MyCategory);
    cl::ParseCommandLineOptions(argc, argv);

    FILE *gcnFile = fopen(InputGcnFilepath.c_str(), "r");
    if (!gcnFile) {
        printf("Couldn't open %s for reading\n", InputGcnFilepath.c_str());
        return 1;
    }
    fseek(gcnFile, 0, SEEK_END);
    size_t len = ftell(gcnFile);
    buf.resize(len);
    fseek(gcnFile, 0, SEEK_SET);
    if (1 != fread(buf.data(), len, 1, gcnFile)) {
        printf("Error reading from %s: %s\n", InputGcnFilepath.c_str(), strerror(errno));
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

        decodeAndConvertGcnCode(gcnBytecode, execModel);
    }

    return err;
}

