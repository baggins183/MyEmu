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
#include "llvm/ADT/bit.h"
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <fstream>
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

// Some taken from llvm-mc


struct ShaderBinaryInfo
{
	uint8_t			m_signature[7];				// 'OrbShdr'
	uint8_t			m_version;					// ShaderBinaryInfoVersion

	unsigned int	m_pssl_or_cg : 1;	// 1 = PSSL / Cg, 0 = IL / shtb
	unsigned int	m_cached : 1;	// 1 = when compile, debugging source was cached.  May only make sense for PSSL=1
	uint32_t		m_type : 4;	// See enum ShaderBinaryType
	uint32_t		m_source_type : 2;	// See enum ShaderSourceType
	unsigned int	m_length : 24;	// Binary code length (does not include this structure or any of its preceding associated tables)

	uint8_t			m_chunkUsageBaseOffsetInDW;			// in DW, which starts at ((uint32_t*)&ShaderBinaryInfo) - m_chunkUsageBaseOffsetInDW; max is currently 7 dwords (128 T# + 32 V# + 20 CB V# + 16 UAV T#/V#)
	uint8_t			m_numInputUsageSlots;				// Up to 16 user data reg slots + 128 extended user data dwords supported by CUE; up to 16 user data reg slots + 240 extended user data dwords supported by InputUsageSlot
	uint8_t         m_isSrt : 1;	// 1 if this shader uses shader resource tables and has an SrtDef table embedded below the input usage table and any extended usage info
	uint8_t         m_isSrtUsedInfoValid : 1;	// 1 if SrtDef::m_isUsed=0 indicates an element is definitely unused; 0 if SrtDef::m_isUsed=0 indicates only that the element is not known to be used (m_isUsed=1 always indicates a resource is known to be used)
	uint8_t         m_isExtendedUsageInfo : 1;	// 1 if this shader has extended usage info for the InputUsage table embedded below the input usage table
	uint8_t         m_reserved2 : 5;	// For future use
	uint8_t         m_reserved3;						// For future use

	uint32_t		m_shaderHash0;				// Association hash first 4 bytes
	uint32_t		m_shaderHash1;				// Association hash second 4 bytes
	uint32_t		m_crc32;					// crc32 of shader + this struct, just up till this field
};

void print_bininfo(const ShaderBinaryInfo &bininfo) {
    uint32_t version = bininfo.m_version;
    uint32_t pssl_or_cg = bininfo.m_pssl_or_cg;
    uint32_t cached = bininfo.m_cached;
    uint32_t type = bininfo.m_type;
    uint32_t source_type = bininfo.m_source_type;
    uint32_t length = bininfo.m_length;
    uint32_t chunkUsageBaseOffsetInDW = bininfo.m_chunkUsageBaseOffsetInDW;
    uint32_t numInputUsageSlots = bininfo.m_numInputUsageSlots;
    uint32_t isSrt = bininfo.m_isSrt;
    uint32_t isSrtUsedInfoValid = bininfo.m_isSrtUsedInfoValid;
    uint32_t isExtendedUsageInfo = bininfo.m_isExtendedUsageInfo;
    uint32_t reserved2 = bininfo.m_reserved2;
    uint32_t reserved3 = bininfo.m_reserved3;
    uint32_t shaderHash0 = bininfo.m_shaderHash0;
    uint32_t shaderHash1 = bininfo.m_shaderHash1;
    uint32_t crc32 = bininfo.m_crc32;

    printf(
		"   m_signature: %.*s\n"
		"   m_version: %x\n"
	    "   m_pssl_or_cg: %x\n"
	    "   m_cached: %x\n"
		"   m_type: %x\n"
		"   m_source_type: %x\n"
	    "   m_length: %i\n"
		"   m_chunkUsageBaseOffsetInDW: %x\n"
		"   m_numInputUsageSlots: %x\n"
        "   m_isSrt: %x\n"
        "   m_isSrtUsedInfoValid: %x\n"
        "   m_isExtendedUsageInfo: %x\n"
        "   m_reserved2: %x\n"
        "   m_reserved3: %x\n"
        "   m_shaderHash0: %x\n"
        "   m_shaderHash1: %x\n"
        "   m_crc32: %x\n",
        7, bininfo.m_signature,
        version,
        pssl_or_cg,
        cached,
        type,
        source_type,
        length,
        chunkUsageBaseOffsetInDW,
        numInputUsageSlots,
        isSrt,
        isSrtUsedInfoValid,
        isExtendedUsageInfo,
        reserved2,
        reserved3,
        shaderHash0,
        shaderHash1,
        crc32
    );
}


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

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

//#include "llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"


#include "AMDGPUGenInstrInfo_INSTRINFO.inc"
#include "AMDGPUGenRegisterInfo_REGINFO.inc"
#include "SIDefines.h"


using namespace llvm;
using namespace mlir::spirv;

bool isSgpr(uint regno) {
    return regno >= AMDGPU::SGPR0 && regno <= AMDGPU::SGPR105;
}

bool isVgpr(uint regno) {
    return regno >= AMDGPU::VGPR0 && regno <= AMDGPU::VGPR255;
}

inline uint sgprOffset(uint regno) {
    assert(isSgpr(regno));
    return regno - AMDGPU::SGPR0;
}

inline uint vgprOffset(uint regno) {
    assert(isVgpr(regno));
    return regno - AMDGPU::VGPR0;
}

enum GcnRegType {
    SGpr,
    VGpr,
    EXEC,
    VCC,
    m0,
    INVALID
};

GcnRegType regnoToType(uint regno) {
    if (isSgpr(regno)) {
        return GcnRegType::SGpr;
    } else if (isVgpr(regno)) {
        return GcnRegType::VGpr;
    }
    switch (regno) {
        case AMDGPU::EXEC:
        case AMDGPU::EXEC_LO:
        case AMDGPU::EXEC_HI:
            return GcnRegType::EXEC;
        case AMDGPU::VCC:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
            return GcnRegType::VCC;
        case AMDGPU::M0:
            return GcnRegType::m0;
        default:
            return GcnRegType::INVALID;
    }
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

typedef VariableOp UniformReg;
typedef VariableOp CCReg;

struct ScalarRegUnion {
    UniformReg uniform;
    CCReg cc;
};

class GcnToSpirvConverter {
public:
    GcnToSpirvConverter(const std::vector<MCInst> &machineCode, ExecutionModel shaderType, const MCSubtargetInfo *STI, MCInstPrinter *IP):
        mlirContext(), 
        builder(mlir::UnknownLoc::get(&mlirContext), &mlirContext),
        shaderType(shaderType),
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
        int16Ty = builder.getI16Type();
        boolTy = builder.getI1Type();
    }

    SpirvConvertResult convert();

private:
    bool convertGcnOp(const MCInst &MI);
    bool buildSpirvDialect();
    bool finalizeModule();
    bool generateSpirvBytecode();

    VariableOp initVgpr(uint vgprno) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto it = usedVgprs.find(vgprno);
        if (it == usedVgprs.end()) {
            auto intPtrTy = PointerType::get(intTy, StorageClass::Function);
            auto initializer = initBuilder.create<ConstantOp>(intTy, initBuilder.getI32IntegerAttr(0));
            auto var = initBuilder.create<VariableOp>(intPtrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), initializer);
#if defined(_DEBUG)
            llvm::SmallString<8> name("v_");
            name += std::to_string(vgprno);
            var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
            auto it = usedVgprs.insert(std::make_pair(vgprno, var));
            return it.first->second;
        } else {
            return it->second;
        }
    }
    VariableOp initSgprCC(uint sgprno) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto it = usedSgprsCC.find(sgprno);
        if (it == usedSgprsCC.end()) {
            auto boolPtrTy = PointerType::get(boolTy, StorageClass::Function);
            auto initializer = initBuilder.create<ConstantOp>(boolTy, initBuilder.getIntegerAttr(boolTy, 0));
            auto var = initBuilder.create<VariableOp>(boolPtrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), initializer);
#if defined(_DEBUG)
            llvm::SmallString<8> name("scc_");
            name += std::to_string(sgprno);
            var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
            auto it = usedSgprsCC.insert(std::make_pair(sgprno, var));
            return it.first->second;
        } else {
            return it->second;
        }
    }
    VariableOp initSgprUniform(uint sgprno) {
        mlir::ImplicitLocOpBuilder initBuilder(builder);
        initBuilder.setInsertionPointToStart(mainEntryBlock);
        auto it = usedSgprsUniform.find(sgprno);
        if (it == usedSgprsUniform.end()) {
            auto intPtrTy = PointerType::get(intTy, StorageClass::Function);
            auto initializer = initBuilder.create<ConstantOp>(intTy, initBuilder.getI32IntegerAttr(0));
            auto var = initBuilder.create<VariableOp>(intPtrTy, StorageClassAttr::get(&mlirContext, StorageClass::Function), initializer);
#if defined(_DEBUG)
            llvm::SmallString<8> name("s_");
            name += std::to_string(sgprno);
            var->setDiscardableAttr(builder.getStringAttr("gcn.varname"), builder.getStringAttr(name));
#endif
            auto it = usedSgprsUniform.insert(std::make_pair(sgprno, var));
            return it.first->second;
        } else {
            return it->second;
        }
    }

    mlir::Value sourceScalarGprCC(uint sgprno) {
        VariableOp cc = initSgprCC(sgprno);
        return builder.create<LoadOp>(cc);
    }

    mlir::Value sourceScalarGprUniform(uint sgprno) {
        VariableOp uni = initSgprUniform(sgprno);
        return builder.create<LoadOp>(uni);
    }

    mlir::Value sourceVectorGpr(uint vgprno) {
        VariableOp vgpr = initVgpr(vgprno);
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
        switch (regnoToType(regno)) {
            case SGpr:
                rv =  sourceScalarGprUniform(sgprOffset(regno));
                break;
            default:
                assert(false && "unhandled");
        }
        if (type != intTy) {
            rv = builder.create<BitcastOp>(type, rv);
        }
        return rv;
    }

    mlir::Value sourceScalarOperandCC(MCOperand &operand) {
        mlir::Value rv;
        if (operand.isImm()) {
            // Only support this in compute (TODO)
            // return bool(laneID & imm)
            assert(false && "Unhandled: condition code source can't be an immediate");
        }

        assert(operand.isReg());
        uint regno = operand.getReg();
        switch (regnoToType(regno)) {
            case SGpr:
                rv =  sourceScalarGprCC(sgprOffset(regno));
                break;
            default:
                assert(false && "unhandled");
                return nullptr;
        }

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
            case SGpr:
                rv =  sourceScalarGprUniform(sgprOffset(regno));
                break;
            case VGpr:
                rv = sourceVectorGpr(vgprOffset(regno));
                break;
            default:
                assert(false && "unhandled");
                return nullptr;
        }
        if (type != intTy) {
            rv = builder.create<BitcastOp>(type, rv);
        }
        return rv;
    }

    void storeScalarResult32(MCOperand &dst, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if (result.getType() != intTy) {
            result = builder.create<BitcastOp>(intTy, result);
        }

        // TODO handle special sgprs (EXEC, VCC, etc)
        // Are all special sgprs 64 bit pairs? - then do nothing
        assert(dst.isReg() && isSgpr(dst.getReg()));
        uint sgprno = sgprOffset(dst.getReg());
        VariableOp sgpr = initSgprUniform(sgprno);
        builder.create<StoreOp>(sgpr, result);
    }

    void storeResultCC(MCOperand &dst, mlir::Value result) {
        assert(result.getType() == boolTy);

        // TODO handle special sgprs (EXEC, VCC, etc)
        uint sgprno = sgprOffset(dst.getReg());
        VariableOp sgpr = initSgprCC(sgprno);
        builder.create<StoreOp>(sgpr, result);
    }

    void storeVectorResult32(MCOperand &dst, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if (result.getType() != intTy) {
            result = builder.create<BitcastOp>(intTy, result);
        }

        assert(dst.isReg() && isVgpr(dst.getReg()));
        uint vgprno = vgprOffset(dst.getReg());
        VariableOp vgpr = initVgpr(vgprno);
        auto store = builder.create<StoreOp>(vgpr, result);
        // Don't write back the result if thread is masked off by EXEC
        // Will need to generate control flow, TODO
        store->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getUnitAttr());
    }

    mlir::MLIRContext mlirContext;
    mlir::ImplicitLocOpBuilder builder;
    ExecutionModel shaderType;

    std::vector<MCInst> machineCode;
    SpirvBytecodeVector spirvModule;
    SpirvConvertResult::ConvertStatus status;

    llvm::DenseMap<uint, VariableOp> usedVgprs;
    llvm::DenseMap<uint, VariableOp> usedSgprsCC;
    llvm::DenseMap<uint, VariableOp> usedSgprsUniform;

    ScalarRegUnion VCCVars;
    CCReg execVar = nullptr; // OpTypeBool only
    
    FuncOp mainFn;
    mlir::Block *mainEntryBlock;
    ModuleOp moduleOp;

    // spirv-val complains about duplicate int types, so mlir must be caching or handling signedness weird.
    // init once and reuse these instead
    mlir::FloatType floatTy;
    mlir::FloatType float16Ty;
    mlir::IntegerType intTy;
    mlir::IntegerType int16Ty;
    mlir::IntegerType boolTy;

    const MCSubtargetInfo *STI;
    MCInstPrinter *IP;
};

bool GcnToSpirvConverter::convertGcnOp(const MCInst &MI) {
    //llvm::SmallVector<mlir::NamedAttribute, 2> attrs 
            //= { mlir::NamedAttribute(builder.getStringAttr("gcn.predicated"), builder.getBoolAttr(true)) };
    //llvm::SmallVector<mlir::Value, 2> args;
    switch (MI.getOpcode()) {
        //case AMDGPU::S_MOV_B32_gfx6_gfx7:
        {
            // For MOV_32 it probably doesn't make sense to copy the CC bits over.
            // The ps4 compiler would probably use MOV_64 if it wanted to move a CC result,
            // because CC results go to an sgpr pair (64 bits)
            // Copy the CC bits as well, and ignore odd numbered sgprs.
            MCOperand dst, src;
            dst = MI.getOperand(0);
            src = MI.getOperand(1);

            GcnRegType dstRegType = regnoToType(dst.getReg());
            bool skipWrite = false;
            

            mlir::Value uni = sourceScalarOperandUniform(src, intTy);
            storeScalarResult32(dst, uni);
            if (src.isReg()) {
                assert(isSgpr(src.getReg()));
                uint sgprno = sgprOffset(src.getReg());
                // Ignore odd numbered : TODO explain
                if ((sgprno & 1) == 0) {
                    mlir::Value srcCC = sourceScalarOperandCC(src);
                    storeResultCC(dst, srcCC);
                }
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
    bool err = true;
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

    for (const MCInst &MI: machineCode) {
        err &= convertGcnOp(MI);
    }
    builder.create<ReturnOp>();
    
    return true;
    // return err; // TODO
}

bool GcnToSpirvConverter::finalizeModule() {
    mlir::Region &region = moduleOp.getBodyRegion();
    builder.setInsertionPointToEnd(&(*region.begin()));
    llvm::SmallVector<mlir::Attribute, 4> interfaceVars;
    builder.create<EntryPointOp>(shaderType, mainFn, interfaceVars);

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

int llvm_mc_main(int argc, char ** argv, const std::vector<unsigned char> &gcnBytecode, fs::path binPath) {
    //llvm::InitializeAllTargetInfos();
    //llvm::InitializeAllTargetMCs();

    //InitLLVM X(argc, argv);

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

    //builder.create<mlir::spirv::BitcastOp>();

    llvm::outs() << "\n";
    std::vector<MCInst> machineCode;
    while (addr < byteView.size()) {
        MCInst MI;
        MCDisassembler::DecodeStatus Res = DisAsm->getInstruction(MI, size, byteView.slice(addr), addr, llvm::nulls());

        switch (Res) {
            case MCDisassembler::DecodeStatus::Success:
                break;
            case MCDisassembler::DecodeStatus::Fail:
            case MCDisassembler::DecodeStatus::SoftFail:
                printf("Failed decoding instruction\n");
                return 1;
        }
        IP->printInst(&MI, 0, "", *STI, llvm::outs());
        llvm::outs() << "\n";

        switch (MI.getOpcode()) {
            case AMDGPU::S_MOV_B32_gfx6_gfx7:
                break;
            default:
                break;
        }

        machineCode.push_back(std::move(MI));

        addr += size;
    }

    ExecutionModel execModel = ExecutionModel::Vertex; // TODO
    GcnToSpirvConverter converter(machineCode, execModel, STI.get(), IP);
    SpirvConvertResult result = converter.convert();
    assert(result);
    SpirvBytecodeVector spirvModule = result.takeSpirvModule();

    fs::path spirvPath = binPath.filename().stem();
    spirvPath += ".spv";
    std::error_code ec;
    raw_fd_ostream spirvFile(spirvPath.c_str(), ec);
    if (ec) {
        llvm::outs() << "Couldn't open " << spirvPath << ": " << ec.message() << "\n";
        return 1;
    }
    spirvFile.write(reinterpret_cast<char *>(spirvModule.data()), spirvModule.size_in_bytes());

    llvm_shutdown();
    return result ? 0 : 1;
}

int main(int argc, char **argv) {
    assert(argc > 0);

    fs::path binPath(argv[1]);
    //bool search = false;

    fs::path outBinPath;
    for (int i = 1; i < argc; i++) {
        if ( std::string(argv[i]) == "-o") { 
            assert(i + 1 < argc);
            outBinPath = argv[++i];
        } else if ( std::string(argv[i]) == "-s") { // search file for GCN. Dump to working directory
            //search = true;
        } else {
            binPath = argv[i];
        }
    }

    if (binPath.empty()) {
        fprintf(stderr, "input binary path not given\n");
        return -1;
    }

    std::vector<unsigned char> buf;
    FILE *f = fopen(binPath.c_str(), "r");
    assert(f);
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);

    buf.resize(len);
    assert(1 == fread(buf.data(), len, 1, f));

    ShaderBinaryInfo bininfo;
    const char *magic = "OrbShdr";

    size_t progStartOff = 0;
    bool foundProgStart = false;
    bool foundBinInfo = false;

    for (size_t i = 0; i <= len - strlen(magic); i++) {
        if ( !memcmp(&buf[i], magic, strlen(magic))) {
            printf("found OrbShdr, position 0x%zx\n", i);
            assert(i + sizeof(ShaderBinaryInfo) <= len);
            memcpy(&bininfo, &buf[i], sizeof(ShaderBinaryInfo));
            printf("\n");
            print_bininfo(bininfo);
            foundBinInfo = true;
            assert(foundProgStart);
            std::vector<unsigned char> programBinary(bininfo.m_length);
            memcpy(programBinary.data(), &buf[progStartOff], bininfo.m_length);
            llvm_mc_main(argc, argv, programBinary, binPath);
        } else if ( *reinterpret_cast<uint32_t *>(&buf[i]) == 0xbeeb03ff) {
            // host is little endian. Reverse when we
            printf("found first instr: offset: 0x%zx\n", progStartOff);
            progStartOff = i;
            foundProgStart = true;
        } else if ( *reinterpret_cast<uint32_t *>(&buf[i]) == 0xff03ebbe) { // s_mov_b32 vcc_hi, #imm
            printf("found first instr: offset: 0x%zx\n", progStartOff);
            printf("was reversed on disk (big endian)\n");
            progStartOff = i;
            foundProgStart = true;
        }

        if (foundProgStart) {
            
        }
    }

    if ( !foundProgStart) {
        printf("couldn't find first instruction\n");
        return 1;
    }

    if ( !foundBinInfo) {
        printf("couldn't find bininfo\n");
        return -1;
    }


//    size_t bytesRead = 0;
//    while (bytesRead < bininfo.m_length) {
//        uint32_t instr;
//
//        instr = *reinterpret_cast<uint32_t *>(&buf[progStartOff + bytesRead]);
//        if ( !fileIsLittleEndian) {
//            instr = bswap_32(instr);
//        }
//
//        memcpy(programBinary.data() + bytesRead, &instr, sizeof(instr));
//        uint32_t temp = htonl(instr);
//        memcpy(outData.data() + bytesRead, &temp, sizeof(temp));
//
//        bytesRead += 4;
//    }

    return 0;
}

