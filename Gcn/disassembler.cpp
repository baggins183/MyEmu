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
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <fstream>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <sys/types.h>
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
    INVALID
};

GcnRegType regnoToType(uint regno) {
    if (isSgpr(regno)) {
        return GcnRegType::SGpr;
    } else if (isVgpr(regno)) {
        return GcnRegType::VGpr;
    }
    return GcnRegType::INVALID;
}

typedef llvm::SmallVector<uint32_t, 0> SpirvBytecodeVector;

class SpirvConvertResult {
public:
    enum ConvertStatus {
        success,
        fail,
        empty,
    };

    operator bool() { return status != ConvertStatus::success; }

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
            mlir::IntegerType int32Ty = initBuilder.getI32Type();
            auto intPtrTy = PointerType::get(int32Ty, StorageClass::Function);
            auto initializer = initBuilder.create<ConstantOp>(int32Ty, initBuilder.getI32IntegerAttr(0));
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
            mlir::IntegerType boolTy = initBuilder.getI1Type();
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
            mlir::IntegerType int32Ty = initBuilder.getI32Type();
            auto intPtrTy = PointerType::get(int32Ty, StorageClass::Function);
            auto initializer = initBuilder.create<ConstantOp>(int32Ty, initBuilder.getI32IntegerAttr(0));
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

        // Not sure if sign extension already happens or if you need to take MCOperand::getImm() and extend
        // it yourself. Treat it like its unextended for now.
        if (auto intTy = dyn_cast<mlir::IntegerType>(type)) {
            switch (intTy.getWidth()) {
                case 32:
                {
                    if (intTy.isSigned()) {
                        int32_t imm = static_cast<int32_t>(operand.getImm() & 0xffffffff);
                        return builder.create<ConstantOp>(type, builder.getSI32IntegerAttr(imm));
                    } else {
                        uint32_t imm = static_cast<uint32_t>(operand.getImm() & 0xffffffff);
                        return builder.create<ConstantOp>(type, builder.getUI32IntegerAttr(imm));     
                    }
                }
                case 64:
                {
                    // Not sure if operand comes sign extended or not, do it in case.
                    uint64_t imm = operand.getImm();
                    if (intTy.isSigned()) {
                        if (imm & 0x80000000) {
                            imm = (0xffffffffULL << 32) | imm; 
                        }
                    } else {
                        // max 64 bit literal should be UINT32_MAX.
                        assert((imm & ~(0xffffffff)) == 0);
                    }
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
                    float imm = static_cast<float>(operand.getImm() & 0xffffffff);
                    return builder.create<ConstantOp>(type, builder.getF32FloatAttr(imm));
                }
                case 64:
                {
                    uint64_t imm = operand.getImm();
                    // extend float by padding LSBs, just in case. Same if already padded or not.
                    imm = static_cast<double>(imm | (imm << 32));
                    return builder.create<ConstantOp>(type, builder.getF64FloatAttr(imm));
                }
                default:
                    assert(false && "unhandled float width");
            }
        }
        assert(false && "unhandled immediate");
        return nullptr;
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
        if (auto intTy = dyn_cast<mlir::IntegerType>(type)) {
            if (intTy.getWidth() == 32 && intTy.isSigned()) {
                return rv;
            }
        }
        rv = builder.create<BitcastOp>(type, rv);
        return rv;
    }

    void storeVectorResult32(MCOperand &dst, mlir::Value result) {
        mlir::Type resultTy = result.getType();
        assert(resultTy.getIntOrFloatBitWidth() == 32);
        if ( !result.getType().isSignedInteger()) {
            result = builder.create<BitcastOp>(builder.getI32Type(), result);
        }

        assert(dst.isReg() && isVgpr(dst.getReg()));
        uint vgprno = vgprOffset(dst.getReg());
        VariableOp vgpr = initVgpr(vgprno);
        builder.create<StoreOp>(vgpr, result);
    }

    mlir::MLIRContext mlirContext;
    mlir::ImplicitLocOpBuilder builder;
    ExecutionModel shaderType;

    std::vector<MCInst> machineCode;
    SpirvBytecodeVector spirvModule;
    SpirvConvertResult::ConvertStatus status;
    llvm::DenseMap<uint, VariableOp> usedSgprsCC;
    llvm::DenseMap<uint, VariableOp> usedSgprsUniform;
    llvm::DenseMap<uint, VariableOp> usedVgprs;
    FuncOp mainFn;
    mlir::Block *mainEntryBlock;
    ModuleOp moduleOp;

    const MCSubtargetInfo *STI;
    MCInstPrinter *IP;
};

bool GcnToSpirvConverter::convertGcnOp(const MCInst &MI) {
    switch (MI.getOpcode()) {
        case AMDGPU::S_MOV_B32_gfx6_gfx7:
        {
            
            break;
        }
        case AMDGPU::V_MUL_F32_e32_gfx6_gfx7:
        {
            assert(MI.getNumOperands() == 3);
            MCOperand dst, src0, vsrc1;
            dst = MI.getOperand(0);
            src0 = MI.getOperand(1);
            vsrc1 = MI.getOperand(2);
            //MCOperand src
            mlir::Type floatTy = builder.getF32Type();
            auto a = sourceVectorOperand(src0, floatTy);
            auto b = sourceVectorOperand(vsrc1, floatTy);
            // overflow, side effects?
            auto mul = builder.create<FMulOp>(a, b);
            mul->setDiscardableAttr(builder.getStringAttr("gcn.predicated"), builder.getBoolAttr(true));
            dst = MI.getOperand(0);
            storeVectorResult32(dst, mul);
            
            break;
        }
        default:
            return false;
            break;
    }
    return true;
}

bool GcnToSpirvConverter::buildSpirvDialect() {
    bool err = true;
    // TODO set module-level stuff (entrypoint, etc)
    moduleOp = builder.create<ModuleOp>(AddressingModel::Logical, MemoryModel::Vulkan);
    mlir::Region &region = moduleOp.getBodyRegion();
    //mlir::Block &entryBlock = region.emplaceBlock();
    assert(region.hasOneBlock());
    builder.setInsertionPointToEnd(&(*region.begin()));

    // TODO: need function arguments for global vars?
    mlir::FunctionType mainFTy = mlir::FunctionType::get(&mlirContext, builder.getNoneType(), builder.getNoneType());
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
    
    return true;
    // return err; // TODO
}

bool GcnToSpirvConverter::finalizeModule() {
    mlir::Region &region = moduleOp.getBodyRegion();
    builder.setInsertionPointToEnd(&(*region.begin()));
    llvm::SmallVector<mlir::Attribute, 4> interfaceVars;
    builder.create<EntryPointOp>(shaderType, mainFn, interfaceVars);

    llvm::SmallVector<Capability, 8> caps = { Capability::Shader };
    llvm::SmallVector<Extension, 4> exts;
    auto vceTriple = VerCapExtAttr::get(Version::V_1_5, caps, exts, &mlirContext);
    moduleOp->setAttr("vce_triple", vceTriple);

    moduleOp.dump();

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

int llvm_mc_main(int argc, char ** argv, const std::vector<unsigned char> &gcnBytecode) {
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
    SpirvBytecodeVector spirvModule = result.takeSpirvModule();

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
            llvm_mc_main(argc, argv, programBinary);
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

