#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <fstream>
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

class GcnInstruction {
public:
    size_t length() const { return 0; }

private:
    llvm::ArrayRef<uint8_t> ref;
};

GcnInstruction *decodeInstruction(unsigned char *data) {
    uint32_t *instr;
    return nullptr;
}



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
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"


using namespace llvm;

int llvm_mc_main(int argc, char ** argv, const std::vector<unsigned char> &gcnBytecode) {
    //llvm::InitializeAllTargetInfos();
    //llvm::InitializeAllTargetMCs();

    //InitLLVM X(argc, argv);

    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();

    const char *tripleName = "amdgcn-unknown-linux-gnu";
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

    //llvm_shutdown();
    return 0;
}

int main(int argc, char **argv) {
    assert(argc > 0);

    fs::path binPath(argv[1]);
    bool search = false;

    fs::path outBinPath;
    for (uint i = 1; i < argc; i++) {
        if ( std::string(argv[i]) == "-o") { 
            assert(i + 1 < argc);
            outBinPath = argv[++i];
        } else if ( std::string(argv[i]) == "-s") { // search file for GCN. Dump to working directory
            search = true;
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
    bool fileIsLittleEndian = false;

    for (size_t i = 0; i <= len - strlen(magic); i++) {
        if ( !memcmp(&buf[i], magic, strlen(magic))) {
            printf("found OrbShdr, position 0x%zx\n", i);
            assert(i + sizeof(ShaderBinaryInfo) <= len);
            memcpy(&bininfo, &buf[i], sizeof(ShaderBinaryInfo));
            printf("\n");
            print_bininfo(bininfo);
            foundBinInfo = true;
        } else if ( *reinterpret_cast<uint32_t *>(&buf[i]) == 0xbeeb03ff) {
            // host is little endian. Reverse when we
            printf("found first instr: offset: 0x%zx\n", progStartOff);
            progStartOff = i;
            foundProgStart = true;
            fileIsLittleEndian = true;
        } else if ( *reinterpret_cast<uint32_t *>(&buf[i]) == 0xff03ebbe) {
            printf("found first instr: offset: 0x%zx\n", progStartOff);
            printf("was reversed on disk (big endian)\n");
            progStartOff = i;
            foundProgStart = true;
            fileIsLittleEndian = false;
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

    std::vector<unsigned char> programBinary(bininfo.m_length);

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

    memcpy(programBinary.data(), &buf[progStartOff], bininfo.m_length);
    llvm_mc_main(argc, argv, programBinary);

    if ( !outBinPath.empty()) {
        FILE *fOut = fopen(outBinPath.c_str(), "w");
        if (!fOut) {
            fprintf(stderr, "couldnt open out path\n");
            return -1;
        }

        assert(1 == fwrite(programBinary.data(), programBinary.size(), 1, fOut));
    }

    return 0;
}

