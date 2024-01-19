#pragma once

#include "SpirvType.h"
#include "SpirvInstruction.h"
#include "SpirvContext.h"
#include "spirv/spirv.hpp"
#include "llvm/ADT/StringRef.h"

struct SGprVars {
    SpirvVariable *uni = nullptr;
    // We treat an SGpr logically as storage for condition bits with cc.
    // Physically, using SGPR s20 as the source or destination for condition bits, means
    // the pair s20, s21 are used for 64 bits:64 threads
    SpirvVariable *cc = nullptr;
};

class SpirvBuilder {
public:
    SpirvBuilder(SpirvContext *spirvContext):
        context(spirvContext),
        SCC(nullptr),
        predicated(false)
    {}

    const SpirvFunction *createFunction(llvm::StringRef name);
    void setInsertPoint(const SpirvFunction *function);
    void addCall(const SpirvFunction *function, BumpVector args);
    void createOp(spv::Op opcode, BumpVector args);
    void createOp(spv::Op opcode, const SpirvType *resultType, BumpVector args);

    SpirvVariable *makeVariable(SpirvType *type, spv::StorageClass storage, llvm::StringRef name);
    SpirvLoad *makeLoad(SpirvInstruction *variable, uint32_t memoryOperand);
    SpirvStore *makeStore(SpirvInstruction *variable, SpirvInstruction *val, uint32_t memoryOperand);
    SpirvLoad *loadVGpr(uint regno);
    SpirvStore *storeVGpr(uint regno, SpirvInstruction *val);
    SpirvInstruction *loadSGprCC(uint regno);
    SpirvInstruction *storeSGprCC(uint regno, SpirvInstruction *val);
    SpirvInstruction *loadUniformGpr(uint regno);
    SpirvInstruction *storeUniformGpr(uint regno, SpirvInstruction *val);
    SpirvInstruction *loadVCC_CC();
    SpirvInstruction *StoreVCC_CC(SpirvInstruction *val);
    SpirvInstruction *makeBitcast(SpirvType *type, SpirvInstruction *from);

    // Should we preserve carry out behavior?
    // set flag (per instruction) to track if carry out is live/dead
    // Then after module create (no emit yet), run dataflow to decide whether to write to VCC/SGPR
    // Then emit code w/ checks when carry out register is live
    // Option 2: Create precise spirv instuctions (objects) including carry-out writes, then remove dead code manually
    //      -Do yourself
    //      -run spirv-opt
    //      -let driver do it
    // Probably optimize ourself because we'll have to emit weird code in case carry out is needed

    // ^ instead, do dataflow on MCInst's and map MCInst* -> scalar dst live, then generate spv

    SpirvFloatType *makeF32Type();
    SpirvFloatType *makeF64Type();
    SpirvIntType *makeI32Type();
    SpirvIntType *makeI64Type();
    SpirvIntType *makeU32Type();
    SpirvIntType *makeU64Type();
    SpirvBoolType *makeBoolType();
    SpirvStructType *makeStructType(const llvm::ArrayRef<SpirvType *> memberTypes);
    SpirvPointerType *makePointerType(spv::StorageClass storage, SpirvType *objectType);

    void setPredicated(bool val) { predicated = val; };

private:
    void initVGpr(uint regno);
    // Initialize the bool-type OpVariable representing this thread's bit in condition codes written to SGpr[regno]
    // and the uint32_t-type OpVariable representing uniform values written to SGpr[regno]
    void initSGpr(uint regno);
    void initVCC();
    void initEXEC();
    void initSCC();
    // TODO: store true to EXEC

    SpirvContext *context;
    llvm::DenseMap<unsigned int, SpirvVariable *> usedVgprs;

    llvm::DenseMap<unsigned int, SGprVars> usedSGprs;
    SGprVars VCC;
    SGprVars EXEC;
    SpirvVariable *SCC; // 1 bit per wavefront
    bool predicated; // set to true to create instructions with predicated=1
};

// Notes

// CMP SDST write : write to SGprCC bool
// SGpr to SGpr move : write to SGprCC and uniform  
// Scalar memory read : write to uniform u32
// ??? also write SGprCC = (1 << invocation_id & val)?

// TODO figure out how to deal with the ambiguous condition bits vs uniform problem with SGPRs.
// Assume the ps4 compiler doesn't do any weird stuff with casting uniforms to thread masks and vice versa?
// Maybe in compute shaders? Probably doesn't matter in graphics

// Assume a value used as a thread mask is always generated by comparison (As SDST, VCC), or copied from one
// of those registers (+ EXEC)
// Maybe special case 

// Simple: SGpr source maps to cc or uniform based on context
// SALU CMP: (loop test, uniform branch) probably source .uni
// SALU Bitwise ops: Perform op on .uni (source S0.uni ... SN.uni) and .cc (source S0.cc ... SN.cc).
// Other SALU ops: source .uni
// SRC in VALU ops: soure .uni
// SRC in addr calculation for memory: probably .uni (unless some sort of mask operand, dunno)

//
// SGpr dest should map to cc or uniform based on context:
// Bitwise ops: Perform op on .uni (source S0.uni ... SN.uni) and .cc (source S0.cc ... SN.cc).
// other SALU ops: perform op only on .uni
// Memory read: only .uni

// Sclar move to SGpr dest
// Source is const: move to .uni
//      Source is const 0 or -1: also move false or true to .cc.
// Source is SGpr: move SRC.uni to DST.uni. Move SRC.cc to DST.cc

// Can we look at when only one SGPR of a pair is used - would imply .uni is being used

// Assume a value used as a scalar is al

// S_MOV_* : copy