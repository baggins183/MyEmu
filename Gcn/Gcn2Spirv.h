#pragma once

//#include "llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.h"
#include "Common.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Support/Allocator.h"
#include <cstddef>
#include <memory>

#include "spirv.hpp"

class SpirvType;
class SpirvInstruction;
class SpirvLiteral;
class SpirvFunction;

class SpirvVisitor {
public:
    virtual void visit(SpirvType *type) = 0;
    virtual void visit(SpirvInstruction *instruction) = 0;
    virtual void visit(SpirvLiteral *literal) = 0;
    virtual void visit(SpirvFunction *function) = 0;

private:
};

class SpirvEmitter : public SpirvVisitor {
public:
    virtual void visit(SpirvType *type) override;
    virtual void visit(SpirvInstruction *instruction) override;
    virtual void visit(SpirvLiteral *literal) override;
    virtual void visit(SpirvFunction *function) override;

private:
};

class SpirvOperand {
public:
    virtual ~SpirvOperand() = default;

    virtual void visit(SpirvVisitor *visitor) = 0;

    virtual bool operator==(const SpirvOperand& RHS) const = 0;
    virtual bool equalsShallow(const SpirvOperand& RHS) const = 0;

    virtual unsigned hash() const = 0;
    virtual unsigned hashShallow() const = 0;
};

using OperandVec = llvm::SmallVector<SpirvOperand *, 4>;
using LiteralVec = llvm::SmallVector<uint32_t, 2>;

class SpirvLiteral : public SpirvOperand {
public:
    SpirvLiteral(LiteralVec words):
        words(words)
    {}

    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); };

    virtual bool operator==(const SpirvOperand& RHS) const override;
    virtual bool equalsShallow(const SpirvOperand& RHS) const override;

    virtual unsigned hash() const override;
    virtual unsigned hashShallow() const override;

    LiteralVec words;
};

namespace llvm {
template <> struct DenseMapInfo<SpirvLiteral *> {
    static constexpr uintptr_t Log2MaxAlign = 12;
    static inline SpirvLiteral* getEmptyKey() {
        uintptr_t Val = static_cast<uintptr_t>(-1);
        Val <<= Log2MaxAlign;
        return reinterpret_cast<SpirvLiteral*>(Val);
    }
    
    static inline SpirvLiteral* getTombstoneKey() {
        uintptr_t Val = static_cast<uintptr_t>(-2);
        Val <<= Log2MaxAlign;
        return reinterpret_cast<SpirvLiteral*>(Val);
    }

    static unsigned getHashValue(const SpirvLiteral *literal);
    static bool isEqual(const SpirvLiteral *LHS, const SpirvLiteral *RHS);
};
}

class SpirvType : public SpirvOperand { // Should only be constructed by SpirvBuilder as to not duplicate types
public:
    SpirvType(spv::Op opcode):
        opcode(opcode)
    {}
    SpirvType(spv::Op opcode, OperandVec operands): 
        opcode(opcode),
        operands(operands)
    {}

    spv::Op getOpcode() const { return opcode; };
    SpirvOperand *getOperand(uint i) const { return operands[i]; }
    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); }

    virtual bool operator==(const SpirvOperand& RHS) const override;
    virtual bool equalsShallow(const SpirvOperand& RHS) const override;

    virtual unsigned hash() const override;
    virtual unsigned hashShallow() const override;

private:
    spv::Op opcode;
    // SpirvOperand is the type but basically only literals and types are allowed
    OperandVec operands;
};

namespace llvm {
template <> struct DenseMapInfo<SpirvType *> {
    static constexpr uintptr_t Log2MaxAlign = 12;
    static inline SpirvType* getEmptyKey() {
        uintptr_t Val = static_cast<uintptr_t>(-1);
        Val <<= Log2MaxAlign;
        return reinterpret_cast<SpirvType*>(Val);
    }
    
    static inline SpirvType* getTombstoneKey() {
        uintptr_t Val = static_cast<uintptr_t>(-2);
        Val <<= Log2MaxAlign;
        return reinterpret_cast<SpirvType*>(Val);
    }

    static unsigned getHashValue(const SpirvType *type);
    static bool isEqual(const SpirvType *LHS, const SpirvType *RHS);
};
}

class SpirvInstruction : public SpirvOperand {
public:
    SpirvInstruction(spv::Op opcode, bool predicated, OperandVec operands, const SpirvType *resultType):
        opcode(opcode),
        predicated(predicated),
        resultType(resultType),
        operands(operands)
    {}

    SpirvInstruction(spv::Op opcode, bool predicated, OperandVec operands):
        SpirvInstruction(opcode, predicated, operands, nullptr)
    {}

    spv::Op getOpcode() const { return opcode; };
    bool isPredicated() const { return predicated; }
    const SpirvOperand *getOperand(uint i) const { return operands[i]; };
    const SpirvType *getResultType() const { return resultType; };
    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); }

    virtual bool operator==(const SpirvOperand& RHS) const override { return this == &RHS; };
    virtual bool equalsShallow(const SpirvOperand& RHS) const override { return this == &RHS; };

    virtual unsigned hash() const override { return (uint64_t) this; };
    virtual unsigned hashShallow() const override { return (uint64_t) this; };

private:
    spv::Op opcode;
    bool predicated; // true if this instruction conditionally executes based on EXEC mask
    const SpirvType *resultType;
    OperandVec operands;
};

class Terminator {
};

class SpirvBlock {
    std::vector<const SpirvInstruction *> instructions;
    llvm::SmallVector<SpirvBlock *, 2> predecessors;
    llvm::SmallVector<SpirvBlock *, 2> successors;

    const auto pred_begin() { return predecessors.begin(); }
    const auto pred_end() { return predecessors.end(); }

    const auto succ_begin() { return successors.begin(); }
    const auto succ_end() { return successors.end(); }
};

class SpirvFunction : public SpirvOperand {
    std::vector<const SpirvBlock *> blocks;
    // OpVariables
    std::vector<const SpirvInstruction *> declarations;
    SpirvBlock *entry;
};

struct SGprVars {
    SpirvInstruction *uni = nullptr;
    // We treat an SGpr logically as storage for condition bits with cc.
    // Physically, using SGPR s20 as the source or destination for condition bits, means
    // the pair s20, s21 are used for 64 bits:64 threads
    SpirvInstruction *cc = nullptr;
};

class SpirvContext {
public:
    SpirvContext() {}

    // Add Spirv structured control flow to check EXEC around vector instructions.
    // The builder should have been set to mark instructions as predicated when translating
    // VALU, VMEM, Export, LDS, and GDS.
    // Also try to reorder predicated instructions adjacent so if's can be fused
    void createPredicatedControlFlow();
    // Remove branches with compile-time constant conditions.
    // Especially cases where EXEC is known to be true
    // (Do SSA-CCP whatever, see spirv-opt)
    void eliminateDeadBranches();
    void elminateDeadFunctions();
    // Create everything before the functions and bodies
    bool createPreamble();
    std::vector<uint8_t> emitSpirvModule();

    SpirvLiteral *allocateLiteral(SpirvLiteral &&obj);
    SpirvType *allocateType(SpirvType &&obj);
    SpirvInstruction *allocateInstruction(SpirvInstruction &&obj);

    SpirvType *getOrInsertType(SpirvType &type);
    SpirvLiteral *makeLiteral(LiteralVec words);
    SpirvLiteral *makeLiteral(uint32_t scalar);

    void setAddressingModel(spv::AddressingModel addrModel) { addressingModel = addrModel; }
    void setMemoryModel(spv::MemoryModel memModel);


    std::vector<spv::Capability> capabilities;
    std::vector<std::string> extensions;

    spv::MemoryModel memoryModel;
    spv::AddressingModel addressingModel;

    // Entry point related fields
    spv::ExecutionModel executionModel; // shader stage type (V, F, ...)
    // entry point always "main"
    SpirvInstruction *entryPoint;
    // "Interface is a list of <id> of global OpVariable instructions"
    std::vector<SpirvInstruction *> interfaces;

    std::vector<SpirvInstruction *> executionModes;

    const SpirvFunction *mainFunction;

private:
    // TODO monolithic allocator for SpirvFunction/other types?
    // BumpPtrAllocator if we can make classes trivial to destruct (cant bcuz SmallVector)? 
    std::vector<std::unique_ptr<SpirvFunction>> functions;

    llvm::DenseSet<SpirvType *> typeSet;
    llvm::DenseSet<SpirvLiteral *> literalSet;

    llvm::SpecificBumpPtrAllocator<SpirvLiteral> literalAllocator;
    llvm::SpecificBumpPtrAllocator<SpirvType> typeAllocator;
    llvm::SpecificBumpPtrAllocator<SpirvInstruction> instructionAllocator;
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
    void addCall(const SpirvFunction *function, OperandVec args);
    void createOp(spv::Op opcode, OperandVec args);
    void createOp(spv::Op opcode, const SpirvType *resultType, OperandVec args);

    SpirvInstruction *makeVariable(SpirvType *objectType, spv::StorageClass storage, llvm::StringRef name);
    SpirvInstruction *makeLoad(SpirvInstruction *variable, uint32_t memoryOperand);
    SpirvInstruction *makeStore(SpirvInstruction *variable, SpirvInstruction *val, uint32_t memoryOperand);
    SpirvInstruction *loadVGpr(uint regno);
    SpirvInstruction *storeVGpr(uint regno, SpirvInstruction *val);
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

    SpirvType *makeF32Type();
    SpirvType *makeF64Type();
    SpirvType *makeI32Type();
    SpirvType *makeI64Type();
    SpirvType *makeU32Type();
    SpirvType *makeU64Type();
    SpirvType *makeBoolType();
    SpirvType *makeStructType(OperandVec memberTypes);
    SpirvType *makePointerType(spv::StorageClass storage, SpirvType *objectType);

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
    llvm::DenseMap<unsigned int, SpirvInstruction *> usedVgprs;

    llvm::DenseMap<unsigned int, SGprVars> usedSGprs;
    SGprVars VCC;
    SGprVars EXEC;
    const SpirvInstruction *SCC; // 1 bit per wavefront
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