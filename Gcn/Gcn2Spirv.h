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

class SpirvVisitor {
public:
    virtual void visit(SpirvType *type) = 0;
    virtual void visit(SpirvInstruction *instruction) = 0;
    virtual void visit(SpirvLiteral *literal) = 0;

private:
};

class SpirvEmitter : public SpirvVisitor {
public:
    virtual void visit(SpirvType *type) override;
    virtual void visit(SpirvInstruction *instruction) override;
    virtual void visit(SpirvLiteral *literal) override;

private:
};

class SpirvTypeEqualityVisitor : public SpirvVisitor {
public:
    virtual void visit(SpirvType *type) override;
    virtual void visit(SpirvInstruction *instruction) override;
    virtual void visit(SpirvLiteral *literal) override;

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

using OperandVec = llvm::SmallVector<const SpirvOperand *, 4>;
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
    const SpirvOperand *getOperand(uint i) const { return operands[i]; }
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
    SpirvInstruction(spv::Op opcode, const SpirvType *resultType, OperandVec operands):
        opcode(opcode),
        resultType(resultType),
        operands(operands)
    {}

    SpirvInstruction(spv::Op opcode, OperandVec operands):
        SpirvInstruction(opcode, nullptr, operands)
    {}

    spv::Op getOpcode() const { return opcode; };
    const SpirvOperand *getOperand(uint i) const { return operands[i]; };
    const SpirvType *getResultType() const { return resultType; };
    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); }

    virtual bool operator==(const SpirvOperand& RHS) const override { return this == &RHS; };
    virtual bool equalsShallow(const SpirvOperand& RHS) const override { return this == &RHS; };

    virtual unsigned hash() const override { return (uint64_t) this; };
    virtual unsigned hashShallow() const override { return (uint64_t) this; };

private:
    spv::Op opcode;
    const SpirvType *resultType;
    OperandVec operands;
};

class SpirvFunction {
    
};

class SpirvContext {
public:
    SpirvContext() {}

    SpirvLiteral *allocateLiteral(SpirvLiteral &&obj);
    SpirvType *allocateType(SpirvType &&obj);
    SpirvInstruction *allocateInstruction(SpirvInstruction &&obj);

    const SpirvType *getOrInsertType(SpirvType &type);
    SpirvLiteral *makeLiteral(LiteralVec words);
    SpirvLiteral *makeLiteral(uint32_t scalar);

private:
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
        VCC(nullptr)
    {}

private:
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

    // debug instructions
    // annotation/decorations (have decorations as attribute of OpVariable instead?)

    // TODO SpirvFunction type?
    std::vector<std::unique_ptr<SpirvFunction>> functions;

public:
    // Create everything but the functions and bodies
    bool createPreamble();
    const SpirvFunction *createFunction(llvm::StringRef name);
    void setInsertPoint(const SpirvFunction *function);
    void addCall(const SpirvFunction *function, OperandVec args);
    void createOp(spv::Op opcode, OperandVec args);
    void createOp(spv::Op opcode, const SpirvType *resultType, OperandVec args);

    const SpirvInstruction *makeLoad(const SpirvInstruction *variable, uint32_t memoryOperand);
    const SpirvInstruction *makeStore(const SpirvInstruction *variable, const SpirvInstruction *val, uint32_t memoryOperand);
    const SpirvInstruction *loadVGpr(uint regno);
    const SpirvInstruction *storeVGpr(uint regno, const SpirvInstruction *val);
    const SpirvInstruction *loadVCC();
    const SpirvInstruction *StoreVCC(SpirvInstruction *val);
    const SpirvInstruction *makeBitcast(const SpirvType *type, const SpirvInstruction *from);

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

    const SpirvType *makeF32Type();
    const SpirvType *makeF64Type();
    const SpirvType *makeI32Type();
    const SpirvType *makeI64Type();
    const SpirvType *makeU32Type();
    const SpirvType *makeU64Type();
    const SpirvType *makeBoolType();
    const SpirvType *makeStructType(OperandVec memberTypes);
    const SpirvType *makePointerType(spv::StorageClass storage, const SpirvType *objectType);

    const SpirvInstruction *makeVariable(const SpirvType *objectType, spv::StorageClass storage, llvm::StringRef name);

private:
    void initVGpr(uint regno);
    void initVCC();

    SpirvContext *context;
    llvm::DenseMap<unsigned int, const SpirvInstruction *> usedVgprs;
    const SpirvInstruction *VCC; // OpVariable
};