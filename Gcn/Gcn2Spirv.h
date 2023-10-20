//#include "llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.h"
#include "Common.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include <memory>

#define GET_INSTRINFO_ENUM 1
#include "AMDGPUGenInstrInfo.inc"
#include "SIDefines.h"

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
    virtual void visit(SpirvVisitor *visitor) = 0;
};

class SpirvLiteral : public SpirvOperand {
public:
    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); };

    llvm::SmallVector<uint32_t, 1> words;
private:
};

class SpirvType : public SpirvOperand { // Should only be constructed by SpirvBuilder as to not duplicate types
    SpirvType(spv::Op opcode, llvm::SmallVector<SpirvOperand *, 4> operands);

    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); }
    spv::Op getOpType();

private:
    spv::Op opcode;
    // SpirvOperand is the type but basically only literals and types are allowed
    llvm::SmallVector<SpirvOperand *, 4> operands;
    uint32_t hash;
};

class SpirvInstruction : public SpirvOperand {
public:
    SpirvInstruction(spv::Op opcode, SpirvType *resultType, llvm::SmallVector<SpirvOperand *, 4> operands):
        opcode(opcode),
        resultType(resultType),
        operands(operands)
        {}
    SpirvInstruction(spv::Op opcode, llvm::SmallVector<SpirvOperand *, 4> operands):
        SpirvInstruction(opcode, nullptr, operands)
        {}

    spv::Op getOpcode();
    const SpirvOperand *getOperand(uint idx);
    const SpirvType *getResultType();
    virtual void visit(SpirvVisitor *visitor) override { visitor->visit(this); }

private:
    spv::Op opcode;
    SpirvType *resultType;
    llvm::SmallVector<SpirvOperand *, 4> operands;
};

class SpirvFunction {
    
};

class SpirvBuilder {
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

    llvm::DenseSet<std::unique_ptr<SpirvType>> types;

    // TODO SpirvFunction type?
    std::vector<std::unique_ptr<SpirvFunction>> functions;

public:
    // Create everything but the functions and bodies
    bool createPreamble();
    const SpirvFunction *createFunction(llvm::StringRef name);
    void setInsertPoint(const SpirvFunction *function);
    void addCall(const SpirvFunction *function, llvm::SmallVector<SpirvOperand *, 4> args);
    void createOp(spv::Op opcode, llvm::SmallVector<SpirvOperand *, 4> args);
    void createOp(spv::Op opcode, const SpirvType *resultType, llvm::SmallVector<SpirvOperand *, 4> args);
    const SpirvInstruction *loadVGpr(uint regNo);
    const SpirvInstruction *storeVGpr(uint regNo, const SpirvInstruction *val);
    const SpirvInstruction *loadVCC();
    const SpirvInstruction *StoreVCC(SpirvOperand val);
    const SpirvInstruction *createBitcast(const SpirvType *type, const SpirvInstruction *from);

    // Should we preserve carry out behavior?
    // set flag (per instruction) to track if carry out is live/dead
    // Then after module create (no emit yet), run dataflow to decide whether to write to VCC/SGPR
    // Then emit code w/ checks when carry out register is live
    // Option 2: Create precise spirv instuctions (objects) including carry-out writes, then remove dead code manually
    //      -Do yourself
    //      -run spirv-opt
    //      -let driver do it
    // Probably optimize ourself because we'll have to emit weird code in case carry out is needed

    const SpirvType *getF32Type();
    const SpirvType *getF64Type();
    const SpirvType *getI32Type();
    const SpirvType *getI64Type();
    const SpirvType *getU32Type();
    const SpirvType *getU64Type();
    const SpirvType *getBoolType();

    SpirvOperand makeLiteral();
};