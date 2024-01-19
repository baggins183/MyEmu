#pragma once

#include "SpirvCommon.h"
#include "SpirvType.h"
#include "spirv.hpp"
#include "llvm/Support/Allocator.h"

class SpirvInstruction;
class SpirvBasicInstruction;
class SpirvBitcast;
class SpirvBinMath;
class SpirvCmp;
class SpirvBitwise;
class SpirvRegRead;
class SpirvRegWrite;
class SpirvBufferRead;
class SpirvBufferWrite;
class SpirvImageRead;
class SpirvImageWrite;
class SpirvTexSamp;

class SpirvVariable;
class SpirvLoad;
class SpirvStore;

class SpirvBlock;
class SpirvFunction;

typedef std::vector<SpirvInstruction *> BumpVector;

class SpirvInstruction {
public:
    bool isPredicated();

    SpirvInstruction(bool predicated):
        predicated(predicated)
        {}
    SpirvInstruction(): SpirvInstruction(false)
        {}

protected:
    bool predicated;
};

class SpirvBasicInstruction : public SpirvInstruction {

private:
    BumpVector operands;
};

struct GcnRegister {
    enum Kind {
            Vgpr,
            Sgpr, // Read/write according to context
            SgprCC,
            SgprUni,
            VCC,
            Exec,
    } kind;

    int regno;

    GcnRegister(GcnRegister::Kind kind, uint regno): kind(kind), regno(regno)
        {}
    GcnRegister(GcnRegister::Kind kind): kind(kind), regno(-1)
        {}
};

class SpirvRegRead : public SpirvInstruction {
public:
    SpirvRegRead(GcnRegister target, bool predicated):
        SpirvInstruction(predicated), target(target)
        {}


private:
    GcnRegister target;
};

// VOPC or VOP3
class SpirvCmp : public SpirvInstruction {
    SpirvCmp(SpirvInstruction *arg0, SpirvInstruction *arg1, bool predicated):
        SpirvInstruction(predicated), arg0(arg0), arg1(arg1)
        {}
private:
    SpirvInstruction *arg0;
    SpirvInstruction *arg1;
    // VOP3 writes to SGpr
    // VOPC writes to VCC
    std::optional<GcnRegister> CondSGpr;
};

class SpirvBitwise : public SpirvInstruction {
    // Codegen:
    // DSTS = AND(S0, S1) -> dst.CC = S0.CC && S1.CC
    //                       dst.Uni = S0.Uni & S1.Uni
    // DSTV = AND(S0, V1) -> DSTV[lane] = S0.Uni & V1[lane]
    // DSTV = AND(V0, V1) -> DSTV[lane] = V0[lane] & V1[lane]

    // DSTS = AND(V0, _)
    // | DSTS = AND(_, V1)
    // should be illegal

private:
    GcnRegister dst;
    GcnRegister src0;
    GcnRegister src1;
};

class SpirvVariable : public SpirvInstruction {
public:
    SpirvVariable(SpirvPointerType *pointerType):
        pointerType(pointerType) 
        {}
    SpirvPointerType *getPointerType() { return pointerType; }

private:
    SpirvPointerType *pointerType;
    
};

class SpirvLoad : public SpirvInstruction {
public:
    SpirvLoad(SpirvInstruction *pointer, uint memoryOperand = 0):
        pointer(pointer), memoryOperand(memoryOperand)
        {}
private:
    SpirvInstruction *pointer;
    // volatile, aligned, etc
    uint memoryOperand;
};

class SpirvStore : public SpirvInstruction {
public:
    SpirvStore(SpirvInstruction *pointer, SpirvInstruction *val, uint memoryOperand = 0):
        pointer(pointer), val(val), memoryOperand(memoryOperand)
        {}
private:
    SpirvInstruction *pointer;
    SpirvInstruction *val;
    // volatile, aligned, etc
    uint memoryOperand;
};

class SpirvBitcast : public SpirvInstruction {
public:
    SpirvBitcast(bool predicated, SpirvType *to, SpirvInstruction *val):
        SpirvInstruction(predicated)
        {}

private:
    SpirvType *to;
    SpirvInstruction *val;
};