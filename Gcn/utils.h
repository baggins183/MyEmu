#pragma once

#include "AMDGPUGenInstrInfo_INSTRINFO.inc"

///////////////////////////////////////////////////////////////////////////////////////////
// CMP instructions
///////////////////////////////////////////////////////////////////////////////////////////

namespace GcnCmp {
enum Op {
    F,
    LT,
    EQ,
    LE,
    GT,
    LG,
    GE,
    O_F,
    U_F,
    NGE,
    NLG,
    NGT,
    NLE,
    NEQ,
    NLT,
    TRU,
};

enum Type {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64
};
}

GcnCmp::Op compareOpcodeToOperation(unsigned int opcode);
bool compareWritesExec(unsigned int opcode);
bool compareIsVop3(unsigned int opcode);
GcnCmp::Type compareOpcodeToOperandType(unsigned int opcode);
