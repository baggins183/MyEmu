#include "Gcn2Spirv.h"
#include "spirv.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

SpirvType *SpirvBuilder::makeF32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    OperandVec operands = { bits };
    SpirvType type(spv::OpTypeFloat, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeF64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    OperandVec operands = { bits };
    SpirvType type(spv::OpTypeFloat, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeI32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    SpirvLiteral *sign = context->makeLiteral(1);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeI64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    SpirvLiteral *sign = context->makeLiteral(1);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeU32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    SpirvLiteral *sign = context->makeLiteral(0);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeU64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    SpirvLiteral *sign = context->makeLiteral(0);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeBoolType() {
    SpirvType type(spv::OpTypeBool);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makeStructType(OperandVec memberTypes) {
    SpirvType type(spv::OpTypeStruct, memberTypes);
    return context->getOrInsertType(type);
}

SpirvType *SpirvBuilder::makePointerType(spv::StorageClass storage, SpirvType *objectType) {
    SpirvLiteral *storageWord = context->makeLiteral(storage);
    OperandVec operands = { storageWord, objectType };
    SpirvType type(spv::OpTypePointer, operands);
    return context->getOrInsertType(type);
}

SpirvInstruction *SpirvBuilder::makeVariable(SpirvType *objectType, spv::StorageClass storage, llvm::StringRef name = "") {
    SpirvLiteral *storageWord = context->makeLiteral(storage);
    SpirvType *resultType = makePointerType(storage, objectType);
    OperandVec operands = { resultType, storageWord };
    SpirvInstruction *rv = context->allocateInstruction(SpirvInstruction(spv::OpVariable, resultType, operands));

    if ( !name.empty()) {
        // TODO add decoration
    }
    return rv;
}

void SpirvBuilder::initVGpr(uint regno) {
    if (usedVgprs.contains(regno)) {
        return;
    }
    SpirvType *type = makeU32Type();
    llvm::SmallString<5> regName;
    regName += "v";
    regName += std::to_string(regno);
    SpirvInstruction *variable = makeVariable(type, spv::StorageClass::StorageClassFunction, regName);
    usedVgprs[regno] = variable;
}

void SpirvBuilder::initSGpr(uint regno) {
    if (usedSGprs.contains(regno)) {
        return;
    }
    SpirvType *uniType = makeU32Type();
    llvm::SmallString<8> regName;
    regName = "s" + std::to_string(regno) + ".uni";
    SpirvInstruction *uniVar = makeVariable(uniType, spv::StorageClass::StorageClassFunction, regName);    
    SpirvType *sccType = makeBoolType();
    regName = "s" + std::to_string(regno) + ".cc";
    SpirvInstruction *scc = makeVariable(sccType, spv::StorageClass::StorageClassFunction, regName);
    usedSGprs[regno] = {
        .uni = uniVar,
        .cc = scc
    };
}

void SpirvBuilder::initVCC() {
    if (VCC.cc) {
        return;
    }
    SpirvType *uniType = makeU32Type();
    SpirvInstruction *uniVar = makeVariable(uniType, spv::StorageClass::StorageClassFunction, "VCC.uni");
    SpirvType *sccType = makeBoolType();
    SpirvInstruction *scc = makeVariable(sccType, spv::StorageClass::StorageClassFunction, "VCC.cc");
    VCC = {
        .uni = uniVar,
        .cc = scc
    };
}

void SpirvBuilder::initSCC() {
    if (SCC) {
        return;
    }
    SpirvType *type = makeBoolType();
    SCC = makeVariable(type, spv::StorageClass::StorageClassFunction, "SCC");
}

SpirvInstruction *SpirvBuilder::makeLoad(SpirvInstruction *variable, uint32_t memoryOperand = 0) {
    OperandVec operands;
    const SpirvType *pointerType = variable->getResultType();
    // TODO get named operand convenience? OpVariable, object type -> idx 1
    auto *resultType = dynamic_cast<const SpirvType *>(pointerType->getOperand(1));
    operands = { variable };
    if (memoryOperand) {
        SpirvLiteral *memoryLit = context->makeLiteral(memoryOperand);
        operands.push_back(memoryLit);
    }
    return context->allocateInstruction(SpirvInstruction(spv::OpLoad, predicated, operands, resultType));
}

SpirvInstruction *SpirvBuilder::makeStore(SpirvInstruction *variable, SpirvInstruction *val, uint32_t memoryOperand = 0) {
    OperandVec operands;
    // TODO get named operand convenience? OpVariable, object type -> idx 1
    operands = { variable, val };
    if (memoryOperand) {
        SpirvLiteral *memoryLit = context->makeLiteral(memoryOperand);
        operands.push_back(memoryLit);
    }
    return context->allocateInstruction(SpirvInstruction(spv::OpStore, predicated, operands));
}

SpirvInstruction *SpirvBuilder::loadVGpr(uint regno) {
    initVGpr(regno);
    SpirvInstruction *variable = usedVgprs[regno];
    return makeLoad(variable);
}

SpirvInstruction *SpirvBuilder::storeVGpr(uint regno, SpirvInstruction *val) {
    initVGpr(regno);
    SpirvInstruction *variable = usedVgprs[regno];
    return makeStore(variable, val);
}

SpirvInstruction *SpirvBuilder::loadVCC_CC() {
    initVCC();
    return makeLoad(VCC.cc);
}

SpirvInstruction *SpirvBuilder::StoreVCC_CC(SpirvInstruction *val) {
    initVCC();
    return makeStore(VCC.cc, val);
}

SpirvInstruction *SpirvBuilder::makeBitcast(SpirvType *type, SpirvInstruction *from) {
    return context->allocateInstruction(SpirvInstruction(spv::OpBitcast, predicated, { from }));
}