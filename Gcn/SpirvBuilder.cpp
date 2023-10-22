#include "Gcn2Spirv.h"
#include "spirv.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

SpirvLiteral *SpirvContext::allocateLiteral(SpirvLiteral &&obj) {
    SpirvLiteral *ptr = literalAllocator.Allocate();
    new (ptr) SpirvLiteral(obj);
    return ptr;
}

SpirvType *SpirvContext::allocateType(SpirvType &&obj) {
    SpirvType *ptr = typeAllocator.Allocate();
    new (ptr) SpirvType(obj);
    return ptr;
}

SpirvInstruction *SpirvContext::allocateInstruction(SpirvInstruction &&obj) {
    SpirvInstruction *ptr = instructionAllocator.Allocate();
    new (ptr) SpirvInstruction(obj);
    return ptr;
}

const SpirvType *SpirvContext::getOrInsertType(SpirvType &type) {
    auto it = typeSet.find(&type);
    if (it == typeSet.end()) {
        SpirvType *newType = allocateType(std::move(type));
        if (newType) {
            typeSet.insert(newType);
        }
        return newType;
    } else {
        return *it;
    }
}

SpirvLiteral *SpirvContext::makeLiteral(LiteralVec words) {
    SpirvLiteral lit(words);
    auto it = literalSet.find(&lit);
    if (it == literalSet.end()) {
        SpirvLiteral *newLit = allocateLiteral(std::move(lit));
        if (newLit) {
            literalSet.insert(newLit);
        }
        return newLit;
    } else {
        return *it;
    }
}

SpirvLiteral *SpirvContext::makeLiteral(uint32_t scalar) {
    LiteralVec words = { scalar };
    return makeLiteral(words);
}

const SpirvType *SpirvBuilder::makeF32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    OperandVec operands = { bits };
    SpirvType type(spv::OpTypeFloat, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeF64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    OperandVec operands = { bits };
    SpirvType type(spv::OpTypeFloat, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeI32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    SpirvLiteral *sign = context->makeLiteral(1);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeI64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    SpirvLiteral *sign = context->makeLiteral(1);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeU32Type() {
    SpirvLiteral *bits = context->makeLiteral(32);
    SpirvLiteral *sign = context->makeLiteral(0);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeU64Type() {
    SpirvLiteral *bits = context->makeLiteral(64);
    SpirvLiteral *sign = context->makeLiteral(0);
    OperandVec operands = { bits, sign };
    SpirvType type(spv::OpTypeInt, operands);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeBoolType() {
    SpirvType type(spv::OpTypeBool);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makeStructType(OperandVec memberTypes) {
    SpirvType type(spv::OpTypeStruct, memberTypes);
    return context->getOrInsertType(type);
}

const SpirvType *SpirvBuilder::makePointerType(spv::StorageClass storage, const SpirvType *objectType) {
    SpirvLiteral *storageWord = context->makeLiteral(storage);
    OperandVec operands = { storageWord, objectType };
    SpirvType type(spv::OpTypePointer, operands);
    return context->getOrInsertType(type);
}

const SpirvInstruction *SpirvBuilder::makeVariable(const SpirvType *objectType, spv::StorageClass storage, llvm::StringRef name = "") {
    SpirvLiteral *storageWord = context->makeLiteral(storage);
    const SpirvType *resultType = makePointerType(storage, objectType);
    OperandVec operands = { resultType, storageWord };
    const SpirvInstruction *rv = context->allocateInstruction(SpirvInstruction(spv::OpVariable, resultType, operands));

    if ( !name.empty()) {
        // TODO add decoration
    }
    return rv;
}

void SpirvBuilder::initVGpr(uint regno) {
    if (usedVgprs.contains(regno)) {
        return;
    }
    const SpirvType *type = makeU32Type();
    llvm::SmallString<5> regName;
    regName += "v";
    regName += std::to_string(regno);
    const SpirvInstruction *variable = makeVariable(type, spv::StorageClass::StorageClassFunction, regName);
    usedVgprs[regno] = variable;
}

const SpirvInstruction *SpirvBuilder::makeLoad(const SpirvInstruction *variable, uint32_t memoryOperand = 0) {
    OperandVec operands;
    const SpirvType *pointerType = variable->getResultType();
    // TODO get named operand convenience? OpVariable, object type -> idx 1
    auto *resultType = dynamic_cast<const SpirvType *>(pointerType->getOperand(1));
    operands = { variable };
    if (memoryOperand) {
        SpirvLiteral *memoryLit = context->makeLiteral(memoryOperand);
        operands.push_back(memoryLit);
    }
    return context->allocateInstruction(SpirvInstruction(spv::OpLoad, resultType, operands));
}

const SpirvInstruction *SpirvBuilder::makeStore(const SpirvInstruction *variable, const SpirvInstruction *val, uint32_t memoryOperand = 0) {
    OperandVec operands;
    // TODO get named operand convenience? OpVariable, object type -> idx 1
    operands = { variable, val };
    if (memoryOperand) {
        SpirvLiteral *memoryLit = context->makeLiteral(memoryOperand);
        operands.push_back(memoryLit);
    }
    return context->allocateInstruction(SpirvInstruction(spv::OpStore, operands));
}

const SpirvInstruction *SpirvBuilder::loadVGpr(uint regno) {
    initVGpr(regno);
    const SpirvInstruction *variable = usedVgprs[regno];
    return makeLoad(variable);
}

const SpirvInstruction *SpirvBuilder::storeVGpr(uint regno, const SpirvInstruction *val) {
    initVGpr(regno);
    const SpirvInstruction *variable = usedVgprs[regno];
    return makeStore(variable, val);
}

void SpirvBuilder::initVCC() {
    if (VCC) {
        return;
    }
    const SpirvType *type = makeBoolType();
    VCC = makeVariable(type, spv::StorageClass::StorageClassFunction);
}

const SpirvInstruction *SpirvBuilder::loadVCC() {
    initVCC();
    return makeLoad(VCC);
}

const SpirvInstruction *SpirvBuilder::StoreVCC(SpirvInstruction *val) {
    initVCC();
    return makeStore(VCC, val);
}

const SpirvInstruction *SpirvBuilder::makeBitcast(const SpirvType *type, const SpirvInstruction *from) {
    return context->allocateInstruction(SpirvInstruction(spv::OpBitcast, { from }));
}