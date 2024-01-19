#include "SpirvBuilder.h"
#include "SpirvCommon.h"
#include "SpirvInstruction.h"
#include "SpirvType.h"
#include "spirv/spirv.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

SpirvFloatType *SpirvBuilder::makeF32Type() {
    SpirvFloatType type(32);
    return context->getOrInsertType(type);
}

SpirvFloatType *SpirvBuilder::makeF64Type() {
    SpirvFloatType type(64);
    return context->getOrInsertType(type);
}

SpirvIntType *SpirvBuilder::makeI32Type() {
    SpirvIntType type(32, true);
    return context->getOrInsertType(type);
}

SpirvIntType *SpirvBuilder::makeI64Type() {
    SpirvIntType type(64, true);
    return context->getOrInsertType(type);
}

SpirvIntType *SpirvBuilder::makeU32Type() {
    SpirvIntType type(32, false);
    return context->getOrInsertType(type);
}

SpirvIntType *SpirvBuilder::makeU64Type() {
    SpirvIntType type(64, false);
    return context->getOrInsertType(type);
}

SpirvBoolType *SpirvBuilder::makeBoolType() {
    SpirvBoolType type;
    return context->getOrInsertType(type);
}

SpirvStructType *SpirvBuilder::makeStructType(const llvm::ArrayRef<SpirvType *> memberTypes) {
    SpirvStructType type(memberTypes);
    return context->getOrInsertType(type);
}

SpirvPointerType *SpirvBuilder::makePointerType(spv::StorageClass storage, SpirvType *dataType) {
    SpirvPointerType type(dataType, storage);
    return context->getOrInsertType(type);
}

SpirvVariable *SpirvBuilder::makeVariable(SpirvType *type, spv::StorageClass storage, llvm::StringRef name) {
    SpirvPointerType *pointerType = makePointerType(storage, type);
    SpirvVariable var(pointerType);
    // TODO decorate with name (and handle name collision)
    return context->allocateInstruction(var);
}

void SpirvBuilder::initVGpr(uint regno) {
    if (usedVgprs.contains(regno)) {
        return;
    }
    SpirvType *type = makeU32Type();
    llvm::SmallString<5> regName;
    regName += "v";
    regName += std::to_string(regno);
    SpirvVariable *variable = makeVariable(type, spv::StorageClass::StorageClassFunction, regName);
    usedVgprs[regno] = variable;
}

void SpirvBuilder::initSGpr(uint regno) {
    if (usedSGprs.contains(regno)) {
        return;
    }
    SpirvType *uniType = makeU32Type();
    llvm::SmallString<8> regName;
    regName = "s" + std::to_string(regno) + ".uni";
    SpirvVariable *uniVar = makeVariable(uniType, spv::StorageClass::StorageClassFunction, regName);    
    SpirvType *sccType = makeBoolType();
    regName = "s" + std::to_string(regno) + ".cc";
    SpirvVariable *scc = makeVariable(sccType, spv::StorageClass::StorageClassFunction, regName);
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
    SpirvVariable *uniVar = makeVariable(uniType, spv::StorageClass::StorageClassFunction, "VCC.uni");
    SpirvType *sccType = makeBoolType();
    SpirvVariable *scc = makeVariable(sccType, spv::StorageClass::StorageClassFunction, "VCC.cc");
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

SpirvLoad *SpirvBuilder::makeLoad(SpirvInstruction *variable, uint32_t memoryOperand = 0) {
    SpirvLoad load(variable, memoryOperand);
    return context->allocateInstruction(load);
}

SpirvStore *SpirvBuilder::makeStore(SpirvInstruction *variable, SpirvInstruction *val, uint memoryOperand = 0) {
    SpirvStore store(variable, val, memoryOperand);
    return context->allocateInstruction(store);
}

SpirvLoad *SpirvBuilder::loadVGpr(uint regno) {
    initVGpr(regno);
    SpirvInstruction *variable = usedVgprs[regno];
    return makeLoad(variable);
}

SpirvStore *SpirvBuilder::storeVGpr(uint regno, SpirvInstruction *val) {
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

SpirvInstruction *SpirvBuilder::makeBitcast(SpirvType *to, SpirvInstruction *val) {
    SpirvBitcast bitcast(predicated, to, val);
    return context->allocateInstruction(bitcast);
}