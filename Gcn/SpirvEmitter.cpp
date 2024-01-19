#include "SpirvEmitter.h"
#include "SpirvCommon.h"
#include "SpirvType.h"
#include "SpirvInstruction.h"

#define ACCEPT_TYPE(Type) \
    void Type::accept(SpirvEmitter &emitter) const { if ( !emitter.hasVisited(static_cast<const SpirvType *>(this))) { emitter.visit(*this); }; }

ACCEPT_TYPE(SpirvFloatType)
ACCEPT_TYPE(SpirvIntType)
ACCEPT_TYPE(SpirvBoolType)
ACCEPT_TYPE(SpirvStructType)
ACCEPT_TYPE(SpirvVecType)
ACCEPT_TYPE(SpirvPointerType)

void SpirvEmitter::visit(const SpirvFloatType &floatTy) {
    push(makeOpcode(spv::Op::OpTypeFloat, 3));
    push(mapResultId(static_cast<const SpirvType *>(&floatTy)));
    push(floatTy.width);
}

void SpirvEmitter::visit(const SpirvIntType &intTy) {
    push(makeOpcode(spv::Op::OpTypeInt, 4));
    push(mapResultId(static_cast<const SpirvType *>(&intTy)));
    push(intTy.width);
    push(intTy.isSigned);
}

void SpirvEmitter::visit(const SpirvBoolType &boolTy) {
    push(makeOpcode(spv::Op::OpTypeBool, 2));
    push(mapResultId(static_cast<const SpirvType *>(&boolTy)));
}

void SpirvEmitter::visit(const SpirvStructType &structTy) {
    push(makeOpcode(spv::Op::OpTypeStruct, 2 + structTy.memberTypes.size()));
    push(mapResultId(static_cast<const SpirvType *>(&structTy)));
    for (SpirvType *t: structTy.memberTypes) {
        push(getResultId(t));
    }
}

void SpirvEmitter::visit(const SpirvVecType &vecTy) {
    push(makeOpcode(spv::Op::OpTypeVector, 4));
    push(mapResultId(static_cast<const SpirvType *>(&vecTy)));
    visit(*vecTy.primitiveType);
    push(getResultId(vecTy.primitiveType));
    push(vecTy.length);
}

void SpirvEmitter::visit(const SpirvPointerType &pointerTy) {
    push(makeOpcode(spv::Op::OpTypePointer, 4));
    push(mapResultId(static_cast<const SpirvType *>(&pointerTy)));
    push(pointerTy.storageClass);
    visit(*pointerTy.objectType);
    push(getResultId(pointerTy.objectType));
}