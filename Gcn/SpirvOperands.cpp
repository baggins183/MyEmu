#include "Common.h"
#include "Gcn2Spirv.h"

bool SpirvType::equalsShallow(const SpirvOperand& RHS) const {
    return this == &RHS;
}

bool SpirvType::operator==(const SpirvOperand& RHS) const {
    auto typeRHS = dynamic_cast<const SpirvType *>(&RHS);
    if (!typeRHS) {
        return false;
    }
    if (opcode != typeRHS->opcode) {
        return false;
    }
    if (operands.size() != typeRHS->operands.size()) {
        return false;
    }
    for (uint i = 0; i < operands.size(); i++) {
        assert(!dynamic_cast<const SpirvInstruction *>(operands[i]));
        if (!operands[i]->equalsShallow(*typeRHS->operands[i])) {
            return false;
        }
    }

    return true;
}

bool SpirvLiteral::equalsShallow(const SpirvOperand& RHS) const {
    auto *litRHS = dynamic_cast<const SpirvLiteral *>(&RHS);
    return litRHS && words == litRHS->words;
}

bool SpirvLiteral::operator==(const SpirvOperand& RHS) const {
    auto literalRHS = dynamic_cast<const SpirvLiteral *>(&RHS);
    if (!literalRHS) {
        return false;
    }
    return words == literalRHS->words;
}

unsigned SpirvType::hashShallow() const {
    const void *thisPtr = this;
    return pjwHash32((unsigned char *)&thisPtr, sizeof(void *));
}

unsigned SpirvType::hash() const {
    uint32_t hash = pjwHash32((unsigned char *)&opcode, sizeof(opcode));
    for (uint i = 0; i < operands.size(); i++) {
        assert(!dynamic_cast<const SpirvInstruction *>(operands[i]));
        uint32_t subHash = operands[i]->hashShallow();
        updatePjwHash32((unsigned char *)&subHash, sizeof(subHash), hash);
    }
    return hash;
}

unsigned SpirvLiteral::hashShallow() const {
    return hash();
}

unsigned SpirvLiteral::hash() const {
    return pjwHash32((unsigned char *)words.data(), words.size_in_bytes());
}

namespace llvm {
unsigned DenseMapInfo<SpirvType *>::getHashValue(const SpirvType *type) {
    return type->hash();
}

bool DenseMapInfo<SpirvType *>::isEqual(const SpirvType *LHS, const SpirvType *RHS) {
    return *LHS == *RHS;
}

unsigned llvm::DenseMapInfo<SpirvLiteral *>::getHashValue(const SpirvLiteral *literal) {
    return literal->hash();
}

bool llvm::DenseMapInfo<SpirvLiteral *>::isEqual(const SpirvLiteral *LHS, const SpirvLiteral *RHS) {
    return *LHS == *RHS;
}
}