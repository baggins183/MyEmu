#include "Gcn2Spirv.h"
#include "spirv.hpp"

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

SpirvType *SpirvContext::getOrInsertType(SpirvType &type) {
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

void SpirvContext::setMemoryModel(spv::MemoryModel memModel) {
    switch(memModel) {
        case spv::MemoryModel::MemoryModelVulkan:
            capabilities.push_back(spv::Capability::CapabilityVulkanMemoryModel);
        default:
            break;
    }

    memoryModel = memModel;
}

void SpirvContext::createPredicatedControlFlow() {

}
// Remove branches with compile-time constant conditions.
// Especially cases where EXEC is known to be true
// (Do SSA-CCP whatever, see spirv-opt)
void SpirvContext::eliminateDeadBranches() {

}
void SpirvContext::elminateDeadFunctions() {

}
// Create everything before the functions and bodies
bool SpirvContext::createPreamble() {
    return true;
}

bool SpirvContext::numberInstructions() {
    
}

std::vector<uint8_t> SpirvContext::emitSpirvModule() {
    SpirvEmitter emitter;
    emitter.visit(mainFunction);
    //emitter.visit(mainFunction);
    return {};
}
