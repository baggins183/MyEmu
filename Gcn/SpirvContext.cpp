#include "SpirvContext.h"
#include "spirv/spirv.hpp"

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

// Create everything before the functions and bodies
bool SpirvContext::createPreamble() {
    return true;
}