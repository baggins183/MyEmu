#pragma once

#include "SpirvCommon.h"
#include "SpirvType.h"
#include "SpirvInstruction.h"
#include "llvm/ADT/DenseSet.h"

#include <cstddef>
#include <set>
#include <unordered_set>

class SpirvContext {
public:
    SpirvContext() {}

    // Add Spirv structured control flow to check EXEC around vector instructions.
    // The builder should have been set to mark instructions as predicated when translating
    // VALU, VMEM, Export, LDS, and GDS.
    // Also try to reorder predicated instructions adjacent so if's can be fused
    void createPredicatedControlFlow();

    // Create everything before the functions and bodies
    bool createPreamble();
    std::vector<uint8_t> emitSpirvModule();
    
    template <typename T>
    T *allocateInstruction(T &instruction) {
        static_assert(std::is_base_of<SpirvInstruction, T>::value, "type parameter of this class must derive from SpirvInstruction");
        T *ptr = allocator.Allocate<T>();
        return new (ptr) T(instruction);
    }


    template <typename T>
    T *getOrInsertType(T &type) {
        static_assert(std::is_base_of<SpirvType, T>::value, "type parameter of this class must derive from SpirvType");
        auto it = typeSet.find(static_cast<SpirvType *>(&type));
        if (it != typeSet.end()) {
            return static_cast<T *>(*it);
        }
        T *ptr = allocator.Allocate<T>();
        return new (ptr) T(type);
    }

    void setAddressingModel(spv::AddressingModel addrModel) { addressingModel = addrModel; }
    void setMemoryModel(spv::MemoryModel memModel);

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

    SpirvFunction *mainFunction;

private:
    std::vector<SpirvFunction *> functions;

    struct SpirvTypeHash {
        size_t operator()(const SpirvType * const &type) const { return type->hash(); };
    };
    std::unordered_set<SpirvType *, SpirvTypeHash> typeSet;

    // TODO: will currently leak std::vector members, e.g. SpirvStructType::memberTypes (and any 
    // other field that is dynamically allocated by a different allocator)
    // need to figure that out
    llvm::BumpPtrAllocator allocator;
};