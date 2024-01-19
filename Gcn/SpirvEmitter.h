#include "SpirvType.h"
#include "SpirvInstruction.h"
#include "llvm/ADT/DenseSet.h"

class SpirvEmitter {
public:
    int mapResultId(const SpirvType *ty) { typeIds[ty] = ++currentResultId; return currentResultId; }
    int mapResultId(const SpirvInstruction *inst) { resultIds[inst] = ++currentResultId; return currentResultId; }
    void push(uint32_t word) { bytes.push_back(word); }

    int getResultId(SpirvType *type) {
        auto it = typeIds.find(type);
        return it == typeIds.end() ? it->second : -1;
    }
    int getResultId(SpirvInstruction *inst) {
        auto it = resultIds.find(inst);
        return it == resultIds.end() ? it->second : -1;
    }

    bool hasVisited(const SpirvType *ty) { return visitedTypes.contains(ty); }

    void visit(const SpirvType &ty) { ty.accept(*this); }

    void visit(const SpirvFloatType &floatTy);
    void visit(const SpirvIntType &intTy);
    void visit(const SpirvBoolType &boolTy);
    void visit(const SpirvStructType &structTy);
    void visit(const SpirvVecType &vecTy);
    void visit(const SpirvPointerType &pointerTy);

private:
    int currentResultId = 0;
    std::vector<uint32_t> bytes;
    std::unordered_map<const SpirvType *, uint> typeIds;
    std::unordered_map<const SpirvInstruction *, uint> resultIds;

    llvm::DenseSet<SpirvType *> visitedTypes;
};