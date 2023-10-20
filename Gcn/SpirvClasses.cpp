    #include "Gcn2Spirv.h"
    
    SpirvType::SpirvType(spv::Op opcode, llvm::SmallVector<SpirvOperand *, 4> operands): 
        opcode(opcode),
        operands(operands)
    {
        hash = pjwHash32((unsigned char *)&opcode, sizeof(opcode));
        for (auto &oprnd: operands) {
            assert(!dynamic_cast<SpirvInstruction *>(oprnd));
            if (auto *subType = dynamic_cast<SpirvType *>(oprnd)) {
                hash = updatePjwHash32((unsigned char *)&subType->hash, sizeof(subType->hash), hash);
            } else if (auto *literal = dynamic_cast<SpirvLiteral *>(oprnd)) {
                hash = updatePjwHash32((unsigned char *)literal->words.data(), literal->words.size_in_bytes(), hash);
            }
        }
    }