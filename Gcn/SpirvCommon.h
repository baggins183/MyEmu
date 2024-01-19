#pragma once

#include <vector>
#include "llvm/Support/Allocator.h"
#include "spirv/spirv.hpp"

static inline uint32_t makeOpcode(spv::Op opcode, uint16_t wordCount) {
    return wordCount << 16 | opcode;
}