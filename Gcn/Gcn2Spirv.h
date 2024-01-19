#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <cstddef>
#include <cstdint>
#include <vector>

bool convertGcnToSpirv(llvm::ArrayRef<uint8_t> gcnBytes, std::vector<uint32_t> &spirv);