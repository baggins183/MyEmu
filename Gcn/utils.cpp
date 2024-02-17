#include "utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include <cassert>
#include <mutex>
#include <unordered_map>

// opcode, writesExec, operandType
#define CMP_OP_TABLE(X) \
        X(llvm::AMDGPU::V_CMP_F_F32_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_F_F32_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_F_F64_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_F_F64_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_F_F32_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_F_F32_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_F_F64_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_F_F64_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_F_I32_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LT_I32_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_EQ_I32_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LE_I32_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GT_I32_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GE_I32_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_F_I32_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LT_I32_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_EQ_I32_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LE_I32_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GT_I32_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GE_I32_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_F_I64_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LT_I64_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_EQ_I64_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LE_I64_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GT_I64_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GE_I64_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_F_I64_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LT_I64_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_EQ_I64_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LE_I64_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GT_I64_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GE_I64_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_F_U32_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LT_U32_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_EQ_U32_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LE_U32_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GT_U32_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GE_U32_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_F_U32_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LT_U32_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_EQ_U32_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LE_U32_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GT_U32_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GE_U32_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_F_U64_e32_gfx6_gfx7, GcnCmp::F, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LT_U64_e32_gfx6_gfx7, GcnCmp::LT, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_EQ_U64_e32_gfx6_gfx7, GcnCmp::EQ, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LE_U64_e32_gfx6_gfx7, GcnCmp::LE, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GT_U64_e32_gfx6_gfx7, GcnCmp::GT, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GE_U64_e32_gfx6_gfx7, GcnCmp::GE, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_F_U64_e32_gfx6_gfx7, GcnCmp::F, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LT_U64_e32_gfx6_gfx7, GcnCmp::LT, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_EQ_U64_e32_gfx6_gfx7, GcnCmp::EQ, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LE_U64_e32_gfx6_gfx7, GcnCmp::LE, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GT_U64_e32_gfx6_gfx7, GcnCmp::GT, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GE_U64_e32_gfx6_gfx7, GcnCmp::GE, true, GcnCmp::U64)

GcnCmp::Op compareOpcodeToOperation(unsigned int opcode) {
    static std::unordered_map<unsigned int, GcnCmp::Op> cmp_opcode_to_op = {
#define OP_COL(opcode, op, writesExec, operandType) { opcode, op },
        CMP_OP_TABLE(OP_COL)
#undef OP_COL
    };

    assert(cmp_opcode_to_op.contains(opcode));
    return cmp_opcode_to_op[opcode];
}

bool compareWritesExec(unsigned int opcode) {
    static std::unordered_map<unsigned int, bool> cmp_opcode_to_writes_exec = {
#define WRITES_EXEC_COL(opcode, op, writesExec, operandType) { opcode, writesExec },
        CMP_OP_TABLE(WRITES_EXEC_COL)
#undef WRITES_EXEC_COL
    };

    assert(cmp_opcode_to_writes_exec.contains(opcode));
    return cmp_opcode_to_writes_exec[opcode];
}

GcnCmp::Type compareOpcodeToOperandType(unsigned int opcode) {
    static std::unordered_map<unsigned int, GcnCmp::Type> cmp_opcode_to_operand_type = {
#define OPERAND_TYPE_COL(opcode, op, writesExec, operandType) { opcode, operandType },
        CMP_OP_TABLE(OPERAND_TYPE_COL)
#undef OPERAND_TYPE_COL
    };

    assert(cmp_opcode_to_operand_type.contains(opcode));
    return cmp_opcode_to_operand_type[opcode];
}