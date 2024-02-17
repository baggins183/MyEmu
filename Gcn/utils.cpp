#include "utils.h"
#include "AMDGPUGenInstrInfo_INSTRINFO.inc"
#include <cassert>
#include <unordered_map>

// opcode, compareOp, writesExec, isVop3, operandType
#define CMP_OP_TABLE(X) \
        X(llvm::AMDGPU::V_CMP_F_F32_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_F_F32_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_F_F64_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_F_F64_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_F_F32_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, false, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_F_F32_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LT_F32_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F32_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LE_F32_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GT_F32_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LG_F32_e32_gfx6_gfx7, GcnCmp::LG, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GE_F32_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_O_F32_e32_gfx6_gfx7, GcnCmp::O_F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_U_F32_e32_gfx6_gfx7, GcnCmp::U_F, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F32_e32_gfx6_gfx7, GcnCmp::NGE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F32_e32_gfx6_gfx7, GcnCmp::NLG, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F32_e32_gfx6_gfx7, GcnCmp::NGT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F32_e32_gfx6_gfx7, GcnCmp::NLE, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F32_e32_gfx6_gfx7, GcnCmp::NEQ, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F32_e32_gfx6_gfx7, GcnCmp::NLT, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F32_e32_gfx6_gfx7, GcnCmp::TRU, true, false, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_F_F64_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, false, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_F_F64_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LT_F64_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F64_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LE_F64_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GT_F64_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LG_F64_e32_gfx6_gfx7, GcnCmp::LG, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GE_F64_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_O_F64_e32_gfx6_gfx7, GcnCmp::O_F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_U_F64_e32_gfx6_gfx7, GcnCmp::U_F, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F64_e32_gfx6_gfx7, GcnCmp::NGE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F64_e32_gfx6_gfx7, GcnCmp::NLG, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F64_e32_gfx6_gfx7, GcnCmp::NGT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F64_e32_gfx6_gfx7, GcnCmp::NLE, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F64_e32_gfx6_gfx7, GcnCmp::NEQ, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F64_e32_gfx6_gfx7, GcnCmp::NLT, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F64_e32_gfx6_gfx7, GcnCmp::TRU, true, false, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_F_I32_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LT_I32_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_EQ_I32_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LE_I32_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GT_I32_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GE_I32_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_F_I32_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LT_I32_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_EQ_I32_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LE_I32_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GT_I32_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GE_I32_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_F_I64_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LT_I64_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_EQ_I64_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LE_I64_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GT_I64_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GE_I64_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_F_I64_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LT_I64_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_EQ_I64_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LE_I64_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GT_I64_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GE_I64_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_F_U32_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LT_U32_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_EQ_U32_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LE_U32_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GT_U32_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GE_U32_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_F_U32_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LT_U32_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_EQ_U32_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LE_U32_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GT_U32_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GE_U32_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_F_U64_e32_gfx6_gfx7, GcnCmp::F, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LT_U64_e32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_EQ_U64_e32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LE_U64_e32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GT_U64_e32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GE_U64_e32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_F_U64_e32_gfx6_gfx7, GcnCmp::F, true, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LT_U64_e32_gfx6_gfx7, GcnCmp::LT, true, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_EQ_U64_e32_gfx6_gfx7, GcnCmp::EQ, true, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LE_U64_e32_gfx6_gfx7, GcnCmp::LE, true, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GT_U64_e32_gfx6_gfx7, GcnCmp::GT, true, false, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GE_U64_e32_gfx6_gfx7, GcnCmp::GE, true, false, GcnCmp::U64) \
        \
        X(llvm::AMDGPU::V_CMP_F_F32_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LT_F32_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_EQ_F32_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LE_F32_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GT_F32_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_LG_F32_e64_gfx6_gfx7, GcnCmp::LG, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_GE_F32_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_O_F32_e64_gfx6_gfx7, GcnCmp::O_F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_U_F32_e64_gfx6_gfx7, GcnCmp::U_F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGE_F32_e64_gfx6_gfx7, GcnCmp::NGE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLG_F32_e64_gfx6_gfx7, GcnCmp::NLG, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NGT_F32_e64_gfx6_gfx7, GcnCmp::NGT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLE_F32_e64_gfx6_gfx7, GcnCmp::NLE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NEQ_F32_e64_gfx6_gfx7, GcnCmp::NEQ, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_NLT_F32_e64_gfx6_gfx7, GcnCmp::NLT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_TRU_F32_e64_gfx6_gfx7, GcnCmp::TRU, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_F_F32_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LT_F32_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_EQ_F32_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LE_F32_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GT_F32_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_LG_F32_e64_gfx6_gfx7, GcnCmp::LG, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_GE_F32_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_O_F32_e64_gfx6_gfx7, GcnCmp::O_F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_U_F32_e64_gfx6_gfx7, GcnCmp::U_F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGE_F32_e64_gfx6_gfx7, GcnCmp::NGE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLG_F32_e64_gfx6_gfx7, GcnCmp::NLG, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NGT_F32_e64_gfx6_gfx7, GcnCmp::NGT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLE_F32_e64_gfx6_gfx7, GcnCmp::NLE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F32_e64_gfx6_gfx7, GcnCmp::NEQ, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_NLT_F32_e64_gfx6_gfx7, GcnCmp::NLT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPX_TRU_F32_e64_gfx6_gfx7, GcnCmp::TRU, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMP_F_F64_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LT_F64_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_EQ_F64_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LE_F64_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GT_F64_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_LG_F64_e64_gfx6_gfx7, GcnCmp::LG, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_GE_F64_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_O_F64_e64_gfx6_gfx7, GcnCmp::O_F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_U_F64_e64_gfx6_gfx7, GcnCmp::U_F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGE_F64_e64_gfx6_gfx7, GcnCmp::NGE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLG_F64_e64_gfx6_gfx7, GcnCmp::NLG, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NGT_F64_e64_gfx6_gfx7, GcnCmp::NGT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLE_F64_e64_gfx6_gfx7, GcnCmp::NLE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NEQ_F64_e64_gfx6_gfx7, GcnCmp::NEQ, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_NLT_F64_e64_gfx6_gfx7, GcnCmp::NLT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_TRU_F64_e64_gfx6_gfx7, GcnCmp::TRU, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_F_F64_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LT_F64_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_EQ_F64_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LE_F64_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GT_F64_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_LG_F64_e64_gfx6_gfx7, GcnCmp::LG, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_GE_F64_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_O_F64_e64_gfx6_gfx7, GcnCmp::O_F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_U_F64_e64_gfx6_gfx7, GcnCmp::U_F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGE_F64_e64_gfx6_gfx7, GcnCmp::NGE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLG_F64_e64_gfx6_gfx7, GcnCmp::NLG, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NGT_F64_e64_gfx6_gfx7, GcnCmp::NGT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLE_F64_e64_gfx6_gfx7, GcnCmp::NLE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NEQ_F64_e64_gfx6_gfx7, GcnCmp::NEQ, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_NLT_F64_e64_gfx6_gfx7, GcnCmp::NLT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPX_TRU_F64_e64_gfx6_gfx7, GcnCmp::TRU, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_F_F32_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LT_F32_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_EQ_F32_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LE_F32_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GT_F32_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_LG_F32_e64_gfx6_gfx7, GcnCmp::LG, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_GE_F32_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_O_F32_e64_gfx6_gfx7, GcnCmp::O_F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_U_F32_e64_gfx6_gfx7, GcnCmp::U_F, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGE_F32_e64_gfx6_gfx7, GcnCmp::NGE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLG_F32_e64_gfx6_gfx7, GcnCmp::NLG, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NGT_F32_e64_gfx6_gfx7, GcnCmp::NGT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLE_F32_e64_gfx6_gfx7, GcnCmp::NLE, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F32_e64_gfx6_gfx7, GcnCmp::NEQ, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_NLT_F32_e64_gfx6_gfx7, GcnCmp::NLT, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_TRU_F32_e64_gfx6_gfx7, GcnCmp::TRU, false, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_F_F32_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LT_F32_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F32_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LE_F32_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GT_F32_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_LG_F32_e64_gfx6_gfx7, GcnCmp::LG, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_GE_F32_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_O_F32_e64_gfx6_gfx7, GcnCmp::O_F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_U_F32_e64_gfx6_gfx7, GcnCmp::U_F, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F32_e64_gfx6_gfx7, GcnCmp::NGE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F32_e64_gfx6_gfx7, GcnCmp::NLG, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F32_e64_gfx6_gfx7, GcnCmp::NGT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F32_e64_gfx6_gfx7, GcnCmp::NLE, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F32_e64_gfx6_gfx7, GcnCmp::NEQ, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F32_e64_gfx6_gfx7, GcnCmp::NLT, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F32_e64_gfx6_gfx7, GcnCmp::TRU, true, true, GcnCmp::F32) \
        X(llvm::AMDGPU::V_CMPS_F_F64_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LT_F64_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_EQ_F64_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LE_F64_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GT_F64_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_LG_F64_e64_gfx6_gfx7, GcnCmp::LG, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_GE_F64_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_O_F64_e64_gfx6_gfx7, GcnCmp::O_F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_U_F64_e64_gfx6_gfx7, GcnCmp::U_F, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGE_F64_e64_gfx6_gfx7, GcnCmp::NGE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLG_F64_e64_gfx6_gfx7, GcnCmp::NLG, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NGT_F64_e64_gfx6_gfx7, GcnCmp::NGT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLE_F64_e64_gfx6_gfx7, GcnCmp::NLE, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NEQ_F64_e64_gfx6_gfx7, GcnCmp::NEQ, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_NLT_F64_e64_gfx6_gfx7, GcnCmp::NLT, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPS_TRU_F64_e64_gfx6_gfx7, GcnCmp::TRU, false, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_F_F64_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LT_F64_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_EQ_F64_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LE_F64_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GT_F64_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_LG_F64_e64_gfx6_gfx7, GcnCmp::LG, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_GE_F64_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_O_F64_e64_gfx6_gfx7, GcnCmp::O_F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_U_F64_e64_gfx6_gfx7, GcnCmp::U_F, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGE_F64_e64_gfx6_gfx7, GcnCmp::NGE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLG_F64_e64_gfx6_gfx7, GcnCmp::NLG, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NGT_F64_e64_gfx6_gfx7, GcnCmp::NGT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLE_F64_e64_gfx6_gfx7, GcnCmp::NLE, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NEQ_F64_e64_gfx6_gfx7, GcnCmp::NEQ, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_NLT_F64_e64_gfx6_gfx7, GcnCmp::NLT, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMPSX_TRU_F64_e64_gfx6_gfx7, GcnCmp::TRU, true, true, GcnCmp::F64) \
        X(llvm::AMDGPU::V_CMP_F_I32_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LT_I32_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_EQ_I32_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_LE_I32_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GT_I32_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_GE_I32_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_F_I32_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LT_I32_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_EQ_I32_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_LE_I32_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GT_I32_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMPX_GE_I32_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::I32) \
        X(llvm::AMDGPU::V_CMP_F_I64_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LT_I64_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_EQ_I64_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_LE_I64_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GT_I64_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_GE_I64_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_F_I64_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LT_I64_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_EQ_I64_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_LE_I64_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GT_I64_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMPX_GE_I64_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::I64) \
        X(llvm::AMDGPU::V_CMP_F_U32_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LT_U32_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_EQ_U32_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_LE_U32_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GT_U32_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_GE_U32_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_F_U32_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LT_U32_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_EQ_U32_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_LE_U32_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GT_U32_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMPX_GE_U32_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::U32) \
        X(llvm::AMDGPU::V_CMP_F_U64_e64_gfx6_gfx7, GcnCmp::F, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LT_U64_e64_gfx6_gfx7, GcnCmp::LT, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_EQ_U64_e64_gfx6_gfx7, GcnCmp::EQ, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_LE_U64_e64_gfx6_gfx7, GcnCmp::LE, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GT_U64_e64_gfx6_gfx7, GcnCmp::GT, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMP_GE_U64_e64_gfx6_gfx7, GcnCmp::GE, false, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_F_U64_e64_gfx6_gfx7, GcnCmp::F, true, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LT_U64_e64_gfx6_gfx7, GcnCmp::LT, true, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_EQ_U64_e64_gfx6_gfx7, GcnCmp::EQ, true, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_LE_U64_e64_gfx6_gfx7, GcnCmp::LE, true, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GT_U64_e64_gfx6_gfx7, GcnCmp::GT, true, true, GcnCmp::U64) \
        X(llvm::AMDGPU::V_CMPX_GE_U64_e64_gfx6_gfx7, GcnCmp::GE, true, true, GcnCmp::U64) \
        \
        X(llvm::AMDGPU::S_CMP_EQ_I32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_LG_I32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_GT_I32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_GE_I32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_LE_I32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_LT_I32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_EQ_I32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_LG_I32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_GT_I32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_GE_I32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_LE_I32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMPK_LT_I32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::I32) \
        X(llvm::AMDGPU::S_CMP_EQ_U32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMP_LG_U32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMP_GT_U32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMP_GE_U32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMP_LE_U32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMP_LT_U32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_EQ_U32_gfx6_gfx7, GcnCmp::EQ, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_LG_U32_gfx6_gfx7, GcnCmp::LG, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_GT_U32_gfx6_gfx7, GcnCmp::GT, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_GE_U32_gfx6_gfx7, GcnCmp::GE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_LE_U32_gfx6_gfx7, GcnCmp::LE, false, false, GcnCmp::U32) \
        X(llvm::AMDGPU::S_CMPK_LT_U32_gfx6_gfx7, GcnCmp::LT, false, false, GcnCmp::U32)

GcnCmp::Op compareOpcodeToOperation(unsigned int opcode) {
    static std::unordered_map<unsigned int, GcnCmp::Op> cmp_opcode_to_op = {
#define OP_COL(opcode, op, writesExec, isVop3, operandType) { opcode, op },
        CMP_OP_TABLE(OP_COL)
#undef OP_COL
    };
    assert(cmp_opcode_to_op.contains(opcode));
    return cmp_opcode_to_op[opcode];
}

bool compareWritesExec(unsigned int opcode) {    
    static std::unordered_map<unsigned int, bool> cmp_opcode_to_writes_exec = {
#define WRITES_EXEC_COL(opcode, op, writesExec, isVop3, operandType) { opcode, writesExec },
        CMP_OP_TABLE(WRITES_EXEC_COL)
#undef WRITES_EXEC_COL
    };

    assert(cmp_opcode_to_writes_exec.contains(opcode));
    return cmp_opcode_to_writes_exec[opcode];
}

bool compareIsVop3(unsigned int opcode) {
    static std::unordered_map<unsigned int, bool> cmp_opcode_to_vop3 = {
#define IS_VOP3_COL(opcode, op, writesExec, isVop3, operandType) { opcode, isVop3 },
        CMP_OP_TABLE(IS_VOP3_COL)
#undef IS_VOP3_COL
    };

    assert(cmp_opcode_to_vop3.contains(opcode));
    return cmp_opcode_to_vop3[opcode];
}

GcnCmp::Type compareOpcodeToOperandType(unsigned int opcode) {
    static std::unordered_map<unsigned int, GcnCmp::Type> cmp_opcode_to_operand_type = {
#define OPERAND_TYPE_COL(opcode, op, writesExec, isVop3, operandType) { opcode, operandType },
        CMP_OP_TABLE(OPERAND_TYPE_COL)
#undef OPERAND_TYPE_COL
    };

    assert(cmp_opcode_to_operand_type.contains(opcode));
    return cmp_opcode_to_operand_type[opcode];
}