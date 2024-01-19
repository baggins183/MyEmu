// Found in GPCS4

#pragma once

#define SCE_ERROR_ERROR_FLAG           0x80000000
#define SCE_ERROR_MAKE_ERROR(fac, sts) (SCE_ERROR_ERROR_FLAG | ((fac) << 16) | (sts))
#define SCE_ERROR_IS_FAILURE(_err)     (((_err)&SCE_ERROR_ERROR_FLAG) == SCE_ERROR_ERROR_FLAG)
#define SCE_ERROR_IS_SUCCESS(_err)     (!((_err)&SCE_ERROR_ERROR_FLAG))

#define SCE_OK            0
#define SCE_ERROR_UNKNOWN SCE_ERROR_MAKE_ERROR(0xFF, 0xFF)


#include "sce_kernel_error.h"
#include "sce_gnm_error.h"
#include "sce_ime_error.h"
#include "sce_userservice_error.h"
#include "sce_systemservice_error.h"
