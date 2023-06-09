#pragma once

#include <elf.h>

typedef void *procparam_t;

void setProcParam(procparam_t);
procparam_t getProcParam();