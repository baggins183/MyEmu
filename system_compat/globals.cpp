#include "system_compat/globals.h"

void *_proc_param = nullptr;

void setProcParam(void *proc_param) {
    _proc_param = proc_param;
}
void *getProcParam() {
    return _proc_param;
}