#include "GcnDialect.h"
#include "GcnOps.h"

using namespace mlir;
using namespace mlir::gcn;

void GcnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GcnOps.cpp.inc"
      >();
}

#include "GcnOpsDialect.cpp.inc"