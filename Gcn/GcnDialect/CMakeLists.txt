#set(LLVM_TARGET_DEFINITIONS GcnOps.td)
add_mlir_dialect(GcnOps gcn)
add_mlir_doc(GcnOps GcnOps Standalone/ -gen-op-doc)
add_mlir_doc(GcnDialect GcnDialect Standalone/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRGcn
        GcnDialect.cpp
        GcnOps.cpp

        #ADDITIONAL_HEADER_DIRS
        #${CMAKE_CURRENT_BINARY_DIR}

        DEPENDS
        MLIRGcnOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

