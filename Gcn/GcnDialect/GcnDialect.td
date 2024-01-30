//===- GcnDialect.td - Gcn dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GCN_DIALECT
#define GCN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Gcn dialect definition.
//===----------------------------------------------------------------------===//

def Gcn_Dialect : Dialect {
    let name = "gcn";
    let summary = "A Gcn out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::gcn";
}

//===----------------------------------------------------------------------===//
// Base Gcn operation definition.
//===----------------------------------------------------------------------===//

class Gcn_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gcn_Dialect, mnemonic, traits>;

#endif // GCN_DIALECT