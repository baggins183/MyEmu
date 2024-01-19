#pragma once

#include "SpirvCommon.h"
#include "spirv.hpp"
#include "llvm/ADT/ArrayRef.h"

class SpirvEmitter;

class SpirvType;
class SpirvFloatType;
class SpirvIntType;
class SpirvBoolType;
class SpirvStructType;
class SpirvVecType;
class SpirvPointerType;

template<typename type> size_t hash_member(type member) {
    std::hash<type> h;
    return h(member);
}

template<typename... types> size_t hash_derived_type( const types & ... members) {
    size_t result = 0;
    ((result ^= hash_member(members)),...);

    return result;
}

#define DEFINE_TYPE_HASH(...) \
    virtual size_t hash() const override { return typeid(*this).hash_code() ^ hash_derived_type(__VA_ARGS__); }

#define EMIT_TYPE_DECLS \
    friend class SpirvEmitter; \
    virtual void accept(SpirvEmitter &emitter) const override;

class SpirvType {
public:
    virtual size_t hash() const = 0;
    virtual bool operator==(SpirvType const& other) const = default;
    virtual void accept(SpirvEmitter &emitter) const = 0;
    virtual ~SpirvType() {};
    SpirvType(const SpirvType& rhs) = default;

    spv::Op getOpcode() const { return opcode; };

//protected:
    SpirvType(spv::Op opcode):
        opcode(opcode)
    {}

    spv::Op opcode;
};

class SpirvFloatType : public SpirvType {
public:
    SpirvFloatType(uint width): SpirvType(spv::OpTypeFloat), width(width)
        {}

    virtual ~SpirvFloatType() override {};

    DEFINE_TYPE_HASH(width)
    EMIT_TYPE_DECLS

private:
    uint width;
};

class SpirvIntType : public SpirvType {
public:
    SpirvIntType(uint width, bool isSigned): SpirvType(spv::OpTypeInt), width(width), isSigned(isSigned)
        {}

    DEFINE_TYPE_HASH(width, isSigned)
    EMIT_TYPE_DECLS

private:
    uint width;
    bool isSigned;
};

class SpirvBoolType : public SpirvType {
public:
    SpirvBoolType(): SpirvType(spv::OpTypeBool)
        {}

    DEFINE_TYPE_HASH()
    EMIT_TYPE_DECLS
};

class SpirvStructType : public SpirvType {
public:
    SpirvStructType(const llvm::ArrayRef<SpirvType *> memberTypes)
        : SpirvType(spv::OpTypeStruct)
    {
        this->memberTypes = memberTypes;
    }

    virtual size_t hash() const override {
        size_t result = typeid(*this).hash_code();
        for (auto t : memberTypes) {
            std::hash<SpirvType *> h;
            result ^= h(t);
        }
        return result;
    }

    EMIT_TYPE_DECLS

private:
    std::vector<SpirvType *> memberTypes;
};

class SpirvVecType : public SpirvType {
    SpirvVecType(SpirvType *primitiveType, uint length):
        SpirvType(spv::Op::OpTypeVector), primitiveType(primitiveType), length(length)
        {}

    DEFINE_TYPE_HASH(primitiveType, length)
    EMIT_TYPE_DECLS

private:
    SpirvType *primitiveType;
    uint length;
};

class SpirvPointerType : public SpirvType {
public:
    SpirvPointerType(SpirvType *objectType, spv::StorageClass storageClass)
        : SpirvType(spv::OpTypePointer), objectType(objectType), storageClass(storageClass)
        {}

    DEFINE_TYPE_HASH(objectType, storageClass)
    EMIT_TYPE_DECLS
    
    SpirvType *getObjectType() { return objectType; }

private:
    SpirvType *objectType;
    spv::StorageClass storageClass;
};