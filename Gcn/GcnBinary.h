// A lot of struct definitions found in Gpcs4

#pragma once

#include <cstdint>
#include <cstdio>

enum class ShaderBinaryStageType : uint8_t
{
	kPixelShader    = 0,  ///< PS stage shader.
	kVertexShader   = 1,  ///< VS stage shader
	kExportShader   = 2,
	kLocalShader    = 3,
	kComputeShader  = 4,  ///< CS stage shader.
	kGeometryShader = 5,
	kUnknown        = 6,
	kHullShader     = 7,  ///< HS stage shader.
	kDomainShader   = 8,  ///< DS stage shader with embedded CS stage frontend shader.
};

enum ShaderInputUsageType
{
	kShaderInputUsageImmResource                = 0x00, ///< Immediate read-only buffer/texture descriptor.
	kShaderInputUsageImmSampler			        = 0x01, ///< Immediate sampler descriptor.
	kShaderInputUsageImmConstBuffer             = 0x02, ///< Immediate constant buffer descriptor.
	kShaderInputUsageImmVertexBuffer            = 0x03, ///< Immediate vertex buffer descriptor.
	kShaderInputUsageImmRwResource				= 0x04, ///< Immediate read/write buffer/texture descriptor.
	kShaderInputUsageImmAluFloatConst		    = 0x05, ///< Immediate float const (scalar or vector).
	kShaderInputUsageImmAluBool32Const		    = 0x06, ///< 32 immediate Booleans packed into one UINT.
	kShaderInputUsageImmGdsCounterRange	        = 0x07, ///< Immediate UINT with GDS address range for counters (used for append/consume buffers).
	kShaderInputUsageImmGdsMemoryRange		    = 0x08, ///< Immediate UINT with GDS address range for storage.
	kShaderInputUsageImmGwsBase                 = 0x09, ///< Immediate UINT with GWS resource base offset.
	kShaderInputUsageImmShaderResourceTable     = 0x0A, ///< Pointer to read/write resource indirection table.
	kShaderInputUsageImmLdsEsGsSize             = 0x0D, ///< Immediate LDS ESGS size used in on-chip GS
	// Skipped several items here...
	kShaderInputUsageSubPtrFetchShader		    = 0x12, ///< Immediate fetch shader subroutine pointer.
	kShaderInputUsagePtrResourceTable           = 0x13, ///< Flat resource table pointer.
	kShaderInputUsagePtrInternalResourceTable   = 0x14, ///< Flat internal resource table pointer.
	kShaderInputUsagePtrSamplerTable		    = 0x15, ///< Flat sampler table pointer.
	kShaderInputUsagePtrConstBufferTable	    = 0x16, ///< Flat const buffer table pointer.
	kShaderInputUsagePtrVertexBufferTable       = 0x17, ///< Flat vertex buffer table pointer.
	kShaderInputUsagePtrSoBufferTable		    = 0x18, ///< Flat stream-out buffer table pointer.
	kShaderInputUsagePtrRwResourceTable		    = 0x19, ///< Flat read/write resource table pointer.
	kShaderInputUsagePtrInternalGlobalTable     = 0x1A, ///< Internal driver table pointer.
	kShaderInputUsagePtrExtendedUserData        = 0x1B, ///< Extended user data pointer.
	kShaderInputUsagePtrIndirectResourceTable   = 0x1C, ///< Pointer to resource indirection table.
	kShaderInputUsagePtrIndirectInternalResourceTable = 0x1D, ///< Pointer to internal resource indirection table.
	kShaderInputUsagePtrIndirectRwResourceTable = 0x1E, ///< Pointer to read/write resource indirection table.
};

class VsShader;
class PsShader;
class ShaderInfo
{
public:

	union
	{
		const void *m_shaderStruct;		///< A pointer to the shader struct -- typeless.
		const VsShader* m_vsShader;
		const PsShader* m_psShader;
	};

	const uint32_t *m_gpuShaderCode;		///< A pointer to the GPU Shader Code which will need to be copied into GPU visible memory.
	uint32_t m_gpuShaderCodeSize;			///< The size of the GPU Shader Code in bytes.
	uint32_t m_reserved;
};

class PipelineStage
{
public:
	/// Represents vertex shader information.
	class VsInfo
	{
	public:
		uint8_t		m_vertexShaderVariant;	///< The <c>PsslVertexVariant</c> such as <c>kVertexVariantVertex</c>, <c>kVertexVariantExport</c>, <c>kVertexVariantLocal</c> etc.
		uint8_t		m_paddingTo32[3];		///< Padding.	
	};

	/// Represents domain shader information.
	class DsInfo
	{
	public:
		uint8_t		m_domainShaderVariant;	///< The <c>PsslDomainVariant</c> such as <c>kDomainVariantVertex</c>, <c>kDomainVariantExport</c> etc. 
		uint8_t		m_paddingTo32[3];		///< Padding.
	};

	/// Represents geometry shader information.
	class GsInfo
	{
	public:
		uint8_t		m_geometryShaderVariant;	///< The <c>PsslGeometryVariant</c> such as <c>kGeometryVariantOnBuffer</c>, <c>kGeometryVariantOnChip</c> etc. 
		uint8_t		m_paddingTo32[3];		///< Padding.
	};

	/// Represents geometry shader information.
	class HsInfo
	{
	public:
		uint8_t		m_hullShaderVariant;	///< The <c>PsslHullVariant</c> such as <c>kHullVariantOnBuffer</c>, <c>kHullVariantOnChip</c> etc. 
		uint8_t		m_paddingTo32[3];		///< Padding.
	};

	/// Stores data as different class types, depending on the type of shader. 
	union
	{
		uint32_t    m_u32;                  ///< An unsigned 32 bit integer. 
		VsInfo      m_vsInfo;				///< The vertex shader information. 
		DsInfo      m_dsInfo;				///< The domain shader information. 
		GsInfo      m_gsInfo;               ///< The geometry shader information.
		HsInfo      m_hsInfo;               ///< The hull shader information.
	};
};

class SystemAttributes
{
public:
	/// Represents CS pipeline stage information.
	class CsInfo
	{
	public:
		uint16_t	m_numThreads[3];        ///< The number of threads.
	};

	/// Represents GS pipeline stage information.
	class GsInfo
	{
	public:
		uint16_t	m_instance;				///< The instance of the GS Shader. 
		uint16_t	m_maxVertexCount;       ///< The maximum number of vertices count.
		uint8_t		m_inputType;            ///< The GS Input Type (<c>PsslGsIoType</c>) such as triangle, line, point, adjacent tri + line, or patch.
		uint8_t		m_outputType;           ///< The GS Output Type (<c>PsslGsIoType</c>) such as triangles, lines, or points.
		uint8_t		m_patchSize;	        ///< The patch size in case of patch topology.
	};

	/// Represents DS pipeline stage information.
	class DsInfo
	{
	public:
		uint8_t		m_domainPatchType;      ///< The <c>PsslHsDsPatchType</c>: triangle, quad, or isoline.  
		uint8_t		m_inputControlPoints;   ///< The number of points in the input patch.
	};

	/// Represents HS pipeline stage information.
	class HsInfo
	{
	public:
		uint8_t		m_domainPatchType;      ///< The <c>PsslHsDsPatchType</c>: triangle, quad, or isoline. 
		uint8_t		m_inputControlPoints;   ///< The number of points in the input patch.
		uint8_t		m_outputTopologyType;   ///< The <c>PsslHsTopologyType</c>: point, line, cwtri, or ccwtri.
		uint8_t		m_partitioningType;	    ///< The <c>PsslHsPartitioningType</c>: integer, powof2, odd_fractional, or even_fractional.

		uint8_t		m_outputControlPoints;  ///< The number of points in the output patch.
		uint8_t		m_patchSize; 		    ///< The size of patch.
		uint8_t		m_paddingTo32[2];       ///< Padding.

		float		m_maxTessFactor;        ///< The maximum tessellation factor.
	};

	/// Stores data as different class types, depending on the type of shader. 
	union
	{
		uint32_t	m_u32[3];               ///< 12 bytes. 
		CsInfo      m_csInfo;               ///< The compute shader information.
		GsInfo      m_gsInfo;               ///< The geometry shader information.
		DsInfo      m_dsInfo;               ///< The domain shader information.
		HsInfo      m_hsInfo;               ///< The hull shader information.
	};
};

class Header
{
public:
	uint8_t				m_formatVersionMajor;         ///< The version of shader binary format: major numbering. 
	uint8_t				m_formatVersionMinor;         ///< The version of shader binary format: minor numbering.
	uint16_t			m_compilerRevision;           ///< The compiler type specific version of shader compiler: this is the svn revision for m_compilerType==kCompilerTypeOrbisPsslc or kCompilerTypeOrbisEsslc or for kCompilerTypeUnspecified (pre-SDK 2.500 versions of these compilers)

	uint32_t			m_associationHash0;           ///< The shader association hash 1.
	uint32_t			m_associationHash1;           ///< The shader association hash 2.

	uint8_t				m_shaderType;                 ///< The <c>PsslShaderType</c>: VS, PS, GS, CS, GS, HS, or DS.
	uint8_t				m_codeType;                   ///< The <c>PsslCodeType</c>: IL, ISA, or SCU.
	uint8_t             m_usesShaderResourceTable;    ///< The shader uses resource table.
	uint8_t		    	m_compilerType : 4;      ///< The <c>PsslCompilerType</c>; 0
	uint8_t				m_paddingTo32 : 4;      // 0; reserved for future use

	uint32_t			m_codeSize;                   ///< The size of code section.

	PipelineStage		m_shaderTypeInfo;             ///< The shader pipeline stage info.
	SystemAttributes	m_shaderSystemAttributeInfo;  ///< The shader system attribute info.
};

class ShaderFileHeader
{
public:
	uint32_t        m_fileHeader;			///< File identifier. Should be equal to kShaderFileHeaderId
	uint16_t        m_majorVersion;			///< Major version of the shader binary.
	uint16_t        m_minorVersion;			///< Minor version of the shader binary.
	uint8_t         m_type;					///< Type of shader. Comes from ShaderType.
	uint8_t			m_shaderHeaderSizeInDW;	///< <c>\<Type\>Shader.computeSize()/4</c>. For example, see CsShader::computeSize().
	uint8_t			m_shaderAuxData;		///< A flag that indicates whether shader auxiliary data is present after end of the shader data ( <c>sizeof(ShaderFileHeader) +</c>
														///< <c>m_shaderHeaderSizeInDW * 4 + ShaderCommonData::m_shaderSize +</c>
														///< <c>ShaderCommonData::m_embeddedConstantBufferSizeInDQW * 16)</c>. Set to 1 to indicate it is
	uint8_t         m_targetGpuModes;		///< Union of all TargetGpuMode values for which this shader binary is valid.
	uint32_t        m_reserved1;			///< Must be 0.
};

class ShaderCommonData
{
public:
	// Memory Layout:
	// - Shader setup data (starting with ShaderCommonData)
	// - n InputUsage (4 bytes each)
	// - immediateConstants
	uint32_t        m_shaderSize : 23;		   ///< The size of the shader binary code block in bytes.
	uint32_t        m_shaderIsUsingSrt : 1;		   ///< A bitflag that indicates if the shader is using a Shader Resource Table.
	uint32_t        m_numInputUsageSlots : 8;           ///< The number of InputUsageSlot entries following the main shader structure.
	uint16_t        m_embeddedConstantBufferSizeInDQW; ///< The size of the embedded constant buffer in 16-byte <c>DWORD</c>s.
	uint16_t        m_scratchSizeInDWPerThread;        ///< The scratch size required by each thread in 4-byte <c>DWORD</c>s.

	/** @brief Calculates and returns the size of the shader code including its embedded CB size in bytes */
	//uint32_t computeShaderCodeSizeInBytes() const { return m_shaderSize + m_embeddedConstantBufferSizeInDQW * 16; }
};

struct ShaderBinaryInfo
{
    uint8_t m_signature[7];  // 'OrbShdr'
    uint8_t m_version;       // ShaderBinaryInfoVersion

    unsigned int m_pssl_or_cg : 1;   // 1 = PSSL / Cg, 0 = IL / shtb
    unsigned int m_cached : 1;       // 1 = when compile, debugging source was cached.  May only make sense for PSSL=1
    uint32_t     m_type : 4;         // See enum ShaderBinaryType
    uint32_t     m_source_type : 2;  // See enum ShaderSourceType
    unsigned int m_length : 24;      // Binary code length (does not include this structure or any of its preceding associated tables)

    uint8_t m_chunkUsageBaseOffsetInDW;  // in DW, which starts at ((uint32_t*)&ShaderBinaryInfo) - m_chunkUsageBaseOffsetInDW; max is currently 7 dwords (128 T# + 32 V# + 20 CB V# + 16 UAV T#/V#)
    uint8_t m_numInputUsageSlots;        // Up to 16 user data reg slots + 128 extended user data dwords supported by CUE; up to 16 user data reg slots + 240 extended user data dwords supported by Gnm::InputUsageSlot
    uint8_t m_isSrt : 1;                 // 1 if this shader uses shader resource tables and has an SrtDef table embedded below the input usage table and any extended usage info
    uint8_t m_isSrtUsedInfoValid : 1;    // 1 if SrtDef::m_isUsed=0 indicates an element is definitely unused; 0 if SrtDef::m_isUsed=0 indicates only that the element is not known to be used (m_isUsed=1 always indicates a resource is known to be used)
    uint8_t m_isExtendedUsageInfo : 1;   // 1 if this shader has extended usage info for the InputUsage table embedded below the input usage table
    uint8_t m_reserved2 : 5;             // For future use
    uint8_t m_reserved3;                 // For future use

    uint32_t m_shaderHash0;  // Association hash first 4 bytes
    uint32_t m_shaderHash1;  // Association hash second 4 bytes
    uint32_t m_crc32;        // crc32 of shader + this struct, just up till this field
};


struct alignas(4) InputUsageSlot
{
    uint8_t m_usageType;      ///< From Gnm::ShaderInputUsageType.
    uint8_t m_apiSlot;        ///< API slot or chunk ID.
    uint8_t m_startRegister;  ///< User data slot.

    union
    {
        struct
        {
            uint8_t m_registerCount : 1;  ///< If 0, count is 4DW; if 1, count is 8DW. Other sizes are defined by the usage type.
            uint8_t m_resourceType : 1;   ///< If 0, resource type <c>V#</c>; if 1, resource type <c>T#</c>, in case of a Gnm::kShaderInputUsageImmResource.
            uint8_t m_reserved : 2;       ///< Unused; must be set to zero.
            uint8_t m_chunkMask : 4;      ///< Internal usage data.
        };
        uint8_t m_srtSizeInDWordMinusOne;  ///< Size of the SRT data; used for Gnm::kShaderInputUsageImmShaderResourceTable.
    };
};

struct VertexInputSemantic
{
    uint8_t m_semantic;
    uint8_t m_vgpr;
    uint8_t m_sizeInElements;
    uint8_t m_reserved;
};


struct VertexExportSemantic
{                      // __cplusplus
    uint8_t m_semantic;      ///< Description to be specified.
    uint8_t m_outIndex : 5;  ///< Equal to exp instruction's paramN
    uint8_t m_reserved : 1;
    uint8_t m_exportF16 : 2;  ///< if (m_exportF16 == 0) this shader exports a 32-bit value to this parameter; if (m_exportF16 & 1) this shader exports a 16-bit float value to the low 16-bits of each channel; if (m_exportF16 & 2) this shader exports a 16-bit float value to the high 16-bits of each channel
};


struct PixelInputSemantic
{
    union
    {
        struct
        {
            uint16_t m_semantic : 8;      ///< The semantic, matched against the semantic value in the VertexExportSemantic table in the VS shader.
            uint16_t m_defaultValue : 2;  ///< The default value supplied to the shader, if m_semantic is not matched in the VS shader. 0={0,0,0,0}, 1={0,0,0,1.0}, 2={1.0,1.0,1.0,0}, 3={1.0,1.0,1.0,1.0}
            uint16_t m_isFlatShaded : 1;  ///< if (m_interpF16 == 0) A bitflag that specifies whether the value interpolation is constant in the shader. It is ignored if <c><i>m_isCustom</i></c> is set; otherwise, it  indicates that a shader reads only { P0 } and that some handling of infinite values in the calculation of P1-P0 and P2-P0 can be disabled.
            uint16_t m_isLinear : 1;      ///< A bitflag that specifies whether the value interpolation is linear in the shader. It is unused by the Gnm runtime.
            uint16_t m_isCustom : 1;      ///< if (m_interpF16 == 0) A bitflag that specifies whether the value interpolation is custom in the shader. It determines whether hardware subtraction should be disabled, supplying { P0, P1, P2 } to the shader instead of { P0, P1-P0, P2-P0 }.
            uint16_t m_reserved : 3;      ///< Unused; set to zero.
        };
        // NEO mode only:
        struct
        {
            uint16_t : 12;                  ///< Description to be specified.
            uint16_t m_defaultValueHi : 2;  ///< if (m_interpF16 != 0) indicates the default value supplied to the shader for the upper 16-bits if m_semantic is not matched in the VS shader, and m_defaultValue indicates the default value for the lower 16-bits.
            uint16_t m_interpF16 : 2;       ///< if (m_interpF16 == 0) this is a 32-bit float or custom value; if (m_interpF16 & 1) the low 16-bits of this parameter expect 16-bit float interpolation and/or default value; if (m_interpF16 & 2) the high 16-bits of this parameter expect 16-bit float interpolation and/or default value
        };
    };
};

// Establish a semantic mapping between 
// VertexExportSemantic and PixelInputSemantic.
// Used to calculate pixel shader input location.
struct PixelSemanticMapping
{
    uint32_t m_outIndex : 5;
    uint32_t m_isCustom : 1;
    uint32_t m_reserved0 : 2;
    uint32_t m_defaultValue : 2;
    uint32_t m_customOrFlat : 1;  // Equal to ( m_isCustom | m_isFlatShaded ) from PixelInputSemantic
    uint32_t m_isLinear : 1;
    uint32_t m_isCustomDup : 1;  // Same as m_isCustom
    uint32_t m_reserved1 : 19;
};

#ifdef __cplusplus
class VsStageRegisters
#else  // __cplusplus
typedef struct VsStageRegisters
#endif // __cplusplus
{
#ifdef __cplusplus
public:
#endif // __cplusplus
	uint32_t           m_spiShaderPgmLoVs; ///< The pointer to shader program (bits 39:8).
	uint32_t           m_spiShaderPgmHiVs; ///< The pointer to shader program (bits 47:40). This must be set to zero.

	uint32_t           m_spiShaderPgmRsrc1Vs;
	uint32_t           m_spiShaderPgmRsrc2Vs;

	uint32_t           m_spiVsOutConfig;
	uint32_t           m_spiShaderPosFormat;
	uint32_t		   m_paClVsOutCntl;

#ifdef __cplusplus


	/** @brief Patches the GPU address of the shader code.

		@param[in] gpuAddress		This address to patch. This must be aligned to a 256-byte boundary.
	 */
	void patchShaderGpuAddress(void *gpuAddress)
	{
		m_spiShaderPgmLoVs = static_cast<uint32_t>(uintptr_t(gpuAddress) >> 8);
		m_spiShaderPgmHiVs = static_cast<uint32_t>(uintptr_t(gpuAddress) >> 40);
	}

	bool isSharingContext(const VsStageRegisters shader) const
	{
		return	!((m_spiVsOutConfig - shader.m_spiVsOutConfig)
			| (m_spiShaderPosFormat - shader.m_spiShaderPosFormat)
			| (m_paClVsOutCntl - shader.m_paClVsOutCntl));
	}

#endif // __cplusplus

#ifdef __cplusplus
};
#else // __cplusplus
		} VsStageRegisters;
#endif // __cplusplus

#ifdef __cplusplus
class PsStageRegisters
#else  // __cplusplus
typedef struct PsStageRegisters
#endif // __cplusplus
{
#ifdef __cplusplus
public:
#endif // __cplusplus
	uint32_t        m_spiShaderPgmLoPs; ///< A pointer to shader program (bits 39:8).
	uint32_t        m_spiShaderPgmHiPs; ///< A pointer to shader program (bits 47:40). This must be set to zero.

	uint32_t        m_spiShaderPgmRsrc1Ps;
	uint32_t        m_spiShaderPgmRsrc2Ps;

	uint32_t        m_spiShaderZFormat;
	uint32_t        m_spiShaderColFormat;

	uint32_t        m_spiPsInputEna;
	uint32_t        m_spiPsInputAddr;

	uint32_t        m_spiPsInControl;
	uint32_t        m_spiBarycCntl;

	uint32_t		m_dbShaderControl;
	uint32_t		m_cbShaderMask;

#ifdef __cplusplus
	void patchShaderGpuAddress(void *gpuAddress)
	{
		m_spiShaderPgmLoPs = static_cast<uint32_t>(uintptr_t(gpuAddress) >> 8);
		m_spiShaderPgmHiPs = static_cast<uint32_t>(uintptr_t(gpuAddress) >> 40);
	}
#endif // __cplusplus
#ifdef __cplusplus
};
#else // __cplusplus
		} PsStageRegisters;
#endif // __cplusplus

class VsShader
{
public:
	ShaderCommonData m_common;			///< The common data for all shader stages.

	VsStageRegisters m_vsStageRegisters;	///< The data to be loaded into the VS shader stage registers. Please see DrawCommandBuffer::setVsShader() for more information.
	// not used if domain shader => vertex shader

	uint8_t m_numInputSemantics;				///< The number of entries in the input semantic table.
	uint8_t m_numExportSemantics;				///< The number of entries in the export semantic table.
	uint8_t m_gsModeOrNumInputSemanticsCs;		///< Stores a union of VsShaderGsMode values for a VsShader or GsShader::getCopyShader(), which are translated into a GsMode constant. For CsVsShader::getVertexShader() with CsVsShader::getComputeShader()->m_version==0, the number of input semantic table entries to use for the CsVsShader::getComputeShader() fetch shader is stored.
	uint8_t m_fetchControl;						///< The user registers that receive vertex and instance offsets for use in the fetch shader.


	/** @brief Patches the GPU address of the shader code.

		@param[in] gpuAddress		This address to patch. This must be aligned to a 256-byte boundary.
	 */
	void patchShaderGpuAddress(void *gpuAddress)
	{
		m_vsStageRegisters.patchShaderGpuAddress(gpuAddress);
	}

	void *getBaseAddress() const
	{
		return (void *)((((uintptr_t)m_vsStageRegisters.m_spiShaderPgmHiVs) << 40) | (((uintptr_t)m_vsStageRegisters.m_spiShaderPgmLoVs) << 8));
	}

	/** @brief Gets a pointer to this shader's input usage slot table that immediately follows this shader's structure in memory.

		@return A pointer to this shader's input usage slot table.
		*/
	const InputUsageSlot       *getInputUsageSlotTable() const { return (const InputUsageSlot *)(this + 1); }

	/** @brief Gets a pointer to this shader's input semantic table that immediately follows the input usage table in memory.

		@return A pointer to this shader's input semantic table.
		*/
	const VertexInputSemantic  *getInputSemanticTable()  const { return (const VertexInputSemantic *)(getInputUsageSlotTable() + m_common.m_numInputUsageSlots); }

	/** @brief Gets a pointer to this shader's export semantic table that immediately follows the input semantic table in memory.

		@return A pointer to this shader's export semantic table.
		*/
	const VertexExportSemantic *getExportSemanticTable() const { return (const VertexExportSemantic *)(getInputSemanticTable() + m_numInputSemantics); }

	/** @brief Computes the total size (in bytes) of the shader binary including this structure, the input usage table, and the input and export semantic tables.

		@return The total size in bytes of this shader binary and its associated tables.
		*/
	uint32_t computeSize() const
	{
		const uint32_t size = sizeof(VsShader) +
			sizeof(InputUsageSlot) * m_common.m_numInputUsageSlots +
			sizeof(VertexInputSemantic) * m_numInputSemantics +
			sizeof(VertexExportSemantic) * m_numExportSemantics;

		return (size + 3) & ~3U;
	}
	/** @brief Gets the user register that contains the vertex offset.

		@return The index of the register containing the vertex offset. A value of 0 indicates no register contains the vertex offset.
		*/
	uint8_t getVertexOffsetUserRegister() const
	{
		return m_fetchControl & 0xf;
	}
	/** @brief Gets the user register that contains the instance offset.

		@return The index of the register containing the instance offset. A value of 0 indicates no register contains the instance offset.
		*/
	uint8_t getInstanceOffsetUserRegister() const
	{
		return (m_fetchControl >> 4) & 0xf;
	}
};


class PsShader
{
public:
	ShaderCommonData m_common;				///< The common data for all shader stages.

	PsStageRegisters  m_psStageRegisters;		///< The data to be loaded into the PS shader stage registers. Please see Gnm::DrawCommandBuffer::setPsShader() for more details.

	uint8_t              m_numInputSemantics;		///< The number of entries in the input semantic table.
	uint8_t              m_reserved[3];				///< Unused

	/** @brief Patches the GPU address of the shader code.

		@param[in] gpuAddress		The address to patch. This must be aligned to a 256-byte boundary.
	 */
	void patchShaderGpuAddress(void *gpuAddress)
	{
		m_psStageRegisters.patchShaderGpuAddress(gpuAddress);
	}

	/** @brief Retrieves the GPU address of the shader code.

		@return The address of the shader code.
		*/
	void *getBaseAddress() const
	{
		return (void *)((((uintptr_t)m_psStageRegisters.m_spiShaderPgmHiPs) << 40) | (((uintptr_t)m_psStageRegisters.m_spiShaderPgmLoPs) << 8));
	}

	/** @brief Gets a pointer to this shader's input usage slot table that immediately follows this shader's structure in memory.

		@return A pointer to this shader's input usage slot table.
		*/
	const InputUsageSlot     *getInputUsageSlotTable()     const { return (const InputUsageSlot *)(this + 1); }

	/** @brief Gets a pointer to this shader's input semantic table that immediately follows the input usage table in memory.

		@return A pointer to this shader's input semantic table.
		*/
	const PixelInputSemantic *getPixelInputSemanticTable() const { return (const PixelInputSemantic *)(getInputUsageSlotTable() + m_common.m_numInputUsageSlots); }

	/** @brief Computes the total size (in bytes) of the shader binary including this structure, the input usage table and the input semantic table.

		@return The total size in bytes of this shader binary and its associated tables.
		*/
	uint32_t computeSize() const
	{
		const uint32_t size = sizeof(PsShader) +
			sizeof(InputUsageSlot) * m_common.m_numInputUsageSlots +
			sizeof(PixelInputSemantic) * m_numInputSemantics;

		return (size + 3) & ~3U;
	}
};


static inline void print_bininfo(const ShaderBinaryInfo &bininfo) {
    printf(
		"   m_signature: %.*s\n"
		"   m_version: %x\n"
	    "   m_pssl_or_cg: %x\n"
	    "   m_cached: %x\n"
		"   m_type: %x\n"
		"   m_source_type: %x\n"
	    "   m_length: %i\n"
		"   m_chunkUsageBaseOffsetInDW: %x\n"
		"   m_numInputUsageSlots: %x\n"
        "   m_isSrt: %x\n"
        "   m_isSrtUsedInfoValid: %x\n"
        "   m_isExtendedUsageInfo: %x\n"
        "   m_reserved2: %x\n"
        "   m_reserved3: %x\n"
        "   m_shaderHash0: %x\n"
        "   m_shaderHash1: %x\n"
        "   m_crc32: %x\n",
        7, bininfo.m_signature,
        bininfo.m_version,
        bininfo.m_pssl_or_cg,
        bininfo.m_cached,
        bininfo.m_type,
        bininfo.m_source_type,
        bininfo.m_length,
        bininfo.m_chunkUsageBaseOffsetInDW,
        bininfo.m_numInputUsageSlots,
        bininfo.m_isSrt,
        bininfo.m_isSrtUsedInfoValid,
        bininfo.m_isExtendedUsageInfo,
        bininfo.m_reserved2,
        bininfo.m_reserved3,
        bininfo.m_shaderHash0,
        bininfo.m_shaderHash1,
        bininfo.m_crc32
    );
}