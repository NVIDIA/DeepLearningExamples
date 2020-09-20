// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the NVIDIA CORPORATION nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include <iostream>
#include <cstring>
#include <assert.h>

using namespace std;
using namespace nvinfer1;

class AddPosEncPlugin: public IPluginV2IOExt {
public:
    AddPosEncPlugin() {};

    AddPosEncPlugin(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }

    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2Ext* clone() const override {
        return new AddPosEncPlugin(&m, sizeof(m));
    }

    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override {
        return pInputDim[0];
    }

    size_t getWorkspaceSize(int nBatch) const override {return 0;}
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "AddPosEncPlugin";}
    const char* getPluginVersion() const override {return "0.0.1";}

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
    {
        m.inputDim = in[0].dims;
        m.dataType = in[0].type;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
    {
        assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= ((inOut[pos].type == DataType::kFLOAT) 
            // || (inOut[pos].type == DataType::kHALF)
        );

        switch (pos) {
            case 0:  // input
                condition &= ((inOut[pos].type == DataType::kFLOAT)  // for seq in fp32
                    || (inOut[pos].type == DataType::kHALF));  // for seq in fp16
                break;
            case 1:  // output
                condition &= ((inOut[pos].type == inOut[0].type));  // the same type as the input
                break;
        }

        return condition;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override
    {
        return inputTypes[0];
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const override
    {
        return false;
    }

private:
    struct {
        Dims inputDim;
        DataType dataType;
    } m;
};

class AddPosEncPluginCreator : public nvinfer1::IPluginCreator {
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new AddPosEncPlugin(serialData, serialLength);
    }
    
    const char* getPluginName() const override {return "AddPosEncPlugin";}
    const char* getPluginVersion() const override {return "0.0.1";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return nullptr;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        return new AddPosEncPlugin();
    }
};
