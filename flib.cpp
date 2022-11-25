#include <stdio.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <assert.h>

#include <tokenizer/tokenizer.h>
#include <ort/onnxruntime_c_api.h>
#include "flib.hpp"

extern "C" {
    const char *truncationStrategy = "only_first";

    OnnxModel::OnnxModel(string model) {
        modelFilepath = model;
    }

    // ORT API helper function
    void OnnxModel::CheckStatus(OrtStatus* status) {
        if (status != NULL) {
        const char* msg = ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        ort->ReleaseStatus(status);
        exit(1);
        }
    }

    // inference session initialization function
    void OnnxModel::initialize() {
        string instanceName = "bviolet-inference";
        std::string modelFilepath = "/usr/pgsql-14/share/embeddings.onnx";
        const OrtApi* ort = NULL;

        // env must be freed
        CheckStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instanceName.c_str(), &env));
        // sessionOptions must be freed
        CheckStatus(ort->CreateSessionOptions(&sessionOptions));
        // session must be freed
        CheckStatus(ort->CreateSession(env, modelFilepath.c_str(), sessionOptions, &session));
        CheckStatus(ort->GetAllocatorWithDefaultOptions(&myallocator));

        CheckStatus(ort->SessionGetInputCount(session, &numInputNodes));

        // mInputName must be freed using allocator
        CheckStatus(ort->SessionGetInputName(session, 0, myallocator, &mInputName));

        inputNames.push_back(mInputName);

        CheckStatus(ort->SessionGetInputTypeInfo(session, 0, &inputTypeInfo));
        CheckStatus(ort->CastTypeInfoToTensorInfo(inputTypeInfo, &inputTensorInfo));

        CheckStatus(ort->GetTensorElementType(inputTensorInfo, &inputType));
        CheckStatus(ort->GetTensorShapeElementCount(inputTensorInfo, &inputTensorSize));
        CheckStatus(ort->GetDimensionsCount(inputTensorInfo, &numDims));
        inputNodeDims.resize(numDims);

        ort->GetDimensions(inputTensorInfo, (int64_t*)inputNodeDims.data(), numDims);
        ort->ReleaseTypeInfo(inputTypeInfo); // free the memory

        CheckStatus(ort->SessionGetOutputCount(session, &numOutputNodes));
        // mOutputName must be freed using allocator
        CheckStatus(ort->SessionGetOutputName(session, 0, myallocator, &mOutputName));

        outputNames.push_back(mOutputName);

        CheckStatus(ort->SessionGetOutputTypeInfo(session, 0, &outputTypeInfo));
        CheckStatus(ort->CastTypeInfoToTensorInfo(outputTypeInfo, &outputTensorInfo));
        CheckStatus(ort->GetTensorElementType(outputTensorInfo, &outputType));

        ort->ReleaseTypeInfo(outputTypeInfo); // free the memory
    };

    // inference session termination function
    void OnnxModel::kill() {
        // free the memory
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(sessionOptions);
        ort->ReleaseEnv(env);   
    };

    // run inference on user query
    float* OnnxModel::run(string query) {
        int is_tensor;
        vector<float> input_ids;
        vector<float> input_mask;
        vector<float> segment_ids;
        vector<vector<float>> input_tensor_values;
        BertTokenizer tokenizer;
        OrtMemoryInfo* memoryInfo;
        OrtValue* input_tensor = NULL; // input_tensor must be freed
        OrtValue* output_tensor = NULL; // output_tensor must be freed

        tokenizer.encode(query, "", input_ids, input_mask, segment_ids, maxSeqLength, truncationStrategy);

        input_tensor_values.push_back(input_ids);
        input_tensor_values.push_back(input_mask);
        input_tensor_values.push_back(segment_ids);

        // memoryInfo must be freed
        CheckStatus(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo));

        CheckStatus(ort->CreateTensorWithDataAsOrtValue(memoryInfo, input_tensor_values.data(), inputTensorSize * sizeof(float), inputNodeDims.data(), numDims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        CheckStatus(ort->IsTensor(input_tensor, &is_tensor));
        assert(is_tensor);

        CheckStatus(ort->Run(session, run_options, inputNames.data(), (const OrtValue* const*)&input_tensor, numInputNodes, outputNames.data(), numOutputNodes, &output_tensor));
        CheckStatus(ort->IsTensor(output_tensor, &is_tensor));
        assert(is_tensor);

        float* floatarr;
        CheckStatus(ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
        assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

        // free the memory
        ort->ReleaseMemoryInfo(memoryInfo); 
        ort->ReleaseValue(output_tensor);
        ort->ReleaseValue(input_tensor);
        
        return floatarr;
    };
}



