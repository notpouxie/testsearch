#ifndef flib_hpp__
#define flib_hpp__
#define maxSeqLength 512

#include <stdio.h>
#include <string>
#include <vector>
#include <ort/onnxruntime_c_api.h>

using namespace std;

constexpr const OrtApi* ort = NULL;
static const OrtTensorTypeAndShapeInfo* inputTensorInfo, *outputTensorInfo;

class OnnxModel
{
public:
    OrtEnv* env;
    OrtSessionOptions* sessionOptions;
    OrtSession* session;
    OrtRunOptions* run_options{nullptr};
    OrtAllocator* myallocator;
    OrtTypeInfo* inputTypeInfo;
    OrtTypeInfo* outputTypeInfo;
    ONNXTensorElementDataType inputType, outputType;
    char* mInputName;
    char* mOutputName;
    string modelFilepath;
    vector<const char*> inputNames;
    vector<const char*> outputNames;
    vector<int64_t> inputNodeDims;
    size_t numDims, numInputNodes, numOutputNodes, inputTensorSize;

    void CheckStatus(OrtStatus* status);
    void initialize();
    void kill();
    float* run(string query);

    OnnxModel() {};
    OnnxModel(string model);
};

#endif

