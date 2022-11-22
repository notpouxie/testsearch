#include <stdio.h>
#include <stdlib.h>
#include <string>
using std :: string;
#include <numeric>
#include <assert.h>
#include <span>
#include <vector>
#include <ort/onnxruntime_c_api.h>
#include "tokenizer.h"


extern "C" {

    #include <postgres.h>
    #include "catalog/pg_type.h"
    #include <fmgr.h>
    #include "utils/builtins.h"
    #include "utils/array.h"
    #include "utils/lsyscache.h"



    #ifndef PG_MODULE_MAGIC 
    PG_MODULE_MAGIC;
    #endif

    void _PG_init(void);
    void _PG_fini(void);

    const OrtApi* ort = NULL;
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtSession* session;
    OrtRunOptions* run_options{nullptr};
    OrtAllocator* myallocator;

    // initialize tokenizer
    BertTokenizer tokenizer;
    BasicTokenizer basictokenizer;

    const char *truncation_strategy = "only_first";

    char* mInputName;
    char* mOutputName;

    int max_seq_length = 512;

    vector<const char*> inputNames;
    vector<const char*> outputNames;

    vector<int64_t> inputNodeDims;

    size_t numDims, numInputNodes, numOutputNodes, inputTensorSize;

    ArrayType *queryArr, *productArr;

    // ORT API helper function
    void CheckStatus(OrtStatus* status) {
    if (status != NULL) {
      const char* msg = ort->GetErrorMessage(status);
      ereport(ERROR, (errmsg(msg)));
      ort->ReleaseStatus(status);
      exit(1);
    }
    }

    // tokenization function

    PG_FUNCTION_INFO_V1(pg_run_session);

    Datum
    pg_run_session(PG_FUNCTION_ARGS) {
        char *user_query;
        user_query = text_to_cstring(PG_GETARG_TEXT_PP(0));

        vector<float> input_ids;
        vector<float> input_mask;
        vector<float> segment_ids;
        vector<vector<float>> input_tensor_values;
        /*vector<vector<float>> input_ids_list;
        vector<vector<float>> input_mask_list;
        vector<vector<float>> segment_ids_list;*/

        tokenizer.encode(user_query, "", input_ids, input_mask, segment_ids, max_seq_length, truncation_strategy);

        input_tensor_values.push_back(input_ids);
        input_tensor_values.push_back(input_mask);
        input_tensor_values.push_back(segment_ids);

        OrtMemoryInfo* memoryInfo;

        CheckStatus(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo));

        //OrtValue* input_ids_tensor = NULL;    
        //OrtValue* input_mask_tensor = NULL;
        //OrtValue* segment_ids_tensor = NULL;

        OrtValue* input_tensor = NULL;
        CheckStatus(ort->CreateTensorWithDataAsOrtValue(memoryInfo, input_tensor_values.data(), inputTensorSize * sizeof(float), inputNodeDims.data(), numDims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

        //CheckStatus(ort->CreateTensorWithDataAsOrtValue(memoryInfo, input_ids.data(), input_ids.size(), inputNodeDims.data(), numDims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ids_tensor));
        //CheckStatus(ort->CreateTensorWithDataAsOrtValue(memoryInfo, input_mask.data(), input_mask.size(), inputNodeDims.data(), numDims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_mask_tensor));
        //CheckStatus(ort->CreateTensorWithDataAsOrtValue(memoryInfo, segment_ids.data(), segment_ids.size(), inputNodeDims.data(), numDims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &segment_ids_tensor));
        // had input_tensor[] initially
        //OrtValue* input_tensor = {std::move(input_ids_tensor), std::move(input_mask_tensor), std::move(segment_ids_tensor)};

        int is_tensor;
        CheckStatus(ort->IsTensor(input_tensor, &is_tensor));
        assert(is_tensor);

        OrtValue* output_tensor = NULL;

        CheckStatus(ort->Run(session, run_options, inputNames.data(), (const OrtValue* const*)&input_tensor, numInputNodes, outputNames.data(), numOutputNodes, &output_tensor));
        CheckStatus(ort->IsTensor(output_tensor, &is_tensor));
        assert(is_tensor);

        float* floatarr;
        CheckStatus(ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
        assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

        // release
        ort->ReleaseMemoryInfo(memoryInfo);

        //ort->ReleaseValue(input_ids_tensor);
        //ort->ReleaseValue(input_mask_tensor);
        //ort->ReleaseValue(segment_ids_tensor);
        ort->ReleaseValue(output_tensor);
        ort->ReleaseValue(input_tensor);

        PG_RETURN_ARRAYTYPE_P(floatarr);
        // if this doesn't work as intended - try solution from "ArrayType from  int[]" bookmark
    }

    // dot product function

    PG_FUNCTION_INFO_V1(pg_dot_product);

    Datum
    pg_dot_product(PG_FUNCTION_ARGS)
    {
        float product = 0.0;
        bool isNull = false;
        queryArr = PG_GETARG_ARRAYTYPE_P(0);
        productArr = PG_GETARG_ARRAYTYPE_P(1);

        Oid q_elemTypeId = ARR_ELEMTYPE(queryArr);
        Oid p_elemTypeId = ARR_ELEMTYPE(productArr);

        float q_element, p_element;
        int16 q_typlen, p_typlen;
        bool q_typbyval, p_typbyval;
        char q_typalign, p_typalign;
        

        if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
            PG_RETURN_NULL();
        }

        if (ARR_NDIM(queryArr) == 0 || (ARR_NDIM(productArr) == 0)) {
            PG_RETURN_NULL();
        }
        if (ARR_NDIM(queryArr) > 1 || (ARR_NDIM(productArr) > 1)) {
            ereport(ERROR, (errmsg("pg_dot_product: one-dimensional arrays are required")));
        }

        Oid elemTypeId = ARR_ELEMTYPE(queryArr);

        if (elemTypeId != ARR_ELEMTYPE(productArr)) {
            ereport(ERROR, (errmsg("pg_dot_product: input arrays must be of the same type")));
        }

        if (ARR_SIZE(queryArr) == ARR_SIZE(productArr)) {

            get_typlenbyvalalign(q_elemTypeId, &q_typlen, &q_typbyval, &q_typalign);
            get_typlenbyvalalign(p_elemTypeId, &p_typlen, &p_typbyval, &p_typalign);

            for (int i = 0; i < ARR_SIZE(queryArr); i++) {
                q_element = DatumGetFloat4(array_ref(queryArr, 1, &i, -1, q_typlen, q_typbyval, q_typalign, &isNull));
                p_element = DatumGetFloat4(array_ref(productArr, 1, &i, -1, p_typlen, p_typbyval, p_typalign, &isNull));
                product += (q_element * p_element);
            }

        }
        else {
            PG_RETURN_NULL();
        }
        
        PG_RETURN_FLOAT4(product);
    }
    
    void _PG_init(void) {

        // if std::string doesn't work, use const char* instead (it's faster aswell)
        std::string modelFilepath = "/usr/pgsql-14/share/embeddings.onnx";
        std::string instance_name = "bviolet-inference";

        // create an OrtEnv (should be freed!)
        CheckStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str(), &env));

        // create OrtSessionOptions (should be freed!)
        CheckStatus(ort->CreateSessionOptions(&session_options));        

        // create a new inference session (should be freed!)
        CheckStatus(ort->CreateSession(env, modelFilepath.c_str(), session_options, &session));

        // create allocator to get model info (should NOT be freed)
        CheckStatus(ort->GetAllocatorWithDefaultOptions(&myallocator));

        /*** Inputs ***/        
        OrtTypeInfo* inputTypeInfo;
        OrtTypeInfo* outputTypeInfo;

        CheckStatus(ort->SessionGetInputCount(session, &numInputNodes));

        // it's said that these variables must be freed using allocator.
        // FIND OUT HOW TO!
        CheckStatus(ort->SessionGetInputName(session, 0, myallocator, &mInputName));
        CheckStatus(ort->SessionGetOutputName(session, 0, myallocator, &mOutputName));

        inputNames.push_back(mInputName);
        outputNames.push_back(mOutputName);

        // get input type information (should be freed!)
        CheckStatus(ort->SessionGetInputTypeInfo(session, 0, &inputTypeInfo));

        const OrtTensorTypeAndShapeInfo* inputTensorInfo;

        CheckStatus(ort->CastTypeInfoToTensorInfo(inputTypeInfo, &inputTensorInfo));
        //deprecated
        //auto inputTensorInfo = inputTypeInfo->GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType inputType;
        CheckStatus(ort->GetTensorElementType(inputTensorInfo, &inputType)); 

        CheckStatus(ort->GetTensorShapeElementCount(inputTensorInfo, &inputTensorSize));

        CheckStatus(ort->GetDimensionsCount(inputTensorInfo, &numDims));

        inputNodeDims.resize(numDims);

        ort->GetDimensions(inputTensorInfo, (int64_t*)inputNodeDims.data(), numDims);

        ort->ReleaseTypeInfo(inputTypeInfo);

        /*** Outputs ***/
        CheckStatus(ort->SessionGetOutputCount(session, &numOutputNodes));;

        CheckStatus(ort->SessionGetOutputTypeInfo(session, 0, &outputTypeInfo));

        const OrtTensorTypeAndShapeInfo* outputTensorInfo;

        CheckStatus(ort->CastTypeInfoToTensorInfo(outputTypeInfo, &outputTensorInfo));
        //deprecated
        //auto outputTensorInfo = outputTypeInfo->GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType;
        CheckStatus(ort->GetTensorElementType(outputTensorInfo, &outputType)); 

        ort->ReleaseTypeInfo(outputTypeInfo);

        // may not be needed, in that case - remove it
        //outputNodeDims = outputTensorInfo.GetShape(); //and then that

    }

    void _PG_fini(void) {
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);   
    }
}