#ifdef __cplusplus
extern "C" {
#endif

    typedef void * MHandle;
    MHandle create_OnnxModel();
    MHandle create_OnnxModel_fname(char* query);
    void initialize(MHandle);
    void kill(MHandle);
    float* run(MHandle, char* query);

#ifdef __cplusplus
}
#endif