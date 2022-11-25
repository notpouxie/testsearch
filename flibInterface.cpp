#include "flibInterface.h"
#include "flib.hpp"

extern "C" {
    MHandle create_OnnxModel() {
        return (OnnxModel*) new OnnxModel();
    };
    MHandle create_OnnxModel_fname(char* query) {
        return (OnnxModel*) new OnnxModel(query);
    };
    void initialize(MHandle p){
        ((OnnxModel*) p)->initialize();
    };
    void kill(MHandle p){
        ((OnnxModel*) p)->kill();
    };
    float* run(MHandle p, char* query){
        return ((OnnxModel*) p)->run(query);
    }
}