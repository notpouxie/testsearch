#include <stdio.h>
#include <stdlib.h>
#include "flibInterface.h"

//const char* query = "молоко простоквашино";

int main() {

    MHandle model = create_OnnxModel();
    char* query = "молоко простоквашино";

    float* arr = malloc(sizeof(float) * 768);

    initialize(model);

    arr = run(model, query);

    if (arr[0] != 0.0) printf("Success!");
    else printf("Failure!");

    free(arr);
    kill(model);

    return 0;
}