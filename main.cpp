#include <iostream>
#include <cstdio>
#include <string>
#include "flib.hpp"
#include "tokenizer.h"
using namespace std;

const string query = "молоко простоквашино";

int main()
{
    OnnxModel model;
    float* arr = new float[768];

    model.initialize();

    arr = model.run(query);

    if (arr[0] != 0.0) {
        cout << "Your query " << query << " is now an array:" << endl;
        cout << arr[0] << " " << arr[1] << " .. " << arr[767] << endl;
    }
    else cout << "Error occured when running inference on a model" << endl;

    model.kill();

    return 0;
}