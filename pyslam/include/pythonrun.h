#pragma once
#include "base.h"


namespace py = pybind11;

class PythonCodeRun {
public:
PythonCodeRun();
void runcode(std::string pycode);

// void addLocals(key,val){
//         locals[key]=val;
// }

py::dict locals;
};
