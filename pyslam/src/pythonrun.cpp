
#include "pythonrun.h"  // std::exp, std::cos



PythonCodeRun::PythonCodeRun(){

}
void
PythonCodeRun::runcode(std::string pycode){
        // Start the Python interpreter
        py::scoped_interpreter guard{};
        using namespace py::literals;
        // Execute Python code, using the variables saved in `locals`
        py::exec(py::str(pycode),py::globals(), locals);
        //
        //   "(
        //
        // import matplotlib.pyplot as plt
        // plt.plot(signal)
        // plt.show()
        //
        // )",py::globals(), locals);
}
// void
// addLocals(key,val){
//         locals[key]=val;
// }
