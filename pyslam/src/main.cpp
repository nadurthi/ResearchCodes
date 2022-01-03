#include <pybind11/pybind11.h>
#include "pcl_registration.h"
// #include <pybind11/eigen.h>




#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)





//using namespace std::chrono_literals;
namespace py = pybind11;

PYBIND11_MODULE(slam, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("registrations", &registrations, R"pbdoc(
        ICP in PCL
    )pbdoc");



    m.def("add", &add,py::return_value_policy::reference_internal,R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
