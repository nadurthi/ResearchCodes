#include <pybind11/pybind11.h>
#include "pcl_registration.h"
#include "localize.h"
#include "donseg.h"



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

    m.def("donsegmentation", &donsegmentation, R"pbdoc(
        ICP in PCL
    )pbdoc");


    m.def("add", &add,py::return_value_policy::reference_internal,R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");



    py::class_<Localize>(m, "Localize")
        .def(py::init<const std::string &>())
        .def("setMapX", &Localize::setMapX)
        .def("computeLikelihood", &Localize::computeLikelihood);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
