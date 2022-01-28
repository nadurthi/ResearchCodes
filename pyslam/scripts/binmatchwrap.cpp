#include <pybind11/pybind11.h>
#include "binmatch.h"
#include "pybind11_json.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


//using namespace std::chrono_literals;
namespace py = pybind11;

PYBIND11_MODULE(binmatch, m) {
        m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";



        m.def("UpsampleMax", &UpsampleMax,R"pbdoc(
        Add two numbers Some other explanation about the add function.)pbdoc");
        m.def("computeHitogram2D", &computeHitogram2D,R"pbdoc(
        Add two numbers Some other explanation about the add function.)pbdoc");
        m.def("getPointCost", &getPointCost,R"pbdoc(
        Add two numbers Some other explanation about the add function.)pbdoc");

        m.def("takejson", &takejson,R"pbdoc(JSON direct reader)pbdoc");


        py::class_<BinMatch>(m, "BinMatch")
        .def(py::init<const std::string &>())
        .def("computeHlevels", &BinMatch::computeHlevels)
        .def("getmatch", &BinMatch::getmatch);


#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
