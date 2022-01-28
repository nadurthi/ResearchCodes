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

        py::class_<BinMatchSol>(m, "BinMatchSol")
        .def_readwrite("H", &BinMatchSol::H)
        .def_readwrite("cost0", &BinMatchSol::cost0)
        .def_readwrite("cost", &BinMatchSol::cost);

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
        .def("getmatch", &BinMatch::getmatch)
        .def_readwrite("mxLVL", &BinMatch::mxLVL)
        .def_readwrite("mn_orig", &BinMatch::mn_orig)
        .def_readwrite("levels", &BinMatch::levels)
        .def_readwrite("HLevels", &BinMatch::HLevels)
        .def_readwrite("dxlevels", &BinMatch::dxlevels);


#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
