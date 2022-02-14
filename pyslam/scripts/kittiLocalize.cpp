
#include "measmapmanagers.h"
#include <pybind11/pybind11.h>
#include "pybind11_json.h"



//using namespace std::chrono_literals;
namespace py = pybind11;

PYBIND11_MODULE(kittilocalize, m) {
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
        .def_readwrite("lvl", &BinMatchSol::lvl)
        .def_readwrite("mxLVL", &BinMatchSol::mxLVL)
        .def_readwrite("cost", &BinMatchSol::cost);

        py::class_<BMatchAndCorrH>(m, "BMatchAndCorrH")
        .def_readwrite("sols", &BMatchAndCorrH::sols)
        .def_readwrite("gHk_corr", &BMatchAndCorrH::gHk_corr);

        py::class_<MapLocalizer>(m, "MapLocalizer")
        .def(py::init<std::string optstr>())
        .def("setOptions", &MapLocalizer::setOptions)
        .def("resetH", &MapLocalizer::resetH)
        .def("addMeas", &MapLocalizer::addMeas)
        .def("addMap", &MapLocalizer::addMap)
        .def("addMap2D", &MapLocalizer::addMap2D)
        .def("setHlevels", &MapLocalizer::setHlevels)
        .def("setgHk", &MapLocalizer::setgHk)
        .def("setLookUpDist", &MapLocalizer::setLookUpDist)
        .def("setRegisteredSeqH", &MapLocalizer::setRegisteredSeqH)
        .def("setRelStates", &MapLocalizer::setRelStates)
        .def("setgHk", &MapLocalizer::setgHk)

        .def("getmeas_eigen", &MapLocalizer::getmeas_eigen)
        .def("MapPcllimits", &MapLocalizer::MapPcllimits)
        .def("getdt", &MapLocalizer::getdt)
        .def("getmaplocal_eigen", &MapLocalizer::getmaplocal_eigen)
        .def("getmap_eigen", &MapLocalizer::getmap_eigen)
        .def("getmap2D_eigen", &MapLocalizer::getmap2D_eigen)
        .def("getvelocities", &MapLocalizer::getvelocities)
        .def("getpositions", &MapLocalizer::getpositions)
        .def("getangularvelocities", &MapLocalizer::getangularvelocities)
        .def("getLikelihoods", &MapLocalizer::getLikelihoods)
        .def("getSeq_gHk", &MapLocalizer::getSeq_gHk)
        .def("getsetSeq_gHk", &MapLocalizer::getsetSeq_gHk)
        .def("getalignSeqMeas_eigen", &MapLocalizer::getalignSeqMeas_eigen)

        .def("BMatchseq", &MapLocalizer::BMatchseq)
        .def("gicp_correction", &MapLocalizer::gicp_correction);


#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
