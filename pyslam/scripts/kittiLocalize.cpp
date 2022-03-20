
#include "measmapmanagers.h"
#include <pybind11/pybind11.h>
#include "pybind11_json.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

//using namespace std::chrono_literals;
namespace py = pybind11;

PYBIND11_MODULE(kittilocal, m) {
        m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

        m.def("pose2Hmat", &pose2Hmat, R"pbdoc(
              ICP in PCL
          )pbdoc");
        m.def("Hmat2pose", &Hmat2pose, R"pbdoc(
                ICP in PCL
            )pbdoc");


        py::class_<BinMatchSol>(m, "BinMatchSol")
        .def_readwrite("H", &BinMatchSol::H)
        .def_readwrite("cost0", &BinMatchSol::cost0)
        .def_readwrite("lvl", &BinMatchSol::lvl)
        .def_readwrite("mxLVL", &BinMatchSol::mxLVL)
        .def_readwrite("cost", &BinMatchSol::cost);


        py::class_<SolBox>(m, "SolBox")
        .def_readwrite("lb", &SolBox::lb)
        .def_readwrite("dx", &SolBox::dx)
        .def_readwrite("cost", &SolBox::cost)
        .def_readwrite("lvl", &SolBox::lvl)
        .def_readwrite("th", &SolBox::th)
        .def_readwrite("flg", &SolBox::flg);

        py::class_<structmeas>(m, "structmeas")
        .def_readwrite("dt", &structmeas::dt)
        .def_readwrite("tk", &structmeas::tk)
        .def_readwrite("X1v", &structmeas::X1v)
        .def_readwrite("X1gv", &structmeas::X1gv)
        .def_readwrite("X1v_roadrem", &structmeas::X1v_roadrem)
        .def_readwrite("X1gv_roadrem", &structmeas::X1gv_roadrem);



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
        .def_readwrite("t0", &BinMatch::t0)
        .def_readwrite("H12mn", &BinMatch::H12mn)
        .def_readwrite("Xth", &BinMatch::Xth)
        .def_readwrite("HLevels", &BinMatch::HLevels)
        .def_readwrite("qvinit", &BinMatch::qvinit)
        .def_readwrite("dxlevels", &BinMatch::dxlevels);


        py::class_<BMatchAndCorrH_async>(m, "BMatchAndCorrH_async")
        .def_readwrite("bmHsol", &BMatchAndCorrH_async::bmHsol)
        .def_readwrite("tk", &BMatchAndCorrH_async::tk)
        .def_readwrite("t0", &BMatchAndCorrH_async::t0)
        .def_readwrite("tf", &BMatchAndCorrH_async::tf)
        .def_readwrite("do_gicp", &BMatchAndCorrH_async::do_gicp)
        .def_readwrite("gHkest_initial", &BMatchAndCorrH_async::gHkest_initial)
        .def_readwrite("gHkest_final", &BMatchAndCorrH_async::gHkest_final)
        .def_readwrite("time_taken", &BMatchAndCorrH_async::time_taken)
        .def_readwrite("isDone", &BMatchAndCorrH_async::isDone);


        py::class_<BMatchAndCorrH>(m, "BMatchAndCorrH")
        .def_readwrite("sols", &BMatchAndCorrH::sols)
        .def_readwrite("gHkcorr", &BMatchAndCorrH::gHkcorr)
        .def_readwrite("isDone", &BMatchAndCorrH::isDone);

        py::class_<MapLocalizer>(m, "MapLocalizer")
        .def(py::init<const std::string &>())
        .def("setOptions", &MapLocalizer::setOptions)
        .def("setOptions_noreset", &MapLocalizer::setOptions_noreset)

        .def("setBMOptions", &MapLocalizer::setBMOptions)
        .def("cleanUp", &MapLocalizer::cleanUp)
        .def("autoReadMeas", &MapLocalizer::autoReadMeas)
        .def("autoReadMeas_async", &MapLocalizer::autoReadMeas_async)

        .def("getMeasQ_eigen", &MapLocalizer::getMeasQ_eigen)
        .def("setquitsim", &MapLocalizer::setquitsim)

        .def("plotsim",&MapLocalizer::plotsim)
        .def("resetH", &MapLocalizer::resetH)
        .def("addMeas_fromQ", &MapLocalizer::addMeas_fromQ)
        .def("addMeas", &MapLocalizer::addMeas)
        .def("addMap", &MapLocalizer::addMap)
        .def("addMap2D", &MapLocalizer::addMap2D)
        .def("setgHk", &MapLocalizer::setgHk)
        .def("setLookUpDist", &MapLocalizer::setLookUpDist)
        .def("setRegisteredSeqH", &MapLocalizer::setRegisteredSeqH)
        .def("setRegisteredSeqH_async", &MapLocalizer::setRegisteredSeqH_async)

        .def("setRelStates", &MapLocalizer::setRelStates)
        .def("setRelStates_async", &MapLocalizer::setRelStates)
        .def("setSeq_gHk", &MapLocalizer::setSeq_gHk)
        .def("computeHlevels", &MapLocalizer::computeHlevels)

        .def("getmeas_eigen", &MapLocalizer::getmeas_eigen)
        .def("MapPcllimits", &MapLocalizer::MapPcllimits)
        .def("getdt", &MapLocalizer::getdt)
        .def("getmaplocal_eigen", &MapLocalizer::getmaplocal_eigen)
        .def("getmap_eigen", &MapLocalizer::getmap_eigen)
        .def("getmap2D_eigen", &MapLocalizer::getmap2D_eigen)
        .def("getvelocities", &MapLocalizer::getvelocities)
        .def("getpositions", &MapLocalizer::getpositions)
        .def("getangularvelocities", &MapLocalizer::getangularvelocities)
        .def("getLikelihoods_octree", &MapLocalizer::getLikelihoods_octree)
        .def("getLikelihoods_lookup", &MapLocalizer::getLikelihoods_lookup)
        .def("getSeq_gHk", &MapLocalizer::getSeq_gHk)
        .def("geti1Hi_seq_vec", &MapLocalizer::geti1Hi_seq_vec)
        .def("getmap2D_noroad_res_eigen", &MapLocalizer::getmap2D_noroad_res_eigen)

        .def("getsetSeq_gHk", &MapLocalizer::getsetSeq_gHk)
        .def("getalignSeqMeas_eigen", &MapLocalizer::getalignSeqMeas_eigen)
        .def("getalignSeqMeas_noroad_eigen", &MapLocalizer::getalignSeqMeas_noroad_eigen)
        .def("BMatchseq", &MapLocalizer::BMatchseq)

        .def("BMatchseq_async", &MapLocalizer::BMatchseq_async)
        .def("getBMatchseq_async", &MapLocalizer::getBMatchseq_async)
        .def("BMatchseq_async_caller", &MapLocalizer::BMatchseq_async_caller)
        .def("gettimers", &MapLocalizer::gettimers);



#ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
