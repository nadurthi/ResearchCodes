#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 




int add(int i, int j) {	

// Note that you need the GIL to access any Python machinery (including returning the result). 
// So before releasing it, make sure to convert all the data you need from Python types to C++ types.

  pybind11::gil_scoped_release release;

// while (true)
// {
    // do something and break
// }

int k= i + j;

pybind11::gil_scoped_acquire acquire;

return k;
}








PYBIND11_MODULE(pclwrapper, m) {
    py::class_<PCLviewerPy>(m, "PCLviewerPy")
        .def(py::init<const std::string &>())
        .def("setName", &PCLviewerPy::setName)
        .def("getName", &PCLviewerPy::getName)
        .def("__repr__",
        [](const Pet &a) {
            return "<example.Pet named '" + a.name + "'>";
        });


    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: pclwrapper

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.doc() = "Add two vectors using pybind11"; // optional module docstring
    m.def("add_arrays", &add_arrays, "Add two NumPy arrays");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

