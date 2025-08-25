#include "cppsrc/my_cpp_code.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT



PYBIND11_MODULE(sm_gco, handle) {
    handle.doc() = "Python Wrappers for shape matching with gco functions";

    // how to wrap functions, easy huh? :D
    handle.def("myfunc0", &mycppcode::myfunc0);
    handle.def("myfunc1", &mycppcode::myfunc1);

    // how to wrap classes, also easy *_* (every class function has to be defined seperatly tho :/ )
    py::class_<mycppcode::MyClass, std::shared_ptr<mycppcode::MyClass>> myclass(handle, "MyClass");
    myclass.def(py::init<std::string>());
    myclass.def("print_name", &mycppcode::MyClass::printName);
}
