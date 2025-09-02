#include "src/gco_shape_matching.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT



PYBIND11_MODULE(sm_gco, handle) {
    handle.doc() = "Python Wrappers for shape matching with gco functions";
    using namespace smgco;

    // how to wrap classes, also easy *_* (every class function has to be defined seperatly tho :/ )
    py::class_<smgco::GCOSM, std::shared_ptr<smgco::GCOSM>> myclass(handle, "GCOSM");
    myclass.def(py::init<const Eigen::MatrixXd, const Eigen::MatrixXi, const Eigen::MatrixXd, const Eigen::MatrixXi, const Eigen::MatrixXd>());
    myclass.def("point_wise", &smgco::GCOSM::pointWise);
    myclass.def("triangle_wise", &smgco::GCOSM::triangleWise);
    myclass.def("set_data_weight", &smgco::GCOSM::setDataWeight);
    myclass.def("set_max_iter", &smgco::GCOSM::setMaxIter);


    py::enum_<COST_MODE>(handle, "COST_MODE")
            .value("SINGLE_LABLE_SPACE_L2", SINGLE_LABLE_SPACE_L2)
            .value("MULTIPLE_LABLE_SPACE_L2", MULTIPLE_LABLE_SPACE_L2)
            .value("MULTIPLE_LABLE_SPACE_SO3", MULTIPLE_LABLE_SPACE_SO3)
            .value("MULTIPLE_LABLE_SPACE_SE3", MULTIPLE_LABLE_SPACE_SE3)
            .value("MULTIPLE_LABLE_SPACE_GEODIST", MULTIPLE_LABLE_SPACE_GEODIST)
            .export_values();
}
