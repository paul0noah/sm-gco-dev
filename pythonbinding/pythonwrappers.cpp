#include "src/gco_shape_matching.hpp"
#include "helper/graph_cycles.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT



PYBIND11_MODULE(sm_gco, handle) {
    handle.doc() = "Python Wrappers for shape matching with gco functions";


    handle.def("get_cycle_triangles", &utils::getCycleTriangles);
    using namespace smgco;

    // how to wrap classes, also easy *_* (every class function has to be defined seperatly tho :/ )
    py::class_<smgco::GCOSM, std::shared_ptr<smgco::GCOSM>> myclass(handle, "GCOSM");
    myclass.def(py::init<const Eigen::MatrixXd, const Eigen::MatrixXi, const Eigen::MatrixXd, const Eigen::MatrixXi, const Eigen::MatrixXd>());
    myclass.def("point_wise", &smgco::GCOSM::pointWise);
    myclass.def("triangle_wise", py::overload_cast<>(&smgco::GCOSM::triangleWise));
    myclass.def("triangle_wise", py::overload_cast<TriangleWiseOpts>(&smgco::GCOSM::triangleWise));
    myclass.def("set_data_weight", &smgco::GCOSM::setDataWeight);
    myclass.def("set_max_iter", &smgco::GCOSM::setMaxIter);


    py::enum_<COST_MODE>(handle, "COST_MODE")
            .value("SINGLE_LABLE_SPACE_L2", SINGLE_LABLE_SPACE_L2)
            .value("MULTIPLE_LABLE_SPACE_L2", MULTIPLE_LABLE_SPACE_L2)
            .value("MULTIPLE_LABLE_SPACE_SO3", MULTIPLE_LABLE_SPACE_SO3)
            .value("MULTIPLE_LABLE_SPACE_SE3", MULTIPLE_LABLE_SPACE_SE3)
            .value("MULTIPLE_LABLE_SPACE_GEODIST", MULTIPLE_LABLE_SPACE_GEODIST)
            .export_values();


    py::class_<TriangleWiseOpts>(handle, "TriangleWiseOpts")
            .def(py::init<>())
            .def_readwrite("cost_mode", &TriangleWiseOpts::costMode)
            .def_readwrite("smooth_scale_before_robust", &TriangleWiseOpts::smoothScaleBeforeRobust)
            .def_readwrite("robust_cost", &TriangleWiseOpts::robustCost)
            .def_readwrite("lambda_se_3", &TriangleWiseOpts::lambdaSe3)
            .def_readwrite("lambda_so_3", &TriangleWiseOpts::lambdaSo3)
            .def_readwrite("unary_weight", &TriangleWiseOpts::unaryWeight)
            .def_readwrite("smooth_weight", &TriangleWiseOpts::smoothWeight)
            .def_readwrite("set_initial_lables", &TriangleWiseOpts::setInitialLables)
            .def_readwrite("lable_space_cycle_size", &TriangleWiseOpts::lableSpaceCycleSize)
            .def_readwrite("lable_space_degenerate", &TriangleWiseOpts::lableSpaceDegnerate)
            .def_readwrite("lable_space_angle_thres", &TriangleWiseOpts::lableSpaceAngleThreshold)
            .def_readwrite("membrane_energy_weight", &TriangleWiseOpts::membraneFactor)
            .def_readwrite("bending_energy_weight", &TriangleWiseOpts::bendingFactor)
            .def_readwrite("wks_energy_weight", &TriangleWiseOpts::wksFactor);

}
