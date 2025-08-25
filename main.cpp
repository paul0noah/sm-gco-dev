#include <igl/decimate.h>
#include <igl/read_triangle_mesh.h>
#include <igl/find.h>
#include <igl/colon.h>
#include <igl/triangle_triangle_adjacency.h>
#include <Eigen/Dense>
#include "cppsrc/my_cpp_code.hpp"

int main(int argc, char *argv[]) {

    mycppcode::MyClass clas("hello");
    clas.printName();

    std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> out = mycppcode::myfunc0();

}
