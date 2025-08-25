#include "my_cpp_code.hpp"

namespace mycppcode {


std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> myfunc0() {
    Eigen::MatrixXi A(10, 2);
    A.setConstant(-1);

    Eigen::MatrixXf B(10, 10);
    B.setConstant(123);

    return std::make_tuple(A, B);
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> myfunc1(Eigen::MatrixXi& input, std::string inputstring) {


    std::cout <<inputstring << std::endl;

    if (input.rows() < 1) {
        std::cout << "input should not be empty" << std::endl;
    }

    Eigen::MatrixXi A(10, 2);
    A.setConstant(-1);

    A(0, 0) = input(0, 0);

    Eigen::MatrixXf B(10, 10);
    B.setConstant(123);

    return std::make_tuple(A, B);
}


MyClass::MyClass(std::string name) : myString(name) {

}

void MyClass::printName() {
    std::cout << myString << std::endl;
}


} // namespace mycppcode
