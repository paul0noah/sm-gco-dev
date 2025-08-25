#include <Eigen/Dense>
#include <iostream>
#include <string>
#ifndef MY_CPP_CODE
#define MY_CPP_CODE


namespace mycppcode {


std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> myfunc0();

std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> myfunc1(Eigen::MatrixXi& input, std::string inputstring);


class MyClass {      // The class
    private:
    std::string myString;  // Attribute (string variable)
    public:             // Access specifier
    MyClass(std::string name);   // constructor
    void printName();
    
};

} // namespace mycppcode


#endif // MY_CPP_CODE
