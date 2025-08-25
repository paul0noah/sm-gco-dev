#include <mex.h>

#include <igl/matlab/parse_rhs.h>
#include <iostream>
#include "cppsrc/my_cpp_code.hpp"


std::string parseStringInput(const mxArray*prhs[], const int index) {

    char* input_buf;
    size_t buflen;

    /* input must be a string */
    if ( mxIsChar(prhs[index]) != 1) {
        std::string err = "Input " + std::to_string(index+1) + " must be a char! (string is not a char)";
        mexErrMsgTxt(err.c_str());
    }

    buflen = (mxGetM(prhs[index]) * mxGetN(prhs[index])) + 1;


    /* copy the string data from prhs[0] into a C string input_ buf.    */
    input_buf = mxArrayToString(prhs[index]);
    if (input_buf == NULL) {
        std::string err = "Could not convert input " + std::to_string(index+1) + " to string";
        mexErrMsgTxt(err.c_str());
    }

    return input_buf;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) {
    const float tol = 1e-5;

    // check and read mex input
    if (false && (nrhs != 4 || nlhs > 4)) {
        mexErrMsgTxt("Usage: [Vred, Fred, Jf2c, If2c] = f2cdecimate_mex(V, F, num_faces, useQslim)");
    }

    const int funcId = (int) *mxGetPr(prhs[0]);

    if (funcId == 0) {
        std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> out = mycppcode::myfunc0();

        Eigen::MatrixXi A = std::get<0>(out);
        Eigen::MatrixXf B = std::get<1>(out);

        plhs[0] = mxCreateDoubleMatrix(A.rows(), A.cols(), mxREAL);
        std::copy(A.data(), A.data() + A.size(), mxGetPr(plhs[0]));

        plhs[1] = mxCreateDoubleMatrix(B.rows(), B.cols(), mxREAL);
        std::copy(B.data(), B.data() + B.size(), mxGetPr(plhs[1]));
    }
    else if (funcId == 1) {
        Eigen::MatrixXd INP;
        igl::matlab::parse_rhs_double(&prhs[1], INP);
        Eigen::MatrixXi INPint = INP.cast<int>();

        std::string string_data = parseStringInput(prhs, 2);

        std::tuple<Eigen::MatrixXi, Eigen::MatrixXf> out = mycppcode::myfunc1(INPint, string_data);

        Eigen::MatrixXi A = std::get<0>(out);
        Eigen::MatrixXf B = std::get<1>(out);

        plhs[0] = mxCreateDoubleMatrix(A.rows(), A.cols(), mxREAL);
        std::copy(A.data(), A.data() + A.size(), mxGetPr(plhs[0]));

        plhs[1] = mxCreateDoubleMatrix(B.rows(), B.cols(), mxREAL);
        std::copy(B.data(), B.data() + B.size(), mxGetPr(plhs[1]));
    }
    else {
        mexErrMsgTxt("Func id not supported");
    }
}
