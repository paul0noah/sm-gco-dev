
#include "helper/utils.hpp"
#include "src/gco_shape_matching.hpp"

int main(int argc, char *argv[]) {

    auto X = utils::getTestShapeX();
    Eigen::MatrixXd VX = std::get<0>(X);
    Eigen::MatrixXi FX = std::get<1>(X);
    const auto Y = utils::getTestShapeY();
    Eigen::MatrixXd VY = std::get<0>(Y);
    Eigen::MatrixXi FY = std::get<1>(Y);
    const Eigen::MatrixXd featDiff = utils::getFeatureDiffXY();


    smgco::GCOSM smGCO(VX, FX, VY, FY, featDiff);

    std::cout << smGCO.pointWise(true) << std::endl;
    std::cout << std::get<1>(smGCO.triangleWise()) << std::endl;
}
