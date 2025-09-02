#include "gco_shape_matching.hpp"
#include <gco/GCoptimization.h>

namespace smgco {



GCOSM::GCOSM(const Eigen::MatrixXd VX_,
             const Eigen::MatrixXi FX,
             const Eigen::MatrixXd VY_,
             const Eigen::MatrixXi FY,
             const Eigen::MatrixXd perVertexFeatureDifference) :
VX(VX_), FX(FX), VY(VY_), FY(FY), perVertexFeatureDifference(perVertexFeatureDifference)
{
    prefix = "[GCOSM] ";
    numIters = -1;
    dataWeight = 1.0;

    std::cout << prefix << "Mean centering shapes ..." << std::endl;
    VX = VX.rowwise() - VX.colwise().mean();
    VY = VY.rowwise() - VY.colwise().mean();
    std::cout << prefix << "Done" << std::endl;

}

void GCOSM::printName() {
    std::cout << prefix << std::endl;
}

void GCOSM::setDataWeight(const float newDataWeight) {
    dataWeight = newDataWeight;
}

void GCOSM::setMaxIter(const int newMaxIter) {
    numIters = newMaxIter;
}

} // namespace smgco
