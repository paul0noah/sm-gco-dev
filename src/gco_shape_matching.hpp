#include <Eigen/Dense>
#include <iostream>
#include <string>
#include "helper/shape.hpp"
#ifndef GCO_SHAPE_MATCHING
#define GCO_SHAPE_MATCHING

#define SCALING_FACTOR 10000.0f

namespace smgco {

class GCOSM {
    private:
    std::string prefix;
    int numIters;
    float dataWeight;
    Eigen::MatrixXd VX;
    const Eigen::MatrixXi FX;
    Eigen::MatrixXd VY;
    const Eigen::MatrixXi FY;
    const Eigen::MatrixXd perVertexFeatureDifference; // should be |VX| x |VY|

    public:
    GCOSM(const Eigen::MatrixXd VX,
          const Eigen::MatrixXi FX,
          const Eigen::MatrixXd VY,
          const Eigen::MatrixXi FY,
          const Eigen::MatrixXd perVertexFeatureDifference);
    void printName();

    Eigen::MatrixXi pointWise(const bool smoothGeodesic=false);
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> triangleWise();

    void setDataWeight(const float newDataWeight);
};

} // namespace smgco


#endif // GCO_SHAPE_MATCHING
