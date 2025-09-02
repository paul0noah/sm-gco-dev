#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <gco/GCoptimization.h>
#include "helper/shape.hpp"
#ifndef GCO_SHAPE_MATCHING
#define GCO_SHAPE_MATCHING

#define SCALING_FACTOR 10000.0f
typedef Eigen::MatrixX<std::tuple<int, int>> TupleMatrixInt;

namespace smgco {
enum COST_MODE {
    SINGLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_SO3,
    MULTIPLE_LABLE_SPACE_SE3,
    MULTIPLE_LABLE_SPACE_GEODIST,
};

typedef struct GCOTrianglewiseExtra {
    COST_MODE costMode;
    // not all of the below matrices are needed for all cost modes
    float lambda;
    int numLables;
    Eigen::MatrixXf p2pDeformation;
    Eigen::MatrixXf VX;
    Eigen::MatrixXi FX;
    Eigen::MatrixXi LableFY;
    TupleMatrixInt commonVXofFX;
    Eigen::MatrixX<Eigen::Quaterniond> quaternoinsXtoY;
    Eigen::MatrixX<Eigen::Vector3f> translationsXtoY;
    Eigen::MatrixXi lableToIndex;
    Eigen::MatrixXf geoDistY;
} GCOTrianglewiseExtra;



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
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> triangleWise(const int costMode);

    void setDataWeight(const float newDataWeight);
    void setMaxIter(const int newMaxIter);
};


void precomputeSmoothCost(const Eigen::MatrixXd& VX,
                          const Eigen::MatrixXi& FX,
                          const Eigen::MatrixXd& VY,
                          const Eigen::MatrixXi& FY,
                          const Eigen::MatrixXi& lableSpace,
                          GCOTrianglewiseExtra& extraData);

/*






 */
inline
GCoptimization::EnergyTermType smoothFnGCOSMTrianglewise(GCoptimization::SiteID s1,
                                                         GCoptimization::SiteID s2,
                                                         GCoptimization::LabelID l1,
                                                         GCoptimization::LabelID l2,
                                                         void* extraDataVoid) {
    GCOTrianglewiseExtra* extraData = static_cast<GCOTrianglewiseExtra*>(extraDataVoid);
    const COST_MODE costMode = extraData->costMode;


    float diff = 0;

    if (costMode == SINGLE_LABLE_SPACE_L2) {
        diff = extraData->p2pDeformation(l1, l2);
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_L2) {
        const std::tuple<int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenSites);
        const int idxX2 = std::get<1>(commonVerticesBetweenSites);
        //const int s1id1 = std::get<2>(commonVerticesBetweenSites);
        //const int s2id1 = std::get<3>(commonVerticesBetweenSites);

        /*const int s1vy0 = extraData->LableFY(l1, s1id0);
         const int s1vy1 = extraData->LableFY(l1, s1id1);
         const int s2vy0 = extraData->LableFY(l2, s2id0);
         const int s2vy1 = extraData->LableFY(l2, s2id1);*/


        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);

        const Eigen::Vector3f tranlsation1 = extraData->translationsXtoY(rowIndex1, colIndex1);
        const Eigen::Vector3f tranlsation2 = extraData->translationsXtoY(rowIndex2, colIndex2);

        Eigen::MatrixXf defTri1 = (Eigen::MatrixXf(2, 3) << extraData->VX.row(idxX1), extraData->VX.row(idxX2)).finished();
        Eigen::MatrixXf defTri2 = defTri1.rowwise() + tranlsation2.transpose();
        defTri1 = defTri1.rowwise() + tranlsation1.transpose();

        const float diff11_21 = (defTri1.row(0) - defTri2.row(0)).norm();
        const float diff11_22 = (defTri1.row(0) - defTri2.row(1)).norm();
        const float diff12_21 = (defTri1.row(1) - defTri2.row(0)).norm();
        const float diff12_22 = (defTri1.row(1) - defTri2.row(1)).norm();

        diff = std::max({diff11_21, diff11_22, diff12_21, diff12_22});

    }
    else if (costMode == MULTIPLE_LABLE_SPACE_SO3) {
        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);
        const Eigen::Quaterniond rotation1 = extraData->quaternoinsXtoY(rowIndex1, colIndex1);
        const Eigen::Quaterniond rotation2 = extraData->quaternoinsXtoY(rowIndex2, colIndex2);
        double innerProductRot = fabs(rotation1.w() * rotation2.w() + rotation1.vec().dot(rotation2.vec()));
        innerProductRot = std::max(std::min(1.0, innerProductRot), -1.0); // clip into value range [-1, ..., 1]

        //diff = 2 * acos(innerProductRot);
        diff = acos(innerProductRot);
        //std::cout << s1 << ", " << s2 << ": " << ", " << l1 << ", " << l2 << ", " << diff << std::endl;
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_SE3) {
        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);
        const Eigen::Quaterniond rotation1 = extraData->quaternoinsXtoY(rowIndex1, colIndex1);
        const Eigen::Matrix3f rot1 = rotation1.toRotationMatrix().cast<float>();
        const Eigen::Quaterniond rotation2 = extraData->quaternoinsXtoY(rowIndex2, colIndex2);
        const Eigen::Matrix3f rot2 = rotation2.toRotationMatrix().cast<float>();

        const std::tuple<int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenSites);
        const int idxX2 = std::get<1>(commonVerticesBetweenSites);

        const Eigen::Vector3f tranlsation1 = extraData->translationsXtoY(rowIndex1, colIndex1);
        const Eigen::Vector3f tranlsation2 = extraData->translationsXtoY(rowIndex2, colIndex2);

        double innerProductRot = fabs(rotation1.w() * rotation2.w() + rotation1.vec().dot(rotation2.vec()));
        innerProductRot = std::max(std::min(1.0, innerProductRot), -1.0); // clip into value range [-1, ..., 1]


        const Eigen::Vector3f vert1 = extraData->VX.row(idxX1);
        const Eigen::Vector3f vert2 = extraData->VX.row(idxX2);

        const Eigen::Vector3f t1Minust2 = tranlsation1 - tranlsation2;

        const Eigen::Vector3f defVert1_1 = rot1 * vert1;
        const Eigen::Vector3f defVert2_1 = rot1 * vert2;

        const Eigen::Vector3f defVert1_2 = rot2 * vert2;
        const Eigen::Vector3f defVert2_2 = rot2 * vert2;

        const float se3_11_12 = (defVert1_1 - defVert1_2 + t1Minust2).norm();
        const float se3_11_22 = (defVert1_1 - defVert2_2 + t1Minust2).norm();
        const float se3_21_12 = (defVert2_1 - defVert1_2 + t1Minust2).norm();
        const float se3_21_22 = (defVert2_1 - defVert2_2 + t1Minust2).norm();

        const float maxse3dist = std::max({se3_11_12, se3_11_22, se3_21_12, se3_21_22});

        //diff = 2 * acos(innerProductRot);
        const float so3dist = acos(innerProductRot);

        diff = extraData->lambda * maxse3dist + so3dist;
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_GEODIST) {
        const std::tuple<int, int> commonVerticesBetweenTriangles = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenTriangles);
        const int idxX2 = std::get<1>(commonVerticesBetweenTriangles);
        const int realLableIndex1 = extraData->lableToIndex(l1, 1);
        const int realLableIndex2 = extraData->lableToIndex(l2, 1);
        int colIdx11 = -1, colIdx12 = -1, colIdx21 = -1, colIdx22 = -1;
        for (int j = 0; j < 3; j++) {
            if (extraData->FX(s1, j) == idxX1) colIdx11 = j;
            if (extraData->FX(s1, j) == idxX2) colIdx12 = j;
            if (extraData->FX(s2, j) == idxX1) colIdx21 = j;
            if (extraData->FX(s2, j) == idxX2) colIdx22 = j;
        }
        const int targetVertex1_1 = extraData->LableFY(realLableIndex1, colIdx11);
        const int targetVertex1_2 = extraData->LableFY(realLableIndex1, colIdx12);
        const int targetVertex2_1 = extraData->LableFY(realLableIndex2, colIdx21);
        const int targetVertex2_2 = extraData->LableFY(realLableIndex2, colIdx22);
        diff = extraData->geoDistY(targetVertex1_1, targetVertex2_1) + extraData->geoDistY(targetVertex1_2, targetVertex2_2);
        //diff = std::log(diff);
    }


    return (int) (SCALING_FACTOR * diff);
}


} // namespace smgco


#endif // GCO_SHAPE_MATCHING
