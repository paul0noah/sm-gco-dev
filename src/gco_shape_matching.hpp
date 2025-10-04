#ifndef GCO_SHAPE_MATCHING
#define GCO_SHAPE_MATCHING
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <gco/GCoptimization.h>
#include "helper/shape.hpp"
#include <unordered_map>
#define USE_CACHING false

#define SCALING_FACTOR 10000.0f
typedef Eigen::MatrixX<std::tuple<int, int>> TupleMatrixInt;

namespace smgco {
enum COST_MODE {
    SINGLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_SO3,
    MULTIPLE_LABLE_SPACE_SE3,
    MULTIPLE_LABLE_SPACE_GEODIST,
    MULTIPLE_LABLE_SPACE_GEODIST_MAX,
    MULTIPLE_LABLE_SPACE_L2DIST,
    MULTIPLE_LABLE_SPACE_L2DIST_MAX
};

struct TriangleWiseOpts {
    COST_MODE costMode = COST_MODE::MULTIPLE_LABLE_SPACE_GEODIST;
    float smoothScaleBeforeRobust = 1.0;
    bool robustCost = false;
    int setInitialLables = 1;
    float lambdaSe3 = 1.0;
    float lambdaSo3 = 1.0;
    float unaryWeight = 1.0;
    float smoothWeight = 1.0;
    int lableSpaceCycleSize = 4;
    float lableSpaceAngleThreshold = M_PI / 2;
    bool lableSpaceDegnerate = true;
    float membraneFactor = 0.0f;
    float bendingFactor = 0.0f;
    float wksFactor = 0.0f;
    float featureFactor = 1.0f;
    bool glueSolution = true;
    int labelOrder = 0; // 0 no order, 1 random, 2 degenerate last, 3 mincost, 4 alternating min cost
};

typedef struct GCOTrianglewiseExtra {
    // not all of the below matrices are needed for all cost modes
    float lambda;
    const TriangleWiseOpts& opts;
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
    std::unordered_map<std::tuple<GCoptimization::LabelID, GCoptimization::LabelID>, int> cache;
    GCOTrianglewiseExtra(TriangleWiseOpts& opts) : opts(opts) {}
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
    std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> triangleWise(TriangleWiseOpts opts);
    std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> triangleWise();

    void setDataWeight(const float newDataWeight);
    void setMaxIter(const int newMaxIter);

    void writeToFile(const std::string& filename);
};


Eigen::MatrixXi glueResult(const Eigen::MatrixXi& result,
                           const Eigen::MatrixXd& VX,
                           const Eigen::MatrixXi& FX,
                           const Eigen::MatrixXd& VY,
                           const Eigen::MatrixXi& FY,
                           const Eigen::MatrixXf& geoDistY=Eigen::MatrixXf(0, 0));

void precomputeSmoothCost(const Eigen::MatrixXd& VX,
                          const Eigen::MatrixXi& FX,
                          const Eigen::MatrixXd& VY,
                          const Eigen::MatrixXi& FY,
                          const Eigen::MatrixXi& lableSpace,
                          GCOTrianglewiseExtra& extraData);

Eigen::MatrixXi buildLableSpace(const Eigen::MatrixXd& VY,
                                const Eigen::MatrixXi& FY,
                                int& numDegenerate,
                                TriangleWiseOpts& opts);


/*






 */
inline
GCoptimization::EnergyTermType smoothFnGCOSMTrianglewise(GCoptimization::SiteID s1,
                                                         GCoptimization::SiteID s2,
                                                         GCoptimization::LabelID l1,
                                                         GCoptimization::LabelID l2,
                                                         void* extraDataVoid) {
    GCOTrianglewiseExtra* extraData = static_cast<GCOTrianglewiseExtra*>(extraDataVoid);
    if (USE_CACHING) {
        const auto& it = extraData->cache.find(std::make_tuple(std::min(l1, l2), std::max(l1, l2)));
        if (it != extraData->cache.end()) {
            return it->second;
            //return it.value();
        }
    }


    const TriangleWiseOpts opts = extraData->opts;
    const COST_MODE costMode = extraData->opts.costMode;


    float diff = 0;

    if (costMode == SINGLE_LABLE_SPACE_L2) {
        diff = extraData->p2pDeformation(l1, l2);
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_L2) {
        const std::tuple<int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenSites);
        const int idxX2 = std::get<1>(commonVerticesBetweenSites);
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

        diff = 2 * acos(innerProductRot);
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

        const float so3dist = 2 * acos(innerProductRot);

        diff = opts.lambdaSe3 * maxse3dist + opts.lambdaSo3 * so3dist;
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_GEODIST ||
             costMode == MULTIPLE_LABLE_SPACE_GEODIST_MAX ||
             costMode == MULTIPLE_LABLE_SPACE_L2DIST ||
             costMode == MULTIPLE_LABLE_SPACE_L2DIST_MAX) {
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


        if (costMode == MULTIPLE_LABLE_SPACE_L2DIST_MAX || costMode == MULTIPLE_LABLE_SPACE_GEODIST_MAX) {
            diff = std::max(extraData->geoDistY(targetVertex1_1, targetVertex2_1), extraData->geoDistY(targetVertex1_2, targetVertex2_2));
        }
        else {
            diff = extraData->geoDistY(targetVertex1_1, targetVertex2_1) + extraData->geoDistY(targetVertex1_2, targetVertex2_2);
        }
    }

    diff = opts.smoothScaleBeforeRobust * diff;

    if (opts.robustCost) {
        diff = std::log(diff);
    }

    diff = opts.smoothWeight * diff;


    const int output = (int) (SCALING_FACTOR * diff );
    if (USE_CACHING) {
        extraData->cache.insert({std::make_tuple(std::min(l1, l2), std::max(l1, l2)), output});
    }

    return output;
}


} // namespace smgco


#endif // GCO_SHAPE_MATCHING
