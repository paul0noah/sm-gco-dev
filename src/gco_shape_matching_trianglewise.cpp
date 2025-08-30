#include "gco_shape_matching.hpp"
#include <gco/GCoptimization.h>
#include <igl/edges.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/exact_geodesic.h>
#include <igl/unique_rows.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>

#include <chrono>

#define GETTIME(x) std::chrono::steady_clock::time_point x = std::chrono::steady_clock::now()
#define DURATION_MS(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count()
#define DURATION_S(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count() / 1000
#define PRINT_SMGCO(x) std::cout << prefix << x << std::endl;

#define isFakeLable(isFakeLable, s1, l1, numLables) {\
const int realLableIdMin =  s1 * numLables;\
const int realLableIdMax = (s1+1) * numLables;\
isFakeLable = l1 < realLableIdMin || l1 >= realLableIdMax;}

typedef Eigen::MatrixX<std::tuple<int, int>> TupleMatrixInt;

namespace smgco {

enum COST_MODE {
    SINGLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_SO3,
    MULTIPLE_LABLE_SPACE_SE3,
};

typedef struct GCOTrianglewiseExtra {
    COST_MODE costMode;
    // not all of the below matrices are needed for all cost modes
    int numLables;
    Eigen::MatrixXf p2pDeformation;
    Eigen::MatrixXf VX;
    Eigen::MatrixXi FX;
    Eigen::MatrixXi LableFY;
    TupleMatrixInt commonVXofFX;
    Eigen::MatrixX<Eigen::Quaterniond>& quaternoinsXtoY;
    Eigen::MatrixX<Eigen::Vector3f>& translationsXtoY;
    Eigen::MatrixXi lableToIndex;
    GCOTrianglewiseExtra(Eigen::MatrixXf& temp0,
                         Eigen::MatrixXi& temp1,
                         TupleMatrixInt& temp2,
                         Eigen::MatrixX<Eigen::Quaterniond>& temp3,
                         Eigen::MatrixX<Eigen::Vector3f>& temp4) :
        p2pDeformation(temp0),
        FX(temp1),
        LableFY(temp1),
        commonVXofFX(temp2),
        quaternoinsXtoY(temp3),
        lableToIndex(temp1),
        translationsXtoY(temp4) {
    };
} GCOTrianglewiseExtra;

typedef struct GCOTrianglewiseExtraData {
    COST_MODE costMode;
    Eigen::MatrixXi data;
    Eigen::MatrixXi lableToIndex;
    int numLables;
    GCOTrianglewiseExtraData(Eigen::MatrixXi& temp0) : data(temp0), lableToIndex(temp0) {
    }
} GCOTrianglewiseExtraData;



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
        const std::tuple<int, int, int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int s1id0 = std::get<0>(commonVerticesBetweenSites);
        const int s2id0 = std::get<1>(commonVerticesBetweenSites);
        const int s1id1 = std::get<2>(commonVerticesBetweenSites);
        const int s2id1 = std::get<3>(commonVerticesBetweenSites);

        const int s1vy0 = extraData->LableFY(l1, s1id0);
        const int s1vy1 = extraData->LableFY(l1, s1id1);
        const int s2vy0 = extraData->LableFY(l2, s2id0);
        const int s2vy1 = extraData->LableFY(l2, s2id1);

        const float def0 = extraData->p2pDeformation(s1vy0, s2vy0);
        const float def1 = extraData->p2pDeformation(s1vy1, s2vy1);

        std::cout << s1 << ", " << s2 << ": " << ", " << s1vy0 << ", " << s1vy1 << ", " << s2vy0 << ", " << s2vy1 << ", " << std::endl;

        if (def0 > def1)
            diff = def0;
        else
            diff = def1;
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

        const Eigen::Vector3f defVert1_1 = rot1 * vert1 + tranlsation1;
        const Eigen::Vector3f defVert2_1 = rot1 * vert2 + tranlsation1;

        const Eigen::Vector3f defVert1_2 = rot2 * vert2 + tranlsation2;
        const Eigen::Vector3f defVert2_2 = rot2 * vert2 + tranlsation2;

        const float l2norm11_12 = (defVert1_1 - defVert1_2).norm();
        const float l2norm11_22 = (defVert1_1 - defVert2_2).norm();
        const float l2norm21_12 = (defVert2_1 - defVert1_2).norm();
        const float l2norm21_22 = (defVert2_1 - defVert2_2).norm();

        const float maxDeform = std::max({l2norm11_12, l2norm11_22, l2norm21_12, l2norm21_22});

        //diff = 2 * acos(innerProductRot);
        //diff = acos(innerProductRot) + maxDeform;

        diff = maxDeform;
    }


    return (int) (SCALING_FACTOR * diff);
}
/*






void precomputeSmoothCost(const Eigen::MatrixXd& VX,
                          const Eigen::MatrixXi& FX,
                          const Eigen::MatrixXd& VY,
                          const Eigen::MatrixXi& FY,
                          const Eigen::MatrixXi& lableSpace,
                          GCOTrianglewiseExtra& extraData) {

    const COST_MODE costMode = extraData.costMode;

    Eigen::MatrixXf p2pDeformation(VY.rows(), VY.rows());
    const bool smoothGeodesic = false;
    // l2 distance between vertices
    p2pDeformation.setZero();
    for (int i = 0; i < VY.rows(); i++) {
        for (int j = 0; j < VY.rows(); j++) {
            p2pDeformation(i, j) = (VY.row(i) - VY.row(j)).norm();
        }
    }

    if (costMode == SINGLE_LABLE_SPACE_L2) {
        Eigen::MatrixXf smoothCost(lableSpace.rows(), lableSpace.rows());
        for (int l1 = 0; l1 < lableSpace.rows(); l1++) {
            const Eigen::VectorXi targetTri1 = lableSpace.row(l1);
            for (int l2 = 0; l2 < lableSpace.rows(); l2++) {

                const Eigen::VectorXi targetTri2 = lableSpace.row(l2);
                const float diff0 = p2pDeformation(targetTri1(0), targetTri2(0));
                const float diff1 = p2pDeformation(targetTri1(1), targetTri2(1));
                const float diff2 = p2pDeformation(targetTri1(2), targetTri2(2));
                smoothCost(l1, l2) = std::max({diff0, diff1, diff2});
            }
        }
        extraData.p2pDeformation = smoothCost;
    }
    if (costMode == MULTIPLE_LABLE_SPACE_L2) {
        TupleMatrixInt commonVertices(FX.rows(), FX.rows());
        Eigen::MatrixXi AdjFX;
        igl::triangle_triangle_adjacency(FX, AdjFX);

        std::vector<int> intersection; intersection.reserve(4);
        for (int i = 0; i < FX.rows(); i++) {
            const int i0 = FX(i, 0), i1 = FX(i, 1), i2 = FX(i, 2);
            for (int k = 0; k < 3; k++) {
                const int j = AdjFX(i, k);
                if (j == -1) continue;

                for (int ii = 0; ii < 3; ii++) {
                    const int vi = FX(i, ii);
                    for (int jj = 0; jj < 3; jj++) {
                        const int vj = FX(j, jj);
                        if (vi == vj) {
                            intersection.push_back(ii);
                            intersection.push_back(jj);
                        }
                    }
                }
                commonVertices(i, j) = std::make_tuple(intersection[0], intersection[1], intersection[2], intersection[3]);
                intersection.clear();
            }
        }
        extraData.commonVXofFX = commonVertices;

        extraData.p2pDeformation = p2pDeformation;
    }
    if (costMode == MULTIPLE_LABLE_SPACE_SO3 || costMode == MULTIPLE_LABLE_SPACE_L2 || costMode == MULTIPLE_LABLE_SPACE_SE3) {
        std::cout << "TODO: take care of other lable space definitions for normals" << std::endl;
        Eigen::MatrixXd NX, NY, TriCentroidsX, TriCentroidsY;
        igl::per_face_normals(VX, FX, NX);
        igl::per_face_normals(VY, FY, NY);

        igl::barycenter(VX, FX, TriCentroidsX);
        igl::barycenter(VY, FY, TriCentroidsY);

        const bool globalTrafo = true;
        if (!globalTrafo) {
            TriCentroidsX.setZero();
            TriCentroidsY.setZero();
        }


        Eigen::MatrixXd EX0 = TriCentroidsX + (VX(FX.col(0), Eigen::all) - VX(FX.col(1), Eigen::all)).rowwise().normalized();
        //Eigen::MatrixXd EX1 = TriCentroidsX + (VX(FX.col(2), Eigen::all) - VX(FX.col(1), Eigen::all)).rowwise().normalized();
        Eigen::MatrixXd EX1(EX0.rows(), EX0.cols());
        for (int i = 0; i < EX0.rows(); i++) {
            const Eigen::Vector3d e = EX0.row(i);
            const Eigen::Vector3d n = NX.row(i);
            EX1.row(i) = TriCentroidsX.row(i) + (e.cross(n)).normalized().transpose();
        }

        Eigen::MatrixXd EY0 = (VY(lableSpace.col(0), Eigen::all) - VY(lableSpace.col(1), Eigen::all)).rowwise().normalized();
        //Eigen::MatrixXd EY1 = (VY(lableSpace.col(2), Eigen::all) - VY(lableSpace.col(1), Eigen::all)).rowwise().normalized();
        //for (int i = 0; i < 3; i++) {
        //    EY1.block(i * FY.rows(), 0, FY.rows(), 3) = TriCentroidsY + EY1.block(i * FY.rows(), 0, FY.rows(), 3);
        //}
        Eigen::MatrixXd EY1(EY0.rows(), EY0.cols());
        for (int i = 0; i < EY0.rows(); i++) {
            const Eigen::Vector3d e = EY0.row(i);
            const int fy = i % FY.rows();
            const Eigen::Vector3d n = NY.row(fy);
            EY1.row(i) = TriCentroidsY.row(fy) + (e.cross(n)).normalized().transpose();
        }

        Eigen::MatrixX<Eigen::Quaterniond> quaternoinsXtoY(FX.rows(), lableSpace.rows());
        Eigen::MatrixX<Eigen::Vector3f> tranlastionsXtoY(FX.rows(), lableSpace.rows());
        for (int x = 0; x < FX.rows(); x++) {

            for (int l = 0; l < lableSpace.rows(); l++) {
                const int fy = l % FY.rows();
                const Eigen::Matrix3d pointsX = globalTrafo ? (Eigen::Matrix3d() << TriCentroidsX.row(x), EX0.row(x), EX1.row(x)).finished() :
                                                              (Eigen::Matrix3d() << NX.row(x), EX0.row(x), EX1.row(x)).finished() ;
                const Eigen::Matrix3d pointsY = globalTrafo ? (Eigen::Matrix3d() << TriCentroidsY.row(fy), EY0.row(l), EY1.row(l)).finished() :
                                                              (Eigen::Matrix3d() << NY.row(fy), EY0.row(l), EY1.row(l)).finished() ;

                /*std::cout << pointsY.row(0).dot(pointsY.row(1)) << std::endl;
                std::cout << pointsY.row(0).dot(pointsY.row(2)) << std::endl;
                std::cout << pointsY.row(1).dot(pointsY.row(2)) << std::endl;
                std::cout << pointsX << std::endl;
                std::cout << pointsY << std::endl;*/


                if (globalTrafo) {
                    const auto tform = Eigen::umeyama(pointsY.transpose(), pointsX.transpose(), false);
                    const Eigen::Matrix3d rot = tform.block(0, 0, 3, 3);
                    const Eigen::Vector3d translation = tform.block(0, 3, 3, 1);
                    /*std::cout << rot << std::endl;
                    std::cout << translation << std::endl;

                    for (int i = 0; i < 3; i++) {
                        const Eigen::Vector3d x = pointsX.row(i);
                        const Eigen::Vector3d y = pointsY.row(i);
                        std::cout << x - rot * y - translation << std::endl;
                        std::cout << y - rot * x - translation << std::endl;
                    }*/
                    assert (std::abs(rot.determinant() - 1.0) <= 1e-6);
                    quaternoinsXtoY(x, l) = Eigen::Quaterniond(rot);
                    tranlastionsXtoY(x, l) = translation.cast<float>();
                }
                else {
                    // see e.g. 4. here: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(pointsX * pointsY.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

                    const double detuv = ( svd.matrixV() * (svd.matrixU().transpose()) ).determinant();
                    const Eigen::DiagonalMatrix<double, 3> d(1, 1, detuv);
                    const Eigen::Matrix3d rot = svd.matrixV() * d * (svd.matrixU().transpose());

                    /*for (int i = 0; i < 3; i++) {
                        const Eigen::Vector3d x = pointsX.row(i);
                        const Eigen::Vector3d y = pointsY.row(i);
                        std::cout << x - rot * y << std::endl;
                        std::cout << y - rot * x << std::endl;
                        std::cout << x - rot.transpose() * y << std::endl;
                        std::cout << y - rot.transpose() * x << std::endl;
                    }*/
                    quaternoinsXtoY(x, l) = Eigen::Quaterniond(rot);

                }

            }
        }
        extraData.quaternoinsXtoY = quaternoinsXtoY;
        extraData.translationsXtoY = tranlastionsXtoY;

    }

    extraData.FX = FX;
    extraData.LableFY = lableSpace;
}

/*






 */
std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise(const int costModeInt) {
    const bool setInitialLables = false;

    const COST_MODE costMode = static_cast<COST_MODE>(costModeInt);
    std::cout << prefix << "Using cost mode = " << costMode << std::endl;

    const int numVertices = FX.rows();
    const int numLables = 3 * FY.rows();
    // any of the three orientations of a triangle is the lable space
    Eigen::MatrixXi lableSpace(numLables, 3);
    lableSpace.block(0, 0, FY.rows(), 3)             = FY;
    lableSpace.block(FY.rows(), 0, FY.rows(), 3)     = FY(Eigen::all, (Eigen::Vector3i() << 1, 2, 0).finished());
    lableSpace.block(2 * FY.rows(), 0, FY.rows(), 3) = FY(Eigen::all, (Eigen::Vector3i() << 2, 0, 1).finished());

    Eigen::MatrixXi AdjFX;
    igl::edges(FX, AdjFX);
    igl::triangle_triangle_adjacency(FX, AdjFX);

    Eigen::MatrixXi result(numVertices, 6);
    result.block(0, 0, FX.rows(), 3) = FX;

    try{
        const int numFakeLables = FX.rows() * numLables;
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numVertices, numFakeLables);
        gc->setVerbosity(1);

        PRINT_SMGCO("Precomputing helpers...");
        GETTIME(t1);
        PRINT_SMGCO(" -> helpers done (" << DURATION_S(t0, t1) << " s)");
        PRINT_SMGCO("Precomputing costs...");

        // Note this could be optimised
        Eigen::MatrixXi data(numVertices, numLables);
        for ( int i = 0; i < numVertices; i++ ) {
            for (int l = 0; l < numLables; l++ ) {
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    sum += perVertexFeatureDifference(FX(i, j), lableSpace(l, j));
                }
                data(i, l) = (int) (SCALING_FACTOR * dataWeight * sum);
            }
        }

        Eigen::MatrixXf temp0(0, 0);
        Eigen::MatrixXi temp1(0, 0);
        TupleMatrixInt temp2(0, 0);
        Eigen::MatrixX<Eigen::Quaterniond> temp3(0, 0);
        Eigen::MatrixX<Eigen::Vector3f> temp4(0, 0);


        for (int i = 0; i < numVertices; i++) {
            for (int l = 0; l < numLables; l++) {
                const int fakeLable = i * numLables + l;
                const int numSites = 1;
                GCoptimization::SparseDataCost* c = new GCoptimization::SparseDataCost[numSites];
                c[0].site = i;
                c[0].cost = data(i, l);
                gc->setDataCost(fakeLable, c, numSites);
            }
        }



        GETTIME(t2);
        PRINT_SMGCO(" -> data cost done (" << DURATION_S(t1, t2) << " s)");


        GCOTrianglewiseExtra extraSmooth(temp0, temp1, temp2, temp3, temp4);
        extraSmooth.costMode = costMode;
        extraSmooth.numLables = numLables;
        extraSmooth.lableToIndex = lableToIndex;
        extraSmooth.VX = VX.cast<float>();
        precomputeSmoothCost(VX, FX, VY, FY, lableSpace, extraSmooth);
        gc->setSmoothCost(smoothFnGCOSMTrianglewise, static_cast<void*>(&extraSmooth));
        GETTIME(t3);
        PRINT_SMGCO(" -> smooth cost done (" << DURATION_S(t2, t3) << " s)");


        for (int f = 0; f < AdjFX.rows(); f++) {
            const int srcId = f;
            for (int j = 0; j < 3; j++) {
                const int targetId = AdjFX(f, j);
                if (targetId >= 0) {
                    const int weight = 1;
                    gc->setNeighbors(srcId, targetId, weight);
                }
            }
        }

        std::cout << prefix << "Before optimization energy is " << gc->compute_energy() / SCALING_FACTOR << std::endl;
        PRINT_SMGCO("Before optimization energy is " << gc->compute_energy() / SCALING_FACTOR);
        GETTIME(t4);
        gc->expansion(numIters);
        GETTIME(t5);
        PRINT_SMGCO("After optimization energy is " << gc->compute_energy() / SCALING_FACTOR);
        PRINT_SMGCO("Optimisation took: " << DURATION_S(t4, t5) << " s");


        for ( int  i = 0; i < numVertices; i++ ) {
            const int lable = gc->whatLabel(i) - i * numLables;
            if (lable < 0 || lable > numLables) {
                PRINT_SMGCO("optimisation led to fake-lable for triangle " << i << ", skipping output writing");
                continue;
            }
            for (int j = 0; j < 3; j++) {
                result(i, j+3) = lableSpace(lable, j);
            }
        }


        delete gc;
    }
    catch (GCException e){
        e.Report();
    }


    Eigen::MatrixXi p2p(3 * result.rows(), 2);
    p2p.block(0, 0, result.rows(), 1)               = result.block(0, 0, result.rows(), 1);
    p2p.block(0, 1, result.rows(), 1)               = result.block(0, 3, result.rows(), 1);
    p2p.block(FX.rows(), 0, result.rows(), 1)       = result.block(0, 1, result.rows(), 1);
    p2p.block(FX.rows(), 1, result.rows(), 1)       = result.block(0, 4, result.rows(), 1);
    p2p.block(2 * FX.rows(), 0, result.rows(), 1)   = result.block(0, 2, result.rows(), 1);
    p2p.block(2 * FX.rows(), 1, result.rows(), 1)   = result.block(0, 5, result.rows(), 1);

    Eigen::MatrixXi p2p_unique, IA, IC;
    igl::unique_rows(p2p, p2p_unique, IA, IC);

    return std::make_tuple(p2p_unique, result);
}

} // namespace smgco
