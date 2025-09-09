#include "gco_shape_matching.hpp"
#include <igl/edges.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/exact_geodesic.h>
#include <igl/unique_rows.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>
#if defined(_OPENMP)
    #include <omp.h>
#else
#endif
#include "helper/graph_cycles.hpp"


namespace smgco {

/*






 */
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
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
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
    if (costMode == MULTIPLE_LABLE_SPACE_L2 || costMode == MULTIPLE_LABLE_SPACE_SE3 || costMode == MULTIPLE_LABLE_SPACE_GEODIST) {
        TupleMatrixInt commonVertices(FX.rows(), FX.rows());
        Eigen::MatrixXi AdjFX;
        igl::triangle_triangle_adjacency(FX, AdjFX);

        #if defined(_OPENMP)
        #pragma omp parallel for
        #else
        std::vector<int> intersection; intersection.reserve(4);
        #endif
        for (int i = 0; i < FX.rows(); i++) {
            #if defined(_OPENMP)
            std::vector<int> intersection; intersection.reserve(4);
            #endif
            const int i0 = FX(i, 0), i1 = FX(i, 1), i2 = FX(i, 2);
            for (int k = 0; k < 3; k++) {
                const int j = AdjFX(i, k);
                if (j == -1) continue;

                for (int ii = 0; ii < 3; ii++) {
                    const int vi = FX(i, ii);
                    for (int jj = 0; jj < 3; jj++) {
                        const int vj = FX(j, jj);
                        if (vi == vj) {
                            intersection.push_back(vi);
                        }
                    }
                }
                commonVertices(i, j) = std::make_tuple(intersection[0], intersection[1]);
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
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
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
    if (costMode == MULTIPLE_LABLE_SPACE_GEODIST) {
        Eigen::MatrixXf geoDistY(VY.rows(), VY.rows());
        #if defined(_OPENMP)
        #pragma omp parallel for
        #else
        Eigen::VectorXi VYsource, FS, VYTarget, FT;
        // all vertices are source, and all are targets
        VYsource.resize(1);
        VYTarget.setLinSpaced(VY.rows(), 0, VY.rows());
        Eigen::VectorXf d;
        #endif
        for (int i = 0; i  < VY.rows(); i++) {
            #if defined(_OPENMP)
            Eigen::VectorXi VYsource, FS, VYTarget, FT;
            // all vertices are source, and all are targets
            VYsource.resize(1);
            VYTarget.setLinSpaced(VY.rows(), 0, VY.rows());
            Eigen::VectorXf d;
            #endif
            VYsource(0) = i;
            igl::exact_geodesic(VY, FY, VYsource, FS, VYTarget, FT, d);
            geoDistY.col(i) = d;
        }
        extraData.geoDistY = geoDistY;
    }

    extraData.FX = FX;
    extraData.LableFY = lableSpace;
}



Eigen::MatrixXi buildLableSpace(const Eigen::MatrixXd& VY,
                                const Eigen::MatrixXi& FY,
                                TriangleWiseOpts& opts) {

    if (opts.lableSpaceDegnerate) {
        if (opts.costMode != COST_MODE::MULTIPLE_LABLE_SPACE_L2 || opts.costMode != COST_MODE::MULTIPLE_LABLE_SPACE_GEODIST) {
            std::cout << "disabling degenerate lables since cost mode does not allow it" << std::endl;
            opts.lableSpaceDegnerate = false;
        }
    }


    const Eigen::MatrixXi cycleTriangles = opts.lableSpaceCycleSize == 3 ? FY : utils::getCycleTriangles(VY, FY, opts.lableSpaceCycleSize, opts.lableSpaceAngleThreshold);
    const int numCycleTris = cycleTriangles.rows();
    int lableSpaceSize = 3 * numCycleTris;

    Eigen::MatrixXi degenerateTris;
    if (opts.lableSpaceDegnerate) {
        degenerateTris = utils::getDegenerateTriangles(FY);
        lableSpaceSize += degenerateTris.rows();
    }

    Eigen::MatrixXi lableSpace(lableSpaceSize, 3);
    if (opts.lableSpaceDegnerate) {
        lableSpace.block(0, 0, degenerateTris.rows(), 3) = degenerateTris;
    }

    lableSpace.block(degenerateTris.rows(), 0, numCycleTris, 3) = cycleTriangles;

    lableSpace.block(degenerateTris.rows() + numCycleTris, 0, numCycleTris, 3) =
                        cycleTriangles(Eigen::all, (Eigen::Vector3i() << 1, 2, 0).finished());

    lableSpace.block(degenerateTris.rows() + 2 * numCycleTris, 0, numCycleTris, 3) =
                        cycleTriangles(Eigen::all, (Eigen::Vector3i() << 2, 0, 1).finished());



    return lableSpace;
}

} // namespace smgco
