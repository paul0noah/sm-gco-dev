#include "gco_shape_matching.hpp"
#include <gco/GCoptimization.h>
#include <igl/edges.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/exact_geodesic.h>
#include <igl/unique_rows.h>
#include <chrono>

#define GETTIME(x) std::chrono::steady_clock::time_point x = std::chrono::steady_clock::now()
#define DURATION_MS(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count()
#define DURATION_S(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count() / 1000

namespace smgco {

typedef struct GCOTrianglewiseExtra {
    Eigen::MatrixXf p2pDeformation;
    Eigen::MatrixXi FX;
    Eigen::MatrixXi LableFY;
} GCOPointwiseExtra;


GCoptimization::EnergyTermType smoothFnGCOSMTrianglewise(GCoptimization::SiteID s1,
                                                         GCoptimization::SiteID s2,
                                                         GCoptimization::LabelID l1,
                                                         GCoptimization::LabelID l2,
                                                         void* extraDataVoid) {
    GCOTrianglewiseExtra* extraData = static_cast<GCOTrianglewiseExtra*>(extraDataVoid);

    /*const Eigen::VectorXi targetTri1 = extraData->LableFY.row(l1);
    const Eigen::VectorXi targetTri2 = extraData->LableFY.row(l2);

    const float diff0 = extraData->p2pDeformation(targetTri1(0), targetTri2(0));
    const float diff1 = extraData->p2pDeformation(targetTri1(1), targetTri2(1));
    const float diff2 = extraData->p2pDeformation(targetTri1(2), targetTri2(2));
     */
    const float diff = extraData->p2pDeformation(l1, l2);

    return (int) (SCALING_FACTOR * diff);
}


void precomputeSmoothCost(const Eigen::MatrixXd& VX,
                          const Eigen::MatrixXi& FX,
                          const Eigen::MatrixXd& VY,
                          const Eigen::MatrixXi& FY,
                          const Eigen::MatrixXi& lableSpace,
                          GCOTrianglewiseExtra& extraData) {
    Eigen::MatrixXf smoothVertexCost(VY.rows(), VY.rows());
    const bool smoothGeodesic = false;
    if (smoothGeodesic) {
        Eigen::VectorXi VYsource, FS, VYTarget, FT;
        // all vertices are source, and all are targets
        VYsource.resize(1);
        VYTarget.setLinSpaced(VY.rows(), 0, VY.rows());
        Eigen::VectorXf d;
        for (int i = 0; i  < VY.rows(); i++) {
            VYsource(0) = i;
            igl::exact_geodesic(VY, FY, VYsource, FS, VYTarget, FT, d);
            smoothVertexCost.col(i) = d;
        }

    }
    else { // smooth l2
        smoothVertexCost.setZero();
        for (int i = 0; i < VY.rows(); i++) {
            for (int j = 0; j < VY.rows(); j++) {
                smoothVertexCost(i, j) = (VY.row(i) - VY.row(j)).norm();
            }
        }
    }

    Eigen::MatrixXf smoothCost(lableSpace.rows(), lableSpace.rows());
    for (int l1 = 0; l1 < lableSpace.rows(); l1++) {
        const Eigen::VectorXi targetTri1 = lableSpace.row(l1);
        for (int l2 = 0; l2 < lableSpace.rows(); l2++) {

            const Eigen::VectorXi targetTri2 = lableSpace.row(l2);
            const float diff0 = smoothVertexCost(targetTri1(0), targetTri2(0));
            const float diff1 = smoothVertexCost(targetTri1(1), targetTri2(1));
            const float diff2 = smoothVertexCost(targetTri1(2), targetTri2(2));
            smoothCost(l1, l2) = std::max({diff0, diff1, diff2});
        }
    }

    extraData.p2pDeformation = smoothCost;
    extraData.FX = FX;
    extraData.LableFY = lableSpace;
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise() {
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
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numVertices, numLables);
        gc->setVerbosity(1);

        std::cout << prefix << "Precomputing costs..." << std::endl;
        GETTIME(t1);
        // Note this could be optimised
        int* data = new int[numVertices * numLables];
        for ( int i = 0; i < numVertices; i++ ) {
            for (int l = 0; l < numLables; l++ ) {
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    sum += perVertexFeatureDifference(FX(i, j), lableSpace(l, j));
                }
                data[i * numLables + l] = (int) (SCALING_FACTOR * dataWeight * sum);
            }
        }


        gc->setDataCost(data);
        GETTIME(t2);
        std::cout << prefix << " -> data cost done (" << DURATION_S(t1, t2) << " s)" << std::endl;


        GCOPointwiseExtra extraData;
        extraData.costMode = costMode;
        precomputeSmoothCost(VX, FX, VY, FY, lableSpace, extraData);
        gc->setSmoothCost(smoothFnGCOSMTrianglewise, static_cast<void*>(&extraData));
        GETTIME(t3);
        std::cout << prefix << " -> smooth cost done (" << DURATION_S(t2, t3) << " s)" << std::endl;


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
        GETTIME(t4);
        gc->expansion(numIters);
        GETTIME(t5);
        std::cout << prefix << "After optimization energy is " << gc->compute_energy() / SCALING_FACTOR << std::endl;
        std::cout << prefix << "Optimisation took: " << DURATION_S(t4, t5) << " s" << std::endl;


        for ( int  i = 0; i < numVertices; i++ ) {
            const int lable = gc->whatLabel(i);
            for (int j = 0; j < 3; j++) {
                result(i, j+3) = lableSpace(lable, j);
            }
        }


        delete gc;
        delete[] data;
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
