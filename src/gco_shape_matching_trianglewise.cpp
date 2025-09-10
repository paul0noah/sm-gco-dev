#include "gco_shape_matching.hpp"
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

namespace smgco {
/*






 */
std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise(TriangleWiseOpts opts) {
    const bool setInitialLables = opts.setInitialLables;

    const COST_MODE costMode = opts.costMode;
    std::cout << prefix << "Using cost mode = " << costMode << std::endl;

    const int numVertices = FX.rows();
    // any of the three orientations of a triangle is the lable space
    const Eigen::MatrixXi lableSpace = buildLableSpace(VY, FY, opts);
    const int numLables = lableSpace.rows();
    std::cout << prefix << "num lables = " << lableSpace.rows() << std::endl;

    Eigen::MatrixXi AdjFX;
    igl::edges(FX, AdjFX);
    igl::triangle_triangle_adjacency(FX, AdjFX);

    Eigen::MatrixXi result(numVertices, 6);
    result.block(0, 0, FX.rows(), 3) = FX;

    try{
        const int numFakeLables = FX.rows() * numLables;
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numVertices, numFakeLables);
        gc->setVerbosity(1);

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

        PRINT_SMGCO("Precomputing helpers...");
        GETTIME(t0);
        Eigen::MatrixXi lableToIndex(numFakeLables, 2);
        for (int lf = 0; lf < numFakeLables; lf++) {
            const int rowIndex = lf / numLables;
            const int colIndex = lf - rowIndex * numLables;
            lableToIndex.row(lf) << rowIndex, colIndex ;
        }



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


        GCOTrianglewiseExtra extraSmooth;
        extraSmooth.costMode = costMode;
        extraSmooth.numLables = numLables;
        extraSmooth.lableToIndex = lableToIndex;
        extraSmooth.VX = VX.cast<float>();
        extraSmooth.lambda = 10; //54 / 27 * M_PI;
        precomputeSmoothCost(VX, FX, VY, FY, lableSpace, extraSmooth);
        gc->setSmoothCost(smoothFnGCOSMTrianglewise, static_cast<void*>(&extraSmooth));
        GETTIME(t3);
        PRINT_SMGCO(" -> smooth cost done (" << DURATION_S(t2, t3) << " s)");




        if (setInitialLables) {
            PRINT_SMGCO("Setting initial lables");
            for (int i = 0; i < numVertices; i++) {
                int minIndex = -1;
                data.row(i).minCoeff(&minIndex);
                gc->setLabel(i, minIndex + i * numLables);
            }
        }

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


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise() {
    TriangleWiseOpts opts;
    return triangleWise(opts);
}

} // namespace smgco
