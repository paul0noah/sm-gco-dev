#include "gco_shape_matching.hpp"
#include <igl/edges.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/exact_geodesic.h>
#include <igl/unique_rows.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>
#include "energy/deformationEnergy.hpp"
#include <chrono>

#define GETTIME(x) std::chrono::steady_clock::time_point x = std::chrono::steady_clock::now()
#define DURATION_MS(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count()
#define DURATION_S(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count() / 1000
#define PRINT_SMGCO(x) std::cout << prefix << x << std::endl;

namespace smgco {
/*






 */
std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise(TriangleWiseOpts opts) {
    GCOTrianglewiseExtra extraSmooth;
    const bool setInitialLables = opts.setInitialLables;

    const COST_MODE costMode = opts.costMode;
    std::cout << prefix << "Using cost mode = " << costMode << std::endl;

    const int numVertices = FX.rows();
    int numDegenerate = 0;
    extraSmooth.LableFY = buildLableSpace(VY, FY, numDegenerate, opts);
    const int numLables = extraSmooth.LableFY.rows();
    PRINT_SMGCO("num lables = " << numVertices * numLables << ", num lables per site = " << numLables);

    Eigen::MatrixXi AdjFX;
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
        extraSmooth.lableToIndex = Eigen::MatrixXi(numFakeLables, 2);
        for (int lf = 0; lf < numFakeLables; lf++) {
            const int rowIndex = lf / numLables;
            const int colIndex = lf - rowIndex * numLables;
            extraSmooth.lableToIndex.row(lf) << rowIndex, colIndex ;
        }



        GETTIME(t1);
        PRINT_SMGCO(" -> helpers done (" << DURATION_S(t0, t1) << " s)");
        PRINT_SMGCO("Precomputing costs...");

        Eigen::MatrixXd energy;
        bool addEnergy = false;
        if (opts.membraneFactor > 1e-8 || opts.bendingFactor > 1e-8 || opts.wksFactor > 1e-8) {
            energy = energyWrapper(VX, FX, VY, FY, extraSmooth.LableFY, numDegenerate, opts.membraneFactor, opts.bendingFactor, opts.wksFactor);
            addEnergy = true;
        }
        Eigen::MatrixXi minLables(numVertices, 1); minLables.setConstant(0);
        for (int i = 0; i < numVertices; i++) {
            float minCost = std::numeric_limits<float>::infinity();
            for (int l = 0; l < numLables; l++) {
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    sum += perVertexFeatureDifference(FX(i, j), extraSmooth.LableFY(l, j));
                }
                if (sum < minCost) {
                    minCost = sum;
                    minLables(i) = l;
                }
                if (addEnergy) {
                    sum += energy(i, l);
                }
                const int dataCost = (int) (SCALING_FACTOR * opts.unaryWeight * sum);

                const int fakeLable = i * numLables + l;
                const int numSites = 1;
                GCoptimization::SparseDataCost* c = new GCoptimization::SparseDataCost[numSites];
                c[0].site = i;
                c[0].cost = dataCost;
                gc->setDataCost(fakeLable, c, numSites);
                delete[] c;
            }
        }
        energy.conservativeResize(0, 0);



        GETTIME(t2);
        PRINT_SMGCO(" -> data cost done (" << DURATION_S(t1, t2) << " s)");


        extraSmooth.costMode = costMode;
        extraSmooth.numLables = numLables;
        extraSmooth.VX = VX.cast<float>();
        extraSmooth.FX = FX;
        if (USE_CACHING) {
            const int cacheSizeGuess = 5 * numLables * AdjFX.rows();
            PRINT_SMGCO("Cache size guess " << cacheSizeGuess);
            extraSmooth.cache.reserve(cacheSizeGuess);
        }

        precomputeSmoothCost(VX, FX, VY, FY, extraSmooth.LableFY, extraSmooth);
        gc->setSmoothCost(smoothFnGCOSMTrianglewise, static_cast<void*>(&extraSmooth));
        GETTIME(t3);
        PRINT_SMGCO(" -> smooth cost done (" << DURATION_S(t2, t3) << " s)");




        if (setInitialLables) {
            PRINT_SMGCO("Setting initial lables");
            for (int i = 0; i < numVertices; i++) {
                int minIndex = minLables(i);
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
                result(i, j+3) = extraSmooth.LableFY(lable, j);
            }
        }

        if (USE_CACHING) {
            PRINT_SMGCO("Cache filled " << extraSmooth.cache.size());
            size_t collisions = 0;
            for (size_t i = 0; i < extraSmooth.cache.bucket_count(); ++i) {
                auto bucket_size = extraSmooth.cache.bucket_size(i);
                if (bucket_size > 1) {
                    collisions += bucket_size - 1; // each extra element in a bucket is a collision
                }
            }
            PRINT_SMGCO("Collisions: " << collisions);
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


    Eigen::MatrixXi gluedSolution, gluedp2p;
    if (opts.glueSolution) {
        gluedSolution = Eigen::MatrixXi(result.rows(), 6);
        gluedp2p = Eigen::MatrixXi(VX.rows(), 2);
        const Eigen::MatrixXi gluedMatches = glueResult(result.block(0, 3, result.rows(), 3), VX, FX, VY, FY);
        for (int i = 0; i < VX.rows(); i++) {
            gluedp2p.row(i) << i, gluedMatches(i);
        }
        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                const int indexX = result(i, j);
                const int bestIndexY = gluedMatches(indexX);
                gluedSolution(i, j) = indexX;
                gluedSolution(i, j+3) = bestIndexY;
            }
        }
    }
    return std::make_tuple(p2p_unique, result, gluedp2p, gluedSolution);
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise() {
    TriangleWiseOpts opts;
    return triangleWise(opts);
}

} // namespace smgco
