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
#if defined(_OPENMP)
int omp_thread_count() {
    // workaround for omp_get_num_threads()
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
inline int getThreadId() {
    return omp_get_thread_num();
}
#else
int omp_thread_count() {
    return 1;
}
inline int getThreadId() {
    return 0;
}
#endif

namespace smgco {
std::string INIT_METHODS[7] = { "NO_INIT",
                                "MIN_LABEL",
                                "MIN_LABEL_NON_DEGENERATE",
                                "TRI_NEIGHBOURS_NON_DEGENERATE",
                                "TRI_NEIGHBOURS",
                                "SINKHORN",
                                "RANDOM"};
std::string ALGORITHMS[8] = { "ALPHA-BETA SWAP",
                                "ALPHA EXPANSION",
                                "SWAP followed by EXPANSION",
                                "EXPANSION followed by SWAP",
                                "LINESEARCH",
                                "LINESEARCH + SITE REORDERING (w.r.t. min smooth cost)",
                                "LINESEARCH + ADAPTIVE",
                                "LINESEARCH + ADAPTIVE + SITE REORDERING (w.r.t. min smooth cost)"};
/*






 */
std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise(TriangleWiseOpts opts) {

    GCOTrianglewiseExtra extraSmooth(opts);
    const int setInitialLables = opts.setInitialLables;

    const COST_MODE costMode = opts.costMode;
    std::cout << prefix << "Using cost mode = " << costMode << std::endl;

    const int numVertices = FX.rows();
    int numDegenerate = 0;
    extraSmooth.LableFY = buildLableSpace(VY, FY, numDegenerate, opts);
    const int numLables = extraSmooth.LableFY.rows();
    extraSmooth.numLables = numLables;
    extraSmooth.VX = VX.cast<float>();
    extraSmooth.FX = FX;
    PRINT_SMGCO("num lables = " << numVertices * numLables << ", num lables per site = " << numLables);

    if (opts.sameLabelCost < 0.0) {
        PRINT_SMGCO("Automatically setting same label cost...");
        opts.sameLabelCost = std::max(1e-3, perVertexFeatureDifference.minCoeff());
        PRINT_SMGCO("   same label cost = " << opts.sameLabelCost);
    }

    Eigen::MatrixXi AdjFX;
    igl::triangle_triangle_adjacency(FX, AdjFX);

    Eigen::MatrixXi result(numVertices, 6);
    result.block(0, 0, FX.rows(), 3) = FX;

    float optimisationTime = 0.0f;
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
                    if (srcId < targetId) // make sure we dont add neighbours twice...
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
        Eigen::MatrixXf siteLabelCost;
        Eigen::MatrixXi siteLabelCostInt;
        if (opts.labelOrder >= 3) {
            siteLabelCost = Eigen::MatrixXf(numVertices * numLables, 1);
        }
        if (opts.algorithm >= 4) {
            siteLabelCostInt = Eigen::MatrixXi(numVertices, numLables);
            siteLabelCostInt.setConstant(-1);
        }
        for (int i = 0; i < numVertices; i++) {
            float minCost = std::numeric_limits<float>::infinity();
            for (int l = 0; l < numLables; l++) {
                const bool isDegenerate =   extraSmooth.LableFY(l, 0) == extraSmooth.LableFY(l, 1) ||
                                            extraSmooth.LableFY(l, 0) == extraSmooth.LableFY(l, 2) ||
                                            extraSmooth.LableFY(l, 1) == extraSmooth.LableFY(l, 2);
                const bool isDegnerateAndCareAboutDegenerate = setInitialLables > 1 ? isDegenerate : false;
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    sum += perVertexFeatureDifference(FX(i, j), extraSmooth.LableFY(l, j));
                }
                sum *= opts.featureFactor;
                if (sum < minCost && !isDegnerateAndCareAboutDegenerate) {
                    minCost = sum;
                    minLables(i) = l;
                }
                if (addEnergy) {
                    sum += energy(i, l);
                }
                if (opts.labelOrder >= 3) {
                    siteLabelCost(i * numLables + l) = sum;
                }
                const int dataCost = (int) (SCALING_FACTOR * opts.unaryWeight * sum);
                if (opts.algorithm >= 4) {
                    siteLabelCostInt(i, l) = dataCost;
                }

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


        if (USE_CACHING) {
            const int cacheSizeGuess = 5 * numLables * AdjFX.rows();
            PRINT_SMGCO("Cache size guess " << cacheSizeGuess);
            extraSmooth.cache.reserve(cacheSizeGuess);
        }

        precomputeSmoothCost(VX, FX, VY, FY, extraSmooth.LableFY, extraSmooth);
        gc->setSmoothCost(smoothFnGCOSMTrianglewise, static_cast<void*>(&extraSmooth));
        GETTIME(t3);
        PRINT_SMGCO(" -> smooth cost done (" << DURATION_S(t2, t3) << " s)");



        PRINT_SMGCO("Setting initial lables in mode " << INIT_METHODS[setInitialLables] << ", " << setInitialLables);
        if (setInitialLables) {
            if (setInitialLables == 1 || setInitialLables == 2) {
                for (int i = 0; i < numVertices; i++) {
                    int minIndex = minLables(i);
                    gc->setLabel(i, minIndex + i * numLables);
                }
            }
            else if (setInitialLables >= 3) {
                // collect lable indexes in which certain vertices of Y appear
                std::vector<std::vector<int>> vertexInLables;
                vertexInLables.reserve(VY.rows());
                for (int i = 0; i < VY.rows(); i++) {
                    std::vector<int> temp; temp.reserve(400);
                    vertexInLables.push_back(std::move(temp));
                }
                for (int l = 0; l < numLables; l++) {
                    for (int j = 0; j < 3; j++) {
                        const Eigen::Vector3i lable = extraSmooth.LableFY.row(l);
                        const int vertexIndex = extraSmooth.LableFY(l, j);
                        if (setInitialLables == 3) {
                            const bool isDegenerate = lable(0) == lable(1) || lable(0) == lable(2) || lable(1) == lable(2);
                            if (isDegenerate) continue;
                        }
                        vertexInLables[vertexIndex].push_back(l);
                    }
                }

                #if defined(_OPENMP)
                #pragma omp parallel for
                #endif
                for (int i = 0; i < numVertices; i++) {
                    // find best sum for each triangle
                    int minIndex = 0;
                    double bestSum = std::numeric_limits<double>::infinity();
                    Eigen::Vector3i adjacentTris = AdjFX.row(i);

                    const int iterStart = setInitialLables == 3 ? numDegenerate : 0;
                    for (int l = iterStart; l < numLables; l++) { // this loop is lable of the center triangle
                        double sum = 0;
                        for (int j = 0; j < 3; j++) {
                            sum += perVertexFeatureDifference(FX(i, j), extraSmooth.LableFY(l, j));
                        }

                        // find best cost for neighbouring lables that are in zero distance of the current lable l
                        for (int adjIdx = 0; adjIdx < 3; adjIdx++) {
                            double bestCost = std::numeric_limits<double>::infinity();
                            bool output = false;
                            const int adjTri = adjacentTris(adjIdx);
                            if (adjTri == -1) continue;
                            const std::tuple<int, int> commonVerticesBetweenTriangles = extraSmooth.commonVXofFX(i, adjTri);
                            const int idxX1 = std::get<0>(commonVerticesBetweenTriangles);
                            const int idxX2 = std::get<1>(commonVerticesBetweenTriangles);
                            int colIdx11 = -1, colIdx12 = -1, colIdx21 = -1, colIdx22 = -1;
                            for (int j = 0; j < 3; j++) {
                                if (FX(i, j) == idxX1) colIdx11 = j;
                                if (FX(i, j) == idxX2) colIdx12 = j;
                                if (FX(adjTri, j) == idxX1) colIdx21 = j;
                                if (FX(adjTri, j) == idxX2) colIdx22 = j;
                            }
                            const int targetVertex1_1 = extraSmooth.LableFY(l, colIdx11);
                            const int targetVertex1_2 = extraSmooth.LableFY(l, colIdx12);

                            //for (int ll = 0; ll < numLables; ll++) {
                            for (const auto& ll : vertexInLables[targetVertex1_1]) {

                                const int targetVertex2_1 = extraSmooth.LableFY(ll, colIdx21);
                                const int targetVertex2_2 = extraSmooth.LableFY(ll, colIdx22);

                                const bool isNeighbouring = targetVertex1_1 == targetVertex2_1 &&
                                                            targetVertex1_2 == targetVertex2_2;
                                if (!isNeighbouring) {
                                    continue;
                                }
                                int remainingVertex1=-1, remainingVertex2=-1, remainingVertexIdx;
                                for (int j = 0; j < 3; j++) {
                                    if (colIdx11 != j && colIdx12 != j) {
                                        remainingVertex1 = extraSmooth.LableFY(l, j);
                                    }
                                    if (colIdx21 != j && colIdx22 != j) {
                                        remainingVertex2 = extraSmooth.LableFY(ll, j);
                                        remainingVertexIdx = j;
                                    }
                                }
                                const bool remainingVertexNotTheSame = setInitialLables == 3 ? remainingVertex1 != remainingVertex2 : true;
                                if (remainingVertexNotTheSame) {
                                    const double cost = perVertexFeatureDifference(FX(adjTri, remainingVertexIdx), extraSmooth.LableFY(ll, remainingVertexIdx));
                                    if (cost < bestCost) {
                                        bestCost = cost;
                                        if (!output) {
                                            output = true;
                                            //std::cout << adjIdx << ": " << extraSmooth.LableFY.row(ll) << std::endl;
                                        }
                                    }
                                }
                            }
                            sum += bestCost;
                        }
                        if (sum < bestSum) {
                            bestSum = sum;
                            minIndex = l;
                        }
                    }
                    #if defined(_OPENMP)
                    #pragma omp critical
                    #endif
                    gc->setLabel(i, minIndex + i * numLables);
                }
            }
            else {
                PRINT_SMGCO("Init mode not supported :( I will not init -> this could cause bad resaults");
            }

            GETTIME(t4);
            PRINT_SMGCO(" -> init lables done (" << DURATION_S(t3, t4) << " s)");
        }

        if (opts.labelOrder == 1) {
            PRINT_SMGCO("Random label order");
            srand(1618);
            const bool setRandom = true;
            gc->setLabelOrder(setRandom);
        }
        if (opts.labelOrder == 2 && numDegenerate > 0) {
            PRINT_SMGCO("Degnerate labels ordered last");
            Eigen::MatrixXi labelOrder(numLables * numVertices, 1);
            const int numNonDegenerate = numLables - numDegenerate;
            const int startIndexDegenerate = numVertices * numNonDegenerate;
            #if defined(_OPENMP)
            #pragma omp parallel
            #endif
            for (int i = 0; i < numVertices; i++) {
                for (int l = 0; l < numLables; l++) {
                    const int currentLabelIndex = i * numLables + l;
                    int newLabelIndex = -1;
                    if (l >= numNonDegenerate) {
                        newLabelIndex = startIndexDegenerate + i * numDegenerate + l - numNonDegenerate;
                    }
                    else {
                        newLabelIndex = i * numNonDegenerate + l;
                    }
                    labelOrder(currentLabelIndex) = newLabelIndex;
                }
            }

            gc->setLabelOrder(labelOrder.data(), numLables * numVertices);
        }
        if (opts.labelOrder == 3) {
            PRINT_SMGCO("Labels ordered according to min cost");
            std::vector<int> sortedLabels;
            utils::argsort(siteLabelCost, sortedLabels);
            gc->setLabelOrder(sortedLabels.data(), numLables * numVertices);
        }
        if (opts.labelOrder == 4) {
            PRINT_SMGCO("Labels ordered according to alternating min cost");
            const auto siteLabelCostR = siteLabelCost.reshaped<Eigen::RowMajor>(numVertices, numLables); // no copy
            Eigen::MatrixXi sortedLabels(numVertices * numLables, 1);
            #if defined(_OPENMP)
            #pragma omp parallel
            #endif
            for (int i = 0; i < numVertices; i++) {
                std::vector<int> sortedLabelsPerSite;
                utils::argsort(siteLabelCostR.row(i), sortedLabelsPerSite);
                for (int l = 0; l < numLables; l++) {
                    // write such that we have [bestSite0, bestSite1, ...., secondBestSite0, secondBestSite1, ....]
                    sortedLabels(l * numVertices + i) = sortedLabelsPerSite[l] + i * numLables;
                }
            }
            gc->setLabelOrder(sortedLabels.data(), numLables * numVertices);
        }


        PRINT_SMGCO("Using algorithm: " << ALGORITHMS[opts.algorithm]);
        PRINT_SMGCO("Before optimization energy is " << gc->compute_energy() / SCALING_FACTOR);

        GETTIME(t4);
        if (opts.algorithm == 1) {
            gc->expansion(numIters);
        }
        else if (opts.algorithm == 0) {
            gc->swap(numIters);
        }
        else if (opts.algorithm == 2) {
            gc->swap(numIters);
            gc->expansion(numIters);
        }
        else if (opts.algorithm == 3) {
            gc->expansion(numIters);
            gc->swap(numIters);
        }
        /*
         >>>>>>>>>>>>> custom alpha expansion (via line search)
         */
        else if (opts.algorithm >= 4) {
            // experimental
            bool progress = true;
            int iter = 0;
            const int maxiter = numIters == -1 ? 123456790 : numIters;
            int oldEnergy = gc->compute_energy();
            Eigen::MatrixXi tryLabelThisIter(numLables * FX.rows(), 1);
            tryLabelThisIter.setConstant(0);
            int tryLabelIndex = 0;
            Eigen::MatrixX<bool> trySiteThisIter(FX.rows(), 1);
            trySiteThisIter.setConstant(true);
            std::vector<int> sortedf(FX.rows());
            for (int i = 0; i < FX.rows(); i++)
                sortedf[i] = i;

            std::cout << opts.algorithm << std::endl;

            while (progress && iter < maxiter) {
                progress = false;
                int successFullExpansions = 0;
                int numExpansions = 0;
                int triedNumSites = 0;

                // determine order
                GETTIME(t00);
                if (opts.algorithm >= 6) {
                    Eigen::MatrixXf pairwiseCosts(FX.rows(), 1);
                    #if defined(_OPENMP)
                    #pragma omp critical
                    #endif
                    for (int f = 0; f < FX.rows(); f++) {
                        int cost = 0;
                        const int currentRealLabel = gc->whatLabel(f);
                        for (int i = 0; i < 3; i++) {
                            const int neighf = AdjFX(f, i);
                            if (neighf == -1) continue;
                            const int neighLabel = gc->whatLabel(neighf);
                            const int smooth = smoothFnGCOSMTrianglewise(f, neighf, currentRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
                            cost += smooth;
                        }
                        pairwiseCosts(f) = cost;
                    }
                    utils::argsort(pairwiseCosts, sortedf);
                }
                for (int findex = 0; findex < FX.rows(); findex++) {
                    const int f = sortedf[findex];

                    if (!trySiteThisIter(f) && (opts.algorithm == 5 || opts.algorithm == 7)) continue;
                    const int currentRealLabel = gc->whatLabel(f);
                    const int currentLabel = currentRealLabel - f * numLables;
                    int cost = siteLabelCostInt(f, currentLabel);
                    for (int i = 0; i < 3; i++) {
                        const int neighf = AdjFX(f, i);
                        if (neighf == -1) continue;
                        const int neighLabel = gc->whatLabel(neighf);// - neighf * numLables;
                        //cost += siteLabelCostInt(neighf, neighLabel);
                        const int smooth = smoothFnGCOSMTrianglewise(f, neighf, currentRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
                        cost += smooth;
                    }
                    triedNumSites++;



                    const int numThreads = omp_thread_count();
                    Eigen::MatrixXi bestLabel(numThreads, 1);
                    bestLabel.setConstant(-1);
                    Eigen::MatrixXi bestCost(numThreads, 1);
                    bestCost.setConstant(cost);
                    Eigen::MatrixXi successFullExpansionCounter(numThreads, 1);
                    successFullExpansionCounter.setConstant(0);
                    Eigen::MatrixXi expansionCounter(numThreads, 1);
                    expansionCounter.setConstant(0);
                    #if defined(_OPENMP)
                    #pragma omp critical
                    #endif
                    for (int l = 0; l < numLables; l++) {
                        if ((opts.algorithm == 5 || opts.algorithm == 7) && tryLabelThisIter(f * numLables + l) < tryLabelIndex)
                            continue; // dont "expand" label
                        const int threadId = getThreadId();
                        const int newLabel = l;
                        const int newRealLabel = l + f * numLables;
                        int newCost =  siteLabelCostInt(f, newLabel);
                        for (int i = 0; i < 3; i++) {
                            const int neighf = AdjFX(f, i);
                            if (neighf == -1) continue;
                            const int neighLabel = gc->whatLabel(neighf);// - neighf * numLables;
                            //newCost += siteLabelCostInt(neighf, neighLabel);
                            const int smooth = smoothFnGCOSMTrianglewise(f, neighf, newRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
                            newCost += smooth;
                        }
                        if (newCost < bestCost(threadId)) {
                            bestCost(threadId) = newCost;
                            bestLabel(threadId) = newRealLabel;
                            tryLabelThisIter(f * numLables + l) = tryLabelIndex + 1; // keep this label for the next (smaller) queue
                            successFullExpansionCounter(threadId) += 1;
                        }
                        expansionCounter(threadId) += 1;
                    }

                    int newBestLabel = -1;
                    int newBestCost = cost;
                    for (int t = 0; t < numThreads; t++) {
                        if (bestCost(t) < newBestCost) {
                            newBestCost = bestCost(t);
                            newBestLabel = bestLabel(t);
                        }
                        successFullExpansions += successFullExpansionCounter(t);
                        numExpansions += expansionCounter(t);
                    }
                    if (newBestLabel != -1) {
                        assert(newBestLabel >= ((f) * numLables) && newBestLabel < ((f+1) * numLables));
                        gc->setLabel(f, newBestLabel);
                        progress = true;
                    }
                    if (newBestLabel == currentRealLabel || newBestLabel == -1) {
                        trySiteThisIter(f) = false; // no progress on this site
                    }

                }

                GETTIME(t01);
                assert(progress == successFullExpansions > 0);
                int newEnergy = gc->compute_energy();
                PRINT_SMGCO("Expansion iter = " << iter << ";     energy = " << newEnergy
                            << ";     # expansions = " << numExpansions
                            << ";     # successfull expansions = " << successFullExpansions
                            << ";     took = " << DURATION_MS(t00, t01) << "ms");
                if (newEnergy > oldEnergy) {
                    PRINT_SMGCO("warning, energy increased")
                }
                oldEnergy = newEnergy;

                // adaptive cycle idea from GCO
                const bool adaptiveAlgorithm = opts.algorithm == 5 || opts.algorithm == 7;
                // No expansion was successful, so try more labels from the previous queue
                if (tryLabelIndex > 0 && adaptiveAlgorithm && successFullExpansions == 0) {
                    progress = true;
                    tryLabelIndex--;
                }
                // Some expansions were successful, so focus on them in a new queue
                if (successFullExpansions > 0) tryLabelIndex++;
                // All expansions were successful, so do another complete sweep
                if (successFullExpansions == numLables * numVertices) tryLabelIndex++;




                iter++;
            }
        }
        GETTIME(t5);
        PRINT_SMGCO("After optimization energy is " << gc->compute_energy() / SCALING_FACTOR);
        PRINT_SMGCO("Optimisation took: " << DURATION_S(t4, t5) << " s");
        optimisationTime = DURATION_MS(t4, t5) / 1000.0f;

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
    else {
        std::vector<std::vector<int>> perVertexMatches(VX.rows());
        gluedp2p = Eigen::MatrixXi(VX.rows(), 2);
        for (int i = 0; i < p2p_unique.rows(); i++) {
            perVertexMatches[p2p_unique(i, 0)].push_back(p2p_unique(i, 1));
        }
        for (int i = 0; i < VX.rows(); i++) {
            double bestEnergy = std::numeric_limits<double>::infinity();
            int bestVertex = 0;
            for (const auto& targetVertex : perVertexMatches[i]) {
                const float energy = perVertexFeatureDifference(i, targetVertex);
                if (energy < bestEnergy) {
                    bestEnergy = energy;
                    bestVertex = targetVertex;
                }
            }
            gluedp2p.row(i) << i, bestVertex;
        }
        gluedSolution = result;
    }
    return std::make_tuple(optimisationTime, gluedp2p, gluedSolution, p2p_unique, result);
}


std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise() {
    TriangleWiseOpts opts;
    return triangleWise(opts);
}

} // namespace smgco
