#include "gco_shape_matching.hpp"
#include <igl/edges.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/exact_geodesic.h>
#include <igl/unique_rows.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>
#include <igl/qslim.h>
#include <igl/decimate.h>
#include "energy/deformationEnergy.hpp"
#include <chrono>



namespace smgco {
std::string ALGORITHMS[10] = { "ALPHA-BETA SWAP",
                                "ALPHA EXPANSION",
                                "SWAP followed by EXPANSION",
                                "EXPANSION followed by SWAP",
                                "LINESEARCH",
                                "LINESEARCH + ADAPTIVE",
                                "LINESEARCH + SITE REORDERING (w.r.t. min smooth cost)",
                                "LINESEARCH + ADAPTIVE + SITE REORDERING (w.r.t. min smooth cost)",
                                "LINESEARCH + SITE REORDERING (w.r.t. max smooth cost)",
                                "LINESEARCH + ADAPTIVE + SITE REORDERING (w.r.t. max smooth cost)"};

unsigned long long GCOSM::computeEnergy(GCoptimizationGeneralGraph* gc,
                                        const Eigen::MatrixXi& AdjFX,
                                        const Eigen::MatrixXi& siteLabelCostInt,
                                        void* extraData) {
    unsigned long long cost = 0;
    const size_t numLables = siteLabelCostInt.cols();
    const size_t numTris = siteLabelCostInt.rows();
    for (int findex = 0; findex < numTris; findex++) {
        const int f = findex;
        const GCoptimization::LabelID currentRealLabel = gc->whatLabel(f);
        const GCoptimization::LabelID currentLabel = currentRealLabel - f * numLables;
        cost += siteLabelCostInt(f, currentLabel);
        for (int i = 0; i < 3; i++) {
            const int neighf = AdjFX(f, i);
            if (neighf == -1) continue;
            const GCoptimization::LabelID neighLabel = gc->whatLabel(neighf);// - neighf * numLables;
            cost += siteLabelCostInt(neighf, neighLabel-  neighf * numLables);
            const long long smooth = smoothFnGCOSMTrianglewise(f, neighf, currentRealLabel, neighLabel, extraData);
            cost += smooth;
        }
    }

    return cost;
}



void GCOSM::triangleWiseAlgorithm(TriangleWiseOpts& opts,
                                  GCOTrianglewiseExtra& extraSmooth,
                                  GCoptimizationGeneralGraph *gc,
                                  const Eigen::MatrixXi& AdjFX,
                                  const Eigen::MatrixXi& siteLabelCostInt,
                                  const Eigen::MatrixXi& ColourFX,
                                  const int numBlue ) {
    PRINT_SMGCO("Using algorithm: " << ALGORITHMS[opts.algorithm]);
    const int numVertices = FX.rows();
    const int numLables = extraSmooth.LableFY.rows();
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
        long long oldEnergy = computeEnergy(gc, AdjFX, siteLabelCostInt, static_cast<void*>(&extraSmooth));
        Eigen::MatrixXi tryLabelThisIter(numLables * FX.rows(), 1);
        tryLabelThisIter.setConstant(0);
        int tryLabelIndex = 0;
        std::vector<unsigned long> sortedf(FX.rows());
        unsigned long redIndex = 0, blueIndex = 0;
        for (int i = 0; i < FX.rows(); i++) {
            if (ColourFX(i)) {
                sortedf[numBlue + redIndex] = i;
                redIndex++;
                continue;
            }
            sortedf[blueIndex] = i;
            blueIndex++;
        }
        assert(redIndex + blueIndex == FX.rows());

        while (progress && iter < maxiter) {
            progress = false;
            unsigned long successFullExpansions = 0;
            unsigned long numExpansions = 0;
            unsigned long triedNumSites = 0;

            // determine order
            GETTIME(t00);
            if (opts.algorithm >= 6) {
                Eigen::MatrixX<long long> pairwiseCosts(FX.rows(), 1);
                #if defined(_OPENMP)
                //#pragma omp parallel
                #endif
                for (unsigned long f = 0; f < FX.rows(); f++) {
                    long long cost = 0;
                    const GCoptimization::LabelID currentRealLabel = gc->whatLabel(f);
                    for (int i = 0; i < 3; i++) {
                        const int neighf = AdjFX(f, i);
                        if (neighf == -1) continue;
                        const GCoptimization::LabelID neighLabel = gc->whatLabel(neighf);
                        const long long smooth = smoothFnGCOSMTrianglewise(f, neighf, currentRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
                        cost += smooth;
                    }
                    if (opts.algorithm >= 8) {
                        cost -= cost;
                    }
                    if (opts.bicolouring && ColourFX(f)) {
                        if (opts.algorithm >= 8) {
                            cost =  std::numeric_limits<long long>::max() + cost;
                        }
                        else {
                            cost = -std::numeric_limits<long long>::max() + cost;
                        }
                    }
                    pairwiseCosts(f) = cost;
                }
                utils::argsort(pairwiseCosts, sortedf);
            }
            for (int findex = 0; findex < FX.rows(); findex++) {
                const int f = sortedf[findex];

                //if (!trySiteThisIter(f) && (opts.algorithm == 5 || opts.algorithm == 7)) continue;
                const GCoptimization::LabelID currentRealLabel = gc->whatLabel(f);
                const GCoptimization::LabelID currentLabel = currentRealLabel - f * numLables;
                unsigned long long cost = siteLabelCostInt(f, currentLabel);
                for (int i = 0; i < 3; i++) {
                    const int neighf = AdjFX(f, i);
                    if (neighf == -1) continue;
                    const GCoptimization::LabelID neighLabel = gc->whatLabel(neighf);// - neighf * numLables;
                    //cost += siteLabelCostInt(neighf, neighLabel);
                    const long long smooth = smoothFnGCOSMTrianglewise(f, neighf, currentRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
                    cost += smooth;
                }
                triedNumSites++;



                const int numThreads = omp_thread_count();
                Eigen::MatrixX<GCoptimization::LabelID> bestLabel(numThreads, 1);
                bestLabel.setConstant(-1);
                Eigen::MatrixX<unsigned long long> bestCost(numThreads, 1);
                bestCost.setConstant(cost);
                Eigen::MatrixX<unsigned long> successFullExpansionCounter(numThreads, 1);
                successFullExpansionCounter.setConstant(0);
                Eigen::MatrixX<unsigned long> expansionCounter(numThreads, 1);
                expansionCounter.setConstant(0);
                #if defined(_OPENMP)
                //#pragma omp parallel
                #endif
                for (GCoptimization::LabelID l = 0; l < numLables; l++) {
                    if ((opts.algorithm == 5 || opts.algorithm == 7 || opts.algorithm == 9) && tryLabelThisIter(f * numLables + l) < tryLabelIndex)
                        continue; // dont "expand" label
                    const int threadId = getThreadId();
                    const GCoptimization::LabelID newLabel = l;
                    const GCoptimization::LabelID newRealLabel = l + f * numLables;
                    unsigned long long newCost = siteLabelCostInt(f, newLabel);
                    for (int i = 0; i < 3; i++) {
                        const int neighf = AdjFX(f, i);
                        if (neighf == -1) continue;
                        const GCoptimization::LabelID neighLabel = gc->whatLabel(neighf);// - neighf * numLables;
                        //newCost += siteLabelCostInt(neighf, neighLabel);
                        const long long smooth = smoothFnGCOSMTrianglewise(f, neighf, newRealLabel, neighLabel, static_cast<void*>(&extraSmooth));
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

                GCoptimization::LabelID newBestLabel = -1;
                unsigned long long newBestCost = cost;
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
            }

            GETTIME(t01);
            assert(progress == successFullExpansions > 0);
            long long newEnergy = computeEnergy(gc, AdjFX, siteLabelCostInt, static_cast<void*>(&extraSmooth));
            PRINT_SMGCO("Expansion iter = " << iter << ";     energy = " << newEnergy
                        << ";     # expansions = " << numExpansions
                        << ";     # successfull expansions = " << successFullExpansions
                        << ";     took = " << DURATION_MS(t00, t01) << "ms");
            if (newEnergy > oldEnergy) {
                PRINT_SMGCO("warning, energy increased")
            }
            oldEnergy = newEnergy;

            // adaptive cycle idea from GCO
            const bool adaptiveAlgorithm = opts.algorithm == 5 || opts.algorithm == 7 || opts.algorithm == 9;
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
}


} // namespace smgco
