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


std::string INIT_METHODS[9] = { "NO_INIT",
                                "MIN_LABEL",
                                "MIN_LABEL_NON_DEGENERATE",
                                "TRI_NEIGHBOURS_NON_DEGENERATE",
                                "TRI_NEIGHBOURS",
                                "SINKHORN",
                                "LOWRES_DECIMATE",
                                "LOWRES_QSLIM",
                                "RANDOM"};
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
/*






 */
std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> GCOSM::triangleWise(TriangleWiseOpts opts) {

    const int numFacesThreshold = 1200;
    if (opts.setInitialLables == 6 && FX.rows() < numFacesThreshold && FY.rows() < numFacesThreshold){
        PRINT_SMGCO("Changing init mode to 4 since number of faces is too small");
        opts.setInitialLables = 4;
    }

    GCOTrianglewiseExtra extraSmooth(opts);
    const int setInitialLables = opts.setInitialLables;

    const COST_MODE costMode = opts.costMode;
    std::cout << prefix << "Using cost mode = " << costMode << std::endl;

    const unsigned long numVertices = FX.rows();
    unsigned long numDegenerate = 0;
    extraSmooth.LableFY = buildLableSpace(VY, FY, numDegenerate, opts);
    const unsigned long numLables = extraSmooth.LableFY.rows();
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
        const unsigned long numFakeLables = FX.rows() * numLables;
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
        for (unsigned long lf = 0; lf < numFakeLables; lf++) {
            const int rowIndex = lf / numLables;
            const int colIndex = lf - rowIndex * numLables;
            extraSmooth.lableToIndex.row(lf) << rowIndex, colIndex ;
        }


        extraSmooth.sharedVertIds.reserve(2 * AdjFX.rows());
        std::vector<int> sharedThis;
        sharedThis.reserve(2);
        std::vector<int> sharedThat;
        sharedThat.reserve(2);
        for (int i = 0; i < FX.rows(); i++) {
            const Eigen::MatrixXi currentFace = FX.row(i);
            for (int j = 0; j < 3; j++) {
                const int neighbouringFaceIndex = AdjFX(i, j);
                if (neighbouringFaceIndex == -1) continue;
                const Eigen::MatrixXi neighbouringFace = FX.row(neighbouringFaceIndex);
                sharedThis.clear();
                sharedThat.clear();

                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        if (currentFace(k) == neighbouringFace(l)) {
                            sharedThis.push_back(k);
                            sharedThat.push_back(l);
                        }
                    }
                }
                const std::tuple<int, int> sharedIdsFirst  = std::make_tuple(sharedThis[0], sharedThat[0]);
                const std::tuple<int, int> sharedIdsSecond = std::make_tuple(sharedThis[1], sharedThat[1]);
                const std::tuple<int, int> key = std::make_tuple(i, neighbouringFaceIndex);
                extraSmooth.sharedVertIds.insert({key, std::make_tuple(sharedIdsFirst, sharedIdsSecond)});
            }
        }

        Eigen::MatrixXf softP;
        if (opts.sinkhornEnergyMod || opts.setInitialLables == 5) {
            softP = perVertexFeatureDifference.cast<float>();
            const float entropy = opts.sinkhornEntropy < -1e8 ? perVertexFeatureDifference.mean() : opts.sinkhornEntropy;
            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int i = 0; i < softP.rows(); i++) {
                for (int j = 0; j < softP.cols(); j++) {
                    softP(i, j) = std::exp(- entropy * softP(i, j)) + 1e-8;
                }
            }

            bool converged = false;
            const int maxIter = opts.sinkhornIters;
            int iter = 0;
            while (iter < maxIter) {
                #if defined(_OPENMP)
                //#pragma omp parallel for
                #endif
                for (int j = 0; j < softP.cols(); j++) {
                    softP.col(j) = softP.col(j) / softP.col(j).sum();
                }


                #if defined(_OPENMP)
                //#pragma omp parallel for
                #endif
                for (int i = 0; i < softP.rows(); i++) {
                    softP.row(i) = softP.row(i) / softP.row(i).sum();
                }

                iter++;
            }


            for (int i = 0; i < softP.rows(); i++) {
                softP.row(i) = softP.row(i) / softP.row(i).maxCoeff();
            }

        }

        Eigen::MatrixXi ColourFX(FX.rows(), 1); ColourFX.setZero();
        int numRed = 0, numBlue = 0;
        if (opts.bicolouring) {
            PRINT_SMGCO("  bicolouring is switched to ON");
            ColourFX = utils::greedyDualTriGraphColouring(FX);
            for (int i = 0; i < FX.rows(); i++) {
                if (ColourFX(i))
                    numRed++;
                else
                    numBlue++;
            }
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
        const double maxEnergy = perVertexFeatureDifference.maxCoeff();
        for (int i = 0; i < numVertices; i++) {
            float minCost = std::numeric_limits<float>::infinity();
            for (unsigned long l = 0; l < numLables; l++) {
                const bool isDegenerate =   extraSmooth.LableFY(l, 0) == extraSmooth.LableFY(l, 1) ||
                                            extraSmooth.LableFY(l, 0) == extraSmooth.LableFY(l, 2) ||
                                            extraSmooth.LableFY(l, 1) == extraSmooth.LableFY(l, 2);
                const bool isDegnerateAndCareAboutDegenerate = setInitialLables > 1 ? isDegenerate : false;
                double sum = 0;
                for (int j = 0; j < 3; j++) {
                    if (opts.sinkhornEnergyMod) {
                        sum += maxEnergy * (1 - softP(FX(i, j), extraSmooth.LableFY(l, j)));
                    }
                    else {
                        sum += perVertexFeatureDifference(FX(i, j), extraSmooth.LableFY(l, j));
                    }
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
                else {
                    const int fakeLable = i * numLables + l;
                    const int numSites = 1;

                    GCoptimization::SparseDataCost* c = new GCoptimization::SparseDataCost[numSites];
                    c[0].site = i;
                    c[0].cost = dataCost;
                    gc->setDataCost(fakeLable, c, numSites);
                    delete[] c;
                }
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



        const int strIndex = setInitialLables >= 10 ? 7 : setInitialLables;
        PRINT_SMGCO("Setting initial lables in mode " << INIT_METHODS[strIndex] << ", " << setInitialLables);
        triangleWiseInit(opts, extraSmooth, gc, AdjFX, minLables, softP, numDegenerate);

        if (opts.labelOrder == 1) {
            PRINT_SMGCO("Random label order");
            srand(1618);
            const bool setRandom = true;
            gc->setLabelOrder(setRandom);
        }
        if (opts.labelOrder == 2 && numDegenerate > 0) {
            PRINT_SMGCO("Degnerate labels ordered last");
            Eigen::MatrixX<GCoptimization::LabelID> labelOrder(numLables * numVertices, 1);
            const int numNonDegenerate = numLables - numDegenerate;
            const int startIndexDegenerate = numVertices * numNonDegenerate;
            #if defined(_OPENMP)
            #pragma omp parallel
            #endif
            for (int i = 0; i < numVertices; i++) {
                for (unsigned long l = 0; l < numLables; l++) {
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
            std::vector<GCoptimization::LabelID> sortedLabels;
            utils::argsort(siteLabelCost, sortedLabels);
            gc->setLabelOrder(sortedLabels.data(), numLables * numVertices);
        }
        if (opts.labelOrder == 4) {
            PRINT_SMGCO("Labels ordered according to alternating min cost");
            const auto siteLabelCostR = siteLabelCost.reshaped<Eigen::RowMajor>(numVertices, numLables); // no copy
            Eigen::MatrixX<GCoptimization::LabelID> sortedLabels(numVertices * numLables, 1);
            #if defined(_OPENMP)
            #pragma omp parallel
            #endif
            for (int i = 0; i < numVertices; i++) {
                std::vector<GCoptimization::LabelID> sortedLabelsPerSite;
                utils::argsort(siteLabelCostR.row(i), sortedLabelsPerSite);
                for (unsigned long l = 0; l < numLables; l++) {
                    // write such that we have [bestSite0, bestSite1, ...., secondBestSite0, secondBestSite1, ....]
                    sortedLabels(l * numVertices + i) = sortedLabelsPerSite[l] + i * numLables;
                }
            }
            gc->setLabelOrder(sortedLabels.data(), numLables * numVertices);
        }


        PRINT_SMGCO("Using algorithm: " << ALGORITHMS[opts.algorithm]);
        PRINT_SMGCO("Before optimization energy is " << computeEnergy(gc, AdjFX, siteLabelCostInt, static_cast<void*>(&extraSmooth)) / SCALING_FACTOR);

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
            triangleWiseAlgorithm(opts, extraSmooth, gc, AdjFX, siteLabelCostInt, ColourFX, numBlue);
        }
        GETTIME(t5);
        PRINT_SMGCO("After optimization energy is " << computeEnergy(gc, AdjFX, siteLabelCostInt, static_cast<void*>(&extraSmooth)) / SCALING_FACTOR);
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
        const std::vector<std::vector<int>> originalPerVertexMatches = perVertexMatches;

        // add all vertices in convex hull of existing vertices
        Eigen::MatrixXf ElenY;
        Eigen::MatrixXi EY;
        igl::edge_lengths(VY, FY, ElenY);
        igl::edges(FY, EY);
        std::vector<std::vector<int>> vertexEdgeAdjacency(VY.rows());
        for (int e = 0; e < EY.rows(); e++) {
            vertexEdgeAdjacency[EY(e, 0)].push_back(e);
            vertexEdgeAdjacency[EY(e, 1)].push_back(e);
        }
        for (int i = 0; i < VX.rows(); i++) {
            float maxGeodist = 0;
            float maxEdgeLength = 0;
            for (const auto& matchedVertex1 : originalPerVertexMatches[i]) {
                for (const auto& matchedVertex2 : originalPerVertexMatches[i]) {
                    const float geodist12 = extraSmooth.geoDistY(matchedVertex1, matchedVertex2);
                    maxGeodist = std::max(geodist12, maxGeodist);
                }
                for (const auto e : vertexEdgeAdjacency[matchedVertex1]) {
                    maxEdgeLength = std::max(maxEdgeLength, ElenY(e));
                }
            }
            maxGeodist += maxEdgeLength;

            // any point on Y that is at most maxGeodist away from any of the original matched points is a viable candidate
            for (int j = 0; j < VY.rows(); j++) {
                float maxGeoDistVj = 0;
                for (const auto& matchedVertex : originalPerVertexMatches[i]) {
                    maxGeoDistVj = std::max(maxGeoDistVj, extraSmooth.geoDistY(j, matchedVertex));
                }
                if (maxGeoDistVj <= maxGeodist) {
                    perVertexMatches[i].push_back(j);
                }
            }
        }

        int numMatches = 0;
        for (int i = 0; i < VX.rows(); i++) {
            numMatches += perVertexMatches[i].size();
        }
        p2p_unique.conservativeResize(numMatches, 2);
        int index = 0;
        for (int i = 0; i < VX.rows(); i++) {
            for (const auto& match : perVertexMatches[i]) {
                p2p_unique.row(index) << i, match;
                index++;
            }
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
