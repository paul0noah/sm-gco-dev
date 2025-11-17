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


void GCOSM::triangleWiseInit(TriangleWiseOpts& opts,
                             GCOTrianglewiseExtra& extraSmooth,
                             GCoptimizationGeneralGraph *gc,
                             const Eigen::MatrixXi& AdjFX,
                             Eigen::MatrixXi& minLables,
                             const Eigen::MatrixXf& softP,
                             const unsigned long numDegenerate) {
    const int numVertices = FX.rows();
    const int numLables = extraSmooth.LableFY.rows();
    const int setInitialLables = opts.setInitialLables;

    const int strIndex = opts.setInitialLables >= 10 ? 7 : opts.setInitialLables;
    PRINT_SMGCO("Using mode " << INIT_METHODS[strIndex] << ", " << opts.setInitialLables);
    std::vector<std::vector<int>> vertexInLables;
    if (setInitialLables >= 3) {
        // collect lable indexes in which certain vertices of Y appear
        vertexInLables.reserve(VY.rows());
        for (int i = 0; i < VY.rows(); i++) {
            std::vector<int> temp; temp.reserve(400);
            vertexInLables.push_back(std::move(temp));
        }
        for (unsigned long l = 0; l < numLables; l++) {
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
    }


    if (setInitialLables) {
        if (setInitialLables > 10) {
            srand(setInitialLables);
            for (int i = 0; i < numVertices; i++) {
                int randomLabel = rand() % (numLables + 1);;
                gc->setLabel(i, randomLabel + i * numLables);
            }
        }
        else if (setInitialLables == 1 || setInitialLables == 2) {
            for (int i = 0; i < numVertices; i++) {
                int minIndex = minLables(i);
                gc->setLabel(i, minIndex + i * numLables);
            }
        }
        else if (setInitialLables == 5) {

            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int i = 0; i < FX.rows(); i++) {
                float bestEnergy = std::numeric_limits<float>::infinity();
                for (unsigned long l = 0; l < numLables; l++) {
                    float energy = 0;
                    for (int j = 0; j < 3; j++) {
                        const float matchingProbability = softP(FX(i, j), extraSmooth.LableFY(l, j));
                        energy += 1.0f - matchingProbability;
                    }
                    if (energy < bestEnergy) {
                        bestEnergy = energy;
                        minLables(i) = l;
                    }
                }
            }

            for (int i = 0; i < FX.rows(); i++) {
                gc->setLabel(i, minLables(i) + i * numLables);
            }

        }
        /*









         */
        else if (setInitialLables == 6 || setInitialLables == 7) {
            Eigen::VectorXi I, J;
            Eigen::MatrixXi FYlr, FXlr;
            Eigen::MatrixXd VYlr, VXlr;
            const int numFacesLowres = 1000;
            if (setInitialLables == 6) {
                igl::decimate(VX, FX, numFacesLowres, VXlr, FXlr, J, I);
                igl::decimate(VY, FY, numFacesLowres, VYlr, FYlr, J, I);
            }
            else {
                igl::qslim(VX, FX, numFacesLowres, VXlr, FXlr, J, I);
                igl::qslim(VY, FY, numFacesLowres, VYlr, FYlr, J, I);
            }
            Eigen::MatrixXf GeoDistX = utils::computeGeodistMatrix(VX, FX);
            Eigen::MatrixXi X_lr_2_hr, Y_lr_2_hr;
            utils::knnsearch(VXlr, VX, X_lr_2_hr);
            utils::knnsearch(VYlr, VY, Y_lr_2_hr);
            const int nn = 11;
            X_lr_2_hr = utils::geodistknnsearch(X_lr_2_hr, VX, FX, GeoDistX, 0.5, nn);
            Y_lr_2_hr = utils::geodistknnsearch(Y_lr_2_hr, VY, FY, extraSmooth.geoDistY, 0.5, nn);
            Eigen::MatrixXd featDifflr(VXlr.rows(), VYlr.rows());
            for (int i = 0; i < VXlr.rows(); i++) {
                for (int j = 0; j < VYlr.rows(); j++) {
                    featDifflr(i, j) = 0.9 * perVertexFeatureDifference(X_lr_2_hr(i, 0), Y_lr_2_hr(j, 0));
                    for (int n = 1; n < nn; n++) {
                        featDifflr(i, j) += 0.01 * perVertexFeatureDifference(X_lr_2_hr(i, n), Y_lr_2_hr(j, n));;
                    }

                }
            }
            X_lr_2_hr.conservativeResize(X_lr_2_hr.rows(), 1);
            Y_lr_2_hr.conservativeResize(Y_lr_2_hr.rows(), 1);
            GCOSM smGCO(VXlr, FXlr, VYlr, FYlr, featDifflr);
            smGCO.updatePrefix("[GCOSM - INIT] ");
            TriangleWiseOpts optsCopy = opts;
            optsCopy.setInitialLables = 4;
            smGCO.setMaxIter(20);
            const auto out = smGCO.triangleWise(optsCopy);
            Eigen::MatrixXi p2plr = std::get<1>(out);
            Eigen::MatrixXi p2plrRaw = std::get<3>(out);

            PRINT_SMGCO(" init: matching extractioin");
            std::vector<std::vector<int>> vertexMatches(VXlr.rows());
            for (int i = 0; i < p2plrRaw.rows(); i++) {
                vertexMatches[p2plrRaw(i, 0)].push_back(Y_lr_2_hr(p2plrRaw(i, 1)));
            }
            Eigen::MatrixXf maxGeoDist(VXlr.rows(), 1);
            maxGeoDist.setZero();
            for (int i = 0; i < VXlr.rows(); i++) {
                for (const auto& targetVertex1 : vertexMatches[i]) {
                    for (const auto& targetVertex2 : vertexMatches[i]) {
                        maxGeoDist(i) = std::max(maxGeoDist(i), extraSmooth.geoDistY(targetVertex1, targetVertex2));
                    }
                }
            }
            Eigen::MatrixXf LX, LY;
            igl::edge_lengths(VX, FX, LX);
            igl::edge_lengths(VY, FY, LY);
            const float threshX = 20 * utils::median(LX);
            const float threshY = 20 * utils::median(LY);

            Eigen::MatrixXi p2phr = p2plr;
            const float matchingDistThreshold = LY.maxCoeff();
            int idx = 0;
            for (int i = 0; i < p2phr.rows(); i++) {
                if (maxGeoDist(i) >= matchingDistThreshold) {
                    continue;
                }
                const int vertexX =  X_lr_2_hr(p2plr(i, 0));
                const int vertexY =  Y_lr_2_hr(p2plr(i, 1));

                p2phr.row(idx) << vertexX, vertexY;
                idx++;
            }
            p2phr.conservativeResize(idx, 2);
            PRINT_SMGCO(" init: kept " << p2phr.rows() << " of " << VXlr.rows()<< " matches from lowres")



            // find closest matched vertices (on x) for each vertex of X
            const int numNNMatchedPoints = 5;
            const Eigen::MatrixXf AllPoints     = GeoDistX(Eigen::all, p2phr.col(0));
            const Eigen::MatrixXf MatchedPoints = GeoDistX(p2phr.col(0), p2phr.col(0));
            Eigen::MatrixXi closestMatchedPoints;
            utils::knnsearch(AllPoints, MatchedPoints, closestMatchedPoints, numNNMatchedPoints);

            PRINT_SMGCO(" init: new cost computation");
            // for each vertex of X. find the best vertex on Y while using lowres matching
            Eigen::MatrixXf newCost(VX.rows(), VY.rows()); newCost.setZero();
            Eigen::MatrixXi interpolatedMatching(VX.rows(), 2);
            const Eigen::MatrixXf& GeoDistY = extraSmooth.geoDistY;
            for (int i = 0; i < VX.rows(); i++) {

                // normalise with maxgeodist of closes match vertices
                float maxGeodistBetweenMatchedOnX = 0.0f, maxGeodistBetweenMatchedOnY = 0.0f;
                for (int k = 0; k < numNNMatchedPoints; k++) {
                    const int matchedIdxK = closestMatchedPoints(i, k);
                    for (int kk = 0; kk < numNNMatchedPoints; kk++) {
                        const int matchedIdxKK = closestMatchedPoints(i, kk);
                        maxGeodistBetweenMatchedOnX = std::max(maxGeodistBetweenMatchedOnX, GeoDistX(p2phr(matchedIdxK, 0), p2phr(matchedIdxKK, 0)));
                        maxGeodistBetweenMatchedOnY = std::max(maxGeodistBetweenMatchedOnY, GeoDistY(p2phr(matchedIdxK, 1), p2phr(matchedIdxKK, 1)));
                    }
                }

                // compute on X
                Eigen::MatrixXf distancesX(1, numNNMatchedPoints);
                for (int k = 0; k < numNNMatchedPoints; k++) {
                    const int matchedIdxK = closestMatchedPoints(i, k);
                    distancesX(0, k) = GeoDistX(i, p2phr(matchedIdxK, 0)) / maxGeodistBetweenMatchedOnX;
                }

                // find best matching on Y
                float bestCost = std::numeric_limits<float>::infinity();
                int bestIndex = 0;
                for (int j = 0; j < VY.rows(); j++) {
                    float maxGeoDistToMatched = 0.0;
                    for (int k = 0; k < numNNMatchedPoints; k++) {
                        const int matchedIdxK = closestMatchedPoints(i, k);
                        maxGeoDistToMatched = std::max(maxGeoDistToMatched, GeoDistY(j, p2phr(matchedIdxK, 1)));
                    }

                    float cost = 0;
                    for (int k = 0; k < numNNMatchedPoints; k++) {
                        const int matchedIdxK = closestMatchedPoints(i, k);
                        const float distanceKonY = GeoDistY(j, p2phr(matchedIdxK, 1)) / maxGeodistBetweenMatchedOnY;
                        cost += std::abs(distanceKonY - distancesX(0, k));
                    }
                    newCost(i, j) = cost * maxGeodistBetweenMatchedOnX * maxGeodistBetweenMatchedOnY;
                    if (maxGeoDistToMatched > maxGeodistBetweenMatchedOnY) {
                        continue;
                    }
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestIndex = j;
                    }
                }

                interpolatedMatching(i, 0) = i;
                interpolatedMatching(i, 1) = bestIndex;
            }



            /*minLables = interpolatedMatching;
            std::cout << minLables.rows() << minLables.cols() << std::endl;
            return;*/


            std::cout << "check costmodes here" << std::endl;
            /*
             newCost.setConstant(1);
             for (int i = 0; i < interpolatedMatching.rows(); i++)
                 newCost(interpolatedMatching(i, 0), interpolatedMatching(i, 1)) = 0;
            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int i = 0; i < FX.rows(); i++) {
                float bestEnergy = std::numeric_limits<float>::infinity();
                for (unsigned long l = 0; l < numLables; l++) {
                    float cost = 0;
                    for (int j = 0; j < 3; j++) {
                        cost += newCost(FX(i, j), extraSmooth.LableFY(l, j));
                    }
                    if (cost < bestEnergy) {
                        bestEnergy = cost;
                        minLables(i) = l;
                    }
                }
            }

            for (int i = 0; i < FX.rows(); i++) {
                gc->setLabel(i, minLables(i) + i * numLables);
            }
            return;*/

            /*
            Eigen::MatrixXf GeoDistXFeat = GeoDistX(Eigen::all, p2phr.col(0));
            Eigen::MatrixXf GeoDistYFeat = extraSmooth.geoDistY(Eigen::all, p2phr.col(1));
            std::cout << threshX << ", " << threshY << std::endl;
            for (int i = 0; i < VX.rows(); i++) {
                for (int j = 0; j < GeoDistXFeat.cols(); j++) {
                    GeoDistXFeat(i, j) = std::min(GeoDistXFeat(i, j), threshX);
                    GeoDistXFeat(i, j) /= threshX;
                }
            }
            for (int i = 0; i < VY.rows(); i++) {
                for (int j = 0; j < GeoDistYFeat.cols(); j++) {
                    GeoDistYFeat(i, j) = std::min(GeoDistYFeat(i, j), threshY);
                    GeoDistYFeat(i, j) /= threshY;
                }
            }

            newCost = 0.1 * perVertexFeatureDifference.cast<float>();

            for (int i = 0; i < VX.rows(); i++) {
                for (int j = 0; j < VY.rows(); j++) {
                    float cost = (GeoDistXFeat.row(i) - GeoDistYFeat.row(j)).cwiseAbs().sum();
                    newCost(i, j) += cost;
                }
            }*/
            //Eigen::MatrixXi t;
            //return std::make_tuple(optimisationTime, p2phr, t, p2phr, t);
            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int i = 0; i < numVertices; i++) {
                // find best sum for each triangle
                int minIndex = 0;
                double bestSum = std::numeric_limits<double>::infinity();
                Eigen::Vector3i adjacentTris = AdjFX.row(i);

                const int iterStart = setInitialLables == 3 ? numDegenerate : 0;
                for (unsigned long l = iterStart; l < numLables; l++) { // this loop is lable of the center triangle
                    double sum = 0;
                    for (int j = 0; j < 3; j++) {
                        sum += newCost(FX(i, j), extraSmooth.LableFY(l, j));
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
                                const double cost = newCost(FX(adjTri, remainingVertexIdx), extraSmooth.LableFY(ll, remainingVertexIdx));
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
            //setMaxIter(0);
        }
        /*









         */
        else if (setInitialLables >= 3) {
            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int i = 0; i < numVertices; i++) {
                // find best sum for each triangle
                int minIndex = 0;
                double bestSum = std::numeric_limits<double>::infinity();
                Eigen::Vector3i adjacentTris = AdjFX.row(i);

                const int iterStart = setInitialLables == 3 ? numDegenerate : 0;
                for (unsigned long l = iterStart; l < numLables; l++) { // this loop is lable of the center triangle
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

       
    }
}

} // namespace smgco
