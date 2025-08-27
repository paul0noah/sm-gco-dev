#include "gco_shape_matching.hpp"
#include <gco/GCoptimization.h>
#include <igl/edges.h>
#include <igl/exact_geodesic.h>

namespace smgco {

typedef struct GCOPointwiseExtra {
    Eigen::MatrixXf p2pDeformation;
} GCOPointwiseExtra;


GCoptimization::EnergyTermType smoothFnGCOSMPointwise(GCoptimization::SiteID s1,
                                                      GCoptimization::SiteID s2,
                                                      GCoptimization::LabelID l1,
                                                      GCoptimization::LabelID l2,
                                                      void* extraDataVoid) {
    GCOPointwiseExtra* extraData = static_cast<GCOPointwiseExtra*>(extraDataVoid);

    const float diff = extraData->p2pDeformation(l1, l2);

    return (int) (SCALING_FACTOR * diff);
}


Eigen::MatrixXi GCOSM::pointWise(const bool smoothGeodesic) {
    const int numVertices = VX.rows();
    const int numLables = VY.rows();

    Eigen::MatrixXi EX;
    igl::edges(FX, EX);

    Eigen::MatrixXi result(numVertices, 1);

    try{
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numVertices, numLables);
        gc->setVerbosity(1);

        // Note this could be optimised
        int* data = new int[numVertices * numLables];
        for ( int i = 0; i < numVertices; i++ ) {
            for (int l = 0; l < numLables; l++ ) {
                data[i * numLables + l] = (int) (SCALING_FACTOR * dataWeight * perVertexFeatureDifference(i, l));
            }
        }

        gc->setDataCost(data);

        Eigen::MatrixXf smoothCost(numLables, numLables);
        if (smoothGeodesic) {
            Eigen::VectorXi VYsource, FS, VYTarget, FT;
            // all vertices are source, and all are targets
            VYsource.resize(1);
            VYTarget.setLinSpaced(VY.rows(), 0, VY.rows());
            Eigen::VectorXf d;
            for (int i = 0; i  < VY.rows(); i++) {
                VYsource(0) = i;
                igl::exact_geodesic(VY, FY, VYsource, FS, VYTarget, FT, d);
                smoothCost.col(i) = d;
            }

        }
        else { // smooth l2
            smoothCost.setZero();
            for (int i = 0; i < VY.rows(); i++) {
                for (int j = 0; j < VY.rows(); j++) {
                    smoothCost(i, j) = (VY.row(i) - VY.row(j)).norm();
                }
            }
        }
        GCOPointwiseExtra extraData;
        extraData.p2pDeformation = smoothCost;


        gc->setSmoothCost(smoothFnGCOSMPointwise, static_cast<void*>(&extraData));

        

        for (int e = 0; e < EX.rows(); e++) {
            const int srcId = EX(e, 0);
            const int targetId = EX(e, 1);
            const int weight = 1;
            gc->setNeighbors(srcId, targetId, weight);
        }

        std::cout << prefix << "Before optimization energy is " << gc->compute_energy() / SCALING_FACTOR << std::endl;
        gc->expansion(numIters);
        std::cout << prefix << "After optimization energy is " << gc->compute_energy() / SCALING_FACTOR << std::endl;


        for ( int  i = 0; i < numVertices; i++ ) {
            result(i) = gc->whatLabel(i);
        }


        delete gc;
        delete[] data;
    }
    catch (GCException e){
        e.Report();
    }

    return result;
}

} // namespace smgco
