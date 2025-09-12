//
//  deformationEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include "deformationEnergy.hpp"
#include "wksEnergy.hpp"
#include "helper/utils.hpp"
#include <chrono>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif


DeformationEnergy::DeformationEnergy(Shape& sA, Shape& sB, Combinations& c, float membraneFactor, float bendingFactor, float wksFactor) :
    membraneEnergy(),
    bendingEnergy(),
    shapeA(sA),
    shapeB(sB),
    combos(c),
    membraneFactor(membraneFactor),
    bendingFactor(bendingFactor),
    wksFactor(wksFactor),
    computed(false) {

}


/* function DeformationEnergy::get
   according to (1):
          | memE(ShapeA, ShapeB) + memE(ShapeB, ShapeA); A, B non-degenerate
   memE = | 2 * memE(ShapeB, ShapeA); A degenerate
          | 2 * memE(ShapeA, ShapeB); B degenerate
   and deformationEnergy = memE + lambda * bendE + mu * wksE
 */
Eigen::MatrixXf DeformationEnergy::get(const int numDegenerate) {
    if(computed) {
        return defEnergy;
    }
    const int numNonDegenerate = combos.FaCombo.rows() - numDegenerate;

    // compute the combinations which get stored in FaCombo and FbCombo respectively
    const Eigen::MatrixXi& FaCombo = combos.FaCombo;
    const Eigen::MatrixXi& FbCombo = combos.FbCombo;

    // lambda models material properties
    const float lambda = 5;
    const float mu = 1;
    Eigen::MatrixXf deformationEnergy(numDegenerate + numNonDegenerate, 1);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //#if defined(_OPENMP)
    //#pragma omp parallel
    //#pragma omp sections
    //#endif
    if (membraneFactor > 1e-8){

        // degenerate cases
        //#if defined(_OPENMP)
        //#pragma omp section
        //#endif
        deformationEnergy.block(0, 0, numDegenerate, 1) = membraneFactor *
            2 * membraneEnergy.get(shapeB, shapeA,
                               FbCombo.block(0, 0, numDegenerate, 3),
                               FaCombo.block(0, 0, numDegenerate, 3));

        // non-degenerate part of membrane energy
        //#if defined(_OPENMP)
        //#pragma omp section
        //#endif
        deformationEnergy.block(numDegenerate, 0, numNonDegenerate, 1) = membraneFactor * (
            membraneEnergy.get(shapeA, shapeB,
                               FaCombo.block(numDegenerate, 0, numNonDegenerate, 3),
                               FbCombo.block(numDegenerate, 0, numNonDegenerate, 3)) +
            membraneEnergy.get(shapeB, shapeA,
                               FbCombo.block(numDegenerate, 0, numNonDegenerate, 3),
                               FaCombo.block(numDegenerate, 0, numNonDegenerate, 3)));


    }

    //deformationEnergy = (deformationEnergy.array() * (1.0f/ deformationEnergy.mean()) ).matrix();
    //std::cout << "Mean membrane energy " <<  deformationEnergy.mean() << std::endl;
    // bending energy is easier to handle ;)
    if (bendingFactor > 0) {
        const Eigen::MatrixXf bendE = bendingEnergy.get(shapeA, shapeB, FaCombo, FbCombo);
        deformationEnergy += bendingFactor * (bendE.array() * (1.0f / bendE.mean())).matrix();
    }
    if (wksFactor > 0) {
        WKSEnergy wksEnergy = WKSEnergy();
        const Eigen::MatrixXf wksE = wksEnergy.get(shapeA, shapeB, FaCombo, FbCombo);
        //std::cout << "Mean wks energy " << wksE.mean() << std::endl;
        //std::cout << "Mean bending energy " << bendingEnergy.get(shapeA, shapeB, FaCombo, FbCombo).mean() << std::endl;
        deformationEnergy += wksFactor * (wksE.array() * (1.0f / wksE.mean()) ).matrix();
    }
    
    const float floatEpsilon = 1e-7;
    const float minCoeff = deformationEnergy.minCoeff();
    if (minCoeff < -FLOAT_EPSI) {
        ASSERT_NEVER_REACH;
    }
    if (minCoeff < FLOAT_EPSI ) {
        deformationEnergy = (deformationEnergy.array() + std::abs(deformationEnergy.minCoeff())).matrix();
    }
    
    defEnergy = deformationEnergy;
    computed = true;
    return deformationEnergy;
}

void DeformationEnergy::modifyEnergyVal(const int index, float newVal) {
    assert(index > 0 && index < defEnergy.rows());
    defEnergy(index) = newVal;
}


void DeformationEnergy::useCustomDeformationEnergy(const Eigen::MatrixXf& Vx2VyCostMatrix, bool useAreaWeighting) {
    const bool useTranspose = Vx2VyCostMatrix.rows() != shapeA.getNumVertices();

    const Eigen::MatrixXi& FaCombo = combos.FaCombo;
    const Eigen::MatrixXi& FbCombo = combos.FbCombo;

    Eigen::ArrayXXf energy(FaCombo.rows(), 3);

    if (useAreaWeighting) {
        // Init curvature and area vectors
        Eigen::VectorXf Aa(shapeA.getNumVertices());
        Eigen::VectorXf Ab(shapeB.getNumVertices());
        WKSEnergy wksEnergy = WKSEnergy();
        wksEnergy.getA(shapeA, Aa);
        wksEnergy.getA(shapeB, Ab);

        energy(Eigen::all, 0) = Aa(FaCombo(Eigen::all, 0)) + Ab(FbCombo(Eigen::all, 0));
        energy(Eigen::all, 1) = Aa(FaCombo(Eigen::all, 1)) + Ab(FbCombo(Eigen::all, 1));
        energy(Eigen::all, 2) = Aa(FaCombo(Eigen::all, 2)) + Ab(FbCombo(Eigen::all, 2));
    }
    else {
        energy.setOnes();

    }

    Eigen::ArrayXXf temp(FaCombo.rows(), 3);
    if (useTranspose) {
        for (int i = 0; i < 3; i++) {
            temp(Eigen::all, i) = Vx2VyCostMatrix(FaCombo(Eigen::all, i), FbCombo(Eigen::all, i));
        }
    }
    else {
        for (int i = 0; i < 3; i++) {
            temp(Eigen::all, i) = Vx2VyCostMatrix(FbCombo(Eigen::all, i), FaCombo(Eigen::all, i));
        }
    }

    // update energy
    energy = energy.cwiseProduct(temp.square());
    defEnergy = energy.matrix().rowwise().sum();
}


/* REFERENCES:
 (1) WINDHEUSER, Thomas, et al. Large‐scale integer linear programming for
     orientation preserving 3d shape matching. In: Computer Graphics Forum.
     Oxford, UK: Blackwell Publishing Ltd, 2011. S. 1471-1480.
 */


Eigen::MatrixXd energyWrapper(const Eigen::MatrixXd& VX,
                              const Eigen::MatrixXi& FX,
                              const Eigen::MatrixXd& VY,
                              const Eigen::MatrixXi& FY,
                              const Eigen::MatrixXi& lableSpace,
                              const int numDegenerate,
                              const float membraneFactor,
                              const float bendingFactor,
                              const float wksFactor) {

    Shape shapeX(VX.cast<float>(), FX), shapeY(VY.cast<float>(), FY);


    Eigen::MatrixXd energy(FX.rows(), lableSpace.rows());
    energy.setConstant(0);
    Combinations combos;
    combos.FbCombo = lableSpace;

    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    for (int f = 0; f < FX.rows(); f++) {
        combos.FaCombo = utils::repelem(FX.row(f), lableSpace.rows(), 1);

        DeformationEnergy defEnergy(shapeX, shapeY, combos, membraneFactor, bendingFactor, wksFactor);
        Eigen::VectorXd energyVec = defEnergy.get(numDegenerate).cast<double>();
        energy.row(f) = energyVec;
    }

    return energy;
}
