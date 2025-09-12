//
//  deformationEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef deformationEnergy_hpp
#define deformationEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"
#include "src/energy/membraneEnergy.hpp"
#include "src/energy/bendingEnergy.hpp"

struct Combinations {
    Eigen::MatrixXi FaCombo;
    Eigen::MatrixXi FbCombo;
};


class DeformationEnergy {
private:
    Shape& shapeA;
    Shape& shapeB;
    
    MembraneEnergy membraneEnergy;
    BendingEnergy bendingEnergy;
    
    Combinations& combos;

    float membraneFactor;
    float bendingFactor;
    float wksFactor;

    Eigen::MatrixXi computeNonDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi computeDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi getTriangle2EdgeMatching(Eigen::MatrixXi &Fx, int numFacesY);
    
    void computeCombinations();
    
    bool computed;
    Eigen::MatrixXf defEnergy;
    
public:
    DeformationEnergy(Shape& sA, Shape& sB, Combinations& c, float membraneFactor, float bendingFactor, float wksFactor);
    Eigen::MatrixXf get(const int numDegenerate);
    void modifyEnergyVal(const int index, float newVal);
    void useCustomDeformationEnergy(const Eigen::MatrixXf& Vx2VyCostMatrix, bool useAreaWeighting);
};


Eigen::MatrixXd energyWrapper(const Eigen::MatrixXd& VX,
                              const Eigen::MatrixXi& FX,
                              const Eigen::MatrixXd& VY,
                              const Eigen::MatrixXi& FY,
                              const Eigen::MatrixXi& lableSpace,
                              const int numDegenerate,
                              const float membraneFactor,
                              const float bendingFactor,
                              const float wksFactor);

#endif /* deformationEnergy_hpp */

