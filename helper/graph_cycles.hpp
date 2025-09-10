//
//  graph_cycles.hpp
//  helper
//
//  Created by Paul Rötzer on 09.09.25.
//

#ifndef graph_cycless_hpp
#define graph_cycless_hpp

#include <Eigen/Dense>

namespace utils {

Eigen::MatrixXi getCycleTriangles(const Eigen::MatrixXd& VX,
                                  const Eigen::MatrixXi& FX,
                                  const int maxCycleLength=4,
                                  const float angleThreshold=M_PI / 2.0);

Eigen::MatrixXi getDegenerateTriangles(const Eigen::MatrixXi& FX);


}

namespace std {
    template<> struct hash<Eigen::Vector3i> {
        std::size_t operator()(Eigen::Vector3i const& vec) const noexcept;
    };
    template<> struct equal_to<Eigen::Vector3i>{
        constexpr bool operator()(Eigen::Vector3i& lhs, Eigen::Vector3i& rhs) const {
            return lhs(0) == rhs(0) && lhs(1) == rhs(1) && lhs(2) == rhs(2);
        }
    };
}

#endif /* graph_cycless_hpp */
