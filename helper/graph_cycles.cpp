//
//  graph_cycles.cpp
//  helper
//
//  Created by Paul Rötzer on 09.09.25.
//

#include "graph_cycles.hpp"


#include "helper/utils.hpp"
#include "src/gco_shape_matching.hpp"
#include <igl/edges.h>
#include <igl/adjacency_list.h>
#include <igl/per_vertex_normals.h>
#include <set>
#include <unordered_set>
#include <stack>
#include <igl/resolve_duplicated_faces.h>
#include <tsl/robin_set.h>
#include <igl/barycenter.h>
#include <igl/signed_distance.h>
#include <igl/per_face_normals.h>
#include <igl/avg_edge_length.h>
#include <igl/slice_mask.h>

namespace utils {


// C++ program to find all the
// cycles in an undirected graph

/*





 */


class BoundedCycles {
public:
    BoundedCycles(int n, int k) : n(n), k(k) {
        adj.resize(n);
        lock.assign(n, INT_MAX);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    std::vector<std::vector<int>> findCycles() {
        std::set<std::vector<int>> unique_cycles; // store canonical cycles
        for (int s = 0; s < n; s++) {
            std::unordered_set<int> reachable = bfsSubgraph(s, k - 1);
            if (!reachable.count(s)) continue;

            lock.assign(n, INT_MAX);
            Blist.clear();
            stack.clear();

            cycleSearch(s, s, 0, reachable, unique_cycles);
        }

        // Convert set to vector for output
        std::vector<std::vector<int>> result(unique_cycles.begin(), unique_cycles.end());
        return result;
    }

private:
    int n, k;
    std::vector<std::vector<int>> adj;
    std::vector<int> stack;
    std::vector<int> lock;
    std::unordered_map<int, std::unordered_set<int>> Blist;

    void cycleSearch(int v, int start, int depth,
                     const std::unordered_set<int>& reachable,
                     std::set<std::vector<int>>& unique_cycles) {
        int blen = INT_MAX;
        lock[v] = depth;
        stack.push_back(v);

        for (int w : adj[v]) {
            if (!reachable.count(w)) continue;

            if (w == start) {
                // Found a cycle
                std::vector<int> cycle(stack);
                cycle.push_back(start);
                std::vector<int> canon = canonicalize(cycle);
                unique_cycles.insert(canon);
                blen = 1;
            }
            else if (depth + 1 < lock[w] && depth + 1 < k) {
                cycleSearch(w, start, depth + 1, reachable, unique_cycles);
                blen = 1;
            }
        }

        if (blen < INT_MAX) {
            relaxLocks(v);
        }
        else {
            for (int w : adj[v]) {
                if (!reachable.count(w)) continue;
                Blist[w].insert(v);
            }
        }

        stack.pop_back();
    }

    std::unordered_set<int> bfsSubgraph(int source, int maxDepth) {
        std::unordered_set<int> visited;
        std::queue<std::pair<int,int>> q;
        visited.insert(source);
        q.push({source, 0});
        while (!q.empty()) {
            auto [v, d] = q.front();
            q.pop();
            if (d < maxDepth) {
                for (int w : adj[v]) {
                    if (!visited.count(w)) {
                        visited.insert(w);
                        q.push({w, d+1});
                    }
                }
            }
        }
        return visited;
    }

    void relaxLocks(int v) {
        std::stack<int> st;
        st.push(v);
        while (!st.empty()) {
            int u = st.top();
            st.pop();
            lock[u] = INT_MAX;
            for (int w : Blist[u]) {
                if (lock[w] != INT_MAX) {
                    st.push(w);
                }
            }
            Blist[u].clear();
        }
    }

    std::vector<int> canonicalize(const std::vector<int>& cycle) {
        // remove last element (duplicate of start) for canonical form
        std::vector<int> tmp(cycle.begin(), cycle.end() - 1);
        int m = tmp.size();

        // find smallest element index
        int min_index = std::min_element(tmp.begin(), tmp.end()) - tmp.begin();

        // rotate so smallest element comes first
        std::vector<int> rotated(m);
        for (int i = 0; i < m; i++)
            rotated[i] = tmp[(min_index + i) % m];

        // also check reversed rotation for lexicographic minimality
        std::vector<int> reversed(m);
        for (int i = 0; i < m; i++)
            reversed[i] = tmp[(min_index - i + m) % m];

        if (reversed < rotated) rotated = reversed;

        // close the cycle
        // rotated.push_back(rotated[0]);
        return rotated;
    }
};



std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> getOrientedEdgeEdgeAdjacency(const Eigen::MatrixXd& Vertices,
                                                                          const Eigen::MatrixXi& Triangles,
                                                                          const float angleThreshold) {


    const int numEdges = 3 * Triangles.rows();
    Eigen::MatrixXi Edges(numEdges, 2);
    Eigen::MatrixXi Edge2NextTriEdge(numEdges, 1);
    int edgeIdx = 0;
    for (int f = 0; f < Triangles.rows(); f++) {
        Edge2NextTriEdge(edgeIdx) = edgeIdx + 1;
        Edges.row(edgeIdx++) << Triangles(f, 0), Triangles(f, 1);
        Edge2NextTriEdge(edgeIdx) = edgeIdx + 1;
        Edges.row(edgeIdx++) << Triangles(f, 1), Triangles(f, 2);
        Edge2NextTriEdge(edgeIdx) = edgeIdx - 2;
        Edges.row(edgeIdx++) << Triangles(f, 2), Triangles(f, 0);
    }

    Eigen::MatrixXd Normals;
    igl::per_vertex_normals(Vertices, Triangles, Normals);



    std::vector<std::vector<int>> vertexEdgeAdjacency;
    vertexEdgeAdjacency.reserve(Vertices.rows());
    std::vector<int> temp; temp.reserve(10);
    for (int i = 0; i < Vertices.rows(); i++) {
        vertexEdgeAdjacency.push_back(temp);
    }

    for (int e = 0; e < Edges.rows(); e++) {
        const int srcVertex = Edges(e, 0);
        vertexEdgeAdjacency[srcVertex].push_back(e);
    }



    std::vector<std::vector<int>> edgeEdgeAdjacency;
    edgeEdgeAdjacency.reserve(Edges.rows());
    for (int e = 0; e < Edges.rows(); e++) {
        edgeEdgeAdjacency.push_back(temp);
    }


    int numEdgeEdges = 0;
    for (int e = 0; e < Edges.rows(); e++) {
        const Eigen::Vector3d currentEdge3d = (Vertices.row((Edges(e, 1))) - Vertices.row((Edges(e, 0)))).normalized();
        const int sourceVertex = Edges(e, 0);
        const int targetVertex = Edges(e, 1);
        const Eigen::Vector3d normal3d = Normals.row(targetVertex);

        for (const auto& nextEdge : vertexEdgeAdjacency[targetVertex]) {
            if (Edges(nextEdge, 1) == sourceVertex) {
                continue;
            }

            const Eigen::Vector3d nextEdge3d = (Vertices.row((Edges(nextEdge, 1))) - Vertices.row((Edges(nextEdge, 0)))).normalized();

            const Eigen::Vector3d crossProduct = (currentEdge3d.cross(nextEdge3d)).normalized();

            const double angle = std::acos(crossProduct.dot(normal3d));

            const bool nextEdgeInTriangle = Edge2NextTriEdge(e) == nextEdge;
            //if (nextEdgeInTriangle)
            //    std::cout <<  angle << std::endl;
            if (angle < angleThreshold || nextEdgeInTriangle) {
                edgeEdgeAdjacency[e].push_back(nextEdge);
                numEdgeEdges++;
            }
        }

    }

    /*
    int e = 0;
    for (const auto& nextEdges : edgeEdgeAdjacency) {
        std::cout << e << " : ";
        for (const auto& edge : nextEdges) {
            std::cout << " " << edge << ", ";
        }
        std::cout << std::endl;
        e++;
    }*/

    Eigen::MatrixXi edgeEdgeAdjacencyEdges(numEdgeEdges, 2);
    int numInsterted = 0;
    for (int e = 0; e < numEdges; e++) {
        for (const auto& nextEdge : edgeEdgeAdjacency.at(e)) {
            edgeEdgeAdjacencyEdges.row(numInsterted) << e, nextEdge;
            numInsterted++;
        }
    }

    return std::make_tuple(Edges, edgeEdgeAdjacencyEdges);
}



int getNumTris(const int maxCycleSize) {
    // cycleLength==3 => 1 tri per cycle
    // cycleLength==4 => 3 tri per cycle
    // cycleLength==5 => 10 tri per cycle
    // cycleLength==6 => 20 tri per cycle
    int index = 0;
    for (int id0 = 0; id0 < maxCycleSize - 2; id0++) {
        for (int id1 = id0+1; id1 < maxCycleSize - 1; id1++) {
            for (int id2 = id1+1; id2 < maxCycleSize; id2++) {
                index++;
            }
        }
    }
    return index;
}



Eigen::MatrixXi getCycleTriangles(const Eigen::MatrixXd& VX,
                                  const Eigen::MatrixXi& FX,
                                  const int maxCycleLength,
                                  const float angleThreshold) {


    const auto out = getOrientedEdgeEdgeAdjacency(VX, FX, angleThreshold);
    const Eigen::MatrixXi Edges = std::get<0>(out);
    const Eigen::MatrixXi edgeEdgeAdjacency = std::get<1>(out);

    BoundedCycles bbc(Edges.rows(), maxCycleLength);
    for (int i = 0; i < edgeEdgeAdjacency.rows(); i++) {
        bbc.addEdge(edgeEdgeAdjacency(i, 0), edgeEdgeAdjacency(i, 1));
    }
    auto cycles = bbc.findCycles();

    const unsigned long numCycles = cycles.size();
    const unsigned long maxNumTris = getNumTris(maxCycleLength);
    Eigen::MatrixXi cycleTris(numCycles * maxNumTris, 3);

    unsigned long index = 0;
    tsl::robin_set<Eigen::Vector3i> uniqueCycleTris;
    Eigen::Vector3i cycleTri; cycleTri.setConstant(-1);
    uniqueCycleTris.insert(cycleTri);
    for (const auto& cycle : cycles) {
        const int cycleSize = cycle.size();
        if (cycleSize < 3) {
            std::cout << "cycle size too small, this should not happen. skipping this cycle" << std::endl;
            continue;
        }

        if (cycleSize == 3) {
            const int v0 = Edges(cycle[0], 0);
            const int v1 = Edges(cycle[1], 0);
            const int v2 = Edges(cycle[2], 0);
            // rotate triangle such that smallest index comes first
            if (v0 < v1 && v0 < v2) {
                cycleTri << v0, v1, v2;
            }
            else if (v1 < v0 && v1 < v2) {
                cycleTri << v1, v2, v0;
            }
            else {
                cycleTri << v2, v0, v1;
            }
            if (uniqueCycleTris.count(cycleTri) < 1) {
                uniqueCycleTris.insert(cycleTri);
                cycleTris.row(index) << cycleTri(0), cycleTri(1), cycleTri(2);
                index++;
            }
        }
        else {
            // cycle of length 5 is {v0 v1 v2 v3, v4}
            // so triangles are
            // {v0 v1 v2}, {v0 v1 v3}, {v0 v1 v4}, {v0 v2 v3}, {v0 v2 v4}, {v0 v3 v4},
            // {v1 v2 v3}, {v1 v2 v4}, {v1 v3 v4},
            // {v2 v3 v4}
            for (int id0 = 0; id0 < cycleSize - 2; id0++) {
                const int v0 = Edges(cycle[id0], 0);
                for (int id1 = id0+1; id1 < cycleSize - 1; id1++) {
                    const int v1 = Edges(cycle[id1], 0);
                    for (int id2 = id1+1; id2 < cycleSize; id2++) {
                        const int v2 = Edges(cycle[id2], 0);

                        // rotate triangle such that smallest index comes first
                        if (v0 < v1 && v0 < v2) {
                            cycleTri << v0, v1, v2;
                        }
                        else if (v1 < v0 && v1 < v2) {
                            cycleTri << v1, v2, v0;
                        }
                        else {
                            cycleTri << v2, v0, v1;
                        }

                        if (uniqueCycleTris.count(cycleTri) < 1) {
                            uniqueCycleTris.insert(cycleTri);
                            cycleTris.row(index) << cycleTri(0), cycleTri(1), cycleTri(2);
                            index++;
                        }
                    }
                }
            }

        }

    }
    cycleTris.conservativeResize(index, 3);

    Eigen::MatrixXd barycenters, barycentersPlusNormal, NX, C,  dist, distPlusNormal;
    Eigen::MatrixXi I;
    igl::barycenter(VX, cycleTris, barycenters);
    igl::per_face_normals(VX, cycleTris, NX);
    const double avgEdgeLength = igl::avg_edge_length(VX, FX);
    barycentersPlusNormal = barycenters + 0.001 * NX;

    igl::signed_distance(barycenters, VX, FX, igl::SIGNED_DISTANCE_TYPE_DEFAULT, dist, I, C, NX);
    igl::signed_distance(barycentersPlusNormal, VX, FX, igl::SIGNED_DISTANCE_TYPE_DEFAULT, distPlusNormal, I, C, NX);


    Eigen::MatrixX<bool> doNotRemove(cycleTris.rows(), 1);
    doNotRemove.setConstant(true);
    for (int i = 0; i < cycleTris.rows(); i++) {
        if (dist(i) > distPlusNormal(i)) {
            // flip the normal of tri
            cycleTri << cycleTris(i, 0), cycleTris(i, 2), cycleTris(i, 1);
            if (uniqueCycleTris.count(cycleTri) >= 1) {
                doNotRemove(i) = false;
            }
            else {
                uniqueCycleTris.insert(cycleTri);
                cycleTris.row(i) = cycleTri;
            }
        }
    }

    return igl::slice_mask(cycleTris, doNotRemove, 1); // lol matlab dimension here
}



Eigen::MatrixXi getDegenerateTriangles(const Eigen::MatrixXi& F) {
    const int numVerts = F.maxCoeff() + 1;
    Eigen::MatrixXi E;
    igl::edges(F, E);
    const int numEdges = E.rows();

    // edge [v0 v1] is in tris [v0 v0 v1], [v0 v1 v0], [v1 v0 v0], [v1 v1 v0], [v0 v1 v1], [v1 v0 v1]
    const int numDegenerateTriangles = numVerts + 6 * numEdges;

    Eigen::MatrixXi degenerateTris(numDegenerateTriangles, 3);
    unsigned long numTrisAdded = 0;

    for (int i = 0; i < numVerts; i++) {
        degenerateTris.row(numTrisAdded) << i, i, i;
        numTrisAdded++;
    }

    for (int e = 0; e < E.rows(); e++) {
        const int v0 = E(e, 0);
        const int v1 = E(e, 1);

        degenerateTris.row(numTrisAdded) << v0, v0, v1;
        numTrisAdded++;

        degenerateTris.row(numTrisAdded) << v0, v1, v0;
        numTrisAdded++;

        degenerateTris.row(numTrisAdded) << v1, v0, v0;
        numTrisAdded++;

        degenerateTris.row(numTrisAdded) << v0, v1, v1;
        numTrisAdded++;

        degenerateTris.row(numTrisAdded) << v1, v0, v1;
        numTrisAdded++;

        degenerateTris.row(numTrisAdded) << v1, v1, v0;
        numTrisAdded++;
    }

    assert(numTrisAdded == numDegenerateTriangles);

    return degenerateTris;
}


}


namespace std {
    std::size_t hash<Eigen::Vector3i>::operator()(Eigen::Vector3i const& vec) const noexcept {
        //size_t idx0hash = std::hash<int>()(edg.idx0);
        //size_t idx1hash = std::hash<int>()(edg.idx1) << 1;
        //return idx0hash ^ idx1hash;
        const int k0 = vec(0);
        const int k1 = vec(1);
        const int k2 = vec(2);
        // size_t maxval is 4,294,967,295
        //                    100 000 000
        return k0 + 1000 * k1 + 1000000 * k2;
    }
}
