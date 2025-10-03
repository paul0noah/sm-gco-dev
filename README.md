# sm-gco-dev


Official repository 


## ⚙️ Installation

`pip install git+https://github.com/paul0noah/sm-gco-dev.git`

## ✨ Usage

```python
import numpy as np
import igl
import gurobipy as gp
from gurobipy import GRB
import time
import scipy.sparse as sp
from sm_gco import GCOSM, COST_MODE, TriangleWiseOpts

# init the options
opts = TriangleWiseOpts()

# pairwise cost modes
opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST      # sum of geodesic distance between shared triangles
#opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX # max of geodesic distance between shared triangles
#opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_L2DIST      # sum of l2 distance between shared triangles
#opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_L2DIST_MAX  # max of l2 distance between shared triangles

opts.smooth_scale_before_robust = 1.0 # scale of the pairwise term before applying robust
opts.robust_cost = False # apply robust cost to pairwise terms, i.e. a logarithm
opts.unary_weight = 1.0 # weight of data term
opts.smooth_weight = 10000.0 # weight of pairwise term

# initialise GCO optimisation
# opts.set_initial_lables = 0 # let gco do the init, i.e. no init
# opts.set_initial_lables = 1 # init with minimum cost label
# opts.set_initial_lables = 2 # init with minimum cost label but not a degenerate label
# opts.set_initial_lables = 3 # init with non degenerate min cost triangle neighbours 
#                               (i.e. the three adjacent triangle to the current tri)
opts.set_initial_lables = 4 # init with degenerate min cost triangle neighbours

# label space settings
opts.lable_space_cycle_size = 4 # max cycle size to extract new triangles from
opts.lable_space_angle_thres = np.pi / 3 # angle to extract orientation preserving triangles from
opts.lable_space_degenerate = True # allow matching of triangles to edges/vertices

# label order settings
# opts.label_order = <any other value> # means no order
# opts.label_order = 1 # random, 2  last, 3 mincost, 4 alternating min cost
# opts.label_order = 2 # degenerate labels come last
# opts.label_order = 3 # mincost labels come first
opts.label_order = 4 # mincost labels come first but are alternating between triangles
#                      i.e. [bestTri0, bestTri1, ..., secondbestTri0, secondbestTri1, ...] 

# use additional energies (if weight is set greater then zero)
# see this and that paper for mem
opts.membrane_energy_weight = 0.0
opts.bending_energy_weight = 0.0
opts.feature_weight = 1.0

# glue the solution to write output
opts.glue_solution = True

# numiterations
maxiter = -1 # run until converged

## TODO: load your own shapes (numpy arrays) with per-vertex features
vx, fx, feat_x = ..., ...
vy, fy, feat_y = ..., ...

## settings
resolve_coupling = True
time_limit = 60 * 60
max_depth = 2
lp_relax = True

## extract data
feature_difference = np.zeros((len(vx), len(vy)))
for i in range(0, len(vx)):
    feature_difference[i, :] = np.linalg.norm(feat_y - feat_x[i, :], axis=1)
fx, _ = igl.orient_outward(vx, fx, np.ones_like(fx)[:, 0])


smgco = GCOSM(vy, fy, vx, fx, feat_diff.T)
smgco.set_max_iter(maxiter)
optime, point_map, tri_tri_matching, raw_matching, raw_tri_tri_matching = smgco.triangle_wise(opts)
print(f"Took {optime} s")

point_map = np.unique(product_space[result_vec.astype('bool'), :-1][:, [0, 2]], axis=0)
```

# 🎓 Attribution
When using this code in your own projects please cite the following
TODO: add veklsers papers

```bibtex
@inproceedings{roetzer2026,
    author    = {Paul Roetzer and many others},
    title     = {SM_GCO},
    booktitle = {???},
    year      = 2025
}
```

