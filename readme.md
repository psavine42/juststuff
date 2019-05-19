
## encoding constraints:

1) Bayesnet
2) 


SOAS

Rinde and Dahl [23]



Algo Types: 
the type of procedural floor plan generation. The types are               Boundaryused, BoundaryRequired
    subdivision,        1               1
    tile placement      1               0
    inside out          0               0
    room growth.        1               0



Misc Techinques:

Seeding.
    given a tiling - where to start each room?
    -constraints -> fn * noise -> seeds
    -Particles moving about (force directed)
    -greedy - largest room first
    - most constrained thing first (by dof)

Operators:
    swap spaces
    slide wall
    break+slide
    reset

Domain:
    meshgrid
    irregular tiling
    continuous bounded


## Constraints

- Overlapping: It describes how space is required or not to be interlocked with another.
- Accessibility: It describes if spaces are connected by means of circulation paths.
- Proximity: it describes the connection between the spaces and distances in between.
- Skin Constraint: It describes if spaces are forced by outline and setback from skin.
- Orientation Constraints: It describes if space should be rotated to face another.
- Prohibit Intersection Constraint
- Build Cost Constraint

## 
Tile-based methods - it should be possible to understand their practical effectiveness by measuring IRL designs to answer - how regular is their wall-grid? 

Wall-grids will probably tend to have fractal-y structure



## Concrete
### related
https://github.com/fogleman/Piet 

### full
** https://nbviewer.jupyter.org/github/MOSEK/Tutorials/blob/master/Optimizer/integer-exact-cover/exactcover.ipynb
https://github.com/cvxgrp/cvxpy/blob/master/examples/floor_packing.py
https://cvxopt.org/examples/book/floorplan.html
https://cvxopt.readthedocs.io/en/latest/solvers.html
https://www.mcs.anl.gov/~itf/dbpp/text/node21.html


https://github.com/subhdas/Spaceplanning_ADSK_PW/ 


## papers
Tree-based:
    Separating Topology and Geometry in Space Planning, B. Medjdoub and B. Yannou

Optimization:
    Optimizing Architectural Layout Design via Mixed Programming, K. Kamol and S. Krung 
    Evolutionary Approach to Generate Space Layout Topologies, J. Damski and J. Gero 

Mixed:
    Global Search + Local Optimization
        Layout Design Optimization, J. Michalek, R. Choudhary and P. Papalambros
        - Simulated Annealing / GA for Global, 
GA: 
    Genetic Algorithms in Supporting Creative Design, E. Tomor and G. Franck
    G.P. + Unfolding Embryology in Automated Layout Planning, A. Doulgerakis
    Assistant tool for Architectural Layout Design by Genetic Algorithm, P. Nilkaew
    
    
Discursive Grammar(?)
    A Discursive Grammar for Customizing mass housing: Case of Siza’s house, J. P. Duarte

Space Adjacency Behavior in Space Planning, Y. Hsu and R. J. Krawczyk


#MiSC

Shear wall layout opotimization

https://naeimdesigntechnologies.wordpress.com/2016/10/21/old-research-generative-algorithms-in-architectural-space-layout-planning/

http://www.pyopt.org/reference/optimizers.fsqp.html


TODO
https://nbviewer.jupyter.org/github/MOSEK/Tutorials/blob/master/Optimizer/integer-exact-cover/exactcover.ipynb