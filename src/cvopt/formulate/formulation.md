

All problems are subject to obvious constraints, aspect, min/max size, adjacency etc 

Continuous Domain 

    - Routing - path from start to N locations 
    - floorplanning - N programmatic elements X repeated once | X is rectagular 
    - facility layout - N programmatic elements X repeated once | X is convex_set
        - Maximize Distances X_ij -> todo
        - Minimize Distances X_ij -> YES
    - Hybrids 
    
 Discrete Domain 
    
    - Routing - path from start to N locations 
    - floorplanning 
        - Tiling - N programmatic elements repeated M times 
        - Covering - N programmatic elements repeated once  
    - Hybrids 

 Hybrid Domain
    some problems are solved by first getting an approximation in descrete space,
    then transformed into optimizable continuous formulations and solved there.



Multi-stage Problems
    
    - some of this is hidden in the interface, but problem stages are available. 

