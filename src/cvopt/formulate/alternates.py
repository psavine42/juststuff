from src.cvopt.formulate.fp_cont import *
from src.cvopt.formulate.stages import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from example.cvopt.utils import *
from src.cvopt.problem import describe_problem
import src.cvopt.cvx_objects as cxo
from src.cvopt.formulate.cont_base import *
from src.cvopt.formulate import *

""" 
Which Formulations can replace each other when something in the problem changes
"""

_alternates_simple = {
    BoxAspect: [BoxAspectLinear ]

}
