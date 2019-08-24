from .formul_test import TestContinuous, TestDiscrete
from src.cvopt.floorplanexample import *

from src.cvopt.formulate.fp_cont import *
from src.cvopt.formulate.stages import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from example.cvopt.utils import *
from src.cvopt.problem import describe_problem
import src.cvopt.cvx_objects as cxo
from src.cvopt.formulate.cont_base import *
from src.cvopt.formulate import *
from example.cvopt import bathroom as fixt
import sys
import pprint
from src.cvopt.interactive import nodewrapper as wraplib
import inspect


class TestComm(TestContinuous):
    def test_simple(self):
        """ local problem to be tested """
        ba = BoxInputList(3, name='BoxInputList.0')
        obj = PerimeterObjective(ba, obj='Maximize', name='PerimeterObjective.0')
        noover = NoOvelapMIP(ba, name='NoOvelapMIP.0')
        bounds = BoundsXYWH(ba, 10, 10, name='BoundsXYWH.0')
        min_perim = NumericBound(ba.W, low=1, name='NumericBound.0')
        min_perim2 = NumericBound(ba.H, low=1, name='NumericBound.1')
        problem = Stage(ba, forms=[obj, bounds, noover, min_perim, min_perim2])
        problem.solve(verbose=True)
        return problem

    def test_simple0(self):
        """ local problem to be tested """
        ba = BoxInputList(3, name='BoxInputList.0')
        obj = PerimeterObjective(ba, obj='Maximize', name='PerimeterObjective.0')
        noover = NoOvelapMIP(ba, name='NoOvelapMIP.0')
        bounds = BoundsXYWH(ba, 10, 10, name='BoundsXYWH.0')
        problem = Stage(ba, forms=[obj, bounds, noover])
        problem.solve(verbose=True)
        return problem

    def test_serialize(self):
        """ """
        sr = wraplib.Serializer()
        problem = self.test_simple()
        json = sr.serialize(problem)
        pprint.pprint(json)
        assert len(json['active']) == 6
        for f in problem.formulations:
            assert f.name in json['active']
        return problem, json

    def test_deserialize(self):
        """ given a serialized problem, the local version == serialized version"""
        problem1 = self.test_simple()
        sr = wraplib.Serializer()
        json = sr.serialize(problem1)
        problem2 = sr.deserialize(json)
        print('------------')
        for item in problem1.formulations:
            print(item.name, item.is_objective, item.objective())
        print('------------')
        problem2.solve(verbose=True)

    def test_deser_solve(self):
        """ recieve a problem, """
        problem, json = self.test_serialize()

    def test_sanitize(self):
        from src.cvopt.interactive.node_serialize import NodeTemplate
        from src.cvopt.interactive.node_serialize import clean_typing_annos
        spec = inspect.getfullargspec(NoOvelapMIP)
        print(spec.args[1:])
        res1 = clean_typing_annos(spec.annotations['box_list'])
        res2 = clean_typing_annos(spec.annotations['others'])
        res3 = clean_typing_annos(spec.annotations['m'])
        print('------------------')
        assert res1 == ['IOType'], str(res1)
        assert res2 == ['IOType', 'NoneType'], str(res2)
        assert res3 == ['int', 'NoneType'], str(res3)
        for x in [res1, res2, res3]:
            print(x)
        return

