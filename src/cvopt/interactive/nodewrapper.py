import cvxpy as cvx
import cvxpy.atoms
import networkx as nx
import src.cvopt.formulate as fr
from src.cvopt.formulate.stages import Stage
from collections import defaultdict as ddict
from .node_serialize import NodeInst, NodeTemplate


class Serializer(object):
    def __init__(self):
        self._templates = {}
        cnt = 0
        for k in dir(fr):
            cls = getattr(fr, k, None)
            if isinstance(cls, type) and issubclass(cls, fr.Formulation):
                templ = NodeTemplate(cls, cnt)
                self._templates[templ.name] = templ
                cnt += 1
        self.len = cnt - 1

    def serialize_spec(self):
        ret = [x.serialize_spec() for x in self._templates.values()]
        ret.sort(key=lambda x: (x['module'], x['name']))
        for i, item in enumerate(ret):
            item['index'] = i
        return ret

    def serialize(self, problem):
        node_to_link = {}
        link_to_node = {}
        active = []
        instances = []
        uid_to_name = {}
        cnt = ddict(int)
        for f in problem.serializable:
            name = f.__class__.__name__
            if name in cnt:
                cnt[name] += 1
            else:
                cnt[name] = 0
            inst_name = name + '.' + str(cnt[name])
            uid_to_name[f.uuid] = inst_name
            active.append(inst_name)

        for f in problem.serializable:
            # gather inputs
            templ = self._templates[f.__class__.__name__]
            item_dict = templ.serialize(f, uid_to_name)
            instances.append(item_dict)

        solution = None
        if problem._problem is not None:
            if problem._problem.solution is not None:
                solution = str(problem._problem.solution)

        #
        json_dict = {
            'solution': solution,  # todo
            'image': None,
            'active': instances,
            'nodeToLink': node_to_link,
            'linkToNode': link_to_node
        }
        return json_dict

    def deserialize(self, data):
        """
        'BoxInputList.0.output.0 -
        """
        nodes = data['active']
        n_nodes = len(nodes)
        inst_dict = {}
        fact_q = []

        for n in nodes:
            template = self._templates[n['name']]
            inst_constructor = NodeInst(template, n)
            if inst_constructor.can_initialize(inst_dict) is True:
                concrete_class = inst_constructor.initialize(inst_dict)
                inst_dict[concrete_class.name] = concrete_class
            else:
                fact_q.append(inst_constructor)
        cnt = 0
        while fact_q:
            cnt += 1
            builder = fact_q.pop(0)
            if builder.can_initialize(inst_dict) is True:
                concrete_class = builder.initialize(inst_dict)
                inst_dict[concrete_class.name] = concrete_class
            else:
                fact_q.append(builder)

            if cnt > n_nodes ** 2:
                print()
                return

        items = list(inst_dict.values())
        problem = Stage([], forms=items)
        return problem

