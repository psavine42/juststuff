import src.cvopt.formulate as fr
import inspect
from cvxpy.utilities.performance_utils import compute_once
from typing import List, Set, Dict, Tuple, Optional, Union
from cvxpy.utilities.canonical import Canonical


def trim_name(cls):
    if cls is None:
        return 'na'
    r = cls.__name__.split('.')
    if len(r) == 1:
        return r[0]
    return '.'.join([r[-2], r[-1]])


def clean_typing_annos(anno):
    anolist = []
    print(anno)
    if hasattr(anno, '__args__'):

        if anno.__args__:
            trm = []
            for arg in anno.__args__:
                trm.append(arg)
            anolist.extend(trm)
        else:
            anolist.append(anno)
    else:
        anolist.append(anno)
    return [trim_name(x) for x in anolist]


class NodeTemplate:
    def __init__(self, klass, index):
        self.inputs = []
        self.index = index
        self.outputs = []
        self.supers = []
        self.klass = klass
        self.name = klass.__name__

        # what is it?
        for sub in klass.mro()[1:]:
            name = self.trim_mod(sub)
            self.supers.append(name)
            if name == 'Formulation':
                break
        self.serialize_spec()

    def trim_mod(self, cls):
        r = cls.__name__.split('.')
        if len(r) == 1:
            return r[0]
        return '.'.join([r[-2], r[-1]])

    @property
    def is_iotype(self):
        return fr.IOType in self.klass.mro()

    @compute_once
    def serialize_spec(self):
        """ serialize the specification for the node """
        spec = inspect.getfullargspec(self.klass.__init__)
        args = list(spec.args[1:])  # self is first
        narg = len(args)
        defaults = [] if spec.defaults is None else list(spec.defaults)
        inputs = []
        for i in range(narg):
            arg = args.pop(-1)
            anno = spec.annotations.get(arg, None)
            anolist = clean_typing_annos(anno)

            # if anno == Union:
            item = dict(
                name=arg,
                required=True if len(defaults) == 0 else False,
                anno=anolist,
            )
            inputs.append(item)
            if len(defaults) > 0:
                defaults.pop(-1)

        for i, item in enumerate(reversed(inputs)):
            item['index'] = i
            self.inputs.append(item)

        outputs = []
        if self.klass.META['constraint'] is True:
            outputs.append({'name': 'constraint', 'anno': 'List[Expression]'})

        if self.klass.META['objective'] is True:
            outputs.append({'name': 'objective', 'anno': 'Objective'})

        if self.klass.creates_var is not False:
            outputs.insert(0, {'name': 'value', 'anno': 'value'})
            if self.is_iotype is True:
                outputs.insert(0, {'name': 'self', 'anno': 'IOType'})
            else:
                outputs.insert(0, {'name': 'self', 'anno': 'Variable'})
        else:
            outputs.insert(0, {'name': 'inputs', 'anno': 'Variable'})
        for i, x in enumerate(outputs):
            x['index'] = i

        self.outputs = outputs
        return {
            'name': self.klass.__name__,
            'index': self.index,
            'meta': self.klass.META,
            'istype': self.supers,
            'group': self.klass.__module__.split('.')[-1],
            'module': self.klass.__module__.split('.')[-1],
            'inputs': self.inputs,
            'outputs': self.outputs
        }

    def serialize(self, instance, uid_to_name):
        """
        sample (jsonified) :
            {
                    "name": "obj",
                    "required": false,
                    "anno": "None",
                    "index": 2,
                    "id": "PerimeterObjective.0.input.2",
                    "from_node": "user_input",
                    "from_index": 0,
                    "value": "Maximize"
                }

        """
        obj_inputs = instance.graph_inputs
        input_serial = self.inputs.copy()
        instance_name = uid_to_name[instance.uuid]

        print('serializing', instance.name)
        for i, input_spec in enumerate(self.inputs):
            reqs = input_spec['required']
            input_obj = obj_inputs[i]
            input_serial[i]['id'] = instance_name + '.input.{}'.format(i)

            print(i, input_obj)
            if input_obj is None and reqs is False:
                input_serial[i]['from_node'] = None
                input_serial[i]['from_index'] = None
                input_serial[i]['val'] = None

            elif input_obj is None and reqs is True:
                print('missing input ', i)

            elif input_obj is not None:
                is_forml = isinstance(input_obj, fr.Formulation)
                is_iotype = isinstance(input_obj, fr.PointList)
                is_canon = isinstance(input_obj, Canonical)
                is_type = isinstance(input_obj, type)

                if not is_forml and not is_iotype \
                        and not is_canon and not is_type:
                    # user_value maps to a slider
                    input_serial[i]['from_node'] = 'user_input'
                    input_serial[i]['from_index'] = 0
                    input_serial[i]['val'] = input_obj

                elif is_forml and is_iotype:
                    input_serial[i]['from_node'] = uid_to_name[input_obj.uuid]
                    input_serial[i]['from_index'] = 0
                    input_serial[i]['val'] = None,

                elif is_forml and not is_iotype:
                    print('not iotype')

                elif is_canon is True:
                    input_serial[i]['from_node'] = 'user_input'
                    input_serial[i]['from_index'] = 0
                    input_serial[i]['val'] = str(input_obj.__class__.__name__)

                elif is_type is True:
                    if Canonical in input_obj.mro():
                        input_serial[i]['from_node'] = 'user_input'
                        input_serial[i]['from_index'] = 0
                        input_serial[i]['val'] = input_obj.__name__
                else:
                    print('ELSE ')

        output_serial = self.outputs.copy()
        inst_forml = isinstance(instance, fr.Formulation)
        inst_iotype = isinstance(instance, fr.IOType)
        for i, output in enumerate(self.outputs):
            output_serial[i]['id'] = instance_name + '.output.{}'.format(i)
            if output['name'] == 'value' and inst_iotype is True:
                output_serial[i]['val'] = instance.value


        base = self.serialize_spec()
        base['inputs'] = input_serial
        base['id'] = instance_name
        return base


class NodeInst:
    def __init__(self, node_def, data):
        """

        self._inputs : [{},
                        {}]

        """
        self._spec = node_def
        self._data = data
        self._inputs = data['inputs']
        self._outputs = data['outputs']
        self._needs = 0
        for item in self._inputs:
            if item['required'] is True:
                self._needs += 1

    def initialize(self, inst_dict):
        """"""
        args = [None] * self._needs
        kwargs = {}
        for i, inst_spec in enumerate(self._inputs):
            key = inst_spec['name']
            req = inst_spec['required']
            src = inst_spec['from_node']
            if src == 'user_input':
                if req is True:
                    args[i] = inst_spec['value']
                else:
                    kwargs[key] = inst_spec['value']
                continue

            elif src is None and req is False:
                kwargs[key] = inst_spec['value']
                continue

            if req is True:
                args[i] = inst_dict[src]
            else:
                kwargs[key] = inst_dict[src]

        idx = self.instance_id
        print('\n\nInitializing :: ', self.name, idx, args, kwargs)
        return self._spec.klass(*args, name=self.name, **kwargs)

    def __repr__(self):
        return str(self._data)

    @property
    def name(self):
        return self._data['id']

    @property
    def instance_id(self):
        return self._data['id'].split('.')[-1]

    def can_initialize(self, inst_dict):
        """ if there are no inputs required - True
            else check inst_dict and if all reqs are present - True
            else False
        """
        for input_spec in self._inputs:
            if input_spec['from_node'] in ['user_input', None]:
                continue
            elif input_spec['from_node'] not in inst_dict:
                print(self.name, 'needs', input_spec['from_node'])
                return False
        return True

