import itertools
import torch

from src.problem.example import *
from src.objectives import DiscProbDim
from src.model.storage import Storage
from src.problem.objective_utils import max_boundnp


def reverse_box_action(state, next_state, inst=None):
    """
    assuming box actions of type [index, xmin, ymin, xmax, ymax]

    Inputs:
        - state      tensor of size [ C, N, M ]
        - next_state tensor of size [ C, N, M ]
    """
    delta = next_state - state
    # get the layer index
    index = np.sum(delta, axis=(1, 2)).argmax()
    # get upper and lower bounds to compute box-action
    inds = np.nonzero(delta[index])
    xymin = np.min(inds, axis=0)
    xymax = np.max(inds, axis=0)
    return np.concatenate((xymin, xymax))


def generate_dataset(dataset_args, state_cls, inst_args, objective_args):
    """
    generate specifications
    from each specifications
        generate intermidiate states by heuristics ?

    Dataset consists of Sample (s_t , a_t , r_t , s_t+1 ) ∼ E

    The actions here are absolute optimal - only works if

    """
    size = dataset_args.num_problems * dataset_args.num_options

    done = False
    count = 0

    while count < size and not done:
        # count = 0
        problem_dict, term_state = problem1(return_state=True)
        instance = state_cls(problem_dict=problem_dict, **inst_args)
        instance.active_state = term_state

        objective = DiscProbDim(None, problem_dict, **objective_args)
        target_code = objective.to_input()

        perfect_perm = list(range(term_state.shape[0]))
        for seq in itertools.permutations(perfect_perm):
            # variations of state | problem
            # zero out one level
            state = term_state.copy()
            for i in seq:
                prev_state = state.copy()
                prev_state[i] = 0
                r, feats = objective.reward(prev_state)
                action = reverse_box_action(prev_state, state)
                datum = {
                    'state': torch.from_numpy(prev_state).float(),
                    'action': torch.from_numpy(action).float(),
                    'code': torch.from_numpy(target_code).float(),
                    'reward': torch.tensor([r]).float(),
                    'next_state': torch.from_numpy(state).float(),
                    'real': 1
                }
                state = prev_state
                yield datum

                count += 1
                if count >= dataset_args.num_options:
                    done = True
                    break


def build_dataset(dataset_args, state_cls, inst_args, objective_args):
    size = dataset_args.num_problems * dataset_args.num_options
    storage = Storage(size)
    for datum in generate_dataset(dataset_args, state_cls, inst_args, objective_args):
        storage.add(datum)
    return storage

