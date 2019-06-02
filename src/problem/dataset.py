import itertools
from scipy.spatial import KDTree

from src.problem.example import *
from src.objectives import DiscProbDim
from src.model.storage import Storage
from src.problem.objective_utils import max_boundnp


def ix_to_one_hot(indices, size):
    x = np.zeros(size)
    # print(indices, x.shape)
    if len(x.shape) == 1:
        x[indices] = 1
    elif len(x.shape) == 2:
        for i in range(x.shape[0]):
            x[i, indices[i]] = 1
    else:
        raise Exception
    return x


def reverse_box_action(state, next_state, inst=None):
    """
    assuming box actions of type [index, xmin, ymin, xmax, ymax]

    Inputs:
        - state      tensor of size [ C, N, M ]
        - next_state tensor of size [ C, N, M ]
    """
    delta = next_state - state
    index = np.sum(delta, axis=(1, 2)).argmax()     # get the updated layer index
    # get upper and lower bounds to compute box-action
    inds = np.stack(np.nonzero(delta[index]), -1)
    xymin = np.min(inds, axis=0)
    xymax = np.max(inds, axis=0)
    cat = np.concatenate((xymin, xymax))
    return index, cat


def to_disc_disc(state, xs):
    a, y = xs
    # print(a, y, state.shape)
    a = ix_to_one_hot(a, (state.shape[0]))
    y = ix_to_one_hot(y, (4, state.shape[-1]))
    return a, y


def to_disc_cont(state, xs):
    a, cat = xs
    a = ix_to_one_hot(a, (state.shape[0]))
    cat = cat / state.shape[1]
    return a, cat


def to_cont_cont(state, xs):
    a, cat = xs
    # print(cat, state.shape)
    a /= state.shape[0]
    cat = cat / state.shape[1]
    return a, cat


def to_vec(state, xs):
    a, cat = to_cont_cont(state, xs)
    return np.concatenate([np.asarray([a]), cat])


def generate_dataset_kd(dataset_args, state_cls, inst_args, objective_args):
    pass



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
        problem = DictProblem(problem_dict)

        instance = state_cls(problem, problem_dict=problem_dict, **inst_args)
        instance.active_state = term_state

        objective = DiscProbDim(problem, problem_dict, **objective_args)
        target_code = objective.to_input()

        # these steps are optimal backtracks
        perfect_perm = list(range(term_state.shape[0]))
        for seq in itertools.permutations(perfect_perm):
            # variations of state | problem
            # zero out one level at each step and save
            state = term_state.copy()
            for i in seq:
                prev_state = state.copy()
                prev_state[i] = 0
                instance.active_state = prev_state
                r, feats = objective.reward(instance)
                action = reverse_box_action(prev_state, state)
                if dataset_args['post_process']:
                    action = dataset_args['post_process'](state, action)
                else:
                    action = to_disc_cont(state, action)

                datum = {
                    'state': prev_state,    # state_t
                    'action': action,       # action_t
                    'code': target_code.astype(float),      # constant | problem
                    'reward': r.astype(float),              # reward_t+1
                    'next_state': state,                    # state_t+1
                    'real': 1,
                    'optimal': 1,
                    'index': i
                }
                state = prev_state
                yield datum

                count += 1
                if count >= dataset_args.num_options:
                    done = True
                    break


def build_dataset(dataset_args, state_cls, inst_args, objective_args, store=False):
    size = dataset_args.num_problems * dataset_args.num_options
    if store is True:
        storage = Storage(size)
        for datum in generate_dataset(dataset_args, state_cls, inst_args, objective_args):
            storage.add(datum)
        return storage
    else:
        return list(generate_dataset(dataset_args, state_cls, inst_args, objective_args))

