from src.model.arguments import Arguments
import datetime


def base_args(title, viz=None):

    args = Arguments()
    ds = datetime.datetime.now().strftime(" %B-%d-%Y-%I:%M%p")
    args.title = title + ds
    args.viz = viz

    # training regime arguments
    args.train = Arguments()
    args.train.lr = 0.0005
    args.train.log_every = 50
    args.train.episodes = 100000
    args.train.testing = False
    # args.train.env_name = 'main'
    args.train.detail_every = 500
    args.train.steps = 10
    args.train.use_gae = True

    # layout instance arguments
    args.inst = Arguments()
    # args.inst.num_spaces = 3
    args.inst.eps = 0.1

    # scenario enviorment
    args.env = Arguments()
    args.env.random_init = True
    args.env.random_objective = False
    args.env.incomplete_reward = -1

    # loss params
    args.loss = Arguments()
    args.loss.gamma = 0.99
    args.loss.entropy_coef = 0.01
    args.loss.gae_lambda = 0.95
    args.loss.value_loss_coef = 0.5
    args.loss.max_grad_norm = 0.5
    args.loss.aux_loss_coef = 0.5

    # model initialization
    args.nn = Arguments()
    args.nn.out_dim = 256
    return args


def super_args(title, viz=None):

    args = Arguments()
    ds = datetime.datetime.now().strftime(" %B-%d-%Y-%I:%M%p")
    args.title = ds + '--' + title
    args.viz = viz

    # training regime arguments
    args.train = Arguments()
    args.train.lr = 5e-4
    args.train.log_every = 50
    args.train.episodes = int(1e3)
    args.train.testing = False
    # args.train.env_name = 'main'
    args.train.detail_every = 500
    args.train.steps = 10
    args.train.use_gae = True

    # layout instance arguments
    args.inst = Arguments()
    args.inst.num_spaces = 3
    args.inst.size = [20, 20]
    args.inst.eps = 0.1
    args.inst.depth = 6

    # scenario enviorment
    args.env = Arguments()
    args.env.random_init = True
    args.env.random_objective = False
    args.env.incomplete_reward = -1

    # loss params
    args.loss = Arguments()
    args.loss.lambda_action = 1.
    args.loss.lambda_reward = 1.
    args.loss.lambda_recons = 1.

    args.obj = Arguments()
    args.obj.use_comps = False

    # model initialization
    args.nn = Arguments()
    args.nn.zdim = 64
    args.nn.out_dim = 256

    args.ds = Arguments()
    args.ds.num_problems = 300
    args.ds.num_options = 12

    # Gradient Based Planner args
    #
    args.gbp = Arguments()
    args.gbp.sigma = 0.25

    args.gbp.num_steps = 3
    args.gbp.num_grad_steps = 10
    args.gbp.num_rollouts = 10
    args.gbp.mu = 0.5
    args.gbp.action_steps = 2
    args.gbp.policy_steps = 10
    args.gbp.dist_size = 5      # same as action size if continuous box action, else S + 4
    return args



