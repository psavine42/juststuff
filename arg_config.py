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
    args.train.log_every = 20
    args.train.episodes = int(1e5)
    args.train.testing = False
    # args.train.env_name = 'main'
    args.train.detail_every = 500
    args.train.steps = 10
    args.train.use_gae = True

    # layout instance arguments
    args.inst = Arguments()
    args.inst.num_spaces = 3
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
    args.nn.zdim = 64
    args.nn.out_dim = 256

    args.ds = Arguments()
    args.ds.num_problems = 400
    args.ds.num_options = 10
    return args


