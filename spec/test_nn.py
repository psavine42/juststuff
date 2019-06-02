import unittest
from src.regimes import *
from src.model.arguments import *
from arg_config import *
from src.algo.recursives import *
import  src.problem.dataset as ds
from src.layout import *
from src.problem.example import *
from torch import autograd
import src.algo.draw as draw
from src.geom.torchgeom import *


class TestGrad(unittest.TestCase):
    def test_g1(self):
        with autograd.detect_anomaly():
            inp = torch.rand(5, 5, requires_grad=True)
            x = inp[1:3, 1:3]
            x = x ** 2
            d = x.sum()
            print(d.grad)
            d.backward(retain_graph=True)
            print(d.grad)
            print(inp.grad)

    def test_gs_grad(self):
        """
        given a grid,
        """
        ga = 16
        sz = int(ga ** 0.5)
        tgt_areas = torch.tensor([[0.5, 0.3, 0.2]])
        # action
        # softmax vecs
        # torch.normal(0.5, 0.2)
        ack = torch.zeros(1, 3, sz, sz).float()
        ack[0, 0, 1:3, 1:3] = 1.
        ack.requires_grad_()

        grid = torch.zeros(1, 3, sz, sz).float()
        # grid[0, 0, ]
        g = grid + ack

        # psuedo loss
        # indexes are samples
        areas = g.sum(dim=(-2, -1)) / ga
        loss_area = torch.abs(areas - tgt_areas).squeeze()

        loss_area[0].backward(retain_graph=True)
        assert torch.is_tensor(ack.grad)
        print('grad', ack.grad)
        print(areas, loss_area)

    def test_gs_grad2(self):
        """
        given a grid,
        """
        tgt_areas = torch.tensor([[0.5, 0.3, 0.2]])
        # action
        # softmax vecs
        # torch.normal(0.5, 0.2)
        ack = torch.zeros(1, 3, 5, 5).float()
        ack[0, 0, 1:3, 1:3] = 1.
        ack.requires_grad_()

        grid = torch.zeros(1, 3, 5, 5).float()
        g = grid + ack

        # psuedo loss
        # indexes are samples
        areas = g.sum(dim=(-2, -1))
        loss_area = torch.abs(areas - tgt_areas).sum()

        loss_area.backward(retain_graph=True)
        print(ack.grad)

    def test_point_acks(self):
        max_actions = 5
        num_regions = 3

        # create a tensor to store bounds
        bounds_mat = torch.zeros(num_regions, max_actions, 4)

        state = torch.zeros(3, 8, 8)

        action1 = torch.tensor([0., 0., 0.5, 0.5])
        action2 = torch.tensor([0.25, 0.25, 0.75, 0.75])

    def test_draw(self):

        N, D, Z, T = 4, 6, 20, 3
        model = draw.DrawModel(T, D, D, Z, N, Z, Z)
        img = model.generate(1)[-1].view(1, D, D).contiguous()
        print(img)
        nt3 = img > (2 / T)
        nt2 = torch.mul(img < (2 / T), img > (1 / T))
        nt1 = img < (1 / T)
        print(nt1)
        print(nt2)
        print(nt3)
        # grad should go through entries where index == max/min bnd
        # it will be stored in this matrix
        # loss_mat = torch.zeros_like(img)

        bins = torch.cat((nt1, nt2, nt3))
        #
        areas = bins.sum(dim=(-1, -2)).float() / D **2
        dils = dilate(bins)
        adjs = adjacencies(bins, dils)
        bnds = max_bound(bins)
        acvx = approx_convex_eval(bins, bounds=bnds)

        print('dil', dils)
        print('bnds', bnds)
        apct = box_aspect(bnds)

        print('1 areas\n', areas)
        print('2 asp\n', apct)
        print('3 ajd\n', adjs)
        print('4 cvx\n', acvx)

        adj = torch.ones(3, 3).triu()

        #
        constr = torch.tensor([[0.5, 0.3, 0.2],  # area
                               [0.5, 0.5, 0.5],  # aspect
                               [0.9, 0.9, 0.9],  # convexity
                               ])

        _adjs = adj.view(-1).unsqueeze(0)
        _cons = constr.view(-1).unsqueeze(0)

        aux = torch.stack((areas, apct, acvx), 1)
        print(aux)
        # F.mse_loss()

        # dilation loss
        # if elements are not adjacent that should, loss is spread
        # to the indices

        # print(bin.shape)
        # amax = torch.argmax(bin.float(), 0)
        # print(amax)
        # mt = max_bound(bin)
        # print(mt)

    def test_areas(self):
        action1 = torch.tensor([0.,   0.,   0.5,  0.5])
        action2 = torch.tensor([0.25, 0.25, 0.75, 0.75])
        action3 = torch.tensor([0.25, 0.25, 0.9,  0.9])
        action4 = torch.tensor([0.4,  0.6,  1.,   1.])
        boxes = [action1, action2, action3, action4]
        indexes = [0, 0, 1, 2]
        num_actions = len(boxes)

        inters = box_intersection(action1, action2)
        diffd = diff_areas(action2, action3)
        print(diffd)

        merged1 = merge_areas(action1, action2)
        print(merged1)

        box_states = [[] for i in range(3) ]
        running_adj = []
        # compute loss statistics after actions
        # when doing this, use a detached nograd copy of the actions?
        for i in range(num_actions):
            box_action = boxes[i]
            idx_action = indexes[i]
            for j in range(len(box_states)):

                s = 0
                num_prev_actions = len(box_states[j])

                while s < num_prev_actions:

                    prev_box = box_states[j].pop(0)

                    # todo possibly keep track of removed chunks and overlaps
                    if idx_action == j:
                        new_bxs = merge_areas(prev_box, box_action)
                    else:
                        new_bxs = diff_areas(prev_box, box_action)

                    # append the returned
                    if torch.is_tensor(new_bxs):
                        box_states[j].append(new_bxs)

                    elif isinstance(new_bxs, (list, tuple)):
                        for nbx in new_bxs:
                            box_states[j].append(nbx)

                    elif new_bxs is None:
                        # if no effect, append the original box
                        box_states[j].append(prev_box)

                    s += 1

            # append the box to its action sequence for furture
            box_states[idx_action].append(box_action)
        for i,b in enumerate(box_states):
            print(i)
            for s in b:
                print(s)



    def test_g2(self):
        with autograd.detect_anomaly():
            x = torch.zeros(5, 5, requires_grad=True)
            inp = torch.tensor([1, 3, 1, 3], requires_grad=True)
            x[inp[0]:inp[1], inp[2]:inp[3]] = 1
            x = x ** 2
            d = x.sum()
            print(d.grad)
            d.backward(retain_graph=True)
            print(d.grad)
            print(inp.grad)

    def test_max(self):
        x = torch.tensor([0.2, 1.5, 0.3], requires_grad=True)
        x2 = x **2
        print(x2.requires_grad)
        mx = x2.argmax().requires_grad_()
        print(mx.requires_grad)
        mx = mx + 1
        print(mx)
        mx.backward(retain_graph=True)
        print(x.grad)

    def test_g3(self):
        """ test grads through indicies
            target is to push indicies to enclose an area
         """
        # 1 x 1 x 5 x 5 with 0 ... 25
        size = 6
        target = 0.5

        src = torch.arange(int(size ** 2), dtype=torch.float).reshape(1, 1, size, size) # .requires_grad_()
        print(src)
        # 1 x 1 x 2 x 2
        indices = torch.tensor([[-1, -1], [0, 1]],
                               dtype=torch.float).reshape(1, 1, -1, 2).requires_grad_()
        print(indices)
        output = F.grid_sample(src, indices).squeeze()
        print(output)  # tensor([[[[  0.,  7.5]]]])
        # rebuild the boundingbox -> [x1, x2] -> [x1, y1], [x2, y2]
        # or get area directly
        rw_ = output / size
        ys = output / (size ** 2)
        mx = torch.abs(target - xs.max() * ys.max())

        print(xs, ys)
        # mx = ((output.max() - output.min()) / 4) ** 2
        print(mx)
        mx.backward(retain_graph=True)
        print(indices.grad)

    def test_mc(self):
        """ directly hit the memory tensor """
        X = torch.diag(torch.FloatTensor([3, 4, 4])).matmul(torch.randn(4, 4, 3))
        # 1. Initialize Parameter
        manifold_param = nn.Parameter(torch.zeros(3, 4, 4))

        def cost(X, w):
            wTX = torch.matmul(w.transpose(1, 0), X)
            wwTX = torch.matmul(w, wTX)
            return torch.sum((X - wwTX) ** 2)

        # 3. Optimize
        optimizer = torch.optim.Adagrad(params=[manifold_param], lr=1e-2)

        for epoch in range(30):
            cost_step = cost(X, manifold_param)
            print(cost_step)
            cost_step.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestAck(unittest.TestCase):
    def test_ind_box(self):
        a1 = OneHot(3)
        a2 = CoordVec(4)
        action_model = ActionReg([a1, a2])
        assert action_model.numel() == 7
        assert action_model.size() == [[3], [4]]

        batch = 2

        xs = torch.randn(batch, 7)
        res = action_model.split(xs)
        # print(res)
        a, x = res
        assert list(a.size()) == [batch, 3]
        assert list(x.size()) == [batch, 4]

        a2, x2 = action_model.apply(lambda x: x + 1, [a, x])
        assert torch.allclose(a2, a + 1)
        assert torch.allclose(x2, x + 1)
        a3, x3 = action_model.apply([nn.Softmax(dim=-1), torch.sigmoid], [a, x])
        assert torch.allclose(a3.sum(), torch.tensor([batch]), atol=1e3), 'got {}'.format(a3.sum())

    def test_all_ix(self):
        a1 = OneHot(3)
        a2 = OneHot([4, 20])
        action_model = ActionReg([a1, a2])
        assert action_model.numel() == 83
        assert action_model.size() == [[3], [4, 20]]
        b = 4
        xs = torch.randn(b, 83)
        res = action_model.split(xs)
        # print(res)
        assert len(res) == 2, 'got {}'.format(len(res))
        a, x = res
        assert list(a.size()) == [b, 3], 'got {}'.format(a.size())
        assert list(x.size()) == [b, 80], 'got {}'.format(x.size())

    def test_all_ix2(self):
        b = 4
        a1 = OneHot(3)
        a2 = OneHot([2, 20])
        a3 = OneHot([2, 20])
        action_model = ActionReg([a1, a2, a3])
        assert action_model.numel() == 83
        assert action_model.size() == [[3], [2, 20], [2, 20]]

        xs = torch.randn(b, 83)
        res = action_model.split(xs)
        assert len(res) == 3, 'got {}'.format(len(res))
        a, x, y = res
        assert list(a.size()) == [3], 'got {}'.format(x.size())
        assert list(x.size()) == [2, 20], 'got {}'.format(x.size())
        assert list(y.size()) == [2, 20], 'got {}'.format(y.size())

    def test_dd(self):
        am = ActionReg([OneHot(3), OneHot([4, 20])], name='dd')
        assert am.numel() == 83, 'got {}'.format(am.numel())

    def test_all_ix3(self):
        a1 = OneHot(3)
        action_model = ActionReg([a1, OneHot(20), OneHot(20), OneHot(20), OneHot(20)])
        assert action_model.numel() == 83
        assert action_model.size() == [[3], [20], [20], [20], [20]]


class TestNN(unittest.TestCase):
    def _setup_test_trainer(self,
                            pclas=PolicySimple,
                            action_model=None,
                            geomtry_fn=None,
                            path=None):
        args = super_args('na-{}{}')
        args.train.testing = True
        zdim = 64

        args.train.log_every = 1
        args.train.episodes = 2
        args.train.testing = True
        args.train.detail_every = 1
        args.train.steps = 10

        enc = EncodeState4(args.inst.num_spaces, zdim)
        dec = DecodeState4(args.inst.num_spaces, zdim)
        print('action_model', action_model)
        action_size = 5 if action_model is None else action_model.numel()

        policy = pclas(zdim, shape=[3, 20, 20], geomtry_fn=geomtry_fn)
        model = GBP(enc, dec, action_size, zdim, 12, policy=policy)
        if path is not None:
            sd = torch.load(path)
            model.load_state_dict(sd)

        return GBPTrainer(None, model=model, argobj=args)

    def test_debug1(self):
        trainer = self._setup_test_trainer()
        args = trainer.args
        args.train.testing = True

        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)

        trainer.train_vanilla_supervised(dataset, args.train)
        trainer.tr_supervised_episode(dataset, args.train)
        trainer.train_distGBP(dataset, args.train)

    def test_debug2(self):
        """
        TEST ALL TRAINING SEQUENCES

        """
        params = [
            (PolicyAllLogitsAG,             ds.to_disc_disc),
            (PolicySimple,                  ds.to_vec),
            (PolicyAllLogitsIndependent,    ds.to_disc_disc),
            (PolicyDiscContAG,              ds.to_disc_cont),
            (PolicyDiscContGA,              ds.to_disc_cont),
            (PolicyDiscContIndependant,     ds.to_disc_cont),
        ]
        step = 0
        for (p, infn) in params:
            print('\n-------------\n{}\n------------\nstarting{} {} {}'.format(
                step, p.__name__, infn, step)
            )

            args = super_args('na-{}{}')
            args.train.testing = True
            args.train.steps = 5
            args.train.episodes = 2

            args.ds.post_process = infn

            args.gbp.action_steps = 2
            args.gbp.policy_steps = 2
            args.ds.num_problems = 10

            dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)

            am = None
            if infn == ds.to_disc_cont:
                am = ActionReg([OneHot(3), CoordVec(4)], name='dc')
            elif infn == ds.to_cont_cont:
                am = ActionReg([CoordVec(1), CoordVec(4)], name='cc')
            elif infn == ds.to_vec:
                am = ActionReg([CoordVec(5)], name='vec')
            elif infn == ds.to_disc_disc:
                am = ActionReg([OneHot(3), OneHot([4, 20])], name='dd')

            trainer = self._setup_test_trainer(
                pclas=p, action_model=am, geomtry_fn=Noop()
            )
            trainer.action_model = am
            trainer.train_vanilla_supervised(dataset, args.train)
            print('vanilla .')
            trainer.tr_supervised_episode(dataset, args.train)
            print('tr_supervised .')
            trainer.train_distGBP(dataset, args.gbp)
            print('distGBP .')
            step += 1

        print('*******************\nALL TESTS\n*******************')

    def test_super_run(self):
        args = super_args('supervised2')
        zdim = 64
        enc = EncodeState4(args.inst.num_spaces, zdim)
        dec = DecodeState4(args.inst.num_spaces, zdim)
        model = GBP(enc, dec, 5, zdim, 12, shared_size=12)
        trainer = GBPTrainer(None, model=model, argobj=args)
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        print(len(dataset))
        trainer.train_vanilla_supervised(dataset, args.train)

    def test_tr_run(self):
        args = super_args('tr_sup')
        zdim = 64
        enc = EncodeState4(args.inst.num_spaces, zdim)
        dec = DecodeState4(args.inst.num_spaces, zdim)

        p = PolicyDiscContAG(zdim, shape=[3, 20, 20], geomtry_fn=Noop)

        model = GBP(enc, dec, 5, zdim, 12, shared_size=12, policy=p)
        trainer = GBPTrainer(None, model=model, argobj=args)
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        print(len(dataset), args.train.episodes * args.ds.num_problems,
                                           args.ds.num_options)
        trainer.train_tr_supervised(dataset, args.train)

    # SETUP FOR SOME GRIDDAGE
    def run_model(self, policy, op):
        args = super_args('supervised2-{}{}'.format(op.__name__, policy.__name__))
        zdim = 100
        enc = EncodeState4(args.inst.num_spaces, zdim)
        dec = DecodeState4(args.inst.num_spaces, zdim)
        p = policy(zdim, shape=[3, 20, 20], geomtry_fn=op)
        model = GBP(enc, dec, 5, zdim, 12, shared_size=12, policy=p)

        trainer = GBPTrainer(None, model=model, argobj=args)
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        print('\n-------------\n steps {}'.format(args.train.episodes * args.ds.num_problems,
                                           args.ds.num_options))

        trainer.train_vanilla_supervised(dataset, args.train)

    # -----------------------------------------------------------------------
    # Supervised Tests -
    #   1) [ [softmax, sigmoid], [ softmax, No-op ] ]
    #   2) [ real_cords, from_center,  ]
    # -----------------------------------------------------------------------
    # LOGITS :[N, S] , GEOM: [ N, 4 ]
    def test_s11(self): self.run_model(PolicyDiscContIndependant, nn.Sigmoid)
    def test_p12(self): self.run_model(PolicyDiscContGA, nn.Sigmoid)
    def test_p13(self): self.run_model(PolicyDiscContAG, nn.Sigmoid)
    #
    def test_p21(self): self.run_model(PolicyDiscContIndependant, Noop)
    def test_p22(self): self.run_model(PolicyDiscContGA, Noop)
    def test_p23(self): self.run_model(PolicyDiscContAG, Noop)
    # Noop > Sigmoid
    # this makes sense, because sigmoid distorts stuff in the middle
    # of the Noo-ops GA/GA are better by a small bit
    # -----------

    # -----------
    def test_l11(self): self.run_model(PolicyAllLogitsIndependent, Noop)
    # def test_l12(self): self.run_model(PolicyAllLogitsAG, Noop)
    def test_l13(self): self.run_model(PolicyAllLogitsRNN, Noop)

    # GBP FOR REAL
    def test_gbp_run(self):
        trainer = self._setup_test_trainer(PolicyDiscContAG)
        args = trainer.args
        args.gbp.policy_steps = 2
        args.gbp.action_steps = 2
        # args.ds.post_process =

        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        trainer.train_distGBP(dataset, args.gbp)

    # Unit ---------------------------------------------------------
    def test_gbp_grad(self):
        """ """
        trainer = self._setup_test_trainer(PolicyDiscContAG, './data/trained/ May-21-2019-12:23AM--tr_sup.pkl')
        args = trainer.args
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        state = trainer.state_to_observation(dataset[1])
        # print(state[0].size(), state[1].size())
        print(trainer._lr)
        best_states, best_actions = trainer.gbp_step(state, args.gbp)
        print(best_actions)

    def test_gbp_gstep(self):
        """ """
        trainer = self._setup_test_trainer(PolicyDiscContAG)
        args = trainer.args
        model = trainer.model

        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj)
        state = trainer.state_to_observation(dataset[1])

        action_noise = D.Normal(torch.zeros(5), torch.zeros(5).fill_(args.gbp.sigma))
        xs_ = action_noise.sample().to(trainer.device)
        xs = torch.tensor(xs_, requires_grad=True, device=trainer.device)

        action = F.softmax(xs.unsqueeze(0))
        states = model.transition(state, xs.unsqueeze(0))
        reward = model.reward(state, action)
        full = model(state)
        print(full[0])
        print(reward)
        print(xs)
        dx = dfdx(reward.sum(), xs)
        print(dx)

    def test_gbp_sz(self, ):
        sig = 0.25 ** 0.5
        zero = torch.zeros(5).fill_(0.5)
        sigma = torch.zeros(5).fill_(sig)
        an2 = D.Normal(zero, sigma)
        # an3 = D.MultivariateNormal(zero, sigma)
        an4 = D.Normal(0.5, sig)

        action_noise = D.Independent(D.Normal(zero, sigma), 1)
        print(action_noise.mean, action_noise.variance)
        print(an2.mean, an2.variance)
        # print(an3.mean, an3.variance)
        print(an4.mean, an4.variance)
        print(an4.sample(torch.Size([5])))

    def test_dataset(self):
        args = super_args('supervised1')
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.obj, store=True)
        print(len(dataset))
        print(dataset.index[0:4])
        print(dataset.action[0:4])
        # check dimensions
        assert list(dataset.state[0].shape) == [3, 20, 20]
        assert list(dataset.next_state[0].shape) == [3, 20, 20]
        assert list(dataset.action[0].shape) == [5], 'got shape {}'.format(dataset.action[0])
        assert all([x[0] < 3 for x in dataset.action])
        assert all([x[0] == i for x, i in zip(dataset.action, dataset.index)])

        assert np.sum(dataset.state[0]) < np.sum(dataset.next_state[0])  # next is terminal
        assert dataset.reward[0] == 1.
        assert np.allclose(dataset.code[0], dataset.code[1]), \
            'target codes should be same for problem'

        assert dataset.reward[1] < 1., \
            'expected < 1, got {}'.format(dataset.reward[1])
        assert np.allclose(dataset.state[0], dataset.next_state[1])
        assert np.sum(dataset.state[2]) == 0, \
            'expected initial state to be all zeros, got {}'.format(np.sum(dataset.state[2]))

    def test_probgen(self):
        problem_dict, term_state = problem1(return_state=True, x=10, y=10)
        print(term_state)


