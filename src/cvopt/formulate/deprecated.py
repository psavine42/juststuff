
# deprecated / not needed ----------------------------------------
class _BoundsXYWH(FormulationR2):
    META = {'constraint': True}

    def __init__(self, space, layout, w, h, **kwargs):
        FormulationR2.__init__(self, space, [layout])
        self.w = w
        self.h = h

    def as_constraint(self):
        fp = self._inputs[0]
        C = [fp.X + 0.5 * fp.W <= 0.5 * self.w,   # 0.5 * self.w,
             fp.Y + 0.5 * fp.H <= 0.5 * self.h,   # 0.5 * self.h,

             0.5 * fp.W - fp.X <= 0.5 * self.w,        # 0.5 * self.w,
             0.5 * fp.H - fp.Y <= 0.5 * self.h,        # 0.5 * self.h,
             # fp.H >= 0.2,
             # fp.W >= 0.2
             # fp.X - 0.5 * fp.W >= 0,
             # fp.Y - 0.5 * fp.H >= 0,
             ]
        return C

    def as_objective(self, **kwargs):
        return None


class PlaceLayout(BoxInputList):
    def __init__(self, space, children,
                 adj_mat,
                 rpm,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """ Stage 2 of layout problem
        from
        An Efficient Multiple-stage Mathematical Programming Method for
        Advanced Single and Multi-Floor Facility Layout Problems

            - Fixed Outline (can work for classical as well)
            -

        """
        FPStage2.__init__(self, space, children)
        self._rpm = rpm
        self._adj_mat = adj_mat
        self._lims = lims
        self._aspect = aspect
        num_in = len(self._inputs)

        self.WF = width if width else Variable(nonneg=True, name='Stage2.W')
        self.HF = height if height else Variable(nonneg=True, name='Stage2.H')

        tris = np.triu(np.ones((num_in, num_in)), 1).sum()
        self.U = Variable(shape=tris, nonneg=True)
        self.V = Variable(shape=tris, nonneg=True)

    def as_constraint(self, *args):
        """

        """
        areas = np.asarray([x.area for x in self._inputs])

        # Area constraints -> can be SDP relaxed
        C = [self.X * self.Y == areas ]                 # 28

        # aspect constraints -> can be SDP relaxed
        C += [
            self.W - self._aspect * self.H <= 0,
            self.H - self._aspect * self.W <= 0,  # 29
        ]

        # common within facility half-bounds
        C += [
            self.X + 0.5 * self.W <= 0.5 * self.WF,
            self.Y + 0.5 * self.H <= 0.5 * self.HF,
            0.5 * self.W - self.X <= 0.5 * self.WF,
            0.5 * self.H - self.Y <= 0.5 * self.HF,     # 26, 27
        ]
        if self._lims:
            C += [
                # within bounds - todo factor out
                self._lims[0] <= self.W,
                self._lims[2] >= self.W,
                self._lims[1] <= self.H,
                self._lims[3] >= self.H,  # 30
            ]

        ij = 0
        for i in range(len(self._inputs)):
            for j in range(i+1, len(self._inputs)):
                C += [
                    self.U[ij] >= self.X[i] - self.X[j],    # 22
                    self.U[ij] >= self.X[j] - self.X[i],    # 23
                    self.V[ij] >= self.Y[i] - self.Y[j],    # 24
                    self.V[ij] >= self.Y[j] - self.Y[i],    # 25
                ]
                ij += 1
        return C

    def as_objective(self, **kwargs):
        O = []
        ij = 0
        for i in range(len(self._inputs)):
            for j in range(i+1, len(self._inputs)):
                O += [
                    self._adj_mat[i, j] * (self.U[ij] + self.V[ij]),  # 21
                ]
                ij += 1
        return Minimize(cvx.sum(cvx.hstack(O)))

    @property
    def action(self):
        geom = cvx.hstack([self.X, self.Y, self.W, self.H])
        return geom


class PlaceLayoutSOC(FPStage2):
    def __init__(self, space, children,
                 adj_mat,
                 rpm,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """
        RPM: upper tri trensor [n x n x 2]
            [0, 0] -> no relation

        """
        FPStage2.__init__(self, space, children)
        self.rpm = rpm
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in)), 1).sum()

        self.U = Variable(shape=tris, nonneg=True)
        self.V = Variable(shape=tris, nonneg=True)
        self.B = Variable(shape=self.X.shape[0], nonneg=True)

    def as_constraint(self):
        """
        """
        num_in = self.X.shape[0]
        tri_i, tri_j = np.triu_indices(num_in, 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()

        A_i = np.asarray([x.area for x in self._inputs])
        a2s = 2 * np.sqrt(A_i)
        C = [
            # lineaerized absolute value constraints
            self.U >= self.X[tri_i] - self.X[tri_j],
            self.U >= self.X[tri_j] - self.X[tri_i],
            self.V >= self.Y[tri_i] - self.Y[tri_j],
            self.V >= self.Y[tri_j] - self.Y[tri_i],

            # SOC constraints on B
            cvx.SOC(self.H - self.W + a2s,      self.H + self.W),
            cvx.SOC(A_i - self.B + 2 * self.H,  self.B + A_i),
            cvx.SOC(A_i - self.B + 2 * self.W,  self.B + A_i),
        ]
        # RPM adjacency constraints
        C += RPM(self.space, self, self.rpm).as_constraint()
        return C

    def as_objective(self, **kwargs):
        return Minimize(cvx.sum(self.U + self.V))


class PlaceLayoutGM(BoxInputList):
    def __init__(self, space, children, rpm,
                 adj_mat=None,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """
        RPM: upper tri trensor [n x n x 2]
            [0, 0] -> no relation
        Novel Convex Optimization Approaches
                for VLSI Floorplanning                      2008    SDP
        """
        FPStage2.__init__(self, space, children)
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        self.B = Variable(shape=self.X.shape[0], pos=True)
        self.U = Variable(shape=tris, pos=True)
        self.V = Variable(shape=tris, pos=True)
        self.WF = width
        self.HF = height
        self.rpm = RPM(self.space, self, rpm)

    def as_constraint(self):
        """
        """
        num_in = self.X.shape[0]
        A_i = np.asarray([x.area for x in self._inputs])
        a2s = np.sqrt(A_i)
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        tri_i, tri_j = np.triu_indices(num_in, 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
        # A_i = np.asarray([x.area for x in self._inputs])
        a2s = np.sqrt(A_i)
        C = [
            # lineaerized absolute value constraints
            self.U >= self.X[tri_i] - self.X[tri_j],
            self.U >= self.X[tri_j] - self.X[tri_i],
            self.V >= self.Y[tri_i] - self.Y[tri_j],
            self.V >= self.Y[tri_j] - self.Y[tri_i]
        ]
        # SDP constraints
        # a = w * h
        C += [cvx.geo_mean(cvx.hstack([self.W[i], self.H[i]])) >= a2s[i] for i in range(num_in)]

        # a = w * h
        # C += [cvx.PSD(cvx.bmat([[self.B[i], self.W[i]],
        #                        [self.W[i], A_i[i]]]), constr_id='bw{}'.format(i)) for i in range(num_in)]

        # C += [cvx.PSD(cvx.bmat([[self.B[i], self.H[i]],
        #                         [self.H[i], A_i[i]]]), constr_id='hw{}'.format(i)) for i in range(num_in)]

        # C += [self.B <= 4]
        # RPM adjacency constraints
        C += self.rpm.as_constraint()

        # within bounds
        # C += BoundsXYWH(self.space, self, self.WF, self.HF).as_constraint()
        return C

    def as_objective(self, **kwargs):
        tri_i, tri_j = np.triu_indices(self.X.shape[0], 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
        o1 = Minimize(cvx.sum(self.U + self.V))
        # o2 = Minimize(cvx.sum(cvx.abs(self.X[tri_i] - self.X[tri_j])
        #                      + cvx.abs(self.Y[tri_i] - self.Y[tri_j])
        #                      ))
        return o1



