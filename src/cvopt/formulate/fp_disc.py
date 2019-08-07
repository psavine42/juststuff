from .formulations import Formulation
import numpy as np
from cvxpy import Parameter, Constant, Variable, Minimize, Maximize
import cvxpy as cvx
import scipy.spatial


"""
some observations to ignore latter:

a vertex subset S of G is said to be convex if it contains the vertices of 
all shortest paths connecting any pair of vertices in S. 
(convex set on graph definition)


floorplanning discrete - supports either problems with N-partitions
or N-tiles which can be placed on mesh

"""
def max_area_objective(A, X):
    return cvx.Maximize(cvx.sum(A @ X))


class FormulationDisc(Formulation):
    DOMAIN = {'discrete'}

    def __init__(self, space, actions=None, **kwargs):
        Formulation.__init__(self, space, **kwargs)
        if actions is not None:
            if isinstance(actions, list):
                self._actions = actions
            else:
                self._actions = [actions]

    def as_constraint(self, *args):
        return []

    def as_objective(self, **kwargs):
        return None

    def display(self):
        data = Formulation.display(self)
        return data


class TileLimit(FormulationDisc):
    def __init__(self, space, tile, upper=None, lower=None, **kwargs):
        """

        """
        FormulationDisc.__init__(self, space, **kwargs)
        self._tile_index = tile
        self._upper = upper
        self._lower = lower
        self.is_constraint = True

    def as_constraint(self, *args):
        C = []
        if self._upper:
            C += [cvx.sum(self._actions[self._tile_index].X) <= self._upper]
        if self._lower:
            C += [cvx.sum(self._actions[self._tile_index].X) >= self._lower]
        return C


class NoOverlappingFaces(FormulationDisc):
    def as_constraint(self, *args):
        M = np.zeros((len(self.space.faces), self.num_actions), dtype=int)
        cnt = 0
        for p in self._actions:
            for mapping in p.maps:
                # ixs = list(mapping.face_map().values())
                M[mapping.faces, cnt] = 1
                cnt += 1
        return [M @ self.stacked <= 1]


class NoOverlappingActions(FormulationDisc):
    def as_constraint(self, *args):
        return [cvx.sum(cvx.hstack(self.stacked)) <= 1]


class LocallyConnectedSet(FormulationDisc):
    """ todo extend this to N steps """
    def __init__(self, space, limit=2, **kwargs):
        FormulationDisc.__init__(self, space, **kwargs)
        self._c = Parameter(value=limit)

    def as_constraint(self, *args):
        """ for the faces of self.space, each partitioning X must have atleast
            2 adjacent faces of same selection,

            or if X is 0, then
            +---+---+---+
            |   | 0 |   |
            +---+---+---+
            | A | A | 0 |
            +---+---+---+
            | A | A |   |
            +---+---+---+
        self._actions is a Variable of size 'space.num_faces'
        """
        N = len(self.space.faces)
        M = np.zeros((N, N), dtype=int)
        for k, vs in self.space.faces.to_faces.items():
            M[list(vs), k] = 1

        M = Parameter(shape=M.shape, value=M, symmetric=True, name='face_adj_mat')
        # print(M)
        C = []
        for action in self._actions:
            # [ N, N ] @ [N, 1] -> N
            # 2 versions of this tested ->
            # v1 - use inverse directly to meet constraint this is empirically slower
            # this enforces connectivity
            C += [self._c <= M @ action.vars + (self._c + 1) * (1 - action.vars)]

            # ------------------
            # v2 - use a slack variable either X or V
            # v = Variable(shape=N, boolean=True, name='indicator.{}.{}'.format(self.name, action.name))
            # C += [
            #     1 <= action.vars + v,
            #     self._adj_lim <= M @ action.vars + 4 * v,
            #     # self._adj_lim <= M @ (1 - action.vars)
            # ]
        return C


class RadiallyConnectedSet(FormulationDisc):
    META = {'constraint': True, 'objective': False}

    def __init__(self, space, action, r, **kwargs):
        FormulationDisc.__init__(self, space, actions=action, **kwargs)
        self._bubble = Variable(shape=2, pos=True, name=self.name)
        self._bx = Variable(shape=1, pos=True, name=self.name )
        self._by = Variable(shape=1, pos=True, name=self.name)
        self._r = r #  Variable(shape=1, pos=True)

    def as_constraint(self, *args):
        """
        Imagine there are bubbles of a fixed size floating about constraining
        the discrete space

        each face is given a coordinate

         X = 1, Xg is coordinate      dist_real <= r
         X = 1, Xg is (0, 0)          dist_fake <= r + 1000

        note -
        2-norm is SIGNIFICANTLY Faster than 1norm.
        """
        N = len(self.space.faces)
        X = self.stacked            # selected faces
        M = 100                     # upper bound

        # centroids.shape[N, 2]
        centroids = np.asarray(self.space.faces.centroids)
        centroids = Parameter(shape=centroids.shape, value=centroids)
        cx = cvx.multiply(centroids[:, 0], X)
        cy = cvx.multiply(centroids[:, 1], X)
        Xg = cvx.vstack([cx, cy]).T

        # C = [] todo [speedups]- is one of these faster ?
        # for i in range(N):
        #     C.append(cvx.norm(cvx.vec(self._bubble - Xg[i, :]), 2) <= r + M * (1 - X[i]))
        v = cvx.vstack([cvx.promote(self._bx, (N,)), cvx.promote(self._by, (N,))]).T
        C = [cvx.norm(v - Xg, 2, axis=1) <= self._r + M * (1 - X)]
        return C

    def as_objective(self, **kwargs):
        """ minimize the x and y domain projections """
        return None # Minimize(self._r)

    def display(self):
        # v = self._bubble.value
        v = [self._bx.value, self._by.value]
        data = [{'x': v[0], 'y': v[1], 'r':self._r, 'index':self.name, 'name':self.name }]
        return {'circles': data}


class Dissallowed(FormulationDisc):
    def as_constraint(self, *args):
        """
         - covers cannot be seperated

        this is somehow vertex covers
        """
        # dual_space = self.space.dual()
        pass


# --------------------------------------------------------------
# FAILS
# --------------------------------------------------------------
class BoxBoundedSet(FormulationDisc):
    def __init__(self, space, action, area, box=None, **kwargs):
        """ """
        FormulationDisc.__init__(self, space, actions=action, **kwargs)
        # assert box is not None or area is not None, 'need a '
        from .fp_cont import MinArea, BoxInputList
        self._box = Variable(shape=4, name='Box_%s' % self.name, pos=True)
        self._area = area # MinArea(self._box, area, method='log')

    def as_constraint(self, *args):
        """

        """
        from .fp_cont import MinArea, BoxInputList
        X = self.stacked            # selected faces
        M = 100                     # upper bound

        # centroids.shape[N, 2]
        centroids = np.asarray(self.space.faces.centroids)
        centrx = Parameter(shape=centroids[:, 0].shape, value=centroids[:, 0])
        centry = Parameter(shape=centroids[:, 1].shape, value=centroids[:, 1])
        cx = cvx.multiply(centrx, X)
        cy = cvx.multiply(centry, X)

        C = [self._box[0] - self._box[2] / 2 <= cx + M * (1 - X),
             self._box[0] + self._box[2] / 2 >= cx,
             self._box[1] - self._box[3] / 2 <= cy + M * (1 - X),
             self._box[1] + self._box[3] / 2 >= cy,
             # MinArea.mean_area()
             self._box[2] >= 1,
             self._box[3] >= 1,
             self._box[3] * self._box[2] <= self._area
             # cvx.geo_mean(cvx.vstack([self._box[3], self._box[2]])) <= np.sqrt(self._area)
             # cvx.log(self._box[3]) + cvx.log(self._box[2]) <= np.log(self._area)
             ]
        return C

    def as_objective(self, **kwargs):
        """ minimize the x and y domain projections """
        return self._obj

    def display(self):
        x, y, w, h = self._box.value
        print('area', w* h)
        data = {'x': x, 'y': y, 'w': w, 'h': w, 'index': self.name, 'name': self.name}
        return {'boxes': [data]}


class GloballyConnectedSet(FormulationDisc):
    def __init__(self, space, action, area, upper=None, **kwargs):
        FormulationDisc.__init__(self, space, actions=action, **kwargs)
        self._bound = upper
        self._area = area

    def as_constraint(self, *args):
        """ aspect in discrete space -
            project set
            p(S_x,y) => S_x'
            p(S_x,y) => S_y'

            S_y' / S_x' <= limit

        """
        import scipy.spatial
        # compute distance matrix of space
        N = len(self.space.faces)

        # centroids.shape[N, 2]
        centroids = np.asarray(self.space.faces.centroids)
        Dij = scipy.spatial.distance_matrix(centroids, centroids)
        Dij[np.tril_indices(N, 1)] = 0

        # aspect_matrix heuristic [symmetric]
        root_upper = np.ceil(np.sqrt(self._area)) + 1
        bnd = np.linalg.norm([root_upper, root_upper], 2)
        print(self._area, bnd)
        # approximate aspect from bounding box area requirements
        # if aspect_matrix[i, j] = 1, then the aspect is
        # return [self.stacked * Dij <= bnd]
        C = []

        mx = cvx.multiply(centroids[:, 0], self.stacked)
        my = cvx.multiply(centroids[:, 1], self.stacked)
        return []

    def as_objective(self, **kwargs):
        """ minimize the x and y domain projections
            todo I am STUCK HERE
        """
        centroids = np.asarray(self.space.faces.centroids)
        n = self._area
        mx = cvx.multiply(centroids[:, 0], self.stacked)
        my = cvx.multiply(centroids[:, 1], self.stacked)

        # my = cvx.multiply(centroids[:, 1], self.stacked)
        u = Variable(shape=n)

        ex = cvx.max(mx - cvx.sum(mx) / n)
        ey = cvx.max(my - cvx.sum(my) / n)
        return Minimize(ex + ey)



