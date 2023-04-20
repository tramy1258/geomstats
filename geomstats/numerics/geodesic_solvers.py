from abc import ABC, abstractmethod

import logging
import numpy as np

import geomstats.backend as gs
from geomstats.numerics.bvp_solvers import ScipySolveBVP
from geomstats.numerics.ivp_solvers import GSIntegrator
from geomstats.numerics.optimizers import ScipyMinimize

# TODO: check uses of space.dim


class ExpSolver(ABC):
    @abstractmethod
    def exp(self, space, tangent_vec, base_point):
        pass

    @abstractmethod
    def geodesic_ivp(self, space, tangent_vec, base_point, t):
        pass


class ExpIVPSolver(ExpSolver):
    def __init__(self, integrator=None):
        if integrator is None:
            integrator = GSIntegrator()

        self.integrator = integrator

    def _solve(self, space, tangent_vec, base_point, t_eval=None):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        if self.integrator.state_is_raveled:
            initial_state = gs.hstack([base_point, tangent_vec])
        else:
            initial_state = gs.stack([base_point, tangent_vec])

        force = self._get_force(space)
        if t_eval is None:
            return self.integrator.integrate(force, initial_state)

        return self.integrator.integrate_t(force, initial_state, t_eval)

    def exp(self, space, tangent_vec, base_point):
        result = self._solve(space, tangent_vec, base_point)
        return self._simplify_exp_result(result, space)

    def geodesic_ivp(self, space, tangent_vec, base_point):
        base_point = gs.broadcast_to(base_point, tangent_vec.shape)

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            result = self._solve(space, tangent_vec, base_point, t_eval=t)
            return self._simplify_result_t(result, space)

        return path

    def _get_force(self, space):
        if self.integrator.state_is_raveled:
            force_ = lambda state, t: self._force_raveled_state(state, t, space=space)
        else:
            force_ = lambda state, t: self._force_unraveled_state(state, t, space=space)

        if self.integrator.tfirst:
            return lambda t, state: force_(state, t)

        return force_

    def _force_raveled_state(self, raveled_initial_state, _, space):
        # input: (n,)

        # assumes unvectorize
        state = gs.reshape(raveled_initial_state, (space.dim, space.dim))

        eq = space.metric.geodesic_equation(state, _)

        return gs.flatten(eq)

    def _force_unraveled_state(self, initial_state, _, space):
        return space.metric.geodesic_equation(initial_state, _)

    def _simplify_exp_result(self, result, space):
        y = result.get_last_y()

        if self.integrator.state_is_raveled:
            return y[..., : space.dim]

        return y[0]

    def _simplify_result_t(self, result, space):
        # assumes several t
        y = result.y

        if self.integrator.state_is_raveled:
            y = y[..., : space.dim]

            if gs.ndim(y) > 2:
                return gs.moveaxis(y, 0, 1)
            return y

        y = y[:, 0, :, ...]
        if gs.ndim(y) > 2:
            return gs.moveaxis(y, 1, 0)
        return y


class LogSolver(ABC):
    @abstractmethod
    def log(self, space, point, base_point):
        pass

    @abstractmethod
    def geodesic_bvp(self, space, point, base_point):
        pass


class _GeodesicBVPFromExpMixins:
    def _geodesic_bvp_single(self, space, t, tangent_vec, base_point):
        tangent_vec_ = gs.einsum("...,...i->...i", t, tangent_vec)
        return space.metric.exp(tangent_vec_, base_point)

    def geodesic_bvp(self, space, point, base_point):
        tangent_vec = self.log(space, point, base_point)
        is_batch = tangent_vec.ndim > space.point_ndim

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            if not is_batch:
                return self._geodesic_bvp_single(space, t, tangent_vec, base_point)

            return gs.stack(
                [
                    self._geodesic_bvp_single(space, t, tangent_vec_, base_point_)
                    for tangent_vec_, base_point_ in zip(tangent_vec, base_point)
                ]
            )

        return path


class _LogBatchMixins:
    @abstractmethod
    def _log_single(self, space, point, base_point):
        pass

    def log(self, space, point, base_point):
        # assumes inability to properly vectorize
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            return self._log_single(space, point, base_point)

        return gs.stack(
            [
                self._log_single(space, point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


class LogShootingSolver:
    def __new__(cls, optimizer=None, initialization=None, flatten=True):
        if flatten:
            return _LogShootingSolverFlatten(
                optimizer=optimizer,
                initialization=initialization,
            )

        return _LogShootingSolverUnflatten(
            optimizer=optimizer,
            initialization=initialization,
        )


class _LogShootingSolverFlatten(_GeodesicBVPFromExpMixins, LogSolver):
    # TODO: add a (linear) initialization here?

    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.flatten(gs.random.rand(*base_point.shape))

    def _objective(self, velocity, space, point, base_point):
        velocity = gs.reshape(velocity, base_point.shape)
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def log(self, space, point, base_point):
        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        tangent_vec = gs.reshape(res.x, base_point.shape)

        return tangent_vec


class _LogShootingSolverUnflatten(
    _LogBatchMixins, _GeodesicBVPFromExpMixins, LogSolver
):
    def __init__(self, optimizer=None, initialization=None):
        if optimizer is None:
            optimizer = ScipyMinimize(jac="autodiff")

        if initialization is None:
            initialization = self._default_initialization

        self.optimizer = optimizer
        self.initialization = initialization

    def _default_initialization(self, space, point, base_point):
        return gs.random.rand(*base_point.shape)

    def _objective(self, velocity, space, point, base_point):
        delta = space.metric.exp(velocity, base_point) - point
        return gs.sum(delta**2)

    def _log_single(self, space, point, base_point):
        objective = lambda velocity: self._objective(velocity, space, point, base_point)
        init_tangent_vec = self.initialization(space, point, base_point)

        res = self.optimizer.optimize(objective, init_tangent_vec)

        return res.x


class LogBVPSolver(_LogBatchMixins, LogSolver):
    def __init__(self, n_nodes=10, integrator=None, initialization=None):
        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization

        self.n_nodes = n_nodes
        self.integrator = integrator
        self.initialization = initialization

        self.grid = self._create_grid()

    def _create_grid(self):
        return gs.linspace(0.0, 1.0, num=self.n_nodes)

    def _default_initialization(self, space, point, base_point):
        point_0, point_1 = base_point, point

        pos_init = gs.transpose(gs.linspace(point_0, point_1, self.n_nodes))

        vel_init = self.n_nodes * (pos_init[:, 1:] - pos_init[:, :-1])
        vel_init = gs.hstack([vel_init, vel_init[:, [-2]]])

        return gs.vstack([pos_init, vel_init])

    def boundary_condition(self, state_0, state_1, space, point_0, point_1):
        pos_0 = state_0[:space.dim]
        pos_1 = state_1[:space.dim]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    def bvp(self, _, raveled_state, space):
        # inputs: n (2*dim) , n_nodes
        # assumes unvectorized

        state = gs.moveaxis(gs.reshape(raveled_state, (2, space.dim, -1)), -2, -1)

        eq = space.metric.geodesic_equation(state, _)

        return gs.reshape(gs.moveaxis(eq, -2, -1), (2 * space.dim, -1))

    def _solve(self, space, point, base_point):
        bvp = lambda t, state: self.bvp(t, state, space)
        bc = lambda state_0, state_1: self.boundary_condition(
            state_0, state_1, space, base_point, point
        )

        y = self.initialization(space, point, base_point)

        return self.integrator.integrate(bvp, bc, self.grid, y)

    def _log_single(self, space, point, base_point):
        res = self._solve(space, point, base_point)
        return self._simplify_log_result(res, space)

    def geodesic_bvp(self, space, point, base_point):
        # TODO: add to docstrings: 0 <= t <= 1

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        is_batch = point.ndim > space.point_ndim
        if not is_batch:
            result = self._solve(space, point, base_point)
        else:
            results = [
                self._solve(space, point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]

        def path(t):
            if not gs.is_array(t):
                t = gs.array([t])

            if not is_batch:
                return self._simplify_result_t(result.sol(t), space)

            return gs.array(
                [self._simplify_result_t(result.sol(t), space) for result in results]
            )

        return path

    def _simplify_log_result(self, result, space):
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (2, space.dim))
        return tangent_vec
    
class LogPolynomialSolver(LogSolver):
# use minimize in optimize
    def __init__(self, n_segments=100, optimizer=None, integrator=None, initialization=None, method="L-BFGS-B", jac=None,
                 bounds=None, tol=None, callback=None, options=None, save_results=False):
        if optimizer is None:
            optimizer = ScipyMinimize(method=method, jac=jac, bounds=bounds, tol=tol, callback=callback, options=options, save_result=save_results)
        
        if integrator is None:
            integrator = ScipySolveBVP()

        if initialization is None:
            initialization = self._default_initialization
        
        self.n_segments = n_segments
        self.initialization = initialization
        self.optimizer = optimizer
        self.integrator = integrator
    
    def _approx_geodesic_bvp(
        self,
        space,
        initial_point,
        end_point,
        degree=5,
        method="BFGS",
        n_times=200,
        jac_on=True,
    ):
        def cost_fun(param):
            """Compute the energy of the polynomial curve defined by param."""
            last_coef = end_point - initial_point - gs.sum(param, axis=0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_curve = [t**i for i in range(degree + 1)]
            t_curve = gs.stack(t_curve)
            curve = gs.einsum("ij,ik->kj", coef, t_curve)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            if curve.min() < 0:
                return np.inf, np.inf, curve, np.nan

            velocity_sqnorm = space.metric.squared_norm(vector=velocity, base_point=curve)
            # print(velocity_sqnorm)
            length = gs.sum(velocity_sqnorm ** (1 / 2)) / n_times
            energy = gs.sum(velocity_sqnorm) / n_times
            return energy, length, curve, velocity     
        
        def cost_jacobian(param):
            """Compute the jacobian of the cost function at polynomial curve."""
            last_coef = end_point - initial_point - gs.sum(param, 0)
            coef = gs.vstack((initial_point, param, last_coef))

            t = gs.linspace(0.0, 1.0, n_times)
            t_position = [t**i for i in range(degree + 1)]
            t_position = gs.stack(t_position)
            position = gs.einsum("ij,ik->kj", coef, t_position)

            t_velocity = [i * t ** (i - 1) for i in range(1, degree + 1)]
            t_velocity = gs.stack(t_velocity)
            velocity = gs.einsum("ij,ik->kj", coef[1:], t_velocity)

            fac1 = gs.stack(
                [
                    k * t ** (k - 1) - degree * t ** (degree - 1)
                    for k in range(1, degree)
                ]
            )
            fac2 = gs.stack([t**k - t**degree for k in range(1, degree)])
            fac3 = (velocity * gs.polygamma(1, position)).T - gs.sum(
                velocity, 1
            ) * gs.polygamma(1, gs.sum(position, 1))
            fac4 = (velocity**2 * gs.polygamma(2, position)).T - gs.sum(
                velocity, 1
            ) ** 2 * gs.polygamma(2, gs.sum(position, 1))

            cost_jac = (
                2 * gs.einsum("ij,kj->ik", fac1, fac3)
                + gs.einsum("ij,kj->ik", fac2, fac4)
            ) / n_times
            return cost_jac.T.reshape(dim * (degree - 1))

        def f2minimize(x):
            """Compute function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            res = cost_fun(param)
            return res[0]

        def jacobian(x):
            """Compute jacobian of the function to minimize."""
            param = gs.transpose(x.reshape((dim, degree - 1)))
            return cost_jacobian(param)
        
        dim = initial_point.shape[0]
        x0 = gs.ones(dim * (degree - 1))
        jac = jacobian if jac_on else None
        sol = self.optimizer.optimize(f2minimize, x0, jac=jac)
        opt_param = sol.x.reshape((dim, degree - 1)).T
        _, dist, curve, velocity = cost_fun(opt_param)

        return dist, curve, velocity
    
    def _default_initialization(self, space, point, base_point, n_segments):
        #_approx_geodesic_bvp
        _, curve, velocity = self._approx_geodesic_bvp(
            space, base_point, point, n_times=n_segments
        )
        return gs.vstack((curve.T, velocity.T))

    def bvp(self, _, raveled_state, space):
        state = gs.moveaxis(
            gs.reshape(raveled_state, (2, space.dim, -1)), -2, -1
        )

        eq = space.metric.geodesic_equation(state, _)

        eq = gs.reshape(gs.moveaxis(eq, -2, -1), (2 * space.dim, -1))
        
        return eq

    def boundary_condition(self, state_0, state_1, space, point_0, point_1):
        pos_0 = state_0[:space.dim]
        pos_1 = state_1[:space.dim]
        return gs.hstack((pos_0 - point_0, pos_1 - point_1))

    # def jac(_, raveled_state):
    #     n_dim = raveled_state.ndim
    #     n_times = raveled_state.shape[1] if n_dim > 1 else 1
    #     position, velocity = raveled_state[: space.dim], raveled_state[space.dim :]

    #     dgamma = space.metric.jacobian_christoffels(gs.transpose(position))

    #     df_dposition = -gs.einsum(
    #         "j...,...ijkl,k...->il...", velocity, dgamma, velocity
    #     )

    #     gamma = space.metric.christoffels(gs.transpose(position))
    #     df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

    #     jac_nw = (
    #         gs.zeros((space.dim, space.dim, raveled_state.shape[1]))
    #         if n_dim > 1
    #         else gs.zeros((space.dim, space.dim))
    #     )
    #     jac_ne = gs.squeeze(
    #         gs.transpose(gs.tile(gs.eye(space.dim), (n_times, 1, 1)))
    #     )
    #     jac_sw = df_dposition
    #     jac_se = df_dvelocity
    #     jac = gs.concatenate(
    #         (
    #             gs.concatenate((jac_nw, jac_ne), axis=1),
    #             gs.concatenate((jac_sw, jac_se), axis=1),
    #         ),
    #         axis=0,
    #     )

    #     return jac

    def geodesic_bvp(self, space, point, base_point, jacobian=False):
        all_results = []
        point, base_point = gs.broadcast_arrays(point, base_point)
        if point.ndim == 1:
            point = gs.expand_dims(point, axis=0)
            base_point = gs.expand_dims(base_point, axis=0)

        fun_jac = self.jac if jacobian else None

        for i in range(point.shape[0]):
            bvp = lambda t, state: self.bvp(t, state, space)
            bc = lambda state_0, state_1: self.boundary_condition(
                state_0, state_1, space, base_point[i], point[i]
            )
            x = gs.linspace(0.0, 1.0, self.n_segments) 
            y = self.initialization(space, point[i], base_point[i], self.n_segments)
            result = self.integrator.integrate(bvp, bc, x, y, fun_jac=fun_jac)
            if result.status == 1:
                logging.warning(
                    "The maximum number of mesh nodes for solving the  "
                    "geodesic boundary value problem is exceeded. "
                    "Result may be inaccurate."
                )
            all_results.append(result)

        def path(t):
            y_t = gs.array([result.sol(t)[:space.dim] for result in all_results])
            return gs.moveaxis(y_t,-1,-2)
            # return gs.expand_dims(gs.squeeze(y_t, axis=-2), axis=-1)

        return path

    def log(self, space, point, base_point, jacobian=False):
        all_results = []
        point, base_point = gs.broadcast_arrays(point, base_point)
        if point.ndim == 1:
            point = gs.expand_dims(point, axis=0)
            base_point = gs.expand_dims(base_point, axis=0)

        def jac(_, raveled_state):
            n_dim = raveled_state.ndim
            n_times = raveled_state.shape[1] if n_dim > 1 else 1
            position, velocity = raveled_state[: space.dim], raveled_state[space.dim :]

            dgamma = space.metric.jacobian_christoffels(gs.transpose(position))

            df_dposition = -gs.einsum(
                "j...,...ijkl,k...->il...", velocity, dgamma, velocity
            )

            gamma = space.metric.christoffels(gs.transpose(position))
            df_dvelocity = -2 * gs.einsum("...ijk,k...->ij...", gamma, velocity)

            jac_nw = (
                gs.zeros((space.dim, space.dim, raveled_state.shape[1]))
                if n_dim > 1
                else gs.zeros((space.dim, space.dim))
            )
            jac_ne = gs.squeeze(
                gs.transpose(gs.tile(gs.eye(space.dim), (n_times, 1, 1)))
            )
            jac_sw = df_dposition
            jac_se = df_dvelocity
            jac = gs.concatenate(
                (
                    gs.concatenate((jac_nw, jac_ne), axis=1),
                    gs.concatenate((jac_sw, jac_se), axis=1),
                ),
                axis=0,
            )

            return jac

        fun_jac = jac if jacobian else None

        for i in range(point.shape[0]):
            bvp = lambda t, state: self.bvp(t, state, space)
            bc = lambda state_0, state_1: self.boundary_condition(
                state_0, state_1, space, base_point[i], point[i]
            )
            x = gs.linspace(0.0, 1.0, self.n_segments) 
            y = self.initialization(space, point[i], base_point[i], self.n_segments)
            print(y)
            #result = self.integrator.integrate(bvp, bc, x, y, fun_jac=fun_jac)
            # if result.status == 1:
            #     logging.warning(
            #         "The maximum number of mesh nodes for solving the  "
            #         "geodesic boundary value problem is exceeded. "
            #         "Result may be inaccurate."
            #     )
            # all_results.append(result)
            
        return gs.squeeze(gs.vstack([self._simplify_result(result, space) for result in all_results]), axis=0)

    def _simplify_result(self, result, space):
        _, tangent_vec = gs.reshape(gs.transpose(result.y)[0], (2, space.dim))

        return tangent_vec