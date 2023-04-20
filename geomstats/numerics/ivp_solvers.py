from abc import ABC, abstractmethod

import scipy

import geomstats.backend as gs
import geomstats.integrator as gs_integrator
from geomstats.errors import check_parameter_accepted_values
from geomstats.numerics._common import result_to_backend_type


def _merge_scipy_results(results, same_t=False):
    keys = list(results[0].keys())
    merged_results = {key: [] for key in keys}

    for result in results:
        for key, value in merged_results.items():
            merged_results[key].append(result[key])

    if same_t:
        merged_results["t"] = gs.moveaxis(merged_results["t"], 0, 1)
        merged_results["y"] = gs.moveaxis(merged_results["y"], 0, 1)

    return merged_results


class OdeResult(scipy.optimize.OptimizeResult):
    # following scipy
    def get_last_y(self):
        # assumes last t is the same
        if gs.is_array(self.y):
            return self.y[-1]

        return gs.stack([y_[-1] for y_ in self.y])


class ODEIVPSolver(ABC):
    def __init__(self, save_result=False, state_is_raveled=False, tfirst=False):
        self.state_is_raveled = state_is_raveled
        self.tfirst = tfirst
        self.save_result = save_result

        self.result_ = None

    @abstractmethod
    def integrate(self, force, initial_state, end_time):
        pass

    @abstractmethod
    def integrate_t(self, force, initial_state, t_eval):
        pass


class GSIntegrator(ODEIVPSolver):
    # TODO: control time
    def __init__(self, n_steps=10, step_type="euler", save_result=False):
        super().__init__(save_result=save_result, state_is_raveled=False, tfirst=False)
        self.step_type = step_type
        self.n_steps = n_steps

    @property
    def step_type(self):
        return self._step_type

    @step_type.setter
    def step_type(self, value):
        if callable(value):
            step_function = value
            value = None
        else:
            check_parameter_accepted_values(
                value, "step_type", gs_integrator.STEP_FUNCTIONS
            )
            step_function = getattr(gs_integrator, gs_integrator.STEP_FUNCTIONS[value])

        self._step_function = step_function
        self._step_type = value

    def step(self, force, state, time, dt):
        return self._step_function(force, state, time, dt)

    def _get_n_fevals(self, n_steps):
        n_evals_step = gs_integrator.FEVALS_PER_STEP[self.step_type]
        return n_evals_step * n_steps

    def _integrate(self, force, initial_state, end_time=1.0):
        dt = end_time / self.n_steps
        states = [initial_state]
        current_state = initial_state

        for i in range(self.n_steps):
            current_state = self.step(
                force=force, state=current_state, time=i * dt, dt=dt
            )
            states.append(current_state)

        return states

    def integrate(self, force, initial_state, end_time=1.0):
        states = self._integrate(force, initial_state, end_time=end_time)

        ts = gs.linspace(0.0, end_time, self.n_steps + 1)
        nfev = self._get_n_fevals(self.n_steps)

        result = OdeResult(t=ts, y=gs.array(states), nfev=nfev, njev=0, sucess=True)

        if self.save_result:
            self.result_ = result

        return result

    def integrate_t(self, force, initial_state, t_eval):
        # TODO: this is a very naive implementation
        # based on previous generic implementation in geomstats
        # resolution gets worst for larger t

        states = []
        initial_states = [
            gs.stack([initial_state[0], t * initial_state[1]]) for t in t_eval
        ]
        for initial_state_ in initial_states:
            states_t = self._integrate(force, initial_state_, end_time=1.0)
            states.append(states_t[-1])

        nfev = self._get_n_fevals(self.n_steps)
        n_t = len(t_eval)
        result = OdeResult(
            t=t_eval, y=gs.stack(states), nfev=n_t * nfev, njev=0, sucess=True
        )

        if self.save_result:
            self.result_ = result

        return result


class ScipySolveIVP(ODEIVPSolver):
    # TODO: remember `vectorized` argument (only for Jacobian approximation)
    # TODO: remember `dense_output` argument

    def __init__(self, method="RK45", save_result=False, **options):
        super().__init__(save_result=save_result, state_is_raveled=True, tfirst=True)
        self.method = method
        self.options = options

    def _integrate(self, force, initial_state, end_time=1.0, t_eval=None):
        # TODO: parallelize?

        # TODO: check this condition
        if initial_state.ndim > 1:
            results = []
            for initial_state_ in initial_state:
                results.append(
                    self._integrate_single_point(
                        force, initial_state_, end_time, t_eval
                    )
                )

            result = OdeResult(_merge_scipy_results(results, same_t=t_eval is not None))

        else:
            result = self._integrate_single_point(
                force, initial_state, end_time, t_eval=t_eval
            )
            result = OdeResult(**result)

        if self.save_result:
            self.result_ = result

        return result

    def integrate(self, force, initial_state, end_time=1.0):
        return self._integrate(force, initial_state, end_time=end_time)

    def integrate_t(self, force, initial_state, t_eval):
        return self._integrate(force, initial_state, end_time=t_eval[-1], t_eval=t_eval)

    def _integrate_single_point(self, force, initial_state, end_time=1.0, t_eval=None):
        def force_(t, state):
            state = gs.from_numpy(state)
            return force(t, state)

        result = scipy.integrate.solve_ivp(
            force_,
            (0.0, end_time),
            initial_state,
            method=self.method,
            t_eval=t_eval,
            **self.options
        )
        result = result_to_backend_type(result)
        result.y = gs.moveaxis(result.y, 0, -1)

        return result
