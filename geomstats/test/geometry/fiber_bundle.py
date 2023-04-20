import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.geometry.base import FiberBundleTestCase
from geomstats.test.geometry.general_linear import GeneralLinearTestCase


class GeneralLinearBuresWassersteinBundle(FiberBundle, GeneralLinear):
    def __init__(self, n):
        super().__init__(
            n=n,
            group=SpecialOrthogonal(n),
        )

    @staticmethod
    def default_metric():
        return MatricesMetric

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None:
            if fiber_point is not None:
                base_point = self.riemannian_submersion(fiber_point)
            else:
                raise ValueError(
                    "Either a point (of the total space) or a "
                    "base point (of the base manifold) must be "
                    "given."
                )
        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


class GeneralLinearBuresWassersteinBundleTestCase(
    FiberBundleTestCase, GeneralLinearTestCase
):
    pass
