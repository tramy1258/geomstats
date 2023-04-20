import pytest

from geomstats.test.geometry.base import OpenSetTestCase, RiemannianMetricTestCase
from geomstats.test.vectorization import generate_vectorization_data


class PoincareBallTestCase(OpenSetTestCase):
    pass


class PoincareBallMetricTestCase(RiemannianMetricTestCase):
    # TODO: complete tests
    def test_mobius_add(self, point_a, point_b, project_first, expected, atol):
        res = self.space.metric.mobius_add(
            point_a, point_b, project_first=project_first
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_mobius_add_vec(self, n_reps, atol):
        # TODO: check project first false
        project_first = True
        point_a = self.data_generator.random_point()
        point_b = self.data_generator.random_point()

        expected = self.space.metric.mobius_add(
            point_a,
            point_b,
            project_first=project_first,
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point_a=point_a,
                    point_b=point_b,
                    project_first=project_first,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_retraction(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_retraction_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.retraction(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
