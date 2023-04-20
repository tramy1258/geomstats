import random

import pytest

from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from geomstats.test.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestCase,
    FullRankCorrelationAffineQuotientMetricTestCase,
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.full_rank_correlation_matrices_data import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 8),
    ],
)
def spaces(request):
    request.cls.space = FullRankCorrelationMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestFullRankCorrelationMatrices(
    FullRankCorrelationMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundles(request):
    n = request.param
    request.cls.space = CorrelationMatricesBundle(n=n)
    request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundles")
class TestCorrelationMatricesBundle(
    CorrelationMatricesBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 5),
    ],
)
def equipped_spaces(request):
    n = request.param
    request.cls.space = space = FullRankCorrelationMatrices(n=n, equip=False)
    space.equip_with_metric(FullRankCorrelationAffineQuotientMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestFullRankCorrelationAffineQuotientMetric(
    FullRankCorrelationAffineQuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationAffineQuotientMetricTestData()
