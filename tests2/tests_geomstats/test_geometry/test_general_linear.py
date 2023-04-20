import random

import pytest

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.test.geometry.general_linear import GeneralLinearTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.general_linear_data import GeneralLinearTestData


@pytest.fixture(
    scope="class",
    params=[
        (2, True),
        (2, False),
        (random.randint(3, 5), True),
        (random.randint(3, 5), False),
    ],
)
def spaces(request):
    n, positive_det = request.param
    request.cls.space = GeneralLinear(n=n, positive_det=positive_det)


@pytest.mark.usefixtures("spaces")
class TestGeneralLinear(GeneralLinearTestCase, metaclass=DataBasedParametrizer):
    testing_data = GeneralLinearTestData()
