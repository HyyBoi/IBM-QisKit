import pytest

from floodcut.config import ScenarioConfig
from floodcut.objective import build_problem
from floodcut.scenario import generate_scenario


@pytest.fixture(scope="session")
def default_problem():
    return build_problem(generate_scenario(ScenarioConfig()), 3)


@pytest.fixture(scope="session", params=[(0, 10), (2, 10), (1, 12)])
def problem_variants(request):
    seed, n = request.param
    return build_problem(generate_scenario(ScenarioConfig(seed=seed, n_sites=n)), 3)
