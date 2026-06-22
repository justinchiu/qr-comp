import local_eval
from baselines.geqrf_baseline import custom_kernel


def test_geqrf_baseline_passes_local_eval() -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=1, seed=1234, case="mixed")
    good, message = local_eval.check_implementation(data, custom_kernel(data))
    assert good, message
