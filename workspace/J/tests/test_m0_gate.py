import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("/usr/local/ASU-unlearning/workspace/J")))
from diagnose_m0 import bootstrap_mean_ci, decide, kl_api_error


def test_go_and_nogo_bounds():
    assert decide(True, True, 0.051, 0.20) == "GO"
    assert decide(True, True, 0.05, 0.20) == "NO-GO"
    assert decide(True, True, 0.2, 0.19) == "NO-GO"
    assert decide(True, False, 0.2, 0.5) == "NO-GO"
    assert decide(False, True, 0.2, 0.5) == "INCONCLUSIVE"


def test_bootstrap_seed_is_stable():
    first = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], seed=1001, samples=20)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], seed=1001, samples=20)
    assert first == second
    assert first["n_authors"] == 4


def test_kl_api_matches_asu_loss_autograd():
    torch.manual_seed(1001)
    student = torch.randn(2, 6, 11)
    teacher = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    labels[:, :2] = -100
    input_ids = torch.randint(0, 11, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    report = kl_api_error(student, teacher, input_ids, labels, mask)
    assert report["sign_ok"]
    assert report["max_rel_error"] < 1e-4
    assert report["ok"]
