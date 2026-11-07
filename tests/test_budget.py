import pytest

from chessfoundation.budget import (
    BudgetExceeded,
    BudgetState,
    check_run,
    parse_colab_usage,
    project_cu,
)
from chessfoundation.config import BudgetConfig


def test_parse_colab_usage():
    state = parse_colab_usage("Current balance: 261.83 compute units\n")
    assert state.remaining_cu == pytest.approx(261.83)


def test_parse_rejects_unreadable_output():
    with pytest.raises(BudgetExceeded):
        parse_colab_usage("no numbers here")


def test_projection_uses_accelerator_rate():
    assert project_cu("L4", 6) == pytest.approx(11.64)
    with pytest.raises(BudgetExceeded):
        project_cu("V100", 1)


def test_refuses_run_over_ceiling():
    with pytest.raises(BudgetExceeded, match="ceiling"):
        check_run(BudgetConfig(), BudgetState(remaining_cu=400), "G4", hours=20)


def test_refuses_run_that_eats_reserve():
    # 6h on L4 is 11.64 CU; 12 CU remaining leaves 0.36 below the 5 CU reserve.
    with pytest.raises(BudgetExceeded, match="reserve"):
        check_run(BudgetConfig(), BudgetState(remaining_cu=12.0), "L4", hours=6)


def test_refuses_run_over_phase_allocation():
    with pytest.raises(BudgetExceeded, match="phase 'pretrain'"):
        check_run(
            BudgetConfig(),
            BudgetState(remaining_cu=200),
            "L4",
            hours=30,
            phase="pretrain",
        )


def test_allows_run_within_every_bound_and_returns_projection():
    projected = check_run(
        BudgetConfig(), BudgetState(remaining_cu=200), "L4", hours=6, phase="pretrain"
    )
    assert projected == pytest.approx(11.64)


def test_unknown_phase_fails():
    with pytest.raises(BudgetExceeded, match="unknown phase"):
        check_run(BudgetConfig(), BudgetState(remaining_cu=200), "L4", 1, phase="nope")
