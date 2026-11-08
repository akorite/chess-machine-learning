import json

import pytest

from chessfoundation.budget import (
    BudgetExceeded,
    BudgetState,
    check_run,
    parse_colab_usage,
    project_cu,
    read_ledger,
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


def test_allows_run_within_every_bound_and_returns_projection(tmp_path):
    projected = check_run(
        BudgetConfig(),
        BudgetState(remaining_cu=200),
        "L4",
        hours=6,
        phase="pretrain",
        ledger_path=tmp_path / "ledger.json",
    )
    assert projected == pytest.approx(11.64)


def test_check_run_accumulates_used_cu_across_runs(tmp_path):
    ledger = tmp_path / "ledger.json"
    state = BudgetState(remaining_cu=200)
    check_run(BudgetConfig(), state, "L4", hours=6, phase="pretrain", ledger_path=ledger)
    check_run(BudgetConfig(), state, "L4", hours=6, phase="pretrain", ledger_path=ledger)
    saved = json.loads(ledger.read_text())
    assert saved["used_cu"] == pytest.approx(23.28)
    assert saved["phases"]["pretrain"] == pytest.approx(23.28)
    assert saved["remaining_cu"] == pytest.approx(200 - 11.64)


def test_ledger_survives_restart_and_loads_prior_spend(tmp_path):
    ledger = tmp_path / "ledger.json"
    state = BudgetState(remaining_cu=200)
    check_run(BudgetConfig(), state, "L4", hours=6, phase="pretrain", ledger_path=ledger)
    assert read_ledger(ledger)["used_cu"] == pytest.approx(11.64)
    # a restart reloads the file and accumulates on top
    check_run(BudgetConfig(), state, "L4", hours=4, phase="data", ledger_path=ledger)
    saved = read_ledger(ledger)
    assert saved["used_cu"] == pytest.approx(19.4)
    assert saved["phases"]["data"] == pytest.approx(7.76)


def test_cumulative_spend_refuses_over_ceiling(tmp_path):
    ledger = tmp_path / "ledger.json"
    state = BudgetState(remaining_cu=500)
    for _ in range(3):
        check_run(BudgetConfig(), state, "L4", hours=20, phase="pretrain", ledger_path=ledger)
    # 3 x 38.8 = 116.4 logged; a fourth run would take cumulative spend past 150
    with pytest.raises(BudgetExceeded, match="cumulative"):
        check_run(BudgetConfig(), state, "L4", hours=20, phase="pretrain", ledger_path=ledger)


def test_unknown_phase_fails():
    with pytest.raises(BudgetExceeded, match="unknown phase"):
        check_run(BudgetConfig(), BudgetState(remaining_cu=200), "L4", 1, phase="nope")
