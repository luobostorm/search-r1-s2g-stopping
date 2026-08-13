import pytest

from kstar.searchr1_v02_eval import (
    normalize_answer,
    official_em,
    strict_em,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("The Beijing.", "beijing"),
        ("  An   Answer  ", "answer"),
        ("U.S.A.", "usa"),
        ("THEATER", "theater"),
    ],
)
def test_normalize_answer_matches_official_order(value, expected):
    assert normalize_answer(value) == expected


def test_strict_and_official_em_remain_distinct():
    gold = ["The United States", "USA"]

    assert strict_em("the united states", gold) == 0
    assert official_em("the united states", gold) == 1
    assert strict_em("USA", gold) == 1
    assert official_em("U.S.A.", gold) == 1


def test_missing_answer_scores_zero():
    assert strict_em(None, ["answer"]) == 0
    assert official_em(None, ["answer"]) == 0


def test_wilson_interval_is_bounded_and_contains_observed_rate():
    lower, upper = wilson_interval(24, 50)

    assert 0.0 <= lower < 0.48 < upper <= 1.0
    assert lower == pytest.approx(0.3479713529)
    assert upper == pytest.approx(0.6148825511)
