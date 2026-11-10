"""gan_check plan: gen_lr and step_mult options, plan content, old plans unchanged.

Pandas only (no torch needed)."""
import pytest

from gainsplit import plans
from gainsplit.config import RunConfig


def test_new_fields_default_so_old_configs_do_not_change():
    c = RunConfig("A4", 10, 0)
    assert (c.gen_lr, c.step_mult) == (0.0, 1)
    assert c.run_id == "A4_p10_s0"
    assert c.steps(100) == max(200, 10 * 4)


def test_steps_with_step_mult():
    assert RunConfig("A1", 1, 0).steps(100) == 200
    assert RunConfig("A1", 1, 0, step_mult=3).steps(100) == 600


@pytest.mark.parametrize("kw", [
    dict(arm="A1", gen_lr=1e-4),             # gen_lr is for GAN arms only
    dict(arm="A3", gen_lr=1e-4, helper_weight=0.3),
    dict(arm="A4", gen_lr=-1e-4),
    dict(arm="A4", step_mult=0),
    dict(arm="A1", step_mult=-2),
])
def test_bad_values_are_rejected(kw):
    with pytest.raises(ValueError):
        RunConfig(**{"budget": 10, "seed": 0, **kw})


def test_good_values_are_accepted():
    RunConfig("A4", 10, 0, gen_lr=2e-4)
    RunConfig("A5", 10, 0, gen_lr=1e-5, helper_weight=0.3)
    RunConfig("A1", 1, 0, step_mult=3)


def test_gan_check_plan():
    runs = plans.gan_check()
    ids = [r.run_id for r in runs]
    assert len(runs) == 12 and len(set(ids)) == 12
    assert not any(r.eval_test for r in runs) and all(r.tag for r in runs)
    old = {r.run_id for name in ("pass1", "pass2a", "pass2b", "pass3", "fgm_tuning") for r in plans.PLANS[name]()}
    assert not old & set(ids)
    glr = [r for r in runs if r.tag.startswith("glr")]
    assert {(r.arm, r.budget, r.gen_lr, r.tag) for r in glr} == {("A4", 10, 1e-5, "glr1e-05"), ("A4", 10, 2e-4, "glr0.0002")}
    long = [r for r in runs if r.tag == "steps3x"]
    assert {(r.arm, r.budget, r.step_mult) for r in long} == {("A4", 1, 3), ("A1", 1, 3)}
    assert {r.seed for r in runs} == {0, 1, 2}
    assert [r.budget for r in runs] == [10] * 6 + [1] * 6      # longest first
    assert "gan_check" in plans.PLANS


def test_old_plans_keep_default_options():
    for name in ("lr_tuning", "w_tuning", "gan_pilot", "pass1", "pass2a", "pass2b", "pass3", "fgm_tuning"):
        for r in plans.PLANS[name]():
            assert (r.gen_lr, r.step_mult) == (0.0, 1)
