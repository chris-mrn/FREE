"""Tests for training/curriculum module."""
import numpy as np
from training.curriculum import CurriculumState


def test_initial_state():
    c = CurriculumState()
    assert c.phase == 0
    assert c.restart_count == 0
    assert c.last_speed_step is None


def test_round_trip_serialization():
    c = CurriculumState()
    c.phase = 2
    c.restart_count = 1
    c.last_speed_step = 100000
    c.t_grid = [0.1, 0.5, 0.9]
    c.v_t    = [1.0, 2.0, 1.5]
    d = c.to_dict()
    c2 = CurriculumState.from_dict(d)
    assert c2.phase == 2
    assert c2.restart_count == 1
    assert c2.last_speed_step == 100000
    assert c2.t_grid == [0.1, 0.5, 0.9]


class _FakeArgs:
    curriculum_start = 10000
    curriculum_blend = 5000
    curriculum_restarts = 2
    curriculum_restart_every = 20000


def test_should_start_curriculum():
    c = CurriculumState()
    args = _FakeArgs()
    assert c.should_start_curriculum(10000, args)
    assert not c.should_start_curriculum(5000, args)


def test_should_end_blend():
    c = CurriculumState()
    c.phase = 1
    c.last_speed_step = 10000
    args = _FakeArgs()
    assert not c.should_end_blend(14999, args)
    assert c.should_end_blend(15000, args)


def test_should_restart():
    c = CurriculumState()
    c.phase = 2
    c.restart_count = 0
    c.last_speed_step = 10000
    args = _FakeArgs()
    assert c.should_restart(30000, args)
    c.restart_count = 2
    assert not c.should_restart(30000, args)
