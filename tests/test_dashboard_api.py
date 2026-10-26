"""Tests for OrbitX 3D Mission Control API endpoints and coordinate calculations."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from scripts.launch_dashboard import OrbitXDashboardHandler, get_conjunctions


def test_conjunctions_data_loader():
    df = get_conjunctions()
    assert len(df) > 0
    assert "debris_name" in df.columns
    assert "miss_distance" in df.columns
    assert "relative_speed" in df.columns


def test_dashboard_api_status_payload():
    handler = OrbitXDashboardHandler.__new__(OrbitXDashboardHandler)
    status = handler.handle_status()
    assert status["mission"] == "OrbitX"
    assert status["status"] == "ONLINE"
    assert "Mayank Gupta" in status["authors"]
    assert "Soumya Sekhar Parida" in status["authors"]


def test_dashboard_api_encounters_payload():
    handler = OrbitXDashboardHandler.__new__(OrbitXDashboardHandler)
    encounters = handler.handle_encounters()
    assert isinstance(encounters, list)
    assert len(encounters) > 0
    first = encounters[0]
    assert "event_id" in first
    assert "miss_distance" in first
    assert first["miss_distance"] > 0


def test_dashboard_api_encounter_detail():
    handler = OrbitXDashboardHandler.__new__(OrbitXDashboardHandler)
    detail = handler.handle_encounter_detail(None)
    assert "target_orbit" in detail
    assert "debris_trajectory" in detail
    assert len(detail["target_orbit"]) > 50
    assert len(detail["debris_trajectory"]) > 20
    assert len(detail["range_series"]) > 20


def test_dashboard_api_rl_avoidance():
    handler = OrbitXDashboardHandler.__new__(OrbitXDashboardHandler)
    rl = handler.handle_rl_avoidance(None)
    assert rl["status"] == "COLLISION AVOIDED"
    assert rl["delta_v_mag"] > 0
    assert rl["cleared_miss_m"] > rl["nominal_miss_m"]
    assert len(rl["nominal_trajectory"]) > 0
    assert len(rl["avoidance_trajectory"]) > 0
