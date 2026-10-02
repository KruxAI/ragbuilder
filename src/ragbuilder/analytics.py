"""Compatibility entry point for usage events from the legacy UI."""
from ragbuilder.core.telemetry import telemetry


def track_event(event_str):
    if event_str == "0":
        telemetry.track_ui_run()
