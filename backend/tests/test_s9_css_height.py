from __future__ import annotations

import pytest


class TestCssHeightFix:
    def test_incident_sidebar_has_fixed_flex_width(self):
        """IncidentSidebar should have flex: 0 0 280px for fixed width."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        assert ".incident-sidebar" in content

        sidebar_start = content.find(".incident-sidebar")
        next_rule = content.find("\n.", sidebar_start + 1)
        sidebar_block = content[sidebar_start:next_rule]

        assert "flex: 0 0 280px" in sidebar_block or "flex: 0 0 280px" in content
        assert "height: 100%" in sidebar_block

    def test_chat_panel_has_flex_1(self):
        """ChatPanel should have flex: 1 to fill remaining space."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        assert ".chat-panel" in content

        chat_start = content.find(".chat-panel")
        next_rule = content.find("\n.", chat_start + 1)
        chat_block = content[chat_start:next_rule]

        assert "flex: 1" in chat_block
        assert "height: 100%" in chat_block

    def test_dashboard_bottom_uses_flexbox_row(self):
        """dashboard-bottom should use flex row, not grid columns."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        dash_start = content.find(".dashboard-bottom")
        next_rule = content.find("\n.", dash_start + 1)
        dash_block = content[dash_start:next_rule]

        assert "flex-direction: row" in dash_block or "flex-direction: row;" in dash_block
        assert "height: 100%" in dash_block

    def test_incident_sidebar_overflow_y_auto(self):
        """incident-list should have overflow-y: auto for scrolling."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        list_start = content.find(".incident-list")
        next_rule = content.find("\n.", list_start + 1)
        list_block = content[list_start:next_rule]

        assert "overflow-y: auto" in list_block

    def test_chat_container_flex_1_overflow_auto(self):
        """chat-container should have flex: 1 and overflow-y: auto."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        container_start = content.find(".chat-container")
        next_rule = content.find("\n.", container_start + 1)
        container_block = content[container_start:next_rule]

        assert "flex: 1" in container_block
        assert "overflow-y: auto" in container_block

    def test_dashboard_height_fill_remaining(self):
        """dashboard should fill remaining viewport height."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        dash_start = content.find(".dashboard")
        next_rule = content.find("\n.", dash_start + 1)
        dash_block = content[dash_start:next_rule]

        assert "height: calc(100vh - 100px)" in dash_block or "height: calc(100vh" in dash_block

    def test_min_height_0_for_flex_children(self):
        """dashboard-bottom should have min-height: 0 for nested flex."""
        with open("frontend-react/src/styles/index.css", "r") as f:
            content = f.read()

        dash_start = content.find(".dashboard-bottom")
        next_rule = content.find("\n.", dash_start + 1)
        dash_block = content[dash_start:next_rule]

        assert "min-height: 0" in dash_block
