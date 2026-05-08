from __future__ import annotations

import json
import os
import pytest


class TestReactUISmoke:
    def test_package_json_has_react_dependencies(self):
        with open('frontend-react/package.json') as f:
            pkg = json.load(f)
        assert 'react' in pkg['dependencies']
        assert 'react-dom' in pkg['dependencies']
        assert 'zustand' in pkg['dependencies']
        assert '@vitejs/plugin-react' in pkg['devDependencies']
        assert 'vite' in pkg['devDependencies']

    def test_vite_config_exists(self):
        assert os.path.exists('frontend-react/vite.config.js')

    def test_index_html_exists(self):
        assert os.path.exists('frontend-react/index.html')

    def test_main_jsx_exists(self):
        assert os.path.exists('frontend-react/src/main.jsx')

    def test_app_jsx_exists(self):
        assert os.path.exists('frontend-react/src/App.jsx')

    def test_css_file_exists(self):
        assert os.path.exists('frontend-react/src/styles/index.css')

    def test_robot_grid_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/RobotGrid.jsx')

    def test_alert_feed_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/AlertFeed.jsx')

    def test_incident_sidebar_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/IncidentSidebar.jsx')

    def test_chat_panel_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/ChatPanel.jsx')

    def test_hitl_card_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/HITLCard.jsx')

    def test_collapsible_step_jsx_exists(self):
        assert os.path.exists('frontend-react/src/components/CollapsibleStep.jsx')

    def test_sse_api_exists(self):
        assert os.path.exists('frontend-react/src/api/sse.js')

    def test_zustand_store_exists(self):
        assert os.path.exists('frontend-react/src/stores/useSimulationStore.js')


        with open('frontend-react/src/styles/index.css') as f:
            css = f.read()
        assert 'word-wrap: break-word' in css
        assert 'white-space: pre-wrap' in css

    def test_css_contains_observation_details(self):
        with open('frontend-react/src/styles/index.css') as f:
            css = f.read()
        assert '.observation-details' in css
        assert '.observation-summary' in css

    def test_hitl_card_css_has_amber_border(self):
        with open('frontend-react/src/styles/index.css') as f:
            css = f.read()
        assert 'accent-amber' in css

    def test_robot_grid_css_has_47_cells(self):
        with open('frontend-react/src/styles/index.css') as f:
            css = f.read()
        assert '.robot-cell' in css
        assert '.robot-cell.alert-l1' in css
        assert '.robot-cell.alert-l2' in css
        assert '.robot-cell.processing' in css
        assert '.robot-cell.done' in css
