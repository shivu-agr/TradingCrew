"""Workflow toolbar elapsed timer is present and cache-busted."""

from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "web" / "frontend"


def test_workflow_elapsed_markup_and_clock():
    html = (FRONTEND / "index.html").read_text()
    js = (FRONTEND / "app.js").read_text()
    css = (FRONTEND / "style.css").read_text()
    assert 'id="workflow-elapsed"' in html
    assert "workflowElapsedLabel" in html
    assert "get workflowElapsedLabel" in js
    assert "formatElapsed" in js
    assert ".workflow-elapsed" in css
    assert "/static/app.js?v=5" in html
    assert "/static/help_content.js?v=8" in html
