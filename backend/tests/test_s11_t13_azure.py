from __future__ import annotations

from pathlib import Path


def test_deploy_readme_exists():
    path = Path(__file__).resolve().parents[2] / "deploy" / "README.md"
    assert path.exists()


def test_deploy_readme_has_connection_string():
    path = Path(__file__).resolve().parents[2] / "deploy" / "README.md"
    content = path.read_text()
    assert "sslmode=require" in content
    assert "postgresql+psycopg" in content


def test_deploy_readme_has_secret_handling():
    path = Path(__file__).resolve().parents[2] / "deploy" / "README.md"
    content = path.read_text()
    assert "DATABASE_URL" in content
    assert "Key Vault" in content or "App Service" in content
