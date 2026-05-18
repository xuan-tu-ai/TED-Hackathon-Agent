from __future__ import annotations

from pathlib import Path


def test_docker_compose_exists():
    path = Path(__file__).resolve().parents[2] / "deploy" / "docker-compose.dev.yml"
    assert path.exists()


def test_docker_compose_has_postgres():
    path = Path(__file__).resolve().parents[2] / "deploy" / "docker-compose.dev.yml"
    content = path.read_text()
    assert "postgres:16-alpine" in content
    assert "5432:5432" in content


def test_docker_data_in_gitignore():
    gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
    content = gitignore.read_text()
    assert ".docker-data/" in content
