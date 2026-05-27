from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def test_src_directory_structure():
    robot_agent = BASE / "app" / "robot_agent"
    assert robot_agent.exists()
    assert (robot_agent / "poller").is_dir()
    assert (robot_agent / "agent").is_dir()
    assert (robot_agent / "memory").is_dir()
    assert (robot_agent / "api").is_dir()
    assert (robot_agent / "simulation").is_dir()
    assert (robot_agent / "seed").is_dir()
    assert (robot_agent / "chroma").is_dir()


def test_agent_knowledge_structure():
    ak = BASE / "agent_knowledge"
    assert ak.exists()
    assert (ak / "rules").is_dir()
    assert (ak / "skills" / "diagnose").is_dir()
    assert (ak / "skills" / "create-plan").is_dir()


def test_config_files_exist():
    assert (BASE / "pyproject.toml").exists()
    assert (BASE / "app").is_dir()
    assert BASE.exists()
    assert (BASE.parent / "frontend").exists()
    assert (BASE / "tests").exists()
    assert (BASE.parent / "artifacts").exists()


def test_init_files_exist():
    robot_agent = BASE / "app" / "robot_agent"
    for subdir in ["poller", "agent", "memory", "api", "simulation", "seed", "chroma"]:
        assert (robot_agent / subdir / "__init__.py").exists()
