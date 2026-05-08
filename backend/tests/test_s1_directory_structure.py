from pathlib import Path


def test_src_directory_structure():
    robot_agent = Path("backend/app/robot_agent")
    assert robot_agent.exists()
    assert (robot_agent / "poller").is_dir()
    assert (robot_agent / "agent").is_dir()
    assert (robot_agent / "memory").is_dir()
    assert (robot_agent / "api").is_dir()
    assert (robot_agent / "simulation").is_dir()
    assert (robot_agent / "seed").is_dir()
    assert (robot_agent / "chroma").is_dir()

    frontend = Path("frontend")
    assert frontend.exists()


def test_agent_knowledge_structure():
    ak = Path("backend/agent_knowledge")
    assert ak.exists()
    assert (ak / "rules").is_dir()
    assert (ak / "skills" / "routing").is_dir()
    assert (ak / "skills" / "create_plan").is_dir()


def test_config_files_exist():
    assert (Path("backend") / "pyproject.toml").exists()
    assert (Path("backend") / "app").is_dir()
    assert Path("backend").exists()
    assert Path("frontend").exists()
    assert (Path("backend") / "tests").exists()
    assert Path("artifacts").exists()


def test_init_files_exist():
    robot_agent = Path("backend/app/robot_agent")
    for subdir in ["poller", "agent", "memory", "api", "simulation", "seed", "chroma"]:
        assert (robot_agent / subdir / "__init__.py").exists()
