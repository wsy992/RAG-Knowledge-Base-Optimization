from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_launcher() -> str:
    return (REPO_ROOT / "scripts" / "run_demo.ps1").read_text(encoding="utf-8")


def test_demo_launcher_loads_repo_dotenv_before_credential_check():
    script = read_launcher()

    assert ".env" in script
    assert "Get-Content" in script
    assert "DEEPSEEK_API_KEY" in script


def test_demo_launcher_exposes_src_for_rag_v2_imports():
    script = read_launcher()

    assert "$env:PYTHONPATH" in script
    assert "Join-Path $repoRoot \"src\"" in script
