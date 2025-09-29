"""
Project Integrity Tests for ExoplanetAI
Тесты целостности проекта ExoplanetAI
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels from tests to Exoplanet_AI root
sys.path.insert(0, str(project_root))

def test_project_structure():
    """Test that project structure is correct"""
    # Check that essential directories exist
    essential_dirs = [
        "backend",
        "frontend",
        "ml",
        "scripts",
        "config"
    ]
    
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"

def test_backend_structure():
    """Test that backend structure is correct"""
    backend_dir = project_root / "backend"
    
    # Check essential files
    essential_files = [
        "main.py",
        "requirements.txt"
    ]
    
    for file_name in essential_files:
        file_path = backend_dir / file_name
        assert file_path.exists(), f"File {file_name} does not exist in backend"
        assert file_path.is_file(), f"{file_name} is not a file"
    
    # Check essential directories
    essential_dirs = [
        "services",
        "models",
        "tests"
    ]
    
    for dir_name in essential_dirs:
        dir_path = backend_dir / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist in backend"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"

def test_ml_structure():
    """Test that ML structure is correct"""
    ml_dir = project_root / "ml"
    
    # Check that ML directory is not nested
    assert not (ml_dir / "ml").exists(), "Nested ml/ml directory found"
    
    # Check essential ML files
    essential_files = [
        "bls_ensemble.py",
        "data_loader.py",
        "inference_engine.py"
    ]
    
    for file_name in essential_files:
        file_path = ml_dir / file_name
        assert file_path.exists(), f"File {file_name} does not exist in ml"
        assert file_path.is_file(), f"{file_name} is not a file"

def test_scripts_structure():
    """Test that scripts structure is correct"""
    scripts_dir = project_root / "scripts"
    
    # Check essential scripts
    essential_scripts = [
        "setup_project.sh",
        "start_all.sh",
        "stop_all.sh"
    ]
    
    for script_name in essential_scripts:
        script_path = scripts_dir / script_name
        assert script_path.exists(), f"Script {script_name} does not exist"
        assert script_path.is_file(), f"{script_name} is not a file"
        
        # Check that scripts are executable
        assert os.access(script_path, os.X_OK), f"Script {script_name} is not executable"

def test_no_merge_conflicts():
    """Test that there are no merge conflicts in Python files"""
    import subprocess
    
    # Search for merge conflict markers in Python files
    try:
        # Look for conflict markers in the project directory, excluding venv and this test file
        result = subprocess.run(
            ["grep", "-r", "--exclude-dir=venv", "--exclude-dir=.git", "--exclude=" + os.path.basename(__file__), "<<<<<<<", "--include=*.py", str(project_root)],
            capture_output=True,
            text=True
        )
        # If grep finds anything, it will return 0, otherwise 1
        # We want to make sure there are NO merge conflicts (return code != 0 means no conflicts found)
        if result.returncode == 0:
            # Check if the matches are only in the test file itself
            lines_with_conflicts = result.stdout.strip().split('\n')
            non_test_conflicts = [line for line in lines_with_conflicts if __file__ not in line and line.strip()]
            if non_test_conflicts:
                assert False, f"Merge conflicts found:\n{chr(10).join(non_test_conflicts)}"
    except FileNotFoundError:
        # If grep is not available, skip this test
        pytest.skip("grep command not available")

def test_requirements_txt():
    """Test that requirements.txt is valid"""
    backend_dir = project_root / "backend"
    req_file = backend_dir / "requirements.txt"
    
    assert req_file.exists(), "requirements.txt does not exist"
    
    # Read requirements and check for obvious issues
    with open(req_file, 'r') as f:
        content = f.read()
    
    # Check that there are no merge conflict markers
    assert "<<<<<<<" not in content, "Merge conflict markers found in requirements.txt"
    assert "=======" not in content, "Merge conflict markers found in requirements.txt"
    assert ">>>>>>>" not in content, "Merge conflict markers found in requirements.txt"
    
    # Check that there are actual requirements
    assert len(content.strip()) > 0, "requirements.txt is empty"
    
    # Check for some essential packages
    essential_packages = ["fastapi", "uvicorn", "pydantic"]
    for package in essential_packages:
        assert package in content, f"Essential package {package} not found in requirements.txt"

def test_no_duplicate_main_files():
    """Test that there are no duplicate main files"""
    backend_dir = project_root / "backend"
    
    # Count main.py files
    main_files = list(backend_dir.glob("main*.py"))
    
    # Should only have one main.py file
    main_py_files = [f for f in main_files if f.name == "main.py"]
    assert len(main_py_files) == 1, f"Expected 1 main.py file, found {len(main_py_files)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])