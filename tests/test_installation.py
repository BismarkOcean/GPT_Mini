"""验证 src 布局下的真实安装，不为未来模型功能提供实现答案。"""

import os
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_installed_namespaces_import_outside_repository(tmp_path):
    """从空工作目录导入，避免误将当前目录/PYTHONPATH 当成正确安装。"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import gpt_mini; import gpt_mini.data; import gpt_mini.model; "
            "import gpt_mini.training; import gpt_mini.evaluation; "
            "import gpt_mini.generation",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
