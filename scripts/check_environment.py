"""仅诊断开发环境；不包含训练、模型或 CUDA kernel 实现。"""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    errors = []
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    if sys.version_info < (3, 11):  # noqa: UP036 - standalone environment probe
        errors.append("Python 3.11+ is required")

    for name in ("gpt-mini", "pytest", "ruff", "build"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = "not installed"

    report["torch"] = "not installed"
    report["cuda_available"] = False
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            report["torch"] = str(torch.__version__)
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                report["cuda_devices"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except (ImportError, OSError, RuntimeError) as error:
            report["torch"] = "import or CUDA query failed"
            errors.append(f"PyTorch diagnostic failed: {error}")

    if (args.require_torch or args.require_cuda) and report["torch"] == "not installed":
        errors.append("Install a compatible PyTorch wheel before this check")
    if args.require_cuda and not report["cuda_available"]:
        errors.append("CUDA is not available to PyTorch")
    report["errors"] = errors
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
