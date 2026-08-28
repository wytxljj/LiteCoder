"""使 `python -m litecoder` 可直接运行。"""
from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
