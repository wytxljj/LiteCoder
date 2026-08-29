"""配置：从环境变量 / .env 读取；凭据通过未入库的 .env 提供。"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析器（自研，不依赖 python-dotenv）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:  # 环境变量优先，.env 兜底
            os.environ[key] = value


class Config:
    """集中管理所有可配置项，均可被环境变量覆盖。"""

    def __init__(self, root: Path | None = None):
        # 项目根目录：src/litecoder/config.py 的上三级
        self.root = (root or Path(__file__).resolve().parent.parent.parent).resolve()
        _load_dotenv(self.root / ".env")

        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "deepseek-chat")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "120"))
        self.max_steps = int(os.environ.get("MAX_STEPS", "20"))
        self.max_context_messages = int(os.environ.get("MAX_CONTEXT_MESSAGES", "40"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY：请在项目根目录的 .env 中配置（参考 .env.example），"
                "或通过环境变量 LLM_API_KEY 提供。"
            )
