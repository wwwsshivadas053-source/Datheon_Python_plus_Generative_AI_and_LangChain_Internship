import sys
import os
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from web.app import app
from core.config import settings

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port or 5000))
    host = os.environ.get("HOST", settings.host or "127.0.0.1")
    print(f"\n==========================================================================")
    print(f">> STRUCTURED LLM PROMPT PIPELINE WEB DASHBOARD RUNNING")
    print(f">> URL: http://{host}:{port}")
    print(f">> Default Provider: {settings.default_provider.upper()}")
    print(f"==========================================================================\n")
    app.run(host=host, port=port, debug=False)
