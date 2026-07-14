#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_dgx_spark.sh
# 工具機可視化介面 — NVIDIA DGX Spark 一鍵環境安裝
#
# 目標平台：DGX Spark (GB10 Blackwell, sm_121, aarch64) + CUDA 13
#           conda 環境 robot_env / Python 3.11 / GPU 版 PyTorch (cu130)
#
# 用法：
#   chmod +x setup_dgx_spark.sh
#   ./setup_dgx_spark.sh
#
# 注意：本腳本刻意「不」用 set -e，讓個別套件失敗不會中斷整體；
#       最後的驗證區塊會列出實際成功 import 的套件，以那個為準。
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ENV_NAME="robot_env"
PY_VER="3.11"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m  %s\n' "$*"; }
err()  { printf '\n\033[1;31m[error]\033[0m %s\n' "$*"; }

# 0. 架構檢查 ──────────────────────────────────────────────────────────────────
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
  warn "偵測到架構為 '$ARCH'，非 aarch64。本腳本針對 DGX Spark (aarch64) 設計，繼續執行風險自負。"
fi

# 1. 系統套件（Qt / OpenGL / 音訊 / 編譯工具）──────────────────────────────────
log "安裝系統相依套件（需要 sudo）..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential cmake git curl ca-certificates \
    libgl1 libglib2.0-0 libegl1 libxkbcommon0 libdbus-1-3 \
    libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    portaudio19-dev libsm6 libxext6 \
    || warn "部分系統套件安裝失敗；若之後 GUI（Qt xcb）或音訊有問題再回頭補裝。"
else
  warn "找不到 apt-get，略過系統套件安裝，請自行確認 Qt / OpenGL 相依已就緒。"
fi

# 2. conda（Miniforge）─────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
  log "未偵測到 conda，安裝 Miniforge (aarch64)..."
  MF="Miniforge3-Linux-aarch64.sh"
  curl -fsSL -o "/tmp/$MF" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/$MF" \
    || { err "下載 Miniforge 失敗，請檢查網路後重試。"; exit 1; }
  bash "/tmp/$MF" -b -p "$HOME/miniforge3"
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# 3. 建立 / 沿用 conda 環境 ────────────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda 環境 '$ENV_NAME' 已存在，沿用之。"
else
  log "建立 conda 環境 '$ENV_NAME'（Python $PY_VER）..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER" \
    || { err "建立 conda 環境失敗。"; exit 1; }
fi
conda activate "$ENV_NAME" || { err "無法 activate '$ENV_NAME'。"; exit 1; }

python -m pip install -U pip wheel setuptools

# 4. PyTorch（Blackwell / CUDA 13）─────────────────────────────────────────────
log "安裝 GPU 版 PyTorch（cu130, aarch64）..."
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" \
  || warn "PyTorch 安裝失敗，GPU 功能將不可用。請確認 DGX Spark 驅動 / CUDA 13 已就緒。"

# 5. 專案相依（torch 已滿足，requirements-dgx.txt 內不含 torch）──────────────────
log "安裝專案相依套件（requirements-dgx.txt）..."
pip install -r "$SCRIPT_DIR/requirements-dgx.txt" \
  || warn "部分相依套件安裝失敗，請看最後的驗證結果確認缺哪一個。"

# 6. RealSense 相機（盡力而為；aarch64 常無現成 wheel）──────────────────────────
log "嘗試安裝 pyrealsense2（RealSense 相機）..."
if pip install "pyrealsense2==2.55.1.6486" 2>/dev/null || pip install pyrealsense2 2>/dev/null; then
  log "pyrealsense2 安裝成功。"
else
  warn "pyrealsense2 無 aarch64 wheel，未安裝。若需 RealSense 相機，"
  warn "請從原始碼編譯 librealsense（含 Python 綁定）——步驟見 README_DGX_SPARK.md。"
fi

# 7. 驗證 ──────────────────────────────────────────────────────────────────────
log "驗證安裝結果..."
python - <<'PY'
import importlib, sys
print("Python :", sys.version.split()[0])
try:
    import torch
    ok = torch.cuda.is_available()
    print("torch  :", torch.__version__,
          "| CUDA available:", ok,
          "| device:", (torch.cuda.get_device_name(0) if ok else "CPU only"))
    if not ok:
        print("  ⚠️  CUDA 不可用 — GPU 加速失效，請檢查驅動 / CUDA 13。")
except Exception as e:
    print("torch  : import 失敗 —", e)

print("-" * 60)
mods = ["numpy","cv2","ultralytics","stable_baselines3","sklearn",
        "xgboost","PyQt5","pandas","matplotlib","gymnasium","urx",
        "langchain","ollama","mysql.connector","speech_recognition"]
for m in mods:
    try:
        importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e:
        print(f"  XX  {m}  —  {e}")
try:
    import pyrealsense2  # noqa
    print("  ok  pyrealsense2")
except Exception:
    print("  --  pyrealsense2（未安裝，RealSense 相機功能不可用）")
PY

log "完成。日後啟用環境：  conda activate $ENV_NAME"
log "啟動介面（擇一）：     python 0506_main.py   或   python 0203_main.py"
