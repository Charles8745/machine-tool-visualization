# 在 NVIDIA DGX Spark 上安裝與執行

本文件說明如何在 **DGX Spark（GB10 Blackwell / sm_121 / aarch64 / CUDA 13）** 上，
從 GitHub clone 本專案並建立可用 GPU 的執行環境。

> 為什麼需要專用步驟？
> Blackwell GB10 只有 **CUDA 13 + PyTorch 2.9↑** 才支援，而 PyTorch 2.9 已不再提供
> Python 3.9 的 wheel。因此 DGX Spark 上使用 **Python 3.11**，並把 `numpy / pandas /
> scipy / matplotlib` 等舊釘版一併升級（詳見 `requirements-dgx.txt` 註解）。
> 原本 Windows/x86 開發環境請繼續用 `environment.yml` / `requirements.txt`。

---

## 一、前置需求

- DGX Spark，已開機並可上網
- 已安裝 NVIDIA 驅動與 **CUDA 13**（DGX OS 出廠內建，確認：`nvidia-smi`）
- `git`（`sudo apt install -y git`）

---

## 二、快速安裝（三步）

```bash
git clone https://github.com/Charles8745/machine-tool-visualization.git
cd machine-tool-visualization
chmod +x setup_dgx_spark.sh && ./setup_dgx_spark.sh
```

`setup_dgx_spark.sh` 會依序：

1. 安裝系統套件（Qt/OpenGL/音訊/編譯工具）
2. 沒有 conda 就自動裝 Miniforge（aarch64）
3. 建立 conda 環境 `robot_env`（Python 3.11）
4. 從 cu130 index 裝 GPU 版 PyTorch
5. 裝 `requirements-dgx.txt` 內的相依套件
6. 盡力安裝 `pyrealsense2`（失敗不影響其他套件）
7. 印出驗證結果（列出成功 import 的套件與 `torch.cuda.is_available()`）

裝完啟用環境：

```bash
conda activate robot_env
```

---

## 三、安裝後仍需手動處理的外部相依

這些不是 pip 套件，`setup` 腳本不會幫你裝：

### 1. MySQL 資料庫（GUI 第一/二頁會用到）
程式碼寫死連 `localhost`、`root` / `123456`（見 `0506_main.py` 開頭）。DGX Spark 上需自備：
```bash
sudo apt install -y mysql-server
sudo systemctl enable --now mysql
sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';"
```
> 若不想用預設弱密碼，請改程式碼中的連線設定。

### 2. Ollama（LLM / 語音助理功能）
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &        # 背景啟動
ollama pull <你用的模型>
```

### 3. RealSense 相機（若 `pyrealsense2` 安裝失敗才需要）
aarch64 常無現成 wheel，需從原始碼編譯 librealsense 及其 Python 綁定：
```bash
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python)
make -j"$(nproc)" && sudo make install
```
編譯後把產生的 `pyrealsense2` 綁定加入 `PYTHONPATH`，或直接複製到環境的 `site-packages`。

---

## 四、硬體連線設定（實機測試前務必確認）

程式碼中寫死的裝置 IP，需與 DGX Spark 所在網段一致：

| 裝置 | IP（程式碼預設） | 檔案 |
|------|------------------|------|
| UR 機械手臂 | `192.168.50.114` | `robot_ur.py` |
| MIR100 移動平台 | `192.168.50.26` | `mir100.py` / `mir_ur5.py` / `tcp_send.py` |
| 腕部相機 | `192.168.0.102` | `wrist_camera.py` |
| TCP 影像/資料 | `192.168.50.46` / `192.168.0.103` | `yolo_detect_pc.py` / `0506_main.py` |

> DGX Spark 需與這些裝置在同一區網（`192.168.50.x` / `192.168.0.x`），
> 或修改上述檔案內的 IP。用 `ping <ip>` 逐一確認連得到。

---

## 五、啟動介面

```bash
conda activate robot_env
python 0506_main.py      # 較新版本
# 或
python 0203_main.py
```

---

## 六、常見問題排除

**`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`**
系統缺 xcb 相依。重跑腳本的系統套件段，或手動：
```bash
sudo apt install -y libxcb-cursor0 libxcb-xinerama0 libxkbcommon0 libgl1
```
若是純 SSH 無顯示器，需 `export DISPLAY=:0`（本機有桌面）或用 X11 forwarding / VNC。

**`torch.cuda.is_available()` 回傳 `False`**
確認 `nvidia-smi` 正常、CUDA 13 已裝；重裝 torch：
```bash
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

**警告：`Found GPU0 NVIDIA GB10 ... cuda capability 12.1 ... Maximum ... 12.0`**
這是 Blackwell sm_121 的已知提示，**可安全忽略**，不影響運算。

**載入 `.pkl` / `.pt` / PPO checkpoint 出錯**
本專案已把 `scikit-learn==1.6.1`、`ultralytics==8.3.121`、`stable-baselines3==2.4.0`
鎖死以維持與模型檔相容；請勿升級這三個。

---

## 檔案對照

| 檔案 | 用途 |
|------|------|
| `setup_dgx_spark.sh` | DGX Spark 一鍵安裝腳本 |
| `requirements-dgx.txt` | aarch64 / Python 3.11 相依清單 |
| `requirements.txt` / `environment.yml` | 原 Windows/x86 + Python 3.9 環境（勿在 DGX Spark 用） |
