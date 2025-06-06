# 使用官方的 Python 3.10 slim 映像檔作為基礎
# slim 版本的映像檔較小，適合生產環境
FROM python:3.10-slim-bookworm

# 設定容器內的工作目錄為 /app
WORKDIR /app

# 將本地的 requirements.txt 檔案複製到容器的 /app 目錄
# 這一層應單獨進行，這樣當 requirements.txt 檔案沒有變化時，
# Docker 可以利用構建緩存，避免重新安裝所有依賴
COPY requirements.txt .

# 升級 pip 並安裝 requirements.txt 中列出的所有 Python 依賴
# --no-cache-dir 選項可以避免 pip 使用緩存，確保每次都下載並安裝最新的依賴包，
# 這有助於解決可能由緩存引起的版本不一致問題
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 將當前本地目錄下除了 .dockerignore 中指定的檔案外的所有檔案，
# 複製到容器的工作目錄 /app
# 這一層應在安裝依賴之後執行，以便可以緩存依賴安裝層
COPY . .

# 定義容器啟動時要執行的預設命令
# 在此案例中，它將運行位於 src/ 目錄下的 main.py 腳本
CMD ["python", "src/main.py"]