import ast
import os
import sys
import importlib.metadata
from stdlib_list import stdlib_list

# 建立常見的 程式碼 import 名稱 -> PyPI 套件名稱的對照表
PACKAGE_MAPPING = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'stable_baselines3': 'stable-baselines3',
    'gymnasium': 'gymnasium',
    'speech_recognition': 'SpeechRecognition',  # 修正語音辨識套件名稱
    'mysql': 'mysql-connector-python',           # 修正資料庫套件名稱
    # 'matlab': 'matlab'
}

def get_imports_from_file(filepath):
    """讀取 Python 檔案並解析出所有 import 的根模組"""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"讀取 {filepath} 時發生錯誤: {e}")
    
    return imports

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_imports = set()
    
    # 1. 取得 Python 內建標準庫清單
    try:
        standard_libs = set(stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
    except:
        standard_libs = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
    
    # 2. 自動抓取當前資料夾下的所有自寫檔案 (去除 .py 副檔名)
    # 這樣腳本就不會再把你寫的 rl, sidebar1 等檔案當成要下載的套件了！
    local_modules = {filename[:-3] for filename in os.listdir(current_dir) if filename.endswith(".py")}
    
    print("🔍 開始掃描 Python 檔案...")
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename != os.path.basename(__file__):
            filepath = os.path.join(current_dir, filename)
            imports = get_imports_from_file(filepath)
            all_imports.update(imports)

    # 3. 過濾掉「內建模組」與「你自己寫的檔案」
    third_party_imports = all_imports - standard_libs - local_modules - {'__future__'}
    
    requirements = []
    print("\n📦 開始檢查當前環境的套件版本...")
    
    for imp in third_party_imports:
        pkg_name = PACKAGE_MAPPING.get(imp, imp)
        
        try:
            version = importlib.metadata.version(pkg_name)
            requirements.append(f"{pkg_name}=={version}")
            print(f"  ✅ 找到套件: {pkg_name} (版本 {version})")
        except importlib.metadata.PackageNotFoundError:
            requirements.append(pkg_name)
            print(f"  ⚠️ 未安裝/無法辨識版本: {pkg_name} (將以最新版代替)")

    req_file_path = os.path.join(current_dir, "requirements.txt")
    with open(req_file_path, "w", encoding="utf-8") as f:
        for req in sorted(requirements):
            f.write(req + "\n")
            
    print(f"\n🎉 完美！已生成乾淨的依賴清單: {req_file_path}")

if __name__ == "__main__":
    main()