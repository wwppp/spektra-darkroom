"""SpektraDarkroom Standalone Portable Packaging Utility
Packages the complete application with an isolated, standalone portable Python runtime
into D:\\app\\spektra-darkroom so it can run immediately on other Windows machines
without requiring Python installation.
"""

import os
import sys
import shutil
import time
import subprocess
import concurrent.futures

SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
DEST_DIR = r"D:\app\spektra-darkroom"
PYTHON_BASE = r"C:\Users\wp\AppData\Local\Programs\Python\Python312"
VENV_SITE_PACKAGES = os.path.join(SOURCE_DIR, ".venv", "Lib", "site-packages")


def copy_file_task(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def fast_copy_tree(src_dir, dst_dir, ignore_dirs=None, ignore_exts=None, max_workers=16):
    ignore_dirs = set(ignore_dirs or [])
    ignore_exts = set(ignore_exts or [])
    tasks = []

    for root, dirs, files in os.walk(src_dir):
        # In-place filter directories to skip traversing
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]

        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path) if rel_path != "." else dst_dir

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ignore_exts:
                continue
            src_file = os.path.join(root, f)
            dst_file = os.path.join(target_root, f)
            tasks.append((src_file, dst_file))

    print(f"  -> 收集到 {len(tasks)} 个待拷贝文件，使用 {max_workers} 线程高速并发写入...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_file_task, s, d) for s, d in tasks]
        concurrent.futures.wait(futures)


def main():
    t0 = time.time()
    print("=" * 65)
    print(">>> SpektraDarkroom 独立便携免安装完整依赖打包系统 <<<")
    print(f"源工程目录: {SOURCE_DIR}")
    print(f"输出发布目录: {DEST_DIR}")
    print("=" * 65)

    if not os.path.exists(PYTHON_BASE):
        print(f"错误: 找不到系统基础 Python 目录: {PYTHON_BASE}")
        sys.exit(1)

    if not os.path.exists(VENV_SITE_PACKAGES):
        print(f"错误: 找不到虚拟环境依赖库目录: {VENV_SITE_PACKAGES}")
        sys.exit(1)

    # 1. 准备目标根目录
    if os.path.exists(DEST_DIR):
        print(f"[1/6] 正在清理旧的目标发布目录: {DEST_DIR}...")
        try:
            shutil.rmtree(DEST_DIR)
        except Exception as e:
            print(f"警告: 无法完全清除目录，尝试覆盖更新: {e}")
    os.makedirs(DEST_DIR, exist_ok=True)

    # 2. 复制应用程序主代码及资源文件
    print("[2/6] 正在复制主程序文件与资源 (ui, resources, profiles, launcher)...")
    app_files = [
        "SpektraDarkroom.exe",
        "main.py",
        "app_core.py",
        "config_manager.py",
        "session_cache_manager.py",
        "path_utils.py",
        "sdc_manager.py",
        "version.py",
        "CHANGELOG.md",
        "DEVELOPMENT_SUMMARY.md"
    ]
    for af in app_files:
        src = os.path.join(SOURCE_DIR, af)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DEST_DIR, af))

    # 复制 ui 目录
    fast_copy_tree(
        os.path.join(SOURCE_DIR, "ui"),
        os.path.join(DEST_DIR, "ui"),
        ignore_dirs={"__pycache__"},
        ignore_exts={".pyc"}
    )

    # 复制 resources 目录
    fast_copy_tree(
        os.path.join(SOURCE_DIR, "resources"),
        os.path.join(DEST_DIR, "resources"),
        ignore_dirs={".lut_cache", "__pycache__"},
        ignore_exts={".pyc"}
    )

    # 3. 部署自包含便携式 Python 核心运行时 (runtime)
    runtime_dir = os.path.join(DEST_DIR, "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    print(f"[3/6] 正在提取并构建完全独立便携式 Python 3.12 核心运行时 -> {runtime_dir}...")

    # 根二进制和核心 dll
    core_binaries = [
        "python.exe",
        "pythonw.exe",
        "python3.dll",
        "python312.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll"
    ]
    for cb in core_binaries:
        src = os.path.join(PYTHON_BASE, cb)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(runtime_dir, cb))

    # DLLs 动态库目录 (含有 _socket, _ssl, _ctypes, pyexpat 等核心 C 扩展)
    print("  -> 部署 Python 标准 C 扩展核心动态库 (DLLs)...")
    fast_copy_tree(
        os.path.join(PYTHON_BASE, "DLLs"),
        os.path.join(runtime_dir, "DLLs")
    )

    # Lib 标准库目录 (排除无用 test 测试目录和旧 site-packages)
    print("  -> 部署 Python 纯 Python 标准库 (Lib)...")
    fast_copy_tree(
        os.path.join(PYTHON_BASE, "Lib"),
        os.path.join(runtime_dir, "Lib"),
        ignore_dirs={"test", "tests", "site-packages", "__pycache__", "idlelib", "turtledemo"},
        ignore_exts={".pyc"}
    )

    # 4. 部署所有第三方依赖库 (PySide6, rawpy, spektrafilm, cv2, numpy, PIL 等)
    print("[4/6] 正在部署全量第三方依赖库 (包含 PySide6, rawpy, spektrafilm 等全部组件)...")
    runtime_site_packages = os.path.join(runtime_dir, "Lib", "site-packages")
    os.makedirs(runtime_site_packages, exist_ok=True)
    fast_copy_tree(
        VENV_SITE_PACKAGES,
        runtime_site_packages,
        ignore_dirs={"__pycache__", "pip", "pip-24.3.1.dist-info"},
        ignore_exts={".pyc"}
    )

    # 确保 runtime 根目录没有任何绝对路径的 pyvenv.cfg
    # 当没有 pyvenv.cfg 时，Python 会自动以自身 runtime 目录作为 sys.prefix，
    # 自动定位同级的 DLLs、Lib、Lib/site-packages，实现 100% 独立便携运行！
    bad_cfg = os.path.join(runtime_dir, "pyvenv.cfg")
    if os.path.exists(bad_cfg):
        os.remove(bad_cfg)

    # 5. 生成便捷启动脚本与文档
    print("[5/6] 正在生成便捷启动批处理与用户说明文件...")

    # 启动快捷批处理
    bat_content = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" "%~dp0SpektraDarkroom.exe" %*
"""
    with open(os.path.join(DEST_DIR, "启动 Spektra 暗房.bat"), "w", encoding="utf-8") as f:
        f.write(bat_content)

    # 调试启动批处理
    debug_bat = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo  SpektraDarkroom 控制台调试启动器
echo ========================================================
echo 正在启动 SpektraDarkroom 并捕获控制台日志...
echo.
"%~dp0runtime\\python.exe" "%~dp0main.py" %*
echo.
echo 程序已退出。
pause
"""
    with open(os.path.join(DEST_DIR, "控制台调试启动.bat"), "w", encoding="utf-8") as f:
        f.write(debug_bat)

    # 软件说明文档
    readme_content = """========================================================================
   SpektraDarkroom - Vulkan / GPU 硬件加速胶片全流程暗房工作站
   便携免安装完整独立发布版 (Portable Standalone Edition)
========================================================================

【关于本程序】
本版本为全内置自包含绿色免安装版，内置专用的 Python 3.12 独立运行时与全套
依赖项（PySide6, Vulkan/OpenGL, spektrafilm, rawpy, OpenCV, NumPy 等）。
无需目标电脑安装任何 Python 环境或第三方库，解压或复制到任意目录即可运行。

【运行方式】
1. 双击运行：【SpektraDarkroom.exe】（推荐，带 Windows 原生毛玻璃开屏加载动画）
2. 脚本运行：双击【启动 Spektra 暗房.bat】
3. 调试运行：双击【控制台调试启动.bat】（可查看实时终端输出与 GPU 状态）

【系统要求】
- 操作系统：Windows 10 / Windows 11 (64位)
- 内存建议：8GB 以上（推荐 16GB 或以上处理高分辨率 RAW）
- 显卡要求：支持 OpenGL 3.3 / Vulkan 硬件加速的独立显卡或现代核显

【快捷键指南】
- 空格键 (Space) / 鼠标双击：切换 1:1 实际像素 (100%) 与全屏自适应缩放
- 鼠标滚轮：以光标为中心自由缩放视口
- 鼠标左键按住拖拽：平移视口底片
- 字母键 C：开启 / 关闭 胶卷相纸 50:50 实时分屏对比
- 字母键 \\ (反斜杠)：按住即时预览数码传感器原生色彩（松开恢复胶片效果）
- Ctrl + Z / Ctrl + Y：撤销 / 重做 调色操作
- Ctrl + O：打开并导入单张或多张 RAW / 图片底片
- Ctrl + Shift + O：批量导入文件夹底片
- Ctrl + E：打开高保真暗房批量导出窗口
- Ctrl + Q：查看后台导出队列管理器

========================================================================
"""
    with open(os.path.join(DEST_DIR, "使用说明.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 6. 独立便携运行验证
    print("[6/6] 正在对目标发布包进行自包含便携运行隔离验证...")
    test_cmd = [
        os.path.join(runtime_dir, "python.exe"),
        "-c",
        "import sys, os; sys.path.insert(0, r'" + DEST_DIR + "'); "
        "import PySide6, cv2, numpy, rawpy, spektrafilm, PIL; "
        "from app_core import SpektraEngine; "
        "eng = SpektraEngine(resources_dir=r'" + os.path.join(DEST_DIR, "resources") + "'); "
        "print('PORTABLE_VERIFY_SUCCESS: Engine profiles loaded:', len(eng.get_film_stocks()))"
    ]
    # 清空环境变量中的系统 Python 路径，确保 100% 依靠 runtime 自身
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)

    proc = subprocess.run(test_cmd, capture_output=True, text=True, env=clean_env)
    print("目标独立运行时自检输出:")
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print("错误: 目标运行时隔离验证失败!")
        print(proc.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    # 统计打包大小
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(DEST_DIR)
        for filename in filenames
    )
    print("=" * 65)
    print(f" 打包构建圆满成功! 耗时: {elapsed:.2f} 秒")
    print(f" 输出目录: {DEST_DIR}")
    print(f" 总文件大小: {total_size / (1024 * 1024):.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
