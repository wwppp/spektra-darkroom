using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;

[assembly: AssemblyTitle("SpektraDarkroom")]
[assembly: AssemblyProduct("SpektraDarkroom")]
[assembly: AssemblyDescription("SpektraDarkroom 物理胶片暗房工作站")]
[assembly: AssemblyCompany("Spektra Darkroom Studio")]
[assembly: AssemblyCopyright("Copyright © 2026 SpektraDarkroom")]
[assembly: AssemblyVersion("0.1.74.75")]
[assembly: AssemblyFileVersion("0.1.74.75")]

namespace SpektraDarkroom
{
    static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [DllImport("shell32.dll", SetLastError = true)]
        private static extern void SetCurrentProcessExplicitAppUserModelID([MarshalAs(UnmanagedType.LPWStr)] string AppID);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("python313.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int Py_Main(int argc, IntPtr[] argv);

        [STAThread]
        static int Main(string[] args)
        {
            // 1. Fast single-instance IPC check
            try
            {
                using (var pipe = new System.IO.Pipes.NamedPipeClientStream(".", "SpektraDarkroom_IPC_SingleInstance", System.IO.Pipes.PipeDirection.Out))
                {
                    pipe.Connect(300);
                    using (var writer = new StreamWriter(pipe, new System.Text.UTF8Encoding(false)))
                    {
                        if (args != null && args.Length > 0)
                        {
                            writer.Write(string.Join("\n", args));
                        }
                        else
                        {
                            writer.Write("\n");
                        }
                        writer.Flush();
                    }
                    return 0;
                }
            }
            catch { }

            // 2. Windows DPI awareness & AppUserModelID
            try { SetProcessDPIAware(); } catch { }
            try { SetCurrentProcessExplicitAppUserModelID("SpektraDarkroom.App.1.0"); } catch { }

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            if (baseDir.TrimEnd('\\', '/').EndsWith("Scripts", StringComparison.OrdinalIgnoreCase))
            {
                baseDir = Path.GetFullPath(Path.Combine(baseDir, "..", ".."));
            }

            string venvDir = Path.Combine(baseDir, ".venv");
            string venvSite = Path.Combine(venvDir, "Lib", "site-packages");

            // 3. Locate python313.dll dynamically
            string dllPath = null;
            string pythonHome = null;

            // 3a. Check runtime directory
            string rtDir = Path.Combine(baseDir, "runtime");
            if (File.Exists(Path.Combine(rtDir, "python313.dll")))
            {
                dllPath = Path.Combine(rtDir, "python313.dll");
                pythonHome = rtDir;
            }

            // 3b. Check .venv/pyvenv.cfg
            if (string.IsNullOrEmpty(dllPath))
            {
                string venvCfg = Path.Combine(venvDir, "pyvenv.cfg");
                if (File.Exists(venvCfg))
                {
                    string[] lines = File.ReadAllLines(venvCfg);
                    foreach (string line in lines)
                    {
                        if (line.Trim().StartsWith("home ="))
                        {
                            string home = line.Substring(line.IndexOf('=') + 1).Trim();
                            string cand = Path.Combine(home, "python313.dll");
                            if (File.Exists(cand))
                            {
                                dllPath = cand;
                                pythonHome = home;
                                break;
                            }
                        }
                    }
                }
            }

            // 3c. Check WindowsApps Python 3.13 packages
            if (string.IsNullOrEmpty(dllPath))
            {
                string progFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
                string winApps = Path.Combine(progFiles, "WindowsApps");
                if (Directory.Exists(winApps))
                {
                    try
                    {
                        foreach (string pkg in Directory.GetDirectories(winApps, "*Python.3.13*"))
                        {
                            string cand = Path.Combine(pkg, "python313.dll");
                            if (File.Exists(cand))
                            {
                                dllPath = cand;
                                pythonHome = pkg;
                                break;
                            }
                        }
                    }
                    catch { }
                }
            }

            if (string.IsNullOrEmpty(dllPath) || !File.Exists(dllPath))
            {
                MessageBox.Show(
                    "找不到 Python 核心运行组件 (python313.dll)。\n请确保已安装 Python 3.13 或已配置完整的 .venv 运行环境。",
                    "SpektraDarkroom",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            string basePrefix = pythonHome ?? Path.GetDirectoryName(dllPath);
            Environment.SetEnvironmentVariable("PYTHONHOME", basePrefix);

            // 4. Construct comprehensive PYTHONPATH for embedded environment
            List<string> paths = new List<string>();
            paths.Add(Path.Combine(basePrefix, "DLLs"));
            paths.Add(Path.Combine(basePrefix, "Lib"));
            paths.Add(basePrefix);
            if (Directory.Exists(venvSite)) paths.Add(venvSite);
            paths.Add(baseDir);

            string pythonPath = string.Join(";", paths.ToArray());
            Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath);
            Environment.SetEnvironmentVariable("PYTHONUNBUFFERED", "1");

            // 5. Load python313.dll
            IntPtr hDll = LoadLibrary(dllPath);
            if (hDll == IntPtr.Zero)
            {
                MessageBox.Show(
                    "加载 python313.dll 失败，Win32 错误代码: " + Marshal.GetLastWin32Error(),
                    "SpektraDarkroom",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            // 6. Execute main.py via Py_Main in native host process
            string mainPy = Path.Combine(baseDir, "main.py");
            if (!File.Exists(mainPy))
            {
                MessageBox.Show(
                    "找不到主程序脚本 main.py。",
                    "SpektraDarkroom",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            List<string> pyArgsList = new List<string>();
            pyArgsList.Add("SpektraDarkroom.exe");
            pyArgsList.Add(mainPy);
            if (args != null)
            {
                foreach (string a in args)
                {
                    if (!string.IsNullOrEmpty(a)) pyArgsList.Add(a);
                }
            }

            IntPtr[] pArgs = new IntPtr[pyArgsList.Count];
            for (int i = 0; i < pyArgsList.Count; i++)
            {
                pArgs[i] = Marshal.StringToHGlobalUni(pyArgsList[i]);
            }

            try
            {
                return Py_Main(pArgs.Length, pArgs);
            }
            finally
            {
                for (int i = 0; i < pArgs.Length; i++)
                {
                    Marshal.FreeHGlobal(pArgs[i]);
                }
            }
        }
    }
}
