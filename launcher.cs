using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

namespace SpektraDarkroom
{
    static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        static void Main(string[] args)
        {
            // Fast IPC check: if an instance is already running, pass args directly and exit immediately
            try
            {
                using (var pipe = new System.IO.Pipes.NamedPipeClientStream(".", "SpektraDarkroom_IPC_SingleInstance", System.IO.Pipes.PipeDirection.Out))
                {
                    pipe.Connect(1000);
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
                    return;
                }
            }
            catch { }

            try
            {
                SetProcessDPIAware();
            }
            catch { }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonExe = Path.Combine(baseDir, "runtime", "python.exe");
            if (!File.Exists(pythonExe))
            {
                pythonExe = Path.Combine(baseDir, "runtime", "pythonw.exe");
            }
            if (!File.Exists(pythonExe))
            {
                pythonExe = Path.Combine(baseDir, "python", "python.exe");
            }
            if (!File.Exists(pythonExe))
            {
                pythonExe = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
            }
            if (!File.Exists(pythonExe))
            {
                pythonExe = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
            }
            if (!File.Exists(pythonExe))
            {
                MessageBox.Show("找不到 Python 运行环境 (runtime/python.exe 或 .venv/Scripts/python.exe)。请确保软件目录完整。", 
                                "SpektraDarkroom", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            string mainPy = Path.Combine(baseDir, "main.py");
            if (!File.Exists(mainPy))
            {
                MessageBox.Show("找不到主程序脚本 main.py。", "SpektraDarkroom", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            string appVersion = GetAppVersion(baseDir);
            string splashPath = Path.Combine(baseDir, "resources", "splash.png");
            SplashForm splash = new SplashForm(splashPath, appVersion);
            splash.Show();
            Application.DoEvents();

            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pythonExe;
            string extraArgs = "";
            if (args != null && args.Length > 0)
            {
                foreach (string arg in args)
                {
                    if (!string.IsNullOrEmpty(arg))
                    {
                        extraArgs += " \"" + arg.Trim().Replace("\"", "\\\"") + "\"";
                    }
                }
            }
            psi.Arguments = "\"" + mainPy + "\" --launched-from-exe" + extraArgs;
            psi.WorkingDirectory = baseDir;
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.CreateNoWindow = true;

            Process proc;
            try
            {
                proc = Process.Start(psi);
            }
            catch (Exception ex)
            {
                splash.Close();
                MessageBox.Show("启动 Python 进程失败: " + ex.Message, "SpektraDarkroom", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            bool isReady = false;
            Thread outThread = new Thread(() =>
            {
                try
                {
                    string line;
                    while ((line = proc.StandardOutput.ReadLine()) != null)
                    {
                        line = line.Trim();
                        if (line.StartsWith("[SPLASH]"))
                        {
                            string msg = line.Substring(8).Trim();
                            if (!splash.IsDisposed && splash.IsHandleCreated)
                            {
                                splash.BeginInvoke((Action)(() => splash.UpdateStatus(msg)));
                            }
                        }
                        else if (line.StartsWith("[READY]"))
                        {
                            isReady = true;
                            if (!splash.IsDisposed && splash.IsHandleCreated)
                            {
                                splash.BeginInvoke((Action)(() => splash.CloseSplash()));
                            }
                            break;
                        }
                    }
                }
                catch { }
            });
            outThread.IsBackground = true;
            outThread.Start();

            Thread monitorThread = new Thread(() =>
            {
                proc.WaitForExit();
                if (!isReady && proc.ExitCode != 0)
                {
                    string err = proc.StandardError.ReadToEnd();
                    if (!splash.IsDisposed && splash.IsHandleCreated)
                    {
                        splash.BeginInvoke((Action)(() =>
                        {
                            splash.Close();
                            MessageBox.Show("程序启动异常退出 (代码 " + proc.ExitCode + "):\n" + err, 
                                            "SpektraDarkroom 启动错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }));
                    }
                }
                else
                {
                    if (!splash.IsDisposed && splash.IsHandleCreated)
                    {
                        splash.BeginInvoke((Action)(() => splash.CloseSplash()));
                    }
                }
            });
            monitorThread.IsBackground = true;
            monitorThread.Start();

            Application.Run(splash);
        }

        static string GetAppVersion(string baseDir)
        {
            try
            {
                string verFile = Path.Combine(baseDir, "version.py");
                if (File.Exists(verFile))
                {
                    string content = File.ReadAllText(verFile, System.Text.Encoding.UTF8);
                    var mMaj = System.Text.RegularExpressions.Regex.Match(content, @"VERSION_MAJOR\s*=\s*(\d+)");
                    var mMin = System.Text.RegularExpressions.Regex.Match(content, @"VERSION_MINOR\s*=\s*(\d+)");
                    var mPat = System.Text.RegularExpressions.Regex.Match(content, @"VERSION_PATCH\s*=\s*(\d+)");
                    if (mMaj.Success && mMin.Success && mPat.Success)
                    {
                        return mMaj.Groups[1].Value + "." + mMin.Groups[1].Value + "." + mPat.Groups[1].Value;
                    }
                }
            }
            catch { }
            return "0.1.16";
        }
    }

    class SplashForm : Form
    {
        private Image splashImage;
        private string appVersion;
        private string currentStatus = "正在初始化环境与启动暗房内核...";
        private System.Windows.Forms.Timer fadeTimer;
        private float currentOpacity = 1.0f;

        public SplashForm(string imagePath, string version = "")
        {
            this.appVersion = version;
            this.FormBorderStyle = FormBorderStyle.None;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.ShowInTaskbar = false;
            this.TopMost = true;
            this.DoubleBuffered = true;
            this.BackColor = Color.FromArgb(14, 15, 20);

            this.ClientSize = new Size(640, 360);

            if (File.Exists(imagePath))
            {
                try
                {
                    splashImage = Image.FromFile(imagePath);
                }
                catch { }
            }
        }

        public void UpdateStatus(string status)
        {
            this.currentStatus = status;
            this.Invalidate(new Rectangle(0, this.ClientSize.Height - 50, this.ClientSize.Width, 50));
        }

        public void CloseSplash()
        {
            fadeTimer = new System.Windows.Forms.Timer();
            fadeTimer.Interval = 20;
            fadeTimer.Tick += (s, e) =>
            {
                currentOpacity -= 0.15f;
                if (currentOpacity <= 0.05f)
                {
                    fadeTimer.Stop();
                    this.Close();
                }
                else
                {
                    this.Opacity = currentOpacity;
                }
            };
            fadeTimer.Start();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            if (splashImage != null)
            {
                g.DrawImage(splashImage, new Rectangle(0, 0, this.ClientSize.Width, this.ClientSize.Height));
            }

            if (!string.IsNullOrEmpty(appVersion))
            {
                string verText = appVersion.StartsWith("v", StringComparison.OrdinalIgnoreCase) ? appVersion : "v" + appVersion;
                using (Font verFont = new Font("Segoe UI", 8.0f, FontStyle.Bold))
                {
                    SizeF textSize = g.MeasureString(verText, verFont);
                    float padH = 6f;
                    float padV = 2f;
                    float badgeW = textSize.Width + padH * 2;
                    float badgeH = textSize.Height + padV * 2;
                    float badgeX = 450f;
                    float badgeY = 117f;

                    RectangleF badgeRect = new RectangleF(badgeX, badgeY, badgeW, badgeH);
                    using (GraphicsPath path = CreateRoundedRectanglePath(badgeRect, 3.5f))
                    {
                        using (SolidBrush bgBrush = new SolidBrush(Color.FromArgb(170, 20, 22, 28)))
                        {
                            g.FillPath(bgBrush, path);
                        }
                        using (Pen borderPen = new Pen(Color.FromArgb(210, 245, 158, 11), 1.0f))
                        {
                            g.DrawPath(borderPen, path);
                        }
                    }
                    using (SolidBrush textBrush = new SolidBrush(Color.FromArgb(245, 158, 11)))
                    {
                        using (StringFormat sf = new StringFormat() { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
                        {
                            g.DrawString(verText, verFont, textBrush, new RectangleF(badgeX, badgeY + 0.5f, badgeW, badgeH), sf);
                        }
                    }
                }
            }

            int barY = this.ClientSize.Height - 44;
            using (SolidBrush barBrush = new SolidBrush(Color.FromArgb(230, 11, 12, 16)))
            {
                g.FillRectangle(barBrush, 0, barY, this.ClientSize.Width, 44);
            }
            using (Pen borderPen = new Pen(Color.FromArgb(40, 44, 56)))
            {
                g.DrawLine(borderPen, 0, barY, this.ClientSize.Width, barY);
            }

            using (SolidBrush dotBrush = new SolidBrush(Color.FromArgb(245, 158, 11)))
            {
                g.FillEllipse(dotBrush, 28, barY + 18, 7, 7);
            }

            using (Font font = new Font("Microsoft YaHei", 9f, FontStyle.Regular))
            using (SolidBrush textBrush = new SolidBrush(Color.FromArgb(203, 213, 225)))
            {
                g.DrawString(currentStatus, font, textBrush, new PointF(46, barY + 13));
            }
        }

        private static GraphicsPath CreateRoundedRectanglePath(RectangleF rect, float radius)
        {
            GraphicsPath path = new GraphicsPath();
            float d = radius * 2.0f;
            path.AddArc(rect.X, rect.Y, d, d, 180, 90);
            path.AddArc(rect.Right - d, rect.Y, d, d, 270, 90);
            path.AddArc(rect.Right - d, rect.Bottom - d, d, d, 0, 90);
            path.AddArc(rect.X, rect.Bottom - d, d, d, 90, 90);
            path.CloseFigure();
            return path;
        }
    }
}
