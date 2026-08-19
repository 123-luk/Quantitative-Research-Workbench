using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("Quant Research Workbench")]
[assembly: System.Reflection.AssemblyDescription("One-click launcher for quant-factor-system")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

internal static class Program
{
    private static string appUrl = "http://127.0.0.1:8501";
    private static string healthUrl = appUrl + "/_stcore/health";
    private static int streamlitPort = 8501;
    private const string ExpectedRoot = @"E:\FINANCIAL ENGINEERING\quant-factor-system";
    private const string ExpectedPython = @"E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe";

    [STAThread]
    private static int Main(string[] args)
    {
        bool smoke = HasArgument(args, "--smoke-test");
        if (smoke)
        {
            streamlitPort = 18501;
            appUrl = "http://127.0.0.1:" + streamlitPort;
            healthUrl = appUrl + "/_stcore/health";
        }
        bool ownsMutex = false;
        using (Mutex mutex = new Mutex(false, @"Local\QuantFactorSystemLauncher"))
        {
            try { ownsMutex = mutex.WaitOne(0, false); }
            catch (AbandonedMutexException) { ownsMutex = true; }

            if (!ownsMutex)
            {
                if (!WaitForServer(null, TimeSpan.FromSeconds(75)))
                    return Fail("Another launcher is running, but the app did not become ready.", smoke);
                if (!smoke) OpenBrowser();
                return 0;
            }

            try
            {
                if (IsHealthy())
                {
                    if (!smoke) OpenBrowser();
                    return 0;
                }

                string root = FindProjectRoot();
                if (root == null)
                    return Fail("Cannot find app\\streamlit_app.py.\n\nKeep this EXE in the project folder or restore:\n" + ExpectedRoot, smoke);

                string python = FindPython(root);
                if (python == null)
                    return Fail("Cannot find the project Python environment.\n\nExpected:\n" + ExpectedPython, smoke);

                Process streamlit = StartStreamlit(python, root);
                if (!WaitForServer(streamlit, TimeSpan.FromSeconds(75)))
                {
                    StopProcess(streamlit);
                    return Fail("Streamlit did not become ready within 75 seconds.\n\nRun run_app.ps1 for console details.", smoke);
                }

                if (smoke) StopProcess(streamlit);
                else OpenBrowser();
                return 0;
            }
            catch (Exception exception)
            {
                return Fail("Unable to start Quant Research Workbench.\n\n" + exception.Message, smoke);
            }
            finally { mutex.ReleaseMutex(); }
        }
    }

    private static bool HasArgument(string[] args, string expected)
    {
        foreach (string argument in args)
            if (string.Equals(argument, expected, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    private static string FindProjectRoot()
    {
        string executableRoot = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
        if (File.Exists(Path.Combine(executableRoot, "app", "streamlit_app.py"))) return executableRoot;
        if (File.Exists(Path.Combine(ExpectedRoot, "app", "streamlit_app.py"))) return ExpectedRoot;
        return null;
    }

    private static string FindPython(string root)
    {
        string parent = Directory.GetParent(root).FullName;
        string[] candidates = {
            Path.Combine(root, ".venv", "Scripts", "python.exe"),
            Path.Combine(parent, ".venv", "quant-factor-system", "Scripts", "python.exe"),
            ExpectedPython
        };
        foreach (string candidate in candidates)
            if (File.Exists(candidate)) return candidate;
        return null;
    }

    private static Process StartStreamlit(string python, string root)
    {
        return Process.Start(new ProcessStartInfo {
            FileName = python,
            Arguments = "-m streamlit run \"app\\streamlit_app.py\" --server.headless true --server.address 127.0.0.1 --server.port " + streamlitPort + " --browser.gatherUsageStats false",
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden
        });
    }

    private static bool WaitForServer(Process process, TimeSpan timeout)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (clock.Elapsed < timeout)
        {
            if (IsHealthy()) return true;
            if (process != null && process.HasExited) return false;
            Thread.Sleep(250);
        }
        return false;
    }

    private static bool IsHealthy()
    {
        try
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(healthUrl);
            request.Method = "GET";
            request.Timeout = 750;
            request.ReadWriteTimeout = 750;
            request.Proxy = null;
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            using (StreamReader reader = new StreamReader(response.GetResponseStream()))
                return response.StatusCode == HttpStatusCode.OK && reader.ReadToEnd().Trim() == "ok";
        }
        catch (WebException) { return false; }
    }

    private static void OpenBrowser()
    {
        Process.Start(new ProcessStartInfo { FileName = appUrl, UseShellExecute = true });
    }

    private static void StopProcess(Process process)
    {
        if (process == null || process.HasExited) return;
        process.Kill();
        process.WaitForExit(5000);
    }

    private static int Fail(string message, bool smoke)
    {
        if (!smoke) MessageBox.Show(message, "Quant Research Workbench", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
