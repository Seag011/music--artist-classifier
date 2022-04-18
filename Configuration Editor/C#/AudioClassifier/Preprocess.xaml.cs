/*
 * Copyright(c) 2021, Pavel Alexeev, pavlik3312@gmail.com
 * All rights reserved.
 * 
 * This source code is licensed under the CC BY-NC-SA 4.0 license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Threading;
using System.Diagnostics;
using System.IO;
using Timer = System.Timers.Timer;

namespace AudioClassifier
{
    /// <summary>
    /// Interaction logic for Preprocess.xaml
    /// </summary>
    /// 

    public static class StartMode
    {
        public const string PROCESS = "process";
        public const string TRAIN = "train";
        public const string WORK = "work";
        public const string ALL = "all";


    }
   
    public partial class Preprocess : Window
    {
        
        private string main_config_path = @"J:\Jupyter\Jupyter\main_config.py";
        private string python_command = "py";

        private Process process;
        private int process_started_name;
        string mode;
        string config_path;


        private double progress;
        public double CurrentProgress {
            get { return progress; }
            set
            {
                if (value > progress)
                    progress = value;
            }
        }        

        public Preprocess(string config_path, string mode)
        {
            InitializeComponent();
            this.config_path = config_path;
            this.mode = mode;
            //sr = ExecuteCommandLine(@"G:\Program Files (x86)\VSsaves\AudioClassifier\AudioClassifier\1.bat");

            main_config_path = Directory.GetCurrentDirectory();


        }
        public void ExecuteCommandLine(String file, String arguments = "")
        {
            ProcessStartInfo startInfo = new ProcessStartInfo();

            //Задаем значение, указывающее необходимость запускать
            //процесс в новом окне.
            //startInfo.CreateNoWindow = false;

            startInfo.WindowStyle = ProcessWindowStyle.Normal; 
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;

            // start command
            startInfo.FileName = "py";// file + " /c " + arguments;

            // arguments
            startInfo.Arguments = arguments;//"\"J:\\Jupyter\\Jupyter\\main_config.py\" -i \"I:\\Downloads\\spectrograms\\new_try\\bards.ini\" -m prepare"; //arguments;

            process = Process.Start(startInfo);

            process_started_name = process.Id;

            process.OutputDataReceived += OnDataRecieved;
            process.ErrorDataReceived += OnDataRecieved;
            process.Exited += OnProcessExit;
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            process.Exited += Process_Exited;
        }

        private void OnDataRecieved(object sender, DataReceivedEventArgs e)
        {
            if (!String.IsNullOrEmpty(e.Data))
            {
                //output.Append(e.Data);
                Dispatcher.Invoke((Action)(() =>
                {
                    Output.Text = Output.Text + "\n" + e.Data;
                    Output.ScrollToEnd();
                }));
            }
        }

        private void OnProcessExit(object sender, EventArgs e)
        {
            progressBar.Value = CurrentProgress;
            //MessageBox.Show
        }
        private  void Process_Exited(object sender, EventArgs e)
        {
            progressBar.Value = 100;

            MessageBox.Show("Работа завершена!", "Выполнено!", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void Process_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {
            this.Dispatcher.Invoke((Action)(() =>
            {
                Output.Text = Output.Text + "\n" + e.Data;
                Output.ScrollToEnd();
                CurrentProgress = 50;
                progressBar.Value = 50;
            }));
        }

        private bool isProceesWork()
        {
            // TODO
            try
            {
                var temp = process.Id;
            }
            catch
            {
                return false;
            }
            return true;
        }
        private void startProcess()
        {
            if (isProceesWork())
            {
                
            }
        }
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            string args = "\"" + main_config_path + "\"" + " -i \"" + config_path + "\" -m " + mode;
            //string args = "\"J:\\Jupyter\\Jupyter\\main_config.py\" -i \"I:\\Downloads\\spectrograms\\new_try\\bards.ini\" -m prepare";
            ExecuteCommandLine(python_command, args);
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            process.Kill();
            process.Close();
            Thread.Sleep(1000);
        }

        private void Output_TextChanged(object sender, TextChangedEventArgs e)
        {
            switch (mode)
            {
                case StartMode.PROCESS:
                    string f = "In derectories found:";
                    f.LastIndexOf(f);
                    
                    break;

                case StartMode.TRAIN:

                    break;

                case StartMode.WORK:

                    break;

                case StartMode.ALL:

                    break;
            }
            progressBar.Value = 75;
        }
    }
}
