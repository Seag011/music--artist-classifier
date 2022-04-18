/*
 * Copyright(c) 2021, Pavel Alexeev, pavlik3312@gmail.com
 * All rights reserved.
 * 
 * This source code is licensed under the CC BY-NC-SA 4.0 license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Configuration;
using IniParser;
using IniParser.Model;
using Microsoft.Win32;
using System.IO;
using Ookii.Dialogs.Wpf;


namespace AudioClassifier
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        const string allowedFormatsDefault = 
        "3gp;aa;aac;aax;act;aiff;alac;amr;ape;au;awb;" +
        "dss;dvf;flac;gsm;iklax;ivs;m4a;m4b;m4p;mmf;" +
        "mp3;mpc;msv;nmf;ogg;oga;mogg;opus;org;ra;rm;" +
        "raw;rf64;tta;voc;vox;wav;wma;wv;webm;8svx;cda";

        ClassificationConfig config = null;
        public ClassificationConfig Config
        { 
            get { return config; }
            set {
                if (value is null)
                    changeMenuAccess(false);
                else
                {
                    changeMenuAccess(true);
                    config = value;
                }
            }
        }

        public bool IsSaved;

        private void changeMenuAccess(bool enable)
        {
            saveConfigMenu.IsEnabled = enable;
            //closeConfigMenu.IsEnabled = enable;
        }

        //public static IEnumerable<TControl> GetChildControls<TControl>(this Control control) where TControl : Control
        //{
        //    var children = (control.Controls != null) ? control.Controls.OfType<TControl>() : Enumerable.Empty<TControl>();
        //    return children.SelectMany(c => GetChildControls<TControl>(c)).Concat(children);
        //}

        public MainWindow()
        {
            InitializeComponent();
            SetListBox(ListFormats, StrToCollection(allowedFormatsDefault));
            setDefaultConfig();
            Config = null;
        }

        public void setDefaultConfig()
        {
            IsSaved = false;
            //config = new ClassificationConfig(path);
            Config = null;
            textBoxDatasetName.Text = "sample_dataset";
            textBoxDataset.Text = "C:/Dataset";
            textBoxTemp.Text = "C:/Dataset Files/Temp";
            textBoxResult.Text = "C:/Dataset Files/Result";
            textBoxClassify.Text = "C:/Dataset Files/Classify";

            sampleRate.Value = 16000;
            sliceLength.Value = 10000;
            layering.Value = 2;

            epoches.Value = 20;
            learningRate.Value = 0.001;
            early_stop.IsChecked = true;
            batchSize.Value = 16;

            //select filters from config
            selectFormats(ListFormats, "mp3;wma;wav;aac;flac");
            IsSaved = false;
        }

        //private void initFormatsList()
        //{
        //    string[] allowedFormats = StrToCollection(allowedFormatsDefault);
            
        //    foreach (var format in allowedFormats)
        //    {
        //        ListFormats.Items.Add(format);
        //    }
        //}
        private void SetListBox(ListBox listBox, string[] str)
        {
            listBox.Items.Clear();
            foreach (var format in str)
            {
                listBox.Items.Add(format);
            }
        }

        private string GetSelectedListBox(ListBox listBox)
        {
            string formats = "";
            foreach (var elem in listBox.Items)
            {
                formats += elem.ToString()+';';
            }
            return formats;
        }

        private string[] StrToCollection(string str_with_delimeters, char delimeter = ';')
        {
            return str_with_delimeters.Split(delimeter);
        }

        private string CollectionToStr(string [] formats, char delimeter = ';')
        {
            return string.Join(delimeter, formats);
        }


        private void choosePath(TextBox box)
        {
            var dialog = new VistaFolderBrowserDialog();
            
            if (dialog.ShowDialog() == true)
            {
                box.Text = dialog.SelectedPath;
            }
        }

        private void generatePaths()
        {
            var dataset_path = textBoxDataset.Text;
            var parent = Directory.GetParent(dataset_path) + @"\";

            textBoxTemp.Text = parent + @"temp";
            textBoxResult.Text = parent + @"results";
            textBoxClassify.Text = parent + @"work";
        }

        private void btnDataset_Click(object sender, RoutedEventArgs e)
        {
            choosePath(textBoxDataset);
            generatePaths();
        }

        private void btnTemp_Click(object sender, RoutedEventArgs e)
        {
            choosePath(textBoxTemp);
        }
        private void btnResult_Click(object sender, RoutedEventArgs e)
        {
            choosePath(textBoxResult);
        }
        private void btnClassify_Click(object sender, RoutedEventArgs e)
        {
            choosePath(textBoxClassify);
        }

        
     
        private bool checkDirectory(string path)
        {
            return Directory.Exists(path);
        }
        private void checkVariables()
        {
            bool paths_check = checkDirectory(textBoxClassify.Text) &&
                   checkDirectory(textBoxDataset.Text) &&
                   checkDirectory(textBoxTemp.Text);
           // bool variables_check = 
        }





        public void loadConfig(string path)
        {
            Config = new ClassificationConfig(path);
            textBoxDatasetName.Text = Config.dataset_name;
            textBoxDataset.Text = Config.dataset;
            textBoxTemp.Text = Config.temp_files;
            textBoxResult.Text = Config.train_results;
            textBoxClassify.Text = Config.work;

            sampleRate.Value = Config.sample_rate;
            sliceLength.Value = Config.slice_length;
            layering.Value = Config.layering;

            epoches.Value = Config.epoches;
            learningRate.Value = Config.learning_rate;
            early_stop.IsChecked = Config.early_stop;
            patience.Value = Config.patience;
            batchSize.Value = Config.batch_size;

            n_fft.Value = Config.n_fft;
            hop_length.Value = Config.hop_length;

            //select filters from Config
            selectFormats(ListFormats, Config.filters);
            IsSaved = false;
        }

        public void saveConfig(string path)
        {            
            if (!(path is null))
                Config = new ClassificationConfig(path, true);

            Config.dataset_name = textBoxDatasetName.Text;
            Config.dataset = textBoxDataset.Text;
            Config.temp_files = textBoxTemp.Text;
            Config.train_results = textBoxResult.Text;
            Config.work = textBoxClassify.Text;

            Config.sample_rate = sampleRate.Value ?? default(int);
            Config.slice_length = sliceLength.Value ?? default(int);
            Config.layering = layering.Value ?? default(int);

            string filters = string.Empty;
            foreach (var format in ListFormats.SelectedItems)
            {
                filters += format + ";";
            }
            Config.filters = filters;

            Config.epoches = epoches.Value ?? default(int);
            Config.learning_rate = learningRate.Value ?? default(double);
            Config.early_stop = early_stop.IsChecked ?? default(bool);
            Config.patience = patience.Value ?? default(int);
            Config.batch_size = batchSize.Value ?? default(int);

            Config.n_fft = n_fft.Value ?? default(int);
            Config.hop_length = hop_length.Value ?? default(int);

            Config.SaveIni();
            IsSaved = true;
        }
        public void saveConfig()
        {
            saveConfig(null);
        }

        private void saveCheck(Action success)
        {
            if (IsSaved)
            {
                success();
            }
            else
            {
                MessageBoxResult result=
                MessageBox.Show("Текущая конфигурация не сохранена. Вы желаете сохранить текущую конфигурацию?",
                    "Нет сохранения!", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning);
                if (result == MessageBoxResult.Yes)
                {
                    saveConfig_Click(this, null);
                    success(); // ERROR: executes even if object doesn saved
                }
                else if (result == MessageBoxResult.No)
                {
                    success();
                }
            }
        }

        private void newConfig_Click(object sender, RoutedEventArgs e)
        {            
            saveCheck( setDefaultConfig );
        }
        private void openConfig_Click(object sender, RoutedEventArgs e)
        {
            saveCheck( setDefaultConfig );

            var dialog = new OpenFileDialog();
            if (dialog.ShowDialog() == true)
            {
                loadConfig(dialog.FileName);                
            }            
        }
        private void saveConfig_Click(object sender, RoutedEventArgs e)
        {
            if (!(Config is null))
                saveConfig();
            else
            {
                saveAsConfig_Click(sender, e);
            }            
        }
        private void saveAsConfig_Click(object sender, RoutedEventArgs e)
        {
            if (isListFormatsEmpty(ListFormats)) //check empty selected formats
            {
                var dialog = new SaveFileDialog();
                dialog.Filter = "INI file (*.ini)|*.ini";
                dialog.FileName = textBoxDatasetName.Text;
                if (dialog.ShowDialog() == true)
                {
                    File.Create(dialog.FileName);
                    saveConfig(dialog.FileName);                   
                }
            }
            else
                MessageBox.Show("Не выбран ни один формат для фильтрации аудиофайлов. Выберите хотя бы один формат.",
                    "Фильтр форматов пуст!", MessageBoxButton.OK, MessageBoxImage.Exclamation);

        }
        private void onChangedValues(object sender, RoutedEventArgs e)
        {
            IsSaved = false;
        }
        private void closeProgramm_Click(object sender, RoutedEventArgs e)
        {

            saveCheck( Close );
        }
        //private void closeConfig_Click(object sender, RoutedEventArgs e)
        //{
        //    saveCheck(setDefaultConfig);
        //}

        #region format_filters
        public void selectFormats(ListBox list, string formats)
        {
            list.UnselectAll();
            var filters = formats.Split(';');

            foreach (var filter in filters)
            {
                int searchIndex = 0; //Use a counter to know on which row you currently are
                foreach (var item in list.Items)
                {
                    if ((string)item == filter)
                    {//If text is found, select row at the current index
                        var elem = list.Items[searchIndex];
                        list.SelectedItems.Add(elem);
                        break;
                    }

                    //afterwards, increase searchIndex ++1
                    searchIndex++;
                }
            }
            list.Focus();
        }

        public bool isListFormatsEmpty(ListBox list)
        {
            return list.SelectedItems.Count > 0;
        }
        private void selectAllFormats_Click(object sender, RoutedEventArgs e)
        {
            ListFormats.SelectAll();           
        }

        private void unselectAllFormats_Click(object sender, RoutedEventArgs e)
        {
            ListFormats.UnselectAll();
        }
        #endregion


        #region process_start
        private void btnStartTransferDataset_Click(object sender, RoutedEventArgs e)
        {
            Preprocess preprocess = new Preprocess(Config.GetPath(), StartMode.PROCESS);
            preprocess.ShowDialog();
        }

        private void btnStartTrain_Click(object sender, RoutedEventArgs e)
        {
            Preprocess preprocess = new Preprocess(Config.GetPath(), StartMode.TRAIN);
            preprocess.ShowDialog();
        }

        private void btnStartClassify_Click(object sender, RoutedEventArgs e)
        {
            Preprocess preprocess = new Preprocess(Config.GetPath(), StartMode.WORK);
            preprocess.ShowDialog();
        }

        private void btnStartAll_Click(object sender, RoutedEventArgs e)
        {
            Preprocess preprocess = new Preprocess(Config.GetPath(), StartMode.ALL);
            preprocess.ShowDialog();
        }
        #endregion
    }
}
