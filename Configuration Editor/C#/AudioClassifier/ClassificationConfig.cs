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
using System.Reflection;
using IniParser;
using IniParser.Model;

namespace AudioClassifier
{
    public class ClassificationConfig
    {
        private IniData config;
        private FileIniDataParser parser = new FileIniDataParser();
        private string path = null;

        public ClassificationConfig()
        {
            parser = new FileIniDataParser();
            config = new IniData();
        }
        public ClassificationConfig(string path)
        {
            parser = new FileIniDataParser();
            ReadConfig(path);
        }

        public string GetPath()
        {
            return path;
        }
        public void ReadConfig(string path)
        {
            this.path = path;
            config = parser.ReadFile(path);
        }

        public void SaveIni(string path)
        {
            if (path.Length > 3 && path.Substring(path.Length - 4) != ".ini")
            {
                path += ".ini";
                this.path = path;
            }
            parser.WriteFile(this.path, config, Encoding.UTF8);
        }

        public void SaveIni()
        {
            SaveIni(this.path);
        }

        /************************* folders *************************/
        public string dataset
        {
            get
            {
                return config["Directories"]["dataset_dir"];
            }
            set
            {
                config["Directories"]["dataset_dir"] = value;
            }
        }
        public string temp_files
        {
            get
            {
                return config["Directories"]["temp_files"];
            }
            set
            {
                config["Directories"]["temp_files"] = value;
            }
        }
        public string train_results
        {
            get
            {
                return config["Directories"]["train_results"];
            }
            set
            {
                config["Directories"]["train_results"] = value;
            }
        }
        public string work
        {
            get
            {
                return config["Directories"]["work"];
            }
            set
            {
                config["Directories"]["work"] = value;
            }
        }
        /***********************************************************/

        /************************* audio ***************************/
        public string dataset_name
        {
            get
            {
                return config["DatasetProcess"]["dataset_name"];
            }
            set
            {
                config["DatasetProcess"]["dataset_name"] = value;
            }
        }
        public int sample_rate
        {
            get
            {
                return int.Parse(config["DatasetProcess"]["sample_rate"]);
            }
            set
            {
                config["DatasetProcess"]["sample_rate"] = value.ToString();
            }
        }
        public int slice_length
        {
            get
            {
                return int.Parse(config["DatasetProcess"]["slice_length"]);
            }
            set
            {
                config["DatasetProcess"]["slice_length"] = value.ToString();
            }
        }
        public int layering
        {
            get
            {
                return int.Parse(config["DatasetProcess"]["layering"]);
            }
            set
            {
                config["DatasetProcess"]["layering"] = value.ToString();
            }
        }

        public string filters
        {
            get
            {
                return config["DatasetProcess"]["filters"];
            }
            set
            {
                config["DatasetProcess"]["filters"] = value;
            }
        }
        /***********************************************************/



        public int epoches
        {
            get
            {
                return int.Parse(config["TrainParameters"]["epoches"]);
            }
            set
            {
                config["TrainParameters"]["epoches"] = value.ToString();
            }
        }
        public double learning_rate
        {
            get
            {
                return double.Parse(config["TrainParameters"]["learning_rate"]);
            }
            set
            {
                config["TrainParameters"]["learning_rate"] = value.ToString();
            }
        }
        public int batch_size
        {
            get
            {
                return int.Parse(config["TrainParameters"]["batch_size"]);
            }
            set
            {
                config["TrainParameters"]["batch_size"] = value.ToString();
            }
        }

        public bool early_stop
        {
            get
            {
                return bool.Parse(config["TrainParameters"]["early_stop"]);
            }
            set
            {
                config["TrainParameters"]["early_stop"] = value.ToString();
            }
        }
    }
}