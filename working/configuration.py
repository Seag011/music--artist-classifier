# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.



import configparser as cp

def getConfig(path):
    config = cp.ConfigParser()
    config.read(path, encoding="utf-8-sig")
    return config