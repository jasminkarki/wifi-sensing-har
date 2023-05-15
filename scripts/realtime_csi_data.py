import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math, os
import glob
import random
from sklearn.decomposition import PCA
import serial
import time
import threading
import os


import pickle
import numpy as np
import pandas as pd
from trainingpipeline import convert_csi_to_amplitude_phase, extract_activity_amp_phase


def get_real_time_data():
    while True:
        sta_dir=os.path.abspath("../firmware/active_sta")
        dest_dir=os.path.abspath("../scripts/tempfile")

        os.system("cp "+sta_dir+"/datalog.csv"+" "+dest_dir+"/datalog.csv")

        command ="cat tempfile/datalog.csv | grep -a \"CSI_DATA,\" > tempfile/tmpdata.csv"
        os.system(command)
        command ="sed -i '1d' tempfile/tmpdata.csv"
        os.system(command)
        command ="tail -n 5000 tempfile/tmpdata.csv > tempfile/tempData.csv"
        os.system(command)
        # tempDF=pd.read_csv("tempData.csv")
        # tempDF.columns=column_names;
        # print(len(tempDF))
        
        # time.sleep()

        # !cat datalog.csv | grep -a "CSI_DATA," > tmpdata.csv
        # !sed -i '1d' tmpdata.csv
        # !tail -n 2000 tmpdata.csv > tempData.csv
        # tempDF=pd.read_csv("tempData.csv")
        # tempDF.columns=column_names;
        # time.sleep(3)
    
t1 = threading.Thread(target=get_real_time_data)
t1.start() 