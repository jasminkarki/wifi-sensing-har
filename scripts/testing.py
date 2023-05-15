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

loaded_pipe = pickle.load(open("../model/pipefinal_svm.pkl","rb"))
column_names=["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","real_time_set","real_timestamp","len","CSI_DATA"]

sta_dir=os.path.abspath("../firmware/active_sta")
dest_dir=os.path.abspath("../scripts/tempfile")

os.system("cp "+sta_dir+"/datalog.csv"+" "+dest_dir+"/datalog.csv")

command ="cat tempfile/datalog.csv | grep -a \"CSI_DATA\" > tempfile/tmpdata.csv"
os.system(command)
command ="sed -i '1d' tempfile/tmpdata.csv"
os.system(command)

# Define the batch size
batch_size = 300
bandWidth = 0
# Read the initial CSV file into a pandas DataFrame
X_values = []
def testData():
    
    # while True: 
    tempDF=pd.read_csv("tempfile/tmpdata.csv")
    tempDF.columns=column_names;
    df = tempDF[(tempDF["bandwidth"]==bandWidth)]
    df.reset_index(inplace=True);
        
    # print(df.columns)
    
    for i in range(0,len(df)-batch_size,batch_size):
        filtered_df = df.iloc[i:i+batch_size]['CSI_DATA']
        csi_rows_raw = []
        for one_row in filtered_df:
            one_row = one_row.strip("[]")
            csi_row_raw = [int(x) for x in one_row.split(" ") if x != '']
            csi_rows_raw.append(csi_row_raw)

        csi_df = pd.DataFrame(csi_rows_raw)
        activity_amplitudes_df, _ = convert_csi_to_amplitude_phase(csi_df)
        # print(activity_amplitudes_df.shape)
        X_realtimetest = extract_activity_amp_phase(activity_amplitudes_df)
        X_realtimetest = X_realtimetest.to_numpy().reshape(1,-1)

        # print(X_realtimetest.shape)

        X_values.append(X_realtimetest)

    return X_values
                
for item in testData():
    # Prediction the data
    Xtest_pred = loaded_pipe.predict(item)

    # Inference
    if (Xtest_pred == 0):
        print("Walking")
    elif (Xtest_pred == 1):
        print("Jogging")
    elif (Xtest_pred == 2):
        print("Idle")
    else:
        print('Nothing')
        pass

testData();
# t2=threading.Thread(target=testData);
# t2.start()