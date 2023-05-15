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

import pickle
import numpy as np
import pandas as pd
from trainingpipeline import convert_csi_to_amplitude_phase, extract_activity_amp_phase

loaded_pipe = pickle.load(open("../model/pipefinal_svm.pkl","rb"))
column_names=["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","real_time_set","real_timestamp","len","CSI_DATA"]

# Define the batch size
batch_size = 300
bandWidth = 0
# Read the initial CSV file into a pandas DataFrame

def testData():
    
    while True: 
        tempDF=pd.read_csv("tempfile/tempData.csv")
        tempDF.columns=column_names;
        tempDF.dropna(inplace=True);
        df = tempDF[(tempDF["bandwidth"]==bandWidth)]
        df.reset_index(inplace=True);
        
        print(df.loc[5]['local_timestamp'])
    
    
        # Initialize a variable to keep track of the last processed row
        last_processed_row = 0
        counter = 0
    

        # Continuously monitor the CSV file for changes
        # Check if there are new rows to process
        if len(df) > (last_processed_row+300):
            # Get the new rows since the last processed row
            new_rows = df.iloc[last_processed_row:]
#             new_rows = df.head(batch_size)
            csi_rows_raw = []
            print(len(new_rows[new_rows['bandwidth'] == bandWidth]))

            filtered_df = new_rows[len(new_rows)-batch_size:]
            print(len(filtered_df))
#                 filtered_df = pd.DataFrame(filtered_rows)
            for one_row in filtered_df['CSI_DATA']:
                one_row = one_row.strip("[]")
                csi_row_raw = [int(x) for x in one_row.split(" ") if x != '']
                csi_rows_raw.append(csi_row_raw)

            csi_df = pd.DataFrame(csi_rows_raw)
            csi_df.dropna(inplace=True)
            activity_amplitudes_df, _ = convert_csi_to_amplitude_phase(csi_df)
            X_realtimetest = extract_activity_amp_phase(activity_amplitudes_df)
            print(f"Counter = {counter} and Sample = {len(X_realtimetest)} \n")
            X_realtimetest = X_realtimetest.to_numpy().reshape(1,-1)

            print(len(X_realtimetest))
            # Prediction the data
            Xtest_pred = loaded_pipe.predict(X_realtimetest)

            # Inference
            if (Xtest_pred == [0]):
                print("Walking")
            elif (Xtest_pred == [1]):
                print("Jogging")
            elif (Xtest_pred == [2]):
                print("Idle")
            else:
                pass

            # Update the counter
            counter+=1
            # Update the last processed row
            last_processed_row = last_processed_row + len(csi_df)
            time.sleep(20)

        else:
            continue


t2=threading.Thread(target=testData);
t2.start()