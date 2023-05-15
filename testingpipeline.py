import pickle
import numpy as np
import pandas as pd
from trainingpipeline import convert_csi_to_amplitude_phase, extract_activity_amp_phase


# Import Pipeline
model_params = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\model_params"
loaded_pipe = pickle.load(open(f"{model_params}\\pipefinal.pkl","rb"))

## Suppose the realtimedata is our previous walking dataset
data1 = pd.read_csv("C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\our_data\\krishna_jog1.csv")
print(len(data1))

# Define the batch size
batch_size = 100
bandWidth = 0
# Read the initial CSV file into a pandas DataFrame
df = data1[(data1["bandwidth"]==bandWidth)]

# Initialize a variable to keep track of the last processed row
last_processed_row = 0
counter = 0

# Continuously monitor the CSV file for changes, select 100 samples with bandwidth 20MHz and make prediction
while True:
    # Check if there are new rows to process
    if len(df) > last_processed_row:
        # Get the new rows since the last processed row
        new_rows = df.iloc[last_processed_row:]
        new_rows = df.head(batch_size)
        csi_rows_raw = []
        # Filter the new rows where bandwidth = 0
        filtered_rows = new_rows[new_rows['bandwidth'] == 0].head(batch_size)
        filtered_df = pd.DataFrame(filtered_rows)
        for one_row in filtered_df['CSI_DATA']:
            one_row = one_row.strip("[]")
            csi_row_raw = [int(x) for x in one_row.split(" ") if x != '']
            csi_rows_raw.append(csi_row_raw)

        csi_df = pd.DataFrame(csi_rows_raw)
        activity_amplitudes_df, _ = convert_csi_to_amplitude_phase(csi_df)
        X_realtimetest = extract_activity_amp_phase(activity_amplitudes_df)
        print(f"Counter = {counter} and Sample = {len(X_realtimetest)} \n")
        X_realtimetest = X_realtimetest.to_numpy().reshape(1,-1)

        # Prediction on the data
        Xtest_pred = loaded_pipe.predict(X_realtimetest)

        # Inference
        if (Xtest_pred == [0]):
            print("Walking")
        elif (Xtest_pred == [1]):
            print("Jogging")
        elif (Xtest_pred == [2]):
            print("No Activity")
        else:
            pass

        # Update the counter
        counter+=1
        # Update the last processed row
        last_processed_row = last_processed_row + len(csi_df)