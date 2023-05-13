import csv, pickle
import numpy as np
import pandas as pd
from trainingpipeline import dataframe_of_CSI, convert_csi_to_amplitude_phase, extract_activity_amp_phase, moving_average, select_data_portion, perform_pca, perform_scaling, test_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import model, scalar, pca
model_params = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\model_params"
loaded_scalar = pickle.load(open(f"{model_params}\\scaling.pkl","rb"))
loaded_pca = pickle.load(open(f"{model_params}\\pca.pkl","rb"))
loaded_model = pickle.load(open(f"{model_params}\\model.pkl","rb"))


## Suppose the data is our previous walking dataset
data1 = pd.read_csv("C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\temp_data\\arjun_walk1.csv")
print(len(data1))

# Define the batch size
batch_size = 50
bandWidth = 0
# Read the initial CSV file into a pandas DataFrame
df = data1[(data1["bandwidth"]==bandWidth)]

# Initialize a variable to keep track of the last processed row
last_processed_row = 0
counter = 0

# Continuously monitor the CSV file for changes
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
        print(f"This is the {counter}th set of data lengthed {csi_df.shape}\n")
        activity_amplitudes_df, _ = convert_csi_to_amplitude_phase(csi_df)
        X_realtimetest = extract_activity_amp_phase(activity_amplitudes_df).to_numpy().flatten()
        # scaled_Xtest = loaded_scalar.transform(X_realtimetest.to_numpy().flatten())
        scaled_Xtest = (X_realtimetest - np.mean(X_realtimetest))/np.std(X_realtimetest)
        pca_Xtest = loaded_pca.transform(scaled_Xtest)
        Xtest_pred = loaded_model.predict(pca_Xtest)
        print(Xtest_pred)

        counter+=1

        # Update the last processed row
        last_processed_row = last_processed_row + len(csi_df)