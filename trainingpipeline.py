### This is the training pipeline file.
## Necessary Imports
import numpy as np
import pandas as pd
import math, os, pickle, random
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

# random.seed(42) # Set seed
# np.random.seed(42)

N_of_samples = 300         # Number of samples to consider an activity
bandWidth = 0

### CHANGE THESE TWO CODE based on your file location
# Define the directory path where the CSV files are located
directory = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\our_data"

# Location where model parameter is stored
model_params = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\model_params"


def dataframe_of_CSI(directory):
    # Create empty DataFrames for walk, noact, and jog
    df_walk = pd.DataFrame()
    df_noact = pd.DataFrame()
    df_jog = pd.DataFrame()

    # Loop through each file in the directory
    for file in os.listdir(directory):
        # Check if the file is a CSV file and contains "walk", "noact", or "jog" in the name
        if file.endswith(".csv") and ("walk" in file or "noact" in file or "jog" in file):
            # Read the CSV file and extract the CSI_DATA column
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            csi_rows_raw = []

            ## Filtering can be done using
            df = df[(df["bandwidth"]==bandWidth)]# & (df["secondary_channel"]==1)]

            ## Ignore first few and last few seconds data
            for one_row in df['CSI_DATA'].iloc[40:-40]:
                one_row = one_row.strip("[]")
                csi_row_raw = [int(x) for x in one_row.split(" ") if x != '']
                csi_rows_raw.append(csi_row_raw)
        
            # Convert the list of lists to a DataFrame and append it to the appropriate DataFrame based on the file name
            csi_df = pd.DataFrame(csi_rows_raw)

            # Check which dataframe we are working on and concat the data
            if "walk" in file:
                df_walk = pd.concat([df_walk, csi_df], axis=0)
            elif "noact" in file:
                df_noact = pd.concat([df_noact, csi_df], axis=0)
            else:
                df_jog = pd.concat([df_jog, csi_df], axis=0)

    return df_walk, df_noact, df_jog


## Extract Amplitude and Phase from the dataframe
def convert_csi_to_amplitude_phase(df):
    """
    Get dataframe and extract amplitude and phase

    Args:
        df (Dataframe): pandas DataFrame

    Returns:
        Dataframe: pandas DataFrames
    """

    total_amplitudes = []
    total_phases = []

    for _, value in enumerate(df.values):
        imaginary = []
        real = []
        amplitudes = [] 
        phases = []

        csi_one_row_lst = value.tolist()

         # Create list of imaginary and real numbers from CSI
        [imaginary.append(csi_one_row_lst[item]) if item%2==0 else real.append(csi_one_row_lst[item]) for item in range(len(csi_one_row_lst))]

        # Transform imaginary and real into amplitude and phase
        val = int(len(csi_one_row_lst)//2)
        for k in range(val):
            amplitudes.append(round(math.sqrt(float(imaginary[k])** 2 + float(real[k])** 2),4))
            phases.append(round(math.atan2(float(imaginary[k]), float(real[k])),4))
        total_amplitudes.append(np.array(amplitudes))
        total_phases.append(np.array(phases))
    
    total_amplitudes_df = pd.DataFrame(total_amplitudes)
    total_phases_df = pd.DataFrame(total_phases)

    return total_amplitudes_df, total_phases_df


def extract_activity_amp_phase(dataFrm):
    """
    Based on sig_mode, 802.11a/g/n received. Here we receive both 802.11a/g and 802.11n
    Either 52 or 56 total sub-carrier would be useful. 
    The first 4 and the last 4 are rejected as null guard.
 
    Args:
        dataFrm (DataFrame): pandas Dataframe

    Returns:
        dataFrm (DataFrame): pandas Dataframe
    """
     
    activity_amp1 = dataFrm.iloc[:,5:32]
    activity_amp2 = dataFrm.iloc[:,33:60] # 33:59 for 802.11ag 33:61 for 802.11n

    activity_amp_df =  pd.concat([activity_amp1, activity_amp2],axis=1)

    return activity_amp_df


# Moving average of the data
def moving_average(df, window_size):
    """
    Compute the moving average with a window of size specified

    Returns:
        dataFrame: pandas DataFrame
    """

    rolling_mean = df.rolling(window=window_size).mean()
    downsampled = rolling_mean.iloc[window_size::window_size, :]
    return downsampled


def select_data_portion(dataFrm,select_size):
    """
    _summary_

    Args:
        dataFrm (_type_): pandas DataFrame
        select_size (int): Sample number that is considered as an activity 

    Returns:
        dataFrame: pandas DataFrame
    """

    selected_df_list = []
    for item in range(0,len(dataFrm)-select_size, select_size):
        selected_df = dataFrm.iloc[item:item+select_size].to_numpy().flatten()  # Select data and flatten
        selected_df_list.append(selected_df)
    selected_df = pd.DataFrame(selected_df_list)
    return selected_df

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y_training, shuffle=True, random_state=random_state)
    return X_train, X_test, y_train, y_test


def test_model(model, X_test):
    """
    Evaluate the trained model on the testing data.
    """ 
    label = model.predict(X_test)
    return label

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the testing data.
    """ 
    score = model.score(X_test, y_test)
    return score


if __name__ == "__main__":
    # Select dataframe for each activity
    walk_df, noact_df, jog_df = dataframe_of_CSI(directory)

    ## Extract walk amplitude and phase
    walk_amplitudes_df, _ = convert_csi_to_amplitude_phase(walk_df)
    walk_df_amps_final = extract_activity_amp_phase(walk_amplitudes_df)

    jog_amplitudes_df, _ = convert_csi_to_amplitude_phase(jog_df)
    jog_df_amps_final = extract_activity_amp_phase(jog_amplitudes_df)

    noact_amplitudes_df, _ = convert_csi_to_amplitude_phase(noact_df)
    noact_df_amps_final = extract_activity_amp_phase(noact_amplitudes_df)

    ## Set N number of samples
    X_walk = select_data_portion(walk_df_amps_final, N_of_samples)
    X_jog = select_data_portion(jog_df_amps_final, N_of_samples)
    X_noact = select_data_portion(noact_df_amps_final, N_of_samples)
    X_training = pd.concat([X_walk,X_jog,X_noact],axis=0,ignore_index=True)

    ## Prepare target label
    y_walk = np.zeros(len(X_walk))
    y_jog = np.ones(len(X_jog))
    y_noact = np.ones(len(X_noact))+1
    y_training = np.concatenate([y_walk, y_jog, y_noact],axis=0)

    ## Train-test split and fit, predict pipeline
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training)
    
    ## SVM
    pipe1 = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=5)),('svc', SVC())])
    pipe1.fit(X_train, y_train)
    y_pred1 = pipe1.predict(X_test)

    print("Model Performance with SVM:", evaluate_model(pipe1, X_test,y_test))
    print(confusion_matrix(y_test, y_pred1))
    print(classification_report(y_test, y_pred1))
    
    ## Dump Model to Target Location
    pickle.dump(pipe1, open(f"{model_params}\\pipefinal_svm.pkl","wb"))

    ## KNN
    pipe2 = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=5)),('knn', KNeighborsClassifier())])
    pipe2.fit(X_train, y_train)
    y_pred2 = pipe2.predict(X_test)

    print("\nModel Performance with KNN:", evaluate_model(pipe2, X_test,y_test))
    print(confusion_matrix(y_test, y_pred2))
    print(classification_report(y_test, y_pred2))

    ## Dump Model to Target Location
    pickle.dump(pipe2, open(f"{model_params}\\pipefinal_knn.pkl","wb"))