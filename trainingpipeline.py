import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math, os, pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


N_of_samples = 50
bandWidth = 0

def dataframe_of_CSI(directory):
    # Create empty DataFrames for walk, up, and jog
    df_walk = pd.DataFrame()
    df_up = pd.DataFrame()
    df_jog = pd.DataFrame()

    # Loop through each file in the directory
    for file in os.listdir(directory):
        # Check if the file is a CSV file and contains "walk", "up", or "jog" in the name
        if file.endswith(".csv") and ("walk" in file or "up" in file or "jog" in file):
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
            elif "up" in file:
                df_up = pd.concat([df_up, csi_df], axis=0)
            else:
                df_jog = pd.concat([df_jog, csi_df], axis=0)

    return df_walk, df_up, df_jog


## Extract Amplitude and Phase from the dataframe
def convert_csi_to_amplitude_phase(df):
    total_amplitudes = []
    total_phases = []

    for i, value in enumerate(df.values):
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
    ## Here, based on sig_mode, 802.11a/g/n received. Here we receive both 802.11a/g and 802.11n
    ## So, either 52 or 56 total sub-carrier would be useful. The first 4 and the last 4 are rejected as null guard.

    activity_amp1 = dataFrm.iloc[:,5:32]
    activity_amp2 = dataFrm.iloc[:,33:60] # 33:59 for 802.11ag 33:61 for 802.11n

    activity_amp_df =  pd.concat([activity_amp1, activity_amp2],axis=1)

    return activity_amp_df


# Moving average of the data
def moving_average(df, window_size):
    """"
    Compute the moving average with a window of size specified
    """

    rolling_mean = df.rolling(window=window_size).mean()
    downsampled = rolling_mean.iloc[window_size::window_size, :]
    return downsampled


def select_data_portion(dataFrm,select_size):
    selected_df_list = []
    for item in range(0,len(dataFrm)-select_size, select_size):
        selected_df = dataFrm.iloc[item:item+select_size].to_numpy().flatten()
        selected_df_list.append(selected_df)
    selected_df = pd.DataFrame(selected_df_list)
    return selected_df


def perform_scaling(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaler, scaled_data


def perform_pca(X, n_components):
    """
    Perform PCA on the data.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca_data = pca.transform(X)
    return pca, pca_data


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train):
    """
    Train SVM model using the training data.
    """
    # svm = SVC(C=10, gamma=0.001)
    svm = SVC()
    svm.fit(X_train, y_train)
    return svm


def train_knn(X_train, y_train):
    """
    Train KNN model using the training data.
    """
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

def train_model(model_type, X_train, y_train):
    """
    Train model of given type using the training data.
    """
    if model_type == 'svm':
        model = train_svm(X_train, y_train)
    elif model_type == 'knn':
        model = train_knn(X_train, y_train)
    else:
        raise ValueError('Invalid model type.')
    return model


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
    # Define the directory path where the CSV files are located
    directory = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\our_data"

    # Select dataframe for each acitivity
    walk_df, up_df, jog_df = dataframe_of_CSI(directory)

    ## Extract walk amplitude and phase
    walk_amplitudes_df, _ = convert_csi_to_amplitude_phase(walk_df)
    walk_df_amps_final = extract_activity_amp_phase(walk_amplitudes_df)

    jog_amplitudes_df, _ = convert_csi_to_amplitude_phase(jog_df)
    jog_df_amps_final = extract_activity_amp_phase(jog_amplitudes_df)

    up_amplitudes_df, _ = convert_csi_to_amplitude_phase(up_df)
    up_df_amps_final = extract_activity_amp_phase(up_amplitudes_df)

    ## Set N number of samples
    X_walk = select_data_portion(walk_df_amps_final, N_of_samples)
    X_jog = select_data_portion(jog_df_amps_final, N_of_samples)
    X_up = select_data_portion(up_df_amps_final, N_of_samples)
    X_training = pd.concat([X_walk,X_jog,X_up],axis=0,ignore_index=True)

    y_walk = np.zeros(len(X_walk))
    y_jog = np.ones(len(X_jog))
    y_up = np.ones(len(X_up))+1
    y_training = np.concatenate([y_walk, y_jog, y_up],axis=0)

    model_params = "C:\\Users\\Dell\\Documents\\Wifi-Sensing-HAR\\data\\model_params"
    scale_obj, scaled_X = perform_scaling(X_training)
    pickle.dump(scale_obj, open(f"{model_params}\\scaling.pkl","wb"))

    pca_obj, pca_X = perform_pca(scaled_X, 0.95)
    pickle.dump(pca_obj, open(f"{model_params}\\pca.pkl","wb"))

    print(np.array(pca_X).shape)
    print(f'Total number of components used after PCA : {pca_obj.n_components_}')

    X_train, X_test, y_train, y_test = train_test_split(pca_X, y_training)
    model = train_model('svm', X_train, y_train)
    # pickle.dump(model, open(f"{model_params}\\model.pkl","wb"))
    y_pred = test_model(model, X_test)
    print(evaluate_model(model, X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))