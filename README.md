# Wifi-Sensing-HAR
Human Activity Recognition using Wifi signal Channel State Information(CSI) data
Device: ESP 32 (easily available in the market)


## Local Implementation

- Clone this repository to your local machine
```
git clone https://github.com/jasminkarki/Wifi-Sensing-HAR
```

- In the directory where you put the cloned repo, create a virtual environment for Python:
```
pip install virtualenv
virtualenv -p python3 venv
```

- Activate your virtual environment
```
source venv/bin/activate
```

- Install packages required
```
pip install -r requirements.txt
```

- Configurations of ESP
    Two ESP32 Node MCU are used in the project. One NodeMCU works as an access point and other NodeMCU works as a station. The firmware for the access point and the station board are inside the `active_ap` and `active_sta`.

    The following configuration has to be done for both the access point and station board.
    1. Serial flasher config > 'idf.py monitor' baud rate > Custom Baud Rate
    2. Serial flasher config > Custom baud rate value > 921600 (Higher Baud Rate can be used if your system supports).
    3. Component config > Common ESP32-related > Channel for console output > Custom UART
    4. Component config > Common ESP32-related > UART console baud rate > 921600
    5. Component config > Wi-Fi > WiFi CSI(Channel State Information) 
    6. Component config > FreeRTOS > Tick rate (Hz) > 1000
    7. ESP32 CSI Tool Config > WiFi Channel Setting
    8. ESP32 CSI Tool Config > WiFi SSID
    9. ESP32 CSI Tool Config > WiFi Password
    10. ESP32 CSI Tool Config > Packet Tx Rate >100 (100 CSI packets per second)
    11. ESP32 CSI Tool Config > (Advanced users only) Should we only collect LLTF?

    Both the firmware should be built with the ESP-IDF and should be flashed to the boards using the command 
    ```
    idf.py -p /dev/ttyUSB0 flash
    ```

    Two boards should be placed at a separation of about 2 meters to collect data. We have not used the external antenna with the board and dependent only on the antenna built on the board itself. So larger separation between two ESPs will not work.

- Real Time Implementation
For the real-time implementation and testing of the algorithm, two python  and one bash scripts should be executed.

1. `csi_data.sh` script should be run inside the active_sta directory for the proper collection of data.

2. `realtime_csi_data.py` scripts is used to acquire the real time data from the serial monitor and save it on a csv file and update the csv file with new data every 3 seconds.

`realtime_system_testing.py` script reads the updated data frame and applies the model for the testing of the real-time data.
