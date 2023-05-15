#!/bin/bash
cd ../firmware/active_sta
source ~/.bashrc
export IDF_TOOLS_PATH=/home/bhusal17/ArjunsWorkspace/StudyMaterials/Learning/MTECH-CPS/SecondSemester/Courses/SmartSensingForIoT/Project/espOldTool
. /home/bhusal17/ArjunsWorkspace/StudyMaterials/Learning/MTECH-CPS/SecondSemester/Courses/SmartSensingForIoT/Project/toolchain/esp/esp-idf/export.sh
idf.py -p /dev/ttyUSB0 monitor >datalog.csv
