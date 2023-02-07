#!/bin/bash

# Add script to download data fiels from bucket p7-data to /backend/resources
"/backend/resources"

python script_download_data_folder.py
#import subprocess
#print("__Download resources folder__")
#subprocess.call(r'python script_download_data_folder.py', shell=True)

# shellcheck disable=SC2068
python $@
