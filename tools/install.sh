#!/bin/bash

# Bu betik CircleCI üzerinde veya lokalde Python sanal ortamı kurar 
# ve requirements.txt içindeki kütüphaneleri yükler.
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt