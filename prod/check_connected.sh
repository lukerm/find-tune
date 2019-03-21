#  Copyright (C) 2019 lukerm
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Complete with MAC address of the bluetooth device you will connect
TARGET_MAC=**:**:**:**:**:**

BT_DEV_MAC=$(pactl list cards short | grep -i 'bluez_card' | sed -E 's/.*bluez_card\.(([0-9A-F]_?){6})/\1/' | cut -c 1-17 | sed 's/_/:/g')
PROFILE=$(pactl list sinks | grep -i 'bluetooth.protocol' | sed -E 's/.*"([a-z0-9_]+)"/\1/')

if [[ $BT_DEV_MAC == $TARGET_MAC ]] && [[ $PROFILE == "a2dp_sink" ]]
then
  echo "connected"
else
  echo "reconnecting ..."
  sudo systemctl disable bluetooth.service
  sudo systemctl stop bluetooth.service
  sleep 2
  sudo systemctl enable bluetooth.service
  sudo systemctl start bluetooth.service
  sleep 5
  echo  "connect $TARGET_MAC" | bluetoothctl
fi
