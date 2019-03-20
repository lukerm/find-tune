TARGET_MAC=**:**:**:**:**:**

BT_DEV_MAC=$(pactl list cards short | grep -i 'bluez_card' | sed -E 's/.*bluez_card\.(([0-9A-F]_?){6})/\1/' | cut -c 1-17 | sed 's/_/:/g')
PROFILE=$(pactl list sinks | grep -i 'bluetooth.protocol' | sed -E 's/.*"([a-z0-9_]+)"/\1/')

if [[ $BT_DEV_MAC == $TARGET_MAC ]] && [[ $PROFILE == "a2dp_sink_woops" ]]
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
