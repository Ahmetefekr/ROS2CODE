#!/bin/bash

# =========================================================
# SÜRÜ İHA BAŞLATMA SCRİPTİS
# =========================================================

WS_PATH="$HOME/SURU-IHA-2026/kokboru_ws"
SETUP="source /opt/ros/humble/setup.bash; source $WS_PATH/install/setup.bash"

echo "SÜRÜ İHA 2026 GÖREV 1 SİSTEMİ BAŞLATILIYOR"

gnome-terminal --window \
  --tab --title="DDS_Koprusu" --command="bash -c 'MicroXRCEAgent udp4 -p 8888; exec bash'" \
  --tab --title="Omurilik_1"  --command="bash -c '$SETUP; ros2 run swarm_core uav_agent --ros-args -p agent_id:=uav_1; exec bash'" \
  --tab --title="Omurilik_2"  --command="bash -c '$SETUP; ros2 run swarm_core uav_agent --ros-args -p agent_id:=uav_2; exec bash'" \
  --tab --title="Omurilik_3"  --command="bash -c '$SETUP; ros2 run swarm_core uav_agent --ros-args -p agent_id:=uav_3; exec bash'" \
  --tab --title="Beyin_1"     --command="bash -c '$SETUP; ros2 run swarm_core swarm_agent --ros-args -p my_id:=1 -p uav_count:=3 -p team_id:=team_1; exec bash'" \
  --tab --title="Beyin_2"     --command="bash -c '$SETUP; ros2 run swarm_core swarm_agent --ros-args -p my_id:=2 -p uav_count:=3 -p team_id:=team_1; exec bash'" \
  --tab --title="Beyin_3"     --command="bash -c '$SETUP; ros2 run swarm_core swarm_agent --ros-args -p my_id:=3 -p uav_count:=3 -p team_id:=team_1; exec bash'" \
  --tab --title="Zeka_Goz"    --command="bash -c 'sleep 2; $SETUP; ros2 run swarm_core camera_node --ros-args -p agent_id:=uav_1; exec bash'"

echo "Bütün sekmeler açıldı."
