#!/bin/bash
src_ip=239.234.234.234
src_port=56234

# src_ip=$1
# src_port=$2

network_stat=network.stat
if [ -e ${network_stat} ]; then
    rm ${network_stat} 
fi

while true; do
    timeout -s  KILL 1s socat udp4-recv:${src_port},bind=${src_ip},ip-add-membership=${src_ip}:0.0.0.0,reuseaddr - > video.es
    echo "$(date -u) $(stat -c%s  video.es)" >> ${network_stat}
    rm video.es

    sleep 10
done
