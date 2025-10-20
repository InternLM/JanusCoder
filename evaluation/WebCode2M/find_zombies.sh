#!/usr/bin/env bash
echo "[*] Zombies:"
ps -eo pid,ppid,user,stat,etime,cmd | awk '$4 ~ /Z/'
echo
echo "[*] Top zombie parents (PPID -> count):"
ps -eo ppid=,stat= | awk '$2 ~ /Z/ {c[$1]++} END{for (p in c) printf "%-8s %d\n", p, c[p]}' \
| sort -k2,2nr | head
echo
echo "[*] Parent details:"
for p in $(ps -eo ppid=,stat= | awk '$2 ~ /Z/ {c[$1]++} END{for (p in c) print p}'); do
  ps -p "$p" -o pid,ppid,stat,etime,cmd
done
echo
echo "# 若确认父进程可重启： sudo systemctl restart <service>"
echo "# 或通知父进程回收：     kill -s CHLD <PPID>"
echo "# 或安全地结束父进程：   kill <PPID>; sleep 2; kill -9 <PPID>"
