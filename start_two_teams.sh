#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-3100}
team1=${3:-2677722}
team2=${4:-2677730}

# start first team (unums 1..5)
for i in {1..5}; do
  python3 ./Run_Player.py -i "$host" -p "$port" -u "$i" -t "$team1" -P 0 -D 0 &
done

# small stagger so one team can register first
sleep 0.5

# start second team (same unums, different team name)
for i in {1..5}; do
  python3 ./Run_Player.py -i "$host" -p "$port" -u "$i" -t "$team2" -P 0 -D 0 &
done

wait