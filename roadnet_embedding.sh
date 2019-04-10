#!/usr/bin/env bash
inputfile=/nfs/project/zhujy/roadnet/mm_nextlink.txt.20180719
formatfile=road_graph/roadnet.adjlist
awk -F "\t" '{gsub(","," ",$2);print $1" "$2}' ${inputfile} > ${formatfile}
deepwalk --format adjlist --input ${formatfile} \
--max-memory-data-size 0 --number-walks 80 --representation-size 128 --walk-length 40 --window-size 10 \
--workers 20 --output road_graph/roadnet.embeddings --undirected ""