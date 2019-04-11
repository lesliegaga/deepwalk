#!/usr/bin/env bash
inputfile=/nfs/project/zhujy/roadnet/mm_nextlink.txt.20180719
formatfile=road_graph/roadnet.adjlist
outputfile=/nfs/project/zhujy/roadnet/output/roadnet.embeddings
awk -F "\t" '{gsub(","," ",$2);print $1" "$2}' ${inputfile} > ${formatfile}
deepwalk --vertex-freq-degree --format adjlist --input ${formatfile} \
--max-memory-data-size 0 --number-walks 40 --representation-size 64 --walk-length 20 --window-size 5 \
--workers 80 --output ${outputfile} --undirected ""