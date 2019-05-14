#!/usr/bin/env bash
#inputfile=/nfs/project/zhujy/roadnet/mm_nextlink.txt.20180719
formatfile=road_graph/roadnet.adjlist
outputfile=/nfs/project/zhujy/roadnet/output2/roadnet.embeddings
#awk -F "\t" '{gsub(","," ",$2);print $1" "$2}' ${inputfile} > ${formatfile}
python -u -m deepwalk --vertex-freq-degree --format adjlist --input ${formatfile} \
--max-memory-data-size 0 --number-walks 80 --representation-size 128 --walk-length 40 --window-size 5 \
--workers 80 --output ${outputfile} --undirected ""