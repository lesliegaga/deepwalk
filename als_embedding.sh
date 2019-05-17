#!/usr/bin/env bash
#inputfile=/nfs/project/zhujy/roadnet/mm_nextlink.txt.20180719
formatfile=road_graph/roadnet.adjlist
outputfile=roadnet.embeddings
formatfile=example/karate.adjlist
outputfile=karate.adjlist.als

python -u als_train.py --format adjlist --input ${formatfile} \
--max-memory-data-size 0 --representation-size 64 \
--output ${outputfile}
# --undirected ""

python -u example_graphs/scoring_new.py --emb ${outputfile} \
--network ${formatfile}
# --undirected ""