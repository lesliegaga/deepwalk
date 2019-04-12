#!/usr/bin/env bash
formatfile=road_graph/roadnet.adjlist
embfile=/nfs/project/zhujy/roadnet/output/roadnet.embeddings
python example_graphs/scoring_new.py --emb ${embfile} \
--network ${formatfile} --undirected ""