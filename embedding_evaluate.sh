#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:.
formatfile=road_graph/roadnet.adjlist
embfile=/nfs/project/zhujy/roadnet/output/roadnet.embeddings
python -u example_graphs/scoring_new.py --emb ${embfile} \
--network ${formatfile} --undirected ""