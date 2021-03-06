#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:.
#inputfile=/nfs/project/zhujy/roadnet/mm_nextlink.txt.20180719
#formatfile=road_graph/roadnet.adjlist
#outputfile=roadnet.embeddings
formatfile=example_graphs/karate.adjlist
outputfile=karate.embedding.als
dim=64
python -u als_train.py --format adjlist --input ${formatfile} --dim ${dim} --output ${outputfile}.tmp \
--num-iters 100000 --reg 0.01 --unobs 0.
# --undirected ""
count=`cat ${outputfile}.tmp|wc -l`
echo "${count} ${dim}" > ${outputfile}
awk '{print NR" "$0}' ${outputfile}.tmp >> ${outputfile}
rm -f ${outputfile}.tmp

python -u example_graphs/scoring_new.py --emb ${outputfile} --network ${formatfile}
# --undirected ""