#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__ = "Bryan Perozzi"

import numpy as np
import sys

from deepwalk import graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gensim.models import KeyedVectors


def main():
    parser = ArgumentParser("scoring",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--emb", required=True, help='Embeddings file')
    parser.add_argument("--network", required=True,
                        help='A .mat file containing the adjacency matrix and node labels of the input network.')
    parser.add_argument("--adj-matrix-name", default='network',
                        help='Variable name of the adjacency matrix inside the .mat file.')
    parser.add_argument("--label-matrix-name", default='group',
                        help='Variable name of the labels matrix inside the .mat file.')
    parser.add_argument("--num-shuffles", default=2, type=int, help='Number of shuffles.')
    parser.add_argument("--all", default=False, action='store_true',
                        help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
                             'By default, only training percents of 10, 50 and 90 are used.')

    args = parser.parse_args()
    # 0. Files
    embeddings_file = args.emb

    # 1. Load Embeddings
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

    # 2. Load labels
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)

    total_right = 0.
    total_num_adj = 0.
    accurate_list = []
    for node in G.keys():
        cur_feat = model[node]
        ctx_map = {}
        for context in G.keys():
            if node != context:
                ctx_feat = model[context]
                ctx_map[context] = np.asscalar(np.inner(cur_feat, ctx_feat))
        sort_ctxlist = sorted(ctx_map.items(), key=lambda x:x[1], reverse=True)
        adj_list = G[node]
        num_adj = len(adj_list)
        right = 0.
        for item in sort_ctxlist[:num_adj]:
            if item[0] in adj_list:
                right += 1
        accurate = right / num_adj if num_adj > 0 else 0.
        accurate_list.append(accurate)
        total_right += right
        total_num_adj += num_adj
    macro_accurate = np.mean(accurate_list)
    micro_accurate = total_right / total_num_adj

    print('Results, using embeddings of dimensionality', model.vector_size)
    print('-------------------')
    print('Macro accurate:', macro_accurate)
    print('Micro accurate:', micro_accurate)
    print('-------------------')


if __name__ == "__main__":
    sys.exit(main())
