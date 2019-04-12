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
    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    args = parser.parse_args()
    # 0. Files
    embeddings_file = args.emb

    # 1. Load Embeddings
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

    # 2. Load labels
    G = graph.load_adjacencylist(args.network, undirected=args.undirected)

    total_right = 0.
    total_num_adj = 0.
    accurate_list = []
    node_count = 0
    for node in G.keys():
        cur_feat = model[str(node)]
        ctx_map = {}
        for context in G.keys():
            if node != context:
                ctx_feat = model[str(context)]
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
        print('-------------------')
        print('cur node:', node)
        print("sort_ctxlist", sort_ctxlist[:num_adj+10])
        print("adj_list:", adj_list)
        print('total adj num:', total_num_adj, " cur node count:", node_count, "total_right:", total_right)
        print('accurate:', accurate)
        macro_accurate = np.mean(accurate_list)
        micro_accurate = total_right / total_num_adj
        print('Macro accurate:', macro_accurate)
        print('Micro accurate:', micro_accurate)
        print('-------------------')
        # if node_count % 100 == 0:
        #     macro_accurate = np.mean(accurate_list)
        #     micro_accurate = total_right / total_num_adj
        #     print('total adj num:', total_num_adj, " cur node count:", node_count)
        #     print('-------------------')
        #     print('Macro accurate:', macro_accurate)
        #     print('Micro accurate:', micro_accurate)
        #     print('-------------------')
        node_count += 1
    macro_accurate = np.mean(accurate_list)
    micro_accurate = total_right / total_num_adj

    print('Results, using embeddings of dimensionality', model.vector_size)
    print('-------------------')
    print('Macro accurate:', macro_accurate)
    print('Micro accurate:', micro_accurate)
    print('-------------------')


if __name__ == "__main__":
    sys.exit(main())
