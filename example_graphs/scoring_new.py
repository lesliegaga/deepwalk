#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__ = "Bryan Perozzi"

import numpy as np
import sys

from deepwalk import graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gensim.models import KeyedVectors

MAX_EVAL_NUM = 10

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
    # sec_G = graph.build_next_step_graph(G, G)
    total_right = 0.
    total_num_adj = 0.
    accurate_list = []
    node_count = 0
    vertex_counts = G.degree(nodes=G.keys())
    s_vertex_counts = sorted(vertex_counts.items(), key=lambda x: x[1], reverse=True)
    for node, degree in s_vertex_counts:
        cur_feat = model[str(node)]
        ctx_map = {}
        for context in G.keys():
            if node != context:
                ctx_feat = model[str(context)]
                ctx_map[context] = np.asscalar(np.inner(cur_feat, ctx_feat))
        sort_ctxlist = sorted(ctx_map.items(), key=lambda x:x[1], reverse=True)
        first_adjlist = G[node]
        # second_adjlist = sec_G[node]
        firt_right = 0.
        sec_right = 0.
        right = 0.
        for item in sort_ctxlist[:len(first_adjlist)]:
            flag = False
            if item[0] in first_adjlist:
                firt_right += 1
                flag = True
            # if item[0] in second_adjlist:
            #     sec_right += 1
            #     flag = True
            if flag:
                right += 1
        accurate = right / len(first_adjlist) if len(first_adjlist) > 0 else 0.
        accurate_list.append(accurate)
        total_right += right
        total_num_adj += len(first_adjlist)
        macro_accurate = np.mean(accurate_list)
        micro_accurate = total_right / total_num_adj
        print('-------------------')
        print('cur node:', node, ' degree:', degree)
        print("sort_ctxlist", sort_ctxlist[:10])
        print("first_adj_list:", first_adjlist)
        # print("second_adjlist:", second_adjlist)
        print('total eval num:', total_num_adj, " cur node count:", node_count,
              "total_right:", total_right, 'firt_right:', firt_right)
        print('accurate:', accurate, 'Macro accurate:', macro_accurate, 'Micro accurate:', micro_accurate)
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
