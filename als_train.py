#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import networkx as nx
import tensorflow as tf

import numpy as np
from deepwalk import graph
import wals
import datetime
from tensorflow.contrib.factorization.python.ops import factorization_ops
from six.moves import range

import psutil
from multiprocessing import cpu_count

# p = psutil.Process(os.getpid())
# try:
#     p.set_cpu_affinity(list(range(cpu_count())))
# except AttributeError:
#     try:
#         p.cpu_affinity(list(range(cpu_count())))
#     except AttributeError:
#         pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def process(args):
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    print("Number of nodes: {}".format(len(G.nodes())))

    nxG = nx.Graph(G)
    Gmat = nx.adjacency_matrix(nxG).tocoo()

    num_iters = args.num_iters

    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # generate model
    input_tensor, row_factor, col_factor, model = wals.wals_model(Gmat,
                                                                  args.dim // 2,
                                                                  args.reg,
                                                                  args.unobs)

    # factorize matrix
    session = wals.simple_train(model, input_tensor, num_iters)

    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)

    # close the training session now that we've evaluated the output

    session.close()

    embedding = np.concatenate((output_row, output_col), axis=1)
    print(embedding)
    # save trained model to job directory
    np.savetxt(args.output, embedding)
    # log results
    train_rmse = wals.get_rmse(output_row, output_col, Gmat)

    log_info = 'train RMSE = %.2f' % train_rmse
    tf.logging.info(log_info)
    print(log_info)

def main():
    parser = ArgumentParser("als",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--dim', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--num-iters', default=10000, type=int,
                        help='number of iters.')

    parser.add_argument('--reg', default=0.01, type=float,
                        help='weights of l2 norm.')

    parser.add_argument('--unobs', default=0., type=float,
                        help='weights of unobserved entries.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    process(args)


if __name__ == "__main__":
    sys.exit(main())
