import argparse
import logging
import os
import sys
import datetime
from pathlib import Path
import torch
import numpy as np
import random
from utils.data import get_data, Data
from models.bpdr import BPDRNet
from utils.sampler import get_neighbor_finder
from utils.helpers import EarlyStopMonitor
from train_eval.evaluation import ranking_evaluation
import math
import time
from tqdm import tqdm
from torch_geometric.data import Data as PyGData, Batch
from torch_geometric.utils import to_undirected
from models.loss_functions import recon_loss
from utils.sampler import temporal_sampling
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score


batch_scores = []
previous_batch_scores = []
epoch_average_scores = []
epoch_previous_scores = []
epoch_positive_previous_scores = []
epoch_negative_previous_scores = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def calculate_harmonic_term(h, batch):
    harmonic_term = 0.0
    num_subgraphs = len(batch.ptr) - 1
    
    for i in range(num_subgraphs):
        subgraph_embeddings = h[batch.ptr[i]:batch.ptr[i + 1]]
    

        pos_embedding = subgraph_embeddings[1]
        distances = torch.norm(subgraph_embeddings - pos_embedding, dim=1)
        distances[1] = float('inf')

        neg_index = torch.argmin(distances)
        neg_embedding = subgraph_embeddings[neg_index]
        
        pos_score = torch.dot(pos_embedding, pos_embedding)
        neg_score = torch.dot(pos_embedding, neg_embedding)
        
        bpr_loss = -F.logsigmoid(pos_score - neg_score)
        
        harmonic_term += bpr_loss

    return harmonic_term / num_subgraphs
    
    
def training_epoch(net, optimizer, train_data, full_data, node_features, edge_features, train_neighbor_finder,
                            batch_size, num_temporal_hops, n_neighbors, num_samples, verbose=False, coalesce_edges_and_time=False, train_randomize_timestamps=False):
    
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    net.train()
    train_start_time = time.time()

    positive_previous_scores = []
    negative_previous_scores = []

    for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
        optimizer.zero_grad()
        batch_loss = []
        start_idx = batch_idx * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
                                                    timestamps_batch, train_neighbor_finder, train_data,
                                                    node_features, edge_features, full_data.n_unique_nodes,
                                                    num_temporal_hops, n_neighbors, coalesce_edges_and_time, train_randomize_timestamps)

        batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
        

        h = net(batch_pos)
        previous_rank_scores, rank_scores = net.predict_proba(h)

        batch_scores.extend(rank_scores.tolist())
        previous_batch_scores.extend(previous_rank_scores.tolist())

        splits = torch.tensor_split(rank_scores, batch_pos.ptr)
        previous_splits = torch.tensor_split(previous_rank_scores, batch_pos.ptr)

        loss = []

        # listwise loss
        for sp, prev_sp in zip(splits[1:-1], previous_splits[1:-1]):
            y = torch.zeros(sp.shape[0], device=net.device)
            y[1] = 1.0

            # previous scores
            positive_previous_scores.append(prev_sp[1].item())
            negative_previous_scores.extend([prev_sp[i].item() for i in range(len(prev_sp)) if i != 1])
            loss.append(-torch.sum(y * F.log_softmax(sp, dim=0)))

        listwise_loss = sum(loss) / len(loss)
        harmonic_term = calculate_harmonic_term(h, batch_pos)
        alpha = 0.3
        total_loss = listwise_loss + alpha * harmonic_term

        total_loss.backward()
        optimizer.step()
        batch_loss.append(total_loss.item())

    avg_epoch_loss = np.array(batch_loss).mean()
    avg_positive_previous_scores = np.mean(positive_previous_scores)
    avg_negative_previous_scores = np.mean(negative_previous_scores)
    train_end_time = time.time()

    return avg_epoch_loss, train_end_time - train_start_time, num_instance, avg_positive_previous_scores, avg_negative_previous_scores


def parse_arguments():
    parser = argparse.ArgumentParser('BandRank Interaction Ranking Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset directory')
    parser.add_argument('--data', type=str, default='wikipedia',
                        help='Dataset name (eg. reddit, wikipedia, mooc, lastfm, enron, uci)')
    parser.add_argument('--prefix', type=str, default='bandrank',
                        help='Prefix to name the checkpoints and models')
    parser.add_argument('--train_batch_size', default=64,
                        type=int, help="Train batch size")
    parser.add_argument('--eval_batch_size', default=256, type=int,
                        help="Evaluation batch size (should experiment to make it as big as possible (based on available GPU memory))")
    parser.add_argument('--num_epochs', default=25, type=int,
                        help="Number of training epochs")
    parser.add_argument('--num_layers', default=3, type=int, help="Number of layers")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")
    parser.add_argument('--time_dim', type=int, default=128,
                        help="Time Embedding dimension size. Give 0 if no time encoding is not to be used")
    parser.add_argument('--num_temporal_hops', type=int, default=3,
                        help="No. of temporal hops for sampling candidates during training.")
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help="No. of neighbors to sample for each candidate node at each temporal hop. This is also the same parameter that samples edges.")
    parser.add_argument('--uniform_sampling', action='store_true',
                        help='Whether to use uniform sampling for temporal neighbors. Default is most recent sampling.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--log_dir', type=str, default="logs/",
                        help="directory for storing logs.")
    parser.add_argument('--saved_models_dir', type=str, default="outputs/saved_models/",
                        help="directory for saved models.")
    parser.add_argument('--saved_checkpoints_dir', type=str, default="outputs/saved_checkpoints/",
                        help="directory for saved checkpoints.")
    parser.add_argument('--verbose', type=int, default=0, help="Verbosity 0/1 for tqdm")
    parser.add_argument('--seed', type=int, default=0, help="deterministic seed for training. this is different from that by used neighbor finder which uses a local random state")
    parser.add_argument('--num_temporal_hops_eval', type=int, default=3,
                        help="No. of temporal hops for sampling candidates during evaluation.")
    parser.add_argument('--num_neighbors_eval', type=int, default=20,
                        help="No. of neighbors to sample for each candidate node at each temporal hop during evaluation. This is also the same parameter that samples edges.")
    parser.add_argument('--no_fourier_time_encoding', action='store_true',
                        help='Whether to not use fourier time encoding')
    parser.add_argument('--coalesce_edges_and_time', action='store_true',
                        help='Whether to coalesce edges and time. make sure no_fourier_time_encoding is set and time_dim is 1. else will raise error')
    parser.add_argument('--train_randomize_timestamps', action='store_true',
                        help='Whether to randomize train timestamps i.e. after sampling  and before going into BPDR')
    parser.add_argument('--no_id_label', action='store_true',
                        help='Whether to not use identity label to distinguish source from destinations. Value used to set label diffusion')
    parser.add_argument('--gpu', type=int, default=1, help='Specify which GPU to use')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for Monte Carlo Dropout')
    parser.add_argument('--N', type=int, default=3, help='Number of frequency bands')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def run(args):
    # setup seed
    setup_seed(args.seed)
    # set logging
    time_of_run = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'train_log_{}_{}_{}.log'.format(args.prefix, args.data, time_of_run))

    logging.basicConfig(filename=log_file, filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    log = logging.getLogger()
    log.info("Logging set up")

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    log.info(args)

    Path(args.saved_models_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = os.path.join(args.saved_models_dir, '{}-{}-{}.pth'.format(args.prefix, args.data, time_of_run))

    Path(args.saved_checkpoints_dir).mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopMonitor(max_round=args.patience)

    # get data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(args.data_dir, args.data, include_padding=True)

    # setup neighbour finder
    train_neighbor_finder = get_neighbor_finder(train_data, uniform=args.uniform_sampling)
    val_neighbor_finder = get_neighbor_finder(full_data, uniform=args.uniform_sampling)
    test_neighbor_finder = get_neighbor_finder(full_data, uniform=args.uniform_sampling)

    # create net and optimizer
    net = BPDRNet(emb_dim=args.emb_dim, edge_attr_size=edge_features.shape[1], edge_time_emb_dim=args.time_dim, num_layers=args.num_layers, use_fourier_features=not args.no_fourier_time_encoding, use_id_label= not args.no_id_label, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # train, val and test
    total_epoch_times = []
    best_model_path = os.path.join(args.saved_models_dir, '{}-{}-best.pth'.format(args.prefix, args.data))
    best_val_mrr = 0.0

    for epoch in range(args.num_epochs):
        # training
        loss, total_epoch_time, num_instance, pos_prev_scores, neg_prev_scores = training_epoch(
            net, optimizer, train_data, full_data, node_features, edge_features, train_neighbor_finder,
            args.train_batch_size, args.num_temporal_hops, args.num_neighbors, args.num_samples,
            args.verbose, args.coalesce_edges_and_time, args.train_randomize_timestamps
        )
        log.info('Epoch: {},  loss: {}, time: {:.3f}s'.format(epoch, loss, total_epoch_time))
        total_epoch_times.append(total_epoch_time)
        
        avg_score = np.mean(batch_scores[-num_instance:])
        prev_score = np.mean(previous_batch_scores[-num_instance:])
        epoch_average_scores.append(avg_score)
        epoch_previous_scores.append(prev_score)
        epoch_positive_previous_scores.append(pos_prev_scores)
        epoch_negative_previous_scores.append(neg_prev_scores)

        # transductive validation
        val_results = ranking_evaluation(net, val_data, full_data, node_features, edge_features, val_neighbor_finder, args.eval_batch_size, args.num_temporal_hops, args.num_neighbors, args.verbose, args.coalesce_edges_and_time)
        log.info('Transductive Validation time:{}'.format(val_results["time"]))
        log.info(val_results)

        # inductive validation
        nn_val_results = ranking_evaluation(net, new_node_val_data, full_data, node_features, edge_features, val_neighbor_finder, args.eval_batch_size, args.num_temporal_hops_eval, args.num_neighbors_eval, args.verbose, args.coalesce_edges_and_time)
        log.info('New Nodes Validation time:{}'.format(nn_val_results["time"]))
        log.info(nn_val_results)
                
        # best model
        if val_results["MRR"] > best_val_mrr:
            best_val_mrr = val_results["MRR"]
            torch.save(net.state_dict(), best_model_path)
            log.info(f'Best model saved with MRR: {best_val_mrr}')

    log.info(f'Loading the best model with MRR: {best_val_mrr}')
    net.load_state_dict(torch.load(best_model_path))
    log.info(f'Loaded the best model for inference')
    net.eval()

    # save model
    log.info('Saving model')
    torch.save(net.state_dict(), model_save_path)
    log.info('Model saved')

    # testing on best loaded model
    test_results = ranking_evaluation(net, test_data, full_data, node_features, edge_features,
                                      test_neighbor_finder, args.eval_batch_size, args.num_temporal_hops_eval,
                                      args.num_neighbors_eval, args.verbose, args.coalesce_edges_and_time)
    log.info('Transductive Testing time:{}'.format(test_results["time"]))
    log.info(test_results)
    nn_test_results = ranking_evaluation(net, new_node_test_data, full_data, node_features, edge_features,
                                         test_neighbor_finder, args.eval_batch_size,
                                         args.num_temporal_hops_eval, args.num_neighbors_eval, args.verbose, args.coalesce_edges_and_time)

    log.info('New Nodes Testing time:{}'.format(nn_test_results["time"]))
    log.info(nn_test_results)

    average_epoch_time = sum(total_epoch_times) / len(total_epoch_times)
    log.info('Average epoch time: {:.3f}s'.format(average_epoch_time))

    average_scores_filename = f'{args.data}_average_scores_{time_of_run}.npy'
    np.save(average_scores_filename, np.array(epoch_average_scores))
    log.info(f'Average scores saved to {average_scores_filename}')

    previous_scores_filename = f'{args.data}_previous_scores_{time_of_run}.npy'
    np.save(previous_scores_filename, np.array(epoch_previous_scores))
    log.info(f'Previous scores saved to {previous_scores_filename}')

    positive_previous_scores_filename = f'{args.data}_positive_previous_scores_{time_of_run}.npy'
    np.save(positive_previous_scores_filename, np.array(epoch_positive_previous_scores))
    log.info(f'Positive previous scores saved to {positive_previous_scores_filename}')

    negative_previous_scores_filename = f'{args.data}_negative_previous_scores_{time_of_run}.npy'
    np.save(negative_previous_scores_filename, np.array(epoch_negative_previous_scores))
    log.info(f'Negative previous scores saved to {negative_previous_scores_filename}')


    log.info('Done!')

if __name__ == '__main__':
    args = parse_arguments()
    run(args)
