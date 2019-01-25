import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Deep learning models for relevance transfer")
    parser.add_argument('--no-cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--model', type=str, default='LSTMBaseline', choices=['LSTMBaseline', 'LSTMRegularized', 'KimCNN'])
    parser.add_argument('--dataset', type=str, default='Robust04', choices=['Robust04', 'Robust05', 'Robust45'])
    parser.add_argument('--dev_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='relevance_transfer/saves')
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--data_dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'datasets'))
    parser.add_argument('--word_vectors_dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word_vectors_file', help='word vectors filename', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument("--output-path", type=str, help='output path for rank file', default="run.core17.lstm.topics.robust00.txt")
    parser.add_argument('--resume-snapshot', action='store_true')

    # Baseline LSTM parameters
    parser.add_argument('--bottleneck-layer', action='store_true')
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--bidirectional', action='store_true')

    # Regularized LSTM parameters
    parser.add_argument('--TAR', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--beta-ema', type=float, default = 0, help="for temporal averaging")
    parser.add_argument('--wdrop', type=float, default=0.0, help="for weight-drop")
    parser.add_argument('--embed-droprate', type=float, default=0.0, help="for embedded droupout")

    # KimCNN parameters
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--output_channel', type=int, default=100)

    # Reranking parameters
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument("--ret-ranks", type=str, help='retrieval rank file', default="run.core17.bm25+rm3.wcro0405.hits10000.txt")
    parser.add_argument("--clf-ranks", type=str, help='classification rank file', default="run.core17.lstm.topics.robust45.txt")

    args = parser.parse_args()
    return args
