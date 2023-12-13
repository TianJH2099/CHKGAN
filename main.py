import argparse
from data_loader import MyDataset
from model import CHKGAT
from train import train

parser = argparse.ArgumentParser(description="Model Setting.")
# program setting
parser.add_argument('--data_name', type=str, default='amazon-book', help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--epoch', type=int, default=100, help='Number of epoch.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing.')
parser.add_argument('--optim', type=str, default='adam', help='Optimizer [sgd, adam].')
parser.add_argument('--device', type=str, default='cuda', help='Device [cuda, cpu].')
parser.add_argument('--log_step', type=int, default=10, help='Log step.')
parser.add_argument('--num_agg', type=int, default=1, help='Number of aggregation.')
parser.add_argument('--agg_step', type=int, default=10, help='Number of aggregation.')
parser.add_argument('--train_ratio', type=float, default=0.98, help='ratio of train and valid')

# file saving setting
parser.add_argument('--log_path', type=str, default=r'mine\log', help='Log path.')
parser.add_argument('--model_path', type=str, default=r'mine\checkpoints', help='Model path.')

# model setting
parser.add_argument('--dim', type=int, default=100, help='Dimension of embedding.')
parser.add_argument('--head_nums', type=int, default=2, help='Number head of multi-head self-attention.')
parser.add_argument('--dropout', type=float, default=0.5, help='feature dropout persent.')
parser.add_argument('--aggregator', type=str, default='sum', help='Type of aggregator [sum, replace, mean].')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold of predict')
parser.add_argument('--pretrain_embed', type=bool, default=True, help='using pretarined embedding of entity and relation')

args = parser.parse_args()

data = MyDataset(args.data_name)
num_user, num_item, num_relation, num_entity  = data._get_num()
model = CHKGAT(args, num_entity=num_entity, num_relation=num_relation, num_item=num_item)

train(args, model, data)
