import argparse

import sys, copy
sys.path.append('..')

from configs.amort import *
from amortizer.ans_trainer import AnSTrainer
# from amortizer.ans_trainer_masking import AnSTrainerParameterMasking

parser = argparse.ArgumentParser(description='Training option')

parser.add_argument('--mode', type=str, default='full', choices=['full', 'base', 'abl0', 'abl1', 'abl2', 'abl3'])
parser.add_argument('--amort', type=str, default='pte', choices=['inn', 'pte'])

# MLP for stat data
parser.add_argument('--mlp_feat', type=int, default=128)
parser.add_argument('--mlp_out', type=int, default=64)

### Encoder for trajectory data
parser.add_argument('--enc_type', type=str, default='transformer', choices=['conv_rnn', 'transformer', 'None'])

# RNN
parser.add_argument('--rnn_feat', type=int, default=16)
parser.add_argument('--rnn_depth', type=int, default=2)

# Transformer
parser.add_argument('--tr_query', type=int, default=16)
parser.add_argument('--tr_out', type=int, default=16)
parser.add_argument('--tr_attn_dropout', type=float, default=0.4)

# INN
parser.add_argument('--inn_block_feat', type=int, default=32)

# PTE
parser.add_argument('--pte_hid_sz', type=int, default=256)
parser.add_argument('--pte_hid_depth', type=int, default=2)


# Training setup
parser.add_argument('--step_per_iter', type=int, default=2048)
parser.add_argument('--batch_sz', type=int, default=64)
parser.add_argument('--n_trial', type=int, default=64)
parser.add_argument('--save_freq', type=int, default=15)
parser.add_argument('--n_iter', type=int, default=150)

# Load
parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--load_model', type=str, default="base_tf_q16_o4_do0.4-mlp_f64_o48-inn_f32-tr_it2048_b64_n64_basee_231121_152827_18_30000000")
parser.add_argument('--load_session', type=str, default="1122_175456")

args = parser.parse_args()

### Configuration setup
cfg = copy.deepcopy(default_ans_config[args.mode]) # if args.masking == 'None' else copy.deepcopy(masking_ans_config[args.masking][args.mode])
name = f"{cfg['name']}_"

if args.amort == 'inn': cfg["point_estimation"] = False
elif args.amort == 'pte': cfg["point_estimation"] = True
cfg["amortizer"]["encoder"]["mlp"]["feat_sz"] = args.mlp_feat
cfg["amortizer"]["encoder"]["mlp"]["out_sz"] = args.mlp_out
cfg["amortizer"]["encoder"]["traj_encoder_type"] = args.enc_type
cfg["amortizer"]["encoder"]["rnn"]["feat_sz"] = args.rnn_feat
cfg["amortizer"]["encoder"]["rnn"]["depth"] = args.rnn_depth
cfg["amortizer"]["encoder"]["transformer"]["query_sz"] = args.tr_query
cfg["amortizer"]["encoder"]["transformer"]["out_sz"] = args.tr_out
cfg["amortizer"]["encoder"]["transformer"]["attn_dropout"] = args.tr_attn_dropout
cfg["amortizer"]["invertible"]["block"]["feat_sz"] = args.inn_block_feat
cfg["amortizer"]["linear"]["hidden_sz"] = args.pte_hid_sz
cfg["amortizer"]["linear"]["hidden_depth"] = args.pte_hid_depth


if args.enc_type == "conv_rnn":
    name += f"rn_f{args.rnn_feat}_d{args.rnn_depth}-"
elif args.enc_type == "transformer":
    name += f"tf_q{args.tr_query}_o{args.tr_out}_do{args.tr_attn_dropout:.1f}-"
elif args.enc_type == "None":
    name += f"so-"      # Statonly
    cfg["amortizer"]["encoder"]["traj_sz"] = 0
    cfg["amortizer"]["encoder"]["traj_encoder_type"] = None

if args.mode in ['abl2', 'abl3']:
    cfg["amortizer"]["encoder"]["traj_encoder_type"] = None

name += f"mlp_f{args.mlp_feat}_o{args.mlp_out}"

if args.amort == 'inn':
    name += f"-inn_f{args.inn_block_feat}"
elif args.amort == 'pte':
    name += f"-pte_{args.pte_hid_sz}x{args.pte_hid_depth}"

name += f"-tr_it{args.step_per_iter}_b{args.batch_sz}_n{args.n_trial}_{cfg['simulator']['policy']}_{cfg['simulator']['checkpt']}"

cfg["name"] = name
trainer = AnSTrainer(config=cfg, mode=args.mode)

if args.load_ckpt:
    trainer.load(
        args.load_model,
        args.load_session
    )

trainer.train(
    n_iter=args.n_iter,
    step_per_iter=args.step_per_iter,
    batch_sz=args.batch_sz,
    n_trial=args.n_trial,
    save_freq=args.save_freq
)