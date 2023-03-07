# DATA
dataset='Tusimple'
data_root = '/media/zxysilent/data/app/Tusimple/train_set'

# TRAIN
# epoch = 100
epoch = 200
batch_size = 102
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
# backbone = '18'
# backbone = '18x' #different pool method
backbone = '18p' #different conv block with pool
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
# note = '_zbt_DSCA'
note = '_zbt_mas_DSCSA'
log_path = ''

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = ''
test_work_dir = './tmp'