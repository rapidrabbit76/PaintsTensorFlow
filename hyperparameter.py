# hyperparame

#  Dont maind
num_files = 0
batch_steps = 0

# for Model
gf_dim = 64
df_dim = 64  # Dimension of gen filters in first conv layer. [64]
c_dim = 3  # n_color 3

lr = 1e-4
beta1 = 0.5
beta2 = 0.9

l1_scaling = 100
l2_scaling = 10

# for Train
epoch = 1
batch_size = 1
log_interval = 10
sampleing_interval = 200
save_interval = 4000

# data & save Path
train_image_datasets_path = ""
train_line_datasets_path = ""
test_image_datasets_path = ""
test_line_datasets_path = ""

samples_path = "./samples"
checkpoint_path = "./ckpt"
log_path = "./board/log"
