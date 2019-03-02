# hyperparame

#  Do not mind
batch_steps = 0

# for Model
gf_dim = 64
df_dim = 64
c_dim = 3

lr = 1e-5
beta1 = 0.9
beta2 = 0.99

l1_scaling = 100
l2_scaling = 10

# for Train
epoch = 2
batch_size = 4

log_interval = 10
sampling_interval = 200
save_interval = 4000

# data & save Path
train_image_datasets_path = "/home/yslee/datasets/PaintsTensorflow/512px/train/image/*.*"
train_line_datasets_path = "/home/yslee/datasets/PaintsTensorflow/512px/train/line/*.*"
test_image_datasets_path = "/home/yslee/datasets/PaintsTensorflow/512px/test/image/*.*"
test_line_datasets_path = "/home/yslee/datasets/PaintsTensorflow/512px/test/line/*.*"
