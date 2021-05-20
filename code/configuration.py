# root_dir = r'C:\Users\anant\Downloads\OpenImages\dummy_data'
root_dir = '/home/aprasannakumar/data/openimages/'
#root_dir = '/home/aprasannakumar/oid/dummy_data/'

data_train_folder = 'train'
data_val_folder = 'validation'
data_train_csv = 'challenge-2019-train-vrd.csv'
data_val_csv = 'challenge-2019-validation-vrd.csv' 

logs_dir = './results/logs_dir'
saved_models_dir = './results/saved_models'

num_classes = 10
batch_size = 32
num_epochs = 1
fraction = 0.4
learning_rate = 1e-4
weight_decay = 1e-6
momentum = 0.9
route = [0, 1, 2]
model = ["resnet", "resnet1", "resnet2", "inception", "inception2"]