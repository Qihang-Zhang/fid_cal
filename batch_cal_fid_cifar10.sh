#cifar10
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/imagenet32
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/cifar10
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/cifar100
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/mnist
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/fashionmnist
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/svhn
$ENV_PATH/pcnn/bin/python fid_cal.py --ref_data_dir ../all_imgs/cifar10 --new_data_dir ../all_imgs/celeba

