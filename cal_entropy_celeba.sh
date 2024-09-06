#celeba
for dataset_q in celeba cifar10 cifar100 fashionmnist imagenet32 mnist svhn 
do
    for dataset_p in celeba cifar10 cifar100 fashionmnist imagenet32 mnist svhn
    do
        $ENV_PATH/pcnn/bin/python cross_entropy_estimate.py --dataset_q $dataset_q --dataset_p $dataset_p
    done
done
