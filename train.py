import os
import re
import csv
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from torch.utils.data import Dataset, Subset
from PIL import Image
from tqdm import tqdm

import faulthandler
faulthandler.enable()

class CustomDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transform.Transforms(size=args.image_size, s=0.5)
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        if args.transfer:
            pattern = re.compile(r"frame_\d+_cell_(\d+)\.0\.png")
            match = pattern.match(self.img_names[idx])
            if match:
                cell_id = match.group(1)
                if cell_id in data_dict:
                    embedding_v = data_dict[cell_id]

            else:
                print(f"Error parsing cell id from image name: {self.img_names[idx]}")
                input("Press Enter to continue...")

            return image, embedding_v
          
        else:
            return image, 0


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), emb_v) in enumerate(tqdm(data_loader)):
        try:
            optimizer.zero_grad()
            x_i = x_i.to(args.cuda_device)
            x_j = x_j.to(args.cuda_device)
            if args.transfer:
                emb_v = emb_v.to(args.cuda_device)
                z_i, z_j, c_i, c_j = model(x_i, x_j, emb_v)
            else:
                z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i, z_j)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_instance + loss_cluster
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
            loss_epoch += loss.item()
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    return loss_epoch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    set_seed(args.seed)

    data_dict = {}
    csv_path = "tracking_area13.csv"
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cell_id = row["cellid"]
            # Parse the features into a tensor
            f_tensor = torch.tensor([float(row[f"f{i}"]) for i in range(64)])
            # Add the tensor to the data dictionary
            data_dict[cell_id] = f_tensor

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200

    elif args.dataset == "microglia":
        #dataset = CustomDataset(img_dir='datasets/microglia/training_center')
        dataset = CustomDataset(img_dir=args.dataset_dir)
        print('Successfully loaded microglia dataset')

    else:
        raise NotImplementedError
    
    # Create subset
    subset_indices = np.random.choice(len(dataset), int(len(dataset)*args.ratio), replace=False)
    sub_dataset = Subset(dataset, subset_indices)
    
    data_loader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    
    if args.generate:
        model = network.T_G_Network(res, args.feature_dim, args.class_num, args.image_size)
    elif args.transfer:
        model = network.T_Network(res, args.feature_dim, args.class_num)
    else:
        model = network.Network(res, args.feature_dim, class_num)
        
    model = model.to(args.cuda_device)
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    # loss_device = torch.device("cuda")
    loss_device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.class_num, args.cluster_temperature, loss_device).to(loss_device)
    
    # train
    with open("loss.txt", "a") as f:
        for epoch in range(args.start_epoch, args.epochs):
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train()
            
            # Save model every epoch
            save_model(args, model, optimizer, epoch)
            
            # Remove the previous epoch's model unless it's a multiple of 10
            if epoch > args.start_epoch and (epoch - 1) % 10 != 0:
                prev_model_fp = os.path.join(args.model_path, f"checkpoint_{epoch - 1}.tar")
                if os.path.exists(prev_model_fp):
                    os.remove(prev_model_fp)
            
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
            f.write(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}\n")
        
    save_model(args, model, optimizer, args.epochs)
