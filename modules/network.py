import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

torch.set_printoptions(profile="full")

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

class T_Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(T_Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.embedding_dim = 64
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.resnet.rep_dim + self.embedding_dim, self.resnet.rep_dim),
            nn.ReLU()
        )
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, embedding_v = None):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        h_i = torch.cat((h_i, embedding_v), dim=1)
        h_j = torch.cat((h_j, embedding_v), dim=1)

        h_i = self.embedding_layer(h_i)
        h_j = self.embedding_layer(h_j)
        
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.network(z)

class ConvGenerator(nn.Module):
    def __init__(self, input_dim, output_channels, img_size):
        super(ConvGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.img_size = img_size
        
        self.fc = nn.Linear(input_dim, 128 * (img_size // 4) * (img_size // 4))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.img_size // 4, self.img_size // 4)
        img = self.conv_blocks(out)
        
        return img

class T_G_Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, image_size=60):
        super(T_G_Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.embedding_dim = 64
        self.rep_dim = self.resnet.rep_dim  # Assuming resnet has an attribute `rep_dim`

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.rep_dim + self.embedding_dim, self.rep_dim),
            nn.ReLU()
        )
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.feature_dim),
        )
        
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

        #self.generator = SimpleGenerator(512, self.resnet.input_dim)
        self.generator = ConvGenerator(512, 1, image_size)

    def forward(self, x_i, x_j=None, embedding_v=None):
        # Obtain initial embeddings from resnet
        h_i = self.resnet(x_i)
        # h_j = self.resnet(x_j)
        
        h_i = torch.cat((h_i, embedding_v), dim=1)
        # h_j = torch.cat((h_j, embedding_v), dim=1)
        
        h_i = self.embedding_layer(h_i)
        # h_j = self.embedding_layer(h_j)
        
        # Generate new images using the generator
        gen_x_i = self.generator(h_i)
        # gen_x_j = self.generator(h_j)
        
        # Save a contrastive diagram between the raw image and the generated one
        # self.save_contrastive(x_i, gen_x_i)

        # Get embeddings for generated images
        gen_h_i = self.resnet(gen_x_i)
        # gen_h_j = self.resnet(gen_x_j)

        # Combine original and generated embeddings

        gen_h_i = torch.cat((gen_h_i, embedding_v), dim=1)
        # gen_h_j = torch.cat((gen_h_j, embedding_v), dim=1)

        # Process through embedding layer       
        gen_h_i = self.embedding_layer(gen_h_i)
        # gen_h_j = self.embedding_layer(gen_h_j)

        # Instance-level projections
        z_i = normalize(self.instance_projector(h_i), dim=1)
        # z_j = normalize(self.instance_projector(h_j), dim=1)
        gen_z_i = normalize(self.instance_projector(gen_h_i), dim=1)
        # gen_z_j = normalize(self.instance_projector(gen_h_j), dim=1)

        # Cluster-level projections
        c_i = self.cluster_projector(h_i)
        # c_j = self.cluster_projector(h_j)
        gen_c_i = self.cluster_projector(gen_h_i)
        # gen_c_j = self.cluster_projector(gen_h_j)

        # Return embeddings and cluster assignments
        # return (z_i, z_j, gen_z_i, gen_z_j), (c_i, c_j, gen_c_i, gen_c_j)
        return z_i, gen_z_i, c_i, gen_c_i

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    def save_contrastive(self, x, gen_x):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(x)
        axs[1].imshow(gen_x)
        plt.savefig("contrastive_diagram.png")
        plt.close()
        
