import argparse
import pytorch_lightning as pl
import torch
from torchvision import transforms
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
from torch.utils.data import DataLoader
from datasets.data import ShapeNetRender, ModelNet40SVM
from util import IOStream, AverageMeter
import torch.nn as nn
import torch.nn.functional as F

#from models.dgcnn import DGCNN, ResNet, DGCNN_partseg
from torchvision.models import resnet50, resnet18



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    
    #Mio commento----
    #device = torch.device('cuda:1')
    #----
    if (torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #------

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(pl.LightningModule):
    def __init__(self,args,cls = -1) -> None:
        super(DGCNN,self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
    
    
        if cls != -1:
            self.linear1 = nn.Linear(self.args.emb_dims*2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=self.args.dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=self.args.dropout)
            self.linear3 = nn.Linear(256, output_channels)
        
        self.cls = cls
        
        self.inv_head = nn.Sequential(
                            nn.Linear(self.args.emb_dims * 2, args.emb_dims),
                            nn.BatchNorm1d(self.args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.args.emb_dims, 256)
                            )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        feat = x
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)
        
        inv_feat = self.inv_head(feat)
        
        return x, inv_feat, feat 
    
class ResNet(pl.LightningModule):
    def __init__(self, model, feat_dim = 2048):
        super(ResNet, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Identity()
        
        self.inv_head = nn.Sequential(
                            nn.Linear(feat_dim, 512, bias = False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256, bias = False)
                            ) 
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.inv_head(x)
        
        return x


class CrosspointLightning (pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        print("Init CrosspointLightning")
        self.args = args

        # ------------
        # models
        # ------------
        if args.model == 'dgcnn':
            self.point_model = DGCNN(args) 
        elif args.model == 'dgcnn_seg':
            self.point_model = DGCNN_partseg(args) #todo in lightning
        else:
            raise Exception("Not implemented")

        self.img_model = ResNet(resnet50(), feat_dim = 2048) 
        self.criterion = NTXentLoss(temperature = 0.1)

        if args.enable_wandb == True:
            wandb.watch(point_model)

        if args.resume:
            model.load_state_dict(torch.load(args.model_path))
            print("Model Loaded !!")
        
    def training_step(self, batch, batch_idx):
        (data_t1 ,data_t2), imgs = batch
        batch_size = data_t1.size()[0]

        data = torch.cat((data_t1, data_t2))
        data = data.transpose(2, 1).contiguous()
        _, point_feats, _ = self.point_model(data)
        img_feats = self.img_model(imgs)

        point_t1_feats = point_feats[:batch_size, :]
        point_t2_feats = point_feats[batch_size: , :]

        loss_imid = self.criterion(point_t1_feats, point_t2_feats)        
        point_feats = torch.stack([point_t1_feats,point_t2_feats]).mean(dim=0)
        loss_cmid = self.criterion(point_feats, img_feats)

        total_loss = loss_imid + loss_cmid

        return total_loss
    
    def forward(self, x):
        return x
    
    def configure_optimizers(self):
        parameters = list(self.point_model.parameters()) + list(self.img_model.parameters())

        # ------------
        # optimizer
        # ------------
        if self.args.use_sgd:
            print("Use SGD")
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-6)
        else:
            print("Use Adam")
            optimizer = torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=1e-6)
        
        # ------------
        # scheduler
        # ------------
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1)


        return [optimizer], [lr_scheduler]


def train(args,io):

    if args.enable_wandb == True:
        print('Wandb Enabled')
        wandb.init(project="CrossPoint", name=args.exp_name)

    
    # ------------
    # data
    # ------------
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_loader = DataLoader(ShapeNetRender(transform, n_imgs = 2), num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
     
    # ------------
    # model
    # ------------
    model = CrosspointLightning(args)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args,max_epochs=-1)
    trainer.fit(model, train_loader)


def cli_main():
    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',choices=['dgcnn', 'dgcnn_seg'],help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--enable_wandb', type=bool, default=False, help='Enable wandb (Default False)')
    args = parser.parse_args()

    # ------------
    # device
    # ------------
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
   

    # ------------
    # training or testing
    # ------------ 
    if not args.eval:
        train(args, io)


if __name__ == '__main__':
    cli_main()