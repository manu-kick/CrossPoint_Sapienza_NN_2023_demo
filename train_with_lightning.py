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

from models.dgcnn import DGCNN, ResNet, DGCNN_partseg
from torchvision.models import resnet50, resnet18



class CrosspointLightning (pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        print("Init CrosspointLightning")
        self.args = args

        # ------------
        # models
        # ------------
        if args.model == 'dgcnn':
            self.point_model = DGCNN(args) #todo in lightning
        elif args.model == 'dgcnn_seg':
            self.point_model = DGCNN_partseg(args) #todo in lightning
        else:
            raise Exception("Not implemented")

        self.img_model = ResNet(resnet50(), feat_dim = 2048) #todo in lightning
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