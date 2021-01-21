from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.domain_adaptation import DA
from reid.datasets.TotalData import TotalData
from reid import models
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.loss import TripletLoss, InvNet
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, IdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

def fix(s):
    import torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(s)
    import random
    random.seed(s)

def get_data(data_dir, height, width, batch_size, num_instances, re=0, workers=8):

    dataset = DA(data_dir)
    test_dataset = TotalData(data_dir)



    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_source_ids

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.Pad(10),
        T.RandomCrop((256,128)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5), 
        T.ColorJitter(brightness=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.ToTensor(),
        normalizer,
        # T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    
    # Train
    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        # shuffle=True, pin_memory=True, drop_last=True)
        sampler=RandomIdentitySampler(dataset.source_train, batch_size, num_instances),
        pin_memory=True, drop_last=True) 

    # Test
    grid_query_loader = DataLoader(
        Preprocessor(test_dataset.grid_query,
                     root=osp.join(test_dataset.grid_images_dir, test_dataset.query_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    grid_gallery_loader = DataLoader(
        Preprocessor(test_dataset.grid_gallery,
                     root=osp.join(test_dataset.grid_images_dir, test_dataset.gallery_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    prid_query_loader = DataLoader(
        Preprocessor(test_dataset.prid_query,
                     root=osp.join(test_dataset.prid_images_dir, test_dataset.query_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    prid_gallery_loader = DataLoader(
        Preprocessor(test_dataset.prid_gallery,
                     root=osp.join(test_dataset.prid_images_dir, test_dataset.gallery_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    viper_query_loader = DataLoader(
        Preprocessor(test_dataset.viper_query,
                     root=osp.join(test_dataset.viper_images_dir, test_dataset.query_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    viper_gallery_loader = DataLoader(
        Preprocessor(test_dataset.viper_gallery,
                     root=osp.join(test_dataset.viper_images_dir, test_dataset.gallery_path), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    ilid_query_loader = DataLoader(
        Preprocessor(test_dataset.ilid_query,
                     root=osp.join(test_dataset.ilid_images_dir, "images"), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)
    ilid_gallery_loader = DataLoader(
        Preprocessor(test_dataset.ilid_gallery,
                     root=osp.join(test_dataset.ilid_images_dir, "images"), transform=test_transformer),
        batch_size=64, num_workers=4,
        shuffle=False, pin_memory=True)


    return dataset, test_dataset, num_classes, source_train_loader, grid_query_loader, grid_gallery_loader,prid_query_loader, prid_gallery_loader,viper_query_loader, viper_gallery_loader, ilid_query_loader, ilid_gallery_loader


def main(args):
    fix(args.seed)
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    
    print(args)
    # Create data loaders
    dataset, test_dataset, num_classes, source_train_loader, grid_query_loader, grid_gallery_loader,prid_query_loader, prid_gallery_loader,viper_query_loader, viper_gallery_loader, ilid_query_loader, ilid_gallery_loader = \
        get_data(args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instance, args.re, args.workers)

    # Create model
    Encoder, Transfer, CamDis = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    invNet = InvNet(args.features, num_classes, args.batch_size, beta=args.beta, knn=args.knn, alpha=args.alpha).cuda()

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        Encoder.load_state_dict(checkpoint['Encoder'])
        Transfer.load_state_dict(checkpoint['Transfer'])
        CamDis.load_state_dict(checkpoint['CamDis'])
        invNet.load_state_dict(checkpoint['InvNet'])
        start_epoch = checkpoint['epoch']

    Encoder = Encoder.cuda()
    Transfer = Transfer.cuda()
    CamDis = CamDis.cuda()

    model = [Encoder, Transfer, CamDis]
    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        # -----------------------------
        v = evaluator.eval_viper(viper_query_loader, viper_gallery_loader, test_dataset.viper_query, test_dataset.viper_gallery, args.output_feature, seed=57)
        p = evaluator.eval_prid(prid_query_loader, prid_gallery_loader, test_dataset.prid_query, test_dataset.prid_gallery, args.output_feature, seed=40)
        g = evaluator.eval_grid(grid_query_loader, grid_gallery_loader, test_dataset.grid_query, test_dataset.grid_gallery, args.output_feature, seed=35)
        l = evaluator.eval_ilids(ilid_query_loader, test_dataset.ilid_query, args.output_feature, seed=24)
        # -----------------------------

    criterion = []
    criterion.append(nn.CrossEntropyLoss().cuda())
    criterion.append(TripletLoss(margin=args.margin))


    # Optimizer
    base_param_ids = set(map(id, Encoder.base.parameters()))
    new_params = [p for p in Encoder.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': Encoder.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer_Encoder = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    # ====
    base_param_ids = set(map(id, Transfer.base.parameters()))
    new_params = [p for p in Transfer.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': Transfer.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer_Transfer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    # ====
    param_groups = [
        {'params':CamDis.parameters(), 'lr_mult':1.0},
    ]
    optimizer_Cam = torch.optim.SGD(param_groups, lr=args.lr,momentum=0.9, weight_decay=5e-4, nesterov=True)

    optimizer = [optimizer_Encoder, optimizer_Transfer, optimizer_Cam]

    # Trainer
    trainer = Trainer(model, criterion, InvNet=invNet)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** ((epoch) // step_size))
        for g in optimizer_Encoder.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Transfer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Cam.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, source_train_loader, optimizer, args.tri_weight, args.adv_weight, args.mem_weight)

        save_checkpoint({
            'Encoder': Encoder.state_dict(),
            'Transfer': Transfer.state_dict(),
            'CamDis': CamDis.state_dict(),
            'InvNet': invNet.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        evaluator = Evaluator(model)
        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # seed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for source")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    parser.add_argument('--num-instance', type=int, default=4)
    parser.add_argument('--tri-weight', type=float, default=0.3)
    parser.add_argument('--margin',type=float,default=0.3)

    parser.add_argument('--adv-weight', type=float, default=0.5)
    parser.add_argument('--mem-weight', type=float, default=0.5)
    parser.add_argument('--knn', type=int, default=3)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
