import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

# FIX 1: Handle numpy warnings properly
# RankWarning was removed in newer numpy versions
warnings.filterwarnings('ignore')
try:
    from numpy.polynomial.polyutils import RankWarning
    warnings.simplefilter('ignore', RankWarning)
except (ImportError, AttributeError):
    # RankWarning doesn't exist in newer numpy, ignore
    pass

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')
# Add a new argument first (at the top with other arguments):
# parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti', 'kitti', 'custom'])
parser.add_argument('--split-dir', type=str, default=None,
                    help='Override split directory (default: dataset/splits/{dataset})')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--grad-accum', default=1, type=int, help='Gradient accumulation steps (effective_bs = bs * grad_accum)')
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--resume-from', type=str, default=None, help='Path to pretrained model or checkpoint')
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    # ← ADD THESE LINES:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)  # ← This is the missing one
    # FIX 2: Better CUDA device handling
    # Check CUDA availability early
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU setup.")
    
    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Verify the GPU is accessible
    try:
        torch.cuda.set_device(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
        print(f"Using GPU {local_rank}: {device_name}")
    except Exception as e:
        raise RuntimeError(f"Cannot access GPU {local_rank}: {e}")
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'kitti':
        trainset = KITTI('dataset/train_fixed1.txt', 'train', size=size)
    elif args.dataset == 'custom':
        from dataset.custom_depth_dataset import CustomDepthDataset
        split_dir = args.split_dir or 'dataset/splits/custom'
        trainset = CustomDepthDataset(f'{split_dir}/train.txt', 'train', img_size=args.img_size, min_depth=args.min_depth, max_depth=args.max_depth)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)

    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = VKITTI2('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'kitti':
        valset = KITTI('dataset/test_fixed1.txt', 'val', size=size)
    elif args.dataset == 'custom':
        from dataset.custom_depth_dataset import CustomDepthDataset
        split_dir = args.split_dir or 'dataset/splits/custom'
        valset = CustomDepthDataset(f'{split_dir}/val.txt', 'val', img_size=args.img_size, min_depth=args.min_depth, max_depth=args.max_depth)
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        if os.path.exists(args.pretrained_from):
            print(f"Loading pretrained weights from: {args.pretrained_from}")
            model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
        else:
            print(f"Warning: Pretrained file not found: {args.pretrained_from}")
    
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    criterion = SiLogLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    


    # Then after optimizer creation, add:
    # start_epoch = 0
    # if args.resume and args.pretrained_from:
    #     if os.path.exists(args.pretrained_from):
    #         checkpoint = torch.load(args.pretrained_from, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         start_epoch = checkpoint['epoch'] + 1
    #         previous_best = checkpoint.get('previous_best', previous_best)
    #         print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    start_epoch = 0

    # Handle resume from checkpoint (if provided)
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint.get('previous_best', previous_best)
        print(f"✓ Resumed from checkpoint at epoch {start_epoch}")
    total_iters = args.epochs * len(trainloader)

    # Paper Section 4.1: "cosine annealing schedule"
    import math
    grad_accum = args.grad_accum

    if rank == 0:
        effective_bs = args.bs * grad_accum * world_size
        logger.info(f'Effective batch size: {effective_bs} (bs={args.bs} x accum={grad_accum} x gpus={world_size})')
        logger.info(f'Total iterations: {total_iters}, LR schedule: cosine annealing')

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, sample in enumerate(trainloader):
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            # Squeeze only if extra channel dim exists (backward compat)
            if depth.dim() == 4:
                depth = depth.squeeze(1)
            if valid_mask.dim() == 4:
                valid_mask = valid_mask.squeeze(1)

            # Note: horizontal flip is now handled inside CustomDepthDataset
            # For non-custom datasets, keep the flip here
            if args.dataset != 'custom' and random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))

            # Gradient accumulation: scale loss and accumulate
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Cosine annealing LR schedule (Paper Section 4.1)
            iters = epoch * len(trainloader) + i
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))

            optimizer.param_groups[0]["lr"] = lr            # encoder (pretrained)
            optimizer.param_groups[1]["lr"] = lr * 10.0     # head (10x)

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            # Squeeze only if extra channel dim exists (backward compat)
            if depth.dim() == 3:
                depth = depth.squeeze(0)
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.squeeze(0)

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        torch.distributed.barrier()
        
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

            # Save best model based on d1 accuracy
            current_d1 = (results['d1'] / nsamples).item()
            if not hasattr(main, '_best_d1') or current_d1 > main._best_d1:
                main._best_d1 = current_d1
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
                logger.info(f'New best model saved! d1={current_d1:.4f}')

            # Save every 5 epochs for comparison
            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, os.path.join(args.save_path, f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()