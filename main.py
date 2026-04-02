import matplotlib
matplotlib.use('Agg')
import os
import glob
import time
import csv
import numpy as np
import random
import argparse
import sys
import datetime
import tqdm
import matplotlib.pyplot as plt
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from dataloader.wildtrack import Wildtrack
from dataloader.multiviewx import MultiviewX
from dataloader.dataloader import GetDataset
from dataloader.concat_dataset import ConcatDataset
from utils.logger import Logger
from utils import basic, utils
from evaluation.evaluate import evaluate
from loss import Loss, FocalLoss
from multiview_model import MultiView_Detection
from encoder import resnet18
from pytorch_metric_learning.losses import SupConLoss
    
###################################################################################################################################
score = []
prev, idx = 0, 0
accum_steps = 16

# Functions Train / Test
def isbest(new, epoch):
    global prev,idx
    if new > prev:
        prev = new
        idx = epoch
        return 1
    return 0

def get_score(score): # 최고 moda 반환
    score = np.asarray(score)
    return score[np.argmax(score,0)[0]]

def cleanup(logdir, idx):
    outputmap = 'map_'+str(idx)+'.jpg'
    outputfile1 = 'test_gta_scene5_'+str(idx)+'.txt'
    outputfile2 = 'test_gta_scene6_'+str(idx)+'.txt'
    retain = [outputmap, outputfile1, outputfile2]
    os.chdir(logdir)
    for img in glob.glob('map_*.jpg'):
        if img not in retain:
            os.remove(img)
    #for outfile in glob.glob('test*.txt'):
    #    if outfile not in retain:
    #        os.remove(outfile)


def sigmoid(x): #시그모이드 함수 klcc loss에서 에러 발생함 텐서 타입?
    # return 1/(1 + np.exp(-x))
    return torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)

def init_fn(worker_id):
    np.random.seed(int(args.seed))
    
def _traget_transform(target, kernel):
    with torch.no_grad():
        target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target
    

def train(model, epoch, data_loader, optimizer, log_interval, scheduler=None):
    model.train()
    tic = time.time()

    losses, ignore_cam, duplicate_cam = 0, 0, 0
    
    remainder = len(data_loader) % accum_steps

    if args.dropview:
        if args.cam_set:
            ignore_cam = random.choice(args.train_cam)
            duplicate_cam = random.choice([i for i in args.train_cam if ignore_cam!=i])
        else:
            ignore_cam = random.randint(0, data_loader.dataset.num_cam-1)
            duplicate_cam = random.choice([i for i in range(data_loader.dataset.num_cam) if ignore_cam!=i])
        print('Ignore cam : ', ignore_cam)
        if not args.avgpool:
            print('Duplicate cam : ', duplicate_cam)
    for batch_idx, (item, target) in enumerate(data_loader):
        data = item['img'].to(device)

        last_batch = (batch_idx + 1) == len(data_loader)
        is_accum_step = (batch_idx + 1) % accum_steps == 0
        

        bev_occupancy, bev_offset, bev_reid, img_reid = model(train_dataset, item['root'], data, ignore_cam, duplicate_cam,args.dropview, args.train_cam)

        
        # occupancy gt
        gt_occ = target['center_bev'].to(bev_occupancy.device)
        gt_valid_bev = target['valid_bev'].to(bev_occupancy.device)
        gt_pid_bev = target['pid_bev'].to(bev_occupancy.device)
        gt_offset = target['offset_bev'].to(bev_occupancy.device)
        
        # img gt
        B, S = target['pid_img'].shape[:2]
        gt_pid_img = basic.pack_seqdim(target['pid_img'].to(bev_occupancy.device), B)
        gt_valid_img = basic.pack_seqdim(target['valid_img'].to(bev_occupancy.device), B)
        
        # bev loss
        # occ_loss = criterion(sigmoid(bev_occupancy), gt_occ)
        occ_loss = occupancy_loss(sigmoid(bev_occupancy), gt_occ) # focal loss
        offset_loss = torch.abs(bev_offset - gt_offset).sum(dim=1, keepdim=True)
        offset_loss = 10 * basic.reduce_masked_mean(offset_loss, gt_valid_bev)
        
        # reid loss
        gt_valid_bev = gt_valid_bev.flatten(1)  # [1, 43200]
        gt_valid_img = gt_valid_img.flatten(1)  # [7, 14400]
        gt_pid_bev = gt_pid_bev.flatten(2).transpose(1,2)     # [1, 43200, 1]
        gt_pid_img = gt_pid_img.flatten(2).transpose(1,2)     # [7, 14400, 1]
        bev_reid = bev_reid.flatten(2).transpose(1,2)  # [1, 43200, C=64]
        img_reid = img_reid.flatten(2).transpose(1,2)  # [7, 14400, C=64]

        gt_reid = torch.cat([gt_pid_bev[gt_valid_bev],
                             gt_pid_img[gt_valid_img]]).squeeze(-1) 
        
        feats = torch.cat([bev_reid[gt_valid_bev],
                           img_reid[gt_valid_img]]) 

        # reid_loss = contrastive_loss(feats, gt_reid)

        #uncertainty weighting
        center_factor = 1 / (2 * torch.exp(model.center_weight))
        occ_loss = 10.0 * center_factor * occ_loss
        center_uncertainty_loss = 0.5 * model.center_weight

        # occ_loss = torch.exp(model.center_weight) * occ_loss
        # center_uncertainty_loss = 0.5 * model.center_weight

        offset_factor = 1 / (2 * torch.exp(model.offset_weight))
        offset_loss = offset_factor * offset_loss
        offset_uncertainty_loss = 0.5 * model.offset_weight

        total_loss = occ_loss + center_uncertainty_loss + offset_loss + offset_uncertainty_loss
        
        remain = remainder > 0 and (batch_idx >= len(data_loader) - remainder)
        denom = remainder if remain else accum_steps
        
        loss = total_loss / denom
        loss.backward()

        if is_accum_step or last_batch:
            optimizer.step()
            optimizer.zero_grad()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        losses += total_loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {}, Batch:{}/{},\tTotal_Loss: {:.9f}, occ_loss: {:.9f}, offset_loss: {:.9f}, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), losses / (batch_idx + 1), occ_loss.item(), offset_loss.item(), (time.time()-tic)/60, sigmoid(bev_occupancy).max()))
            pass

    print('Train Epoch: {}, Batch:{}, \tTotal_Loss: {:.9f}, occ_loss: {:.9f}, offset_loss: {:.9f}, \tTime: {:.3f}(min)'.format(
            epoch, len(data_loader), losses / len(data_loader), occ_loss.item(), offset_loss.item(), (time.time()-tic)/60))

    return losses / len(data_loader)


def test(model, epoch, data_loader, res_fpath=None, visualize=False):
    model.eval()

    all_res_list = []    
    for batch_idx, (item, target) in enumerate(data_loader):
        data = item['img'].to(device)
        with torch.no_grad():
            bev_occupancy, bev_offset, bev_reid, img_reid = model(test_dataset, item['root'], data, 0, 0, False, args.test_cam)

            xy, scores, id = utils.decoder(sigmoid(bev_occupancy), bev_offset, bev_reid, K=60)
            xy = xy[0]  # (K, 2) K개의 객체 좌표
            scores = scores[0]  # (K, 1) 각 객체의 confidence score
            # id = id[0]  # (K, C) 각 객체의 ID
            keep = scores.squeeze(-1) > float(args.cls_thres)
            xy = xy[keep]
            scores = scores[keep]

            if res_fpath is not None:
                frame_val = torch.as_tensor(item['frame'], device=xy.device, dtype=xy.dtype)
                dataset_idx = torch.tensor(
                    data_loader.dataset.dataset_list.index(item['root'][0]),
                    device=xy.device,
                    dtype=xy.dtype,
                )
                if data_loader.dataset.dicts[item['root'][0]]['base'].base.indexing == 'xy':
                    xy_save = xy
                else:
                    xy_save = xy[:, [1, 0]]
                all_res_list.append(torch.cat([
                    torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype) * frame_val,
                    xy_save * data_loader.dataset.dicts[item['root'][0]]['base'].grid_reduce,
                    scores,
                    torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype) * dataset_idx], dim=1))

    if res_fpath is not None:
        all_res_list = torch.cat(all_res_list, dim=0)
        np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.cpu().numpy(), '%.8f')
        recall, precision, moda, modp = [], [], [], []
        for dataset_idx, dataset_name in enumerate(data_loader.dataset.dataset_list):
            res_list = []
            all_res_list_temp = all_res_list[all_res_list[:, 4] == dataset_idx, :]
            for frame in np.unique(all_res_list_temp[:, 0].cpu().numpy()):
                res = all_res_list_temp[all_res_list_temp[:, 0] == frame, :]
                positions = res[:, 1:3]
                count = positions.shape[0]
                res_list.append(torch.cat([torch.ones([count, 1], device=positions.device) * frame, positions], dim=1))
            res_list = torch.cat(res_list, dim=0).cpu().numpy() if res_list else np.empty([0, 3])
            res_map_grid = res_list[res_list[:, 0] == int(item['frame']), 1:]
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/test_' + str(epoch) + '.txt', res_list[:, :3], '%d')

            lrecall, lprecision, lmoda, lmodp = evaluate(
                os.path.abspath(os.path.dirname(res_fpath)) + '/test_' + str(epoch) + '.txt',
                os.path.abspath(data_loader.dataset.dicts[dataset_name]['base'].base.gt_fname),
                data_loader.dataset.dicts[dataset_name]['base'].base.__name__)

            moda.append(lmoda)
            modp.append(lmodp)
            precision.append(lprecision)
            recall.append(lrecall)

        moda_avg = np.average(moda)
        modp_avg = np.average(modp)
        prec_avg = np.average(precision)
        rec_avg = np.average(recall)
        score.append([moda_avg, modp_avg, prec_avg, rec_avg])

        print(f'####### moda: {moda_avg:.1f}%, modp: {modp_avg:.1f}%, prec: {prec_avg:.1f}%, recall: {rec_avg:.1f}% #######')
    else:
        moda = 0

    map_res_nms = np.zeros(data_loader.dataset.dicts[dataset_name]['base'].reducedgrid_shape)
    for ij in res_map_grid:
        i, j = (ij / data_loader.dataset.dicts[dataset_name]['base'].grid_reduce).astype(int)
        if data_loader.dataset.dicts[dataset_name]['base'].base.indexing == 'xy':
            i, j = j, i
            map_res_nms[i, j] = 1
        else:
            map_res_nms[i, j] = 1
    
    #
    map_res_nms = _traget_transform(torch.from_numpy(map_res_nms).unsqueeze(0).unsqueeze(0).float(), data_loader.dataset.dicts[dataset_name]['base'].map_kernel) #gaussian kernel 적용
    map_res_nms = F.interpolate(map_res_nms, data_loader.dataset.dicts[dataset_name]['base'].reducedgrid_shape).squeeze().numpy() #reducedgrid_shape로 보간
    map_res_nms = np.uint8(255.0*map_res_nms) #이미지로 저장
    
    if visualize and epoch!=0:
        fig = plt.figure()
        subplt0 = fig.add_subplot(131, title="output")
        subplt1 = fig.add_subplot(132, title="target")
        subplt2 = fig.add_subplot(133, title="after nms")
        subplt0.imshow(sigmoid(bev_occupancy).detach().cpu().numpy().squeeze())
        subplt1.imshow(target['center_bev'].detach().cpu().numpy().squeeze())
        subplt2.imshow(map_res_nms)
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, 'map_'+str(epoch)+'.jpg'))
        plt.close(fig)
        
    return np.average(moda)


###################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx', 'gmvd_test','gmvd_train'], help='Choose dataset wildtrack/multiviewx (default: wildtrack)')
    parser.add_argument('-l', '--loss', type=str, default='klcc', choices=['klcc', 'mse'], help='Choose loss function klcc/mse. (default:klcc)' )
    parser.add_argument('-pr', '--pretrained', default=False, action='store_true', help='Use pretrained weights (default: True)')
    parser.add_argument('-cd', '--cross_dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--max_lr', type=float, default=1e-2, metavar='max LR', help=' max learning rate (default: 1e-2)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--p', type=float, default=1.0, help='hyper parameter to control CC loss')
    parser.add_argument('--k', type=float, default=1.0, help='hyper parameter to control KLDiv loss')
    parser.add_argument('--earlystop',  type=int, default=0, help='Store chkpt for particular epoch number (default: 0)')
    parser.add_argument('--avgpool', default=False, action='store_true', help='Enable average pooling (default: False)')
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='Choose optimizer.(default : SGD)')
    parser.add_argument('--step_size',default=1, type=int)
    parser.add_argument('--gamma',default=0.1, type=float)
    parser.add_argument('--lr_sched',type=str, default='onecycle_lr', choices=['step_lr', 'onecycle_lr'], help='Choose lr scheduler. (default:onecycle_lr)')
    parser.add_argument('--cam_set', default=False, action='store_true', help='Enable different camera set training and testing (default: False)')
    parser.add_argument("--train_cam", nargs="+", default=[])
    parser.add_argument("--test_cam", nargs="+", default=[])
    parser.add_argument('--dropview', default=False, action='store_true', help='Enable drop view training(default: False)')
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        
    if args.loss=='klcc':
        args.cls_thres = sigmoid(args.cls_thres)
        args.momentum = 0.9
        if args.batch_size != 0 and args.batch_size <= 2:
            args.lr = 5e-4 * args.batch_size
            args.max_lr = 5e-3 * args.batch_size

    if not args.avgpool and args.loss=='mse':
        args.lr = 0.1
        args.max_lr = 0.1

    if args.dropview:
        args.avgpool = True
        
    args.train_cam = list(map(int, args.train_cam))
    args.test_cam = list(map(int, args.test_cam))
    
    #Logging
    logdir = f'logs/{args.dataset}_'+datetime.datetime.today().strftime('%d_%m_%Y_%H_%M')
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    
    if args.cam_set:
        #if not args.avgpool:
        #    args.avgpool = True
        #    print('\nSetting {avgpool = True}\n')
        if len(args.train_cam) == 0 or len(args.test_cam) == 0:
            print('\nTrain/Test camera set is empty... Setting {cam_set = False}.\n')
            args.cam_set = False
            exit()
            
    args.train_cam = [x-1 for x in args.train_cam]
    print(args.train_cam)
    args.test_cam = [x-1 for x in args.test_cam]
    print(args.test_cam)
    
     # Dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])
    
    if 'wildtrack' in args.dataset:
        n_ids = 1201
        data_path = os.path.expanduser('/mnt/c/users/minch/lab/Wildtrack')
        base0 = Wildtrack(data_path, args.cam_set, args.train_cam, args.test_cam)
    elif 'multiviewx' in args.dataset:
        n_ids = 350
        data_path = os.path.expanduser('/mnt/c/users/minch/lab/MultiviewX')
        base0 = MultiviewX(data_path, args.cam_set, args.train_cam, args.test_cam)
        
    elif 'gmvd_train' in args.dataset or 'gmvd_test' in args.dataset:
        n_ids = 7000
        train_dataset_list = []
        test_dataset_list = []
        
        if 'gmvd_train' == args.dataset: 
            f = open('train_datapath.csv')
        elif 'gmvd_test' == args.dataset:
            f = open('test_datapath.csv')
            #data_path = f.readlines()
        data_path = csv.reader(f)
        #for i in range(len(data_path)):
        for i,data_row in enumerate(data_path):
            #print(data_row[1])
            train_ratio = float(data_row[2])
            sample_require = int(data_row[3])
            path = os.path.expanduser(str(data_row[1]))
            if data_row[1].split('/')[-1]!='Wildtrack':
                base = MultiviewX(path, args.cam_set, args.train_cam, args.test_cam)
            else:
                base = Wildtrack(path, args.cam_set, args.train_cam, args.test_cam)
                
            if data_row[0]=='train':
                # Train data
                dataset_obj = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                train_dataset_list.append(dataset_obj)
                
            else:
                # test data
                print("inside test")
                dataset_obj = GetDataset(base, train=False, transform=train_transform, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                test_dataset_list.append(dataset_obj)
        
        if 'gmvd_test' == args.dataset:
            train_dataset_list = test_dataset_list

    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
        

    # Train and Test set
    if args.dataset == 'gmvd_train' or args.dataset == 'gmvd_test':
        
        print("Training datasets")
        train_dataset = ConcatDataset(*train_dataset_list)
        print("Testing datasets")
        test_dataset = ConcatDataset(*test_dataset_list)
    
    elif args.dataset == 'wildtrack' or args.dataset == 'multiviewx':
        # train_dataset_ = GetDataset(base0, train=True, transform=train_transform, grid_reduce=4, img_reduce=4)
        # test_dataset_ = GetDataset(base0, train=False, transform=train_transform, grid_reduce=4, img_reduce=4) #기존GMVD
        train_dataset_ = GetDataset(base0, train=True, transform=train_transform, grid_reduce=4, img_reduce=8) #EarlyBird decoder
        test_dataset_ = GetDataset(base0, train=False, transform=train_transform, grid_reduce=4, img_reduce=8)
        
        print("Training datasets")
        train_dataset = ConcatDataset(train_dataset_)
        print("Testing datasets")
        test_dataset = ConcatDataset(test_dataset_)
    
    # Train and Test Data Loader
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)
    #value ,index = next(iter(train_dataset))
    #print(value)
    #print(index)


    print(f'Train Data : {len(train_loader)*args.batch_size}')
    print(f'Test Data : {len(test_loader)}')
    
    # Model
    resnet_model = resnet18(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
    model = MultiView_Detection(resnet_model, logdir, args.loss, args.avgpool, args.cam_set, len(args.train_cam), n_ids=n_ids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    '''
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        print(torch.cuda.get_device_properties(device))
        model = nn.DataParallel(model)
    '''
    # print(model)
    
    # Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR Scheduler
    actual_steps = len(train_loader) // accum_steps + (1 if len(train_loader) % accum_steps != 0 else 0)
    if args.lr_sched == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_sched == 'onecycle_lr':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=actual_steps,
                                                        epochs=args.epochs)
    else:
        scheduler = None
    
    # Loss
    criterion = Loss(args.loss, args.k, args.p).to(device)
    occupancy_loss = FocalLoss().to(device)
    classification_loss = torch.nn.CrossEntropyLoss().to(device)
    contrastive_loss = SupConLoss().to(device)

    #test_save_file = 'test_'+str(args.dataset)+'.txt'
    if args.resume is None:
        for epoch in tqdm.tqdm(range(1, args.epochs+1)):
            print('Training...')
            train_loss = train(model, epoch, train_loader, optimizer, args.log_interval, scheduler)
            
            print('Testing...')
            MODA = test(model, epoch, test_loader, os.path.join(logdir, 'test_'+str(args.dataset)+'_'+str(epoch)+'.txt'), visualize=True)
            
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
            
            # Save Dictionary
            if isbest(score[epoch-1][0], epoch) and epoch>1:
                torch.save(model.state_dict(), os.path.join(logdir, 'Best_Multiview_Detection_'+str(args.dataset)+'.pth'))
            
            #if MODA < 50.0 and epoch>5:
            #    break
            torch.save(model.state_dict(), os.path.join(logdir, 'Multiview_Detection_'+str(args.dataset)+'_'+str(epoch)+'.pth'))
            
        moda, modp, prec, rec = score[idx]
        print(f'####### epoch: {idx} ,moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {prec:.1f}%, recall: {rec:.1f}% #######')
        
        #torch.save(model.state_dict(), os.path.join(logdir, 'Multiview_Detection_'+str(args.dataset)+'_'+str(epoch)+'.pth'))
        print("Training Completed..")
            

    else :
        resume_dir = args.resume
        resume_fname = resume_dir #+ '/Multiview_Detection_'+str(args.cross_dataset)+'.pth'
        model.load_state_dict({k.replace('world_classifier','map_classifier'):v for k,v in torch.load(resume_fname).items()})
        model.eval()
        
    print('Test loaded model...')
    test(model, 1, test_loader, os.path.join(logdir, 'test_'+str(args.dataset)+'.txt'), visualize=True)
    moda, modp, precision, recall = score[idx]
    print(f'####### moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}% #######')
    if args.resume is None:
        cleanup(logdir, idx)
