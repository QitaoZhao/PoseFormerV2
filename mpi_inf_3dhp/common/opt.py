import argparse
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-frame-kept', '--number-of-kept-frames', default='27', type=int, metavar='N',
                        help='how many frames are kept')
        self.parser.add_argument('-coeff-kept', '--number-of-kept-coeffs', type=int, metavar='N', help='how many coefficients are kept')
        self.parser.add_argument('--naive', type=bool, default=False)
        
        self.parser.add_argument('--layers', default=4, type=int)
        self.parser.add_argument('--channel', default=256, type=int)
        self.parser.add_argument('--d_hid', default=512, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', type=int, default=0)
        self.parser.add_argument('--test', type=int, default=1)
        self.parser.add_argument('--nepoch', type=int, default=80)
        self.parser.add_argument('-b','--batchSize', type=int, default=160)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=80)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('-f','--frames', type=int, default=243)
        self.parser.add_argument('--pad', type=int, default=121)
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--reload', type=int, default=0)
        self.parser.add_argument('--refine_reload', type=int, default=0)
        self.parser.add_argument('-c','--checkpoint', type=str, default='model')
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--previous_refine_name', type=str, default='')
        self.parser.add_argument('--manualSeed', type=int, default=1)

        self.parser.add_argument('--MAE', action='store_true')
        self.parser.add_argument('-tmr','--temporal_mask_rate', type=float, default=0)
        self.parser.add_argument('-smn', '--spatial_mask_num', type=int, default=0)
        self.parser.add_argument('-tds', '--t_downsample', type=int, default=1)

        self.parser.add_argument('--MAE_reload', type=int, default=0)
        self.parser.add_argument('-r', '--resume', action='store_true')




    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        self.opt.pad = (self.opt.frames-1) // 2

        stride_num = {
                '9': [1, 3, 3],
                '27':  [3, 3, 3],
                '351': [3, 9, 13],
                '81': [3, 3, 3, 3],
                '243': [3, 3, 3, 3, 3],
            }

        if str(self.opt.frames) in stride_num:
            self.opt.stride_num = stride_num[str(self.opt.frames)]
        else:
            self.opt.stride_num = None
            print('no stride_num')
            # exit()

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'
        #self.opt.subjects_test = 'S11'

        #if self.opt.train:
        logtime = time.strftime('%m%d_%H%M_%S_')

        ckp_suffix = ''
        if self.opt.refine:
            ckp_suffix='_refine'
        elif self.opt.MAE:
            ckp_suffix = '_pretrain'
        else:
            ckp_suffix = '_STMO'
        self.opt.checkpoint = 'checkpoint/'+self.opt.checkpoint + '_%d'%(self.opt.pad*2+1) + \
            '%s'%ckp_suffix

        if not os.path.exists(self.opt.checkpoint):
            os.makedirs(self.opt.checkpoint)

        if self.opt.train:
            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt





        
