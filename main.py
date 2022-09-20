from email.policy import strict
import nni

import logging
import pickle
import json
import torch
import os.path as osp
from parser import create_parser

import warnings
warnings.filterwarnings('ignore')

from methods import SimDesign
from API import Recorder
from utils import *


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = SimDesign(self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)

    def test(self):
        test_perplexity, test_recovery, test_subcat_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))

        for cat in test_subcat_recovery.keys():
            print_log('Category {0} Rec: {1:.4f}\n'.format(cat, test_subcat_recovery[cat]))

        return test_perplexity, test_recovery


if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__

    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    print(config)
    
    svpath = '/gaozhangyang/experiments/ProDesign/results/ProDesign/'
    exp = Exp(args)

    exp.method.model.load_state_dict(torch.load(svpath+'checkpoint.pth'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_perp, test_rec = exp.test()

    nni.report_final_result(test_rec.item())