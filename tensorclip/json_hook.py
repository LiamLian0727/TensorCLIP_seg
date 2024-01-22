# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict, defaultdict

import json
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class JsonHock(Hook):
    def __init__(self,
                 by_epoch=False,
                 interval=1,
                 log_save_dir='loss.json'):

        self.by_epoch = by_epoch
        self.interval = interval
        self.log_save_dir = log_save_dir
        self.json = None

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def before_run(self, runner):
        self.json = defaultdict(list)

    def after_train_iter(self, runner):
        # cur_iter = self.get_iter(runner, inner_iter=True)
        # print(cur_iter, runner.outputs['log_vars'])
        for key, value in runner.outputs['log_vars'].items():
            if 'loss' in key:
                self.json[key].append(f'{value:.3f}')

    def after_run(self, runner):
        if self.json is not None:
            with open(self.log_save_dir, 'w') as json_file:
                json_file.write(json.dumps(self.json))
