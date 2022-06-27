from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from rsimhe.core import pre_eval_to_metrics, metrics, eval_metrics
from rsimhe.utils import get_root_logger
from rsimhe.datasets.builder import DATASETS
from rsimhe.datasets.pipelines import Compose

from rsimhe.ops import resize

from PIL import Image

import torch
import os


@DATASETS.register_module()
class CustomDepthDataset(Dataset):
    """Custom dataset for supervised monocular rsimhe esitmation. 
    An example of file structure. is as followed.
    .. code-block:: none
        ├── data
        │   ├── custom
        │   │   ├── train
        │   │   │   ├── rgb
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   │   ├── rsimhe
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   ├── val
        │   │   │   ...
        │   │   │   ...

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        data_root (str, optional): Data root for img_dir.
        test_mode (bool): test_mode=True
        min_rsimhe=1e-3: Default min rsimhe value.
        max_rsimhe=10: Default max rsimhe value.
    """

    def __init__(self,
                 pipeline,
                 data_root,
                 test_mode=True,
                 min_rsimhe=1e-3,
                 max_rsimhe=10,
                 rsimhe_scale=1):

        self.pipeline = Compose(pipeline)
        self.img_path = os.path.join(data_root, 'rgb')
        self.rsimhe_path = os.path.join(data_root, 'rsimhe')
        self.test_mode = test_mode
        self.min_rsimhe = min_rsimhe
        self.max_rsimhe = max_rsimhe
        self.rsimhe_scale = rsimhe_scale

        # load annotations
        self.img_infos = self.load_annotations(self.img_path, self.rsimhe_path)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, rsimhe_dir):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory. Load all the images under the root.
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        imgs = os.listdir(img_dir)
        imgs.sort()

        if self.test_mode is not True:
            rsimhes = os.listdir(rsimhe_dir)
            rsimhes.sort()

            for img, rsimhe in zip(imgs, rsimhes):
                img_info = dict()
                img_info['filename'] = img
                img_info['ann'] = dict(rsimhe_map=rsimhe)
                img_infos.append(img_info)
        
        else:

            for img in imgs:
                img_info = dict()
                img_info['filename'] = img
                img_infos.append(img_info)

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())

        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['rsimhe_fields'] = []
        results['img_prefix'] = self.img_path
        results['rsimhe_prefix'] = self.rsimhe_path
        results['rsimhe_scale'] = self.rsimhe_scale

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']
    
    # waiting to be done
    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.rsimhe_scale) # Do not convert to np.uint16 for ensembling. # .astype(np.uint16)
        return results

    # design your own evaluation pipeline
    def pre_eval(self, preds, indices):
        pass

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        pass