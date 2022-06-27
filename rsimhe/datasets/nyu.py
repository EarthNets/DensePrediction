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

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

@DATASETS.register_module()
class NYUDataset(Dataset):
    """NYU dataset for rsimhe estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── NYU
        │   │   ├── nyu_train.txt
        │   │   ├── nuy_test.txt
        │   │   ├── scenes_xxxx (xxxx. No. of the scenes)
        │   │   │   ├── data_1
        │   │   │   ├── data_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── scenes (test set, no scene No.)
        │   │   │   ├── data_1 ...
    split file format:
    input_image: /kitchen_0028b/rgb_00045.jpg
    gt_rsimhe:    /kitchen_0028b/sync_rsimhe_00045.png
    focal:       518.8579
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.png'
        ann_dir (str, optional): Path to annotation directory. Default: None
        rsimhe_map_suffix (str): Suffix of rsimhe maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
    """
 
    def __init__(self,
                 pipeline,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 rsimhe_scale=1000,
                 garg_crop=False,
                 eigen_crop=True,
                 min_rsimhe=1e-3,
                 max_rsimhe=10):

        self.pipeline = Compose(pipeline)
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.rsimhe_scale = rsimhe_scale
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.min_rsimhe = min_rsimhe
        self.max_rsimhe = max_rsimhe
        
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.data_root, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, data_root, split):
        """Load annotation from directory.
        Args:
            data_root (str): Data root for img_dir/ann_dir.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        self.invalid_rsimhe_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    rsimhe_map = line.strip().split(" ")[1]
                    if rsimhe_map == 'None':
                        self.invalid_rsimhe_num += 1
                        continue
                    img_info['ann'] = dict(rsimhe_map=osp.join(data_root, remove_leading_slash(rsimhe_map)))
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = osp.join(data_root, remove_leading_slash(img_name))
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_rsimhe_num} invalid pairs are filtered', logger=get_root_logger())
        return img_infos
    
    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['rsimhe_fields'] = []
        results['rsimhe_scale'] = self.rsimhe_scale

        # train/test share the same cam param
        results['cam_intrinsic'] = \
            [[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
             [5.1946961112127485e+02, 0, 2.5373616633400465e+02],
             [0                     , 0, 1                    ]]

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

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_rsimhe_maps(self):
        """Get ground truth rsimhe maps for evaluation."""

        for img_info in self.img_infos:
            rsimhe_map = img_info['ann']['rsimhe_map']
            rsimhe_map_gt = np.asarray(Image.open(rsimhe_map), dtype=np.float32) / self.rsimhe_scale
            yield rsimhe_map_gt

    def eval_mask(self, rsimhe_gt):
        rsimhe_gt = np.squeeze(rsimhe_gt)
        valid_mask = np.logical_and(rsimhe_gt > self.min_rsimhe, rsimhe_gt < self.max_rsimhe)
        if self.garg_crop or self.eigen_crop:
            gt_height, gt_width = rsimhe_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif self.eigen_crop:
                # eval_mask = np.ones(valid_mask.shape)
                eval_mask[45:471, 41:601] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the rsimhe estimation, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            rsimhe_map = self.img_infos[index]['ann']['rsimhe_map']

            rsimhe_map_gt = np.asarray(Image.open(rsimhe_map), dtype=np.float32) / self.rsimhe_scale
            rsimhe_map_gt = np.expand_dims(rsimhe_map_gt, axis=0)
            # rsimhe_map_gt = self.eval_nyu_crop(rsimhe_map_gt)
            valid_mask = self.eval_mask(rsimhe_map_gt)
            
            eval = metrics(rsimhe_map_gt[valid_mask], pred[valid_mask], self.min_rsimhe, self.max_rsimhe)
            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict rsimhe map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_rsimhe_maps = self.get_gt_rsimhe_maps()
            ret_metrics = eval_metrics(
                gt_rsimhe_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i*9: i*9 + 9]
            values = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
