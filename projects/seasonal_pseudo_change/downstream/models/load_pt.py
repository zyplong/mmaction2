import os
import torch
from mmaction.datasets.base import BaseDataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class PTFeatureDataset(BaseDataset):
    def __init__(self, ann_file, data_prefix, pipeline, test_mode=False):
        self.data_prefix = data_prefix  # 先保存原始 data_prefix
        super().__init__(ann_file=ann_file, data_prefix=data_prefix, pipeline=pipeline, test_mode=test_mode)

        if isinstance(self.data_prefix, dict):
            if 'video_path' in self.data_prefix:
                self.data_prefix = self.data_prefix['video_path']
            else:
                raise ValueError('data_prefix must contain "video_path" key if passed as dict.')

        self.data_list = self.load_data_list()

    def _join_prefix(self):  # ✅ 加上这个方法覆盖掉父类行为
        pass

    def _load_metainfo(self, metainfo=None):
        return {}

    def load_annotations(self):
        split = []
        with open(self.ann_file, 'r') as f:
            for line in f.readlines()[1:]:
                name, label = line.strip().split(',')
                sample = dict(
                    video_path=os.path.join(self.data_prefix, f'{name}.pt'),
                    label=int(label)
                )
                split.append(sample)
        #self.data_address = [item['video_path'] for item in split]
        return split

    def load_data_list(self):
        return self.load_annotations()

    def __len__(self):
        return len(self.data_list)

    def prepare_train_frames(self, idx):
        results = self.data_list[idx].copy()
        feat = torch.load(results['video_path'])  # [20, 2048, 4, 4]
        feat = feat.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2048, 20, 4, 4]
        results['inputs'] = feat
        results['data_sample'] = dict(gt_label=results['label'])
        return results

    def prepare_test_frames(self, idx):
        return self.prepare_train_frames(idx)
