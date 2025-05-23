# import torch
# from mmaction.registry import TRANSFORMS
#
# @TRANSFORMS.register_module()
# class LoadPTFeature:
#     def transform(self, results: dict) -> dict:
#         feat = torch.load(results['video_path'])              # (T, C, H, W)
#         feat = feat.permute(1, 0, 2, 3).unsqueeze(0)          # → (1, C, T, H, W)
#         results['inputs'] = feat
#         results['data_sample'] = dict(gt_label=results['label'])
#         return results

# projects/seasonal_pseudo_change/downstream/pipelines/load_pt_feature.py

import torch
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadPTFeature:

    def __call__(self, results: dict) -> dict:
        feat = torch.load(results['video_path'])              # (T, C, H, W)
        feat = feat.permute(1, 0, 2, 3).unsqueeze(0)           # → (1, C, T, H, W)
        results['inputs'] = feat
        results['data_sample'] = dict(gt_label=results['label'])
        return results
