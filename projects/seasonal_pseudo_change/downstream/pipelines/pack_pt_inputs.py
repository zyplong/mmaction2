# # from mmaction.registry import TRANSFORMS
# # from mmcv.transforms import BaseTransform
# #
# # @TRANSFORMS.register_module()
# # class PackPTInputs(BaseTransform):
# #     def transform(self, results):
# #         return dict(
# #             inputs=results['inputs'],  # [1, 2048, 20, 4, 4]
# #             data_sample=results['data_sample']  # dict(gt_label=...)
# #         )
# from mmaction.registry import TRANSFORMS
# from mmcv.transforms import BaseTransform
# from mmaction.structures import ActionDataSample
#
#
# @TRANSFORMS.register_module()
# class PackPTInputs(BaseTransform):
#     def transform(self, results):
#         # 输入张量 [1, 2048, 20, 4, 4]
#         inputs = results['inputs']
#
#         # 将 dict(gt_label=...) → ActionDataSample(gt_label=...)
#         data_sample = ActionDataSample()
#         if 'gt_label' in results:
#             data_sample.gt_label = results['gt_label']
#
#         return dict(
#             inputs=inputs,
#             data_samples=[data_sample]  # 注意是 list
#         )

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmaction.structures import ActionDataSample

@TRANSFORMS.register_module()
class PackPTInputs(BaseTransform):
    def transform(self, results):
        data_sample = ActionDataSample()
        if 'gt_label' in results:
            data_sample.gt_label = results['gt_label']

        return dict(
            inputs=results['inputs'],  # [1, 2048, 20, 4, 4]
            data_samples=[data_sample]  # 必须是 list
        )
