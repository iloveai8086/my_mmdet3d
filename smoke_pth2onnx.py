import torch

from mmdet3d.apis import init_model
from mmdet3d.models.dense_heads.smoke_mono3d_head import SMOKEMono3DHead


# import torch.onnx.symbolic_opset9 as sym
# def groupnorm(g, input, num_groups, weight, bias, eps):
#     return sym.batch_norm(g, input, weight, bias, eps, True, num_groups)
# from torch.onnx import register_custom_op_symbolic
# register_custom_op_symbolic('mydomain::GroupNorm', groupnorm, 9)




class SMOKEOnnx(torch.nn.Module):
    def __init__(self, model):
        super(SMOKEOnnx, self).__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head

    def forward(self, img, topk=100):
        out = self.backbone(img)
        out = self.neck(out)
        cls_scores, bbox_preds = self.head(out)  # （1，3，96，320）（1，8，96，320）3就是numclass，因为是kitti所以是3

        # get_local_maximum
        # https://gitee.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L190
        hmax = torch.nn.functional.max_pool2d(cls_scores[0], 3, stride=1, padding=1)  # 这个3应该是池化核的大小，（1，3，96，320）
        keep = (hmax == cls_scores[0]).float()  # （1，3，96，320） 取完最大值，最大值和原始对比得到每一个位置是不是有物体，此时是只有0-1
        scores = cls_scores[0] * keep                               # (1, 3, H/4, W/4)

        # get_topk_from_heatmap
        # https://gitee.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L207
        # batch, _, height, width = int(scores.size()[0]),int(scores.size()[1]),int(scores.size()[2]),int(scores.size()[3])
        batch, _, height, width = scores.size()  # 1 3 96 320
        scores = scores.view(batch, -1)  # （1，92160）96*320是特征图，3乘上去，这是直接把类别也干进去了，每个位置三个类别？
        topk_scores, topk_indices = torch.topk(scores, topk)        # (1, 100), (1, 100)   默认最后一个维度
        # topk_clses = topk_inds // (height * width)                # (1, 100)
        topk_clses = torch.floor(topk_indices / (height * width))   # (1, 100)  topk_indices都是几万几万的，看看在特征图的什么位置，除以30720，取整，这个变量只有012，代表类别？
        topk_inds = topk_indices % (height * width)                 # （1，100）第几个格子那种索引，最多到30720
        # topk_ys = topk_inds // width                              # (1, 100)
        topk_ys = torch.floor(topk_inds / width)                    # (1, 100)  5000/320，判断在第几行
        topk_xs = (topk_inds % width).int().float()                 # (1, 100)  5000%320 判断在当前行的第几列，就有了坐标，为什么trt他不用points？
        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()], dim=1)    # (100, 2)

        # transpose_and_gather_feat
        # https://gitee.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L255
        bbox_pred = bbox_preds[0].permute(0, 2, 3, 1).contiguous()  # (1, H/4, W/4, 8)  （1，96，320，8）
        bbox_pred = bbox_pred.view(-1, 8)                           # (H*W/16, 8)  （30720，8）
        topk_inds = topk_inds.view(-1)                              # (100)
        # bbox_pred = bbox_pred[topk_inds, :]                       # (100, 8)   好像是inplace操作
        bbox_p = bbox_pred[topk_inds, :]                            # (100, 8)   好像是inplace操作，不是inplace 30720里面找索引取数据
        topk_clses = topk_clses.view(-1)                            # (100)
        topk_scores = topk_scores.view(-1)                          # (100)

        # return bbox_pred, points, topk_clses.float(), topk_scores
        return bbox_p, topk_scores, topk_indices.float()


def export_onnx(onnx_file_path):
    # https://gitee.com/open-mmlab/mmdetection3d/tree/master/configs/smoke
    config_file = 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py'
    checkpoint_file = 'checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth'
    checkpoint_model = init_model(config_file, checkpoint_file)

    smoke = SMOKEOnnx(checkpoint_model)
    dummy_img = torch.randn(1, 3, 384, 1280, device='cuda:0')
    out = smoke(dummy_img)
    torch.onnx.export(smoke, dummy_img, onnx_file_path, opset_version=13)
    print('Saved SMOKE onnx file: {}'.format(onnx_file_path))


if __name__ == '__main__':
    export_onnx('smoke_dla34.onnx')
    
