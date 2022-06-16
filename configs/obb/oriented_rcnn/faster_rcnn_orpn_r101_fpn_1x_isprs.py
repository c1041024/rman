_base_ = './faster_rcnn_orpn_r50_fpn_1x_isprs.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
