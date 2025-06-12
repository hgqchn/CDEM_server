import sys
import os

import torch

from model import modules
import dem_utils
import utils
import numpy as np



def generate_sr_dem(lr_dem,scale,model):
    """

    :param lr_dem: shape H,W dem ndarray
    :param scale:
    :param model:
    :return:
    """
    lr_4dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    lr_height,lr_width = lr_dem.shape
    hr_height=lr_height*scale
    hr_width=lr_width*scale
    device=next(model.parameters()).device
    hr_coord = utils.get_pixel_center_coord_tensor((hr_height, hr_width))
    hr_coord = hr_coord.to(device)
    hr_coord = hr_coord.unsqueeze(0) # Add batch dimension
    # 模型运算
    out=model(hr_coord)
    coord_grad=out['model_in']
    model_out=out['model_out']

    _, trans = dem_utils.tensor_maxmin_norm(lr_4dtensor, (-1, 1), 1e-6,
                                                       None)
    sr_value = dem_utils.value_denorm(model_out, trans)

    # sr
    sr_4dtensor = sr_value.view(1, 1, hr_height, hr_width).detach().cpu()
    #
    sr_dem_np=sr_4dtensor.squeeze().squeeze().numpy()
    return sr_dem_np

def get_hegiht_dxdy(coords,model,dem_width,dem_height,trans,dem_res=30):
    """

    :param coords:  shape: (B,2)  (y,x) coordinate of pixel in image, e.g. x from 1 to dem_width
    :param model:
    :param dem_width: width of hr dem
    :param dem_height: height of hr dem
    :param trans: norm parameters from lr dem
    :param dem_res: resolution of the hr dem, default 30m
    :return:
    """

    if isinstance(coords, np.ndarray):
        coords=torch.from_numpy(coords)
    coords = coords.float()
    # 得到像素中心的归一化的坐标
    coords[:, 0] = -1+(2*coords[:, 0]-1)/dem_height
    coords[:, 1] = -1+(2*coords[:, 1]-1)/dem_width



    device=next(model.parameters()).device
    coords=coords.to(device)
    coords = coords.unsqueeze(0)  # Add batch dimension

    out=model(coords)
    coord_grad=out['model_in']
    model_out=out['model_out']
    sr_value = dem_utils.value_denorm(model_out, trans)
    dydx = utils.gradient(sr_value, coord_grad, grad_outputs=torch.ones_like(model_out))

    dy=dydx[..., 0]  # dy
    dx=dydx[..., 1]
    dy=dy*2/dem_height*1/dem_res  # 2/64 坐标缩放
    dx=dx*2/dem_width*1/dem_res

    return sr_value,dx,dy

if __name__ == '__main__':

    lr_file=r'test_data/lr.TIF'
    lr_dem = utils.read_dem(lr_file)

    sr_file=r'test_data/sr.TIF'
    sr_dem = utils.read_dem(sr_file)
    sr_4dtensor = torch.from_numpy(sr_dem).unsqueeze(0).unsqueeze(0)

    weight_path=r"test_data/dem0_0_mlp_params.pth"
    model_config=utils.get_config_from_yaml(r'config.yaml')['model_config']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mlp_hidden_features = model_config['target_hidden']
    mlp_hidden_layers = model_config['target_hidden_layers']
    lr_resolution = model_config['image_resolution']
    use_pe=model_config['use_pe']

    model=modules.SimpleMLPNet(hidden_features=mlp_hidden_features,
                               num_hidden_layers=mlp_hidden_layers,
                               image_resolution=lr_resolution,
                               use_pe=use_pe, )

    sd=torch.load(weight_path,weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 用法1
    # 生成超分辨率dem
    # sr_dem_np=generate_sr_dem(lr_dem=lr_dem,scale=4,model=model)
    # print(np.allclose(sr_dem_np,sr_dem))

    # 用法2
    # 直接获取输入坐标的高程值，dx，dy偏导数
    lr_4dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    _, trans = dem_utils.tensor_maxmin_norm(lr_4dtensor, (-1, 1), 1e-6,
                                                       None)
    # 边缘像素的值不准需要去掉
    # 也就是y，x索引从2开始
    coords=np.array([[2, 2], [2, 3],[3,2],[3,3]])  # (y,x) coordinate of pixel in image

    # hr dem的宽高
    dem_height,dem_width=64,64
    height,dx,dy=get_hegiht_dxdy(coords,model,dem_width,dem_height,trans)

    pass
