import numpy as np
import torch
import torch.nn as nn
import math

def img_nparray_2D_2tensor3D(img):
    """
    numpy array shape: H,W
    numpy array dtype: float32
    :param img: input numpy array
    :return: torch tensor C,H,W  float32
    """
    # H W 1
    img=np.expand_dims(img,-1)
    # 1 H W
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def nparray2tensor(data):
    # numpy array to torch tensor
    return torch.from_numpy(np.ascontiguousarray(data)).float()

def data_maxmin_norm(data, norm_range=(-1,1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    :param data: numpy array
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :return:  trans: np.array([data_min,norm_min,scale])
            (norm_data-norm_min)/scale+data_min
    """
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    if minmax_height:
        data_min=minmax_height[0]
        data_max=minmax_height[1]
    else:
        data_max=data.max()
        data_min=data.min()
    norm_data= (norm_max-norm_min) * (data - data_min) /(data_max - data_min + epsilon) + norm_min
    scale=(norm_max-norm_min)/(data_max - data_min + epsilon)
    trans=np.array([data_min,norm_min,scale]).astype(np.float32)
    return norm_data, trans


def data_maxmin_denorm(norm_data,trans):
    """
    :param norm_data: numpy array
    :param trans: np.array([data_min,norm_min,scale])
    :return: normalized numpy array
    """
    data_min=trans[0]
    norm_min=trans[1]
    scale=trans[2]
    data= (norm_data - norm_min) / scale + data_min
    return data

def array_maxmin_tensor(data, norm_range=(-1, 1), epsilon=1e-6):
    """

    :param data: H W dem numpy array
    :param norm_range:
    :param epsilon:
    :return: normlized tensor size (1 H W), trans tensor size (3)
    """
    # H W dem numpy data -> 1 H W tensor

    norm_data, data_trans = data_maxmin_norm(data,norm_range=norm_range, epsilon=epsilon)
    # H W ndarray-> 1 H W tensor
    input_tensor = img_nparray_2D_2tensor3D(norm_data)
    #
    data_trans_tensor = nparray2tensor(data_trans)
    # input tensor: 1 H W
    # data_trans_tensor: 3
    return input_tensor, data_trans_tensor

def tensor_maxmin_norm(data, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    for single DEM
    :param data: tensor： 1 H W or 1 1 H W
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :param minmax_height: given min max of data
    :return:  trans: [data_min,norm_min,scale])
            (norm_data-norm_min)/scale+data_min
    """
    assert data.ndim == 3 or (data.ndim == 4 and data.shape[0]==1)
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    if minmax_height:
        data_min=minmax_height[0]
        data_max=minmax_height[1]
    else:
        data_max=data.max()
        data_min=data.min()
    norm_data= (norm_max-norm_min) * (data - data_min) /(data_max - data_min + epsilon) + norm_min
    scale=(norm_max-norm_min)/(data_max - data_min + epsilon)
    trans=torch.tensor([data_min,norm_min,scale])
    # 1 1 H W
    if len(data.shape)==4:
        # 1,3
        trans=trans.unsqueeze(0)
    return norm_data, trans

def value_denorm(norm_value, trans):

    # value: B,L,1
    # trans: B,3

    # 确保输入张量在同一设备上
    if norm_value.device != trans.device:
        trans = trans.to(norm_value.device)

    # B 3->B 1-> B 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1)
    # B L 1
    new_data= (norm_value - norm_min) / scale + data_min
    return new_data

def tensor4D_maxmin_norm(data, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    based on each sample's max and min or
    :param data: tensor：B 1 H W
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :return:
        norm_data, B 1 H W
        trans: B 3
            [data_min,norm_min,scale])
            ### (norm_data-norm_min)/scale+data_min
    """
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    batchsize=data.shape[0]
    if minmax_height:
        data_min = torch.full([batchsize], minmax_height[0], dtype=data.dtype)
        data_max = torch.full([batchsize], minmax_height[1], dtype=data.dtype)
    else:
        # B
        data_max = data.amax(dim=(1, 2, 3))
        # B
        data_min = data.amin(dim=(1, 2, 3))

    # 将标量扩展为与数据相同形状进行广播计算
    data_min_expanded = data_min.view(batchsize, 1, 1, 1)
    data_max_expanded = data_max.view(batchsize, 1, 1, 1)

    # B 1 H W
    norm_data = (norm_max - norm_min) * (data - data_min_expanded) / (
            data_max_expanded - data_min_expanded + epsilon) + norm_min

    # B 3
    trans = torch.stack([data_min,
                         torch.full_like(data_min, norm_min),
                         (norm_max - norm_min) / (data_max - data_min + epsilon)], dim=1)

    return norm_data, trans


def tensor_maxmin_trans(data,trans):
    """
    for single DEM
    :param data: tensor： 1 H W
    :param trans: tensor： 3
    :return: tensor： 1 H W
    """
    data_min=trans[0]
    norm_min=trans[1]
    scale=trans[2]
    norm_data=(data-data_min)*scale+norm_min
    return norm_data

def tensor4D_maxmin_denorm(norm_data, trans):
    """
    dem tensor data maxmin denormalization

    :param norm_data: B 1 H W
    :param trans: B 3
    :return: B 1 H W tensor
    """
    # 确保输入张量在同一设备上
    if norm_data.device != trans.device:
        trans = trans.to(norm_data.device)

    # B 3->B 1-> B 1 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1, 1)
    # B 1 H W
    new_data= (norm_data - norm_min) / scale + data_min
    return new_data


class Slope_torch(nn.Module):
    def __init__(self,pixel_size=1):
        #pixel_size在实际计算坡度时使用,为空间分辨率
        super(Slope_torch, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)
        self.pixel_size = pixel_size
        # -1 0 1
        # -2 0 2
        # -1 0 -1
        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1
        # -1 -2 -1
        # 0 0 0
        # 1 2 1
        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * self.pixel_size)
        weight2 = weight2 / (8 * self.pixel_size)
        # nn.Parameter 注册为模型参数
        self.weight1 =nn.Parameter(torch.tensor(weight1)) # 自定义的权值
        self.weight2 =nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False
    def forward(self, x,return_dxdy=False):
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        # 坡度值
        slope = torch.arctan(slope) * 180 / math.pi

        if return_dxdy:
            return dx, dy, slope
        else:
            return slope

    def forward_dxdy(self, x):
        """
        计算dx和dy
        :param x: 输入张量
        :return: dx, dy, grad_norm
        """
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        grad_norm= torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))

        return dx, dy,grad_norm

#坡向
class Aspect_torch(nn.Module):
    def __init__(self):
        super(Aspect_torch, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))

        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias =nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        # west point to east
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        # north point to south
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        # torch.atan2(-dy, dx) 返回向量(dx,-dy)与x轴正方向的弧度，范围在-pi到pi，逆时针为正
        aspect = 180/math.pi*torch.atan2(-dy, dx)

        # angle from north
        aspect = torch.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect

    def forward_rad(self, x):
        # west point to east
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        # north point to south
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        # torch.atan2(-dy, dx) 返回向量(dx,-dy)与x轴正方向的弧度，范围在-pi到pi，逆时针为正
        aspect_rad = torch.atan2(-dy, dx)

        return aspect_rad


Slope_net = Slope_torch(pixel_size=30)
Aspect_net = Aspect_torch()

def cal_DEM_metric(demA, demB, padding=None, device=None, reduction="mean",slope_net=Slope_net, aspect_net=Aspect_net):
    """
    input ndarray or tensor
    :param demA demB: B 1 H W
    :param padding: to be cropped
    :reduction: batch mean or none
    :return: {
        'height_mae':
        'slope_mae':
        'aspect_mae':
        'height_rmse':
        'slope_rmse':
        'aspect_rmse':
        }
    """
    B=demA.shape[0]
    if padding:
        demA = demA[...,padding:-padding, padding:-padding]
        demB = demB[...,padding:-padding, padding:-padding]
    if isinstance(demA,np.ndarray):
        demA_tensor = torch.from_numpy(demA)
        demB_tensor = torch.from_numpy(demB)
    elif isinstance(demA, torch.Tensor):
        demA_tensor=demA
        demB_tensor=demB

    slope_net=slope_net
    aspect_net=aspect_net
    if device:
        slope_net.to(device)
        aspect_net.to(device)
        demA_tensor=demA_tensor.to(device)
        demB_tensor=demB_tensor.to(device)
    else:
        if demA_tensor.is_cuda:
            device=demA_tensor.device
            slope_net.to(device)
            aspect_net.to(device)
    with torch.inference_mode():
        demA_slope = slope_net(demA_tensor)
        demB_slope = slope_net(demB_tensor)
        demA_aspect = aspect_net(demA_tensor)
        demB_aspect = aspect_net(demB_tensor)

    height_mae=torch.abs(demA_tensor - demB_tensor).mean(dim=(1,2,3))
    height_rmse=torch.sqrt(torch.mean(torch.pow(demA_tensor - demB_tensor, 2), dim=(1,2,3)))
    height_max_error,_=torch.abs(demA_tensor - demB_tensor).view(B,-1).max(dim=1)

    slope_mae=torch.abs(demA_slope - demB_slope).mean(dim=(1,2,3))
    slope_rmse=torch.sqrt(torch.mean(torch.pow(demA_slope - demB_slope, 2), dim=(1,2,3)))
    slope_max_error,_=torch.abs(demA_slope - demB_slope).view(B,-1).max(dim=1)

    aspect_mae=torch.abs(demA_aspect - demB_aspect).mean(dim=(1,2,3))
    aspect_rmse=torch.sqrt(torch.mean(torch.pow(demA_aspect - demB_aspect, 2), dim=(1,2,3)))
    aspect_max_error,_=torch.abs(demA_aspect - demB_aspect).view(B,-1).max(dim=1)
    #B,1 -> 1
    if reduction=="mean":
        height_mae=height_mae.mean()
        height_rmse=height_rmse.mean()
        slope_mae=slope_mae.mean()
        aspect_mae=aspect_mae.mean()
        slope_rmse=slope_rmse.mean()
        aspect_rmse=aspect_rmse.mean()

    return {
        'height_mae': height_mae.cpu().numpy(),
        'height_rmse': height_rmse.cpu().numpy(),
        #'height_max_error': height_max_error.cpu().numpy(),
        'slope_mae':slope_mae.cpu().numpy(),
        'slope_rmse': slope_rmse.cpu().numpy(),
        #'slope_max_error': slope_max_error.cpu().numpy(),
        'aspect_mae':aspect_mae.cpu().numpy(),
        'aspect_rmse':aspect_rmse.cpu().numpy(),
    }


