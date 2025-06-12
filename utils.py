import numpy as np

import imageio.v3 as imageio
import torch

from omegaconf import OmegaConf,DictConfig

torch.set_printoptions(precision=6)

def read_dem(dem_file: str):
    """

    :param dem_file:
    :return: H W
    """
    file_suffix = dem_file.split('.')[-1]
    if file_suffix.lower() in ['tif', 'tiff']:
        # 和pillow库速度差不多
        # dem_data = np.array(Image.open(dem_file))
        dem_data = imageio.imread(dem_file)
    elif file_suffix == 'dem':
        # 后缀是dem的文件，其实就是ASCII文件，用读取txt的方式读取
        dem_data = np.loadtxt(dem_file, dtype=np.float32, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_suffix}")
    return dem_data.astype(np.float32)

def write_dem(dem_data, dem_file):
    imageio.imwrite(dem_file, dem_data)

def compose_kwargs(**kwargs):
    """
    compose keyword arguments to a string
    :param kwargs:
    :return:
    """
    str_res=""
    for key, value in kwargs.items():
            str_res=str_res+(f"{key}:{str(value)}")+" "*4
    return str_res


def gradient(y, x, grad_outputs=None):
    """
    Calculate the gradient of y with respect to x.
    :param y:
    :param x:
    :param grad_outputs:
    :return: shape same as x, gradient of y with respect to x
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def save_config_to_yaml(config_dict: dict, file_path: str) -> None:
    """
    将字典类型的配置信息保存为 YAML 文件。

    参数:
        config_dict (dict): 要保存的配置字典。
        file_path (str): YAML 文件的完整保存路径。
    """
    # 将字典转换为 OmegaConf 对象
    cfg = OmegaConf.create(config_dict)
    # 保存到 YAML
    OmegaConf.save(config=cfg, f=file_path)

def get_config_from_yaml(yaml_file,return_dict=True):
    with open(yaml_file, 'r') as f:
        config = OmegaConf.load(f)
    return config if not return_dict else OmegaConf.to_container(config,resolve=True, throw_on_missing=True)

# 根据输入的形状和范围（默认-1,1）
# shape=(H,W)
# 生成网格坐标 (纵坐标，横坐标) 对应于H，W
def get_pixel_center_coord_tensor(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        ranges:2*2大小的 range[0]=() range[1]=() 网格边界的值
        生成网格中心点对应的坐标
    """
    coord_seqs = []
    # shape = [H, W]
    # i=0,n=H
    # i=1,n=W

    shape= (shape,shape) if isinstance(shape, int) else shape

    # shape H,W
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]

        # 0.5 pixel width
        r = (v1 - v0) / (2 * n)
        # seq范围[vo+r,v1-r]，n个数，间隔2r
        # torch.arange(n) 0~n-1
        # (v0+r,v0+3r,v0+5r,...,v1-r)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        # 存储H 和 W的坐标
        coord_seqs.append(seq)

    # *coord_seqs 是H W
    # 输入是先纵轴坐标后横轴坐标
    # 指定index为ij形式，则生成的两个网格坐标数组大小是H W，对应列索引和行索引,取值就是[H][W]。
    # stack后，大小变为H W 2  2中首先是对应的H坐标然后是W坐标
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    # 数组展平(H*W,2)默认为True
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

if __name__=="__main__":

    pass