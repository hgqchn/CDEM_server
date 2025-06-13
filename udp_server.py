import sys
import os
import socket
import json

import torch
from model import modules
import dem_utils
import utils
import numpy as np
import math


# 配置服务器IP和端口
SERVER_IP = '0.0.0.0'      # 监听所有网卡
SERVER_PORT = 9999

#DEM_INR 配置
lr_file_path=r'test_data/lr.TIF'
weight_file_path=r"test_data/dem0_0_mlp_params.pth"
model_config_path=r'config.yaml'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DEM_INR:
    def __init__(self,
                 lr_file=lr_file_path,
                 weight_file=weight_file_path,
                 model_config_file=model_config_path,
                 scale=4,
                 hr_res=30,  # 超分辨率DEM的空间分辨率
                 device=device):
        lr_dem = utils.read_dem(lr_file)
        lr_height,lr_width = lr_dem.shape
        hr_height,hr_width = lr_height*scale, lr_width*scale
        self.scale=scale
        self.hr_height=hr_height
        self.hr_width=hr_width
        self.device = device
        self.hr_res=hr_res
        model_config = utils.get_config_from_yaml(model_config_file)['model_config']

        mlp_hidden_features = model_config['target_hidden']
        mlp_hidden_layers = model_config['target_hidden_layers']
        lr_resolution = model_config['image_resolution']
        use_pe = model_config['use_pe']
        model = modules.SimpleMLPNet(hidden_features=mlp_hidden_features,
                                     num_hidden_layers=mlp_hidden_layers,
                                     image_resolution=lr_resolution,
                                     use_pe=use_pe, )

        sd = torch.load(weight_file, weights_only=True)
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        self.model = model

        lr_4dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
        _, trans = dem_utils.tensor_maxmin_norm(lr_4dtensor, (-1, 1), 1e-6,
                                                None)
        self.trans=trans

    def coords_norm(self,coords):
        """

        :param coords: shape: (B,2)  (y,x) coordinate of pixel in image, e.g. x from 1 to dem_width, y from 1 to dem_height. left up pixel is (1,1)
        :return:
        """
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        if isinstance(coords, list):
            coords=torch.tensor(coords)
        coords = coords.float()
        # 得到像素中心的归一化的坐标
        # y坐标
        coords[:, 0] = -1 + (2 * coords[:, 0] - 1) / self.hr_height
        # x坐标
        coords[:, 1] = -1 + (2 * coords[:, 1] - 1) / self.hr_width
        coords = coords.unsqueeze(0)
        coords = coords.to(self.device)
        return coords


    def get_height(self,coords):
        """

        :param coords: shape: (B,2),  (y,x) coordinate of pixel in image, e.g. x from 1 to dem_width, y from 1 to dem_height. left up pixel is (1,1)
        :return: shape: (B,1), height value of pixel at (y,x) coordinate
        """
        coords=self.coords_norm(coords)
        out = self.model(coords)
        model_out = out['model_out']
        height_value = dem_utils.value_denorm(model_out, self.trans)
        height_value=height_value.detach().cpu().numpy()
        return height_value

    def get_height_dxdy(self, coords,angle):
        """

        :param coords: 形状为B,2的坐标数组
                        每个元素为(y,x)坐标，y从1到dem_height，x从1到dem_width
                        左上角像素中心坐标为(1,1)，y方向为从上到下（北到南），x方向为从左到右（西到东）
        :param angle: 前进方向与正北方向的夹角，单位为弧度
        :return:
            height_value: 形状为(B,1)的高程值数组
            dx: 形状为(B,1)的x方向偏导数，即从西向东
            dy: 形状为(B,1)的y方向偏导数，即从北向南
            slope: 形状为(B,1)的前进方向的坡度值，
        """

        coords=self.coords_norm(coords)
        out = self.model(coords)
        coord_grad = out['model_in']
        model_out = out['model_out']
        height_value = dem_utils.value_denorm(model_out, self.trans)
        dydx = utils.gradient(height_value, coord_grad, grad_outputs=torch.ones_like(model_out))
        dy = dydx[..., 0]  # dy
        dx = dydx[..., 1]
        dy = dy * 2 / self.hr_height * 1 / self.hr_res  # 2/64 坐标缩放
        dx = dx * 2 / self.hr_width * 1 / self.hr_res
        height_value=height_value.detach().cpu().numpy()
        dx= dx.detach().cpu().numpy()
        dy= dy.detach().cpu().numpy()
        slope=dx*math.sin(angle)+dy*math.cos(angle)

        return height_value, dx, dy, slope



class UDPServer:
    def __init__(self, dem_inr,host='0.0.0.0', port=8888, buffer_size=1024):
        # buffer_size: 接收数据的缓冲区大小。 一组x,y坐标大小为8字节
        self.dem_inr = dem_inr

        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        print(f"[UDPServer] Listening on {self.host}:{self.port}")

    def start(self):
        try:
            while True:
                data, client_addr = self.socket.recvfrom(self.buffer_size)
                msg = data.decode('utf-8')
                request = json.loads(msg)
                flag = request.get("flag")
                params= request.get("params")

                print(f"[RECV] flag: {flag}")
                result = self.dispatch(flag, params)

                response = json.dumps(result)
                self.socket.sendto(response.encode('utf-8'), client_addr)
        except KeyboardInterrupt:
            print("\n[UDPServer] Shutting down.")
        finally:
            self.socket.close()

    def dispatch(self, flag: str, params:dict):
        """根据 flag 分派不同处理函数"""
        if flag == "height":
            res=self.dem_inr.get_height(**params)
            return {
                "height": res.tolist()
            }
        elif flag == "slope":
            res= self.dem_inr.get_height_dxdy(**params)
            height, dx, dy, slope = res
            return {
                "height": height.tolist(),
                "dx": dx.tolist(),
                "dy": dy.tolist(),
                "slope": slope.tolist()
            }
        else:
            return "error"




if __name__ == "__main__":

    dem_inr = DEM_INR()
    udp_server = UDPServer(dem_inr, host=SERVER_IP, port=SERVER_PORT)
    udp_server.start()

    pass
