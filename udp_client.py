import sys
import os
import socket
import json
import math
# 配置目标服务器IP和端口
SERVER_IP = '127.0.0.1'
SERVER_PORT = 9999


if __name__== "__main__":
    # 创建 UDP 套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    buffer_size = 1024
    msg1={
        "flag": "height",
        "params": {
            "coords": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        }
    }

    msg1_json=json.dumps(msg1).encode('utf-8')
    # 发送数据
    client_socket.sendto(msg1_json, (SERVER_IP, SERVER_PORT))
    # 接收响应
    data, server_addr = client_socket.recvfrom(buffer_size)
    rec1=json.loads(data.decode('utf-8'))
    print(f"[RESPONSE] {rec1}")

    msg2={
        "flag": "slope",
        "params": {
            "coords": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "angle": math.pi/3
        }
    }
    msg2_json=json.dumps(msg2).encode('utf-8')
    # 发送数据
    client_socket.sendto(msg2_json, (SERVER_IP, SERVER_PORT))
    # 接收响应
    data, server_addr = client_socket.recvfrom(buffer_size)
    rec2=json.loads(data.decode('utf-8'))
    print(f"[RESPONSE] {rec2}")

