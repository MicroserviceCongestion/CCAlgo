import os
import threading
import time
from socket import socket
from adaptive_qps import AdaptiveQPSHandler  # 导入自定义的QPS处理模块

SERVER_IP = 'localhost'
SERVER_PORT = 32012

CALCULATE_INTERVAL_MS = 500


class QpsManager:
    def __init__(self, handler, cs):
        self.qps = 0
        self.cpu = 0.0
        self.handler = handler
        self.cs = cs
        handler.reset()
        threading.Thread(target=self.socket_handler).start()
        threading.Thread(target=self.qps_handler).start()

    def socket_handler(self):
        while True:
            data = self.cs.recv(1024)
            if not data:
                break
            msg = data.decode()
            for line in msg.splitlines():
                sp = line.split(':')
                if len(sp) != 2:
                    continue
                if sp[0] == 'QPS':
                    self.qps = int(sp[1])
                    # print('received QPS: ', self.qps)
                if sp[0] == 'CPU':
                    self.cpu = float(sp[1])
                    # print('received CPU: ', self.cpu)
        self.cs.close()

    def qps_handler(self):
        while True:
            time.sleep(CALCULATE_INTERVAL_MS / 1000)
            print('Current QPS:', self.qps)
            print('Current CPU:', self.cpu)
            result = float(self.handler.get_max_qps(qps=self.qps, cpu=self.cpu))
            print('MaxQpsRate:', result)
            self.cs.send(f'MaxQpsRate:{result}\n'.encode())


def start_server():
    ss = socket()
    ss.bind((SERVER_IP, SERVER_PORT))
    ss.listen()
    print(f'Server started at {SERVER_IP}:{SERVER_PORT}, calculation interval: {CALCULATE_INTERVAL_MS}ms', flush=True)
    managers = {}
    while True:
        cs, addr = ss.accept()
        print(f'Client connected from {addr}', flush=True)
        handler = AdaptiveQPSHandler()
        managers[addr] = QpsManager(handler, cs)


if __name__ == '__main__':
    if os.getenv("CCALGO_SERVER_IP") is not None:
        SERVER_IP = os.getenv("CCALGO_SERVER_IP")
    if os.getenv("CCALGO_SERVER_PORT") is not None:
        SERVER_PORT = os.getenv("CCALGO_SERVER_PORT")
    if os.getenv("CCALGO_CALCULATE_INTERVAL_MS") is not None:
        CALCULATE_INTERVAL_MS = os.getenv("CCALGO_CALCULATE_INTERVAL_MS")
    start_server()
