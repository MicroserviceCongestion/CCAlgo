import threading
from socket import socket
from adaptive_qps import AdaptiveQPSHandler  # 导入自定义的QPS处理模块

SERVER_IP = 'localhost'
SERVER_PORT = 32012

qps = 0


def socket_handler(cs: socket):
    while True:
        data = cs.recv(1024)
        if not data:
            break
        msg = data.decode()
        for line in msg.splitlines():
            sp = line.split(':')
            if len(sp) != 2:
                continue
            if sp[0] == 'QPS':
                global qps
                qps = int(sp[1])
                # if qps > 0:
                #     print('Received QPS:', qps)
    cs.close()


def qps_handler(cs: socket, handler: AdaptiveQPSHandler):
    while True:
        print('CurrentQPS:', qps)
        handler.reset(qps)
        result = int(handler.get_max_qps(qps))
        print('MaxQPS:', result)
        cs.send(f'MaxQPS:{result}\n'.encode())


def start_server():
    ss = socket()
    ss.bind((SERVER_IP, SERVER_PORT))
    ss.listen()
    print(f'Server started at {SERVER_IP}:{SERVER_PORT}')
    while True:
        cs, addr = ss.accept()
        print(f'Client connected from {addr}')
        handler = AdaptiveQPSHandler()
        threading.Thread(target=socket_handler, args=(cs,)).start()
        threading.Thread(target=qps_handler, args=(cs, handler)).start()


if __name__ == '__main__':
    start_server()
