import threading
from socket import socket

SERVER_IP = 'localhost'
SERVER_PORT = 32012


def get_max_qps(qps: int) -> int:
    return max(int(qps * 0.5), 10)


def socket_handler(cs: socket):
    while True:
        data = cs.recv(1024)
        if not data:
            break
        msg = data.decode()
        for line in msg.splitlines():
            print(line)
            sp = line.split(':')
            if len(sp) != 2:
                continue
            result = get_max_qps(int(sp[1]))
            if sp[0] == 'QPS':
                print('Received QPS:', sp[1], 'Result:', result)
                cs.send(f'MaxQPS:{result}\n'.encode())

    cs.close()


def start_server():
    ss = socket()
    ss.bind((SERVER_IP, SERVER_PORT))
    ss.listen()
    print(f'Server started at {SERVER_IP}:{SERVER_PORT}')
    while True:
        cs, addr = ss.accept()
        print(f'Client connected from {addr}')
        threading.Thread(target=socket_handler, args=(cs,)).start()


if __name__ == '__main__':
    start_server()
