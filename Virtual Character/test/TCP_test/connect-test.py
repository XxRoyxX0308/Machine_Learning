import socket

HOST = '127.0.12.12'
PORT = 1212

###伺服器端
##s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
##s.bind((HOST, PORT))
##s.listen(5)
##
##print('server start at: %s:%s' % (HOST, PORT))
##print('wait for connection...')
##
##while True:
##    conn, addr = s.accept()
##    print('connected by ' + str(addr))
##
##    while True:
##        indata = conn.recv(1024)
##        if len(indata) == 0: # connection closed
##            conn.close()
##            print('client closed connection.')
##            break
##        print('recv: ' + indata.decode())
##
##        outdata = 'echo ' + indata.decode()
##        conn.send(outdata.encode())


        
#客戶端
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    outdata = input('please input message: ')
    print('send: ' + outdata)
    s.send(outdata.encode())
##    
##    indata = s.recv(1024)
##    if len(indata) == 0: # connection closed
##        s.close()
##        print('server closed connection.')
##        break
##    print('recv: ' + indata.decode())
