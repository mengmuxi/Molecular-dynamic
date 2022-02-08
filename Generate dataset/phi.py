import numpy as np
import math
from os.path import dirname, join as pjoin
import os
import glob
from tqdm import tqdm
#Define Variables
v=0.03
eta=0
L=5
N=1000
T=1000
w=0.3*np.pi
system = 100
#文件夹生成
data_path = "./N1000_L5_ETA0w0.3/train"
φ_path = "./N1000_L5_ETA0w0.3/train_φ"
if not os.path.isdir(data_path):
    os.makedirs(data_path)
if not os.path.isdir(φ_path):
    os.makedirs(φ_path)
else:
    data_list = glob.glob(data_path + "/*.dat")
    for i in range(len(data_list)):
        os.remove(data_list[i])
    va_list = glob.glob(φ_path + "/*.dat")
    for i in range(len(va_list)):
        os.remove(va_list[i])


va = []
x_av = N / (L * L)
dis_min_1 = []
φ=[]
def initialize():
    global x,y,theta,theta_new,order,order_new,order_inst
    va_x = 0
    va_y = 0
    x=np.random.uniform(0,L,N)
    y=np.random.uniform(0,L,N)
    theta=np.random.uniform(-np.pi,np.pi,N)
    theta_new=np.random.uniform(-np.pi,np.pi,N)
    order=0
    order_new=1
    order_inst=[]
    for i in range(N):
        va_x = (va_x + v * np.cos(theta[i]))
        va_y = (va_y + v * np.sin(theta[i]))
    va.append(np.sqrt(va_x ** 2 + va_y ** 2) / (N * v))

#Functions
def dist(i,j):
    return np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)


def update():
    va_x = 0
    va_y = 0

    for i in range(N):
        theta_x = 0
        theta_y = 0
        num = 0
        for j in range(N):
            y_ij = y[j] - y[i]
            x_ij = x[j]-x[i]
            angle1 = theta[i]
            angle2 = math.atan2(y_ij, x_ij)
            if angle1*angle2 >= 0:
                angle = abs(angle1-angle2)
            else:
                angle = abs(angle1)+abs(angle2)
                if angle > np.pi:
                    angle = 2*np.pi-angle

            if dist(i, j) < 1 and angle < w/2:
                num = num+1
                theta_x = theta_x+v*np.cos(theta[j])
                theta_y = theta_y+v*np.sin(theta[j])
        if num == 0:
            theta_new[i] = theta[i]
        else:
            theta_x = theta_x/num
            theta_y = theta_y/num
            theta_new[i] = math.atan2(theta_y,theta_x) + np.random.uniform(-eta/2,eta/2)

    for i in range(N):
        theta[i] = theta_new[i]
        x[i] = (x[i] + v * np.cos(theta[i])) % L
        y[i] = (y[i] + v * np.sin(theta[i])) % L
        va_x = (va_x + v * np.cos(theta[i]))
        va_y = (va_y + v * np.sin(theta[i]))
    va.append(np.sqrt(va_x ** 2 + va_y ** 2) / (N * v))

    dis_min_1 = []
    for a in range(1,L+1):#x轴
        for b in range(1,L+1):#y轴
            s = 0
            for k in range(N):
                if a-1 < x[k] < a and b-1 < y[k] < b:
                    s = s+1
            #dis_min_2.append(s)#所有时间的每个格子的粒子数在同一个列表
            dis_min_1.append(s)#每个时间一个列表


    count_2 = 0#计算φ
    for p in range(L*L):
        count_2 = count_2+abs(dis_min_1[p]-x_av)
        dis__=count_2/(2*x_av*(L*L-1))
    φ.append(dis__)

if __name__ == '__main__':
    for icon in range (1,system+1):#改
        φ=[]
        #va=[]
        initialize()
        partciles = np.zeros((N, 4))
        for i in range(N):
            partciles[i, 0] = x[i]
            partciles[i, 1] = y[i]
            partciles[i, 2] = v * np.cos(theta[i])
            partciles[i, 3] = v * np.sin(theta[i])
        save_path = pjoin(data_path,'Con'+str(icon)+'eta'+str(eta)+'N'+str(N)+'L'+str(L)+'w'+str(w/np.pi)+'t'+str(0)+'.csv')
        np.savetxt(save_path, partciles, delimiter=", ")

        for t in tqdm(range(1,T+1)):
            update()
            partciles = np.zeros((N, 4))
            for i in range(N):
                partciles[i,0]=x[i]
                partciles[i,1]=y[i]
                partciles[i,2] = v * np.cos(theta[i])
                partciles[i,3] = v * np.sin(theta[i])

            save_path = pjoin(data_path,
                              'Con' + str(icon) + 'eta' + str(eta) + 'N' + str(N) + 'L' + str(L) + 'w' + str(w/np.pi) + 't' + str(t) + '.csv')

            np.savetxt(save_path,partciles , delimiter=", ")



        save_φ_path =pjoin(φ_path,'Con' + str(icon) + 'eta' + str(eta) + 'N' + str(N) + 'L' + str(L) + 'w' + str(w/np.pi)+'t'+ str(t)+'.csv')
        np.savetxt(save_φ_path, φ, delimiter=", ")