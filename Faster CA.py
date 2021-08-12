import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from numba import njit

def popn_generation(z):
    proportions = [0.6896 - (z/10000), 0.31, 0.0003, 0.0001, (z/10000), 0]
    population = [0, 1, 2, 3, 4, 5]
    popn = np.random.choice(a=population, p=proportions, size=(100, 100), replace=True)
    return popn

def fitness_func(modeled_q, real_q):
    q = np.sum(np.absolute(modeled_q - real_q))
    fitness = (1 / q)
    return fitness

def unpack_para(parameter1):
    Pei = int(parameter1[0:10], 2) / 1000
    Piq = int(parameter1[10:20], 2) / 1000
    Pir = int(parameter1[20:30], 2) / 1000
    Pqr = int(parameter1[30:40], 2) / 1000
    Pe = int(parameter1[40:50], 2) / 1000
    Pi = int(parameter1[50:60], 2) / 1000
    Pb = int(parameter1[60:70], 2) / 1000
    Tei = int(parameter1[70:80], 2)
    Tiq = int(parameter1[80:90], 2)
    Tqr = int(parameter1[90:100], 2)
    Tir = int(parameter1[100:110], 2)
    d = int(parameter1[110:120], 2)
    return np.array([Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d])

@njit()
def nebor2(population1, i, j, d, Pe, Pi):
    start1 = i - d
    end1 = i + d
    start2 = j - d
    end2 = j + d
    if start1 < 0:
        start1 = 0
    if start2 < 0:
        start2 = 0
    if end1 > 99:
        end1 = 98
    if end2 > 99:
        end2 = 98

    ne = population1[start1:end1+1,start2:end2+1]
    a = (1-Pi)**np.count_nonzero(ne == 3)
    b = (1-Pe)**np.count_nonzero(ne == 2)
    return 1 - (a * b)
@njit()
def loop(popn,modeled_q1,parameters,total):
    time = np.zeros_like(popn)
    for t in range(0,modeled_q1.size):
        z = 0
        new = np.zeros_like(popn)
        for i in range(0,100):
            for j in range(0,100):
                cell = popn[i,j]
                tem = time[i,j]
                if cell == 0 or cell == 5:
                    new[i,j] = cell
                elif cell == 1:
                    if nebor2(popn,i,j,parameters[11],parameters[4], parameters[5]) >= np.random.random():
                        new[i,j] = 2
                    else:
                        new[i,j] = 1
                elif cell == 2 and  t>= parameters[7] + tem and np.random.random() <= parameters[0]:
                    new[i,j] = 3
                    time[i,j] = t
                elif cell == 3:
                    if t >= parameters[10] + tem and np.random.random() <= parameters[2]:
                        new[i,j] = 5
                    elif t >= parameters[8] + tem and np.random.random() <= parameters[1]:
                        new[i,j] = 4
                        time[i,j] = t
                        z += 1
                    else: 
                        new[i,j] = 3
                elif cell == 4 and t>= parameters[9] + tem and np.random.random() <= parameters[3]:
                    new[i,j] = 5
                    z -= 1
                else:
                    new[i,j] = cell
        popn = new

        print(np.count_nonzero(popn == 1), np.count_nonzero(popn == 2), np.count_nonzero(popn== 3), np.count_nonzero(popn==4),np.count_nonzero(popn==5))
        modeled_q1[t] = modeled_q1[t-1] + z
        if modeled_q1[t] == total:
            modeled_q1[t:] = total
            return modeled_q1
    return modeled_q1

def tr(real_q):
    popn = popn_generation(real_q[0])
    z = np.count_nonzero(popn == 4)
    total = np.count_nonzero(popn > 0)
    modeled_q = np.zeros(real_q.size + 100)
    modeled_q[-1] = z
    print(modeled_q.size)
    p = unpack_para('011110100000100010011010000011000110001110111110000011001101100000110100000011010000110110000001101000011111010000000011')
    print(p)
    modeled_q1 = loop(popn.copy(),modeled_q.copy(),p,total)
    print(fitness_func(modeled_q1[0:real_q.size],real_q))
    y = np.arange(0,np.size(modeled_q1))
    temp = np.zeros(real_q.size + 100)
    for i in range(0, real_q.size):
        temp[i] = real_q[i]
    real_q = temp
    plt.plot(y,real_q,'o', label = "real_q")
    plt.plot(y,modeled_q1, label = "modeled_q")
    temp_data = open("model.csv", "w+")
    temp_data.close()
    model = pd.DataFrame(modeled_q).to_csv('model.csv')
    plt.show()
    return
df = pd.read_csv("lol.csv", usecols=['Confirmed'])
real_q = df["Confirmed"].to_numpy()
tr(real_q)