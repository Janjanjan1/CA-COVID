import time
from PIL import Image, ImageShow, ImagePalette
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from pandas.core.algorithms import mode
np.set_printoptions(precision= 15)


def popn_generation(z):
    proportions = [0.6896 - (z/10000), 0.31, 0.0003, 0.0001, (z/10000), 0]
    population = [0, 1, 2, 3, 4, 5]
    popn = np.random.choice(a=population, p=proportions, size=(100, 100), replace=True)
    return popn

def nebor(population1, i, j, d, Pe, Pi):
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
    ne = population1[start1:end1,start2:end2]
    a = (1-Pi)**np.count_nonzero(ne == 3)
    b = (1-Pe)**np.count_nonzero(ne == 2)
    return 1 - (a * b)

@jit(nopython= True)
def vCond(population, time, t, Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir):
    if population == 2 and t >= Tei + time and np.random.random() <= Pei:
        population = 3
        time = t
    elif population == 3:
        if t >= Tir + time and np.random.random() <= Pir:
            population = 5
        elif t >= Tiq + time and np.random.random() <= Piq:
            population = 4
            time = t
    elif population == 4 and t >= Tqr + time and np.random.random() <= Pqr:
        population = 5
    return population, time



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
    return Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d


def CA(parameter, real_q, population1, Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d):
    population = population1.copy()
    modeled_q = np.zeros(real_q.size + 30)
    print(modeled_q.shape)
    time = np.zeros_like(population)
    total = np.count_nonzero(population == 1) + np.count_nonzero(population == 2) + np.count_nonzero(population == 3)
    im =  []
    for t in range(0, modeled_q.size):
        # neighborhoods = neighborsv(x, y, d, Pe, Pi, population, neighborhoods)
        print(np.count_nonzero(population == 1), np.count_nonzero(population == 2), np.count_nonzero(population == 3),
              np.count_nonzero(population == 4), np.count_nonzero(population == 5))
        x, y = population.shape
        for i in range(0, x):
            for j in range(0, y):
                if population[i, j] == 5 or 0:
                    pass
                if population[i, j] == 1:
                    neighborhood = nebor(population, i, j, d, Pe, Pi)
                    if neighborhood >= np.random.random():
                        population[i, j] = 2
                        time[i, j] = t
                else:
                    population[i, j], time[i, j] = vCond(population[i, j], time[i, j], t, Pei, Piq,
                                                         Pir, Pqr, Pe,
                                                         Pi, Pb, Tei, Tiq, Tqr,
                                                         Tir)

        modeled_q[t] = np.count_nonzero(population == 4)
        im.append(Image.fromarray(population).convert('P'))
        if modeled_q[t] == total:
            modeled_q[t:] = total
            break
    neighborhoods = np.zeros_like(population, dtype=float)
    return modeled_q,im


df = pd.read_csv("data.csv", usecols=['Confirmed'])
real_q = df["Confirmed"].to_numpy()
parameters= 1
population1 = popn_generation(real_q[0])
parameter = input("Enter Parameters: ")
Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d = unpack_para(parameter)
print(Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d )
modeled_q,im = CA(parameter, real_q, population1, Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d)
palette = [0,0,0,255, 255, 255,255,69,0,255,0,0,255,192,203,0,128,0]
im[0].save('spread.gif', save_all=True, append_images=im[1:],
           optimize=True, duration=5, loop=True, palette = palette)
print(real_q.size)
y = np.arange(0,np.size(modeled_q))
print(y.shape)
temp = np.zeros(real_q.size + 30)
for i in range(0, real_q.size):
    temp[i] = real_q[i]
real_q = temp
print(modeled_q.shape)
print(modeled_q)
print(real_q)
plt.plot(y,real_q,'o', label = "real_q")
plt.plot(y,modeled_q, label = "modeled_q")
temp_data = open("model.csv", "w+")
temp_data.close()
model = pd.DataFrame(modeled_q).to_csv('model.csv')

plt.show()

