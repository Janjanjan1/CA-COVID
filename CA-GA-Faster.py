import numpy as np
from numba import njit
import time as tem
import pandas as pd


def parameter_gen(n):
    binary_repr_v = np.vectorize(np.binary_repr)
    b = np.array(np.random.randint(0, 1000, (n // 2, 7)))
    c = np.array(np.random.randint(0, 30, (n // 2, 3)))
    e = np.array(np.random.randint(20, 100, (n // 2, 1)))
    f = np.array(np.random.randint(0, 10, (n // 2, 1)))
    d = np.append(b, c, axis=1)
    d = np.append(d, e, axis=1)
    d = np.append(d, f, axis=1)
    d = binary_repr_v(d, 10)
    new = []
    for row in d:
        conc = ""
        for cell in row:
            conc = conc + cell
        new.append(conc)
    new = np.array(new).reshape(n // 2, 1)

    fitness = np.empty(shape=(n // 2, 2), dtype=tuple)
    fitness[:, 1] = new[:, 0]
    fitness[:, 0] = np.zeros(shape=(n // 2))
    fitness = np.core.records.fromarrays(fitness.transpose(),
                                         names='fits, gene',
                                         formats='f8, U120')
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


def popn_generation(z):
    proportions = [0.6896 - (z/10000), 0.31, 0.0003, 0.0001, (z/10000), 0]
    population = [0, 1, 2, 3, 4, 5]
    popn = np.random.choice(a=population, p=proportions, size=(100, 100), replace=True)
    return popn


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
        # print(np.count_nonzero(popn == 1), np.count_nonzero(popn == 2), np.count_nonzero(popn== 3), np.count_nonzero(popn==4),np.count_nonzero(popn==5))
        modeled_q1[t] = modeled_q1[t-1] + z
        if modeled_q1[t] == total:
            modeled_q1[t:] = total
            return modeled_q1
        
    return modeled_q1

def crossingover(parameters):
    index = np.arange(np.size(parameters))
    index1 = index + 1
    probability = (np.size(index1) + 1 - index1) / np.sum(index1)
    probability = np.flip(probability, axis=0)
    parameter = parameters['gene']
    for i in range(0, int(0.3 * np.size(parameters)), 2):
        p1 = parameter[np.random.choice(a=index, p=probability)]
        p2 = p1
        while p2 == p1:
            p2 = parameter[np.random.choice(a=index, p=probability)]
        slice_point = np.random.randint(1, 120)
        temp_p1 = p1[0:slice_point]
        p1 = p1[slice_point:len(p1)]
        temp_p2 = p2[0:slice_point]
        p2 = p2[slice_point:len(p2)]
        of1 = temp_p1 + p2
        of2 = temp_p2 + p1
        if np.random.random() < 0.8:
            of1 = list(of1)
            a = np.random.randint(0, 119)
            b = np.random.randint(a, 121)
            of1[a:b] = np.random.choice(a=['0', '1'], size=len(of1[a:b]))
            of1 = "".join(of1)

        if np.random.random() < 0.8:
            of2 = list(of2)
            a = np.random.randint(0, 119)
            b = np.random.randint(a, 121)
            of2[a:b] = np.random.choice(a=['0', '1'], size=len(of2[a:b]))
            of2 = "".join(of2)
        parameter[i] = of1
        if i + 1 >= int(0.5 * np.size(parameters)):
            pass
        else:
            parameter[i + 1] = of2
        for i in range(int(0.3 * np.size(parameters)), int(0.9 * np.size(parameters))):
            parameter[i] = parameter_gen(2)[0][1]

    return parameters
@njit()
def fitness_func(modeled_q, real_q):
    q = np.sum(np.absolute(modeled_q - real_q))
    fitness = (1 / q)
    return fitness

def main(real_q):
    popn = popn_generation(real_q[0])
    parameters = parameter_gen(50)
    z = np.count_nonzero(popn == 4)
    total = np.count_nonzero(popn > 0)
    modeled_q = np.zeros_like(real_q)
    modeled_q[-1] = z
    for gen in range(0, 1000):
        print("Generation " + str(gen))
        a = tem.time()
        parameter = parameters['gene']
        fit = parameters['fits']
        for i in range(0, int(np.size(parameters) * 0.9)):
            p = unpack_para(parameter[i])
            fit[i] = fitness_func(loop(popn.copy(),modeled_q.copy(),p,total), real_q)
        print(tem.time() - a)
        parameters = np.sort(parameters, order='fits')
        print(parameters)
        parameters = crossingover(parameters)
    return parameters


df = pd.read_csv("data.csv", usecols=['Confirmed'])
real_q = df["Confirmed"].to_numpy()
parameters = main(real_q)

