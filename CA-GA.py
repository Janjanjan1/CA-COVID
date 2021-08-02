import numpy as np
import time as tem
import pandas as pd
from numba import jit
from PIL import Image, ImageShow, ImagePalette
import matplotlib.pyplot as plt


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


def B2i(string, powers):
    string = list(string)
    string = np.array(string, dtype=int)
    i = np.multiply(powers, string)
    return i


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


def fitness_func(modeled_q, real_q):
    q = np.sum(np.absolute(modeled_q - real_q))
    fitness = (1 / q)
    return fitness


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


def u1npack_para(parameter1):
    powers = np.flip(2 ** np.arange(10))
    Pei = B2i(parameter1[0:10], powers) / 1000
    Piq = B2i(parameter1[10:20], powers) / 1000
    Pir = B2i(parameter1[20:30], powers) / 1000
    Pqr = B2i(parameter1[30:40], powers) / 1000
    Pe = B2i(parameter1[40:50], powers) / 1000
    Pi = B2i(parameter1[50:60], powers) / 1000
    Pb = B2i(parameter1[60:70], powers) / 1000
    Tei = B2i(parameter1[70:80], powers)
    Tiq = B2i(parameter1[80:90], powers)
    Tqr = B2i(parameter1[90:100], powers)
    Tir = B2i(parameter1[100:110], powers)
    d = B2i(parameter1[110:120], powers)
    return Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d


def popn_generation(z):
    proportions = [0.6896 - (z/10000), 0.31, 0.0003, 0.0001, (z/10000), 0]
    population = [0, 1, 2, 3, 4, 5]
    popn = np.random.choice(a=population, p=proportions, size=(100, 100), replace=True)
    return popn


def neighborsv(i, j, d, Pe, Pi, population1, neighborhoods):
    ne = (population1.take(range(i - d, i + d + 1), axis=0, mode='wrap').take(range(j - d, j + d + 1), axis=1,
                                                                              mode='wrap'))
    neighborhoods[i, j] = (1 - ((1 - Pi) ** (np.count_nonzero(ne == 3)) * (1 - Pe) ** (np.count_nonzero(ne == 2))))
    return


def nebor(population1, i, j, d, Pe, Pi):
    ne = (population1.take(range(i - d, i + d + 1), axis=0, mode='wrap').take(range(j - d, j + d + 1), axis=1,
                                                                              mode='wrap'))
    return 1 - ((1 - Pi) ** (np.count_nonzero(ne == 3)) * (1 - Pe) ** (np.count_nonzero(ne == 2)))


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

    ne = population1[start1:end1,start2:end2]
    a = (1-Pi)**np.count_nonzero(ne == 3)
    b = (1-Pe)**np.count_nonzero(ne == 2)
    return 1 - (a * b)


def vCond(population, time, t, Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir,z):
    if population == 2 and t >= Tei + time and np.random.random() <= Pei:
        population = 3
        time = t
    elif population == 3:
        if t >= Tir + time and np.random.random() <= Pir:
            population = 5
        elif t >= Tiq + time and np.random.random() <= Piq:
            population = 4
            time = t
            z +=1
    elif population == 4 and t >= Tqr + time and np.random.random() <= Pqr:
        population = 5
        z -= 1
    return population, time,z


def CA(parameter, real_q, population1, Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d,z):
    population = population1.copy()
    modeled_q1= np.zeros_like(real_q)
    modeled_q1[-1] = z
    time = np.zeros_like(population)
    total = np.count_nonzero(population == 1) + np.count_nonzero(population == 2) + np.count_nonzero(population == 3)
    x, y = population.shape
    for t in range(0, modeled_q1.size):
        # neighborhoods = neighborsv(x, y, d, Pe, Pi, population, neighborhoods)
        # print(np.count_nonzero(population == 1), np.count_nonzero(population == 2), np.count_nonzero(population == 3),
        #       np.count_nonzero(population == 4), np.count_nonzero(population == 5))
        z = 0
        for i in range(0, x):
            for j in range(0, y):
                if population[i, j] == 5 or 0:
                    pass
                if population[i, j] == 1:
                    neighborhood = nebor2(population, i, j, d, Pe, Pi)
                    if neighborhood >= np.random.random():
                        population[i, j] = 2
                        time[i, j] = t
                else:
                    population[i, j], time[i, j],z = vCond(population[i, j], time[i, j], t, Pei, Piq,
                                                         Pir, Pqr, Pe,
                                                         Pi, Pb, Tei, Tiq, Tqr,
                                                         Tir,z)

        modeled_q1[t] = modeled_q1[t-1] + z
        if modeled_q1[t] == total:
            modeled_q1[t:] = total
            break
    neighborhoods = np.zeros_like(population, dtype=float)
    return modeled_q1




def main(real_q):
    population1 = popn_generation(real_q[0])
    parameters = parameter_gen(30)
    neighborhoods = np.zeros_like(population1, dtype=float)
    # neighbors_vec = np.vectorize(neighborsv, otypes=[], excluded={2, 3, 4, 5, 6})
    # vecCond = np.vectorize(vCond, otypes=[np.ndarray, np.ndarray])
    parameter = parameters['gene']
    z = np.count_nonzero(population1 == 4)
    for t in range(0, 10):
        print("Generation " + str(t))
        a = tem.time()
        parameter = parameters['gene']
        fit = parameters['fits']
        for i in range(0, int(np.size(parameters) * 0.9)):
            Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir, d = unpack_para(parameter[i])
            modeled_q = CA(parameter[i], real_q, population1.copy(), Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir,
                           d,z)

            fit[i] = fitness_func(modeled_q, real_q)
        print(tem.time() - a)
        parameters = np.sort(parameters, order='fits')
        print(parameters)
        print(z)
        parameters = crossingover(parameters)

    for i in range(0, np.size(parameters)):
        modeled_q =  CA(parameter[i], real_q, population1.copy(), Pei, Piq, Pir, Pqr, Pe, Pi, Pb, Tei, Tiq, Tqr, Tir,
                           d,z)
        fit[i] = fitness_func(modeled_q, real_q)
    parameters = np.sort(parameters, order='fits')
    return parameters


df = pd.read_csv("lol.csv", usecols=['Confirmed'])
real_q = df["Confirmed"].to_numpy()
parameters = main(real_q)
print(parameters[-1])
print(unpack_para(parameters[-1][1]))
print(parameters)
