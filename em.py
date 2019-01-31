import numpy as np
import math

log_tau = math.log(math.pi * 2)

def data_generate(means, covas, nb_gauss, size):
    data = np.empty([nb_gauss, size, len(means[0])])  # len(means[0]) = dim
    for i in range (nb_gauss):
        data[i] = np.random.multivariate_normal(means[i], covas[i], size)

    data.resize(nb_gauss * size * len(means[0]))
    data.resize(nb_gauss * size, len(means[0]), 1)
    return data

def cova_generate(nb_gauss, dim, range_val):
    covas = np.empty([nb_gauss, dim, dim])
    for i in range (nb_gauss):
        var = range_val * (i + 1) 
        mat = np.random.rand(dim, dim) * range_val
        covas[i] = 0.5 * (mat + mat.T) + np.eye(dim) * var
    return covas

def mean_generate(data, nb_gauss):
    dim = len(data[0])
    means = np.zeros([nb_gauss, dim, 1])
    card = np.zeros([nb_gauss])

    rand_idx = np.random.randint(0, nb_gauss, len(data))
    for i in range (len(data)):
        idx = rand_idx[i]
        means[idx] += data[i]
        card[idx] += 1

    for i in range (nb_gauss):
        means[i] /= card[i]

    return means

def p_log(x, mean, cova):
    det = np.linalg.det(cova)
    fraction = len(mean) * log_tau + math.log(det)

    center = x - mean
    center.resize(dim, 1)

    return -0.5 * (fraction + (center.T).dot(np.linalg.inv(cova)).dot(center))

def update_cova(data, mean, p_cache, gauss):
    dim = len(data[0])
    num = np.zeros([dim, dim])

    for k in range (len(data)):
        center = data[k] - mean
        num += p_cache[k][gauss] * center.dot(center.T)

    return num / s

def expectation(data, p_cache, means, covas, alpha):
    maxi = -math.inf

    for i in range(len(data)): # compute llj
        tab = p_cache[i]

        for j in range(nb_gauss):
            tab[j] = p_log(data[i], means[j], covas[j])
            tab[j] += math.log(alpha[j])
            if maxi < tab[j]:
                maxi = tab[j]

    llk = 0
    for i in range (len(data)): #normalize
        tab = p_cache[i]
        for j in range (nb_gauss):
            tab[j] = math.exp(tab[j] - maxi)
            llk += tab[j]

    for j in range(len(data)):
        tab = p_cache[j]
        tot = np.sum(tab) + np.finfo(float).eps

        for k in range(nb_gauss):
            tab[k] /= tot

    return llk

np.random.seed(0)

nb_gauss = 3
size = 10000
dim = 4
range_val = 5
range_init = 10

means_init = np.random.rand(nb_gauss, dim)
for i in range (nb_gauss):
    means_init[i] *= range_init * (i + 1)
covas_init = cova_generate(nb_gauss, dim, range_init)

data = data_generate(means_init, covas_init, nb_gauss, size)

"""
print ("covariance")
print (covas_init)
print ("means")
print (means_init)
"""

means = mean_generate(data, nb_gauss)
covas = cova_generate(nb_gauss, dim, range_val)
alpha = np.repeat(1.0 / nb_gauss, nb_gauss)

p_cache = np.empty([len(data), nb_gauss])
prev = -1
itera = 0
max_iter = 10

for itera in range (1, max_iter + 1):
    llk = expectation(data, p_cache, means, covas, alpha)

    if abs(llk - prev) < 0.001:
        break

    print('iter:', itera, 'llk =', llk)
    prev = llk

    for i in range (nb_gauss): #Maximization
        s = 0
        for j in range (len(data)):
            s += p_cache[j][i]

        alpha[i] = s / len(data)

        num = np.zeros([dim, 1]) # Update mean
        for j in range (len(data)):
            num += data[j] * p_cache[j][i]
        means[i] = num / s

        covas[i] = update_cova(data, means[i], p_cache, i)

"""
means.resize(nb_gauss, dim)
print ("covariance")
print (covas)
print ("means")
print (means)
"""

score = 0
histo = np.zeros([nb_gauss], dtype=int)
histo_idx = [-1] * nb_gauss

for i in range(len(data)):
    if (i + 1) % size == 0:
        b_idx = np.argmax(histo)
        if histo_idx[b_idx] != -1:
            print ("Fail training EM")
        else:
            score += histo[b_idx]
        print (histo)
        histo = np.zeros([nb_gauss], dtype=int)
        histo_idx[b_idx] = 0

    b_idx = np.argmax(p_cache[i])
    histo[b_idx] += 1

print ("end with :", itera, "iterations")
print ("Score:", score, "/", len(data))
