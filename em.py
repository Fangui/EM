import math
import numpy as np
import random

def sdp_mat(dim, sd):
    M = np.array(np.random.rand(dim, dim))
    M = 0.5 * (M + M.T)
    M = M + sd * np.eye(dim)
    return M

"""
Generate a symmetric definite positive matrice
"""
def get_sdp_mat(sd_diff, nb_gauss, dim):
    SDClass = np.random.rand(1, nb_gauss) + sd_diff

    return [sdp_mat(dim, i) for i in SDClass[0]]

def gen_gmm(theta, nb_pts, nb_gauss, mu, cov):
    r = np.random.multinomial(nb_pts, theta)

    data = []
    for i in range (nb_gauss):
        gmm = np.random.multivariate_normal(mu[i], cov[i], r[i])
        for j in range (r[i]):
            data.append(gmm[j])
    """
    data = np.random.multivariate_normal(mu[0], cov[0], nb_pts) #fixme  [nb_pts, dim]
    """
    return data

def update_mu(mu, s_weight, data, weights_array):
    for d in range (len(data[0])):
        mu[d] = 0

    for i in range (len(data)):
        for d in range (dim):
            mu[d] += data[i][d] * weights_array[i]

    mu /= s_weight

def var_flor(s, f):
    l = np.linalg.cholesky(s)

    l_inv = np.linalg.inv(l)
    t = l_inv * s * l_inv.T

    u, d = np.linalg.eig(t)

    for i in range (len(d)):
        d[i][i] = max(1, d[i][i])

    t_ = u * d * u.T

    return l * t_ * l.T

def update_cov(s_weight, data_array, mu_array, weights_array):
    cov = np.zeros([dim, dim])

    for i in range (len(data_array)):
        mat = np.asmatrix(data_array[i]) - np.asmatrix(mu_array)
        cov += weights_array[i] * (mat.T @ mat)

    return cov / s_weight

def log_multivar(point, dim, mu, cov):
    q =  -(dim / 2) * np.log (2 * math.pi) - 0.5 * np.log(np.linalg.det(cov))
    e = -0.5 * (point - mu).T @ np.linalg.inv(cov) @ (point - mu)
    return q + e

def compute_weight(weights, nb_gauss, dim, alpha, data, mu, cov):
    llk = 0
    for i in range (len(data)):
        maxi = -math.inf
        for j in range (nb_gauss):
            weights[j][i] = log_multivar(data[i], dim, mu[j], cov[j])
            weights[j][i] += np.log(alpha[j])

            if weights[j][i] > maxi:
                maxi = weights[j][i]
            llk += weights[j][i]

        tot = 0
        for j in range (nb_gauss):
            val = np.exp(weights[j][i] - maxi)
            weights[j][i] = val
            tot += val

        for j in range (nb_gauss):
            weights[j][i] /= tot

    return llk

def compare_mu(a, b):
    dif = np.empty([len(a), len(a[0])])

    for i in range (len(a)):
        for j in range (len(a[0])):
            dif[i][j] = np.abs(a[i][j] - b[i][j])

    return dif

def compare_cov(a, b):
    dif = np.empty([len(a), len(a[0]), len(a[0][0])])

    for i in range (len(a)):
        for j in range (len(a[0])):
            for k in range (len(a[0][0])):
                dif[i][j][k] = np.abs(a[i][j][k] - b[i][j][k])

    return dif

def debug_print(target_mu, target_cov, mu, cov):
    print ("### Target parameters")
    print ("mu:")
    print (target_mu)

    print ("cov:")
    print (target_cov)

    print ("### Generate parameters")
    print ("Generate mu:")
    print (mu)
    print ("Generate cov:")
    print (cov)


if __name__ == '__main__':
    nb_pts = 15000
    nb_gauss = 3
    dim = 5

    distanceBTWclasses = 30

    alpha = np.repeat(1.0 / nb_gauss, nb_gauss)
    target_mu = [(np.random.random(dim) * distanceBTWclasses * i) for i in range(1, nb_gauss + 1)]
    target_cov = get_sdp_mat(4, nb_gauss, dim)

    data = gen_gmm(alpha, nb_pts, nb_gauss, target_mu, target_cov)

    cov = get_sdp_mat(4, nb_gauss, dim)
    mu = [(np.random.random(dim) * distanceBTWclasses * i) for i in range(1, nb_gauss + 1)]

    #debug_print(target_mu, target_cov, mu, cov)
    weights = np.empty([nb_gauss, nb_pts])

    prev = -math.inf
    llk = -9999999 # FIXME
    i = 0

#   print (compare_mu(target_mu, mu))
    while i < 10 or llk - prev > 0.01:
        if i > 30:
            break
        """
        if False and i % 10 == 0:
            plot_data(data)
        """
        i += 1
        prev = llk
        llk = compute_weight(weights, nb_gauss, dim, alpha, data, mu, cov)
        for j in range (nb_gauss):
            s_weight = np.sum(weights[j])

            alpha[j] = s_weight / nb_pts
            update_mu(mu[j], s_weight, data, weights[j])
            cov[j] = update_cov(s_weight, data, mu[j], weights[j])

    print("###After EM:")

    print ("diff mu")
    print (compare_mu(target_mu, mu))

    print ("diff cov")
    print (compare_cov(target_cov, cov))

    print ("stop after " + str(i) + " iterations")
    print (prev)
    print (llk)
