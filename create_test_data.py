import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def get_Q(Q_sig, dt=1):
    Q = Q_sig * np.asarray([
        [(1/3)*np.power(dt, 3), 0, (1/2)*np.power(dt, 2), 0],
        [0, (1/3)*np.power(dt, 3), 0, (1/2)*np.power(dt, 2)],
        [(1/2) * np.power(dt, 2), 0, dt, 0],
        [0, (1/2) * np.power(dt, 2), 0, dt]
    ])
    return Q


def get_R(R_sig):
    R = R_sig * np.eye(2)
    return R

# Time
t = 200
# transformation matrix
F = np.asarray([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# measurement model
H = np.asanyarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
# Let's assume two Kalman filters
Q = [get_Q(0.3), get_Q(0.1)]
R = [get_R(1), get_R(3)]

# Create path
kf_ind = 0
kf_change_pnt = [100]
x_tminus1 = np.asarray([[0], [0], [1], [1]])
path = []
meas = []
for i in range(t):
    x_t_ = F @ x_tminus1
    x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
    x_t = x_t.reshape((4, 1))
    x_tminus1 = x_t
    y_t_ = H @ x_t
    y_t = np.random.multivariate_normal(np.squeeze(y_t_), R[kf_ind])
    y_t = y_t.reshape((2, 1))
    if i in kf_change_pnt:
        kf_ind += 1
    path.append(x_t)
    meas.append(y_t)
path = np.squeeze(np.asarray(path))
meas = np.squeeze(np.asarray(meas))
plt.figure()
plt.plot(meas[:,0], meas[:,1], 'r-o', label='measurements')
plt.plot(path[:,0], path[:,1], 'b-o', label='True trajectory')
plt.legend()

print('F:')
print(F)
print('H:')
print(H)
for i in range(len(kf_change_pnt)+1):
    print('Q_' + str(i) + ': ')
    print(Q[i])
    print('R_' + str(i) + ': ')
    print(R[i])
data = pd.DataFrame(meas)
data.to_csv('data/measurement1.csv', index=False, header=False)
