import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def get_Q(Q_sig, dt=1):
    Q = (Q_sig ** 2) * np.asarray([
        [(1/3)*np.power(dt, 3), 0, (1/2)*np.power(dt, 2), 0],
        [0, (1/3)*np.power(dt, 3), 0, (1/2)*np.power(dt, 2)],
        [(1/2) * np.power(dt, 2), 0, dt, 0],
        [0, (1/2) * np.power(dt, 2), 0, dt]
    ])
    return Q


def get_Q_RW(Q_sig):
    Q = (Q_sig ** 2) * np.diag([1, 1, 0, 0])
    return Q

def get_R(R_sig):
    R = (R_sig ** 2) * np.eye(2)
    return R

def get_R_RW(R_sig):
    R = (R_sig ** 2) * np.eye(2)
    return R

# transformation matrix
F_CV = np.asarray([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
F_RW = np.eye(4)
# measurement model
H_CV = np.asanyarray([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])
H_RW = np.asanyarray([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])


def create_path_constant_volocity_one_model():
    # Time
    t = 200
    # Let's assume two Kalman filters
    Q = [get_Q(0.5)]
    R = [get_R(1.0)]

    kf_ind = 0
    kf_change_pnt = []
    x_tminus1 = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    path = []
    meas = []
    for i in range(t):
        x_t_ = F_CV @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((4, 1))
        y_t_ = H_CV @ x_t
        y_t = np.random.multivariate_normal(np.squeeze(y_t_), R[kf_ind])
        y_t = y_t.reshape((2, 1))
        if i in kf_change_pnt:
            kf_ind += 1
        path.append(x_t)
        meas.append(y_t)
        x_tminus1 = x_t
    path = np.squeeze(np.asarray(path))
    meas = np.squeeze(np.asarray(meas))
    plt.figure()
    plt.plot(meas[:, 0], meas[:, 1], 'r-o', label='measurements')
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='True trajectory')
    plt.legend()

    print('F:')
    print(F_CV)
    print('H:')
    print(H_CV)
    for i in range(len(kf_change_pnt) + 1):
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
    meas_df = pd.DataFrame(meas)
    meas_df.to_csv('data/measurement1.csv', index=False, header=False)
    truth_df = pd.DataFrame(path)
    truth_df.to_csv('data/groundtruth1.csv', index=False, header=False)


def create_path_constant_volocity():
    # Time
    t = 100
    # Let's assume two Kalman filters
    Q = [get_Q(1.0), get_Q(6.0), get_Q(1.0)]
    R = [get_R(1.05), get_R(0.85), get_R(1.5)]

    kf_ind = 0
    kf_change_pnt = [65, 80]
    x_tminus1 = np.asarray([[0.0], [0.0], [-0.5], [-0.5]])
    path = []
    meas = []
    for i in range(t):
        x_t_ = F_CV @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((4, 1))
        y_t_ = H_CV @ x_t
        y_t = np.random.multivariate_normal(np.squeeze(y_t_), R[kf_ind])
        y_t = y_t.reshape((2, 1))
        if i in kf_change_pnt:
            kf_ind += 1
        path.append(x_t)
        meas.append(y_t)
        x_tminus1 = x_t
    path = np.squeeze(np.asarray(path))
    meas = np.squeeze(np.asarray(meas))
    plt.figure()
    plt.plot(meas[:, 0], meas[:, 1], 'r-o', label='measurements')
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='True trajectory')
    plt.legend()

    print('F:')
    print(F_CV)
    print('H:')
    print(H_CV)
    for i in range(len(kf_change_pnt) + 1):
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
    meas_df = pd.DataFrame(meas)
    meas_df.to_csv('data/measurement2.csv', index=False, header=False)
    truth_df = pd.DataFrame(path)
    truth_df.to_csv('data/groundtruth2.csv', index=False, header=False)


# Create path
def create_path_random_walk():
    # Time
    t = 200
    # Let's assume two Kalman filters
    Q = [get_Q_RW(2.0), get_Q_RW(10.0)]
    R = [get_R_RW(1.5), get_R_RW(1.75)]

    kf_ind = 0
    kf_change_pnt = [80]
    x_tminus1 = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    path = []
    meas = []
    for i in range(t):
        x_t_ = F_RW @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((4, 1))
        x_tminus1 = x_t
        y_t_ = H_RW @ x_t
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
    print(F_RW)
    print('H:')
    print(H_RW)
    for i in range(len(kf_change_pnt)+1):
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
    meas_df = pd.DataFrame(meas)
    meas_df.to_csv('data/measurement3.csv', index=False, header=False)
    truth_df = pd.DataFrame(path)
    truth_df.to_csv('data/groundtruth3.csv', index=False, header=False)


def create_path_cv_rw():
    # Time
    t = 200
    # Let's assume two Kalman filters
    Q = [get_Q_RW(6.0), get_Q(0.5)]
    R = [get_R_RW(0.45), get_R(0.5)]
    F = [F_RW, F_CV]
    H = [H_RW, H_CV]

    kf_ind = 0
    kf_change_pnt = [100]
    x_tminus1 = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    path = []
    meas = []
    for i in range(t):
        x_t_ = F[kf_ind] @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((4, 1))
        y_t_ = H[kf_ind] @ x_t
        y_t = np.random.multivariate_normal(np.squeeze(y_t_), R[kf_ind])
        y_t = y_t.reshape((2, 1))
        if i in kf_change_pnt:
            kf_ind += 1
        path.append(x_t)
        meas.append(y_t)
        x_tminus1 = x_t
    path = np.squeeze(np.asarray(path))
    meas = np.squeeze(np.asarray(meas))
    plt.figure()
    plt.plot(meas[:, 0], meas[:, 1], 'r-o', label='measurements')
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='True trajectory')
    plt.legend()

    print('F:')
    print(F_CV)
    print('H:')
    print(H_CV)
    for i in range(len(kf_change_pnt) + 1):
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
    meas_df = pd.DataFrame(meas)
    meas_df.to_csv('data/measurement4.csv', index=False, header=False)
    truth_df = pd.DataFrame(path)
    truth_df.to_csv('data/groundtruth4.csv', index=False, header=False)

# create_path_constant_volocity_one_model()
# create_path_constant_volocity()
# create_path_random_walk()
create_path_cv_rw()


