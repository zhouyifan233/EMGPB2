from emgpb2.models import *
from simulation.path_drawer import draw_path as drawer
import pandas as pd


def create_path_constant_volocity_one_model(output_measurements='data/measurement1.csv',
                                            output_groundtruth='data/groundtruth1.csv',
                                            q=0.5,
                                            r=1.0,
                                            state_dim=4,
                                            obs_dim=2,
                                            t=200):
    # create constant velocity model
    constant_velocity = ConstantVelocity(dt=1.0, q=q, r=r, state_dim=state_dim, obs_dim=obs_dim)
    Q = constant_velocity.Q
    R = constant_velocity.R
    F = constant_velocity.A
    H = constant_velocity.H
    # Start point
    x_tminus1 = np.asarray([[0.0], [0.0], [0.0], [0.0]])

    path = []
    meas = []
    for i in range(t):
        x_t_ = F @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q)
        x_t = x_t.reshape((4, 1))
        y_t_ = H @ x_t
        y_t = np.random.multivariate_normal(np.squeeze(y_t_), R)
        y_t = y_t.reshape((2, 1))
        path.append(x_t)
        meas.append(y_t)
        x_tminus1 = x_t
    path = np.squeeze(np.asarray(path))
    meas = np.squeeze(np.asarray(meas))
    # drawer(meas, path)
    print('F:')
    print(F)
    print('H:')
    print(H)
    print('Q: ')
    print(Q)
    print('R: ')
    print(R)
    print('=====================================================')

    if output_measurements is not None:
        meas_df = pd.DataFrame(meas)
        meas_df.to_csv(output_measurements, index=False, header=False)
    if output_groundtruth is not None:
        truth_df = pd.DataFrame(path)
        truth_df.to_csv(output_groundtruth, index=False, header=False)

    return meas


def create_path_constant_volocity_multi_model(output_measurements='data/measurement2.csv',
                                              output_groundtruth='data/groundtruth2.csv',
                                              q: list = [1.0, 6.0],
                                              r: list = [0.75, 0.5],
                                              state_dim=4,
                                              obs_dim=2,
                                              t=200,
                                              change_pnt=[100]):
    # create constant velocity model
    num_of_models = len(q)
    constant_velocity_list = []
    for i in range(num_of_models):
        constant_velocity_list.append(ConstantVelocity(dt=1.0, q=q[i], r=r[i], state_dim=state_dim, obs_dim=obs_dim))
    Q = []
    R = []
    F = []
    H = []
    for i in range(num_of_models):
        Q.append(constant_velocity_list[i].Q)
        R.append(constant_velocity_list[i].R)
        F.append(constant_velocity_list[i].A)
        H.append(constant_velocity_list[i].H)
    # Start point
    x_tminus1 = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    # model switching controller
    kf_ind = 0
    kf_change_pnt = change_pnt

    path = []
    meas = []
    for i in range(t):
        x_t_ = F[kf_ind] @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((state_dim, 1))
        y_t_ = H[kf_ind] @ x_t
        y_t = np.random.multivariate_normal(np.squeeze(y_t_), R[kf_ind])
        y_t = y_t.reshape((obs_dim, 1))
        if i in kf_change_pnt:
            kf_ind += 1
        path.append(x_t)
        meas.append(y_t)
        x_tminus1 = x_t
    path = np.squeeze(np.asarray(path))
    meas = np.squeeze(np.asarray(meas))
    # drawer(meas, path)
    for i in range(len(kf_change_pnt) + 1):
        print('F_' + str(i) + ': ')
        print(F[i])
        print('H_' + str(i) + ': ')
        print(H[i])
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
        print('-----------------------------------')
    print('=====================================================')
    if output_measurements is not None:
        meas_df = pd.DataFrame(meas)
        meas_df.to_csv(output_measurements, index=False, header=False)
    if output_groundtruth is not None:
        truth_df = pd.DataFrame(path)
        truth_df.to_csv(output_groundtruth, index=False, header=False)

    return meas


def create_path_random_walk_multi_model(output_measurements='data/measurement3.csv',
                                        output_groundtruth='data/groundtruth3.csv',
                                        q: list = [2.0, 10.0],
                                        r: list = [1.0, 0.8],
                                        state_dim=2,
                                        t=500,
                                        change_pnt=[300]):
    # create constant velocity model
    num_of_models = len(q)
    random_walk_list = []
    for i in range(num_of_models):
        random_walk_list.append(RandomWalk(q=q[i], r=r[i], state_dim=state_dim))
    Q = []
    R = []
    F = []
    H = []
    for i in range(num_of_models):
        Q.append(random_walk_list[i].Q)
        R.append(random_walk_list[i].R)
        F.append(random_walk_list[i].A)
        H.append(random_walk_list[i].H)
    # Start point
    x_tminus1 = np.asarray([[0.0], [0.0]])
    # model switching controller
    kf_ind = 0
    kf_change_pnt = change_pnt

    path = []
    meas = []
    for i in range(t):
        x_t_ = F[kf_ind] @ x_tminus1
        x_t = np.random.multivariate_normal(np.squeeze(x_t_), Q[kf_ind])
        x_t = x_t.reshape((2, 1))
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
    # drawer(meas, path)
    for i in range(len(kf_change_pnt) + 1):
        print('F_' + str(i) + ': ')
        print(F[i])
        print('H_' + str(i) + ': ')
        print(H[i])
        print('Q_' + str(i) + ': ')
        print(Q[i])
        print('R_' + str(i) + ': ')
        print(R[i])
        print('-----------------------------------')
    print('=====================================================')
    if output_measurements is not None:
        meas_df = pd.DataFrame(meas)
        meas_df.to_csv(output_measurements, index=False, header=False)
    if output_groundtruth is not None:
        truth_df = pd.DataFrame(path)
        truth_df.to_csv(output_groundtruth, index=False, header=False)

    return meas


if __name__ == '__main__':
    create_path_constant_volocity_one_model()
    create_path_constant_volocity_multi_model()
    create_path_random_walk_multi_model()

