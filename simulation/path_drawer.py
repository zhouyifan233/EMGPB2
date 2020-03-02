import matplotlib.pyplot as plt


def draw_path(meas, path):
    plt.figure()
    plt.plot(meas[:, 0], meas[:, 1], 'r-o', label='measurements')
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='True trajectory')
    plt.legend()

