import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from bayou.datastructures import Gaussian, GMM, GMMSequence
from bayou.models import LinearModel
from bayou.filters.skf import GPB2 as GPB2f
from bayou.smoothers.skf import GPB2 as GPB2s


file_list = glob.glob(r'\dataset\test\fishing\*.csv')

N_MODELS = 2

m1a = LinearModel(A=np.eye(2), Q=np.eye(2), H=np.eye(2), R=0.0001 * np.eye(2))
m1a.Q = np.loadtxt('f-Q1.csv', delimiter=',')
m2a = LinearModel(A=np.eye(2), Q=np.eye(2), H=np.eye(2), R=0.0001 * np.eye(2))
m2a.Q = np.loadtxt('f-Q2.csv', delimiter=',')
skf_model_1 = [m1a, m2a]

m1b = LinearModel(A=np.eye(2), Q=np.eye(2), H=np.eye(2), R=0.0001 * np.eye(2))
m1b.Q = np.loadtxt('c-Q1.csv', delimiter=',')
m2b = LinearModel(A=np.eye(2), Q=np.eye(2), H=np.eye(2), R=0.0001 * np.eye(2))
m2b.Q = np.loadtxt('c-Q2.csv', delimiter=',')
skf_model_2 = [m1b, m2b]

models = [skf_model_1, skf_model_2]

Za = np.loadtxt('f-Z.csv', delimiter=',')
Zb = np.loadtxt('c-Z.csv', delimiter=',')
Z = [Za, Zb]

init_gaussian = Gaussian(np.zeros([2,1]), 0.001 * np.eye(2))
initial_state = GMM([init_gaussian, init_gaussian])

predictions = []

for file in tqdm(file_list):
    data = np.loadtxt(file, delimiter=',')
    measurements = np.expand_dims(data, axis=-1)

    L = []
    for n in range(N_MODELS):
        gmmsequence = GMMSequence(measurements, initial_state)
        gmmsequence = GPB2f.filter_sequence(gmmsequence, models[n], Z[n])
        L.append(np.sum(gmmsequence.measurement_likelihood))
    predictions.append(np.argmax(L))
