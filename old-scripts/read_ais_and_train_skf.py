import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from datetime import timedelta
from bayou.datastructures import Gaussian, GMM, GMMSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.skf import SKF

# Learn fishing vessels
#ShipTypes = ["Cargo", "Cargo-Hazardous", "Passenger", "Tanker", "Tanker-Hazardous", "Tug",
#             "Vessel-DredgingorUnderwater Ops", "Vessel-Fishing", "Vessel-Pleasure Craft"]
ShipTypes = ["Cargo", "Passenger", "Tanker", "Tug", "Vessel-Fishing"]

for this_ShipType in ShipTypes:
    # Read and process data
    data_ais = pd.read_csv("D:/AIS-data/group_by_type/"+ this_ShipType +".csv")
    ais_groupby_mmsi = pd.DataFrame.groupby(data_ais, "MMSI")
    seperated_fish_tracks = []
    for this_boat in ais_groupby_mmsi:
        this_mmsi = this_boat[0]
        this_time = list(this_boat[1]["Time"])
        this_lat = list(this_boat[1]["Latitude"])
        this_lon = list(this_boat[1]["Longitude"])
        this_continuous_track_idx = []
        prev_time = []
        prev_lat = []
        prev_lon = []
        for i, this_row_time in enumerate(this_time):
            proc_time = datetime.strptime(this_row_time, "%Y-%m-%dT%H:%M:%SZ")
            if prev_time == []:
                prev_time = proc_time
                prev_lat = this_lat[i]
                prev_lon = this_lon[i]
                this_continuous_track_idx.append(i)
            else:
                if (this_lat[i] < 90) & (this_lat[i] > -90):
                    if ((proc_time - prev_time) < timedelta(hours=1.5)) & (np.abs(prev_lat-this_lat[i]) <= 1.0) &\
                            (np.abs(prev_lon-this_lon[i]) <= 1.0):
                        this_continuous_track_idx.append(i)
                    else:
                        if len(this_continuous_track_idx) >= 80:
                            this_continuous_track = {}
                            this_continuous_track["mmsi"] = this_mmsi
                            this_continuous_track["lat"] = [this_lat[k] for k in this_continuous_track_idx]
                            this_continuous_track["lon"] = [this_lon[k] for k in this_continuous_track_idx]
                            seperated_fish_tracks.append(this_continuous_track)
                            this_continuous_track_idx = []
                        else:
                            this_continuous_track_idx = [i]
                    prev_time = proc_time
                    prev_lat = this_lat[i]
                    prev_lon = this_lon[i]
                else:
                    if len(this_continuous_track_idx) >= 80:
                        this_continuous_track = {}
                        this_continuous_track["mmsi"] = this_mmsi
                        this_continuous_track["lat"] = [this_lat[k] for k in this_continuous_track_idx]
                        this_continuous_track["lon"] = [this_lon[k] for k in this_continuous_track_idx]
                        seperated_fish_tracks.append(this_continuous_track)
                        this_continuous_track_idx = []
                    else:
                        this_continuous_track_idx = []
                    prev_time = []
                    prev_lat = []
                    prev_lon = []

        if len(this_continuous_track_idx) >= 80:
            this_continuous_track = {}
            this_continuous_track["mmsi"] = this_mmsi
            this_continuous_track["lat"] = [this_lat[k] for k in this_continuous_track_idx]
            this_continuous_track["lon"] = [this_lon[k] for k in this_continuous_track_idx]
            seperated_fish_tracks.append(this_continuous_track)
            this_continuous_track_idx = []
    print("Reading data finished ...")

    # Train GPB2
    # Train a CV model for Latitude
    # prepare data and initial model
    dataset_Lat = []
    gmmsequences_list_lat = []
    for i, this_track in enumerate(seperated_fish_tracks):
        this_track_array = np.asarray(this_track["lat"]) * 100
        g1 = Gaussian(np.array([[this_track_array[0]], [0]]), 10 * np.eye(2))
        g2 = Gaussian(np.array([[this_track_array[0]]]), 10 * np.eye(1))
        #g1 = Gaussian(np.ones([2, 1]), 10 * np.eye(2))
        #g2 = Gaussian(np.ones([1, 1]), 10 * np.eye(1))
        initial_gmm_state = GMM([g1, g2])
        gmmsequence_lat = GMMSequence(np.expand_dims(this_track_array, axis=-1), initial_gmm_state)
        gmmsequences_list_lat.append(gmmsequence_lat)
        if i > 1000:
            break
    gmmsequences_list_lon = []
    for i, this_track in enumerate(seperated_fish_tracks):
        this_track_array = np.asarray(this_track["lon"]) * 100
        g1 = Gaussian(np.array([[this_track_array[0]], [0]]), 10 * np.eye(2))
        g2 = Gaussian(np.array([[this_track_array[0]]]), 10 * np.eye(1))
        #g1 = Gaussian(np.ones([2, 1]), 10 * np.eye(2))
        #g2 = Gaussian(np.ones([1, 1]), 10 * np.eye(1))
        initial_gmm_state = GMM([g1, g2])
        gmmsequence_lon = GMMSequence(np.expand_dims(np.asarray(this_track["lon"]), axis=-1), initial_gmm_state)
        gmmsequences_list_lon.append(gmmsequence_lon)
        if i > 1000:
            break

    # initialise model
    # Linear Model
    #m1 = LinearModel(np.eye(2), np.eye(2), np.eye(2)[:1], np.eye(1))  #A, Q, H, R
    # Constant Velocity + Random Walk
    m1 = ConstantVelocity(dt=1.0, q=1.0, r=0.1, state_dim=2, obs_dim=1)
    m2 = LinearModel(np.eye(1), np.eye(1), np.eye(1), 0.1*np.eye(1)) #A  Q  H  R
    initial_models = [m1, m2]
    Z = np.ones([2, 2]) / 2

    print("Preparing data and initialise models finished... Only 2000 tracks are used for the following training ... ")
    # train latitude
    new_models_lat, Z, dataset, LL = SKF.EM(gmmsequences_list_lat, initial_models, Z,
                                            max_iters=10, threshold=0.0001, learn_H=False, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=True, learn_Z=True,
                                            keep_Q_structure=False, diagonal_Q=False, wishart_prior=False)

    print(this_ShipType + " -- Latitude:")
    print("Model 1:")
    print("-------A--------")
    print(new_models_lat[0].A)
    print("-------H--------")
    print(new_models_lat[0].H)
    print("-------Q--------")
    print(new_models_lat[0].Q)
    print("-------R--------")
    print(new_models_lat[0].R)

    print("Model 2: ")
    print("-------A--------")
    print(new_models_lat[1].A)
    print("-------H--------")
    print(new_models_lat[1].H)
    print("-------Q--------")
    print(new_models_lat[1].Q)
    print("-------R--------")
    print(new_models_lat[1].R)
    print("======================================")

    # train latitude
    new_models_lon, Z, dataset, LL = SKF.EM(gmmsequences_list_lon, initial_models, Z,
                                            max_iters=10, threshold=0.0001, learn_H=False, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=True, learn_Z=True,
                                            keep_Q_structure=False, diagonal_Q=False, wishart_prior=False)

    print(this_ShipType + " -- Longitude:")
    print("Model 1:")
    print("-------A--------")
    print(new_models_lon[0].A)
    print("-------H--------")
    print(new_models_lon[0].H)
    print("-------Q--------")
    print(new_models_lon[0].Q)
    print("-------R--------")
    print(new_models_lon[0].R)

    print("Model 2: ")
    print("-------A--------")
    print(new_models_lon[1].A)
    print("-------H--------")
    print(new_models_lon[1].H)
    print("-------Q--------")
    print(new_models_lon[1].Q)
    print("-------R--------")
    print(new_models_lon[1].R)
    print("======================================")
    Dynamics = {}
    Dynamics["Latitude"] = new_models_lat
    Dynamics["Longitude"] = new_models_lon

    fid = open("models/" + this_ShipType + "-skf.data", "wb")
    pickle.dump(Dynamics, fid)
    fid.close()


