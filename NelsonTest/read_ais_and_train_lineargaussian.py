import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from datetime import timedelta
from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian

# We have these types in processed-ais folder
#ShipTypes = ["Cargo", "Cargo-Hazardous", "Passenger", "Tanker", "Tanker-Hazardous", "Tug",
#             "Vessel-DredgingorUnderwater Ops", "Vessel-Fishing", "Vessel-Pleasure Craft"]

# BUT, we only learn the dynamic of the following types.
ShipTypes = ["Cargo", "Passenger", "Tanker", "Tug", "Vessel-Fishing"]

# The path of the processed csv-files. The output of "rawais_to_groups_by_shiptypes.py"
input_path = "processed-ais/"

# iterate each type.
for this_ShipType in ShipTypes:
    # Read and pre-process data
    # The basic task is to transform the ais data into lists with only Latitude and Longitude.
    # In order to prevent coordinates jumping, data points of one track should be consecutive (time interval == 1 hour).
    # Thus one ship's track might be separated into several sub-tracks.
    # Any (sub-)track should have more than 80 data points (better for learning, I think).
    data_ais = pd.read_csv(input_path + this_ShipType +".csv")
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
                # remove the invalid latitudes.
                if (this_lat[i] < 90) & (this_lat[i] > -90):
                    # limit the time interval and lat lon changes (I think any ship cannot be faster than 1 lat/lon per hour).
                    if ((proc_time - prev_time) < timedelta(hours=1.5)) & (np.abs(prev_lat-this_lat[i]) <= 1.0) &\
                            (np.abs(prev_lon-this_lon[i]) <= 1.0):
                        this_continuous_track_idx.append(i)
                    else:
                        # the track should be longer than 80 points
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
                    # the track should be longer than 80 points
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
    # seperated_fish_tracks is a list of the dictionary that includes MMSIs, latitudes and longitudes.

    ##################################################################################################
    # Train Gaussian Linear (Single Kalman filter)
    # Latitudes and Longitudes are learnt separately
    ##################################################################################################
    # Train a Constant Velocity (CV) model for Latitudes
    # prepare latitude dataset for learning
    dataset_lat = []
    for i, this_track in enumerate(seperated_fish_tracks):
        this_track_array = np.asarray(this_track["lat"]) * 100
        initial_state = Gaussian(np.array([[this_track_array[0]], [0]]), 10*np.eye(2))
        sequence = GaussianSequence(np.expand_dims(this_track_array, axis=-1), initial_state)
        dataset_lat.append(sequence)
        # We only use the first 3000 tracks. Larger number means longer time.
        # For testing, you can use 500.
        if i > 3000:
            break
    # prepare longitude dataset for learning
    dataset_lon = []
    for i, this_track in enumerate(seperated_fish_tracks):
        this_track_array = np.asarray(this_track["lon"]) * 100
        initial_state = Gaussian(np.array([[this_track_array[0]], [0]]), 10*np.eye(2))
        sequence = GaussianSequence(np.expand_dims(this_track_array, axis=-1), initial_state)
        dataset_lon.append(sequence)
        if i > 3000:
            break

    # initialise CV model
    initial_model = ConstantVelocity(dt=1.0, q=1, r=0.1, state_dim=2, obs_dim=1)

    print("Preparing data and initialise models finished... Only 2000 tracks are used for the following training ... ")
    # train latitude model
    new_models_lat, dataset, LLs = LinearGaussian.EM(dataset_lat, initial_model,
                                            max_iters=20, threshold=0.00001,
                                            learn_H=False, learn_R=True,
                                            learn_A=False, learn_Q=True, learn_init_state=True,
                                            keep_Q_structure=False, diagonal_Q=False)

    print(this_ShipType + " -- Latitude:")
    print("Model :")
    print("-------A--------")
    print(new_models_lat.A)
    print("-------H--------")
    print(new_models_lat.H)
    print("-------Q--------")
    print(new_models_lat.Q)
    print("-------R--------")
    print(new_models_lat.R)
    print("======================================")

    # train longitude model
    new_models_lon, dataset, LLs = LinearGaussian.EM(dataset_lon, initial_model,
                                                     max_iters=20, threshold=0.00001,
                                                     learn_H=False, learn_R=True,
                                                     learn_A=False, learn_Q=True, learn_init_state=True,
                                                     keep_Q_structure=False, diagonal_Q=False)

    print(this_ShipType + " -- Longitude:")
    print("Model:")
    print("-------A--------")
    print(new_models_lon.A)
    print("-------H--------")
    print(new_models_lon.H)
    print("-------Q--------")
    print(new_models_lon.Q)
    print("-------R--------")
    print(new_models_lon.R)

    print("======================================")
    Dynamics = {}
    Dynamics["Latitude"] = new_models_lat
    Dynamics["Longitude"] = new_models_lon

    # save the model in binary
    fid = open("models/" + this_ShipType + ".data", "wb")
    pickle.dump(Dynamics, fid)
    fid.close()

