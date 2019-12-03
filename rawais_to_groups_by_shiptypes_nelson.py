import pandas as pd
import copy
import numpy as np

"""This script processes the raw ais data. It does a few simple filter, so short tracks are removed.
 The outputs are some csv-files and each one contains useful ais data of one type."""

# The path of raw-ais data
data = pd.read_csv("raw-AIS/Nelson_Fused_AIS.csv", low_memory=False)

# The path of the output csv-files
output_path = "processed-ais/nelson/"

# We only keep the ship tracks with consistent MMSI, LRIMO, Name, Type.
# We now perform a strict restriction: a track must have more than 120 data points..
data_groups = pd.DataFrame.groupby(data, "mmsi")
ShipTypeSet = set()
MMSI_vs_Type = {}
for idx, group_obj in enumerate(data_groups):
    this_MMSI = group_obj[0]
    ShipName = str(group_obj[1].shipName.tolist()[0])
    ShipTypeRaw = str(group_obj[1].shipTypeText.tolist()[0])
    if (type(this_MMSI) == float) and (len(group_obj[1]) >= 1) and (ShipTypeRaw != "nan"):
        #ShipTypeSet.add(this_ShipType)
        ShipType = ShipTypeRaw.split(",")[0]
        ShipTypeSet.add(ShipType)
        MMSI_vs_Type[this_MMSI] = ShipType
print("Get valid group index and generated ship type set...")

# Count the number of ships in each type
ShipTypeCount = {}
dataframe_idx_groupedby_type = {}
for thisShipType in ShipTypeSet:
    ShipTypeCount[thisShipType] = 0
    dataframe_idx_groupedby_type[thisShipType] = []
for this_mmsi, this_type in MMSI_vs_Type.items():
    ShipTypeCount[this_type] += 1
    if len(data_groups.get_group(this_mmsi).index) > 1:
        dataframe_idx_groupedby_type[this_type].extend(data_groups.indices[this_mmsi])
    else:
        dataframe_idx_groupedby_type[this_type].append(data_groups.indices[this_mmsi])
print("Get dataframe idx by type...")

# We generate csv-files of the following types: Cargo: 468 ships // Other Type: 8062 ships // Dredging or underwater ops: 3974 ships
print("Starts to write files")
ship_types = dataframe_idx_groupedby_type.keys()
for this_ship_type in ship_types:
    if this_ship_type == "Cargo" or this_ship_type == "Fishing" or \
            this_ship_type == "Passenger" or this_ship_type == "Pleasure Craft" or \
            this_ship_type == "Tug" or this_ship_type == "Sailing":
        filename = copy.deepcopy(this_ship_type)
        # The csv-file includes MMSI, ship name, combined ship types, raw ship types, raw additional info, Lat, Lon, Time
        this_dataframe = data.iloc[dataframe_idx_groupedby_type[this_ship_type], [0, 1, 4, 5, 16, 17, 18]]
        this_dataframe.insert(6, "ShipType", filename)
        this_dataframe.to_csv(output_path + filename + ".csv", index=False, float_format='%.8f')
        print(filename + " finished...")
