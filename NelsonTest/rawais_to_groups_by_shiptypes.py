import pandas as pd
import copy

"""This script processes the raw ais data. It does a few simple filter, so short tracks are removed.
 The outputs are some csv-files and each one contains useful ais data of one type."""

# The path of raw-ais data
data1 = pd.read_csv("D:/AIS-data/AIS_2017-01-23_2017-01-28.csv")
data2 = pd.read_csv("D:/AIS-data/AIS_2017-01-28_2017-02-02.csv")
data3 = pd.read_csv("D:/AIS-data/AIS_2017-02-02_2017-02-07.csv")
data = pd.concat((data1, data2, data3))

# The path of the output csv-files
output_path = "D:/AIS-data/processed-ais-group_by_type/processed-ais/"

# We only keep the ship tracks with consistent MMSI, LRIMO, Name, Type.
# We now perform a strict restriction: a track must have more than 120 data points..
data_groups = pd.DataFrame.groupby(data, "MMSI")
ShipTypePart1Set = set()
ShipTypePart2Set = set()
ShipTypeSet = set()
MMSI_vs_Type = {}
for idx, group_obj in enumerate(data_groups):
    LRIMOShipNo = set(group_obj[1].LRIMOShipNo.tolist())
    ShipName = set(group_obj[1].ShipName.tolist())
    ShipTypePart1 = set(group_obj[1].ShipType.tolist())
    ShipTypePart2 = set(group_obj[1].AdditionalInfo.tolist())
    if len(LRIMOShipNo) == 1 and len(ShipName) == 1 and len(ShipTypePart1) == 1\
            and len(ShipTypePart2) == 1 and len(group_obj[1]) >= 120:
        this_ShipTypePart1 = ShipTypePart1.pop()
        this_ShipTypePart2 = ShipTypePart2.pop()
        this_ShipType = str(this_ShipTypePart1)+"-"+str(this_ShipTypePart2)
        ShipTypeSet.add(this_ShipType)
        MMSI_vs_Type[group_obj[0]] = this_ShipType
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
        # The following commented lines are for generating test files.
        if len(dataframe_idx_groupedby_type[this_type]) > 3000:
            continue
        dataframe_idx_groupedby_type[this_type].extend(data_groups.indices[this_mmsi])
    else:
        dataframe_idx_groupedby_type[this_type].append(data_groups.indices[this_mmsi])
print("Get dataframe idx by type...")

# Combine Cargo Hazardous A, B, C, D together (into Cargo-Hazardous)
dataframe_idx_groupedby_type["Cargo-Hazardous"] = dataframe_idx_groupedby_type["Cargo-Hazardous Cat. A"]
dataframe_idx_groupedby_type["Cargo-Hazardous"].extend(dataframe_idx_groupedby_type["Cargo-Hazardous Cat. B"])
dataframe_idx_groupedby_type["Cargo-Hazardous"].extend(dataframe_idx_groupedby_type["Cargo-Hazardous Cat. C"])
dataframe_idx_groupedby_type["Cargo-Hazardous"].extend(dataframe_idx_groupedby_type["Cargo-Hazardous Cat. D"])

# Combine Tanker-Hazardous A, B, C, D together (into Tank-Hazardous)
dataframe_idx_groupedby_type["Tanker-Hazardous"] = dataframe_idx_groupedby_type["Tanker-Hazardous Cat. A"]
dataframe_idx_groupedby_type["Tanker-Hazardous"].extend(dataframe_idx_groupedby_type["Tanker-Hazardous Cat. B"])
dataframe_idx_groupedby_type["Tanker-Hazardous"].extend(dataframe_idx_groupedby_type["Tanker-Hazardous Cat. C"])
dataframe_idx_groupedby_type["Tanker-Hazardous"].extend(dataframe_idx_groupedby_type["Tanker-Hazardous Cat. D"])

# We generate csv-files of the following types: Vessel-Dredging/Underwater Ops: 468 ships // Tanker: 8062 ships // Tug: 3974 ships
# Vessel-Fishing 2559 // Passenger 2108 // Vessel-Pleasure Craft 665 // Cargo 21018 //
# Cargo-Hazardous 1420+148+108+246 // Tanker-Hazardous 803+590+257+348
print("Starts to write files")
ship_types = dataframe_idx_groupedby_type.keys()
for this_ship_type in ship_types:
    if this_ship_type == "Vessel-Dredging/Underwater Ops" or this_ship_type == "Tanker-nan" or \
            this_ship_type == "Tug-nan" or this_ship_type == "Vessel-Fishing" or \
            this_ship_type == "Passenger-nan" or this_ship_type == "Vessel-Pleasure Craft" or \
            this_ship_type == "Cargo-nan" or this_ship_type == "Cargo-Hazardous" or \
            this_ship_type == "Tanker-Hazardous":
        filename = copy.deepcopy(this_ship_type)
        filename = filename.replace("-nan", "")
        filename = filename.replace("/", "or")
        # The csv-file includes MMSI, ship name, combined ship types, raw ship types, raw additional info, Lat, Lon, Time
        this_dataframe = data.iloc[dataframe_idx_groupedby_type[this_ship_type], [3, 2, 1, 15, 8, 9, 19]]
        this_dataframe.insert(2, "ShipTypeComplete", filename)
        this_dataframe.to_csv(output_path + filename + ".csv", index=False, float_format='%.8f')
        print(filename + " finished...")
