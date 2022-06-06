# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import os
import numpy as np
from tqdm import *

# deal with ISMN data
# Data availability 	from 1981-01-08 to 1999-12-28
# STATION: ["GUYUAN", "HUANXIAN", "LUSHI", "NANYANG", "TIANSHUI", "TONGWEI", "XIFENGZH", "XINXIAN", "YONGNING",
# "ZHUMADIA"], note that dont contains XILINGUO(only have layer with 0 - 0.5m)
# Depths of soil moisture measurements
# 0.00 - 0.05 m 0
# 0.05 - 0.10 m 1
# 0.10 - 0.20 m 2
# 0.20 - 0.30 m 3
# 0.30 - 0.40 m 4
# 0.40 - 0.50 m 5
# 0.50 - 0.60 m 6
# 0.60 - 0.70 m 7
# 0.70 - 0.80 m 8
# 0.80 - 0.90 m 9
# 0.90 - 1.00 m 10

layers = {
    0: {
        "start": 0.00,
        "end": 0.05,
    },
    1: {
        "start": 0.05,
        "end": 0.10,
    },
    2: {
        "start": 0.10,
        "end": 0.20,
    },
    3: {
        "start": 0.20,
        "end": 0.30,
    },
    4: {
        "start": 0.30,
        "end": 0.40,
    },
    5: {
        "start": 0.40,
        "end": 0.50,
    },
    6: {
        "start": 0.50,
        "end": 0.60,
    },
    7: {
        "start": 0.60,
        "end": 0.70,
    },
    8: {
        "start": 0.70,
        "end": 0.80,
    },
    9: {
        "start": 0.80,
        "end": 0.90,
    },
    10: {
        "start": 0.90,
        "end": 1.00,
    },
}


class dataProcessISMN:
    """ dataProcess for International Soil Moisture Network(ISMN)
    such as:
        read txt file to excel, based on time index, time without data set to empty
        combine different layers into one file

     source data: https://ismn.geo.tuwien.ac.at/en/
    """
    def __init__(self, home, stations):
        """ init function
        input:
            home: str, home path that contains stations
            stations_path: list of str, the str of station folders' names
        """
        self.home = home
        self.stations = stations

    def __call__(self, start=None, end=None):
        """ ReadtoExcel.__call__ """
        # read txt to excel
        self.read_txt_to_excel()

        # combine Layers
        if (start is not None) and (end is not None):
            self.combineLayers(start, end)

    def read_txt_to_excel(self):
        """ read txt file to excel
        txt files: each station have multiple sm layers(10)
        excel: m(time, some time do not have sm, set Nan) * n(layers)
        time without data set to empty
        """
        # general set
        stations = self.stations

        # set date index, the period of source data between 1981-01-08 to 1999-12-28
        years = list(range(1981, 2000))
        months = list(range(1, 13))
        days = [8, 18, 28]
        pd_index = [f"{year}/{month}/{day}" for year in years for month in months for day in days]  # time index
        pd_index = pd.to_datetime(pd_index)

        # loop for reading to excel
        self.ret_df = {}
        for station in tqdm(stations, colour="green", desc=f"Read to excel for {len(stations)} stations"):
            # fill_value = np.NAN, then put txt sm(the time have sm) into Dataframe
            result = pd.DataFrame(np.full((len(pd_index), 11), fill_value=np.NAN), index=pd_index)
            stms = [os.path.join(home, station, d) for d in os.listdir(os.path.join(home, station)) if d[-4:] == ".stm"]
            for i in range(len(stms)):
                with open(stms[i]) as f:
                    str_ = f.read()
                str_ = str_.splitlines()
                index_ = pd.to_datetime([i_[:10] for i_ in str_[2:]])
                data_ = pd.Series([float(i_[19:25]) for i_ in str_[2:]], index=index_)
                for j in range(len(data_)):
                    result.loc[data_.index[j], i] = data_.loc[data_.index[j]]
            result.to_excel(f"{station}.xlsx")
            self.ret_df[station] = result

    def combineLayers(self, start, end, station_excel_home=None, out_home="./"):
        """ Combine different layers into one file
        input:
            start/end: list of float, control the layers to combine
                start = [0.1, 0.6]
                end = [0.3, 0.8]
                2 layers: 0.1 - 0.3, 0.6 - 0.8
            station_excel_home: str, station_excel home path, it contains files such as {station}.xlsx
        """
        # general set
        stations = self.stations

        # read file -> ret_df
        if station_excel_home is None:
            try:
                ret_df = self.ret_df
            except ValueError("you should input station_excel_home or read_txt_to_excel first"):
                return
        else:
            ret_df = {}
            for station in stations:
                ret_df[station] = pd.read_excel(os.path.join(station_excel_home, f"{station}.xlsx"), index_col=0)

        # cal index
        layer_start = [layers[i]["start"] for i in range(len(layers))]
        layer_end = [layers[i]["end"] for i in range(len(layers))]
        index_start = []
        index_end = []
        for i in range(len(start)):
            if start[i] not in layer_start:
                raise ValueError("start should in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]")

            index_start.append(layer_start.index(start[i]))

            if end[i] not in layer_end:
                raise ValueError("end should in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]")

            index_end.append(layer_end.index(end[i]))

        # loop for combineLayers
        ret_combineLayers = {}
        for station in tqdm(stations, colour="green", desc=f"Combine Layers for {len(stations)} stations"):
            ret_df_ = ret_df[station]

            # loop for each layer: index_start[i]-index_end[i]
            ret_combine = pd.DataFrame(index=ret_df_.index)
            for i in range(len(index_start)):
                ret_combine.loc[:, f"{start[i]}_{end[i]}"] = ret_df_.iloc[:, index_start[i]: index_end[i] + 1].values.sum(axis=1)
                # unit: dm, it can combine by sum()

            # save and output
            ret_combineLayers[station] = ret_combine
            ret_combine.to_excel(os.path.join(out_home, f"{station}_combineLayers.xlsx"))

        return ret_combineLayers


if __name__ == "__main__":
    home = "G:/data_zxd/SM/SM_ISMN/CHINA"
    stations = ["GUYUAN", "HUANXIAN", "LUSHI", "NANYANG", "TIANSHUI", "TONGWEI", "XIFENGZH", "XINXIAN", "YONGNING", "ZHUMADIA"]
    rte = dataProcessISMN(home, stations)
    # rte.read_txt_to_excel()
    # rte.combineLayers(start=[0, 0.1, 0], end=[0.1, 0.6, 0.6], station_excel_home="G:/data_zxd/SM/SM_ISMN/CHINA/1.Station_xlsx")
    rte.combineLayers(start=[0], end=[0.6], station_excel_home="G:/data_zxd/SM/SM_ISMN/CHINA/1.Station_xlsx")