# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import numpy as np
import pandas as pd
import time
from tqdm import *
from pykrige.ok import OrdinaryKriging
from plot_func import map_plot
from Interpolation import geoInterpolation


# general set
home = "H:/research/flash_drough/code/VIC/Mete_param"
data_home = "G:/data_zxd/水文气象/中国国家级地面气象站基本气象要素日值数据集(V3.0)/SURF_CLI_CHN_MUL_DAY_V3.0/datasets"
PRE_path = os.path.join(data_home, "PRE")
TEM_path = os.path.join(data_home, "TEM")
WIN_path = os.path.join(data_home, "WIN")
coord_path = os.path.join("H:/research/flash_drough/code/VIC/coord.txt")

# read meteStations_LP_Data_postSelect and coord
meteStations_LP_Data_postSelect = pd.read_excel(os.path.join(home, "meteStations_LP_Data_postSelect.xlsx"),
                                                sheet_name=1)
coord = pd.read_csv(coord_path, sep=",")


# search stations in LP and data stations
def SearchStationsInLPandData():
    """ Search Stations In LP and Data stations
    output meteStations_LP_Data
    """
    # path
    meteStations_LP_path = os.path.join(home, "meteStations_LP.xlsx")
    meteStations_China_path = os.path.join(home, "SURF_CLI_CHN_MUL_DAY_STATION.xls")

    # read
    meteStations_LP = pd.read_excel(meteStations_LP_path)
    meteStations_China = pd.read_excel(meteStations_China_path)
    meteStations_China = meteStations_China.set_index("区站号")

    # search station in meteStations_LP and meteStations_China (data stations)
    LP_stationid = meteStations_LP.loc[:, "stationid"].values
    meteStations_China_LP = meteStations_China.loc[LP_stationid, :]
    meteStations_China_LP["stationid"] = meteStations_China_LP.index
    meteStations_China_LP = meteStations_China_LP.merge(meteStations_LP, on="stationid")

    # save
    meteStations_China_LP.to_excel("meteStations_LP_Data.xlsx")

# SearchStationsInLPandData()


# combine files into one file (given station and file list)
def CombineFiles(file_list, stationid, data_column, date_column, save="out.xlsx"):
    """ Combine files into one file
    file form: txt file
    such as pre:
        stationid               date_column=4          data_column=8
        50527 4913 11945   6766 1951  1  1      0      0      0 0 0 0

        序号	中文名	数据类型	单位
        1	区站号	Number(5)
        2	纬度	Number(5)	（度、分）
        3	经度	Number(6)	（度、分）
        4	观测场拔海高度	Number(7)	0.1米
        5	年	Number(5)	年
        6	月	Number(3)	月
        7	日	Number(3)	日
        8	20-8时降水量	Number(7)	0.1mm
        9	8-20时降水量	Number(7)	0.1mm
        10	20-20时累计降水量	Number(7)	0.1mm
        11	20-8时降水量质量控制码	Number(2)
        12	8-20时累计降水量质量控制码	Number(2)
        13	20-20时降水量质量控制码	Number(2)

    input:
        file_list: list of str, contains the path of files to combine
        stationid: iterator, all stations in stationid will be extract
        data_column: int, the column of data in files
        date_column: int, the start column of date in files, date_column(year) - +1(month) - +2(day)
        save: save path

    output:
        save.xlsx
                station1 station2 ... stationN (in stationsid)
        date1
        date2
        ...
        daten
    """
    # loop for each file
    for i in tqdm(range(len(file_list))):
        p = file_list[i]
        p_array = np.loadtxt(p)

        # loop for each station in this file
        for id in stationid:
            p_array_ = p_array[p_array[:, 0] == id]
            date_ = [f"{int(p_array_[j, date_column])}-{int(p_array_[j, date_column + 1])}-{int(p_array_[j, date_column + 2])}"
                     for j in range(p_array_.shape[0])]
            p_df_ = pd.DataFrame(p_array_[:, data_column], index=date_, columns=[f"{id}"])

            # combine all stations in this file
            if id == stationid[0]:
                p_df = p_df_
            else:
                p_df = p_df.join(p_df_)

        # combine all file output
        if i == 0:
            df_ret = p_df
        else:
            df_ret = pd.concat([df_ret, p_df], axis=0)

    # save
    df_ret.to_csv(save, sep=" ")


def combineFilesLP():
    """ combine files in the LP
    output: pre_LP_{start_date}_{end_date}.csv
    """
    # extract date: 1983-1998
    start_date = 198301
    end_date = 199812
    PRE_path_list = [os.path.join(PRE_path, p) for p in os.listdir(PRE_path) if p.endswith("TXT")]
    TEM_path_list = [os.path.join(TEM_path, p) for p in os.listdir(TEM_path) if p.endswith("TXT")]
    WIN_path_list = [os.path.join(WIN_path, p) for p in os.listdir(WIN_path) if p.endswith("TXT")]

    PRE_path_list = [p for p in PRE_path_list if start_date <= int(p[-10: -4]) <= end_date]
    TEM_path_list = [p for p in TEM_path_list if start_date <= int(p[-10: -4]) <= end_date]
    WIN_path_list = [p for p in WIN_path_list if start_date <= int(p[-10: -4]) <= end_date]

    # read meteStations_LP_Data_postSelect
    meteStations_LP_Data_postSelect = pd.read_excel(os.path.join(home, "meteStations_LP_Data_postSelect.xlsx"),
                                                    sheet_name=1)
    stationid = meteStations_LP_Data_postSelect.loc[:, "stationid"]

    # combine files
    CombineFiles(PRE_path_list, stationid, save=f"pre_LP_{start_date}_{end_date}.csv", data_column=8, date_column=4)
    CombineFiles(TEM_path_list, stationid, save=f"temMax_LP_{start_date}_{end_date}.csv", data_column=8, date_column=4)
    CombineFiles(TEM_path_list, stationid, save=f"temMin_LP_{start_date}_{end_date}.csv", data_column=9, date_column=4)
    CombineFiles(WIN_path_list, stationid, save=f"winSpeed_LP_{start_date}_{end_date}.csv", data_column=7, date_column=4)

# combineFilesLP()


def preprocessingData():
    """ preprocessingData, modify some data into standard form
    output: pre_LP_198301_199812_AfterPreprocessing.csv
    """
    # read combined file
    pre = pd.read_csv(os.path.join(home, "pre_LP_198301_199812.csv"), index_col=0)
    temMax = pd.read_csv(os.path.join(home, "temMax_LP_198301_199812.csv"), index_col=0)
    temMin = pd.read_csv(os.path.join(home, "temMin_LP_198301_199812.csv"), index_col=0)
    winSpeed = pd.read_csv(os.path.join(home, "winSpeed_LP_198301_199812.csv"), index_col=0)

    # preprocessing
    # pre: 32766 ->0, 100000 + x -> x; 32700 -> 0, 32xxx -> xxx, 31xxx -> xxx, 30xxx -> xxx
    pre[pre == 32766] = 0
    pre = pre.applymap(lambda x: x if x < 100000 else x % 100000)

    pre[pre == 32700] = 0
    pre = pre.applymap(lambda x: x if x < 30000 else x % 10000 % 1000)

    # temMax and temMin: 32766 ->0, 100000 + x -> x; 10000 + x -> x; -10000 + x -> x
    temMax[temMax == 32766] = 0
    temMax = temMax.applymap(lambda x: x if x < 100000 else x % 100000)
    temMin[temMin == 32766] = 0
    temMin = temMin.applymap(lambda x: x if x < 100000 else x % 100000)

    temMax = temMax.applymap(lambda x: x if x < 10000 else x % 10000)
    temMax = temMax.applymap(lambda x: x if x > -10000 else x % -10000)
    temMin = temMin.applymap(lambda x: x if x < 10000 else x % 10000)
    temMin = temMin.applymap(lambda x: x if x > -10000 else x % -10000)

    # winSpeed: 32766 ->0, 100000 + x -> x; 1000 + x -> x
    winSpeed[winSpeed == 32766] = 0
    winSpeed = winSpeed.applymap(lambda x: x if x < 100000 else x % 100000)
    winSpeed = winSpeed.applymap(lambda x: x if x < 1000 else x % 1000)

    # save
    pre.to_csv("pre_LP_198301_199812_AfterPreprocessing.csv", sep=" ")
    temMax.to_csv("temMax_LP_198301_199812_AfterPreprocessing.csv", sep=" ")
    temMin.to_csv("temMin_LP_198301_199812_AfterPreprocessing.csv", sep=" ")
    winSpeed.to_csv("winSpeed_LP_198301_199812_AfterPreprocessing.csv", sep=" ")


# preprocessingData()

# interpolation
def interpolation():
    # read data
    pre = pd.read_csv(os.path.join(home, "pre_LP_198301_199812_AfterPreprocessing.csv"), index_col=0)
    temMax = pd.read_csv(os.path.join(home, "temMax_LP_198301_199812_AfterPreprocessing.csv"), index_col=0)
    temMin = pd.read_csv(os.path.join(home, "temMin_LP_198301_199812_AfterPreprocessing.csv"), index_col=0)
    winSpeed = pd.read_csv(os.path.join(home, "winSpeed_LP_198301_199812_AfterPreprocessing.csv"), index_col=0)

    # set date
    date = pd.date_range("19830101", "19981231", freq="D")

    # create lat/lon of grid for interpolating box
    grid_lon = np.linspace(min(coord.lon), max(coord.lon), int((max(coord.lon) - min(coord.lon)) / 0.25) + 1)
    grid_lat = np.linspace(min(coord.lat), max(coord.lat), int((max(coord.lat) - min(coord.lat)) / 0.25) + 1)
    np.save("raster_grid_lon", grid_lon)
    np.save("raster_grid_lat", grid_lat)

    # station lat lon based on meteStations_LP_Data_postSelect
    stationid = pre.columns.values
    station_lon = np.array([meteStations_LP_Data_postSelect[meteStations_LP_Data_postSelect["stationid"] == int(id)].Lon.values[0] for id in stationid])
    station_lat = np.array([meteStations_LP_Data_postSelect[meteStations_LP_Data_postSelect["stationid"] == int(id)].Lat.values[0] for id in stationid])

    # pre set
    dataset = [pre.values, temMax.values, temMin.values, winSpeed.values]
    cblabel = ["pre", "temMax", "temMin", "winSpeed"]
    save_path = ["pre_LP_198301_199812_coord_df.csv", "temMax_LP_198301_199812_coord_df.csv",
                 "temMin_LP_198301_199812_coord_df.csv", "winSpeed_LP_198301_199812_coord_df.csv"]

    # interpolating using pykridge "spherical" variogram
    pki = geoInterpolation.PykrigeInterpolation()
    dir_kridge = os.path.join(home, "pykridge_spherical")

    if not os.path.exists(dir_kridge):
        os.mkdir(dir_kridge)  # create dir to save
    save_path_kridge = [os.path.join(dir_kridge, path) for path in save_path]

    for i in range(4):
        dst_data_points, dst_data_grid, dst_coord, sameData = pki(coord.lat.values, coord.lon.values,
                                                                  station_lat, station_lon, dataset[i],
                                                                  variogram_model="spherical",
                                                                  okparam=None, style="points", det=0.25,
                                                                  save_grid_lat_lon=False, plot=[10],
                                                                  cblabel=cblabel[i], src_label=stationid)
        df_ = pd.DataFrame(dst_data_points, index=date, columns=coord.index)
        df_.to_csv(save_path_kridge[i], sep=" ")

    # interpolating using IDW with search_number = 5
    idwi = geoInterpolation.IDWInterpolation()
    dir_IDW = os.path.join(home, "IDW_5")

    if not os.path.exists(dir_IDW):
        os.mkdir(dir_IDW)  # create dir to save
    save_path_IDW = [os.path.join(dir_IDW, path) for path in save_path]

    for i in range(4):
        dst_data_points, dst_data_grid, dst_coord_gridbox = idwi(coord.lat.values, coord.lon.values, station_lat,
                                                                 station_lon, dataset[i], power=2,
                                                                 search_number=5, style="points", det=0.25,
                                                                 save_grid_lat_lon=False, plot=[10],
                                                                 cblabel=cblabel[i], src_label=stationid)
        df_ = pd.DataFrame(dst_data_points, index=date, columns=coord.index)
        df_.to_csv(save_path_IDW[i], sep=" ")


# interpolation()

# cal climotological average precipitation for Soil param, Average annual precipitation - mm
# (daily sum) 0.1mm -> (yearly sum) mm -> multi-year average
def Average_annual_precipitation():
    pre = pd.read_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/pre_LP_198301_199812_coord_df.csv", sep=" ",
                      index_col=0)
    pre = pre * 0.1
    pre = pre.set_index(pd.to_datetime(pre.index))
    pre_year = pre.groupby(pre.index.year).sum()
    pre_year_average = pre_year.mean()

    # save
    pre_year_average.to_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/pre_LP_198301_199812_coord_Average_annual"
                            "_precipitation.csv", sep=" ")


# Average_annual_precipitation()


# format
def format_forcing():
    # read data
    pre = pd.read_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/pre_LP_198301_199812_coord_df.csv", sep=" ",
                      index_col=0)
    temMax = pd.read_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/temMax_LP_198301_199812_coord_df.csv", sep=" ",
                      index_col=0)
    temMin = pd.read_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/temMin_LP_198301_199812_coord_df.csv", sep=" ",
                      index_col=0)
    winSpeed = pd.read_csv("H:/research/flash_drough/code/VIC/Mete_param/IDW_5/winSpeed_LP_198301_199812_coord_df.csv",
                           sep=" ", index_col=0)

    # unit format
    '''
    pre: (daily sum) 0.1mm -> (daily sum) mm - PREC: Total precipitation (rain and snow), factor: 0.1
    temMax/temMin: (daily extremes) 0.1 C -> (daily extremes) C - TMAX/MIN: Maxi/Minimum daily temperature, factor: 0.1
    winSpeed: (daily mean) 0.1 m/s -> (daily mean) m/s - WIND: Wind speed, factor: 0.1
    '''
    pre = pre * 0.1 * 0.8
    temMax = temMax * 0.1 * 1.2
    temMin = temMin * 0.1 * 1.2
    winSpeed = winSpeed * 0.1

    # VIC format
    # loop for each grid to create forcing data
    prefix = "forcing_"

    dir_forcing = os.path.join(home, "forcing_p_sub_20%_t_add_20%")
    if not os.path.exists(dir_forcing):
        os.mkdir(dir_forcing)  # create dir to save

    for i in tqdm(coord.index, colour="green"):
        lat = coord.loc[i, "lat"]
        lon = coord.loc[i, "lon"]
        file_path = os.path.join(dir_forcing, f"{prefix}{lat:.4f}_{lon:.4f}")

        pre_ = pre.loc[:, str(i)]
        temMax_ = temMax.loc[:, str(i)]
        temMin_ = temMin.loc[:, str(i)]
        winSpeed_ = winSpeed.loc[:, str(i)]

        frames = [pre_, temMax_, temMin_, winSpeed_]

        ret = pd.concat(frames, axis=1)
        ret.columns = ["pre", "temMax", "temMin", "winSpeed"]
        ret_array = ret.values

        np.savetxt(file_path, ret_array, fmt="%.2f")


format_forcing()
