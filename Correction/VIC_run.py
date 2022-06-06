# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Soil_param.Soil_param_Preparation import Soil_param_Preparation
import spotpy
import sys
import time
import re
from Calibration.Calibration_ISMN import ISMN_SoilData
from scipy.stats import pearsonr


class vic_model:

    def __init__(self):
        pass

    def get_obs(self, start="19840101", end="19981231"):
        """
        get observation
        input:
            start/end: str, define the period to get from obs which used for calibrating
        """
        print("get_obs")

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        observation_path = os.path.join(__location__, "Calibration/Calibration_GLDAS/", f"SoilMoi0_100cm_inst_{start}_{end}_D.npy")

        observation = np.load(observation_path)
        observation = observation[:, 1:]

        # reshape
        # print(observation.shape)
        # observation = observation.reshape((-1, ))  # row first, col second
        # observation = observation.mean(axis=1)

        return observation

    def get_simulation(self, start="19840101", end="19981231"):
        """
        get simulation
        input:
            start/end: str, define the period to get from simulation which used for calibrating
        """
        print("get_simulation")

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # path set
        coord_path = os.path.join(__location__, "coord.txt")
        result_path = os.path.join(__location__, "result")

        # read coord
        coord = pd.read_csv(coord_path, sep=",")

        # loop for each stations
        for i in coord.index:
            coord_lat = coord.loc[i, "lat"]
            coord_lon = coord.loc[i, "lon"]
            filename = f"soil_{coord_lat:.4f}_{coord_lon:.4f}"
            data = pd.read_csv(os.path.join(result_path, filename), sep="\t", header=None)

            # date search
            year = [f"{data.loc[index, 0]}" for index in data.index]
            month = [f"{data.loc[index, 1]}" if data.loc[index, 1] > 10 else f"0{data.loc[index, 1]}" for index in data.index]
            day = [f"{data.loc[index, 2]}" if data.loc[index, 2] > 10 else f"0{data.loc[index, 2]}" for index in data.index]
            date = [year[i] + month[i] + day[i] for i in range(len(year))]
            start_index = date.index(start)
            end_index = date.index(end)

            # data extract and combine
            layer1_soil_moisture = data.iloc[start_index: end_index + 1, 4]
            layer2_soil_moisture = data.iloc[start_index: end_index + 1, 5]
            layer1_2_combine = (layer1_soil_moisture + layer2_soil_moisture).values

            # save
            if i == 0:
                simulation = np.zeros((len(layer1_2_combine), len(coord)))  # dates, grids
                simulation.astype(layer1_2_combine.dtype)

            simulation[:, i] = layer1_2_combine

        # reshape
        # print(simulation.shape)
        # simulation = simulation.reshape((-1, ))  # row first, col second
        # simulation = simulation.mean(axis=1)

        return simulation

    def run_vic(self, binfilt=0.35, Ds=0.7, Dsmax=7, Ws=0.02, depth1=0.1, depth2=0.9, depth3=2.0, start="19830101",
                end="19981231", skip_year="0"):
        """
        run vic model
        input:
            params in vic: float
                binfilt=0.35, Ds=0.7, Dsmax=7, Ws=0.02, depth1=0.1, depth2=0.5, depth3=2.0
            start/end: str, the period to run vic model, note the span up period (ususlly one year)
            skip_year: skip_year in GlobalParameter, to define the year not run out, usually used for the span up period
        """
        print("run_vic")

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        os.chdir(__location__)

        # specify file paths to pass into system commands
        GlobalParameter = os.path.join(__location__, './GlobalParameterFile')

        # modify start/end in globalparameter
        with open(GlobalParameter, "r") as f:
            globalparameter = f.readlines()

        for i in range(len(globalparameter)):
            line = globalparameter[i]
            if line.startswith("STARTYEAR"):
                globalparameter[i] = re.sub(r"\d{4}", start[:4], line)
            elif line.startswith("STARTMONTH"):
                globalparameter[i] = re.sub(r"\d{2}", start[4:6], line)
            elif line.startswith("STARTDAY"):
                globalparameter[i] = re.sub(r"\d{2}", start[6:], line)
            elif line.startswith("ENDYEAR"):
                globalparameter[i] = re.sub(r"\d{4}", end[:4], line)
            elif line.startswith("ENDMONTH"):
                globalparameter[i] = re.sub(r"\d{2}", end[4:6], line)
            elif line.startswith("ENDDAY"):
                globalparameter[i] = re.sub(r"\d{2}", end[6:], line)
            elif line.startswith("SKIPYEAR"):
                globalparameter[i] = re.sub(r"\d", skip_year, line)

        with open(GlobalParameter, "w") as f:
            f.writelines(globalparameter)

        # create soil parameter file with new set of variables
        SP = Soil_param_Preparation()
        SP(binfilt=binfilt, Ds=Ds, Dsmax=Dsmax, Ws=Ws, depth1=depth1, depth2=depth2, depth3=depth3)

        # modify soil parameter to turn off some grids run
        # put the coord_on.txt in src folder: (__location__, "../coord_on.txt")

        # execute the vic model
        os.system('./vicNl -g {0} 2> ./result/vic.log'.format(GlobalParameter))


class EvaIndicators:

    def __init__(self, simulation_data, actual_data):
        """ init function
        input:
            simulation_data: 1D iterator, simulation data
            actual_data: 1D iterator, actual data
            note: len(simulation_data) == len(actual_data)
        """
        self.simulation_data = np.array(simulation_data)
        self.actual_data = np.array(actual_data)

    def RMSE(self):
        """ RMSE """
        rmse = (sum((self.simulation_data - self.actual_data) ** 2) / len(self.simulation_data)) ** 0.5
        return rmse

    def RRMSE(self):
        """ RRMSE """
        rrmse = (sum((self.simulation_data - self.actual_data) ** 2)) ** 0.5 / len(self.simulation_data) / self.actual_data.mean()
        return rrmse

    def R(self, confidence: float = 0.95):
        """ R: Pearson Correlation analysis """
        r, p_value = pearsonr(self.simulation_data, self.actual_data)
        significance = 0
        if p_value < 1 - confidence:
            if r > 0:
                significance = 1
            elif r < 0:
                significance = -1

        return r, p_value, significance

    def Ce(self):
        """ Nash coefficient of efficiency """
        ce = 1 - sum((self.actual_data - self.simulation_data) ** 2) / sum((self.actual_data - self.actual_data.mean()) ** 2)
        return ce

    def Bias(self):
        """ Bias """
        bias = (self.actual_data - self.simulation_data).mean()
        return bias


def runvic():
    # run vic
    run = True

    # general set
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # model set
    start_run = "19830101"
    end_run = "19981231"
    skip_year = "1"
    vic = vic_model()
    params = dict(binfilt=0.349908, Ds=0.309337, Dsmax=10.8751, Ws=0.795941, depth3=1.25963)

    # model run
    if run:
        # check and clean result folder
        result_path = os.path.join(__location__, "./result")
        if os.path.exists(result_path):
            sub_files = os.listdir(result_path)
            if len(sub_files) != 0:
                [os.remove(os.path.join(result_path, file)) for file in sub_files]
        else:
            os.mkdir(result_path)

        vic.run_vic(binfilt=params["binfilt"], Ds=params["Ds"], Dsmax=params["Dsmax"], Ws=params["Ws"],
                    depth3=params["depth3"], start=start_run, end=end_run, skip_year=skip_year)

    # all
    start_all = "19840101"
    end_all = "19981231"
    vic_all = vic.get_simulation(start_all, end_all)
    np.save(f"All_SoilMoi0_100cm_inst_{start_all}_{end_all}_D_vic", vic_all)

    # cali time
    start_cali = "19840101"
    end_cali = "19891231"
    vic_cali = vic.get_simulation(start_cali, end_cali)
    np.save(f"Cali_SoilMoi0_100cm_inst_{start_cali}_{end_cali}_D_vic", vic_cali)

    # vali time
    start_vali = "19900101"
    end_vali = "19951231"
    vic_vali = vic.get_simulation(start_vali, end_vali)
    np.save(f"Vali_SoilMoi0_100cm_inst_{start_vali}_{end_vali}_D_vic", vic_vali)

    # simu time
    start_simu = "19960101"
    end_simu = "19981231"
    vic_simu = vic.get_simulation(start_simu, end_simu)
    np.save(f"Simu_SoilMoi0_100cm_inst_{start_simu}_{end_simu}_D_vic", vic_simu)


if __name__ == "__main__":
    # general set
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # runvic
    runvic()

