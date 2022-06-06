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


class vic_model():

    def __init__(self):
        pass

    def get_obs(self, stations=None, start="19840101", end="19981231"):
        """
        get observation
        input:
            stations: list of str, define the stations to get from observation
            start/end: str, define the period to get from obs which used for calibrating
        """
        print("get_obs")

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
        stations = ["GUYUAN", "HUANXIAN", "TIANSHUI", "TONGWEI", "XIFENGZH",
                    "YONGNING"] if stations is None else stations
        observation_path = [os.path.join(__location__, "Calibration/ISMN_SMdata", station + "_combineLayers.xlsx")
                            for station in stations]
        for i in range(len(stations)):
            station = stations[i]

            # observation data
            observation_data = pd.read_excel(observation_path[i], index_col=0)
            observation_data.index = pd.to_datetime(observation_data.index)  # set date index
            observation_data.dropna(inplace=True)  # drop Nan
            observation_data = observation_data * 100

            # set date between start - end + 1
            start_index = np.where(observation_data.index >= pd.Timestamp(start))[0][0]
            end_index = np.where(observation_data.index <= pd.Timestamp(end))[0][-1]
            observation_data = observation_data.iloc[start_index: end_index + 1, :]

            # to np array
            observation_data = observation_data.values

            # combine
            if i == 0:
                observation = observation_data.flatten()
            else:
                observation = np.hstack((observation, observation_data.flatten()))

        self.observation = observation

        return observation

    def get_simulation(self, stations=None, start="19840101", end="19981231"):
        """
        get simulation
        input:
            stations: list of str, define the stations to get from simulation (the grid related to stations, based coord)
            start/end: str, define the period to get from simulation which used for calibrating
        """
        print("get_simulation")

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # got simulation
        stations = ["GUYUAN", "HUANXIAN", "TIANSHUI", "TONGWEI", "XIFENGZH",
                    "YONGNING"] if stations is None else stations
        coord_path = [os.path.join(__location__, "Calibration/ISMN_SMdata", station_ + ".txt") for station_ in
                      stations]
        observation_path = [os.path.join(__location__, "Calibration/ISMN_SMdata", station_ + "_combineLayers.xlsx") for
                            station_ in stations]
        result_path = os.path.join(__location__, "result")

        # loop for each stations
        for i in range(len(stations)):
            station_ = stations[i]
            coord_ = pd.read_csv(coord_path[i])

            # observation data
            observation_data = pd.read_excel(observation_path[i], index_col=0)
            observation_data.index = pd.to_datetime(observation_data.index)  # set date index
            observation_data.dropna(inplace=True)  # drop Nan
            observation_data = observation_data

            # set date between start - end + 1
            start_index = np.where(observation_data.index >= pd.Timestamp(start))[0][0]
            end_index = np.where(observation_data.index <= pd.Timestamp(end))[0][-1]
            observation_data = observation_data.iloc[start_index: end_index + 1, :]

            # observation date (not None)
            observation_date = observation_data.index

            # loop for resampling multi-grids to the observation position
            model_data_all_grid = pd.DataFrame(index=observation_data.index, columns=coord_.index)

            # loop for resampling multi-grids to the observation position
            for j in coord_.index:
                coord_lat = coord_.loc[j, "lat"]
                coord_lon = coord_.loc[j, "lon"]
                filename = f"soil_{coord_lat:.4f}_{coord_lon:.4f}"

                data = pd.read_csv(os.path.join(result_path, filename), sep="\t", header=None)
                date_str = [f"{data.iloc[k, 0]}-{data.iloc[k, 1]}-{data.iloc[k, 2]}" for k in range(len(data))]

                data.index = pd.to_datetime(date_str)  # set date index

                # combine first(0-0.1), second(0.1-0.6) layers
                model_data_all_grid.loc[:, j] = data.loc[observation_date, 4] + data.loc[observation_date, 5]

            model_data = model_data_all_grid.mean(axis=1)  # average all multi-grids

            # to np array
            model_data = model_data.values.flatten()

            # combine
            if i == 0:
                simulation = model_data
            else:
                simulation = np.hstack((simulation, model_data))

        return simulation

    def run_vic(self, binfilt=0.35, Ds=0.7, Dsmax=7, Ws=0.02, depth1=0.1, depth2=0.5, depth3=2.0, start="19830101",
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


class spotpy_setup():
    """ spotpy setup class """

    def __init__(self, stations=None, start_run="19830101", end_run="19981231", start_cali="19840101", end_cali="19981231",
                 skip_year="1", depth2=0.1):
        """
        init function
            input:
                stations: list of str, define the stations for calibration
                start/end_run: str, define the period to run vic model
                start/end_cali: str, define the period to calibrate
                skip_year: str, define the year for span up and not output
        """
        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.owd = __location__

        self.stations = stations
        self.start_run = start_run
        self.end_run = end_run
        self.start_cali = start_cali
        self.end_cali = end_cali
        self.skip_year = skip_year
        self.depth2 = depth2

        self.vicmodel = vic_model()
        self.params = [spotpy.parameter.Uniform('binfilt', 0, 0.35),
                       spotpy.parameter.Uniform('Ds', 0., 0.5),
                       spotpy.parameter.Uniform('Dsmax', 5, 20),
                       spotpy.parameter.Uniform('Ws', 0.5, 1.),
                       spotpy.parameter.Uniform('depth3', 0.3, 3.0),
                       ]
        return

    def simulation(self, params):
        """ simulation based on params """
        self.vicmodel.run_vic(binfilt=params[0], Ds=params[1], Dsmax=params[2], Ws=params[3], depth2=self.depth2,
                              depth3=params[4],
                              start=self.start_run, end=self.end_run, skip_year=self.skip_year)
        simulation = self.vicmodel.get_simulation(stations=self.stations, start=self.start_cali, end=self.end_cali)

        return simulation

    def evaluation(self):
        """ evaluation, return observation """
        observation = self.vicmodel.get_obs(stations=self.stations, start=self.start_cali, end=self.end_cali)
        return observation

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = spotpy.objectivefunctions.rmse(evaluation, simulation)
        return objectivefunction

    def parameters(self):
        return spotpy.parameter.generate(self.params)


def findBestSim(dbPath):
    csv = pd.read_csv(dbPath)

    results = np.array(csv)

    likes = np.array(csv.like1)

    idx = np.abs(likes).argmin()  # return the first index of elements in min(likes)

    bestobjf = results[idx, 0] * -1
    binfilt = results[idx, 1]
    Ds = results[idx, 2]
    Dsmax = results[idx, 3]
    Ws = results[idx, 4]
    depth3 = results[idx, 5]

    params = [binfilt, Ds, Dsmax, Ws, depth3]

    return params, bestobjf


def calibrate(parallel_spotpy="seq", plot=False, depth2=0.3):
    """ calibrateparallel_spotpy
    input:
        parallel_spotpy: str, "seq" or "mpi" or "mpc" or "umpc", run spotpy with multi-cores or not,
                        if "mpi" use 'mpirun -np num python VIC_calibration.py repetitions' to calibrate
    """
    # general set
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # check and clean result folder
    result_path = os.path.join(__location__, "./result")
    if os.path.exists(result_path):
        sub_files = os.listdir(result_path)
        if len(sub_files) != 0:
            [os.remove(os.path.join(result_path, file)) for file in sub_files]
    else:
        os.mkdir(result_path)

    # calibration setup object
    stations = ["GUYUAN", "HUANXIAN", "TIANSHUI", "TONGWEI", "XIFENGZH"]
    start_run = "19830101"
    end_run = "19891231"
    start_cali = "19840101"
    end_cali = "19891231"

    # combine the first and second layers
    depth1 = 0.1
    depth2 = depth2
    dp_ISMN = ISMN_SoilData.dataProcessISMN(home="./Calibration/ISMN_SMdata/original_data/", stations=stations)
    dp_ISMN.combineLayers(start=[0], end=[round(depth1 + depth2, 1)],
                          station_excel_home="./Calibration/ISMN_SMdata/original_data/",
                          out_home="./Calibration/ISMN_SMdata/")

    # initialize calibration algorithm with
    cal_setup = spotpy_setup(stations, start_run=start_run, end_run=end_run, start_cali=start_cali, end_cali=end_cali,
                             skip_year="1", depth2=depth2)
    outCal = './result/SCEUA_VIC'
    sampler = spotpy.algorithms.sceua(cal_setup, dbname=outCal, dbformat='csv', parallel=parallel_spotpy)

    # print Optimization_direction, it should compared with objectivefunction
    print("Optimization_direction: ", sampler.optimization_direction)  # minimized

    # run calibration process
    sampler.sample(int(sys.argv[1]), ngs=4)  # input the maximum number of function evaluations allowed (repetitions)

    # plot
    if plot:
        plot_calibration(out_csv=outCal + ".csv", start_cali=start_cali, end_cali=end_cali, stations=stations, depth2=depth2)

    return


def plot_calibration(out_csv, start_cali="19840101", end_cali="19891231", depth2=0.3, stations=None):
    """ plot SCEUA_objectivefunction trace"""

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # plot
    results = pd.read_csv(out_csv)
    fig = plt.figure(1, figsize=(9, 5))
    plt.plot(results.like1)
    plt.ylabel('Calibration Indicator')
    plt.xlabel('Iteration')
    fig.savefig(os.path.join(__location__, 'SCEUA_objectivefunctiontrace.png'), dpi=300)

    # best simulation
    params, bestobjf = findBestSim(out_csv)
    print('Best parameter set: {0}'.format(params))

    # use cali period to out simulation
    lastRun = vic_model()
    print("last Run start")
    lastRun.run_vic(params[0], params[1], params[2], params[3], depth2=depth2, depth3=params[4],
                    start=start_cali, end=end_cali, skip_year="0")
    best_simulation = lastRun.get_simulation(stations=stations, start=start_cali, end=end_cali)
    observation = lastRun.get_obs(stations=stations, start=start_cali, end=end_cali)
    print(best_simulation, observation)

    # plot best simulation
    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
    ax.plot(observation, 'r.', markersize=3, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel('Data')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_best_modelrun.png', dpi=300)


if __name__ == "__main__":
    start = time.time()
    calibrate(parallel_spotpy="seq", plot=True)
    end = time.time()
    print("Elapsed time: ", end - start)
