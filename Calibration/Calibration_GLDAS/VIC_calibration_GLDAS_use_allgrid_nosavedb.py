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
import gc


class vic_model():

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

        observation_path = os.path.join(__location__, "Calibration/Calibration_GLDAS/", f"Cali_SoilMoi0_100cm_inst_{start}_{end}_D.npy")

        observation = np.load(observation_path)
        observation = observation[:, 1:]

        # reshape
        # print(observation.shape)
        # observation = observation.reshape((-1, ))  # row first, col second
        observation = observation.mean(axis=1)

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
        simulation = np.zeros((2192, 1166))  # 2192-dates, 1167-grids
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
                simulation.astype(layer1_2_combine.dtype)

            simulation[:, i] = layer1_2_combine

        # reshape
        # print(simulation.shape)
        # simulation = simulation.reshape((-1, ))  # row first, col second
        simulation = simulation.mean(axis=1)

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


class spotpy_setup():
    """ spotpy setup class """

    def __init__(self, start_run="19830101", end_run="19981231", start_cali="19840101", end_cali="19981231",
                 skip_year="1"):
        """
        init function
            input:
                start/end_run: str, define the period to run vic model
                start/end_cali: str, define the period to calibrate
                skip_year: str, define the year for span up and not output
        """
        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.owd = __location__

        self.start_run = start_run
        self.end_run = end_run
        self.start_cali = start_cali
        self.end_cali = end_cali
        self.skip_year = skip_year

        self.vicmodel = vic_model()
        self.params = [spotpy.parameter.Uniform('binfilt', 0.001, 0.4),
                       spotpy.parameter.Uniform('Ds', 0.001, 1),
                       spotpy.parameter.Uniform('Dsmax', 5, 20),
                       spotpy.parameter.Uniform('Ws', 0.1, 1.),
                       spotpy.parameter.Uniform('depth3', 0.5, 3.0),
                       ]
        return

    def simulation(self, params):
        """ simulation based on params """
        self.vicmodel.run_vic(binfilt=params[0], Ds=params[1], Dsmax=params[2], Ws=params[3],
                              depth3=params[4],
                              start=self.start_run, end=self.end_run, skip_year=self.skip_year)
        simulation = self.vicmodel.get_simulation(start=self.start_cali, end=self.end_cali)

        return simulation

    def evaluation(self):
        """ evaluation, return observation """
        observation = self.vicmodel.get_obs(start=self.start_cali, end=self.end_cali)
        return observation

    def objectivefunction(self, simulation, evaluation):
        # objectivefunction = spotpy.objectivefunctions.rmse(evaluation, simulation)
        # objectivefunction = ((sum((simulation - evaluation) ** 2) / len(simulation)) ** 0.5).mean()
        objectivefunction = -(1 - sum((evaluation - simulation) ** 2) / sum((evaluation - evaluation.mean()) ** 2))
        del evaluation, simulation
        gc.collect()
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


def calibrate(parallel_spotpy="seq", plot=False):
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
    start_run = "19830101"
    end_run = "19891231"
    start_cali = "19840101"
    end_cali = "19891231"
    skip_year = "1"

    # initialize calibration algorithm with
    cal_setup = spotpy_setup(start_run=start_run, end_run=end_run, start_cali=start_cali, end_cali=end_cali,
                             skip_year=skip_year)
    outCal = './result/SCEUA_VIC'
    sampler = spotpy.algorithms.sceua(cal_setup, dbname=outCal, dbformat='csv', parallel=parallel_spotpy, save_sim=False)

    # print Optimization_direction, it should compared with objectivefunction
    print("Optimization_direction: ", sampler.optimization_direction)  # minimized

    # run calibration process
    sampler.sample(int(sys.argv[1]), ngs=4)  # input the maximum number of function evaluations allowed (repetitions)

    # plot
    if plot:
        plot_calibration(out_csv=outCal + ".csv", start_cali=start_cali, end_cali=end_cali, skip_year="0")

    return


def plot_calibration(out_csv, start_cali="19840101", end_cali="19891231", skip_year="0"):
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
    lastRun.run_vic(binfilt=params[0], Ds=params[1], Dsmax=params[2], Ws=params[3], depth3=params[4],
                    start=start_cali, end=end_cali, skip_year=skip_year)
    best_simulation = lastRun.get_simulation(start=start_cali, end=end_cali)
    observation = lastRun.get_obs(start=start_cali, end=end_cali)
    # best_simulation = best_simulation.mean(axis=1)
    # observation = observation.mean(axis=1)
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
