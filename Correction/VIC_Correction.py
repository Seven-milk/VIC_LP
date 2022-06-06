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
import draw_plot
import statsmodels.api as sm
import fitter
from scipy import stats
from tqdm import *


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


def evaluate_function(vic, obs, date, show=True, save_on=False, save_format="jpg"):
    # cal evaIndicators
    ei = EvaIndicators(vic, obs)
    evaluation = dict(RMSE=ei.RMSE(), RRMSE=ei.RRMSE(), R=ei.R(), Ce=ei.Ce(), Bias=ei.Bias())

    # plot compare series
    fig_compare_series = draw_plot.Figure()
    draw_compare_series = draw_plot.Draw(fig_compare_series.ax, fig_compare_series, legend_on=True, labelx=None,
                          labely="Soil Moisture / mm")
    fig_compare_series.ax.set_xlabel("Date")
    draw_obs = draw_plot.PlotDraw(date, obs, color="black", linestyle="solid", label='Observation', linewidth=0.7)
    draw_vic = draw_plot.PlotDraw(date, vic, 'o', markersize=0.6, label='Simulation', markerfacecolor="r",
                                  markeredgecolor='r')
    draw_compare_series.adddraw(draw_obs, draw_vic)

    # plot compare scatter
    fig_compare_scatter = draw_plot.Figure()
    draw_compare_scatter = draw_plot.Draw(fig_compare_scatter.ax, fig_compare_scatter, labelx=None,
                          labely="Simulation / mm", legend_on=True)
    draw_compare_scatter.ax.set_xlabel("Observation / mm")
    draw_scatter = draw_plot.PlotDraw(obs, vic, 'o', markersize=0.6, markerfacecolor="gray", markeredgecolor="gray")

    # fit line
    data = np.vstack((np.array(obs), np.array(vic)))
    data_sort = data.T[np.lexsort(data[::-1,:])].T
    x = data_sort[0, :]
    y = data_sort[1, :]

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    result = model.fit()
    r2 = result.rsquared
    params = result.params
    k_ = params[1]
    evaluation["R2"] = r2
    evaluation["OLS"] = f"y={params[1]}x + {params[0]}"

    text = "r=%.2f" % evaluation["R"][0]

    x_pre = np.linspace(x[0], x[-1], 100)
    X_pre = sm.add_constant(x_pre)
    pred_ols = result.get_prediction(X_pre)
    pred_mean = pred_ols.summary_frame()["mean"]

    draw_mean = draw_plot.PlotDraw(x_pre, pred_mean, color='k', linewidth=1, zorder=20, label="OLS: " + text)
    draw_compare_scatter.adddraw(draw_scatter, draw_mean)

    # df_out
    df_out = pd.DataFrame(evaluation)

    if show:
        plt.show()

    if save_on:
        if save_format == "jpg":
            fig_compare_series.savejpg(save_on + "_compare_series")
            fig_compare_scatter.savejpg(save_on + "_compare_scatter")
        elif save_format == "svg":
            fig_compare_series.savesvg(save_on + "_compare_series")
            fig_compare_scatter.savesvg(save_on + "_compare_scatter")

        df_out.to_csv(save_on + "evaluate.csv")

    return df_out


def evaluate():
    # general set
    show = False
    save_on = "Before_Correction"
    save_format = "jpg"

    # data
    cali_vic = np.load(os.path.join(__location__, "Cali_SoilMoi0_100cm_inst_19840101_19891231_D_vic.npy"))
    vali_vic = np.load(os.path.join(__location__, "Vali_SoilMoi0_100cm_inst_19900101_19951231_D_vic.npy"))
    simu_vic = np.load(os.path.join(__location__, "Simu_SoilMoi0_100cm_inst_19960101_19981231_D_vic.npy"))

    cali_obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19840101_19891231_D.npy"))
    vali_obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19900101_19951231_D.npy"))
    simu_obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19960101_19981231_D.npy"))

    cali_vic = cali_vic[:, 1:].mean(axis=1)
    vali_vic = vali_vic[:, 1:].mean(axis=1)
    simu_vic = simu_vic[:, 1:].mean(axis=1)

    cali_obs = cali_obs[:, 1:].mean(axis=1)
    vali_obs = vali_obs[:, 1:].mean(axis=1)
    simu_obs = simu_obs[:, 1:].mean(axis=1)

    # date
    date_cali = pd.date_range("19840101", "19891231", freq="D")
    date_vali = pd.date_range("19900101", "19951231", freq="D")
    date_simu = pd.date_range("19960101", "19981231", freq="D")

    # cali
    evaluate_function(cali_vic, cali_obs, date_cali, show=show, save_on=save_on + "cali" if save_on else save_on,
                      save_format=save_format)
    evaluate_function(vali_vic, vali_obs, date_vali, show=show, save_on=save_on + "vali" if save_on else save_on,
                      save_format=save_format)
    evaluate_function(simu_vic, simu_obs, date_simu, show=show, save_on=save_on + "simu" if save_on else save_on,
                      save_format=save_format)


def CDF_Match_correction_function(simu, obs, date):
    # date
    date_month = [d.month for d in date]
    correction_simu = np.zeros(simu.shape)

    # loop for each grid to corrction
    for i in tqdm(range(simu.shape[1]), colour="green", desc="loop for each grid to corrction"):
        simu_grid = simu[:, i]
        obs_grid = obs[:, i]

        # loop for each month
        for j in range(12):
            # extract month data
            month = j + 1
            date_month_index = [m == month for m in date_month]
            simu_grid_month = simu_grid[date_month_index]
            obs_grid_month = obs_grid[date_month_index]

            # fit distribution
            fitter_simu = fitter.Fitter(simu_grid_month, distributions="norm")
            fitter_obs = fitter.Fitter(obs_grid_month, distributions="norm")
            fitter_simu.fit()
            fitter_obs.fit()

            best_simu = fitter_simu.get_best(method="bic")
            best_obs = fitter_obs.get_best(method="bic")

            name_simu = list(best_simu.keys())[0]
            params_simu = best_simu[name_simu]
            name_obs = list(best_obs.keys())[0]
            params_obs = best_obs[name_obs]

            distribution_simu = getattr(stats, name_simu)(*params_simu)
            distribution_obs = getattr(stats, name_obs)(*params_obs)

            cdf_simu = distribution_simu.cdf(simu_grid_month)
            correction_simu_ = distribution_obs.ppf(cdf_simu)

            # nan and inf
            index_nan = np.isnan(correction_simu_)
            index_inf = np.isinf(correction_simu_)

            if sum(index_nan) > 0:
                correction_simu_[index_nan] = simu_grid_month[index_nan]
            if sum(index_inf) > 0:
                correction_simu_[index_inf] = simu_grid_month[index_inf]

            correction_simu[date_month_index, i] = correction_simu_

    return correction_simu


def CDF_Match_correction():
    # data
    obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19480101_20141231_D.npy"))
    simu = np.load(os.path.join(__location__, "All_SoilMoi0_100cm_inst_19840101_19981231_D_vic.npy"))

    # date
    period_simu = [19840101, 19981231]
    date = pd.date_range("19840101", "19981231", freq="D")

    # find index
    obs_date = np.array(obs[:, 0])
    obs_date -= 0.12
    obs_date = obs_date.astype(int)
    period_simu_index = [np.where(obs_date == period_simu[0])[0][0],
                         np.where(obs_date == period_simu[1])[0][0]]

    obs = obs[period_simu_index[0]: period_simu_index[1] + 1, :]
    obs = obs[:, 1:]

    # correction
    correction_simu = CDF_Match_correction_function(simu, obs, date)

    # save
    np.save("Correction_All_SoilMoi0_100cm_inst_19840101_19981231_D_vic", correction_simu)


def Regression_correction_function(simu_base, simu, obs, date):
    # date
    date_month = [d.month for d in date]
    correction_simu = np.zeros(simu.shape)

    # loop for each grid to correction
    for i in tqdm(range(simu.shape[1]), colour="green", desc="loop for each grid to corrction"):
        simu_base_grid = simu_base[:, i]
        simu_grid = simu[:, i]
        obs_grid = obs[:, i]

        # loop for each month
        for j in range(12):
            # extract month data
            month = j + 1
            date_month_index = [m == month for m in date_month]
            simu_base_grid_month = simu_base_grid[date_month_index]
            simu_grid_month = simu_grid[date_month_index]
            obs_grid_month = obs_grid[date_month_index]

            # fit regression: simu base - obs
            data = np.vstack((np.array(simu_base_grid_month), np.array(obs_grid_month)))
            data_sort = data.T[np.lexsort(data[::-1,:])].T
            x = data_sort[0, :]
            y = data_sort[1, :]
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            result = model.fit()

            # predict: simu - fit regression
            X_pre = sm.add_constant(simu_grid_month)
            pred_ols = result.get_prediction(X_pre)
            pred_mean = pred_ols.summary_frame()["mean"]

            # save result
            correction_simu[date_month_index, i] = pred_mean

    return correction_simu


def Regression_correction():
    # data
    obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19480101_20141231_D.npy"))
    simu = np.load(os.path.join(__location__, "All_SoilMoi0_100cm_inst_19840101_19981231_D_vic.npy"))
    simu_base = np.load(os.path.join(__location__, "All_SoilMoi0_100cm_inst_19840101_19981231_D_vic_base.npy"))

    # date
    period_simu = [19840101, 19981231]
    date = pd.date_range("19840101", "19981231", freq="D")

    # find index
    obs_date = np.array(obs[:, 0])
    obs_date -= 0.12
    obs_date = obs_date.astype(int)
    period_simu_index = [np.where(obs_date == period_simu[0])[0][0],
                         np.where(obs_date == period_simu[1])[0][0]]

    obs = obs[period_simu_index[0]: period_simu_index[1] + 1, :]
    obs = obs[:, 1:]

    # correction
    correction_simu = Regression_correction_function(simu_base, simu, obs, date)

    # save
    np.save("Correction_All_SoilMoi0_100cm_inst_19840101_19981231_D_vic", correction_simu)


def correction_compare_plot(obs, simu, simu_correction, date, show=True, save_on=False, save_format="jpg"):
    # cal evaIndicators
    ei_simu = EvaIndicators(simu, obs)
    ei_simu_correction = EvaIndicators(simu_correction, obs)

    # df_out
    df_out = pd.DataFrame(index=["simu", "simu_correction"], columns=["RMSE", "RRMSE", "R", "Ce", "Bias"])
    df_out.loc["simu", :] = [ei_simu.RRMSE(), ei_simu.RRMSE(), ei_simu.R()[0], ei_simu.Ce(), ei_simu.Bias()]
    df_out.loc["simu_correction", :] = [ei_simu_correction.RRMSE(),
                                        ei_simu_correction.RRMSE(),
                                        ei_simu_correction.R()[0],
                                        ei_simu_correction.Ce(),
                                        ei_simu_correction.Bias()]

    # plot compare series
    fig_compare_series = draw_plot.Figure()
    draw_compare_series = draw_plot.Draw(fig_compare_series.ax, fig_compare_series, legend_on=True, labelx=None,
                          labely="Soil Moisture / mm")
    fig_compare_series.ax.set_xlabel("Date")
    draw_obs = draw_plot.PlotDraw(date, obs, color="black", linestyle="solid", label='Observation', linewidth=0.7)
    draw_simu = draw_plot.ScatterDraw(date, simu, s=0.2, c='r', label='Simulation: %.2f' % df_out.loc["simu", "R"])
    draw_simu_correction = draw_plot.ScatterDraw(date, simu_correction, s=0.2, c='b', label='Correction: %.2f' % df_out.loc["simu_correction", "R"])
    draw_compare_series.adddraw(draw_obs, draw_simu, draw_simu_correction)

    if show:
        plt.show()

    if save_on:
        if save_format == "jpg":
            fig_compare_series.savejpg(save_on + "_correction_compare")
        elif save_format == "svg":
            fig_compare_series.savesvg(save_on + "_correction_compare")

        df_out.to_csv(save_on + "_evaluate_correction_compare.csv")

    return df_out


def correction_compare():
    # general set
    show = False
    save_on = "correction_compare"
    save_format = "jpg"

    # data read
    obs = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19480101_20141231_D.npy"))
    simu = np.load(os.path.join(__location__, "All_SoilMoi0_100cm_inst_19840101_19981231_D_vic.npy"))
    simu_correction = np.load(os.path.join(__location__, "Correction_All_SoilMoi0_100cm_inst_19840101_19981231_D_vic.npy"))

    # date
    period_simu = [19840101, 19981231]
    date = pd.date_range("19840101", "19981231", freq="D")

    # find index
    obs_date = np.array(obs[:, 0])
    obs_date -= 0.12
    obs_date = obs_date.astype(int)
    period_simu_index = [np.where(obs_date == period_simu[0])[0][0],
                         np.where(obs_date == period_simu[1])[0][0]]

    obs = obs[period_simu_index[0]: period_simu_index[1] + 1, :]
    obs = obs[:, 1:]

    # mean
    obs = obs.mean(axis=1)
    simu = simu.mean(axis=1)
    simu_correction = simu_correction.mean(axis=1)

    # compare
    correction_compare_plot(obs, simu, simu_correction, date, show=show, save_on=save_on, save_format=save_format)


if __name__ == "__main__":
    # general set
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # runvic
    # runvic()

    # evaluate
    evaluate()

    # correction
    # CDF_Match_correction()
    # Regression_correction()

    # correction_compare
    # correction_compare()
