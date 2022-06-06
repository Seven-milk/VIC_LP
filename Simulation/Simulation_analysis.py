# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import numpy as np
import pandas as pd
import FlashDrought
from FlashDrought import FlashDrought_Frozen as FD
import pretreatment_data
import Nonparamfit, Univariatefit
from scipy import stats
import Statistical_data
import Analysis_spatial
import map_plot, draw_plot


def replaceSimuIntoGLDAS(GLDAS_sm, simu_sm, home_path):
    GLDAS_VIC_sm = np.array(GLDAS_sm)

    # find index
    period_simu = [19840101, 19981231]
    GLDAS_date = np.array(GLDAS_sm[:, 0])
    GLDAS_date -= 0.12
    GLDAS_date = GLDAS_date.astype(int)
    period_simu_index = [np.where(GLDAS_date == period_simu[0])[0][0],
                         np.where(GLDAS_date == period_simu[1])[0][0]]

    GLDAS_VIC_sm[period_simu_index[0]: period_simu_index[1] + 1, 1:] = simu_sm

    # save
    np.save(os.path.join(home_path, "GLDAS_VIC_SoilMoi0_100cm_inst_19480101_20141231_D.npy"), GLDAS_VIC_sm)
    return GLDAS_VIC_sm


def UpscaleD_to_Pentad(GLDAS_VIC_sm, home_path):
    save_path = os.path.join(home_path, "GLDAS_VIC_SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy")
    GLDAS_VIC_sm_Pentad_Upscale = pretreatment_data.UpscaleTime(original_series=GLDAS_VIC_sm[:, 1:], multiple=5,
                                                                original_date=GLDAS_VIC_sm[:, 0],
                                                                save_path=save_path,
                                                                combine=True, info="")
    GLDAS_VIC_sm_Pentad = GLDAS_VIC_sm_Pentad_Upscale()
    return GLDAS_VIC_sm_Pentad


def Cal_smpercentile_multiple_distribution(sm_, home_path):
    save_path = os.path.join(home_path, "GLDAS_VIC_SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile")

    sm = sm_[:, 1:]
    date = sm_[:, 0]
    format = '%Y%m%d'

    # nonparam
    nonparamdistribution = Nonparamfit.Gringorten()

    # distributions
    distribution = [Univariatefit.UnivariateDistribution(stats.expon), Univariatefit.UnivariateDistribution(stats.gamma),
                    Univariatefit.UnivariateDistribution(stats.beta), Univariatefit.UnivariateDistribution(stats.lognorm),
                    Univariatefit.UnivariateDistribution(stats.logistic), Univariatefit.UnivariateDistribution(stats.pareto),
                    Univariatefit.UnivariateDistribution(stats.weibull_min), Univariatefit.UnivariateDistribution(stats.genextreme)]

    # cal sm percentile
    cspmd = pretreatment_data.CalSmPercentileMultiDistribution(sm, date, format, distribution=distribution,
                                                               nonparamdistribution=nonparamdistribution,
                                                               info="multiple distribution sm percentile",
                                                               save_path=save_path)
    sm_percentile, distribution_ret = cspmd()
    return sm_percentile, distribution_ret


def extract_simu_period(sm_percentile, home_path):
    # find index
    period_simu = [19840101, 19981231]
    sm_date = np.array(sm_percentile[:, 0])
    sm_date -= 0.12
    sm_date = sm_date.astype(int)
    period_simu_index = [np.where(sm_date >= period_simu[0])[0][0],
                         np.where(sm_date <= period_simu[1])[0][-1]]

    sm_percentile_simu = sm_percentile[period_simu_index[0]: period_simu_index[1] + 1, 1:]
    save_path = os.path.join(home_path, f"GLDAS_VIC_SoilMoi0_100cm_inst_{sm_date[period_simu_index[0]]}_{sm_date[period_simu_index[1]]}_Pentad_muldis_SmPercentile.npy")
    np.save(save_path, sm_percentile_simu)
    return sm_percentile_simu


def loop_pretreatment_data():
    # home path
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # listdir
    dirs_simu = [d for d in os.listdir(__location__) if d.startswith("forcing")]

    # loop for dirs to pretreatment data
    for i in range(len(dirs_simu)):
        dir_simu = dirs_simu[i]

        # print info
        print(f"pretreatment data for {dir_simu}")

        # read data
        GLDAS_sm = np.load(os.path.join(__location__, "SoilMoi0_100cm_inst_19480101_20141231_D.npy"))
        simu_sm = np.load(os.path.join(__location__, dir_simu, "Correction_Regression", "Correction_All_SoilMoi0_100cm_inst_19840101_19981231_D_vic.npy"))

        # replaceSimuIntoGLDAS
        home_path = os.path.join(__location__, dir_simu, "Simulation_analysis")
        GLDAS_VIC_sm = replaceSimuIntoGLDAS(GLDAS_sm, simu_sm, home_path)

        # UpscaleD_to_Pentad
        GLDAS_VIC_sm_Pentad = UpscaleD_to_Pentad(GLDAS_VIC_sm, home_path)

        # Cal_smpercentile_multiple_distribution
        sm_percentile, _ = Cal_smpercentile_multiple_distribution(GLDAS_VIC_sm_Pentad, home_path)

        # extract_simu_period
        sm_percentile_simu = extract_simu_period(sm_percentile, home_path)


def loop_cal_grid_static():
    # home path
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # listdir
    dirs_simu = [d for d in os.listdir(__location__) if d.startswith("forcing")]

    # loop for dirs to cal grid_static
    for i in range(len(dirs_simu)):
        dir_simu = dirs_simu[i]

        # print info
        print(f"cal grid_static for {dir_simu}")

        # read data
        sm_percentile_simu = np.load(os.path.join(__location__, dir_simu, "Simulation_analysis",
                                                  "GLDAS_VIC_SoilMoi0_100cm_inst_19840104_19981231_Pentad_muldis_SmPercentile.npy"))

        # cal grid static
        std = Statistical_data.StaticalData()
        grid_DFD_ret = std.gridDroughtFDTimingStatistics(drought_index=sm_percentile_simu, date_pentad=date_simu,
                                                         save_on=None)
        grid_DFD_ret["grid_static"].to_excel(os.path.join(__location__, dir_simu, "Simulation_analysis", "grid_static.xlsx"))


def loop_analysis_spatial():
    # home path
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # listdir
    dirs_simu = [d for d in os.listdir(__location__) if d.startswith("forcing")]

    # map_boundys
    # map_boundys = "not_set"
    map_boundys = np.array([[[0, 60], [0, 30]],
                            [[0, 20], [-0.2, 0]],
                            [[0, 100], [0, 2]],
                            [[0, 0.35], [0.025, 0.125]]])

    # loop for dirs to analysis spatial
    for i in range(len(dirs_simu)):
        dir_simu = dirs_simu[i]

        # print info
        print(f"cal grid_static for {dir_simu}")

        grid_static = pd.read_excel(os.path.join(__location__, dir_simu, "Simulation_analysis", "grid_static.xlsx"), index_col=0)

        fig_map, fig_boxplot = Analysis_spatial.droughtFDParamsSpatialAnalysis(extent, det, lat, lon, boundry_shpMap,
                                                                               grid_static, chinese_font_on=True,
                                                                               show=False, save_on=False,
                                                                               save_format="jpg", map_boundys=map_boundys)
        fig_map.fig.savefig(os.path.join(__location__, dir_simu, "Simulation_analysis", "droughtFDParamsSpatialAnalysis.jpg"),
                                         dpi=fig_map.dpi, bbox_inches='tight')
        fig_boxplot.fig.savefig(os.path.join(__location__, dir_simu, "Simulation_analysis", "droughtFDParamsSpatialAnalysis_boxplot.jpg"),
                                         dpi=fig_boxplot.dpi, bbox_inches='tight')

        # df_out
        if i == 0:
            df_out = pd.DataFrame(index=grid_static.columns, columns=dirs_simu)

        df_out.loc[:, dir_simu] = grid_static.mean()

    # df_out save
    df_out.to_excel("analysis_spatial.xlsx")


def droughtFDParamsSpatialAnalysis(data, extent, det, lat, lon, boundry_shpMap,
                                   chinese_font_on=True, show=False,
                                   save_on="droughtFDParamsSpatialAnalysis", save_format="jpg",
                                   map_boundys=False):
    # data set
    shape_ = data.shape
    cb_label = np.array(["数目", "PR", "候", "PR/候"]) if chinese_font_on else np.array(["number", "pentad$\cdot$PR", "pentad", "PR/pentad"])
    cb_label = cb_label.reshape((1, -1))
    cb_label = cb_label.repeat(shape_[0], axis=0)
    row_names = [f"方案{i}" for i in range(1, 8)] if chinese_font_on else [f"Scheme {i}" for i in range(1, 8)]
    col_names = ["总骤旱数目", "骤旱烈度均值", "骤旱历时均值", "RI均值"] if chinese_font_on else \
        ["Total flash drought number", "Mean flash drought severity",
         "Mean flash drought duration", "Mean rate of intensification"]

    # map plot
    font_family = "Microsoft YaHei" if chinese_font_on else "Arial"
    # fig_kwargs = dict(family=font_family, expand_factor_row=6, figsize=(4, 6), wspace=0.5, hspace=0.5)
    fig_kwargs = dict(family=font_family, expand_factor_row=6, figsize=(4, 4), wspace=0.5, hspace=0.1) # 4,4
    map_kwargs = None
    if isinstance(map_boundys, str):
        map_boundys = None
    elif map_boundys is not None:
        map_boundys = map_boundys
    else:
        map_boundys = np.array([[[0, 30], [-0.2, 0], [0, 2], [0.025, 0.125]]])
        map_boundys = map_boundys.repeat(shape_[0], axis=0)

    raster_kwargs = dict(map_boundry=map_boundys)
    cb_kwargs = dict(shrink=0.6)
    annotationOrder_kwargs = None
    annotationRows_kwargs = dict(xy=(-0.5, 0.5))
    annotationSubNames_kwargs = None
    order_texts = None

    arm = map_plot.AnnotationRasterMap3(data, extent, det, lat, lon, boundry_shpMap,
                                        row_names=row_names, col_names=col_names,
                                        fig_kwargs=fig_kwargs, map_kwargs=map_kwargs,
                                        raster_kwargs=raster_kwargs, cb_kwargs=cb_kwargs,
                                        annotationOrder_kwargs=annotationOrder_kwargs, order_texts=order_texts,
                                        annotationRows_kwargs=annotationRows_kwargs, cb_label=cb_label,
                                        sub_names=None, annotationSubNames_kwargs=annotationSubNames_kwargs,
                                        annotationOrder=False)
    fig_map = arm()

    # show
    if show:
        fig_map.show()

    # save
    if save_on:
        if save_format == "jpg":
            fig_map.savejpg(save_on)
        elif save_format == "svg":
            fig_map.savesvg(save_on)
        else:
            raise ValueError("only support jpg and svg save_format!")

    return fig_map


def loop_analysis_spatial_combine():
    # home path
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # listdir
    dirs_simu = ["forcing",
                 "forcing_p_add_20%",
                 "forcing_p_sub_20%",
                 "forcing_t_add_20%",
                 "forcing_t_sub_20%",
                 "forcing_p_sub_20%_t_add_20%",
                 "forcing_p_add_20%_t_sub_20%"]

    # map_boundys
    map_boundys = np.array([[[0, 30], [-0.2, 0], [0, 2], [0.025, 0.125]]])
    map_boundys = map_boundys.repeat(len(dirs_simu), axis=0)

    # data
    data_all = []

    # loop for dirs to collect data
    for i in range(len(dirs_simu)):
        dir_simu = dirs_simu[i]
        # print info
        print(f"cal grid_static for {dir_simu}")
        grid_static = pd.read_excel(os.path.join(__location__, dir_simu, "Simulation_analysis", "grid_static.xlsx"), index_col=0)
        data = [grid_static["FD_number"].values, grid_static["FDS_mean"].values,
                grid_static["FDD_mean"].values, grid_static["RI_mean_mean"].values]

        data_all.append(data)

    # data_all to np.array
    data_all = np.array(data_all)

    # plot
    fig_map = droughtFDParamsSpatialAnalysis(data_all, extent, det, lat, lon, boundry_shpMap,
                                             chinese_font_on=True, show=False, save_on=False, save_format="jpg",
                                             map_boundys=map_boundys)

    fig_map.fig.savefig("droughtFDParamsSpatialAnalysis_combine.jpg", dpi=fig_map.dpi, bbox_inches='tight')


if __name__ == "__main__":
    # general set
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # period
    period_simu = [19840101, 19981231]
    date_simu = pd.date_range("19840101", "19981231", freq="D")
    date_all = pd.date_range("19480101", "20141231", freq="D")

    # map set
    root = "H"
    home = f"{root}:/research/flash_drough/"
    coord_path = os.path.join(home, "coord.txt")
    coord = pd.read_csv(coord_path, sep=",")

    lon = coord.loc[:, "lon"].values
    lat = coord.loc[:, "lat"].values
    det = 0.25
    res_grid = 1
    res_label = 2
    lat_min = min(lat)
    lon_min = min(lon)
    lat_max = max(lat)
    lon_max = max(lon)
    extent = [lon_min, lon_max, lat_min, lat_max]
    boundry_shp = [f"{root}:/GIS/Flash_drought/f'r_project.shp"]
    Helong_shp = [f"{root}:/GIS/Flash_drought/he_long_projection.shp"]
    boundry_shpMap = map_plot.ShpMap(boundry_shp)

    # loop_pretreatment_data
    # loop_pretreatment_data()

    # loop_cal_grid_static
    # loop_cal_grid_static()

    # loop_analysis_spatial
    # loop_analysis_spatial()

    # loop_analysis_spatial_combine
    loop_analysis_spatial_combine()

    # flashDroughtAnalysis
    # flashDroughtAnalysis()















