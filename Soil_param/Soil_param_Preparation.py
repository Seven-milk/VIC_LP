# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
"""
HWSD classification: top: 0-30cm, sub: 30-100cm
    1 clay(heavy)
    2 silty clay
    3 clay
    4 silty clay loam
    5 clay loam
    6 silt
    7 silt loam
    8 sandy clay
    9 loam
    10 sandy clay loam
    11 sandy loam
    12 loamy sand
    13 sand

variable for Calibrating
binfilt = 0.35
Ds = 0.7
Dsmax = 7
Ws = 0.02
depth1 = 0.1  # 0 - 0.1
depth2 = 0.5  # 0.1 - 0.6
depth3 = 2.0  # 0.6 - 2.6

"""

import pandas as pd
import os
import json
from tqdm import *


class Soil_param_Preparation:

    def __init__(self):
        pass

    def __call__(self, binfilt=0.35, Ds=0.7, Dsmax=7, Ws=0.02, depth1=0.1, depth2=0.5, depth3=2.0):
        # local path
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        Soil_param_out = os.path.join(__location__, "../Soil_param.txt")

        try:
            # read files: coord dem HWSD_Sclass HWSD_Tclass soil_type_attributes
            coord = pd.read_csv(os.path.join(__location__, "../coord.txt"), sep=",")

            # control the grid on, if not exit, turn all grid on
            if os.path.exists(os.path.join(__location__, "../coord_on.txt")):
                coord_on = pd.read_csv(os.path.join(__location__, "../coord_on.txt"), sep=",")
            else:
                coord_on = pd.read_csv(os.path.join(__location__, "../coord.txt"), sep=",")
            dem_mean_lp = pd.read_csv(os.path.join(__location__, "../Soil_param/dem_mean_lp.txt"), sep=",")
            HWSD_Tclass = pd.read_csv(os.path.join(__location__, "../Soil_param/HWSD_Tclass.txt"))
            HWSD_Sclass = pd.read_csv(os.path.join(__location__, "../Soil_param/HWSD_Sclass.txt"))

            # read soil_type_attributes and change the form
            with open(os.path.join(__location__, "../Soil_param/soil_type_attributes.json")) as file:
                soil_type_attributes = json.load(file)
                soil_type_attributes = soil_type_attributes["classAttributes"]
                temp = {}
                for i in range(len(soil_type_attributes)):
                    soil_type_attributes_ = soil_type_attributes[i]
                    temp[soil_type_attributes_["class"]] = soil_type_attributes_["properties"]
                soil_type_attributes = temp

            # read Average_annual_precipitation
            Average_annual_precipitation = pd.read_csv(os.path.join(__location__, "../Mete_param/pre_LP_198301_199812_coord"
                                                                                  "_Average_annual_precipitation.csv"),
                                                       sep=" ", index_col=0)

        except:
            print("Make sure 'coord.txt' 'Soil_param/dem_mean_lp.txt' 'Soil_param/HWSD_Tclass.txt'"
                          "'Soil_param/HWSD_Sclass.txt' 'Soil_param/soil_type_attributes.json'"
                          "'Mete_param/pre_LP_198301_199812_coord_Average_annual_precipitation.csv'"
                          "in this directory")

            raise ValueError("Input Error")

        # variable for predefining
        c = 2  # Exponent used in baseflow curve
        phis = -9999
        avg_t = 27  # average temperature of soil
        dp = 4  # depth that soil temp does not change
        tsoil_den = 2685  # top layer soil density
        ssoil_den = 2685  # bottom layer soil density
        rough = 0.01  # bare soil roughness coefficient
        srough = 0.001  # snow roughness coefficient
        fs_act = 1  # boolean value to run frozen soil algorithm

        # make Soil_param file
        with open(Soil_param_out, "w") as f:
            lat_on = coord_on.loc[:, "lat"]
            lon_on = coord_on.loc[:, "lon"]
            coord_on_ = list(zip(lat_on, lon_on))

            # loop for each grid
            for i in tqdm(coord.index, colour="green"):
                gridcel = i + 1

                lat = coord.loc[i, "lat"]
                lon = coord.loc[i, "lon"]
                coord_ = list(zip([lat], [lon]))
                run_cell = 1 if coord_[0] in coord_on_ else 0

                Topsoil_class = HWSD_Tclass.loc[i, "MAJORITY"]
                Subsoil_class = HWSD_Sclass.loc[i, "MAJORITY"]

                # deal with 0 type (nodata), seach nearby grid
                j = 0
                while Topsoil_class == 0:
                    if Subsoil_class != 0:
                        Topsoil_class = Subsoil_class
                    else:
                        try:
                            Topsoil_class = HWSD_Tclass.loc[i - 1 - j, "MAJORITY"]
                        except:
                            Topsoil_class = HWSD_Tclass.loc[i + 1 + j, "MAJORITY"]
                    j += 1

                j = 0
                while Subsoil_class == 0:
                    if Topsoil_class != 0:
                        Subsoil_class = Topsoil_class
                    else:
                        try:
                            Subsoil_class = HWSD_Sclass.loc[i - 1 - j, "MAJORITY"]
                        except:
                            Subsoil_class = HWSD_Sclass.loc[i + 1 + j, "MAJORITY"]
                    j += 1

                texpt = 3 + (2 * float(soil_type_attributes[str(Topsoil_class)]['SlopeRCurve']))
                sexpt = 3 + (2 * float(soil_type_attributes[str(Subsoil_class)]['SlopeRCurve']))

                tksat = (float(soil_type_attributes[str(Topsoil_class)]['SatHydraulicCapacity']) * 240)
                sksat = (float(soil_type_attributes[str(Subsoil_class)]['SatHydraulicCapacity']) * 240)

                elev = dem_mean_lp.loc[i, "MEAN"]

                tbub = soil_type_attributes[str(Topsoil_class)]['BubblingPressure']
                sbub = soil_type_attributes[str(Subsoil_class)]['BubblingPressure']
                tquartz = soil_type_attributes[str(Topsoil_class)]['Quartz']
                squartz = soil_type_attributes[str(Subsoil_class)]['Quartz']
                tbulk_den = float(soil_type_attributes[str(Topsoil_class)]['BulkDensity']) * 1000
                sbulk_den = float(soil_type_attributes[str(Subsoil_class)]['BulkDensity']) * 1000

                off_gmt = lon * 24 / 360  # time zone offset from GMT

                twrc_frac = float(soil_type_attributes[str(Topsoil_class)]['FieldCapacity']) \
                            / float(soil_type_attributes[str(Topsoil_class)]['Porosity'])  # top layer critical point
                swrc_frac = float(soil_type_attributes[str(Subsoil_class)]['FieldCapacity']) \
                            / float(soil_type_attributes[str(Subsoil_class)]['Porosity'])  # sub layer critical point
                twpwp_frac = float(soil_type_attributes[str(Topsoil_class)]['WiltingPoint']) \
                             / float(soil_type_attributes[str(Topsoil_class)]['Porosity'])  # top layer wilting point
                swpwp_frac = float(soil_type_attributes[str(Subsoil_class)]['WiltingPoint']) \
                             / float(soil_type_attributes[str(Subsoil_class)]['Porosity'])  # sub layer wilting point

                # annprecip  # climotological average precipitation
                annprecip = Average_annual_precipitation.loc[i].values[0]

                tresid = soil_type_attributes[str(Topsoil_class)]['Residual']  # top layer residual moisture
                sresid = soil_type_attributes[str(Subsoil_class)]['Residual']  # sub layer residual moisture

                init_moist1 = (float(tbulk_den) / tsoil_den) * depth1 * 1000  # top layer inital moisture conditions
                init_moist2 = (float(sbulk_den) / ssoil_den) * depth2 * 1000  # second layer initial moisture conditions
                init_moist3 = (float(sbulk_den) / ssoil_den) * depth3 * 1000  # bottom layer initial moisture conditions

                # write
                f.write(f"{str(run_cell)} {str(gridcel)} "
                        f"{lat:.4f}\t{lon:.4f}\t"
                        f"{str(binfilt)}\t{str(Ds)}\t{str(Dsmax)}\t{str(Ws)}\t{str(c)}\t"
                        f"{str(texpt)}\t{str(texpt)}\t{str(sexpt)}\t"
                        f"{str(tksat)}\t{str(tksat)}\t{str(sksat)}\t"
                        f"{str(phis)}\t{str(phis)}\t{str(phis)}\t"
                        f"{str(init_moist1)}\t{str(init_moist2)}\t{init_moist3}\t"
                        f"{str(elev)}\t"
                        f"{str(depth1)}\t{str(depth2)}\t{str(depth3)}\t"
                        f"{str(avg_t)}\t{str(dp)}\t"
                        f"{str(tbub)}\t{str(tbub)}\t{str(sbub)}\t{str(tquartz)}\t{str(tquartz)}\t{str(squartz)}\t"
                        f"{str(tbulk_den)}\t{str(tbulk_den)}\t{str(sbulk_den)}\t"
                        f"{str(tsoil_den)}\t{str(tsoil_den)}\t{str(ssoil_den)}\t"
                        f"{str(off_gmt)}\t"
                        f"{str(twrc_frac)}\t{str(twrc_frac)}\t{str(swrc_frac)}\t"
                        f"{str(twpwp_frac)}\t{str(twpwp_frac)}\t{str(swpwp_frac)}\t"
                        f"{str(rough)}\t{srough}\t"
                        f"{str(annprecip)}\t"
                        f"{str(tresid)}\t{str(tresid)}\t{str(sresid)}\t{str(fs_act)}\n")


if __name__ == "__main__":
    sp = Soil_param_Preparation()
    sp()