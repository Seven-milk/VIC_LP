# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import json
import os
import pandas as pd
import numpy as np


def Veg_param_Preparation(coord_path, attribute_path, umd_path_home, Veglib_path, Veg_param_out):
    # sep set
    sep = " "

    # read coord
    coord = pd.read_csv(coord_path, sep=",")

    # read veg_type_attribute
    with open(attribute_path) as file:
        attribute = json.load(file)
        attribute = attribute["classAttributes"]

    # read umd_landcover_qd_c01-11
    umd_path = [os.path.join(umd_path_home, path) for path in os.listdir(umd_path_home)]  # x = np.loadtxt(umd_path[0])
    umd_path.sort()
    umd_landcover_Cv_all = [np.loadtxt(path) for path in umd_path]  # 1-11 class

    # read Veglib
    Veglib = pd.read_csv(Veglib_path, sep="\t", index_col=0)  # index is class

    # cal index for each coord
    lat_global = np.arange(89.875, -90.125, -0.25)  # -90.125 = -89.875 - 0.25 (last not contain)
    lon_global = np.arange(-179.875, 180.125, 0.25)
    row_index = []
    col_index = []
    for i in coord.index:
        lat_coord = coord.loc[i, "lat"]
        lon_coord = coord.loc[i, "lon"]
        row_index_ = np.where(lat_global == lat_coord)[0][0]
        col_index_ = np.where(lon_global == lon_coord)[0][0]

        row_index.append(row_index_)
        col_index.append(col_index_)

    # make Veg_param file
    with open(Veg_param_out, "w") as f:
        # loop for each grid
        for i in coord.index:
            gridcel = i + 1
            f.write(str(gridcel) + sep)  # write gridcel

            Nveg = 0
            veg_class = []
            Cv = []
            root_depth1 = []
            root_depth2 = []
            root_depth3 = []
            root_fract1 = []
            root_fract2 = []
            root_fract3 = []

            # loop for each class of all class (11)
            for j in range(len(umd_landcover_Cv_all)):
                veg_class_ = j + 1
                Cv_ = umd_landcover_Cv_all[j][row_index[i], col_index[i]]
                if Cv_ != 0:
                    veg_class.append(veg_class_)
                    Cv.append(Cv_ / 100)  # Conversion to percentage
                    Nveg += 1
                    root_depth1.append(attribute[veg_class_]["properties"]["rootd1"])
                    root_depth2.append(attribute[veg_class_]["properties"]["rootd2"])
                    root_depth3.append(attribute[veg_class_]["properties"]["rootd3"])
                    root_fract1.append(attribute[veg_class_]["properties"]["rootfr1"])
                    root_fract2.append(attribute[veg_class_]["properties"]["rootfr2"])
                    root_fract3.append(attribute[veg_class_]["properties"]["rootfr3"])

            f.write(str(Nveg) + "\n")  # write Nveg   sep + sep.join(6 * str(0))

            # loop for each class in this grid
            for k in range(Nveg):
                f.write("\t")  # format
                f.write(str(veg_class[k]) + sep)  # write veg_class
                f.write(f"{Cv[k]:.4f}" + sep)  # write Cv
                f.write(root_depth1[k] + sep)  # write root_depth1
                f.write(root_fract1[k] + sep)  # write root_fract1
                f.write(root_depth2[k] + sep)  # write root_depth2
                f.write(root_fract2[k] + sep)  # write root_fract2
                f.write(root_depth3[k] + sep)  # write root_depth3
                f.write(root_fract3[k] + "\n")  # write root_fract3

                # write LAI for month 1-12
                # f.write(6 * " ")  # format
                # LAI = [str(element)for element in list(Veglib.loc[1, "JAN-LAI": "DEC-LAI"].values)]
                # f.write(sep.join(LAI) + "\n")  # write LAI 1-12

            print(f"grid{i} complete, all grid is {len(coord.index)}")


if __name__ == "__main__":
    # general set
    home = "H:/research/flash_drough/code/VIC"
    coord_path = os.path.join(home, "coord.txt")
    attribute_path = os.path.join(home, "Veg_param_lib/veg_type_attributes_rheas.json")
    umd_path_home = "H:/research/flash_drough/code/VIC/Veg_param_lib/umd_landcover_qd"
    Veglib_path = os.path.join(home, "Veg_param_lib/Veglib.txt")
    Veg_param_out = "Veg_param.txt"
    Veg_param_Preparation(coord_path, attribute_path, umd_path_home, Veglib_path, Veg_param_out)
