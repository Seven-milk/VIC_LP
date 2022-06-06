# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import os


def TwoDimension2OneDimension(file, out_path, value_name):
    """ Transform 2D (lat/lon) file to 1D (num: lat_lon) file
    input:
        file: 2D file (lat, lon), pd.Dataframe
        out: 1D file path
        value_name: str, define value name
    """
    out = pd.DataFrame(columns=["lat", "lon", value_name])
    num = 0  # all = 1440 * 720 = 1,036,800A
    for i in file.index:
        for j in file.columns:
            out = out.append(pd.DataFrame({"lat": [i], "lon": [j], value_name:
                [file.loc[i, j]]}), ignore_index=True)
            print(num, ": ", "lat ", i, " lon ", j)
            num += 1

    # save
    out.to_excel(out_path, header=True, index=True)


if __name__ == "__main__":
    # path set
    home = "G:/data_zxd/landuse/UMD_landcover_classification/ISLSCP II University of Maryland Global Land Cover" \
           " Classifications/umd_landcover_qdeg"
    classification_path = os.path.join(home, "umd_landcover_class_qd.xlsx")
    classification = pd.read_excel(classification_path, sheet_name=0, header=0, index_col=0)

    # transform classification(umd_landcover_class_qd_out.xlsx) into 1D file
    TwoDimension2OneDimension(classification, "umd_landcover_class_qd_out.xlsx", "classification")