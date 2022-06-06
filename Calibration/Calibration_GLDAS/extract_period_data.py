# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import numpy as np

# period set
period_cali = [19840101, 19891231]
period_vali = [19900101, 19951231]
period_simu = [19960101, 19981231]

# path set
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

original_data_path = os.path.join(__location__, "SoilMoi0_100cm_inst_19480101_20141231_D.npy")
cali_path = os.path.join(__location__, f"SoilMoi0_100cm_inst_{period_cali[0]}_{period_cali[1]}_D")
vali_path = os.path.join(__location__, f"SoilMoi0_100cm_inst_{period_vali[0]}_{period_vali[1]}_D")
simu_path = os.path.join(__location__, f"SoilMoi0_100cm_inst_{period_simu[0]}_{period_simu[1]}_D")

# read data
original_data = np.load(original_data_path)

# find index
original_date = np.array(original_data[:, 0])
original_date -= 0.12
original_date = original_date.astype(int)
period_cali_index = [np.where(original_date == period_cali[0])[0][0], np.where(original_date == period_cali[1])[0][0]]
period_vali_index = [np.where(original_date == period_vali[0])[0][0], np.where(original_date == period_vali[1])[0][0]]
period_simu_index = [np.where(original_date == period_simu[0])[0][0], np.where(original_date == period_simu[1])[0][0]]

# extract and save
cali_data = original_data[period_cali_index[0]: period_cali_index[1] + 1, :]
vali_data = original_data[period_vali_index[0]: period_vali_index[1] + 1, :]
simu_data = original_data[period_simu_index[0]: period_simu_index[1] + 1, :]

np.save(cali_path, cali_data)
np.save(vali_path, vali_data)
np.save(simu_path, simu_data)