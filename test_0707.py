import summer_utiles as su
import os
import numpy as np
import matplotlib.pyplot as plt
import binascii
from scipy import io

caffe_dir = "D:\CAFFE_DATA"
caffe_file_list = os.listdir(caffe_dir)
caffe_file_list_mat = [file for file in caffe_file_list if file.endswith(".mat")]
caffe_data_path = os.path.join('D:\CAFFE_DATA',caffe_file_list_mat[000])

fpga_dir = "D:\FPGA_DATA"
fpga_1st_list = os.listdir(fpga_dir)
fpga_1st_path = os.path.join('D:\FPGA_DATA',fpga_1st_list[1])
fpga_2nd_list = os.listdir(fpga_1st_path)
fpga_2nd_path = os.path.join('D:\FPGA_DATA',fpga_1st_list[1],fpga_2nd_list[0])
fpga_last_list = os.listdir(fpga_2nd_path)
fpga_file_list_txt = [file for file in fpga_last_list if file.endswith(".txt")]
fpga_data_path = os.path.join('D:\FPGA_DATA',fpga_1st_list[1],fpga_2nd_list[0],fpga_file_list_txt[0])

#key = io.loadmat(caffe_data_path)
caffe = su.parsedata(caffe_data_path,'conv1_1','mat')
fpga = su.parsedata(fpga_data_path,'conv1_1','txt')
