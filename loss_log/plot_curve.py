"""
plot loss
"""
from matplotlib.ticker import FuncFormatter
import numpy as np 
import matplotlib.pyplot as plt 

loss_file = "res_pose_5_5_facelandmark_branches_s.txt"

upper_face_loss_coarse = []
upper_face_loss_refine = []
bottom_face_loss_coarse = []
bottom_face_loss_refine = []

with open(loss_file, 'r') as file:
	loss_record = [loss.strip() for loss in file.readlines()]

for loss in loss_record:
	loss = loss.split(' ')
	upper_face_loss_coarse.append(float(loss[2]))
	upper_face_loss_refine.append(float(loss[0]))
	bottom_face_loss_coarse.append(float(loss[3]))
	bottom_face_loss_refine.append(float(loss[1]))

plt.plot(bottom_face_loss_coarse, label='bottom_face_loss_coarse')
plt.plot(upper_face_loss_coarse, label='upper_face_loss_coarse')
plt.plot(bottom_face_loss_refine, label='bottom_face_loss_refine')
plt.plot(upper_face_loss_refine, label='upper_face_loss_refine')
ax = plt.gca()
ax.set_yscale("linear")
# ax.yaxis.set_major_formatter(formatter)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(0, 0.0025)
# plt.grid()
plt.legend(loc='best')
plt.savefig('./log/loss.jpg')
# plt.show()
# plt.cla()
