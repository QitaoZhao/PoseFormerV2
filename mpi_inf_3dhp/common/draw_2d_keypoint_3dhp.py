import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import scipy.io as scio

keypoints = np.load('../dataset/data_test_3dhp.npz',allow_pickle=True)
image = mpimg.imread(r'..\3dhp_test\TS6\imageSequence\img_000061.jpg')

parents=[1,15,1,2,3,1,5,6,14,8,9,14,11,12,-1,14,15]
joints_right_2d=[2, 3, 4, 8, 9, 10]
colors_2d = np.full(17, 'midnightblue')
colors_2d[joints_right_2d] = 'yellowgreen'


data=keypoints['data'].item()

data_sequence = data["TS6"]
valid_frame = data_sequence["valid"].astype(bool)

valid_cnt = 0
image_cnt = 0
for i in range(len(valid_frame)):
    if valid_frame[i] == True:
        valid_cnt+=1
        #TS1:101, TS4:81, TS5:71, TS6:11
        if valid_cnt==11:
            image_cnt = i
            break

#TS1:1040, TS4:960, TS5:70, TS6:60
#equals to image_cnt
test = data_sequence['data_2d'][60]
#TS1:100, TS4:80, TS5:70, TS6:10
#equals to image_cnt-1
data_2d = data_sequence['data_2d'][valid_frame][10]
#data_2d = data["TS3"]['data_2d'][364]

plt.axis("off")
# plt.xlim(0,1000)
# plt.ylim(0,1000)
plt.imshow(image)

for j, j_parent in enumerate(parents):
    if j_parent == -1:
        continue

    plt.plot([data_2d[j, 0], data_2d[j_parent, 0]],
                            [data_2d[j, 1], data_2d[j_parent, 1]], linewidth=1,color='pink')

plt.scatter(data_2d[:, 0], data_2d[:, 1], 10, color=colors_2d, edgecolors='white', zorder=10)

#plt.show()
plt.savefig("./plot/3dhp_test_2d.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
plt.close()
print("")
