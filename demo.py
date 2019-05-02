import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import cv2

limbSeq = [[7, 11],  [11, 10], [10, 9], [7, 12],[12, 13], [13, 14],[7, 6], [6, 2], [2, 1], [1, 0], [6, 3], [3, 4], [4, 5],[7, 8], [11, 8], [12, 8]]
mapIdx = [[22, 23], [24, 25], [26, 27], [16, 17], [18, 19], [20, 21], [2, 3],  [10, 11], [12, 13], [14, 15], [4, 5], [6, 7], [8, 9],[0, 1], [28, 29], [30, 31]]
colors = [[85, 255, 0], [255, 85, 0],[0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0],  [255, 170, 0], [255, 255, 0], [170, 255, 0],]

boxsize = 384
scale_search = [0.6, 1, 1.3]
stride = 8
padValue = 0.
thre_point = 0.05
thre_line = 0.05
stickwidth = 2

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def construct_model(weight_name):

    from model.deeplabv3 import AtrousPose
    model = AtrousPose()
    model.load_state_dict(torch.load(weight_name))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    return model


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def normalize(origin_img):
    origin_img = np.array(origin_img, dtype=np.float32)
    origin_img = origin_img.astype(np.float32) / 255.

    return origin_img


def process(model, input_path):
    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape

    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 16))  # num_point
    paf_avg = np.zeros((height, width, 32))  # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, 16, padValue)

        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        mask = np.ones((1, 1, int(input_img.shape[2] / stride), int(input_img.shape[3] / stride)), dtype=np.float32)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())
        vec, heat = model(input_var)
        # get the heatmap
        heatmap = heat.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # ind = 1
    # for j in range(0, 8, 2):
    #   U = paf_avg[:,:,j] * -1
    #   V = paf_avg[:,:,j+1]
    #   X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    #   M = np.zeros(U.shape, dtype='bool')
    #   M[U**2 + V**2 < 0.5 * 0.5] = True
    #   U = ma.masked_array(U, mask=M)
    #   V = ma.masked_array(V, mask=M)

    #   plt.axis('off')
    #   plt.subplot(2,4,ind)
    #   ind+=1
    #   # plt.imshow(origin_img[:,:,[2,1,0]], alpha = .5)
    #   s = 4
    #   plt.imshow(origin_img[:,:,[2,1,0]], alpha = .5)
    #   Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
    #              scale=20, headaxislength=2, alpha=.5, width=0.001, color='r')

      # fig = plt.gcf()
      # fig.set_size_inches(20, 20)
    
    # w, h = origin_img.shape[0:2] 
    # for j in range(0, 32, 2):
    #   print(j, j+1)
    #   paf = np.abs(paf_avg[:, :, j])
    #   paf += np.abs(paf_avg[:, :, j + 1])
    #   # print(paf.shape)
    #   paf[paf > 1] = 1
    #   paf = paf.reshape((w, h, 1))
    #   paf *= 255
    #   paf = paf.astype(np.uint8)
    #   paf = cv2.applyColorMap(paf, cv2.COLORMAP_JET)
    #   # paf = paf.reshape((448,448,1))
    #   # paf /= 255
    #   # result = paf * 0.4 + img * 0.5
    #   plt.imshow(origin_img)
    #   plt.imshow(paf, alpha=0.5)
    #   plt.show()
    #   plt.close()
    # ind = 5
    # for j in range(0, 4):
    #   heatmap = heatmap_avg[:, :, j]
    #   heatmap = heatmap.reshape((1000, 800, 1))
    #   heatmap *= 255
    #   heatmap = heatmap.astype(np.uint8)
    #   heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #   # heatmap = heatmap.reshape((448,448,1))
    #   # heatmap /= 255
    #   # result = heatmap * 0.4 + img * 0.5
    #   plt.axis('off')
    #   plt.subplot(2,4,ind)
    #   ind+=1
    #   plt.imshow(origin_img)
    #   plt.imshow(heatmap, alpha=0.5)
    #   # plt.show()
    #   # plt.close()
    # plt.axis('off')
    # plt.savefig('star_vector.jpg')
    # plt.show()
    # plt.close()
    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0
    for part in range(0, 16):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 10  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)
                    
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    if norm == 0:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts)
                    else:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*origin_img.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 17))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k])

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 14:
                    row = -1 * np.ones(17)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.45:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # draw points
    canvas = cv2.imread(input_path)

    for i in range(14):
      for n in range(len(subset)):
          index = subset[n][np.array(limbSeq[i])]
          if -1 in index:
              continue
          Y = candidate[index.astype(int), 0]
          X = candidate[index.astype(int), 1]
          cv2.line(canvas, (int(Y[1]), int(X[1])), (int(Y[0]), int(X[0])), [0, 0, 120],10,)

    for n in range(len(subset)):
        for i in range(15):
            index = subset[n, i]
            if index < 0:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            cv2.circle(canvas, (int(mY), int(mX)), 3, [255, 255, 255], thickness=-1, lineType=4)

    return canvas


if __name__ == '__main__':

    input_image = './images/liuxiang.jpg'
    output = "./images/result.jpg"
    weight_name = './legacy/dilated_pose_done.pth'
    # load model
    model = construct_model(weight_name)
    tic = time.time()
    print('start processing...')

    # generate image with body parts
    canvas = process(model, input_image)

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)
