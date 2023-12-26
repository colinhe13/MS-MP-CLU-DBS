import numpy as np
import cv2
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# 产生种子半色调图像的函数
# img: 原图像
# d: 滤波器参数
# g: 种子半色调的灰度级数 0-255/255
def get_seed(img, d, g):
    # fs = 7
    # gaulen = int((fs - 1) / 2)
    # GF = np.zeros((fs, fs))
    #
    # for k in range(-gaulen, gaulen + 1):
    #     for l in range(-gaulen, gaulen + 1):
    #         c = (k ** 2 + l ** 2) / (2 * d ** 2)
    #         GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
    #
    # CPP = np.zeros((13, 13))
    #
    # CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))

    HalfCPPSize = 12
    CPP = cal_cpp(d)

    im = np.array(img, dtype=np.float64)

    rows, cols = im.shape

    # imr = np.full((rows, cols), 1 - g, dtype=np.float64)
    imr = np.full((rows, cols), g, dtype=np.float64)

    dst = np.random.rand(rows, cols) > g
    # dst = np.where(dst, 1.0, 0.0)
    dst = np.where(dst, 0.0, 1.0)

    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')
    ESP_MIN = 0

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):  # -1 0 1
                    if not (0 <= i + y < rows):  # i=0 y=0/1; i=rows-1 y=-1/0; i=rows y=-1
                        continue
                    for x in range(-1, 2):  # -1 0 1
                        if not (0 <= j + x < cols):  # j=0 x=0/1; j=cols-1 x=-1/0; j=cols x=-1
                            continue
                        if y == 0 and x == 0:
                            continue
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        # 源码EPS
                        EPS = (a0 * a0 + a1 * a1) * CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y
                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1
        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break
        # break
        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)
    print("------- seed generated -------")
    return dst


# 从种子半色调生成初始化半色调
# 每个8*8区域内计算原始图像平均灰度值，然后通过在种子相同位置的随机翻转匹配灰度
def initialHalftone(img, dst):
    # cv2.imshow("before initial", dst)
    # cv2.waitKey(0)
    im = np.array(img, dtype=np.float64)
    rows, cols = im.shape
    size = 8
    rowsi = rows // size
    colsi = cols // size

    # 0->white 1->black
    for i in range(rowsi):
        for j in range(colsi):
            countg = 0
            gray_value = 0.0
            for y in range(size):
                for x in range(size):
                    gray_value += im[i * size + y, j * size + x] / 255

            # gray_value = 1 - (gray_value / size ** 2)
            # gray_value = gray_value / size ** 2

            # 0-->white  1-->black
            # bnum = round(size ** 2 * gray_value)
            bnum = round(gray_value)
            for y in range(size):
                for x in range(size):
                    # if dst[i * size + y, j * size + x] == 0:
                    #     dst[i * size + y, j * size + x] = 1
                    if dst[i * size + y, j * size + x] == 1:
                        dst[i * size + y, j * size + x] = 0
                    # if dst[i * size + y, j * size + x] == 1:
                    #     bnum -= 1
            while bnum > 0:
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                if dst[i * size + y, j * size + x] == 0:
                    dst[i * size + y, j * size + x] = 1
                    bnum -= 1
                    countg += 1
            # while bnum < 0:
            #     x = np.random.randint(0, size)
            #     y = np.random.randint(0, size)
            #     if dst[i * size + y, j * size + x] == 1:
            #         dst[i * size + y, j * size + x] = 0
            #         bnum += 1
            #     countg += 1
            # print("countg = ", countg)
    # cv2.imshow("after initial", dst)
    # cv2.waitKey(0)
    return dst


# dbs函数
# img: 原图像
# d: 滤波器参数
def dbs(img, d):
    # fs = 7
    # gaulen = int((fs - 1) / 2)
    # GF = np.zeros((fs, fs))
    #
    # for k in range(-gaulen, gaulen + 1):
    #     for l in range(-gaulen, gaulen + 1):
    #         c = (k ** 2 + l ** 2) / (2 * d ** 2)
    #         GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
    #
    # CPP = np.zeros((13, 13))
    # HalfCPPSize = 6
    # CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))

    HalfCPPSize = 12
    CPP = cal_cpp(d)

    im = np.array(img, dtype=np.float64)
    # print(im)

    rows, cols = im.shape

    dst = np.random.rand(rows, cols) > 0.5
    dst = np.where(dst, 1.0, 0.0)

    imr = im / 255.0
    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')
    ESP_MIN = 0

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):  # -1 0 1
                    if not (0 <= i + y < rows):  # i=0 y=0/1; i=rows-1 y=-1/0; i=rows y=-1
                        continue
                    for x in range(-1, 2):  # -1 0 1
                        if not (0 <= j + x < cols):  # j=0 x=0/1; j=cols-1 x=-1/0; j=cols x=-1
                            continue
                        if y == 0 and x == 0:
                            if dst[i, j] == 1:
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        # 源码EPS
                        EPS = (a0 * a0 + a1 * a1) * CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1

        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break
        # break
        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)

    return dst


# clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
def clu_dbs(img, dst, d, d1, dst0):
    HalfCPPSize = 12
    # HalfCPPSize = 24
    CPP = cal_cpp(d)
    CPP1 = cal_cpp(d1)

    im = np.array(img, dtype=np.float64)

    rows, cols = im.shape

    imr = im / 255.0

    gray_value = 0.0
    for y in range(rows):
        for x in range(cols):
            gray_value += dst[y, x]
    gray_value = gray_value / (rows * cols)

    # TODO 确定Err0的计算方式
    # Err0 = dst - np.full((rows, cols), gray_value, dtype=np.float64)
    Err0 = dst - imr
    # Err0 = dst0 - np.full((rows, cols), gray_value, dtype=np.float64)

    D_CEP0 = signal.correlate2d(Err0, CPP - CPP1, mode='full')
    # D_CEP0 = signal.convolve2d(Err0, CPP, mode='full') - signal.convolve2d(Err0, CPP1, mode='full')

    dst = initialHalftone(im, dst)

    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')

    # # 绘制热度图
    # plt.imshow(CEP, cmap='rainbow', vmin=-0.1, vmax=0.1)  # viridis 是一种常用的热度图配色方案
    # plt.colorbar()  # 显示颜色条，对应数值和颜色的关系
    # # 添加标题
    # plt.title('CEP')
    # # 显示图形
    # plt.show()

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):
                    if not (0 <= i + y < rows):
                        continue
                    for x in range(-1, 2):
                        if not (0 <= j + x < cols):
                            continue
                        if y == 0 and x == 0:
                            if dst[i, j] == 1:
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        # EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                        #       2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                        #       2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                        #       2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                              2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j] + \
                              2 * a0 * a1 * a0 * CEP[HalfCPPSize + i + y, HalfCPPSize + j + x] - \
                              2 * a0 * a1 * a0 * D_CEP0[HalfCPPSize + i + y, HalfCPPSize + j + x]

                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1

        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break

        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)

    # cv2.imshow("after cludbs", dst)
    # cv2.waitKey(0)

    return dst


# mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# p: pass(迭代次数)
def mp_clu_dbs(img, dst, d, d1, p, dst0):
    CountInit = 0
    for i in range(p):
        print("------- pass " + str(i + 1) + " / " + str(p) + " -------")
        # if CountInit == 0:
        #     CountInit += 0.25
        # elif CountInit == 0.25:
        #     CountInit += 0.25
        # else:
        #     CountInit = 1

        # if CountInit < 1:
        #     CountInit += 0.1
        # else:
        #     CountInit = 1

        CountInit = 1

        imgi = img * CountInit
        dst = clu_dbs(imgi, dst, d, d1, dst0)
        print("gray_value = " + str(np.count_nonzero(dst) / 256))
        # cv2.imshow("aftermp"+str(i), dst)

    # cv2.waitKey(0)
    return dst


# ms_mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# s: stage(阶段数)
# p: pass(迭代次数)
def ms_mp_clu_dbs(img, dst, d, d1, s, p, dst0):
    for i in range(s):
        print("======= stage " + str(i + 1) + " / " + str(p) + " =======")
        imgi = img / s * (i + 1)
        dst = mp_clu_dbs(imgi, dst, d, d1, p, dst0)
        dst0 = dst
        # cv2.imshow("afterms", dst)
        # print("gray_value = " + str(np.count_nonzero(dst) / 127 ** 2))
        # cv2.waitKey(0)
    return dst


# 生成ms_mp_clu_dbs网屏的函数
# height: 网屏高度
# width: 网屏宽度
def screen_msmpcludbs(height, width, g, d, d1, s, p):
    screen = np.full((height, width), 0, dtype=np.uint8)

    # 首先生成灰度级127/255的ms_mp_clu_dbs半色调图像并存入screen数组
    mid_img = np.full((height, width), 127, dtype=np.uint8)
    dst = get_seed(screen, d, g)
    dst = ms_mp_clu_dbs(mid_img, dst, d, d1, s, p)
    # np.save("output/gray_values_127.npy", dst)
    for i in range(height):
        for j in range(width):
            if dst[i, j] == 0:
                screen[i, j] = 127
            else:
                screen[i, j] = 255

    # 调整灰度级127/255图像中黑色像素的数量
    screen = fill_screen(screen, 127, 127, d)

    # 生成灰度级126/255到0/255的ms_mp_clu_dbs半色调图像
    for k in range(126, -1, -1):
        screen = fill_screen(screen, k + 1, k, d)

    # 生成灰度级128/255到255/255的ms_mp_clu_dbs半色调图像
    for k in range(128, 256):
        screen = fill_screen(screen, k - 1, k, d)

    return screen


# 根据输入的灰度级数在网屏矩阵中填入对应级数的函数
# screen: 网屏矩阵(不是图像)
# cur_values: 当前灰度级数
# target_values: 目标灰度级数
# d: 滤波器参数
def fill_screen(screen, cur_values, target_values, d, d1):
    height, width = screen.shape
    CPP = cal_cpp(d)
    CPP1 = cal_cpp(d1)
    HalfCPPSize = 12

    im = np.full((height, width), target_values, dtype=np.float64)
    imr = im / 255.0

    dst = screen <= cur_values
    dst = np.where(dst, 1.0, 0.0)
    Err0 = dst - np.full((height, width), target_values / 255.0, dtype=np.float64)
    D_CEP0 = signal.correlate2d(Err0, CPP - CPP1, mode='full')

    # 情况1 目标值=当前值：重新检查网屏中灰度值是否符合要求
    if target_values == cur_values:
        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP1, mode='full')
        num_black = np.count_nonzero(dst)
        num_target = round(target_values / 255 * height * width)
        while num_black != num_target:
            # 如果黑色像素数量过多
            if num_black > num_target:
                # 找到使误差降低最多或增加最少的黑色像素1，并将其修改为白色0
                EPS_MIN, EPS = 0, 0
                a0c, Cpx, Cpy = 0, 0, 0
                flag = True
                # 先找EPS降低
                if flag:
                    flag = False
                    EPS_MIN = 0
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 1:
                                a0 = -1
                                EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                      2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    flag = True
                    print("working on i = ", Cpy, "  j = ", Cpx)
                    print("most decrease EPS = ", EPS_MIN)
                    dst[Cpy, Cpx] += a0c
                    num_black -= 1
                    screen[Cpy, Cpx] = 255.0
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        flag = False
                        break
                # 如果数量还是不够，再找EPS增加
                if not flag:
                    EPS_MIN = 100000
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 1:
                                a0 = -1
                                EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                      2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                    print("working on i = ", Cpy, "  j = ", Cpx)
                    print("min increase EPS = ", EPS_MIN)
                    dst[Cpy, Cpx] += a0c
                    num_black -= 1
                    screen[Cpy, Cpx] = 255.0
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        flag = False
                        break
            else:
                # 找到使误差降低最多或增加最少的白色像素0，并将其修改为黑色1
                EPS_MIN, EPS = 0, 0
                a0c, Cpx, Cpy = 0, 0, 0
                flag = True
                # 先找EPS降低
                if flag:
                    flag = False
                    EPS_MIN = 0
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 0:
                                a0 = 1
                                EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                      2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    flag = True
                    print("working on i = ", Cpy, "  j = ", Cpx)
                    print("min increase EPS = ", EPS_MIN)
                    dst[Cpy, Cpx] += a0c
                    num_black += 1
                    screen[Cpy, Cpx] = target_values
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        flag = False
                        break
                # 如果数量还是不够，再找EPS增加
                if not flag:
                    EPS_MIN = 100000
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 0:
                                a0 = 1
                                EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                      2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                    print("working on i = ", Cpy, "  j = ", Cpx)
                    print("min increase EPS = ", EPS_MIN)
                    dst[Cpy, Cpx] += a0c
                    num_black += 1
                    screen[Cpy, Cpx] = target_values
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        flag = False
                        break
        return screen

    # 情况2 目标值<当前值：找到并修改使误差降低最多或增加最少的黑色像素
    elif target_values < cur_values:

        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP1, mode='full')

        num_black = np.count_nonzero(dst)
        num_target = round(target_values / 255 * height * width)

        while num_black != num_target:
            # 找到使误差降低最多或增加最少的黑色像素0，并将其修改为白色1
            EPS_MIN, EPS = 0, 0
            a0c, Cpx, Cpy = 0, 0, 0
            flag = True
            # 先找EPS降低
            while flag:
                flag = False
                EPS_MIN = 0
                for i in range(height):
                    for j in range(width):
                        if dst[i, j] == 1:
                            a0 = -1
                            EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                  2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                  2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                            if EPS_MIN > EPS:
                                EPS_MIN = EPS
                                a0c = a0
                                Cpx = j
                                Cpy = i
                                flag = True
                # 如果没有能使EPS降低的像素，跳出循环，开始找EPS增加最少的像素
                if not flag:
                    flag = True
                    break
                print("working on i = ", Cpy, "  j = ", Cpx)
                print("most decrease EPS = ", EPS_MIN)
                dst[Cpy, Cpx] += a0c
                # screen[Cpy, Cpx] = target_values
                screen[Cpy, Cpx] = 255.0
                num_black -= 1
                for y in range(-HalfCPPSize, HalfCPPSize + 1):
                    for x in range(-HalfCPPSize, HalfCPPSize + 1):
                        CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                            y + HalfCPPSize, x + HalfCPPSize]
                if num_black == num_target:
                    flag = False
            # 如果数量还是不够，再找EPS增加
            while flag:
                flag = False
                EPS_MIN = 100000
                for i in range(height):
                    for j in range(width):
                        if dst[i, j] == 1:
                            a0 = -1
                            EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                                  2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                                  2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                            if EPS_MIN > EPS:
                                EPS_MIN = EPS
                                a0c = a0
                                Cpx = j
                                Cpy = i
                                flag = True
                print("working on i = ", Cpy, "  j = ", Cpx)
                print("min increase EPS = ", EPS_MIN)
                dst[Cpy, Cpx] += a0c
                # screen[Cpy, Cpx] = target_values
                screen[Cpy, Cpx] = 255.0
                num_black -= 1
                for y in range(-HalfCPPSize, HalfCPPSize + 1):
                    for x in range(-HalfCPPSize, HalfCPPSize + 1):
                        CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                            y + HalfCPPSize, x + HalfCPPSize]
                if num_black == num_target:
                    flag = False
                    break
        return screen

    # 情况3 目标值>当前值：找到并修改使误差降低最多或增加最少的白色像素
    else:
        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP1, mode='full')

        num_black = np.count_nonzero(dst)
        num_target = round(target_values / 255 * height * width)

        # 找到使误差降低最多或增加最少的白色像素1，并将其修改为黑色0
        EPS_MIN, EPS = 0, 0
        a0c, Cpx, Cpy = 0, 0, 0
        flag = True
        # 先找EPS降低
        while flag:
            flag = False
            EPS_MIN = 0
            for i in range(height):
                for j in range(width):
                    if dst[i, j] == 0:
                        a0 = 1
                        EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                              2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            Cpx = j
                            Cpy = i
                            flag = True
            # 如果没有能使EPS降低的像素，跳出循环，开始找EPS增加最少的像素
            if not flag:
                flag = True
                break
            print("working on i = ", Cpy, "  j = ", Cpx)
            print("most decrease EPS = ", EPS_MIN)
            dst[Cpy, Cpx] += a0c
            num_black += 1
            screen[Cpy, Cpx] = target_values
            for y in range(-HalfCPPSize, HalfCPPSize + 1):
                for x in range(-HalfCPPSize, HalfCPPSize + 1):
                    CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                        y + HalfCPPSize, x + HalfCPPSize]
            if num_black == num_target:
                flag = False
        # 如果数量还是不够，再找EPS增加
        while flag:
            flag = False
            EPS_MIN = 100000
            for i in range(height):
                for j in range(width):
                    if dst[i, j] == 0:
                        a0 = 1
                        EPS = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                              2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            Cpx = j
                            Cpy = i
                            flag = True
            if flag:
                print("working on i = ", Cpy, "  j = ", Cpx)
                print("min increase EPS = ", EPS_MIN)
                dst[Cpy, Cpx] += a0c
                num_black += 1
                screen[Cpy, Cpx] = target_values
                for y in range(-HalfCPPSize, HalfCPPSize + 1):
                    for x in range(-HalfCPPSize, HalfCPPSize + 1):
                        CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                            y + HalfCPPSize, x + HalfCPPSize]
            if num_black == num_target:
                flag = False
                break
        return screen


# 计算CPP的函数(CPP在整个计算过程中不改变)
def cal_cpp(d):
    fs = 13
    # fs = 25
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            # 计算到中心的距离的二范数
            dist = np.linalg.norm([k, l])
            c = dist ** 2 / (2 * d ** 2)
            # c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    # CPP = np.zeros((13, 13))
    CPP = np.zeros((25, 25))
    # CPP = np.zeros((49, 49))
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))
    # CPP = signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0)
    return CPP


# 灰度匹配(将当前图像灰度精确为目标灰度)
def gray_matching(path, d, d1, gray_value):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    HalfCPPSize = 12
    CPP = cal_cpp(d)
    CPP1 = cal_cpp(d1)
    # 黑白翻转二值化 0=white 1=black
    dst = img > 127
    dst = np.where(dst, 0.0, 1.0)
    imr = dst
    rows, cols = dst.shape
    # 计算需要调整的像素数量
    n = np.count_nonzero(dst) - round(gray_value / 255 * height * width)
    # 进行切换操作
    Err0 = dst - np.full((rows, cols), gray_value, dtype=np.float64)
    D_CEP0 = signal.correlate2d(Err0, CPP - CPP1, mode='full')
    # 初始化操作

    Err = dst - imr
    CEP = signal.correlate2d(Err, CPP, mode='full')

    if n == 0:
        return
    if n < 0:  # 需要增加黑色像素
        while n != 0:
            CountB = 0
            for i in range(rows):
                for j in range(cols):
                    a0c = 0
                    a1c = 0
                    Cpx = 0
                    Cpy = 0
                    EPS_MIN = 0

                    a0 = 1
                    a1 = 0
                    EPS = CPP1[HalfCPPSize, HalfCPPSize] + \
                          2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                          2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]


    else:  # 需要减少黑色像素
        return

        # cv2.imshow("dst", dst * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


# 特定翻转操作(找到并修改使误差降低最多或增加最少的黑色/白色像素)
def particular_toggle():
    return


def get_dataset():
    for i in range(1, 10001):
        img = cv2.imread("F:/Ander/Downloads/BossBase-1.01-cover/" + str(i) + ".pgm", cv2.IMREAD_GRAYSCALE)
        # 反色
        for k in range(img.shape[0]):
            for j in range(img.shape[1]):
                p = img[k, j]
                img[k, j] = 255 - p
        dst = np.load("output/seed/seed_7_size_512_bossbase.npy")
        dst0 = dst
        dst = mp_clu_dbs(img, dst, 1.4, 1.7, 5, dst0)
        save_filepath = "output/bb/14_17_5/" + str(i) + "_14_17_5.pgm"
        print("------------------------------------------------------------------" + "\n"
        # "       Program Completed       " + "\n"
                                                                                     "       Process     " + str(
            i) + "/10000" + "\n"
                            "       Result has Saved to " + save_filepath + "\n"
                                                                            "------------------------------------------------------------------")
        dst = np.where(dst, 0.0, 1.0)
        cv2.imwrite(save_filepath, dst * 255)


def generate_gray_127(size):
    # 生成大小为size的灰度级127/255的图像
    img = np.full((size, size), 127, dtype=np.uint8)
    return img


def m():
    # 生成256*256的灰度级127/255的图像
    # img = np.full((256, 256), 127, dtype=np.uint8)
    img = np.full((512, 512), 127, dtype=np.uint8)
    # img = np.full((128, 128), 64, dtype=np.uint8)

    dst = get_seed(img, 1.3, 7.57 / 255)
    # dst = get_seed(img, 1.3, 25.5 / 255)
    # dst = np.load("output/seed/seed_7_size_512_bossbase.npy")
    dst0 = dst

    # dst = clu_dbs(img, dst, 1.3, 1.7)
    # dst = mp_clu_dbs(img, dst, 1.3, 1.7, 10, dst0)
    dst = ms_mp_clu_dbs(img, dst, 1.3, 1.7, 5, 5, dst0)

    save_filepath = "output/231208/msmpcludbs_d13_17_s5_p10_gray127_size512.png"

    print("##################################################################" + "\n"
                                                                                 "       Program Completed       " + "\n"
                                                                                                                     "       Result has Saved to " + save_filepath + "\n"
                                                                                                                                                                     "##################################################################")

    dst = np.where(dst, 0.0, 1.0)
    cv2.imwrite(save_filepath, dst * 255)

    # cv2.imshow("image", dst)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()


def view_cpp():
    CPP = cal_cpp(1.3)
    CPP1 = cal_cpp(1.7)

    # 获取矩阵的维度
    rows, cols = CPP.shape
    # 创建行和列的坐标
    x = np.arange(cols)
    y = np.arange(rows)
    # 创建坐标网格
    x, y = np.meshgrid(x, y)
    # 创建三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维曲面图
    ax.plot_surface(x, y, CPP, cmap='rainbow')
    # 绘制三维曲面图，不填充颜色
    # ax.plot_trisurf(x.flatten(), y.flatten(), CPP.flatten(), facecolors='none', cmap='rainbow', edgecolor='b')
    # 添加标题
    # ax.set_title('Cpp1_r')
    # 显示图形
    plt.show()

def get_screen():
    screen = cv2.imread("output/231208/screen/screen_gray127.png", cv2.IMREAD_GRAYSCALE)

    # screen = np.where(screen < 1, 1, 255)
    # screen = fill_screen(screen, 1, 0, 1.3, 1.7)
    # screen = np.where(screen != 255, 0.0, 255.0)
    # save_filepath = "output/231208/screen/screen_gray0"
    # cv2.imshow("screen", screen)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 生成灰度级126至0的screen
    # for k in range(126, -1, -1):
    #     screen = np.where(screen < k+1, k+1, 255)
    #     screen = fill_screen(screen, k+1, k, 1.3, 1.7)
    #     screen = np.where(screen != 255, 0.0, 255.0)
    #     save_filepath = "output/231208/screen/screen_gray" + str(k)
    #     cv2.imwrite(save_filepath + ".png", screen)

    # 生成灰度级128至255的screen
    # for k in range(128, 256):
    #     screen = np.where(screen < k, k-1, 255)
    #     screen = fill_screen(screen, k-1, k, 1.3, 1.7)
    #     screen = np.where(screen != 255, 0.0, 255.0)
    #     save_filepath = "output/231208/screen/screen_gray" + str(k)
    #     cv2.imwrite(save_filepath + ".png", screen)

    # 读取图像，生成screen矩阵
    matrx = np.zeros(screen.shape, dtype=np.float64)
    for k in range(0, 256):
        img = cv2.imread("output/231208/screen/screen_gray" + str(k) + ".png", cv2.IMREAD_GRAYSCALE)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] == 0 and matrx[i, j] == 0:
                    matrx[i, j] = k
    np.save("output/231208/screen/screen_matrx.npy", matrx)


    # screen = np.where(screen < 127, 127, 255)
    # screen = fill_screen(screen, 127, 128, 1.3, 1.7)
    # screen = np.where(screen != 255, 0.0, 255.0)
    # os.mkdir("output/231208/screen")
    # save_filepath = "output/231208/screen/screen_gray128"
    # cv2.imwrite(save_filepath + ".png", screen)
    # np.save(save_filepath + ".npy", screen)

def send_notification():
    notification_script = """
    display notification "程序运行结束，请查看" with title "Pycharm" sound name "Glass"
    """
    os.system(f"osascript -e '{notification_script}'")

m()
send_notification()
# view_cpp()
# gray_matching("output/231208/msmpcludbs_d13_17_s5_p10_gray127.png", 1.3, 127)
# get_screen()

# cv2.imshow("screen", screen)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cpp = cal_cpp(1.3)
# print(cpp)
