import numpy as np
import scipy.signal as signal


# Seed Halftone
# img: Continuous-Tone Image
# d: Gaussian standard deviation
# g: The absorptance(gray level) of Seed halftone 0-255/255
def seed_halftone(img, d, g):
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    CPP = np.zeros((13, 13))
    HalfCPPSize = 6
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))

    im = np.array(img, dtype=np.float64)
    rows, cols = im.shape

    f = np.full((rows, cols), g, dtype=np.float64)
    dst = np.random.rand(rows, cols) > g
    dst = np.where(dst, 0.0, 1.0)

    Err = dst - f
    CPE = signal.correlate2d(Err, CPP, mode='full')

    # Only swap operation to generate seed halftone
    while True:
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
                        EPS = (a0 * a0 + a1 * a1) * CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CPE[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CPE[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y
                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CPE[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CPE[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1
        if CountB == 0:
            break
    return dst


# Initial Halftone
# img: Continuous-Tone Image
# dst: Seed halftone
def initialHalftone(img, dst):
    im = np.array(img, dtype=np.float64)
    rows, cols = im.shape
    size = 8
    rowsi = rows // size
    colsi = cols // size

    for i in range(rowsi):
        for j in range(colsi):
            countg = 0
            gray_value = 0.0
            for y in range(size):
                for x in range(size):
                    gray_value += im[i * size + y, j * size + x] / 255
            bnum = round(gray_value)
            for y in range(size):
                for x in range(size):
                    if dst[i * size + y, j * size + x] == 1:
                        dst[i * size + y, j * size + x] = 0
            while bnum > 0:
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                if dst[i * size + y, j * size + x] == 0:
                    dst[i * size + y, j * size + x] = 1
                    bnum -= 1
                    countg += 1
    return dst


# CLU-DBS
# img: Continuous-Tone Image
# d: Gaussian standard deviation that is used to initialize total squared error
# d1: Gaussian standard deviation that is used to update total squared error
def clu_dbs(img, dst, d, d1, g0):
    # Build the gaussian filter Cpp and Cpp1
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))
    GF1 = np.zeros((fs, fs))
    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            c1 = (k ** 2 + l ** 2) / (2 * d1 ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
            GF1[k + gaulen, l + gaulen] = np.exp(-c1) / (2 * np.pi * d1 ** 2)
    CPP, CPP1 = np.zeros((13, 13)), np.zeros((13, 13))
    HalfCPPSize = 6
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))
    CPP1 = CPP1 + (signal.convolve2d(GF1, GF1, mode='full', boundary='fill', fillvalue=0))

    im = np.array(img, dtype=np.float64)
    rows, cols = im.shape
    f = im / 255.0

    Err0 = g0 - f
    D_CPE0 = signal.correlate2d(Err0, CPP - CPP1, mode='full')

    # Initial halftone pattern from seed halftone
    dst = initialHalftone(im, dst)

    Err = dst - f
    CPE = signal.correlate2d(Err, CPP, mode='full')

    while True:
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
                        # The changes of cost metric after trial toggle or swap operation
                        EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CPE[HalfCPPSize + i, HalfCPPSize + j] - \
                              2 * a0 * D_CPE0[HalfCPPSize + i, HalfCPPSize + j] + \
                              2 * a0 * a1 * a0 * CPE[HalfCPPSize + i + y, HalfCPPSize + j + x] - \
                              2 * a0 * a1 * a0 * D_CPE0[HalfCPPSize + i + y, HalfCPPSize + j + x]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y
                # If delta cost metric < 0 then accept trial toggle or swap operation and update Cpe
                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CPE[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CPE[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1
        if CountB == 0:
            break
    return dst


# MP-CLU-DBS
# img: Continuous-Tone Image
# d: Gaussian standard deviation that is used to initialize total squared error
# d1: Gaussian standard deviation that is used to update total squared error
# p: Pass
# g0: Initial halftone pattern for being optimized
def mp_clu_dbs(img, dst, d, d1, p, g0):
    for i in range(p):
        dst = clu_dbs(img, dst, d, d1, g0)
    return dst


# MS-MP-CLU-DBS
# img: Continuous-Tone Image
# d: Gaussian standard deviation that is used to initialize total squared error
# d1: Gaussian standard deviation that is used to update total squared error
# s: Stage
# p: Pass
# g0: Initial halftone pattern for being optimized
def ms_mp_clu_dbs(img, dst, d, d1, s, p, g0):
    for i in range(s):
        imgi = img / s * (i + 1)
        dst = mp_clu_dbs(imgi, dst, d, d1, p, g0)
        g0 = dst
    return dst


if __name__ == '__main__':

    img = np.full((128, 128), 127, dtype=np.uint8)
    dst = seed_halftone(img, 1.3, 7.57 / 255)
    g0 = dst

    dst = ms_mp_clu_dbs(img, dst, 1.3, 1.7, 5, 10, g0)

