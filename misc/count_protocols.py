import numpy as np
import scipy.special
import math
np.random.seed(0)

d = np.full((100, 100), 0.0)
t = np.full(100, 0.0)

class DataFrame:
    def __init__(self, leftSum, occ, maximum):
        self.leftSum = leftSum
        self.occ = occ
        self.maximum = maximum


def calculate_d(end):
    for i in range(4, end):
        check = [DataFrame(i, 0, 4)]
        while check:
            el = check[-1]
            del check[-1]

            if el.leftSum == 0:
                d[i][el.occ] += 1
                continue

            for j in range(el.maximum, i + 1):
                if el.leftSum - j >= 0:
                    tmp = DataFrame(el.leftSum - j, el.occ + 1, j)
                    check.append(tmp)
    print("Calculated d")

def gen_d(end):
    calculate_d(end)

    for i in range(end):
        sum = 0
        print("i: {} \t".format(i), end='')
        for j in range(30):
            sum += d[i][j]
            if (i < 30):
                print(d[i][j], end=' ')
        print(' sum: {}'.format(sum))



def calculate_t(end):
    for i in range(1, end):
        # print("i: {} \t".format(i), end='')
        sum = 0.0
        tmp = scipy.special.comb(719 - 2 * (i - 1), i - 1)
        sum += tmp
        # print(tmp, end=' ')
        tmp = scipy.special.comb(718 - 2 * (i - 1), i - 1)
        sum += tmp
        # print(tmp, end=' ')
        tmp = scipy.special.comb(720 - 2 * i, i)
        sum += tmp
        t[i] = sum
        # print(tmp, end=' ')
        # print(sum)
    print("Calculated t")


def calculate_all():
    calculate_d(81)
    calculate_t(21)
    sum = 0
    for i in range(4, 81):
        for j in range(1, 20):
            sum += d[i][j] * t[j] * math.factorial(j)
    print(sum)
    print("{:e}".format(sum))

if __name__ == '__main__':
    # gen_d(81)
    # calculate_t(end)
    calculate_all()

