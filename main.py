import warnings
from train import *
from preData import *


if __name__ == '__main__':
    args = parse_args()
    print(args)
    resList = []
    repeat = 10000
    warnings.filterwarnings("ignore")

    for i in range(repeat):
        # ******************5-cv训练代码******************
        averages = fold_valid(args)
        resList.append(averages)
        print(averages)


    print("finish")