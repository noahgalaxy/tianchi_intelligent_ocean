import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
import math

font_set = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=12)

csv_root_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
csv_suffix_path = r'.csv'
csv_paths = []
for i in [3, 4, 6, 7, 8, 10, 14, 200, 45, 400]:
    csv_paths.append(csv_root_path + '\\' + str(i) +csv_suffix_path )
def _3d():
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(df.head())
        print(df.index)
        #plt.figure(figsize=(20, 20))
        x = df['x']
        y = df['y']
        z = df['time'].to_numpy()
        z = np.arange(len(z))
        #z = z[np.newaxis, :]
        print('z shape' ,z.shape)
        fig = plt.figure(figsize= (20, 20))
        num = f'12{i+1}'
        ax1 = fig.add_subplot(num, projection='3d')
        ax1 = Axes3D(fig)  # 创建3D图的2种方式，第一种通过Axes3D将图片从二维变成三维，第二种通过在add_subplot(111,projection='3d'）将子图坐标修改成三维
        ax1.plot(x,y,z,'bo--')  # 参数与二维折现图不同的在于多了一个Z轴的数据

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('time')
    plt.show()

def _2d_plot():
    plt.figure(figsize= (20, 15))
    num = len(csv_paths)
    row_num = math.ceil(num / 4)
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(df.head())
        print(df.index)
        #plt.figure(figsize=(20, 20))
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        title = df['type'].unique()[0]
        print('title', title)
        #plt.tight_layout()
        plt.subplot(row_num, 4, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        plt.plot(x, y)
        plt.plot(x[0], y[0], color= 'red', marker= 'o')
        plt.plot(x[-1], y[-1], color='blue', marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title, fontproperties=font_set)

    plt.show()
_2d_plot()