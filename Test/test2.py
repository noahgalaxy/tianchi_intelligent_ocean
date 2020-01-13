'''
class he:
    @classmethod
    def a(cls, a, b):
        print('a', a + b)

    @classmethod
    def b(cls, a, b):
        print('b', a + b)

he.a(1, 2)
'''

import pandas as pd, numpy as np
from matplotlib import pyplot as plt

csv_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102\3.csv'
df = pd.read_csv(csv_path)
print(df.head(5))
'''
plt.hist(df['速度'], bins= 10, label= 'speed', histtype= 'stepfilled')
plt.show()
print(df['速度'].mean())
'''

'''
fig, ax = plt.subplots(figsize=(8,6),ncols=2,nrows= 2)
ax[0][0].hist(df['速度'] / df['速度'].mean(), bins= 10, histtype= 'stepfilled')
ax[0][0].set_title('X')
ax[0][0].legend()
ax[0][1].hist(df['y'] / df['y'].mean(), bins= 10)
ax[0][1].set_title('Y')
ax[0][1].legend()
ax[1][0].hist(df['速度'] / df['速度'].mean(), bins= 10)
ax[1][0].set_title('Speed')
ax[1][0].legend()
ax[1][1].hist(df['方向'] / df['方向'].mean(), bins= 10)
ax[1][1].set_title('Direction')
ax[1][1].legend()
plt.show()
'''

id_list  = np.random.choice(range(7000), size= (100))
image_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Image'
for i in id_list:
    csv_path = r'C:\Users\Nolan\Desktop\py\Inteligent_Ocean\Dataset\hy_round1_train_20200102'
    csv_path = csv_path + '\\' + f'{i}.csv'
    df = pd.read_csv(csv_path)
    # 不加平均
    title = df['type'].unique()[-1]
    print(title, end= ' ')
    print(i)
    fig, ax = plt.subplots(figsize=(8,6),ncols=2,nrows= 2)
    # 可以选展开成一维的，这样下面就直接一维数组选择就行了
    ax.flaten()
    if title == '拖网':
        ti = 'tuowang'
    elif title == '围网':
        ti = 'weiwang'
    else:
        ti = 'ciwang'

    fig.suptitle(ti)
    ax[0][0].hist(df['x'], bins= 40, histtype= 'stepfilled')
    ax[0][0].set_title('X')

    ax[0][1].hist(df['y'] , bins= 40)
    ax[0][1].set_title('Y')

    ax[1][0].hist(df['速度'], bins= 40)
    ax[1][0].set_title('Speed')

    ax[1][1].hist(df['方向'], bins= 40)
    ax[1][1].set_title('Direction')

    #plt.legend()

    #plt.show()
    #fig.pause(1)
    fig.savefig(image_path + '\\' +f'{i}_' + title + '.jpg')
    plt.close()
