import bt
import sys
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


'''
基本上金電價差應該是低相關  拿來做發散
所以會把電子/金融當作指標
然後隨便取個適合的MA

發現最新值>MA，開始向上發散的時候
多2電子 空3金融

反之最新值<MA，開始向下發散的時候
空2電子  多3金融
'''


ew = 2 
fw = 3
freq = '1D'
maPeriod = 20
start_date = '1990-01-01'


dfList = []
for file in glob.glob(sys.path[0] + '/dataset/*XF1-分鐘-成交價.csv'):
    tmp = pd.read_csv(file, index_col=0)[['Value_Close']]
    tmp.rename(columns={'Value_Close': file.split('-')[0].split('/')[-1]}, inplace=True)
    tmp.index = pd.to_datetime(tmp.index)
    dfList.append(tmp)
df = pd.concat(dfList, axis=1)
df = df.resample(freq).last()
df.fillna(method='ffill', inplace=True)
print(df.tail(3))


df['Ind'] = df['EXF1'] / df['FXF1']
df['Ind_SMA'] = df['Ind'].rolling(maPeriod).mean()
weights = df.copy(deep=True)
weights.dropna(inplace=True)


weights['FXF1'] = weights['Ind'] > weights['Ind_SMA']
weights['FXF1'] = weights['FXF1'].replace({True: -1*fw, False: fw})
weights['EXF1'] = weights['Ind'] > weights['Ind_SMA']
weights['EXF1'] = weights['EXF1'].replace({True: ew, False: -1*ew})
weights.drop(['Ind', 'Ind_SMA'], axis=1, inplace=True)


s = bt.Strategy('ExF', [
    bt.algos.SelectAll(),
    bt.algos.WeighTarget(weights=weights),
    bt.algos.Rebalance()
])
t = bt.Backtest(s, df[['FXF1', 'EXF1']].loc[df.index > start_date])
res = bt.run(t)
print('\n', res.display())


now = datetime.datetime.now()
file_dst = sys.path[0] + '/result/ExF-%s.pkl' % now
with open(file_dst, 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
file_dst = sys.path[0] + '/trade/ExF-%s.pkl' % now
with open(file_dst, 'wb') as f:
    pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)