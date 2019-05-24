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
另外一種就是等市值去製造新商品
電子*4000*2-金融*1000*3+常數(讓數值變正)
然後無腦套短、長MA穿越就行XDD
'''


ew = 2 
fw = 3
maLong = 20
maShort = 5
freq = '1D'
start_date = '1990-01-01'


dfList = []
for file in glob.glob(sys.path[0] + '/dataset/*.csv'):
    tmp = pd.read_csv(file, index_col=0)[['Value_Close']]
    tmp.rename(columns={'Value_Close': file.split('-')[0].split('/')[-1]}, inplace=True)
    tmp.index = pd.to_datetime(tmp.index)
    dfList.append(tmp)
df = pd.concat(dfList, axis=1)
df = df.resample(freq).last()
df.fillna(method='ffill', inplace=True)
print(df.tail(3))


df['NewIndex'] = df['EXF1']*4000*ew - df['FXF1']*1000*fw
constant = abs(df['NewIndex'].min()) + df['NewIndex'].loc[df['NewIndex'] > 0].mean()
# to make sure the index is always positive
df['NewIndex'] = df['NewIndex'] + constant
df['NI_maShort'] = df['NewIndex'].rolling(maShort).mean()
df['NI_maLong'] = df['NewIndex'].rolling(maLong).mean()
weights = df.copy(deep=True)
weights.dropna(inplace=True)


weights['FXF1'] = weights['NI_maShort'] > weights['NI_maLong']
weights['FXF1'] = weights['FXF1'].replace({True: -1*fw, False: fw})
weights['EXF1'] = weights['NI_maShort'] > weights['NI_maLong']
weights['EXF1'] = weights['EXF1'].replace({True: ew, False: -1*ew})
weights = weights[['FXF1', 'EXF1']]


s = bt.Strategy('EnF', [
    bt.algos.SelectAll(),
    bt.algos.WeighTarget(weights=weights),
    bt.algos.Rebalance()
])
t = bt.Backtest(s, df[['FXF1', 'EXF1']].loc[df.index > start_date])
res = bt.run(t)
print('\n', res.display())


file_dst = sys.path[0] + '/result/EnF-%s.pkl' % datetime.datetime.now()
with open(file_dst, 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)