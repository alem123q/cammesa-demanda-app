import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
R='/home/claude/cammesa-demanda-app/'
s=pd.read_csv(R+'data/processed/serie_regional.csv')
w=pd.read_csv(R+'data/external/weather_monthly_regions.csv')
e=pd.read_csv(R+'data/external/emae_monthly.csv')
S=s.pivot(index='indice_tiempo',columns='region',values='demanda_MWh')/1000
W=w.set_index(['region','indice_tiempo'])
E=e.set_index('indice_tiempo')['emae']
def shift_month(m,k):
    p=pd.Period(m,'M')-k; return str(p)
rows=[]
for t in range(12,len(S)):
    f=S.index[t]; mes=int(f[5:])
    for r in S.columns:
        f12=shift_month(f,12)
        rows.append(dict(fecha=f,region=r,mes=mes,trimestre=(mes-1)//3+1,y=S[r].iloc[t],lag1=S[r].iloc[t-1],lag12=S[r].iloc[t-12],
          hdd=W.loc[(r,f),'hdd'],cdd=W.loc[(r,f),'cdd'],hdd12=W.loc[(r,f12),'hdd'],cdd12=W.loc[(r,f12),'cdd'],
          tmean=W.loc[(r,f),'temp_mean'],tmean12=W.loc[(r,f12),'temp_mean'],
          emae_yoy=np.log(E[f]/E[f12]), emae_yoy_l2=np.log(E[shift_month(f,2)]/E[shift_month(f,14)])))
D=pd.DataFrame(rows)
D['d_hdd']=D.hdd-D.hdd12; D['d_cdd']=D.cdd-D.cdd12
tr,te=D[D.fecha<'2019-09'],D[D.fecha>='2019-09']
print(len(tr),len(te))
base=['lag1','lag12','mes','trimestre']
def X(df,num): return df.assign(lag1=np.log1p(df.lag1),lag12=np.log1p(df.lag12))[num+['region']]
def pipe(num,est,inter=None):
    tf=[('num',StandardScaler(),num),('region',OneHotEncoder(drop='first',handle_unknown='ignore'),['region'])]
    return make_pipeline(ColumnTransformer(tf),est)
def rmse(a,b): return mean_squared_error(a,b)**.5
def evaluate(name,num,est):
    # validation: TimeSeriesSplit(4, test_size=27) on train ordered by fecha
    trs=tr.sort_values(['fecha','region']).reset_index(drop=True)
    v=[]
    for a,b in TimeSeriesSplit(4,test_size=27).split(trs):
        m=pipe(num,est).fit(X(trs.iloc[a],num),np.log1p(trs.y.iloc[a]))
        v.append(rmse(trs.y.iloc[b],np.expm1(m.predict(X(trs.iloc[b],num)))))
    m=pipe(num,est).fit(X(tr,num),np.log1p(tr.y)); p=np.expm1(m.predict(X(te,num)))
    return dict(modelo=name,val=np.mean(v),test=rmse(te.y,p)),p
out=[];preds={}
cands={
 'Ridge base':base,
 '+ clima (grados-día del mes)':base+['hdd','cdd'],
 '+ clima (dif. vs año anterior)':base+['d_hdd','d_cdd'],
 '+ EMAE interanual (mismo mes)':base+['emae_yoy'],
 '+ EMAE interanual (rezago 2, publicado)':base+['emae_yoy_l2'],
 '+ clima dif + EMAE mismo mes':base+['d_hdd','d_cdd','emae_yoy'],
 '+ clima dif + EMAE rezago 2':base+['d_hdd','d_cdd','emae_yoy_l2'],
}
for n,num in cands.items():
    r,p=evaluate(n,num,Ridge(alpha=3.16)); out.append(r); preds[n]=p
out.append(dict(modelo='Regla: mismo mes año anterior',val=np.nan,test=rmse(te.y,te.lag12)))
print(pd.DataFrame(out).round(1).to_string(index=False))
D.to_csv('analysis/tabla_ext.csv',index=False)
