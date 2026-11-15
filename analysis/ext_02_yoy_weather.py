import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
D=pd.read_csv('analysis/tabla_ext.csv')
s=pd.read_csv('/home/claude/cammesa-demanda-app/data/processed/serie_regional.csv')
S=s.set_index(['region','indice_tiempo'])['demanda_MWh']/1000
D['lag13']=[S.get((r,str(pd.Period(f,'M')-13)),np.nan) for r,f in zip(D.region,D.fecha)]
D['yoy1']=np.log(D.lag1/D.lag13)
D['target']=np.log(D.y/D.lag12)
D=D.dropna(subset=['yoy1']).reset_index(drop=True)
tr,te=D[D.fecha<'2019-09'].sort_values(['fecha','region']).reset_index(drop=True),D[D.fecha>='2019-09']
regs=sorted(D.region.unique())
def feats(df,cols,inter):
    F=df[cols].copy() if cols else pd.DataFrame(index=df.index)
    if inter:
        for r in regs:
            for c in inter: F[f'{c}_{r}']=df[c]*(df.region==r)
    return F
def rmse(a,b): return mean_squared_error(a,b)**.5
def run(name,cols,inter=(),alpha=1.0):
    def fit_pred(a,b):
        Fa,Fb=feats(a,cols,inter),feats(b,cols,inter)
        mu,sd=Fa.mean(),Fa.std().replace(0,1)
        m=Ridge(alpha=alpha).fit((Fa-mu)/sd,a.target)
        return b.lag12*np.exp(m.predict((Fb-mu)/sd))
    v=[rmse(tr.y.iloc[b],fit_pred(tr.iloc[a],tr.iloc[b])) for a,b in TimeSeriesSplit(4,test_size=27).split(tr)]
    return dict(modelo=name,val=round(np.mean(v),1),test=round(rmse(te.y,fit_pred(tr,te)),1))
res=[dict(modelo='Regla (cambio interanual = 0)',val=round(np.mean([rmse(tr.y.iloc[b],tr.lag12.iloc[b]) for a,b in TimeSeriesSplit(4,test_size=27).split(tr)]),1),test=round(rmse(te.y,te.lag12),1))]
for alpha in (1.0,10.0):
    res+= [run(f'yoy1 (a={alpha})',['yoy1'],alpha=alpha),
     run(f'yoy1 + clima (a={alpha})',['yoy1','d_hdd','d_cdd'],alpha=alpha),
     run(f'yoy1 + clima x región (a={alpha})',['yoy1'],('d_hdd','d_cdd'),alpha=alpha),
     run(f'yoy1 + clima x región + EMAE (a={alpha})',['yoy1','emae_yoy'],('d_hdd','d_cdd'),alpha=alpha),
     run(f'yoy1 + clima x región + EMAE rez2 (a={alpha})',['yoy1','emae_yoy_l2'],('d_hdd','d_cdd'),alpha=alpha),
     run(f'clima x región + EMAE (sin yoy1) (a={alpha})',['emae_yoy'],('d_hdd','d_cdd'),alpha=alpha)]
print(len(tr),len(te)); print(pd.DataFrame(res).to_string(index=False))
# --- weather known at forecast time: climatological normal (mean of 2016-2018 for that region-month) vs last year's actual
w=pd.read_csv('/home/claude/cammesa-demanda-app/data/external/weather_monthly_regions.csv')
w['m']=w.indice_tiempo.str[5:]
norm=w[w.indice_tiempo<'2019-01'].groupby(['region','m'])[['hdd','cdd']].mean()
D['m']=D.fecha.str[5:]
D['n_hdd']=[norm.loc[(r,m),'hdd'] for r,m in zip(D.region,D.m)]; D['n_cdd']=[norm.loc[(r,m),'cdd'] for r,m in zip(D.region,D.m)]
D['dn_hdd']=D.n_hdd-D.hdd12; D['dn_cdd']=D.n_cdd-D.cdd12
tr,te=D[D.fecha<'2019-09'].sort_values(['fecha','region']).reset_index(drop=True),D[D.fecha>='2019-09']
print(pd.DataFrame([run('yoy1 + clima NORMAL (sin ver el mes)',['yoy1','dn_hdd','dn_cdd']),
  run('yoy1 + clima NORMAL + EMAE rez2',['yoy1','dn_hdd','dn_cdd','emae_yoy_l2']),
  run('yoy1 + clima REAL',['yoy1','d_hdd','d_cdd'])]).to_string(index=False))
Fa=tr[['yoy1','d_hdd','d_cdd']]; m=LinearRegression().fit(Fa,tr.target); print(dict(zip(Fa.columns,m.coef_.round(5))))
