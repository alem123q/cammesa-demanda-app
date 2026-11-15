import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
D=pd.read_csv('analysis/tabla_ext.csv'); D['target']=np.log(D.y/D.lag12)
D['d_hdd']=D.hdd-D.hdd12; D['d_cdd']=D.cdd-D.cdd12
cols=['d_hdd','d_cdd','emae_yoy']
m=LinearRegression().fit(D[cols],D.target); print('coef',dict(zip(cols,m.coef_.round(5))),'int',round(m.intercept_,4),'R2',round(m.score(D[cols],D.target),2))
for name,a,b in [('caída sep18-ago19','2018-09','2019-08'),('recuperación sep19-feb20','2019-09','2020-02')]:
    P=D[(D.fecha>=a)&(D.fecha<=b)]
    wts=P.lag12/P.lag12.sum()   # weight by size
    real=(wts*P.target).sum()
    contr={c:(wts*P[c]*m.coef_[i]).sum() for i,c in enumerate(cols)}
    tot=P.y.sum()/P.lag12.sum()-1
    print(name,'real %.1f%%'%(100*real),{k:'%.1f%%'%(100*v) for k,v in contr.items()},'resto %.1f%%'%(100*(real-sum(contr.values())-m.intercept_)),'interc %.1f%%'%(100*m.intercept_), 'tot', round(100*tot,1))
