#!/usr/bin/env python3
"""W2 (R1/R3): AMCL covariance calibration with a moving-only subset and CIs.

q = e^T Sigma^-1 e, e = p_gt - p_hat, over BENIGN samples (before the spoof drives
the error past SPOOF_E). We additionally report the MOVING subset (robot >0.3 m
from start) so trivially-covered stationary samples at the origin do not inflate
coverage, and give bootstrap 95% CIs. chi^2_2 quantile at level a is -2 ln(1-a);
E[chi^2_2]=2.
"""
import json, glob, math, bisect, os, hashlib

D = 'experiment_results/gazebo_s1_s6/poscov'
SPOOF_E = 0.40; MATCH_TOL = 0.35; MOVE_MIN = 0.30
NBOOT = 2000

def chi2_2_q(a): return -2.0 * math.log(1.0 - a)

def load_posmon(f):
    ts, xy = [], []
    for line in open(f):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except: continue
        if d.get('x') is None: continue
        ts.append(d['t']); xy.append((d['x'], d['y']))
    return ts, xy

def nearest(ts, xy, t):
    if not ts: return None
    i = bisect.bisect_left(ts, t); best=None
    for j in (i-1, i):
        if 0<=j<len(ts) and (best is None or abs(ts[j]-t)<abs(ts[best]-t)): best=j
    if best is None or abs(ts[best]-t)>MATCH_TOL: return None
    return xy[best]

def _rng(seed):
    x = int(hashlib.md5(str(seed).encode()).hexdigest(), 16) & 0xffffffff
    while True:
        x = (1103515245*x + 12345) & 0x7fffffff
        yield x / 0x7fffffff

def boot_ci(vals, stat, nb=NBOOT):
    n=len(vals); g=_rng(len(vals)); out=[]
    for _ in range(nb):
        samp=[vals[int(next(g)*n)] for _ in range(n)]
        out.append(stat(samp))
    out.sort(); return out[int(0.025*nb)], out[int(0.975*nb)]

def coverage(qs, a, k=1.0): return sum(1 for q in qs if q/k<=chi2_2_q(a))/len(qs)

def collect():
    allq=[]; movq=[]
    for cf in sorted(glob.glob(f'{D}/cov_v*.jsonl')):
        pf=cf.replace('cov_v','posmon_v').replace('.jsonl','.log')
        if not os.path.exists(pf): continue
        pts,pxy=load_posmon(pf)
        for r in (json.loads(l) for l in open(cf) if l.strip()):
            g=nearest(pts,pxy,r['t'])
            if g is None: continue
            ex=g[0]-r['amcl_x']; ey=g[1]-r['amcl_y']
            if ex*ex+ey*ey > SPOOF_E*SPOOF_E: break
            sxx,sxy,syy=r['sxx'],r['sxy'],r['syy']; det=sxx*syy-sxy*sxy
            if det<=1e-12 or sxx<=0 or syy<=0: continue
            q=ex*ex*(syy/det)+2*ex*ey*(-sxy/det)+ey*ey*(sxx/det)
            allq.append(q)
            if g[0]*g[0]+g[1]*g[1] > MOVE_MIN*MOVE_MIN: movq.append(q)
    return allq, movq

def report(name, qs):
    if not qs: print(f'[{name}] no samples'); return None
    qs=sorted(qs); mean_q=sum(qs)/len(qs); kappa=mean_q/2.0
    mlo,mhi=boot_ci(qs, lambda s: sum(s)/len(s))
    print(f'[{name}] n={len(qs)}  mean(q)={mean_q:.2f} [{mlo:.2f},{mhi:.2f}]  '
          f'median={qs[len(qs)//2]:.2f}  kappa={kappa:.2f}')
    print(f'  {"nom":>4} {"thr":>7} {"emp":>6} {"95% CI":>13}')
    res={}
    for a in (0.90,0.95,0.99):
        c=coverage(qs,a); lo,hi=boot_ci(qs, lambda s: coverage(s,a))
        print(f'  {int(a*100):>3}% {chi2_2_q(a):7.3f} {c*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]')
        res[a]=(c,lo,hi)
    return {'n':len(qs),'mean_q':mean_q,'mean_ci':[mlo,mhi],'kappa':kappa,
            'coverage':{str(a):list(res[a]) for a in res}}

def main():
    allq, movq = collect()
    out={}
    out['all']=report('all benign', allq)
    out['moving']=report(f'moving (>{MOVE_MIN}m)', movq)
    json.dump(out, open(f'{D}/coverage_summary.json','w'), indent=2)
    print(f'\nwrote {D}/coverage_summary.json')

if __name__=='__main__':
    os.chdir('/home/jim/ros2_motion_planning_tutorials'); main()
