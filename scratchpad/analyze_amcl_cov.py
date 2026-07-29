#!/usr/bin/env python3
"""W2: AMCL covariance-calibration analysis (posmon-matched ground truth).

For each AMCL sample (p_hat, Sigma) we take the concurrent Gazebo ground-truth pose
from the position-monitor log (nearest time), form e = p_gt - p_hat and the
Mahalanobis distance q = e^T Sigma^-1 e over BENIGN samples (before the LiDAR spoof
drives the estimate away). We compare the empirical distribution of q to chi^2_2 and
report nominal-vs-empirical coverage; if AMCL is over-confident we fit a scalar
kappa (Sigma <- kappa*Sigma). chi^2_2 quantile at level a is -2 ln(1-a); E[chi^2_2]=2.
"""
import json, glob, math, bisect, os

D = 'experiment_results/gazebo_s1_s6/poscov'
SPOOF_E = 0.40   # m: error above which a trial is treated as spoofed (benign cut)
MATCH_TOL = 0.35 # s: max time gap for amcl<->gt match

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
    i = bisect.bisect_left(ts, t)
    best=None
    for j in (i-1, i):
        if 0<=j<len(ts) and (best is None or abs(ts[j]-t)<abs(ts[best]-t)): best=j
    if best is None or abs(ts[best]-t)>MATCH_TOL: return None
    return xy[best]

def main():
    qs=[]; nf=nall=nmatch=nbenign=0
    for cf in sorted(glob.glob(f'{D}/cov_v*.jsonl')):
        pf = cf.replace('cov_v','posmon_v').replace('.jsonl','.log')
        if not os.path.exists(pf): continue
        nf+=1
        pts, pxy = load_posmon(pf)
        rows=[json.loads(l) for l in open(cf) if l.strip()]
        for r in rows:
            nall+=1
            g = nearest(pts, pxy, r['t'])
            if g is None: continue
            nmatch+=1
            ex=g[0]-r['amcl_x']; ey=g[1]-r['amcl_y']
            if ex*ex+ey*ey > SPOOF_E*SPOOF_E:
                break  # spoof onset for this trial -> stop
            sxx,sxy,syy=r['sxx'],r['sxy'],r['syy']
            det=sxx*syy-sxy*sxy
            if det<=1e-12 or sxx<=0 or syy<=0: continue
            q = ex*ex*(syy/det) + 2*ex*ey*(-sxy/det) + ey*ey*(sxx/det)
            qs.append(q); nbenign+=1
    if not qs:
        print(f"no benign q (files={nf} samples={nall} matched={nmatch})"); return
    qs.sort()
    mean_q=sum(qs)/len(qs); med_q=qs[len(qs)//2]; kappa=mean_q/2.0
    def cov(k): return {a: sum(1 for q in qs if q/k<=chi2_2_q(a))/len(qs) for a in (0.90,0.95,0.99)}
    raw=cov(1.0); cal=cov(kappa)
    print(f"files={nf} amcl_samples={nall} matched={nmatch} benign={nbenign}")
    print(f"mean(q)={mean_q:.2f} median(q)={med_q:.2f} (ideal chi2_2: 2.00 / 1.39)")
    print(f"kappa=mean(q)/2={kappa:.2f} (>1 => AMCL over-confident)\n")
    print(f"{'nominal':>8} {'chi2thr':>8} {'emp(raw)':>10} {'emp(kappa)':>11}")
    for a in (0.90,0.95,0.99):
        print(f"{a*100:7.0f}% {chi2_2_q(a):8.3f} {raw[a]*100:9.1f}% {cal[a]*100:10.1f}%")
    json.dump({'benign_samples':nbenign,'mean_q':mean_q,'median_q':med_q,'kappa':kappa,
               'coverage_raw':{str(a):raw[a] for a in raw},
               'coverage_kappa':{str(a):cal[a] for a in cal}},
              open(f'{D}/coverage_summary.json','w'), indent=2)
    print(f"\nwrote {D}/coverage_summary.json")

if __name__=='__main__':
    os.chdir('/home/jim/ros2_motion_planning_tutorials'); main()
