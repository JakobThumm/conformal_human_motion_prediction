import os, numpy as np, cloudpickle
from conformal_human_motion_prediction.utils.eval_utils import convert_covariance_matrices_to_set
from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_NAMES_13
from conformal_human_motion_prediction.motion_prediction.h36m_settings import SET_LIKELIHOOD
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
RUNS=[("baseline","results/motion_prediction/motion_prediction_results_validation.cloudpickle"),
("control","results/coverage_experiments/cov_control/motion_prediction_results_validation.cloudpickle"),
("P1","results/coverage_experiments/cov_p1_noise/motion_prediction_results_validation.cloudpickle"),
("P2","results/coverage_experiments/cov_p2_pinball/motion_prediction_results_validation.cloudpickle"),
("P1+P2","results/coverage_experiments/cov_p1p2/motion_prediction_results_validation.cloudpickle"),
("P1+P2+P4","results/coverage_experiments/cov_p1p2p4/motion_prediction_results_validation.cloudpickle"),
("P2+P4","results/coverage_experiments/cov_p2p4/motion_prediction_results_validation.cloudpickle"),
("P2hi+P4","results/coverage_experiments/cov_p2hi_p4/motion_prediction_results_validation.cloudpickle")]
RUNS=[(n,p) for n,p in RUNS if os.path.exists(os.path.join(ROOT,p))]
LA,RA=JOINT_NAMES_13.index("LAnkle"),JOINT_NAMES_13.index("RAnkle")
def load(p):
    r=cloudpickle.load(open(os.path.join(ROOT,p),'rb'))
    pred=np.asarray(r["predictions"],np.float64);tgt=np.asarray(r["targets"],np.float64)
    cov=np.asarray(r["covariance_matrices"],np.float64);last=np.asarray(r["last_input_poses"],np.float64)
    N,T,J,_=pred.shape
    in_cov=last[:,J*3:J*3+J*9].reshape(N,J,3,3)
    in_set=convert_covariance_matrices_to_set(in_cov,SET_LIKELIHOOD)/1000.0
    rad=convert_covariance_matrices_to_set(cov,SET_LIKELIHOOD)  # native mm
    dist=np.linalg.norm(pred-tgt,axis=-1)
    vTF=~(np.all(pred==0.0,axis=(2,3))|np.all(tgt==0.0,axis=(2,3)))
    valid=np.repeat(vTF[:,:,None],J,axis=2)
    inTJ=np.repeat(in_set[:,None,:],T,axis=1)
    return pred,tgt,rad,dist,valid,inTJ,J
def cov_at(scale,rad,dist,valid):
    return (dist[valid] <= scale*rad[valid]).mean()
print(f"{'variant':>10} | {'iso-scale':>9} | {'meanVol m3':>10} | {'cov[.5,.75)':>11} | {'cov[.75,1)':>10} | {'cov>=1':>7} | {'RAnkle':>7} | {'LAnkle':>7}")
print("-"*100)
for name,p in RUNS:
    pred,tgt,rad,dist,valid,inTJ,J=load(p)
    # bisect scale to hit 99.5% overall
    lo,hi=0.1,50.0
    for _ in range(40):
        mid=(lo+hi)/2
        if cov_at(mid,rad,dist,valid)<0.995: lo=mid
        else: hi=mid
    s=(lo+hi)/2
    r=rad*s
    within=dist<=r
    cf=within[valid]; inf=inTJ[valid]; rf=r[valid]/1000.0
    vol=4/3*np.pi*(rf.mean()**3)
    def strat(a,b):
        m=(inf>=a)&(inf<b); return 100*cf[m].mean() if m.any() else float('nan')
    ra=100*within[:,:,RA][valid[:,:,RA]].mean(); la=100*within[:,:,LA][valid[:,:,LA]].mean()
    print(f"{name:>10} | {s:>9.2f} | {vol:>10.4f} | {strat(0.5,0.75):>11.2f} | {strat(0.75,1.0):>10.2f} | {strat(1.0,1e9):>7.2f} | {ra:>7.2f} | {la:>7.2f}")
print("\nAll rows calibrated (global scale) to 99.50% OVERALL coverage. High-strata/ankle cols then show")
print("RESIDUAL conditional miscalibration: closer to 99.5 = better conditionally calibrated by training.")
