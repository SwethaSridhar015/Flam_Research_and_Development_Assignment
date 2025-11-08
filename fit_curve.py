#!/usr/bin/env python3
import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass
from math import atan2, degrees, radians, sin, cos
from scipy.optimize import least_squares

CSV_PATH = "xy_data.csv"
T_MIN, T_MAX, N_T = 6.0, 60.0, 800
PLOT_DIR = "plots"

plt.rcParams.update({
    "figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10
})

@dataclass
class Fit:
    theta_deg: float; M: float; X: float
    mae_n: float; rmse_n: float; l1_mean: float; l1_total: float
    desmos: str; latex: str

def read_xy(p):
    d = pd.read_csv(p)
    if not {"x","y"}.issubset(d.columns): raise ValueError("need x,y")
    return d[["x","y"]].to_numpy(float)

def pca_theta(X):
    mu = X.mean(0); C = np.cov((X-mu).T)
    w,V = np.linalg.eigh(C); v = V[:, np.argmax(w)]
    th = (degrees(atan2(v[1], v[0]))+180)%180
    if th>90: th = 180-th
    return float(np.clip(th,.001,50))

def solve_X(mu_x, mu_y, th):
    s,c = sin(th),cos(th)
    return mu_x - (mu_y-42)*c/max(s,1e-12)

def project_tn(X, th, X0):
    s,c = sin(th),cos(th); x,y = X[:,0],X[:,1]
    t = (x-X0)*c + (y-42)*s
    n = -(x-X0)*s + (y-42)*c
    return t,n

def estimate_M(t,n):
    s0 = np.sin(0.3*t)
    m = (np.abs(s0)>1e-3)&(np.sign(s0)==np.sign(n))
    if m.sum()<10: m = np.abs(s0)>1e-3
    w = np.abs(t[m]); z = np.log(np.clip(np.abs(n[m]/s0[m]),1e-12,None))
    M = (w*z).sum()/(w*w).sum() if len(w)>0 else 0.0
    return float(np.clip(M,-.05,.05))

def model(th,M,X0,t):
    x = t*np.cos(th) - np.exp(M*np.abs(t))*np.sin(0.3*t)*np.sin(th) + X0
    y = 42 + t*np.sin(th) + np.exp(M*np.abs(t))*np.sin(0.3*t)*np.cos(th)
    return x,y

def residuals_normal(p,X):
    thd,M,X0 = p; th = radians(thd)
    t,n = project_tn(X,th,X0)
    return n - np.exp(M*np.abs(t))*np.sin(0.3*t)

def metrics_normal(X,thd,M,X0):
    th = radians(thd); t,n = project_tn(X,th,X0)
    nh = np.exp(M*np.abs(t))*np.sin(0.3*t)
    r = n-nh
    mae = float(np.mean(np.abs(r))); rmse = float(np.sqrt(np.mean(r*r)))
    return mae,rmse,t,r

def l1_uniform_t(X,thd,M,X0):
    th = radians(thd)
    t_data,_ = project_tn(X,th,X0)
    m = (t_data>=T_MIN)&(t_data<=T_MAX)
    t_data = t_data[m]; XY = X[m]
    i = np.argsort(t_data)
    t_sorted = t_data[i]; xs = XY[i,0]; ys = XY[i,1]
    tg = np.linspace(max(T_MIN,t_sorted.min()), min(T_MAX,t_sorted.max()), N_T)
    xe = np.interp(tg,t_sorted,xs); ye = np.interp(tg,t_sorted,ys)
    xp,yp = model(th,M,X0,tg)
    l1 = np.abs(xe-xp)+np.abs(ye-yp)
    return float(l1.mean()), float(l1.sum()), tg, xe, ye, xp, yp

def desmos(th,M,X0):
    return f"(t*cos({th:.10f})-exp({M:.10f}*abs(t))*sin(0.3*t)*sin({th:.10f})+{X0:.10f},42+t*sin({th:.10f})+exp({M:.10f}*abs(t))*sin(0.3*t)*cos({th:.10f}))"

def latex(th,M,X0):
    return (r"\left(t\cos("+f"{th:.10f}"+r")-e^{"+f"{M:.10f}"+r"\left|t\right|}\sin(0.3t)\sin("+f"{th:.10f}"+r")+"
            +f"{X0:.10f}"+r",42+t\sin("+f"{th:.10f}"+r")+e^{"+f"{M:.10f}"+r"\left|t\right|}\sin(0.3t)\cos("+f"{th:.10f}"+r")\right)")

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    X = read_xy(CSV_PATH)

    thd0 = pca_theta(X); th0 = radians(thd0)
    X0 = solve_X(*X.mean(0), th0)
    M0 = estimate_M(*project_tn(X,th0,X0))
    res = least_squares(residuals_normal, [thd0,M0,X0], args=(X,),
                        bounds=([.001,-.05,0],[50,.05,100]),
                        loss="soft_l1", f_scale=1.0, max_nfev=2000)
    thd,M,Xoff = res.x; th = radians(thd)

    mae,rmse,t_all,r_all = metrics_normal(X,thd,M,Xoff)
    l1m,l1s,tg,xe,ye,xp,yp = l1_uniform_t(X,thd,M,Xoff)
    des,ltx = desmos(th,M,Xoff), latex(th,M,Xoff)

    result = {
        "theta_deg": round(thd,6), "M": round(M,6), "X": round(Xoff,6),
        "normal_mae": round(mae,6), "normal_rmse": round(rmse,6),
        "l1_mean": round(l1m,6), "l1_total": round(l1s,6),
        "desmos": des, "latex": ltx, "plots": {}
    }

    print(f"\nθ(deg)={thd:.6f}\nM={M:.6f}\nX={Xoff:.6f}")
    print(f"Normal_MAE={mae:.6f}  Normal_RMSE={rmse:.6f}")
    print(f"L1_mean={l1m:.6f}  L1_total={l1s:.6f}")
    print("\nDesmos:\n"+des+"\n\nLaTeX:\n"+ltx)

    # --- 1) points + fitted curve ---
    t_line = np.linspace(T_MIN,T_MAX,1200)
    x_fit,y_fit = model(th,M,Xoff,t_line)
    amp = np.exp(M*np.abs(t_line)); v = np.array([-sin(th),cos(th)])
    x_up,y_up = x_fit + amp*v[0], y_fit + amp*v[1]
    x_dn,y_dn = x_fit - amp*v[0], y_fit - amp*v[1]
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0],X[:,1], s=35, color="#3498db", alpha=0.85, label="data")
    plt.plot(x_fit,y_fit, color="#e74c3c", lw=2.2, label="fit")
    plt.plot(x_up,y_up,"--", color="#95a5a6", lw=1.2, alpha=0.9)
    plt.plot(x_dn,y_dn,"--", color="#95a5a6", lw=1.2, alpha=0.9)
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Observed data and fitted curve")
    plt.axis("equal"); plt.legend(frameon=False); plt.tight_layout()
    f1=os.path.join(PLOT_DIR,"01_points_and_fit.png"); plt.savefig(f1,dpi=220)

    # --- 2) residuals vs t ---
    plt.figure(figsize=(8,4.5))
    plt.plot(t_all, r_all, ".", ms=3, alpha=0.8)
    plt.axhline(0, lw=1, color="#7f8c8d")
    plt.xlabel("t"); plt.ylabel("normal residual"); plt.title("Residuals vs t")
    plt.tight_layout(); f2=os.path.join(PLOT_DIR,"02_residuals_vs_t.png"); plt.savefig(f2,dpi=220)

    # --- 3) expected vs predicted ---
    fig,axs = plt.subplots(1,2, figsize=(11,4.6), sharex=True)
    axs[0].plot(tg, xe, lw=2, label="x expected"); axs[0].plot(tg, xp, lw=1.8, label="x predicted")
    axs[1].plot(tg, ye, lw=2, label="y expected"); axs[1].plot(tg, yp, lw=1.8, label="y predicted")
    axs[0].set_title("x(t)"); axs[1].set_title("y(t)")
    for ax in axs: ax.legend(); ax.set_xlabel("t"); ax.set_ylabel("value")
    fig.tight_layout(); f3=os.path.join(PLOT_DIR,"03_expected_vs_pred.png"); fig.savefig(f3,dpi=220)

    # --- 4) residual histogram ---
    mu, sd = float(np.mean(r_all)), float(np.std(r_all))
    plt.figure(figsize=(8,4.5))
    plt.hist(r_all, bins=40, edgecolor="white", color="#4c72b0", alpha=0.9)
    plt.axvline(mu, color="#c0392b", lw=2, label=f"mean={mu:.3g}")
    plt.axvline(mu+sd, color="#27ae60", lw=1.5, ls="--", label=f"±1σ={sd:.3g}")
    plt.axvline(mu-sd, color="#27ae60", lw=1.5, ls="--")
    plt.xlabel("normal residual"); plt.ylabel("count"); plt.title("Residual distribution")
    plt.legend(); plt.tight_layout(); f4=os.path.join(PLOT_DIR,"04_residual_hist.png"); plt.savefig(f4,dpi=220)

    result["plots"] = {"fit": f1, "residuals_vs_t": f2, "expected_vs_pred": f3, "residual_hist": f4}
    with open(os.path.join(PLOT_DIR,"results_summary.json"),"w") as f: json.dump(result,f,indent=2)
    plt.show()

if __name__=="__main__":
    main()
