#!/usr/bin/env python3
"""Create native, standalone panel figures from the result JSON files.

No image/PDF cropping is used: every output has a fresh Figure, Axes and
legend.  The panel filenames remain stable for manuscript and slide links.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = {"fmqa":"#1565c0", "std":"#e53935", "xy":"#2e7d32", "opt":"#212121"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.labelsize":11,
                     "axes.titlesize":12, "legend.fontsize":9, "grid.alpha":.32,
                     "grid.linestyle":"--", "lines.linewidth":2})

def data(exp, name):
    return json.loads((ROOT/exp/"json"/name).read_text())

def save(exp, stem, draw):
    fig, ax = plt.subplots(figsize=(7.2,5.1), dpi=300)
    draw(ax)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper center", bbox_to_anchor=(.5,-.19), ncol=2, framealpha=.95)
        fig.subplots_adjust(bottom=.28)
    for kind in ("png","pdf"):
        d=ROOT/exp/kind/"panels"; d.mkdir(parents=True,exist_ok=True)
        fig.savefig(d/f"{stem}.{kind}", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"{exp}/pdf/panels/{stem}.pdf")

def titles(ax,t,x,y): ax.set_title(t,fontweight="bold"); ax.set_xlabel(x); ax.set_ylabel(y)

def optimization():
    d=data("optimization","results_two_bb_optimization.json")
    keys=("FMQA_default","FMQA_adaptive","Standard_QAOA","FM_XY_QAOA")
    labels=("FMQA\ndefault","FMQA\nadaptive","Standard\nQAOA","FM-XY\nQAOA")
    cols=("#90caf9",C["fmqa"],C["std"],C["xy"])
    for i,(case,name) in enumerate((("BB1_materials","BB-1 Materials Catalyst"),("BB2_random","BB-2 Fully-Connected Random"))):
        direct=d[case]["part1_direct"]
        for metric,stem,y,ylim,letter in (("feasibility_rate","feasibility","Feasibility rate (%)",(0,115),"ab"[i]),("success_probability","success_probability","Ground-state hit probability (%)",(0,9.5),"cd"[i])):
            def draw(ax,m=metric,ttl=name,lim=ylim):
                vals=[direct[k][m]*100 for k in keys]; bars=ax.bar(labels,vals,color=cols,edgecolor="black")
                for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+(2 if m=="feasibility_rate" else .18),f"{v:.2f}%",ha="center",fontweight="bold")
                ax.set_ylim(*lim); titles(ax,f"{ttl}: {y}","Method",y)
            save("optimization",f"panel_{letter}_bb{i+1}_{stem}",draw)
        b=d[case]["part2_bbo"]; x=np.arange(101)
        def curve(ax,b=b,name=name,case=case):
            for k,l,col,mark,style in (("FMQA","FMQA",C["fmqa"],"s","-"),("Standard_QAOA","Standard QAOA",C["std"],"^","--"),("FM_XY_QAOA","FM-XY-QAOA",C["xy"],"o","-")):
                ax.plot(x,b[k]["best_history"],label=l,color=col,marker=mark,markevery=20,linestyle=style)
            ax.axhline(d[case]["exact_min"],color=C["opt"],linestyle=":",label="Exact minimum"); ax.set_xlim(0,100); titles(ax,f"{name}: 100-Cycle BBO Trajectory","BBO cycle","Best objective value (lower is better)")
        save("optimization",f"panel_{'ef'[i]}_bb{i+1}_bbo_trajectory",curve)

def sensitivity(exp, filename, std=False):
    d=data(exp,filename)
    for i,(case,name) in enumerate((("BB1_materials","BB-1 Materials Catalyst"),("BB2_random","BB-2 Fully-Connected Random"))):
        z=d[case]; scan=z["scan_data"] if std else z["sweep_data"]; x=[p["lambda"] for p in scan]
        if std:
            specs=(("ab"[i],"feasibility", "theoretical_feasibility","empirical_feasibility","Feasibility rate (%)"),("cd"[i],"success_probability","theoretical_p_opt","empirical_p_opt","Ground-state probability (%)"),("ef"[i],"feasible_energy","best_feas_energy","mean_feas_energy","Objective value (lower is better)"))
        else:
            specs=(("ab"[i],"feasibility","feasibility_rate",None,"Feasibility rate (%)"),("cd"[i],"success_probability","success_probability",None,"Ground-state probability (%)"),("ef"[i],"energy","best_feasible_energy","mean_feasible_energy","Objective value (lower is better)"))
        for letter,stem,a,b,ylabel in specs:
            def draw(ax,a=a,b=b,stem=stem,z=z,name=name):
                vals=lambda key:[np.nan if p[key] is None else p[key]*(100 if stem!="feasible_energy" and stem!="energy" else 1) for p in scan]
                ax.plot(x,vals(a),color=C["std"] if std else C["fmqa"],marker="o",label=("Standard QAOA" if std else "FMQA"))
                if b: ax.plot(x,vals(b),color="#ff7043",marker="^",linestyle="--",label="Sampled / mean feasible")
                if stem=="feasibility": ax.axhline(100,color=C["xy"],label="FM-XY-QAOA (100%)")
                elif stem=="success_probability":
                    ref=(z["fm_xy_qaoa_reference"] if std else z["fm_xy_qaoa_baseline"])["success_probability"]*100
                    ax.axhline(ref,color=C["xy"],label="FM-XY-QAOA reference")
                else: ax.axhline(z["exact_min"],color=C["opt"],linestyle=":",label="Exact minimum")
                titles(ax,f"{name}: {ylabel} vs Penalty","Penalty coefficient $\\lambda$",ylabel)
            save(exp,f"panel_{letter}_bb{i+1}_{stem}",draw)

def scaling():
    d=data("qubit_scaling","results_qubit_scaling.json")["data"]; n=[r["N"] for r in d]
    defs=(("a","feasibility","feasibility_rate","Feasibility rate (%) [log]"),("c","ground_state_probability","p_opt","Ground-state probability (%) [log]"),("d","optimality_gap","opt_gap","Optimality gap"),("e","runtime","runtime_sec","Execution time (s) [log]"))
    for letter,stem,key,ylabel in defs:
        def draw(ax,key=key,ylabel=ylabel,stem=stem):
            for method,label,col,mark in (("FM_XY_QAOA","FM-XY-QAOA",C["xy"],"o"),("Standard_QAOA","Standard QAOA",C["std"],"^"),("FMQA","FMQA",C["fmqa"],"s")):
                r=[q for q in d if q[method][key] is not None]; y=[q[method][key]*(100 if key in ("feasibility_rate","p_opt") else 1) for q in r]; ax.plot([q["N"] for q in r],y,label=label,color=col,marker=mark)
            if "log" in ylabel: ax.set_yscale("log")
            if key=="opt_gap": ax.axhline(0,color="#777",linestyle=":",label="Exact minimum")
            titles(ax,f"{stem.replace('_',' ').title()} vs Number of Qubits","Number of qubits $N$",ylabel)
        save("qubit_scaling",f"panel_{letter}_{stem}",draw)
    for letter,stem,ys,label in (("b","state_space",("hilbert_dim","subspace_dim"),("Full Hilbert space","Valid One-Hot subspace")),("f","memory",("ram_bytes_full","ram_bytes_sub"),("Full state-vector RAM","Subspace state-vector RAM"))):
        def draw(ax,ys=ys,label=label,stem=stem):
            scale=1024**3 if stem=="memory" else 1
            ax.plot(n,[q[ys[0]]/scale for q in d],color=C["std"],marker="^",label=label[0]); ax.plot(n,[q[ys[1]]/scale for q in d],color=C["xy"],marker="o",label=label[1]); ax.set_yscale("log"); titles(ax,stem.replace("_"," ").title(),"Number of qubits $N$","Memory (GB) [log]" if stem=="memory" else "Number of states [log]")
        save("qubit_scaling",f"panel_{letter}_{stem}",draw)

def robustness():
    d=data("bbo_seed_robustness","results_bbo_seed_robustness.json")
    for i,(case,p) in enumerate(d["problems"].items()):
        def draw(ax,p=p,case=case):
            x=np.arange(101)
            for m,col in (("FMQA",C["fmqa"]),("Standard QAOA",C["std"]),("FM-XY-QAOA",C["xy"])):
                h=np.asarray([r["best_history"] for r in p["methods"][m]]); ax.plot(x,h.mean(0),label=m,color=col,marker="o",markevery=20); ax.fill_between(x,h.mean(0)-h.std(0,ddof=1),h.mean(0)+h.std(0,ddof=1),color=col,alpha=.15)
            ax.axhline(p["exact_min"],color=C["opt"],linestyle=":",label="Exact minimum"); titles(ax,f"{case}: BBO Seed Robustness","BBO cycle","Best objective value (mean ± SD)")
        save("bbo_seed_robustness",f"panel_{'ab'[i]}_{case.lower()}",draw)

def main():
    optimization(); sensitivity("penalty_sensitivity","results_penalty_sensitivity.json"); sensitivity("std_qaoa_sensitivity","results_std_qaoa_sensitivity.json",True); scaling(); robustness()
if __name__=="__main__": main()
