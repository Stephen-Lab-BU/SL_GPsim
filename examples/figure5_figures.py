"""Render the Figure 5 uncertainty analysis from the saved draws; no fitting here."""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from empirical_violin import violin
from figure5_data import AMPS, FIELDS, METHODS, OUT, broadband, slope, truth

COL = ["#4c72b0", "#55a868", "#dd8452"]
POST = "#825d9e"
INK = "#242d36"
AMP_LABELS = ["0.5", "1", "2", "4"]
SHORT = ["specparam", "SL multiplicative", "SL additive"]
mpl.rcParams.update({"axes.spines.top":False, "axes.spines.right":False, "axes.edgecolor":"#a4abb0",
    "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK, "text.color":INK,
    "font.size":9, "axes.titlesize":11, "axes.titleweight":"bold", "legend.frameon":False,
    "svg.fonttype":"none", "pdf.fonttype":42, "figure.facecolor":"white"})

def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", pad=8)

def sheet(title, subtitle, size=(12,7.5)):
    fig=plt.figure(figsize=size)
    fig.text(.055,.957,title,fontsize=21,weight="bold",va="top")
    fig.text(.055,.902,subtitle,fontsize=10.5,va="top",color="#46535b")
    return fig

def height_axis(ax):
    ax.set_xticks(range(4),AMP_LABELS); ax.set_xlim(-.6,3.6)
    ax.set_xlabel("Generating rhythm amplitude")

def htruth(ax, value):
    ax.axhline(value,ls="--",lw=1.05,color=INK,zorder=0)

def bb_curve(f, theta):
    return broadband(f,*theta[[1,2,3]])

def rhythm_curve(f, theta, m):
    g=np.exp(-.5*((f-theta[5])/theta[7])**2)
    if m==0:
        return bb_curve(f,theta)*np.expm1(np.log(10)*theta[6]*g)
    g+=np.exp(-.5*((f+theta[5])/theta[7])**2)
    return theta[6]*g if m==2 else bb_curve(f,theta)*np.expm1(np.log(10)*theta[6]*g)

def contrasts(records, seed=202611, n=20000):
    """Resample independent records within level; preserve method pairing."""
    rng=np.random.default_rng(seed)
    low,high=records[0,:,:,0],records[-1,:,:,0]
    indices=rng.integers(0,len(low),(2,n,len(low)))
    return np.nanmean(high[indices[1]],axis=1)-np.nanmean(low[indices[0]],axis=1)

def mainpage(records, boots, delta):
    fig=sheet("A change in rhythm amplitude can shift the fitted broadband slope",
        "Figure 5 uncertainty analysis   /   30 independent 30 s recordings per amplitude   /   8 Hz rhythm, fixed knee and exponent")
    gs=fig.add_gridspec(2,4,left=.07,right=.985,top=.81,bottom=.17,hspace=.70,wspace=.43)
    f=boots[0]["f"]
    s_true=slope(f,truth(f,0)[0]); alphas=[.32,.5,.73,1]
    for j,title in enumerate(["Generating components",*SHORT]):
        a=fig.add_subplot(gs[0,j]); panel(a, "ABCD"[j], title)
        for i,A in enumerate(AMPS):
            bg,r=truth(f,A) if j==0 else (bb_curve(f,boots[i]["observed"][j-1]),rhythm_curve(f,boots[i]["observed"][j-1],j-1))
            c=INK if j==0 else COL[j-1]
            a.loglog(f,r,color=c,alpha=alphas[i],lw=1.25)
            a.loglog(f,bg,color=c,ls="--",alpha=alphas[i],lw=1)
        a.axvspan(40,60,alpha=.10,color="#a73441")
        a.set(xlim=(1,200),ylim=(2e-5,8),xlabel="Frequency (Hz)")
        a.set_xticks([1,8,50,200],["1","8","50","200"])
        if j==0:a.set_ylabel("Component PSD")
    a=fig.add_subplot(gs[1,0]);panel(a,"E","High − low mean slope")
    for j in range(3):
        q=np.quantile(delta[:,j],[.025,.975])
        a.plot(q,[j,j],color=COL[j],lw=3)
        a.scatter(records[-1,:,j,0].mean()-records[0,:,j,0].mean(),j,c=COL[j],s=24,zorder=3)
    a.axvline(0,color=INK,ls="--",lw=1);a.set_yticks(range(3),["specparam","SL mult.","SL add."])
    a.invert_yaxis();a.margins(y=.25);a.set_xlabel("Difference in slope")
    ylim=(records[:,:,:,0].min()-.018,records[:,:,:,0].max()+.018)
    rng=np.random.default_rng(11)
    for j in range(3):
        a=fig.add_subplot(gs[1,j+1]);panel(a,"FGH"[j],SHORT[j])
        for l in range(4):
            v=records[l,:,j,0];violin(a,v,l,COL[j],interval=False,bins=11)
            a.scatter(l+rng.uniform(-.09,.09,len(v)),v,s=4,c=INK,alpha=.69,zorder=3)
            a.plot(l,v.mean(),"D",color="white",mec=INK,ms=4.2)
        htruth(a,s_true);height_axis(a);a.set_ylim(ylim)
        if j==0:a.set_ylabel("40–60 Hz broadband slope")
    fig.text(.055,.085,"A–D   Solid curves show rhythmic excess in linear power. Dashed curves show broadband. Darker shades indicate larger amplitudes. B–D use record 1.",fontsize=9)
    fig.text(.055,.055,"E   95% intervals for the mean contrast resample 30 records per endpoint.   F–H   Each dot is one fit. Diamonds are means. The dashed line is the generating slope.",fontsize=9)
    return fig

def within_page(boot,post,truth_s):
    fig=sheet("How uncertain is the slope for one recording?",
        "The first simulation seed at each amplitude was selected before fitting. Each pair of shapes refers to the same observed spectrum.")
    gs=fig.add_gridspec(2,3,left=.08,right=.98,top=.79,bottom=.18,hspace=.68,wspace=.32)
    extent=np.concatenate([v["draws"][:,:,0].ravel() for v in boot]+[v[:,:,0].ravel() for row in post for v in row])
    for j in range(3):
        a=fig.add_subplot(gs[0,j]);panel(a,"ABC"[j],SHORT[j])
        for l in range(4):
            v=boot[l]["draws"][:,j,0]
            violin(a,v,l,COL[j],side=-1 if j else 0)
            if j:violin(a,post[l][j-1][:,:,0],l,POST,side=1)
            a.plot(l,boot[l]["observed"][j,0],"D",ms=3.2,color=INK)
        htruth(a,truth_s);height_axis(a);a.set_ylim(np.nanmin(extent)-.01,np.nanmax(extent)+.01)
        if j==0:a.set_ylabel("40–60 Hz broadband slope")
        a=fig.add_subplot(gs[1,j]);panel(a,"DEF"[j],"Knee and slope in refits")
        for l,marker in [(0,"o"),(3,"^")]:
            v=boot[l]["draws"][:,j,:];mask=np.isfinite(v[:,4])
            a.scatter(v[mask,4],v[mask,0],s=7,alpha=.33,marker=marker,
                      c=COL[j] if l==3 else "#818b94",linewidths=0,label=f"A = {AMPS[l]:g}",rasterized=True)
        a.axvline(np.sqrt(60),ls=":",lw=1,c=INK);htruth(a,truth_s)
        a.set_xlabel("Fitted knee (Hz)")
        if j==0:a.set_ylabel("40–60 Hz broadband slope")
        a.legend(fontsize=8,loc="lower right",handletextpad=.2)
    fig.text(.055,.091,"A–C   Method color shows the time-domain bootstrap. Purple shows the SL posterior. Bars span the central 95%, white dots mark medians, and black diamonds are observed fits.",fontsize=9)
    fig.text(.055,.06,"D–F   Each point is a joint bootstrap refit. Only finite positive knee frequencies are shown. Dotted vertical and dashed horizontal lines mark generating values.",fontsize=9)
    return fig

def mechanism(ctrl, f):
    fig=sheet("Where does specparam put the added rhythmic power?",
        "Fit exact spectra on the same 1–200 Hz grid. These controls remove sampling variation and multitaper estimation from the comparison.")
    gs=fig.add_gridspec(2,2,left=.09,right=.97,top=.80,bottom=.17,hspace=.58,wspace=.29)
    axes=[fig.add_subplot(idx) for idx in gs]
    a,b,c,d=axes
    for l,A in enumerate((.5,4.)):
        color=("#73899c","#235788")[l]
        bg,r=truth(f,A)
        row=ctrl[(ctrl.control=="additive truth")&(ctrl.amplitude==A)].iloc[0]
        theta=row[list(FIELDS)].to_numpy(float)
        a.plot(f,np.log10(1+r/bg),color=color,lw=2,label=f"A = {A:g}")
        a.plot(f,theta[6]*np.exp(-.5*((f-theta[5])/theta[7])**2),ls="--",color=color,lw=1.5)
        b.semilogx(f,bb_curve(f,theta)/bg,color=color,lw=2,label=f"A = {A:g}")
    panel(a,"A","Rhythmic excess above broadband"); a.set(xlim=(1,26),xlabel="Frequency (Hz)",ylabel="Excess in log₁₀ power")
    a.legend(fontsize=8);panel(b,"B","Fitted / generating broadband")
    b.axhline(1,color=INK,ls="--",lw=1);b.axvspan(40,60,color="#c15e72",alpha=.16)
    b.set(xlim=(1,200),xlabel="Frequency (Hz)",ylabel="Power ratio");b.set_xticks([1,8,50,200],["1","8","50","200"]);b.legend(fontsize=8)
    for ax,pop,title,letter in [(c,"additive truth","Additive generating spectrum","C"),
                                 (d,"multiplicative truth","Log-Gaussian generating spectrum","D")]:
        for key,label,col,ls in [(pop,"specparam",COL[0],"-"),
                  (pop+" (joint fit)","Joint log-power fit","#a84954","-")]:
            v=ctrl[ctrl.control==key]; ax.plot(range(4),v.slope40_60,color=col,ls=ls,marker="o",ms=4,label=label)
        htruth(ax,slope(f,truth(f,0)[0]));height_axis(ax);panel(ax,letter,title)
        ax.set_ylim(-2.075,-1.918);ax.set_ylabel("40–60 Hz broadband slope");ax.legend(fontsize=8)
    fig.text(.055,.089,"A   Solid lines are log₁₀(1 + rhythm / broadband). Dashed lines are specparam’s fitted Gaussian bumps. B shows the compensating change in the fitted broadband.",fontsize=9)
    fig.text(.055,.058,"C–D   The joint fit optimizes the broadband and log-peak parameters together. The multiplicative control matches the additive spectrum’s total power at 8 Hz.",fontsize=9)
    return fig

def parameter_page(m,boot,post):
    subtitle = ("Four representative recordings   /   Method-colored MAP refits and purple posterior draws   /   Bars span the central 95%"
                if m else "Four representative recordings   /   399 specparam bootstrap refits per recording   /   Bars span the central 95%")
    fig=sheet(f"{SHORT[m]} parameter distributions", subtitle)
    gs=fig.add_gridspec(2,3,left=.075,right=.975,top=.79,bottom=.17,hspace=.64,wspace=.33)
    params=[(1,"Log₁₀ numerator"),(2,"Broadband exponent"),(4,"Knee (Hz)"),(5,"Peak center (Hz)"),
            (6,"Linear peak amplitude" if m==2 else "Log₁₀ peak amplitude"),(7,"Peak width σ (Hz)")]
    for i,(k,name) in enumerate(params):
        a=fig.add_subplot(gs[i//3,i%3]);panel(a,"ABCDEF"[i],name)
        for l,A in enumerate(AMPS):
            v=boot[l]["draws"][:,m,k]
            violin(a,v,l,COL[m],side=-1 if m else 0)
            if m: violin(a,post[l][m-1][:,:,k],l,POST,side=1)
            a.plot(l,boot[l]["observed"][m,k],"D",ms=3,c=INK)
            if k in (1,2,4) or (m==2 and k in (5,6,7)):
                value={1:.5,2:2.,4:np.sqrt(60),5:8.,6:A,7:3.}[k]
                a.hlines(value,l-.38,l+.38,ls="--",color=INK,lw=.8)
        height_axis(a)
    fig.text(.055,.087,"Black diamonds mark observed fits. Dashed segments mark the generating broadband parameters and, for the additive model, the generating peak parameters.",fontsize=9)
    fig.text(.055,.057,"Widths are scaled within each half-violin. Peak and knee-frequency summaries use finite estimates. Raw specparam knee coefficients and finite counts are included in the tables.",fontsize=9)
    return fig

def interval_page(perfit, truth_s):
    fig=sheet("Specparam uncertainty across all 120 recordings",
        "Each row is one 30 s record. Each interval comes from simulating and refitting 399 surrogate records using that recording’s own multitaper pilot.")
    gs=fig.add_gridspec(1,4,left=.065,right=.97,top=.80,bottom=.18,wspace=.13)
    lo=min(perfit.lower.min(),truth_s)-.01;hi=max(perfit.upper.max(),truth_s)+.01
    for l,A in enumerate(AMPS):
        a=fig.add_subplot(gs[0,l]);panel(a,"ABCD"[l],f"Amplitude {A:g}")
        dat=perfit[perfit.level==l].sort_values("record")
        a.axvline(truth_s,color=INK,ls="--",lw=1.2)
        for row in dat.itertuples():
            color=COL[0] if row.lower<=truth_s<=row.upper else "#a54952"
            a.plot([row.lower,row.upper],[row.record]*2,c=color,lw=1.3)
            a.scatter(row.observed,row.record,s=9,c=color)
        a.set(xlim=(lo,hi),ylim=(30.8,.2),xlabel="40–60 Hz broadband slope")
        if l==0:a.set_ylabel("Recording number")
        else:a.tick_params(axis="y",labelleft=False)
        a.set_yticks([1,5,10,15,20,25,30])
    fig.text(.055,.089,"Dots give original fits. Lines give percentile intervals for the refitted estimator. Red intervals miss the known broadband slope, shown by the dashed vertical line.",fontsize=9)
    fig.text(.055,.056,"The fraction covering truth measures recovery in this specific simulation. Bootstrap refits can repeat the same model-dependent error as the original estimator.",fontsize=9)
    return fig

def render():
    files=lambda p: dict(np.load(OUT/p))
    b=[files(f"boot_{l}_0.npz") for l in range(4)]
    p=[[files(f"post_{l}_0_{m}.npz")["draws"] for m in range(2)] for l in range(4)]
    rec=np.stack([files(f"records_{l}.npz")["fits"] for l in range(4)])
    for l,data in enumerate(b):
        np.testing.assert_allclose(rec[l,0], data["observed"],equal_nan=True)
    f=b[0]["f"];true=float(slope(f,truth(f,0)[0]))
    delta=contrasts(rec); ctrl=pd.read_csv(OUT/"population_controls.csv")
    spec_rows=[]; summaries=[];diag=[]
    for l in range(4):
        for r in range(30):
            data=b[l] if r==0 else files(f"spec_{l}_{r}.npz")
            fits=data["draws"][:,0,:] if r==0 else data["draws"]
            point=data["observed"][0] if r==0 else data["observed"]
            np.testing.assert_allclose(point,rec[l,r,0],equal_nan=True)
            lo,hi=np.nanquantile(fits[:,0],[.025,.975])
            spec_rows.append((l,AMPS[l],r+1,point[0],lo,hi,np.isfinite(fits[:,0]).sum(),lo<=true<=hi))
        for j in range(3):
            groups=[("bootstrap",b[l]["draws"][:,j,:])]
            if j:
                groups.append(("posterior",p[l][j-1].reshape(-1,len(FIELDS))))
                run=files(f"post_{l}_0_{j-1}.npz")
                df=pd.read_csv(OUT/f"post_{l}_0_{j-1}.csv")
                diag.append((AMPS[l],METHODS[j],np.sum(run["divergences"]),run["depth_hits"].item(),
                             df.r_hat.max(),df.ess_bulk.min(),df.ess_tail.min(),min(run["bfmi"])))
            for kind,rows in groups:
                for k,name in enumerate(FIELDS):
                    vals=rows[:,k];valid=vals[np.isfinite(vals)]
                    q=np.quantile(valid,[.025,.5,.975]) if len(valid) else [np.nan]*3
                    summaries.append([AMPS[l],METHODS[j],kind,name,len(valid),len(vals),b[l]["observed"][j,k],*q])
    stats=pd.DataFrame(spec_rows,columns=["level","amplitude","record","observed","lower","upper","finite","covers_truth"])
    stats.to_csv(OUT/"specparam_120_intervals.csv",index=False)
    pd.DataFrame(summaries,columns=["amplitude","method","distribution","parameter","finite","attempted","observed","q025","q50","q975"]).to_csv(OUT/"parameter_intervals.csv",index=False)
    pd.DataFrame(diag,columns=["amplitude","method","divergences","max_depth_hits","max_rhat","min_bulk_ess","min_tail_ess","min_bfmi"]).to_csv(OUT/"posterior_diagnostics.csv",index=False)
    effects=rec[-1,:,:,0].mean(0)-rec[0,:,:,0].mean(0)
    eq=np.quantile(delta,[.025,.975],axis=0)
    table=pd.DataFrame({"method":METHODS,"mean_low":rec[0,:,:,0].mean(0),"mean_high":rec[-1,:,:,0].mean(0),"high_minus_low":effects,"lower":eq[0],"upper":eq[1]})
    table.to_csv(OUT/"group_contrasts.csv",index=False)
    print(table.to_string(index=False));print(stats.groupby("amplitude").covers_truth.sum())
    pages=[("Figure5_overview",mainpage(rec,b,delta)),("Within_record",within_page(b,p,true)),
           ("Mechanism_controls",mechanism(ctrl,f)),("Specparam_120_intervals",interval_page(stats,true))]
    pages.extend((f"Parameters_{j}",parameter_page(j,b,p)) for j in range(3))
    with PdfPages(OUT/"Figure5_bootstrap.pdf") as pdf:
        for name,fig in pages:
            fig.savefig(OUT/(name+".svg"),dpi=300)
            fig.savefig(OUT/(name+".png"),dpi=500)
            if name=="Figure5_overview":fig.savefig(OUT/"Figure5_preview.png",dpi=140)
            pdf.savefig(fig,dpi=300);plt.close(fig)

if __name__=="__main__":
    render()
