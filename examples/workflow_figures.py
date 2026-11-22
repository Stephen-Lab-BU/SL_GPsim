"""Render the two workflows from workflow_data.py outputs. All figure coordinates are normalized."""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullLocator
from matplotlib import font_manager
from empirical_violin import violin

OUT=Path(__file__).resolve().parent/"out"
d=np.load(OUT/"spectra.npz")
b=np.load(OUT/"parameters.npz")
f, fa=d["f"],d["fa"]
band=(f>=3)&(f<=45)
ba=(fa>=3)&(fa<=45)
posterior = np.load(OUT / "posterior.npz")["draws"].reshape(-1, 6)
n_boot = b["draws"].shape[0]
t=np.arange(257)/128
ink="#152B3C"; muted="#455968"; edge="#C6D2D8"; grid="#E3E8E9"
blue="#3178A6"; teal="#008575"; orange="#BA5434"; purple="#80579B"
fonts={font.name for font in font_manager.fontManager.ttflist}
mpl.rcParams.update({"font.family":"Lato" if "Lato" in fonts else "DejaVu Sans",
                     "pdf.fonttype":42,"svg.fonttype":"none","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":edge,"axes.labelcolor":muted,"axes.labelsize":10,
                     "xtick.color":muted,"ytick.color":muted,"xtick.labelsize":9,
                     "ytick.labelsize":9,"legend.frameon":False,"legend.fontsize":9,
                     "lines.solid_capstyle":"round","figure.facecolor":"white"})

def txt(fig,x,y,s,size=11,color=ink,weight="normal",**kw):
    return fig.text(x,y,s,fontsize=size,color=color,weight=weight,ha=kw.pop("ha","left"),
                    va=kw.pop("va","center"),**kw)

def rule(fig,x1,y1,x2,y2,c=edge,lw=0.8):
    fig.add_artist(Line2D([x1,x2],[y1,y2],transform=fig.transFigure,color=c,lw=lw))

def arrow(fig,start,end):
    fig.add_artist(FancyArrowPatch(start,end,transform=fig.transFigure,arrowstyle="-|>",
        color="#768A97",mutation_scale=13,lw=1.25,shrinkA=0,shrinkB=0))

def heading(fig,x,y,letter,label):
    txt(fig,x,y,letter,size=11,weight="bold",color="white",ha="center",
        bbox=dict(boxstyle="circle,pad=0.28",facecolor=ink,edgecolor="none"))
    txt(fig,x+.022,y,label,size=16,weight="bold")

def base(title,subtitle,sz):
    fig=plt.figure(figsize=sz)
    txt(fig,.041,.948,title,26,weight="bold")
    txt(fig,.042,.907,subtitle,13,color=muted)
    rule(fig,.041,.878,.96,.878)
    return fig

def axes(fig,rect):
    a=fig.add_axes(rect)
    a.grid(axis="y",color=grid,lw=.7);a.set_axisbelow(True)
    a.tick_params(axis="both",length=3,width=.7,pad=4)
    a.margins(x=0)
    return a

def freq(a,ylog=False,showx=True):
    a.set_xlim(3,45);a.set_xticks([5,10,20,30,40])
    if showx: a.set_xlabel("Frequency (Hz)",labelpad=3)
    else: a.tick_params(axis="x",labelbottom=False)
    if ylog:
        a.set_yscale("log");a.set_ylim(.002,3.6)
        a.set_yticks([.01,.1,1],labels=["0.01","0.1","1"])
        a.yaxis.set_minor_locator(NullLocator())

def trace(a,x,col):
    a.axhline(0,color=edge,lw=.7)
    a.plot(t,x[:257],color=col,lw=.94)
    a.set_xlim(0,2);a.set_xticks([0,.5,1,1.5,2]);a.set_xlabel("Time (s)",labelpad=3)
    a.set_ylim(-10,10);a.set_yticks([-8,0,8])

def legend(a,cols=1):
    a.legend(loc="upper right",handlelength=1.65,columnspacing=.8,ncols=cols,
             borderpad=.35,facecolor="white",frameon=True,edgecolor="none",framealpha=.92)

def commontruth(a,arr,qual=None):
    a.plot(f[band],arr[band],color=ink,lw=1.85,label=qual or "Ground truth",zorder=4)

def make_uni():
    fig=base("Univariate spectral decomposition",
             "8 s at 128 Hz  •  Additive ground truth: exponent 1.60, knee 1.54 Hz, beta center 20 Hz, σ = 1.80 Hz, peak height 0.70 µV²/Hz",(15.3,10.2))
    for x,letter,label in [(.049,"A","Generating spectrum"),(.376,"B","Gaussian time series"),(.701,"C","Direct spectral estimate")]:
        heading(fig,x,.837,letter,label)
    a=axes(fig,[.083,.569,.224,.199])
    a.plot(f[band],d["background"][band],color=purple,lw=1.45,ls="--",label="Broadband")
    a.plot(f[band],d["beta"][band],color=orange,lw=1.5,ls=":",label="Beta")
    commontruth(a,d["true"],"Total")
    freq(a,True);a.set_ylabel("PSD (µV²/Hz)")
    handles,labels=a.get_legend_handles_labels()
    a.legend([handles[i] for i in [2,0,1]],[labels[i] for i in [2,0,1]],loc="upper right",
             facecolor="white",frameon=True,edgecolor="none",framealpha=.9)
    a=axes(fig,[.402,.569,.22,.199]);trace(a,d["x"][:,0],ink);a.set_ylabel("Voltage (µV)")
    txt(fig,.514,.79,"First 2 s of the 8 s record",9.7,color=muted,ha="center")
    a=axes(fig,[.73,.569,.222,.199])
    a.plot(fa[ba],d["estimate"][ba],color=blue,lw=.92,label="Multitaper")
    commontruth(a,d["true"]);freq(a,True);a.set_ylabel("PSD (µV²/Hz)")
    legend(a)
    txt(fig,.84,.79,"DPSS multitaper, NW = 2, K = 3",9.7,color=muted,ha="center")
    arrow(fig,(.323,.67),(.355,.67));arrow(fig,(.648,.67),(.679,.67))
    # Connect the estimated spectrum to the bootstrap fit panel.
    rule(fig,.869,.504,.869,.491,c="#768A97",lw=1.15)
    rule(fig,.869,.491,.049,.491,c="#768A97",lw=1.15)
    arrow(fig,(.049,.491),(.049,.462))
    heading(fig,.049,.44,"D","Both methods fit a knee")
    txt(fig,.044,.402,f"Fits use the same 3–45 Hz bins. Brackets show 95% percentile intervals from {n_boot:,} paired bootstrap refits, or equal-tailed intervals from {len(posterior):,} posterior draws.",11.1,color=muted)
    centers=[.28,.408,.54,.668,.794,.914]
    headers=["Offset","Exponent","Knee (Hz)","Beta center (Hz)","Beta height¹","Beta width²"]
    for x,h in zip(centers,headers):
        txt(fig,x,.354,h,12,weight="bold",ha="center")
    q=np.quantile(posterior,[.025,.5,.975],axis=0)
    idx1=[0,1,3,4,5,6]; idx2=[7,8,9,10,11,12]
    rows=[("specparam","Observed fit · bootstrap CI",orange,idx1),
          ("SL_specdecomp MAP","Observed fit · bootstrap CI",teal,idx2),
          ("SL_specdecomp posterior","Median · credible interval",purple,None)]
    for i,(label,detail,color,indices) in enumerate(rows):
        y=.302-i*.078
        rule(fig,.043,y+.028,.957,y+.028)
        txt(fig,.045,y,label,12.6,color=color,weight="bold")
        txt(fig,.045,y-.02,detail,9.3,color=muted)
        for j,x in enumerate(centers):
            if indices is None:
                lo,value,hi=q[:,j]
            else:
                k=indices[j]; value=b["observed"][k]; lo=b["lo"][k]; hi=b["hi"][k]
            txt(fig,x,y+.001,f"{value:.2f}",14,color=color,weight="bold",ha="center")
            txt(fig,x,y-.024,f"[{lo:.2f}, {hi:.2f}]",10.8,ha="center")
    rule(fig,.043,.105,.957,.105)
    txt(fig,.044,.083,"¹ specparam reports log₁₀ excess power. SL reports additive µV²/Hz.   ² specparam reports bandwidth (2σ) in Hz. SL reports σ in Hz.",10.2,color=muted)
    txt(fig,.044,.059,f"specparam returned a positive knee frequency in {b['valid'][3]}/{n_boot} fits and a 15–25 Hz peak in {b['valid'][4]}/{n_boot}. The corresponding intervals condition on those fits.",10.2,color=muted)
    txt(fig,.044,.036,f"Its native knee coefficient k was {b['observed'][2]:.3g} (95% bootstrap CI {b['lo'][2]:.3g} to {b['hi'][2]:.3g}). For positive k and exponent χ, the half-power knee frequency is k raised to 1/χ.",10.2,color=muted)
    return fig

def make_multi():
    fig=base("Multivariate spectral simulation",
             "Two simultaneous channels  •  One 8 s record at 128 Hz  •  Ground truth and direct estimate shown over 3–45 Hz",(15.3,10.2))
    for x,l,name in [(.049,"A","Generating spectral matrix"),(.379,"B","Joint time series"),(.684,"C","Direct spectral matrix estimate")]:
        heading(fig,x,.836,l,name)
    txt(fig,.048,.793,"One-sided density (µV²/Hz)",10.5,color=muted)
    txt(fig,.377,.793,"First 2 s of both channels",10.5,color=muted)
    txt(fig,.682,.793,"DPSS multitaper, NW = 2, K = 3",10.5,color=muted)
    spec=d["matrix"]; est=d["sm"]
    components=[("Channel 1 power",spec[:,0,0].real,est[:,0,0].real),
                ("Channel 2 power",spec[:,1,1].real,est[:,1,1].real),
                ("Real cross-spectrum",spec[:,1,0].real,est[:,1,0].real),
                ("Imaginary cross-spectrum",spec[:,1,0].imag,est[:,1,0].imag)]
    rows=[.622,.464,.306,.148]
    for col,left in enumerate([.084,.721]):
        for j,(name,g,e) in enumerate(components):
            ax=axes(fig,[left,rows[j],.232,.099]);freq(ax,j<2,j==3)
            if j>=2:
                ax.set_ylim(-1,1);ax.set_yticks([-.8,0,.8]);ax.axhline(0,color=edge,lw=.85)
            if col: ax.plot(fa[ba],e[ba],color=blue,lw=1.0,label="Multitaper")
            commontruth(ax,g)
            label=["S₁₁","S₂₂","Re S₂₁","Im S₂₁"][j]
            txt(fig,left,rows[j]+.116,name+"  ·  "+label,10.6,weight="bold")
    # Ground truth and estimate retain consistent colors for all four entries.
    handles=[Line2D([],[],color=ink,lw=2,label="Ground truth"),Line2D([],[],color=blue,lw=1.4,label="Multitaper")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.835,.061),ncols=2,fontsize=11,frameon=False)
    for j,(bottom,col) in enumerate([(.569,teal),(.408,orange)]):
        ax=axes(fig,[.402,bottom,.213,.113])
        trace(ax,d["x2"][:,j],col)
        if j==0:ax.set_xlabel("");ax.tick_params(axis="x",labelbottom=False)
        ax.set_ylabel("µV",labelpad=2)
        txt(fig,.40,bottom+.132,f"Channel {j+1}",11,weight="bold",color=col)
    arrow(fig,(.327,.523),(.36,.523));arrow(fig,(.628,.523),(.667,.523))
    rule(fig,.393,.241,.616,.241)
    txt(fig,.393,.213,"Cross-channel structure",12,weight="bold")
    txt(fig,.393,.162,"Both channels contain a 20 Hz rhythm.\nIn channel 2, that rhythm is delayed by\ntwo samples (15.6 ms). Channel 2 also\ncontains an independent 10 Hz rhythm.",
        10.4,color=muted,linespacing=1.42)
    txt(fig,.044,.061,"S₂₁ uses X₂ × conjugate(X₁). The delay gives a −112.5° phase at 20 Hz; S₁₂ is the conjugate of S₂₁.",10.4,color=muted)
    return fig

def make_posterior():
    fig=base("Bootstrap sampling distributions and the SL posterior",
             "One observed 8 s spectrum  •  Six fitted parameters  •  Same additive Gamma model, fit band and prior rules",(15.3,10.2))
    handles=[Line2D([],[],color=teal,lw=2.2,label=f"{n_boot:,} bootstrap MAP fits"),
             Line2D([],[],color=purple,lw=2.2,label=f"{len(posterior):,} posterior draws"),
             Line2D([],[],color=ink,lw=1.3,ls="--",label="Generating value")]
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.036,.844),ncols=3,fontsize=12,frameon=False)
    txt(fig,.044,.801,"Violin widths show relative draw counts. Bars span the central 95% and white points mark medians.",11.8,color=muted)
    post=posterior
    names=["Broadband offset (log₁₀ numerator)","Broadband exponent","Knee frequency (Hz)","Beta center (Hz)","Beta height (µV²/Hz)","Beta width σ (Hz)"]
    truth=[np.log10(8),1.6,2**(1/1.6),20,.70,1.8]
    for i,title in enumerate(names):
        row,col=divmod(i,3)
        left=[.078,.401,.723][col]; bottom=[.536,.217][row]
        a=axes(fig,[left,bottom,.228,.172])
        a.set_title(title,loc="left",fontsize=12.3,weight="bold",color=ink,pad=9)
        a.set_yticks([]);a.set_ylim(-.6,1.6)
        a.axvline(truth[i],ls="--",color=ink,lw=1.1)
        boot=b["draws"][:,7+i]
        violin(a,boot,1,teal,horizontal=True,halfwidth=.46,bins=35)
        violin(a,post[:,i],0,purple,horizontal=True,halfwidth=.46,bins=35)
        a.margins(x=.04)
        bl,bh=b["lo"][7+i],b["hi"][7+i]
        pl,ph=np.quantile(post[:,i],[.025,.975])
        txt(fig,left,bottom-.042,f"Bootstrap CI    {bl:.2f}–{bh:.2f}",10.4,color=teal,weight="bold")
        txt(fig,left,bottom-.065,f"Posterior CrI    {pl:.2f}–{ph:.2f}",10.4,color=purple,weight="bold")
    rule(fig,.043,.113,.957,.113)
    bound_count = np.isclose(b["draws"][:, 9], 1, rtol=1e-6).sum()
    txt(fig,.044,.089,f"The generating knee was 1.54 Hz, below the 3 Hz fit cutoff. The MAP estimate reached its 1 Hz lower bound in {bound_count}/{n_boot} bootstrap draws. The prior is uniform in log knee frequency from 1 to 45 Hz.",10.1,color=muted)
    txt(fig,.044,.063,"MAP refits retaper each simulated record and recompute data-scaled priors. The posterior uses the observed PSD and a product of Gamma likelihoods across frequency bins.",10.3,color=muted)
    run = json.loads((OUT / "nuts_run.json").read_text())
    txt(fig,.044,.036,
        f"PyMC NUTS used {run['chains']} chains with {run['draws_per_chain']:,} retained draws each. "
        f"Divergences {run['divergences']}, maximum R-hat {run['max_rhat']:.3f}, "
        f"minimum bulk ESS {run['min_ess_bulk']:,.0f}.",10.3,color=muted)
    return fig


def make_code():
    fig=base("Simulating the records in these figures",
             "Build a spectrum from named components, then draw an eight-second Gaussian record.",(15.3,10.2))
    heading(fig,.049,.823,"A","One channel")
    heading(fig,.534,.823,"B","Two channels")
    txt(fig,.045,.771,"The raw knee coefficient is 2. The half-power knee is 1.54 Hz.",11,color=muted)
    txt(fig,.531,.771,"Channel 2 carries the shared beta rhythm two samples later.",11,color=muted)
    body_left="""import numpy as np
from SL_GPsim import Broadband, GaussianPeak, SpectralGrid

grid = SpectralGrid(fs=128, duration=8)
bg = Broadband(scale=8, knee=2, exponent=1.6)
beta = GaussianPeak(center_hz=20, sigma_hz=1.8,
                    height=0.70)
single = bg + beta
x = grid.simulate(single, random_state=7)"""
    body_right="""from SL_GPsim import Independent, Shared

private = (Broadband(scale=5.4, knee=4, exponent=1.4)
           + GaussianPeak(center_hz=10, sigma_hz=1.1,
                          height=0.44))
pair = (Independent(bg, private)
        + Shared(beta, gains=[1, 0.7],
                 delays_seconds=[0, 0.015625]))
xy = grid.simulate(pair, random_state=112)"""
    for left,txtcode in [(.045,body_left),(.531,body_right)]:
        txt(fig,left,.729,txtcode,size=10.0,va="top",fontfamily="DejaVu Sans Mono",linespacing=1.47)
    rule(fig,.044,.469,.957,.469)
    heading(fig,.049,.427,"C","Evaluate or export the generating model")
    heading(fig,.534,.427,"D","Work with a recording")
    left_code = """f = np.linspace(0, grid.fs / 2, 10_001)
curve = single(f)           # any frequencies in Hz
matrix_curve = pair(f)      # full complex CSD
target = grid.sample(pair)  # exact simulation bins
settings = pair.parameters  # JSON-compatible"""
    right_code = """from SL_GPsim import multitaper_csd, simulate_like

t = grid.times              # seconds
freqs, estimate = multitaper_csd(xy, fs=128)
replicate = simulate_like(xy, fs=128, random_state=7)"""
    for left,txtcode in [(.045,left_code),(.531,right_code)]:
        txt(fig,left,.378,txtcode,size=10.1,va="top",fontfamily="DejaVu Sans Mono",linespacing=1.48)
    txt(fig,.045,.172,"Eight seconds at 128 Hz gives 1,024 time samples. The default embedding has 1,025 nonnegative frequency bins.",11,color=muted)
    txt(fig,.045,.123,"The smooth curve can be evaluated anywhere in frequency. The simulation target applies the real-FFT convention at DC and even-grid Nyquist.",11,color=muted)
    return fig


if __name__=="__main__":
    with PdfPages(OUT/"Spectral_workflows.pdf") as pdf:
        for name,func in [("Univariate_workflow",make_uni),("Bootstrap_vs_posterior",make_posterior),("Multivariate_workflow",make_multi),("Simulation_code",make_code)]:
            fig=func()
            fig.savefig(OUT/(name+".png"),dpi=500)
            fig.savefig(OUT/(name+".svg"))
            pdf.savefig(fig)
            fig.savefig(OUT/(name+"_preview.png"),dpi=95)
            plt.close(fig)
