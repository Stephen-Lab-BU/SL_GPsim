"""Regenerate README figures. Run workflow_data.py and workflow_posterior.py first."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
from SL_GPsim.figure_style import COLUMN_WIDTH, FIGURE_STYLE, INK, ORANGE, PURPLE, TEAL
from simulate_spectra import figure_models
from SL_GPsim import SpectralGrid, multitaper_csd

def save(fig, name, dest):
    for letter, ax in zip("ab", fig.axes):
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, va="bottom", weight="bold", fontsize=8)
        ax.set_title(ax.get_title(), loc="left", fontsize=7)
        ax.set_title("", loc="center")
    fig.savefig(dest / (name + ".svg"), metadata={"Date": None})
    plt.close(fig)


@mpl.rc_context(FIGURE_STYLE)
def main():
    dest = root / "docs/figures"
    grid = SpectralGrid(fs=128, duration=8)
    background, rhythm, single, pair = figure_models()
    x = grid.simulate(single, random_state=7)
    xy = grid.simulate(pair, random_state=112)
    f, psd = multitaper_csd(x, fs=128)
    _, csd = multitaper_csd(xy, fs=128)
    f_plot = np.linspace(0.125, 48, 1001)
    t = grid.times[:128]

    fig,(a,b)=plt.subplots(1,2,figsize=(COLUMN_WIDTH, 2.55),layout="constrained")
    a.plot(t,x[:128,0],color=TEAL,lw=1.25)
    a.set(title="One simulated channel",xlabel="Time (s)",ylabel="Amplitude (a.u.)",xlim=(0,1))
    band=(f>0)&(f<=48)
    b.semilogy(f[band],psd[band,0,0].real,color="#92C7C4",lw=1,label="Multitaper estimate")
    b.semilogy(f_plot,single(f_plot),color=INK,lw=1.15,label="Generating spectrum")
    b.semilogy(f_plot,background(f_plot),color=ORANGE,lw=1.2,ls="--",label="Broadband alone")
    b.set(title="Spectrum of the full 8 s",xlabel="Frequency (Hz)",ylabel="Power (a.u.²/Hz)",xlim=(0,48))
    b.legend(fontsize=6.5)
    save(fig, "single-channel", dest)

    fig,(a,b)=plt.subplots(1,2,figsize=(COLUMN_WIDTH, 2.55),layout="constrained")
    mask=(f>=12)&(f<=28)
    scale=np.sqrt(csd[:,0,0].real*csd[:,1,1].real)
    coh=np.abs(csd[:,1,0]/scale)**2
    sf=f[mask]; mat=pair(sf)
    truth_coh=np.abs(mat[:,1,0])**2/(mat[:,0,0].real*mat[:,1,1].real)
    a.plot(sf,coh[mask],color="#8ABDBB",lw=1.2,label="Multitaper estimate")
    a.plot(sf,truth_coh,color=INK,lw=1.15,label="Generating spectrum")
    a.set(title="Coherence across channels",xlabel="Frequency (Hz)",ylabel="Magnitude-squared coherence",ylim=(-.035,1.05),xlim=(12,28))
    a.legend(fontsize=6.5,loc="lower center")
    phase=np.angle(csd[mask,1,0],deg=True)
    b.scatter(sf,phase,s=6,color=TEAL,label="Multitaper phase")
    b.plot(sf,np.angle(mat[:,1,0],deg=True),color=INK,lw=1.15,label="Generating phase")
    b.scatter(20,-112.5,s=28,facecolor=ORANGE,zorder=4)
    b.set(title="Phase of channel 2 relative to 1",xlabel="Frequency (Hz)",ylabel="Phase (degrees)",ylim=(-185,185),yticks=[-180,-90,0,90,180],xlim=(12,28))
    b.legend(loc="upper right",fontsize=6.5)
    save(fig, "shared-rhythm", dest)

    boot=np.load(root / "examples/out/parameters.npz")["draws"]
    post=np.load(root / "examples/out/posterior.npz")["draws"].reshape(-1,6)
    fig,(a,b)=plt.subplots(1,2,figsize=(COLUMN_WIDTH, 2.55),layout="constrained")
    for ax,ib,ip,title,units,limits,nbins in [(a,8,1,"Broadband exponent","Exponent",(1,2.3),36),(b,9,2,"Half-power knee","Frequency (Hz)",(1,8),40)]:
        for draw,color,label in [(boot[:,ib],TEAL,"Bootstrap MAP refits"),(post[:,ip],PURPLE,"Posterior draws")]:
            edges=np.linspace(*limits,nbins+1)
            counts=np.histogram(draw,bins=edges)[0]
            ax.stairs(100*counts/len(draw),edges,fill=True,alpha=.23,color=color)
            ax.stairs(100*counts/len(draw),edges,color=color,lw=1,label=label)
        ax.set(title=title,xlabel=units,ylabel="Draws per bin (%)",xlim=limits)
    a.legend(loc="upper right",fontsize=6.5)
    save(fig, "bootstrap-posterior", dest)


if __name__ == "__main__":
    main()
