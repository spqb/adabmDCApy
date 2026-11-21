"""Conceptual 1D illustration of replacing an explicit rung by a reservoir."""
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parent
blue,orange,teal,ink='#285b8c','#df8439','#37856b','#21354c'
x=np.linspace(-3.3,8.,1500)
means=[0,2.3,4.6]
rng=np.random.default_rng(27)
# Illustrative draws, not samples of a fitted biological model.
samples=np.clip(rng.normal(size=180),-2.65,2.65)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink})
fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharex=True,sharey=True)
for i,ax in enumerate(axes):
    for k,mu in enumerate(means):
        d=norm.pdf(x-mu)
        if i==1 and k==0:
            ax.plot(x,d,color=teal,lw=1.5,ls=(0,(4,3)),alpha=.65)
            continue
        c=orange if k==2 else blue
        ax.fill_between(x,d,color=c,alpha=.10)
        ax.plot(x,d,color=c,lw=2.2)
    ax.axhline(0,color='#bac5d1',lw=.8)
    if i==0:
        ax.text(0,-.055,'Explicit model',ha='center',fontsize=11,color=blue)
        for a,b in [(0,2.3),(2.3,4.6)]:
            ax.annotate('',xy=(b-.3,.47),xytext=(a+.3,.47),
                        arrowprops=dict(arrowstyle='<->',color=ink,lw=1.2))
        ax.text(2.3,.535,'MCMC + swaps',ha='center',fontsize=12)
    else:
        box=FancyBboxPatch((-2.9,-.015),5.8,.135,boxstyle='round,pad=0.012,rounding_size=0.08',
                           facecolor='#edf6f1',edgecolor=teal,lw=1,zorder=4)
        ax.add_patch(box)
        rows=rng.uniform(.005,.10,len(samples))
        ax.scatter(samples,rows,s=10,color=teal,alpha=.65,linewidths=0,zorder=5)
        picked=np.array([24,67,118])
        ax.scatter(samples[picked],rows[picked],s=35,facecolor='#edf6f1',edgecolor=ink,linewidths=1,zorder=6)
        ax.text(0,-.055,'Reservoir\nEquilibrium configurations',ha='center',fontsize=10.5,color=teal)
        ax.annotate('Draw configurations',xy=(0,.405),xytext=(-.1,.535),
                    ha='center',fontsize=11,color=teal,
                    arrowprops=dict(arrowstyle='->',color=teal,lw=1.5,connectionstyle='arc3,rad=-.25'))
        ax.annotate('',xy=(2.,.47),xytext=(.3,.47),
                    arrowprops=dict(arrowstyle='<->',color=ink,lw=1.2))
        ax.text(1.15,.51,'Swap',ha='center',fontsize=11)
        ax.annotate('',xy=(4.3,.47),xytext=(2.6,.47),
                    arrowprops=dict(arrowstyle='<->',color=ink,lw=1.2))
        ax.text(3.45,.51,'Swap',ha='center',fontsize=11)
        ax.text(4.6,-.055,'Active ladder',ha='center',fontsize=11,color=ink)
    ax.set_xlim(-3.3,8.);ax.set_ylim(-.10,.64);ax.set_axis_off()
    ax.set_title(['Explicit ladder','Reservoir + active ladder'][i],fontsize=15,weight='bold',pad=15)
# Transformation arrow between panels.
axes[0].annotate('',xy=(.523,.53),xytext=(.477,.53),xycoords=fig.transFigure,
                 arrowprops=dict(arrowstyle='->',color='#8292a3',lw=1.8))
fig.legend(handles=[Line2D([0],[0],color=blue,lw=2,label='Fixed ladder models'),
                    Line2D([0],[0],color=orange,lw=2,label='Evolving model'),
                    Line2D([0],[0],color=teal,marker='o',ls='',markersize=5,label='Stored configurations')],
           loc='lower center',bbox_to_anchor=(.5,.02),frameon=False,ncol=3,fontsize=11)
fig.subplots_adjust(left=.025,right=.985,top=.82,bottom=.19,wspace=.16)
for ext in ['png','pdf','svg']:
    p=OUT/f'ptt_reservoir_schematic.{ext}'
    fig.savefig(p,dpi=300,facecolor='white')
    print(p)
