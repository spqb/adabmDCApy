"""Conceptual ladder compression: discard old rungs, retain an equilibrium reservoir."""
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parent
blue,orange,teal,grey,ink='#285b8c','#df8439','#37856b','#9ba8b5','#21354c'
x=np.linspace(-2.7,12.,2000)
means=[0,2.1,4.2,6.3,8.4]
rng=np.random.default_rng(27)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink})
fig,axes=plt.subplots(2,1,figsize=(12.5,7.5),sharex=True,sharey=True)
for i,ax in enumerate(axes):
    for k,mu in enumerate(means):
        d=norm.pdf(x-mu,scale=.85)
        if i==1 and k<3:
            continue
        c=orange if k==4 else blue
        if i==0 and k<3:
            c=grey
        ax.fill_between(x,d,color=c,alpha=.10)
        ax.plot(x,d,color=c,lw=2.1)
    ax.axhline(0,color='#c7cfd8',lw=.8)
    ax.set_xlim(-2.7,11.5);ax.set_ylim(-.22,.69);ax.set_axis_off()
    if i==0:
        ax.set_title('Before compression',loc='left',fontsize=15,weight='bold',pad=10)
        ax.annotate('',xy=(5.3,-.055),xytext=(-1.15,-.055),
                    arrowprops=dict(arrowstyle='|-|',color=grey,lw=1.2))
        ax.text(2.1,-.14,'Earlier rungs: discarded after reservoir collection',ha='center',fontsize=11,color='#71808f')
        ax.annotate('Collect equilibrium configurations',xy=(6.3,.47),xytext=(5.3,.62),
                    ha='center',fontsize=11,color=teal,
                    arrowprops=dict(arrowstyle='->',color=teal,lw=1.3,connectionstyle='arc3,rad=-.2'))
    else:
        ax.set_title('After compression',loc='left',fontsize=15,weight='bold',pad=10)
        box=FancyBboxPatch((-.5,.02),4.6,.37,boxstyle='round,pad=0.02,rounding_size=0.07',
                           facecolor='#edf6f1',edgecolor=teal,lw=1.25,zorder=4)
        ax.add_patch(box)
        # The box is a collection, not a spatial projection of the configurations.
        sx=rng.uniform(-.3,3.9,170);sy=rng.uniform(.055,.35,170)
        ax.scatter(sx,sy,s=12,color=teal,alpha=.68,linewidths=0,zorder=5)
        ax.text(1.8,-.075,'Reservoir',ha='center',fontsize=12,color=teal,weight='bold')
        ax.text(1.8,-.15,'Equilibrium configurations of the new bottom rung',ha='center',fontsize=10,color=teal)
        ax.annotate('',xy=(5.8,.17),xytext=(4.15,.17),
                    arrowprops=dict(arrowstyle='->',color=teal,lw=1.4),zorder=6)
        ax.text(4.95,.26,'Draw configurations',ha='center',fontsize=10,color=teal)
        ax.annotate('',xy=(8.05,.56),xytext=(6.65,.56),
                    arrowprops=dict(arrowstyle='<->',color=ink,lw=1.2))
        ax.text(7.35,.60,'Swap',ha='center',fontsize=11)
        ax.text(6.3,-.095,'New bottom rung',ha='center',fontsize=11,color=blue)
        ax.text(8.4,-.095,'Endpoint',ha='center',fontsize=11,color=orange)
fig.legend(handles=[Line2D([0],[0],color=grey,lw=2,label='Earlier ladder models'),
                    Line2D([0],[0],color=blue,lw=2,label='Retained fixed model'),
                    Line2D([0],[0],color=orange,lw=2,label='Evolving model'),
                    Line2D([0],[0],color=teal,marker='o',ls='',markersize=5,label='Stored configurations')],
           loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,ncol=4,fontsize=10)
fig.subplots_adjust(left=.05,right=.98,top=.92,bottom=.13,hspace=.36)
for ext in ['png','pdf','svg']:
    path=OUT/f'ptt_reservoir_compression.{ext}'
    fig.savefig(path,dpi=300,facecolor='white')
    print(path)
