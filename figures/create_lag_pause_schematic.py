"""Conceptual training lag and relaxation during a pause; synthetic densities."""
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent
ink,blue,orange,green='#21354c','#285b8c','#df8439','#37856b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':ink})
fig=plt.figure(figsize=(13.5,8.1),facecolor='white')
gs=fig.add_gridspec(2,3,height_ratios=[1.25,1],hspace=.65,wspace=.20)
x=np.linspace(-3.8,4.5,1000)
model=.55*norm.pdf(x,-1.1,.7)+.45*norm.pdf(x,1.45,.65)
chain_weights=[.12,.27,.41]
for i,w in enumerate(chain_weights):
    ax=fig.add_subplot(gs[0,i])
    chains=(1-w)*norm.pdf(x,-1.1,.7)+w*norm.pdf(x,1.45,.65)
    ax.fill_between(x,model,color=orange,alpha=.12)
    ax.plot(x,model,color=orange,lw=2.4)
    ax.fill_between(x,chains,color=blue,alpha=.10)
    ax.plot(x,chains,color=blue,lw=2.1)
    ax.set_xlim(-3.5,4.0);ax.set_ylim(0,.60);ax.set_axis_off()
    ax.axhline(0,color='#c4ced7',lw=.8)
    ax.set_title(['Model moves; chains lag','Pause: freeze parameters','Chains catch up'][i],
                 loc='left',fontsize=13,weight='bold',pad=14)
    if i==0:
        ax.annotate('New basin is underpopulated',xy=(1.45,.09),xytext=(-.2,.48),
                    ha='center',fontsize=10,color=blue,
                    arrowprops=dict(arrowstyle='->',color=blue,lw=1.1,connectionstyle='arc3,rad=-.2'))
    elif i==1:
        ax.annotate('',xy=(1.2,.34),xytext=(-.65,.34),
                    arrowprops=dict(arrowstyle='->',color=blue,lw=1.5,connectionstyle='arc3,rad=-.3'))
        ax.text(.5,-.10,'MCMC + swaps continue',transform=ax.transAxes,ha='center',color=blue,fontsize=11)
    else:
        ax.text(.5,-.10,'Resume parameter updates',transform=ax.transAxes,ha='center',color=green,fontsize=11)
# Lower plot: schematic lag rises under forcing, decays with parameters frozen.
ax=fig.add_subplot(gs[1,:])
start,end=4.,8.
t1=np.linspace(0,start,180);y1=.045+.255*(t1/start)**1.7
t2=np.linspace(start,end,180);y2=.30*np.exp(-np.log(.30/.125)*(t2-start)/(end-start))
t3=np.linspace(end,10.3,100);y3=.125+.023*(t3-end)+.006*np.sin((t3-end)*3)
ax.axvspan(start,end,color=green,alpha=.09,lw=0)
ax.axhline(.25,color='#a0adba',ls=(0,(4,3)),lw=1)
ax.axhline(.125,color=green,ls=(0,(4,3)),lw=1)
ax.plot(t1,y1,color=orange,lw=2.7)
ax.plot(t2,y2,color=green,lw=2.7)
ax.plot(t3,y3,color=orange,lw=2.7)
ax.scatter([start,end],[.30,.125],c=[ink,green],s=45,edgecolors='white',lw=1,zorder=5)
ax.text(10.45,.263,'Trigger: 0.25',va='bottom',fontsize=10,color='#71808f')
ax.text(10.45,.138,'Resume target: 0.125',va='bottom',fontsize=10,color=green)
ax.text(2.,.355,'Training',ha='center',fontsize=11,color=orange)
ax.text(6.,.355,'Lag-pause',ha='center',fontsize=11,color=green,weight='bold')
ax.text(9.2,.355,'Training',ha='center',fontsize=11,color=orange)
ax.annotate('Stop updates',xy=(start,.30),xytext=(start-.8,.325),ha='right',fontsize=9,color=ink,
            arrowprops=dict(arrowstyle='-',color=ink,lw=.7))
ax.set_xlim(-.1,12.1);ax.set_ylim(0,.39)
ax.set_xlabel('Sampling / training time',labelpad=10)
ax.set_ylabel('Estimated lag',labelpad=10)
ax.set_xticks([]);ax.set_yticks([])
for sp in ['top','right']:ax.spines[sp].set_visible(False)
for sp in ['left','bottom']:ax.spines[sp].set_color('#b8c4ce')
fig.text(.06,.965,'Let the chains catch up',fontsize=23,weight='bold',color=ink)
fig.text(.06,.923,'Freeze the model while correlations with past parameter updates relax.',fontsize=12,color='#697d90')
fig.legend(handles=[Line2D([0],[0],color=orange,lw=2.4,label='Current model: equilibrium distribution'),
                    Line2D([0],[0],color=blue,lw=2.1,label='Actual chain population')],
           loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=11)
fig.text(.5,.045,r'$\mathrm{lag}_O(t)\;\approx\;\operatorname{Cov}\!\left[z_O(x_t),\,M\right]$'
         '   •   Memory of past energy changes',ha='center',fontsize=13,color=ink)
fig.subplots_adjust(left=.065,right=.96,top=.78,bottom=.16)
for ext in ['png','pdf','svg']:
    p=OUT/f'lag_pause_training_schematic.{ext}'
    fig.savefig(p,dpi=280,facecolor='white')
    print(p)
