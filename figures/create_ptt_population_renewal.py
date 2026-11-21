"""Illustrative population renewal: synthetic ages, exact implemented threshold."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parent
ink,old,fresh,grey='#21354c','#df8439','#285b8c','#8295a5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':ink})
fig=plt.figure(figsize=(14,7.3),facecolor='white')
gs=fig.add_gridspec(2,3,height_ratios=[1.15,1],left=.055,right=.965,bottom=.14,top=.76,wspace=.20,hspace=.65)
rng=np.random.default_rng(54)
counts=[[100,100,100,100],[0,20,45,65],[0,0,0,1]]
for frame,nn in enumerate(counts):
    ax=fig.add_subplot(gs[0,frame])
    for k,nold in enumerate(nn):
        x,y=np.meshgrid(np.arange(10),np.arange(10));x=x.ravel()+k*13;y=y.ravel()
        colors=np.full(100,fresh,dtype='<U7');colors[rng.permutation(100)[:nold]]=old
        ax.add_patch(FancyBboxPatch((k*13-1,-1),11,11,boxstyle='round,pad=0.0,rounding_size=.5',
                                    facecolor='#f7f9fb',edgecolor='#dce3e9',lw=.8,zorder=0))
        ax.scatter(x,y,c=colors,s=7,linewidths=0,zorder=2)
        ax.text(k*13+4.5,11.,['Profile','Rung 1','Rung 2','Endpoint'][k],ha='center',fontsize=8.5,
                color=fresh if k in [0,3] else ink)
        ax.text(k*13+4.5,-3.,str(nold)+' old',ha='center',fontsize=9,color=old if nold else grey)
        if k<3:
            ax.annotate('',xy=(k*13+12,4.5),xytext=(k*13+10,4.5),
                        arrowprops=dict(arrowstyle='<->',color=grey,lw=.7))
    ax.annotate('Exact redraws',xy=(4.5,-4.5),xytext=(4.5,-8),ha='center',fontsize=9,color=fresh,
                arrowprops=dict(arrowstyle='->',color=fresh,lw=1.1))
    ax.set_title(['Mark the initial population','Fresh configurations propagate','Renewal criterion reached'][frame],
                 loc='left',fontsize=12,weight='bold',pad=14)
    ax.text(.5,-.23,[r'$N_{\mathrm{old}}=400$',r'$N_{\mathrm{old}}=130$',r'$N_{\mathrm{old}}=1\leq\varepsilon N$'][frame],
            transform=ax.transAxes,ha='center',fontsize=13,color=fresh if frame==2 else ink)
    ax.set_xlim(-2,50);ax.set_ylim(-11,13);ax.set_aspect('equal');ax.set_axis_off()
ax=fig.add_subplot(gs[1,:])
ax.set_position([.22, .18, .65, .17])
t=np.linspace(0,100,401);n=400*np.exp(-np.log(400)*t/100)
ax.plot(t,n,color=old,lw=2.7)
ax.axhline(1,color=fresh,ls=(0,(5,3)),lw=1.3)
mid=np.log(400/130)/np.log(400)*100
ax.scatter([0,mid,100],[400,130,1],s=50,c=[ink,old,fresh],edgecolors='white',lw=1,zorder=4)
ax.axvspan(100,111,color=fresh,alpha=.07,lw=0)
ax.text(109,1.55,r'Renewal: $N_{\mathrm{old}}\leq\varepsilon N=1$',ha='right',fontsize=10,color=fresh)
ax.text(.02,1.12,'Old configurations across all rungs',transform=ax.transAxes,fontsize=11,color=ink)
ax.set_yscale('log');ax.set_ylim(.65,650);ax.set_xlim(-2,111)
ax.set_yticks([1,20,400],labels=['1','20','400']);ax.set_xticks([])
ax.minorticks_off();ax.set_ylabel(r'$N_{\mathrm{old}}$',labelpad=12)
ax.set_xlabel('PTT exchange rounds',labelpad=10)
for sp in ['top','right']:ax.spines[sp].set_visible(False)
for sp in ['left','bottom']:ax.spines[sp].set_color('#bac7d1')
ax.tick_params(axis='y',colors='#697d90',length=3)
fig.text(.055,.95,'Population renewal in PTT sampling',fontsize=23,weight='bold')
fig.text(.055,.90,'Fresh configurations enter at the profile; swaps carry their age labels through the ladder.',fontsize=12,color='#697d90')
fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=old,markersize=6,label='Old: present at the reference round'),
                    Line2D([0],[0],marker='o',ls='',color=fresh,markersize=6,label='Fresh: drawn at the bottom since then')],
           loc='upper center',bbox_to_anchor=(.5,.865),ncol=2,frameon=False,fontsize=10.5)
fig.text(.5,.055,r'$R=4,\quad N=100,\quad\varepsilon=0.01\qquad N_{\mathrm{old}}\leq\varepsilon N\;\Longleftrightarrow\;G\leq\varepsilon/R$',
         ha='center',fontsize=13)
fig.text(.5,.018,'Schematic populations and decay · Local MCMC moves do not reset age labels · Fresh does not mean statistically independent',
         ha='center',fontsize=9,color='#697d90')
for ext in ['png','pdf','svg']:
    p=OUT/f'ptt_population_renewal.{ext}'
    fig.savefig(p,dpi=280,facecolor='white')
    print(p)
