"""Schematic of the PTT snapshot criterion using translated unit Gaussians.
For these illustrative distributions, mean swap acceptance = erfc(delta/2).
The shaded marginal overlap is illustrative, not the acceptance probability.
"""
from pathlib import Path
import numpy as np
from scipy.special import erfcinv
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent
x=np.linspace(-3.5,6.,1600)
acceptances=[.8,.5,.25]
shifts=2*erfcinv(acceptances)
reference=norm.pdf(x)
blue,orange,green,purple='#285b8c','#df8439','#37856b','#9b80b3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#21354c'})
fig,axes=plt.subplots(1,3,figsize=(13.5,4.6),sharex=True,sharey=True)
for i,(ax,acc,shift) in enumerate(zip(axes,acceptances,shifts)):
    current=norm.pdf(x-shift)
    ax.fill_between(x,reference,color=blue,alpha=.08)
    ax.fill_between(x,current,color=orange,alpha=.13)
    ax.fill_between(x,np.minimum(reference,current),color=purple,alpha=.38)
    ax.plot(x,reference,color=blue,lw=2)
    ax.plot(x,current,color=orange,lw=2.3)
    if i==2:
        snapshot=norm.pdf(x-shifts[1])
        ax.plot(x,snapshot,color=green,lw=2,ls=(0,(4,2)),zorder=5)
        ax.annotate('Insert stored snapshot',xy=(shifts[1],norm.pdf(0)),xytext=(shift+1.45,.49),
                    ha='center',fontsize=11,color=green,
                    arrowprops=dict(arrowstyle='->',color=green,lw=1.2,connectionstyle='arc3,rad=.2'))
    else:
        ax.annotate('',xy=(shift+.85,.46),xytext=(shift-.1,.46),
                    arrowprops=dict(arrowstyle='->',color=orange,lw=1.6))
    ax.set_title(rf'$\langle A_{{\mathrm{{swap}}}}\rangle\ '+('\\approx' if i else '=')+rf'\ {acc:.2f}$',pad=16,fontsize=16)
    if i==0:
        caption='Continue training'
    elif i==1:
        caption='Store snapshot'
    else:
        caption='Add model to ladder'
    ax.text(.5,-.08,caption,ha='center',transform=ax.transAxes,fontsize=12,
            color=green if i==2 else '#21354c',weight='bold' if i==2 else 'normal')
    ax.set_xlim(-3.4,5.8);ax.set_ylim(0,.57)
    ax.axhline(0,color='#bac5d1',lw=.8)
    ax.set_axis_off()
fig.legend(handles=[Line2D([0],[0],color=blue,lw=2,label='Last ladder model'),
                    Line2D([0],[0],color=orange,lw=2,label='Training model'),
                    Patch(facecolor=purple,alpha=.38,label='Distribution overlap'),
                    Line2D([0],[0],color=green,lw=2,ls='--',label='Stored snapshot')],
           loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False,fontsize=10)
axes[0].annotate('',xy=(.94,.105),xytext=(.06,.105),xycoords=fig.transFigure,
                 arrowprops=dict(arrowstyle='->',color='#8292a3',lw=1.25))
fig.text(.5,.045,'Training time',ha='center',fontsize=11,color='#65788e')
fig.subplots_adjust(left=.035,right=.98,top=.79,bottom=.25,wspace=.16)
for ext in ['png','pdf','svg']:
    p=OUT/f'ptt_ladder_insertion_schematic.{ext}'
    fig.savefig(p,dpi=300,facecolor='white')
    print(p)
