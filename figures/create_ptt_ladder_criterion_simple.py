"""Conceptual ladder growth, omitting internal snapshot bookkeeping."""
from pathlib import Path
import numpy as np
from scipy.special import erfcinv
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
blue,orange='#285b8c','#df8439'
delta=float(2*erfcinv(.24))
x=np.linspace(-3.5,2*delta+3.5,2000)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#21354c'})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharex=True,sharey=True)
for i,ax in enumerate(axes):
    fixed=[j*delta for j in range(i+1)]
    moving=(i+1)*delta
    for mu in fixed:
        d=norm.pdf(x-mu)
        ax.fill_between(x,d,color=blue,alpha=.07)
        ax.plot(x,d,color=blue,lw=2.2)
    evolving=norm.pdf(x-moving)
    ax.fill_between(x,evolving,color=orange,alpha=.13)
    ax.plot(x,evolving,color=orange,lw=2.4)
    overlap=np.minimum(norm.pdf(x-fixed[-1]),evolving)
    ax.fill_between(x,overlap,color=orange,alpha=.26)
    ax.annotate('',xy=(moving+.9,.46),xytext=(moving-.2,.46),
                arrowprops=dict(arrowstyle='->',color=orange,lw=1.7))
    ax.set_title(r'$\langle A_{\mathrm{swap}}\rangle < 0.25$',fontsize=17,pad=15)
    ax.text(.5,-.09,'Add model to ladder',transform=ax.transAxes,ha='center',fontsize=12,weight='bold')
    ax.set_xlim(x[0],x[-1]);ax.set_ylim(0,.55)
    ax.axhline(0,color='#bac5d1',lw=.8)
    ax.set_axis_off()
fig.legend(handles=[Line2D([0],[0],color=blue,lw=2.2,label='Fixed ladder models'),
                    Line2D([0],[0],color=orange,lw=2.4,label='Evolving model')],
           loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False,fontsize=12)
axes[0].annotate('',xy=(.95,.10),xytext=(.05,.10),xycoords=fig.transFigure,
                 arrowprops=dict(arrowstyle='->',color='#8292a3',lw=1.3))
fig.text(.5,.035,'Training time',ha='center',color='#65788e')
fig.subplots_adjust(left=.035,right=.975,top=.77,bottom=.25,wspace=.15)
for ext in ['png','pdf','svg']:
    path=OUT/f'ptt_ladder_insertion_simple.{ext}'
    fig.savefig(path,dpi=300,facecolor='white')
    print(path)
