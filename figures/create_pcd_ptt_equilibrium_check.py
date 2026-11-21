"""Illustrative stationarity check, not an empirical result or convergence proof."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
ink,blue,orange,grey='#21354c','#285b8c','#df8439','#b0bcc7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink})
rng=np.random.default_rng(12)
n=650
component=rng.random(n)<.34
base=rng.normal(size=(n,2))@np.array([[.55,.08],[.12,.45]])
base+=np.where(component[:,None],np.array([1.15,.75]),np.array([-.75,-.35]))
ptt=rng.normal(size=(n,2))@np.array([[.55,.08],[.12,.45]])
ptt+=np.where((rng.random(n)<.34)[:,None],np.array([1.15,.75]),np.array([-.75,-.35]))
pcd=base@np.array([[.78,.12],[-.1,.85]])+np.array([2.9,-.45])

def contours(ax,shift=(0,0),color=blue,style='-',alpha=1):
    x,y=np.meshgrid(np.linspace(-3.3,5.8,240),np.linspace(-2.7,2.7,180))
    x=x-shift[0];y=y-shift[1]
    d=.66*np.exp(-.5*(((x+.75)/.60)**2+((y+.35)/.47)**2))+.34*np.exp(-.5*(((x-1.15)/.60)**2+((y-.75)/.47)**2))
    ax.contour(x+shift[0],y+shift[1],d,levels=[.05,.16,.32],colors=color,linewidths=1.15,linestyles=style,alpha=alpha)

fig=plt.figure(figsize=(13.2,7.0),facecolor='white')
gs=fig.add_gridspec(2,2,left=.17,right=.965,bottom=.15,top=.77,hspace=.12,wspace=.50)
for row in range(2):
    for col in range(2):
        ax=fig.add_subplot(gs[row,col])
        if col==0:
            ax.scatter(*base.T,s=6,c=grey,alpha=.50,linewidths=0,rasterized=True)
            contours(ax,color='#8295a5')
        elif row==0:
            ax.scatter(*ptt.T,s=6,c=blue,alpha=.45,linewidths=0,rasterized=True)
            contours(ax)
        else:
            contours(ax,color=grey,style='--',alpha=.8)
            ax.scatter(*pcd.T,s=6,c=orange,alpha=.5,linewidths=0,rasterized=True)
            contours(ax,shift=(2.9,-.45),color=orange)
            ax.annotate('',xy=(2.15,-1.45),xytext=(-.15,-1.15),
                        arrowprops=dict(arrowstyle='->',color=orange,lw=1.5,connectionstyle='arc3,rad=.12'))
        ax.set_xlim(-2.7,5.65);ax.set_ylim(-2.2,2.15);ax.set_aspect('equal');ax.set_axis_off()
        if row==0:
            ax.set_title(['Same initial population','After 50,000 MCMC steps'][col],loc='left',fontsize=13,weight='bold',pad=15)
# Annotations outside the plotting regions.
fig.text(.045,.60,'PTT-trained\nmodel',fontsize=15,weight='bold',color=blue,va='center')
fig.text(.045,.33,'PCD-trained\nmodel',fontsize=15,weight='bold',color=orange,va='center')
for yy in [.60,.33]:
    fig.add_artist(FancyArrowPatch((.495,yy),(.62,yy),transform=fig.transFigure,
                                  arrowstyle='->',mutation_scale=16,lw=1.5,color='#8295a5'))
fig.text(.557,.635,'MCMC',ha='center',fontsize=11,color=blue)
fig.text(.557,.365,'MCMC',ha='center',fontsize=11,color=orange)
fig.text(.80,.455,'No systematic distribution drift',ha='center',fontsize=11,color=blue)
fig.text(.80,.14,'Distribution shifts under the PCD-trained model',ha='center',fontsize=10.5,color=orange)
fig.text(.045,.935,'Does the distribution remain stationary?',fontsize=23,weight='bold')
fig.text(.045,.88,'Initialize both runs from equilibrium samples generated with PTT.',fontsize=12,color='#697d90')
fig.text(.5,.085,'Compare distributions, not individual configurations.',ha='center',fontsize=12,color=ink)
fig.text(.5,.035,'Schematic expected outcome · Stability alone does not establish equilibrium; a shifted population still requires convergence checks.',
         ha='center',fontsize=9,color='#697d90')
for ext in ['png','pdf','svg']:
    p=OUT/f'pcd_ptt_equilibrium_check.{ext}'
    fig.savefig(p,dpi=280,facecolor='white')
    print(p)
