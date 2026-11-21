"""Illustrative likelihood landscape and local KL-constrained parameter update."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
ink,orange,teal='#21354c','#df8439','#37856b'
# Concave quadratic: the same Fisher matrix defines likelihood curvature and local KL.
F=np.array([[1.6,.62],[.62,.75]])
optimum=np.array([.65,.85]); current=np.array([-1.65,-1.15])
def likelihood(p):
    d=np.asarray(p)-optimum
    return 3.-.5*np.einsum('...i,ij,...j->...',d,F,d)
grad=F@(optimum-current)
nominal=.95*grad
budget=.60
predicted=.5*nominal@F@nominal
adapted=nominal*np.sqrt(budget/predicted)
end=current+adapted
nom_end=current+nominal
v,e=np.linalg.eigh(F)
t=np.linspace(0,2*np.pi,300)
ellipse=current+(e@(np.sqrt(2*budget/v)[:,None]*np.array([np.cos(t),np.sin(t)]))).T
xs=np.linspace(-2.8,3.5,150);ys=np.linspace(-2.5,2.65,150)
X,Y=np.meshgrid(xs,ys);Z=likelihood(np.stack([X,Y],axis=-1))
floor=-15.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':ink})
fig=plt.figure(figsize=(12,9),facecolor='white')
ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
cmap=LinearSegmentedColormap.from_list('landscape',['#dce8f0','#8db4c8','#448b9c','#327d79','#a8c6a0','#efd6a0'])
ax.plot_surface(X,Y,Z,cmap=cmap,alpha=.91,linewidth=0,antialiased=True,rstride=2,cstride=2,zorder=2)
ax.contour(X,Y,Z,levels=np.linspace(-12,2.8,10),zdir='z',offset=floor,colors='#c3d0d9',linewidths=.65,zorder=0)
# KL trust region projected onto the parameter plane.
ax.plot(ellipse[:,0],ellipse[:,1],floor+.04,color=teal,lw=2.2,zorder=3)
ax.plot_trisurf(ellipse[:,0],ellipse[:,1],np.full(len(ellipse),floor+.02),color=teal,alpha=.12,zorder=1)
# Lift the same boundary onto the likelihood surface.
ax.plot(ellipse[:,0],ellipse[:,1],likelihood(ellipse)+.055,color=teal,lw=2,ls=(0,(4,2)),zorder=4)
for p,c in [(current,ink),(end,teal)]:
    z=likelihood(p)
    ax.plot([p[0],p[0]],[p[1],p[1]],[floor,z],color=c,lw=.85,ls=':',alpha=.5,zorder=3)
    ax.scatter(*p,z+.10,s=75,c=c,edgecolors='white',linewidths=1.2,depthshade=False,zorder=7)
# Draw update paths slightly above the surface, to preserve visibility.
for step,color,style in [(nominal,orange,'--'),(adapted,teal,'-')]:
    path=current+np.linspace(0,1,100)[:,None]*step
    ax.plot(path[:,0],path[:,1],likelihood(path)+.12,color=color,lw=3 if style=='-' else 2,ls=style,zorder=6)
    a=path[-5];b=path[-1];za=likelihood(a)+.12;zb=likelihood(b)+.12
    ax.quiver(*a,za,*(b-a),zb-za,color=color,arrow_length_ratio=.9,linewidth=2,zorder=7)
ax.scatter(*nom_end,likelihood(nom_end)+.12,s=55,marker='x',color=orange,linewidths=2,depthshade=False,zorder=7)
ax.text(current[0]-.15,current[1]-.1,likelihood(current)+1.35,r'Current model $\theta$',color=ink,fontsize=11,zorder=8)
ax.text(end[0]-.15,end[1]+.08,likelihood(end)+1.0,'Adaptive step',color=teal,fontsize=11,zorder=8)
ax.text(nom_end[0]+.12,nom_end[1],likelihood(nom_end)+.6,'Nominal step',color=orange,fontsize=11,zorder=8)
ax.text(ellipse[:,0].min()-.15,current[1]-.55,floor+.3,'KL trust region',color=teal,fontsize=11,zorder=8)
ax.set_xlabel(r'Field direction $\Delta h$',labelpad=13)
ax.set_ylabel(r'Coupling direction $\Delta J$',labelpad=13)
ax.set_zlabel('Log-likelihood',labelpad=10)
ax.set_xlim(xs[0],xs[-1]);ax.set_ylim(ys[0],ys[-1]);ax.set_zlim(floor,5.7)
ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
ax.view_init(elev=29,azim=-62)
ax.set_box_aspect((1.2,1,.85))
for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
    axis.pane.fill=False
    axis.pane.set_edgecolor('#edf1f4')
    axis.line.set_color('#a8b6c3')
ax.grid(False)
fig.text(.075,.94,'Adaptive learning rate',fontsize=23,weight='bold',color=ink)
fig.text(.075,.895,'Improve likelihood while limiting the change in the model distribution.',fontsize=12,color='#697d90')
fig.text(.5,.065,r'$D_{\mathrm{KL}}(p_\theta\,\|\,p_{\theta+\Delta\theta})\;\approx\;\frac{1}{2}\Delta\theta^\top F\,\Delta\theta\;\leq\;\varepsilon$',ha='center',fontsize=17,color=ink)
fig.subplots_adjust(left=.02,right=.96,bottom=.12,top=.86)
for ext in ['png','pdf','svg']:
    p=OUT/f'adaptive_learning_rate_landscape.{ext}'
    fig.savefig(p,dpi=260,facecolor='white')
    print(p)
