"""Independent PCD temperature sampling; contours, endpoints and chain traces."""
from pathlib import Path
import json,time
import h5py
import numpy as np
import torch
from adabmDCA.api import load_model
from adabmDCA.sampling import prepare_fixed_model_sampler
from adabmDCA.utils import init_chains
from adabmDCA.statmech import compute_energy
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/RF00379/sampling'
PREFIX=OUT/'pca_temperature_pcd_beta_2_3_to_1'
TEMPERATURES=[1.5,1.3,1.15,1.]
N=2000
SEED=147
with np.load(OUT/'pca_ladder_distributions_projections.npz') as f:
    mean=f['mean']; basis=f['components']
def project(x):
    return (x.detach().cpu().numpy().reshape(len(x),-1)-mean)@basis
cache=Path(str(PREFIX)+'_samples.h5')
if not cache.exists():
    model=load_model(ROOT/'models/RF00379/params.dat.gz',alphabet='rna',device='cpu')
    kernel,params=prepare_fixed_model_sampler('metropolized_gibbs',torch.device('cpu'),'float32',model.params)
    results=[]
    for i,T in enumerate(TEMPERATURES):
        previous = OUT/'pca_temperature_pcd_samples.h5'
        with h5py.File(previous) as old:
            matches = np.flatnonzero(np.isclose(old['temperatures'][()], T))
            if len(matches):
                g = old[f'temperature_{matches[0]}']
                results.append(tuple(g[key][()] for key in ['sequences', 'pca', 'chain_traces_pca', 'diagnostics']))
                print(f'T={T:g}: reused existing PCD samples', flush=True)
                continue
        torch.manual_seed(SEED+i)
        chains=init_chains(num_chains=N,L=model.metadata.length,q=model.metadata.num_states,device=torch.device('cpu'),dtype=torch.float32)
        history=[]
        t=time.time()
        for sweep in range(0,2000,100):
            chains=kernel(chains=chains,params=params,nsweeps=100,beta=1/T)
            energies=compute_energy(chains,model.params).detach().cpu().numpy()
            pc=project(chains)
            history.append([sweep+100,float(energies.mean()),float(energies.std()/np.sqrt(N)),*pc.mean(0).tolist(),*pc.std(0).tolist()])
            if (sweep+100)%500==0:
                print(f'T={T:g}: {sweep+100}/2000 sweeps, mean energy={energies.mean():.3f}, elapsed={time.time()-t:.1f}s',flush=True)
        traces=[project(chains[:3])]
        for _ in range(40):
            chains=kernel(chains=chains,params=params,nsweeps=5,beta=1/T)
            traces.append(project(chains[:3]))
        results.append((chains.argmax(-1).numpy(),project(chains),np.array(traces),np.array(history)))
    with h5py.File(cache,'x') as f:
        f.attrs.update(strategy='pcd',sampler='metropolized_gibbs',model=str(ROOT/'models/RF00379/params.dat.gz'),tokens=model.tokens,
                       device='cpu',seed=SEED,burn_in_sweeps=2000,trajectory_sweeps=200,trace_stride=5,
                       description='Independent temperatures, no replica swaps; finite-run MCMC samples')
        f.create_dataset('temperatures',data=TEMPERATURES)
        f.create_dataset('betas',data=1/np.array(TEMPERATURES))
        f.create_dataset('seeds',data=[148,148,149,147])
        for i,(seq,pc,traces,history) in enumerate(results):
            g=f.create_group(f'temperature_{i}')
            g.create_dataset('sequences',data=seq,compression='gzip')
            g.create_dataset('pca',data=pc)
            g.create_dataset('chain_traces_pca',data=traces)
            g.create_dataset('diagnostics',data=history)
            g.attrs['diagnostic_columns']='sweep,mean_energy,energy_sem,mean_PC1,mean_PC2,std_PC1,std_PC2'
with h5py.File(cache) as f:
    projected=[f[f'temperature_{i}/pca'][()] for i in range(4)]
    traces=[f[f'temperature_{i}/chain_traces_pca'][()] for i in range(4)]
    for i,T in enumerate(TEMPERATURES):
        print('Diagnostic',T,f[f'temperature_{i}/diagnostics'][-5:, :].tolist(),flush=True)
all_points=np.vstack(projected+ [t.reshape(-1,2) for t in traces])
lo,hi=all_points.min(0),all_points.max(0)
pad=(hi-lo)*.08+.35;lo-=pad;hi+=pad
edges=[np.linspace(lo[j],hi[j],241) for j in range(2)]
centres=[(e[:-1]+e[1:])/2 for e in edges]
sigma=[.22/(e[1]-e[0]) for e in edges]
fig,axes=plt.subplots(1,4,figsize=(16,4.7),sharex=True,sharey=True)
for ax,T,p,tr in zip(axes,TEMPERATURES,projected,traces):
    hist=np.histogram2d(*p.T,bins=edges)[0]
    density=gaussian_filter(hist,sigma=sigma,mode='constant').T
    density/=density.sum()
    ranked=np.sort(density.ravel())[::-1]; cumulative=ranked.cumsum()
    levels=[ranked[np.searchsorted(cumulative,m)] for m in [.95,.80,.50]]
    ax.contourf(*centres,density,levels=[*levels,density.max()*1.001],colors=['#fcead9','#f4c393','#e69a57'])
    ax.contour(*centres,density,levels=levels,colors='#c9742d',linewidths=.8)
    ax.scatter(*p.T,s=3,color='#285b8c',alpha=.30,linewidths=0,rasterized=True,zorder=3)
    tr = tr[-11:]  # Show the last 50 sweeps to keep the illustration readable.
    for j,c in enumerate(['#224d70','#31796b','#824b89']):
        ax.plot(*tr[:,j].T,color=c,lw=.85,alpha=.85,zorder=4)
        ax.scatter(*tr[0,j],s=18,facecolor='white',edgecolor=c,linewidth=.8,zorder=5)
        ax.scatter(*tr[-1,j],s=26,color=c,marker='>',zorder=5)
    ax.set_title(rf'$\beta = {1/T:.3f}$',fontsize=13,color='#172b45',pad=8)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_axis_off()
fig.subplots_adjust(left=.01,right=.99,bottom=.025,top=.91,wspace=.055)
for ext in ['png','pdf','svg']:
    fig.savefig(f'{PREFIX}.{ext}',dpi=300,facecolor='white')
print('Saved',PREFIX,flush=True)
