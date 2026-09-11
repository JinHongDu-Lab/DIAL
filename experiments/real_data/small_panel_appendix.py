"""Authoritative G.4.2 plots: raw GACV and c=1 adaptive DIAL-mu, 50 matched seeds.

python -m experiments.real_data.small_panel_appendix --margin-root results/endpoint_margin_appendix --out figures
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dial-small-panel-matplotlib')
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter,ScalarFormatter
import numpy as np
import pandas as pd
from .robustness_plot import load
from .robustness import ROOT
from experiments.style import RCPARAMS,DATASET_LABEL,GRID,INK_SOFT

PANELS=['small6','large6','all']
TITLES=['small6 (36.5B)','large6 (203B)','all, 21 judges (367.5B+, 3 paid APIs)']
NH={'arena_33k':300,'mt_bench':80,'pandalm':60}
METHODS=[('human_only','Human','#52514e','-',None),('consensus_cal','Cons-Cal','#1baf7a','--','v'),('dial_mu',r'DIAL-$\mu$ (raw)','#4279ad',':','s'),('dial_margin_1',r'DIAL-$\mu$','#c0392b','-','o')]


def figure_data(margin_root):
    adaptive=pd.read_csv(Path(margin_root)/'rows.csv')
    adaptive=adaptive[adaptive.method=='dial_margin_1'].copy()
    frames=[]
    for ds in NH:
        raw=load(ds,ROOT)
        raw=raw[raw.sweep.isin(['budget','llm_budget']) & raw.panel.isin(PANELS) & raw.method.isin(['human_only','consensus_cal','dial_mu']) & (raw.seed<50)]
        raw=raw[(raw.sweep=='budget')|((raw.sweep=='llm_budget')&(raw.n_H_level==NH[ds]))].copy()
        # Human-only predictions use the same calibration draw for every judge panel.
        human=raw[(raw.panel=='all')&(raw.method=='human_only')]
        raw=raw[raw.method!='human_only']
        raw=pd.concat([raw]+[human.assign(panel=p) for p in PANELS],ignore_index=True)
        raw['dataset']=ds
        ad=adaptive[(adaptive.dataset==ds)&adaptive.panel.isin(PANELS)&(adaptive.seed<50)]
        ad=ad[(ad.sweep=='budget')|((ad.sweep=='llm_budget')&(ad.n_H_level==NH[ds]))]
        keys=['sweep','panel','level','n_H_level','seed']
        raw_keys=set(map(tuple,raw[raw.method=='dial_mu'][keys].to_numpy()))
        ad_keys=set(map(tuple,ad[keys].to_numpy()))
        assert raw_keys==ad_keys,(ds,'raw/adaptive cell mismatch',len(raw_keys),len(ad_keys))
        frames.extend([raw,ad])
    data=pd.concat(frames,ignore_index=True)
    keys=['dataset','sweep','panel','level','n_H_level','method']
    assert not data.duplicated(keys+['seed']).any()
    assert data.groupby(keys).seed.apply(lambda x:set(x)==set(range(50))).all()
    summary=data.groupby(keys).agg(excess=('excess','mean'),se=('excess','sem'),n_H=('n_H','mean'),n_L=('n_L','mean'),n=('seed','size')).reset_index()
    return summary


def make_figures(margin_root=ROOT/'results'/'endpoint_margin_appendix',out=ROOT/'figures'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=figure_data(margin_root)
    summary.to_csv(out/'small_panel_appendix_plot_data.csv',index=False)
    with plt.rc_context(RCPARAMS):
        for sweep,filename in [('budget','fig_small_panel'),('llm_budget','fig_small_panel_llmbudget')]:
            fig,axes=plt.subplots(3,3,figsize=(7.4,6.6),sharey='row')
            for i,ds in enumerate(NH):
                row=summary[(summary.dataset==ds)&(summary.sweep==sweep)]
                nonhuman=row[row.method!='human_only']
                upper=max(.01,float((nonhuman.excess+1.96*nonhuman.se).max())*1.08)
                for j,panel in enumerate(PANELS):
                    ax=axes[i,j]
                    for method,label,color,ls,marker in METHODS:
                        g=row[(row.panel==panel)&(row.method==method)].sort_values('n_H' if sweep=='budget' else 'level')
                        x=g.n_H if sweep=='budget' else g.level
                        if method=='human_only' and sweep=='llm_budget':
                            ax.axhline(g.excess.mean(),color=color,lw=1,label=label)
                        else:
                            ax.plot(x,g.excess,color=color,ls=ls,marker=marker,ms=2.5,lw=1.45 if method=='dial_margin_1' else 1.05,label=label)
                            if method!='human_only':ax.fill_between(x,g.excess-1.96*g.se,g.excess+1.96*g.se,color=color,alpha=.085,lw=0)
                    ax.set_xscale('log');ax.xaxis.set_minor_formatter(NullFormatter())
                    if sweep=='budget':
                        ax.set_xticks([50,200,800,3200] if ds=='arena_33k' else [20,80,320]);ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.set_ylim(-.002,upper);ax.grid(color=GRID,lw=.5);ax.tick_params(length=2.5,labelsize=6.5)
                    if i==0:ax.set_title(TITLES[j],fontsize=7.7)
                    if j==0:ax.set_ylabel(f'{DATASET_LABEL[ds]}\nexcess held-out log loss')
                    if i==2:ax.set_xlabel(r'human calibration labels $n_{\mathrm{H}}$' if sweep=='budget' else r'LLM comparisons $n_{\mathrm{L}}$')
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.005))
            if sweep=='llm_budget':fig.suptitle(r'Fixed human budget $n_{\mathrm{H}}=300\,/\,80\,/\,60$',fontsize=8.5,y=.995)
            fig.tight_layout(rect=(0,.03,1,.975 if sweep=='llm_budget' else 1))
            fig.savefig(out/f'{filename}.pdf',bbox_inches='tight')
            fig.savefig(out/f'{filename}.png',dpi=160,bbox_inches='tight')
            plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--margin-root',type=Path,default=ROOT/'results'/'endpoint_margin_appendix');p.add_argument('--out',type=Path,default=ROOT/'figures');a=p.parse_args()
    summary=make_figures(a.margin_root,a.out)
    print(f'Exported both G.4.2 figures: {len(summary)} plotted method/setting rows; all 50 seeds verified.')

if __name__=='__main__':main()
