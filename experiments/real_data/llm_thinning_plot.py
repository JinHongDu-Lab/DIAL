"""Plot low-human-budget thinning results; no manuscript files are changed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .llm_thinning import OUT, BUDGETS
from .robustness import ROOT


def main():
    df = pd.read_json(OUT/'rows.jsonl', lines=True)
    comp = pd.read_csv(OUT/'comparisons.csv')
    for metric, ylabel, suffix in [('excess','Excess human test log loss',''),('ref_kendall',"Kendall's tau",'_tau')]:
        fig, axes = plt.subplots(3,3,figsize=(12,9),sharex=True)
        for row,dataset in enumerate(BUDGETS):
            for col,panel in enumerate(['all','small6','large6']):
                ax=axes[row,col]
                for nh,color in zip(BUDGETS[dataset],['#1764ab','#dd7024']):
                    g=df[(df.dataset==dataset)&(df.panel==panel)&(df.n_H==nh)&(df.method=='dial_mu')]
                    agg=g.groupby('fraction')[metric].agg(['mean','std','count'])
                    ax.errorbar(agg.index,agg['mean'],yerr=agg['std']/np.sqrt(agg['count']),color=color,marker='o',ms=3,label=f'DIAL, nH={nh}')
                    cons=df[(df.dataset==dataset)&(df.panel==panel)&(df.n_H==nh)&(df.method=='consensus_cal')&(df.fraction==1)][metric].mean()
                    ax.axhline(cons,color=color,ls='--',alpha=.8,label=f'Full Cons-Cal, nH={nh}')
                    sel=comp[(comp.dataset==dataset)&(comp.panel==panel)&(comp.n_H==nh)&(comp.method=='dial_select_nL_gacv')].iloc[0]
                    ax.axhline(sel['excess' if metric=='excess' else 'tau'],color=color,ls=':',alpha=.8,label=f'GACV-selected nL, nH={nh}')
                ax.set_xscale('log')
                ax.set_title(f'{dataset} / {panel}')
                ax.grid(alpha=.15)
                if col==0: ax.set_ylabel(ylabel)
                if row==2: ax.set_xlabel('Fraction of available LLM rows retained')
        handles,labels=axes[0,0].get_legend_handles_labels()
        # Budgets differ by dataset: each row's first panel carries its own legend.
        for row in range(3): axes[row,0].legend(fontsize=6,loc='best')
        fig.suptitle('LLM thinning with scarce human labels (50 paired seeds; bars = Monte Carlo SE)',fontsize=12)
        fig.tight_layout()
        fig.savefig(ROOT/'figures'/f'fig_llm_thinning{suffix}.pdf')
        fig.savefig(OUT/f'fig_llm_thinning{suffix}.png',dpi=140)
        plt.close(fig)

if __name__=='__main__': main()
