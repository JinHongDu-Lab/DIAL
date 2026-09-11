"""Budget curves for the direct GACV endpoint-margin experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dial-endpoint-matplotlib')
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/endpoint_margin'));a=p.parse_args()
    data=pd.read_csv(a.root/'summary.csv')
    methods=[('dial_mu','GACV','#467bb4','-'),('consensus_cal','Cons-Cal','#333333','--'),('dial_margin_0.5','c=0.5','#aa75b5',':'),('dial_margin_1','c=1','#d25925','-'),('dial_margin_2','c=2','#54a26c','-.')]
    for sweep,xlabel in [('budget','Human calibration comparisons'),('llm_budget','LLM training comparisons')]:
        fig,axes=plt.subplots(3,2,figsize=(10,10),sharey='row')
        for i,dataset in enumerate(['arena_33k','mt_bench','pandalm']):
            for j,panel in enumerate(['all','small6']):
                ax=axes[i,j]
                for key,label,color,ls in methods:
                    g=data[(data.dataset==dataset)&(data.panel==panel)&(data.sweep==sweep)&(data.method==key)].sort_values('n_H' if sweep=='budget' else 'n_L')
                    x=g.n_H if sweep=='budget' else g.n_L
                    ax.plot(x,g.excess,label=label,color=color,ls=ls,marker='o',ms=3,lw=1.5)
                    if key=='dial_margin_1':ax.fill_between(x,g.excess-g.excess_se,g.excess+g.excess_se,color=color,alpha=.12)
                ax.set_xscale('log')
                ax.xaxis.set_minor_formatter(NullFormatter())
                if sweep=='budget':
                    ax.set_xticks([50,200,800,3200] if dataset=='arena_33k' else [20,80,320])
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.grid(alpha=.15);ax.set_title(f'{dataset} / {panel}')
                if j==0:ax.set_ylabel('Excess human test log loss')
                if i==2:ax.set_xlabel(xlabel)
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower center',ncol=5,frameon=False)
        fig.suptitle('Direct GACV endpoint margin: 20 paired seeds (c=1 band: ±1 MCSE)')
        fig.tight_layout(rect=(0,.04,1,.97))
        fig.savefig(a.root/f'endpoint_margin_{sweep}.pdf')
        fig.savefig(a.root/f'endpoint_margin_{sweep}.png',dpi=130)
        plt.close(fig)

if __name__=='__main__':main()
