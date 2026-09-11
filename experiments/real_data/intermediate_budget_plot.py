"""Plot the prechosen intermediate-LLM-budget pilot, without selecting settings."""
from pathlib import Path
import argparse,json
from .small_panel_appendix import plt,np,pd,RCPARAMS,DATASET_LABEL,GRID,PANELS,METHODS,ScalarFormatter,NullFormatter


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/tmp/dial-intermediate-budget'));a=p.parse_args();root=a.root
    x=pd.read_csv(root/'rows.csv');seeds=json.loads((root/'design.json').read_text())['seeds']
    keys=['dataset','panel','n_H_level','method'];assert not x.duplicated(keys+['seed']).any()
    assert x.groupby(keys).seed.apply(lambda z:set(z)==set(range(seeds))).all()
    assert len(x)==57*3*seeds
    base=x[x.method=='consensus_cal'].set_index(['dataset','panel','n_H_level','seed'])
    z=x.join(base.excess.rename('cons_excess'),on=['dataset','panel','n_H_level','seed'],validate='many_to_one');z['delta_vs_cons']=z.excess-z.cons_excess
    g=z.reset_index().groupby(keys).agg(n_H=('n_H','mean'),excess=('excess','mean'),se=('excess','sem'),delta_vs_cons=('delta_vs_cons','mean'),paired_mcse=('delta_vs_cons','sem')).reset_index()
    g.to_csv(root/'plot_data.csv',index=False)
    with plt.rc_context(RCPARAMS):
        fig,axes=plt.subplots(3,3,figsize=(7.4,6.6),sharey='row')
        for i,(ds,nl) in enumerate([('arena_33k',2000),('mt_bench',160),('pandalm',100)]):
            row=g[g.dataset==ds];upper=max(.01,float((row.excess+1.96*row.se).max())*1.08)
            for j,panel in enumerate(PANELS):
                ax=axes[i,j]
                for method,label,color,ls,marker in METHODS[1:]:
                    q=row[(row.panel==panel)&(row.method==method)].sort_values('n_H')
                    ax.plot(q.n_H,q.excess,color=color,ls=ls,marker=marker,ms=3,label=label)
                    ax.fill_between(q.n_H,q.excess-1.96*q.se,q.excess+1.96*q.se,color=color,alpha=.085,lw=0)
                ax.set_xscale('log');ax.set_xticks([50,200,800,3200] if i==0 else [20,80,320]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter());ax.set_ylim(-.002,upper);ax.grid(color=GRID,lw=.5)
                if i==0:ax.set_title(panel)
                if j==0:ax.set_ylabel(f'{DATASET_LABEL[ds]}, $n_L={nl}$\nexcess held-out log loss')
                if i==2:ax.set_xlabel(r'human calibration labels $n_H$')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'Intermediate LLM budget: {seeds} matched seeds',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ['pdf','png']:fig.savefig(root/f'fig_intermediate_budget.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    print(g[g.method=='dial_margin_1'][['dataset','panel','n_H','excess','delta_vs_cons','paired_mcse']].round(5).to_string(index=False))

if __name__=='__main__':main()
