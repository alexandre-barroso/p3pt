"""Rebuild seven editable tables and the manuscript's coverage figure."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from reproduce import compare
from tables import assets,integer

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--anonymous',action='store_true');a=p.parse_args()
    r=json.loads(a.results.read_text());compare(r['science'],json.loads((HERE/'reference_results.json').read_text()))
    out=a.output.resolve()
    if out.exists() or any(s in {'source_material','master_references'} for s in out.parts):raise ValueError('Use a new output directory outside protected inputs.')
    out.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    # Keep font discovery/cache writes within this output, too.
    from xml.sax.saxutils import escape
    config=out/'fonts.conf';fontcache=out/'font-cache';fontcache.mkdir()
    config.write_text('<fontconfig><dir>'+escape(str(Path(matplotlib.get_data_path())/'fonts/ttf'))+'</dir><cachedir>'+escape(str(fontcache))+'</cachedir></fontconfig>\n')
    os.environ['FONTCONFIG_FILE']=str(config)
    import matplotlib.pyplot as plt
    tables,figures,cov=assets(r['science']);compare(tables,json.loads((HERE/'expected_tables.json').read_text()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.6,3.3),layout='constrained')
    for name,only,marker,col,off in [('Palavra','only_all_position_correct','o','0.15',.09),('Terminal5','only_terminal5_correct','s','0.52',-.09)]:
        for y,label in enumerate(['<5','5']):
            x=cov[label];num=x['both_correct']+x[only];v=100*num/x['observations']
            ax.scatter(v,y+off,label=name if y==0 else None,marker=marker,color=col,s=45,zorder=3)
            ax.annotate(f'{integer(num)}/{integer(x["observations"])}',(v,y+off),xytext=(-6,0),textcoords='offset points',ha='right',va='center',fontsize=9)
    ax.set_yticks([0,1],['Menos de cinco letras\n(n = 241)','Cinco letras\n(n = 2.250)']);ax.set_ylim(-.45,1.65);ax.set_xlim(0,100)
    ax.set_xlabel('Recuperação antepenúltima (%)');ax.set_ylabel('Cobertura terminal');ax.grid(axis='x',color='0.9',linewidth=.7);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',frameon=False,ncol=2)
    author='' if a.anonymous else 'Alexandre Menezes Barroso'
    for ext in ['pdf','png','jpg','svg']:
        md={'Creator':author,'Title':'Recuperação por cobertura terminal'} if ext in ['pdf','svg'] else {'Author':author,'Title':'Recuperação por cobertura terminal'} if ext=='png' else None
        kwargs={'dpi':400}
        if md:kwargs['metadata']=md
        fig.savefig(out/('terminal_coverage.'+ext),**kwargs)
    plt.close(fig)
    for name,obj in [('tables',tables),('figures',figures)]:
        (out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    files=sorted(p for p in out.iterdir() if p.is_file())
    manifest={'status':'TABLE_VALUES_MATCH_REFERENCE_FIGURE_RENDERED','tables':len(tables),'figures':1,'anonymous':a.anonymous,
      'input_sha256':hashlib.sha256(a.results.read_bytes()).hexdigest(),
      'files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'tables':7,'figures':1,'formats':['PDF','PNG','JPEG','SVG']}))
if __name__=='__main__':main()
