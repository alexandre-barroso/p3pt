"""Recompute terminal controls and the complete joint census from public PSL.

Only the separately supplied public CSV is read. No source lexicon, manuscript,
model, vocabulary or training-record export is needed or produced.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np
import scipy
import model as sp
import terminal as tc
import census as cs

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(actual,expected,path='science'):
    """Strict structural comparison; finite floats allow 1e-10 abs/rel error."""
    if isinstance(expected,bool) or expected is None or isinstance(expected,(str,int)):
        if type(actual) is not type(expected) or actual!=expected:raise ValueError('Mismatch: '+path)
        return 0.0
    if isinstance(expected,float):
        if type(actual) not in (float,int) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Numeric mismatch: '+path)
        return abs(actual-expected)
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Keys differ: '+path)
        return max((compare(actual[k],v,path+'.'+k) for k,v in expected.items()),default=0.0)
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Lengths differ: '+path)
        return max((compare(a,b,path+f'[{i}]') for i,(a,b) in enumerate(zip(actual,expected))),default=0.0)
    raise TypeError(path)

def run(psl,output):
    if not __debug__:raise RuntimeError('Do not use Python -O.')
    psl=psl.resolve(strict=True);output=output.resolve()
    if output.exists() or output==psl or any(x in {'source_material','master_references'} for x in output.parts):raise ValueError('Output must be new and outside protected inputs.')
    if sha(psl)!=sp.PSL_SHA:raise ValueError('Unexpected PSL version.')
    versions={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'unicode':sp.ud.unidata_version}
    if sys.version_info[:3]!=(3,12,3) or np.__version__!='2.4.4' or scipy.__version__!='1.17.1' or sp.ud.unidata_version!='15.0.0':raise RuntimeError('Use the pinned numerical environment documented in README.md.')
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'IN_PROGRESS',
      'input_sha256':sp.PSL_SHA,'versions':versions,'test_was_previously_examined':True,
      'confirmatory':False,'source_hashes':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
      'checks':cs.checks(),'fit_events':[]}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        rows,population=sp.read_types(psl);words=[w for w,_ in rows];labels=np.array([y for _,y in rows]);split=np.array([sp.assignment(w) for w in words])
        sets=[[words[i] for i in np.flatnonzero(split==s)] for s in range(3)];ys=[labels[split==s] for s in range(3)]
        groups=[{sp.stripped(w) for w in ws} for ws in sets]
        assert all(not groups[a]&groups[b] for a in range(3) for b in range(a))
        science={'population':population,'splits':{n:{'types':len(sets[s]),'groups':len(groups[s]),'class_counts':np.bincount(ys[s],minlength=3).tolist()} for s,n in enumerate(['train','validation','test'])},'models':{},'contrasts':{}}
        trained={};full_fit=None
        # The context fit reproduces the earlier three-C whole-spelling model.
        # The six current procedures use the amended four-C grid.
        configurations=[(rep,family,transform,tc.GRID) for rep,transform in tc.REPRESENTATIONS.items() for family in tc.FAMILIES]
        configurations.append(('full_context','all_position',sp.canonical,[.1,1.,10.]))
        for rep,family,transform,grid in configurations:
            name=rep+'__'+family;tw,vw=[[transform(w) for w in ws] for ws in sets[:2]]
            vocab=tc.vocabulary(tw,family);x=tc.matrix(tw,vocab,family);vx=tc.matrix(vw,vocab,family)
            fits=[];candidates=[]
            metric=sp.metrics if rep=='full_context' else tc.metrics
            for C in grid:
                start=time.perf_counter();theta,optimizer=sp.fit(x,ys[0],C)
                candidates.append({'C':C,'validation':metric(ys[1],sp.probabilities(theta,vx))});fits.append(theta)
                event={'model':name,'C':C,'seconds':time.perf_counter()-start,'optimizer':optimizer}
                record['fit_events'].append(event);save();print(json.dumps({'fit':len(record['fit_events']),'of':27,'model':name,'C':C}),flush=True)
            best=min(range(len(grid)),key=lambda i:candidates[i]['validation']['log_loss'])
            info={'feature_dimension':len(vocab),'selected_C':grid[best],'candidate_fits':candidates}
            trained[name]=(vocab,fits[best],transform,family,info)
        assert len(record['fit_events'])==27
        record['all_selection_complete_before_reproduction_test_scoring']=True;save()
        probs={}
        for name,(vocab,theta,transform,family,info) in trained.items():
            tx=tc.matrix([transform(w) for w in sets[2]],vocab,family);prob=sp.probabilities(theta,tx)
            info['test']=(sp.metrics if name.startswith('full_context') else tc.metrics)(ys[2],prob)
            if name.startswith('full_context'):full_fit=info
            else:science['models'][name]=info;probs[name]=prob
        science['context']={'full':full_fit,'all_marks_test_oracle':sp.oracle([sp.stripped(w) for w in sets[2]],ys[2])}
        y,ws=ys[2],sets[2];syllables=tc.syllable_strata(psl,ws)
        transformed={rep:[f(w) for w in ws] for rep,f in tc.REPRESENTATIONS.items()}
        for rep in tc.REPRESENTATIONS:
            w,t5,t4=[rep+'__'+f for f in tc.FAMILIES]
            for label,an,bn,tn in [('all_minus_terminal1to5',w,t5,t5),('all_minus_subset1to4',w,t4,t4),('terminal1to5_minus_subset1to4',t5,t4,t5)]:
                a,b=probs[an],probs[bn];vocab,_,_,family,_=trained[tn]
                science['contrasts'][rep+'__'+label]={'a':an,'b':bn,**tc.contrast(y,a,b),
                  'strata':tc.stratified(ws,y,a,b,transformed[rep],vocab,family,syllables)}
        for family in tc.FAMILIES:
            an='acute_circumflex_removed__'+family;bn='all_marks_removed__'+family
            science['contrasts']['selective_minus_all_marks__'+family]={'a':an,'b':bn,**tc.contrast(y,probs[an],probs[bn])}
        primary='all_marks_removed__all_minus_terminal1to5';science['primary_contrast']=primary;p=science['contrasts'][primary];ap=y==2
        p['antepenult_conditional_bootstrap']=sp.group_interval([w for w,inc in zip(ws,ap) if inc],y[ap],probs[p['a']][ap],probs[p['b']][ap],seed=20260927)
        p['bootstrap_scope']=('Observed AP-bearing fully folded groups, 2000 percentile draws; conditional on fixed fit, selection and reused test; excludes family, training, tuning, question-selection and participant uncertainty')
        a,b=probs[p['a']].argmax(axis=1),probs[p['b']].argmax(axis=1);vocab=trained[p['b']][0];folded=transformed['all_marks_removed']
        ap_rows=[]
        for i in np.flatnonzero(ap):
            i=int(i);ap_rows.append({'test_index':i,'ending_group':tc.ending_stratum(ws[i]),'recorded_syllable_count':syllables[ws[i]],
              'longest_training_terminal_feature_letters':tc.longest_terminal(folded[i],vocab,'terminal_tagged_1to5'),
              'character_length':len(folded[i]),'all_position_class':int(a[i]),'terminal5_class':int(b[i]),'paired_correctness':cs.four_way(int(a[i]),int(b[i]))})
        joint=cs.joint_table(ap_rows);universe={'ending_group':cs.ENDINGS,'recorded_syllable_count':sorted({r['recorded_syllable_count'] for r in ap_rows}),
          'longest_training_terminal_feature_letters':list(range(6)),'character_length':sorted({r['character_length'] for r in ap_rows})}
        one,two,binary=cs.projected_tables(joint,universe)
        science['census']={'overall_ap_counts':cs.counts(r['paired_correctness'] for r in ap_rows),'joint_cells':joint,'one_dimensional_margins':one,'two_dimensional_margins':two,
          'binary_five_letter_coverage_crossings':binary,'canonical_vs_folded_character_length_disagreements':sum(len(ws[i])!=len(folded[i]) for i in np.flatnonzero(ap)),
          'public_examples_selection':{'namespace':cs.EXAMPLE_NAMESPACE,'rule':'Minimum SHA256(namespace + canonical spelling) in each paired-outcome × coverage5/<5 cell; no replacement',
            'cells':cs.selected_examples(ap_rows,ws),'not_representative_sample':True}}
        reference=json.loads((HERE/'reference_results.json').read_text())
        record['maximum_absolute_difference']=compare(science,reference)
        record['science']=science;record['input_unchanged']=sha(psl)==sp.PSL_SHA
        assert record['input_unchanged'] and record['source_hashes']=={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
        record['status']='PASS_FRESH_RECOMPUTATION';record['completed_utc']=datetime.now(timezone.utc).isoformat();save()
    except Exception as error:
        record.update(status='FAILED_RETAINED_CHECKPOINT',error=type(error).__name__+': '+str(error));save();raise
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--psl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.psl,a.output);print(json.dumps({'status':r['status'],'maximum_absolute_difference':r['maximum_absolute_difference']}))
