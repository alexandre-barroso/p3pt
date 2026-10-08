"""Single entry point for synthetic checks, formal replay and numerical results."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import census
import model
import reproduce

HERE=Path(__file__).resolve().parent
REQUIRED_FILES={'model.py','marks.py','terminal.py','census.py','reproduce.py','verify.py','draw_figures.py','tables.py',
 'reference_results.json','expected_tables.json','requirements.txt','lean/CollisionBound.lean','lean/lean-toolchain',
 'README.md','PROTOCOL.md','claim_map.json','CITATION.cff','RUN_RECORD.json','.github/workflows/checks.yml'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def synthetic():
    result=census.checks();cases=0
    # Exhaust all labels for two alternating synthetic input groups, including
    # empty groups and a candidate prediction label absent from the data.
    for n in range(6):
        for ys in itertools.product(range(3),repeat=n):
            groups=[i%2 for i in range(n)]
            measured=model.oracle([str(g) for g in groups],ys)['minimum_errors'] if n else 0
            brute=min(sum(pred[g]!=y for g,y in zip(groups,ys)) for pred in itertools.product(range(4),repeat=2))
            if measured!=brute:raise ValueError('Finite group-count mismatch.')
            cases+=1
    x={'matrix':[[1,2],[3,4]],'ratio':.25,'passed':True,'empty':None}
    reproduce.compare(x,deepcopy(x));bad=[dict(x,ratio=float('nan')),dict(x,ratio=.5),dict(x,passed=1),dict(x,extra=0),dict(x,matrix=[[1,2]])]
    for altered in bad:
        try:reproduce.compare(altered,x)
        except ValueError:pass
        else:raise ValueError('Corruption accepted.')
    return {'terminal_and_census':result,'exhaustive_finite_count_cases':cases,'corruptions_rejected':len(bad),'scope':'Synthetic checks do not establish the empirical findings.'}

def formal(fresh,run):
    elan=shutil.which('elan')
    if not elan:raise RuntimeError('An existing Elan/Lean 4.34.0 installation is required; no tools are installed automatically.')
    pin='leanprover/lean4:v4.34.0';installed=run([elan,'toolchain','list'],'installed_toolchains')
    if pin not in {line.strip().split(' ',1)[0] for line in installed.splitlines()}:raise RuntimeError('Pinned Lean is not locally installed.')
    lean=[elan,'run',pin,'lean'];version=run(lean+['--version'],'lean_version')
    if 'version 4.34.0,' not in version:raise RuntimeError('Unexpected Lean version.')
    prefix=Path(run(lean+['--print-prefix'],'lean_prefix').strip())
    output=run(lean+['-o',str(fresh/'CollisionBound.olean'),str(HERE/'lean/CollisionBound.lean')],'lean_build')
    required={'class_error_lower_bound','class_error_bound_attainable','indistinguishable_labels_force_error','aggregate_error_lower_bound','aggregate_error_bound_attainable'};audits={}
    for line in output.splitlines():
        m=re.match(r"'NormalizationAudit\.(\w+)' (.*)",line)
        if not m:continue
        name,rest=m.groups()
        if 'does not depend on any axioms' in rest:axioms=[]
        else:
            match=re.search(r'axioms: \[(.*)\]',rest)
            if not match:raise ValueError('Unrecognized axiom audit.')
            axioms=match[1].split(', ') if match[1] else []
        if set(axioms)-{'propext','Classical.choice','Quot.sound'}:raise ValueError('Disallowed proof dependency.')
        audits[name]=axioms
    if set(audits)!=required:raise ValueError('Incomplete theorem audit.')
    checker=prefix/'bin/leanchecker'
    if not checker.is_file():raise RuntimeError('Compatible leanchecker unavailable.')
    env=dict(os.environ,LEAN_PATH=str(fresh),PYTHONDONTWRITEBYTECODE='1')
    run([str(checker),'--fresh','-v','CollisionBound'],'lean_fresh_replay',env=env)
    return {'declaration_axioms':audits,'scope':'Known finite label-count identity; no proof of linguistic labels, parser correctness, prediction accuracy, novelty or population validity. Fresh kernel replay on this host, not an independent kernel implementation.'}

def main():
    if not __debug__:raise RuntimeError('Do not use Python -O.')
    ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--psl',type=Path)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--synthetic-only',action='store_true');modes.add_argument('--existing-results',type=Path)
    ap.add_argument('--python-only',action='store_true',help='Allowed only with --synthetic-only; omits the formal and empirical gates.')
    a=ap.parse_args()
    if a.python_only and not a.synthetic_only:ap.error('--python-only requires --synthetic-only')
    if not a.synthetic_only and not a.existing_results and not a.psl:ap.error('Full reproduction requires --psl.')
    work=a.work_dir.resolve()
    if any(x in {'source_material','master_references'} for x in work.parts) or work.is_relative_to(HERE):raise ValueError('Use a work directory outside this repository and protected inputs.')
    work.mkdir(parents=True,exist_ok=True);fresh=Path(tempfile.mkdtemp(prefix='terminal-verify-',dir=work));commands=[]
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'IN_PROGRESS','commands':commands}
    def run(cmd,name,env=None,timeout=7200):
        log=fresh/(name+'.log')
        with log.open('w') as f:p=subprocess.run(cmd,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        commands.append({'name':name,'exit_code':p.returncode,'log':log.name})
        if p.returncode:raise RuntimeError(name+' failed; see '+str(log))
        return log.read_text()
    try:
        manifest=json.loads((HERE/'verification_inputs.json').read_text())
        if not isinstance(manifest,dict) or set(manifest)!=REQUIRED_FILES:raise ValueError('Incomplete or unexpected manifest file set.')
        if not all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in manifest.values()):raise ValueError('Invalid manifest digest.')
        for rel,expected in manifest.items():
            p=HERE/rel
            if p.is_symlink() or not p.resolve(strict=True).is_relative_to(HERE) or sha(p)!=expected:raise ValueError('Repository manifest mismatch: '+rel)
        report['source_hashes']=manifest;report['synthetic']=synthetic()
        if not a.python_only:report['formal']=formal(fresh,run)
        if a.synthetic_only:report['status']='PASS_SYNTHETIC_PYTHON_ONLY' if a.python_only else 'PASS_SYNTHETIC_AND_FORMAL_NO_EMPIRICAL_CHECK'
        else:
            result=a.existing_results.resolve(strict=True) if a.existing_results else fresh/'results.json'
            if not a.existing_results:
                run([sys.executable,'-B',str(HERE/'reproduce.py'),'--psl',str(a.psl.resolve(strict=True)),'--output',str(result)],'reproduce',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            actual=json.loads(result.read_text())
            if actual.get('status')!='PASS_FRESH_RECOMPUTATION' or actual.get('input_unchanged') is not True or actual.get('input_sha256')!=model.PSL_SHA:raise ValueError('Missing input/result binding.')
            if actual.get('test_was_previously_examined') is not True or actual.get('all_selection_complete_before_reproduction_test_scoring') is not True or len(actual.get('fit_events',[]))!=27:raise ValueError('Incomplete fit/chronology record.')
            versions=actual.get('versions',{})
            if versions.get('numpy')!='2.4.4' or versions.get('scipy')!='1.17.1' or versions.get('unicode')!='15.0.0' or not re.match(r'^3\.12\.3(?:\s|$)',versions.get('python','')):raise ValueError('Unexpected recorded numerical environment.')
            for filename,expected in actual['source_hashes'].items():
                if sha(HERE/filename)!=expected:raise ValueError('Numerical implementation changed: '+filename)
            maximum=reproduce.compare(actual['science'],json.loads((HERE/'reference_results.json').read_text()))
            report['empirical']={'result_sha256':sha(result),'maximum_absolute_difference':maximum,'mode':'Existing aggregates, no refit' if a.existing_results else 'Fresh 27-fit reproduction'}
            run([sys.executable,'-B',str(HERE/'draw_figures.py'),'--results',str(result),'--output',str(fresh/'figures'),'--anonymous'],'tables_and_figure')
            report['status']='PASS_EXISTING_AGGREGATES_AND_FORMAL_NO_REFIT' if a.existing_results else 'PASS_FRESH_RECOMPUTATION_AND_FORMAL_REPLAY'
        for rel,expected in manifest.items():
            if sha(HERE/rel)!=expected:raise ValueError('Repository changed during verification: '+rel)
    except Exception as e:report.update(status='FAIL',error=type(e).__name__+': '+str(e))
    (fresh/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'record':str(fresh/'verification.json')}),flush=True)
    if report['status']=='FAIL':raise RuntimeError(report['error'])
if __name__=='__main__':main()
