"""Original table formatting from the current manuscript pipeline."""
def integer(x):return f'{x:,}'.replace(',','.')


def pt(x,n=4):return f'{x:.{n}f}'.replace('.',',')


def table(c,h,r,n):return {'caption':c,'headers':h,'rows':r,'note':n}


ENDING={'ends_ico_ica_icos_icas':'ico/ica/icos/icas','ends_logo_loga_logos_logas':'logo/loga/logos/logas',
 'remainder_not_morphologically_classified':'Restante'}


COUNTS=['observations','both_correct','only_all_position_correct','only_terminal5_correct','neither_correct']


def assets(science):
 s=science; t=science; j=science['census']
 tables={'population':table('Tipos, grupos gráficos sem marcas e classes de referência no PSL.',
  ['Partição','Tipos','Grupos','Última','Penúltima','Antepenúltima'],
  [[lab,integer(s['splits'][k]['types']),integer(s['splits'][k]['groups']),*map(integer,s['splits'][k]['class_counts'])]
   for k,lab in [('train','Treino'),('validation','Validação'),('test','Teste')]],
  'Cada tipo tem um único rótulo. Os grupos gráficos são disjuntos entre partições; os tipos têm peso uniforme.')}
 rows=[]
 for rep,label in [('all_marks_removed','Sem marcas'),('acute_circumflex_removed','Retirada seletiva')]:
  for family,name in [('all_position','Palavra'),('terminal_tagged_1to5','Terminal5'),('terminal_subset_1to4','Terminal4')]:
   m=t['models'][rep+'__'+family];v=m['test']
   rows.append([label,name,pt(100*v['accuracy']),pt(v['macro_f1']),pt(v['log_loss'],6),integer(v['confusion_true_rows'][2][2]),pt(100*v['recall_by_class'][2])])
 tables['terminal_results']=table('Desempenho dos seis procedimentos no teste de 19.266 tipos.',
  ['Entrada','Modelo','Ac. (%)','F1 macro','Perda log.','Acertos AP','R AP (%)'],rows,
  'AP: classe antepenúltima (2.491 tipos); R: recuperação; Ac.: acurácia. Retirada seletiva remove agudo e circunflexo. Cada procedimento tem vocabulário, normalização e ajuste próprios; todos selecionaram C = 10 por perda de validação.')
 q=j['overall_ap_counts']
 tables['paired_ap']=table('Acertos pareados dos 2.491 tipos antepenúltimos sem marcas.',
  ['Palavra','Terminal5 acerta','Terminal5 erra','Total'],
  [['Acerta',integer(q['both_correct']),integer(q['only_all_position_correct']),integer(q['all_position_correct'])],
   ['Erra',integer(q['only_terminal5_correct']),integer(q['neither_correct']),integer(q['observations']-q['all_position_correct'])],
   ['Total',integer(q['terminal5_correct']),integer(q['observations']-q['terminal5_correct']),integer(q['observations'])]],
  'A diferença líquida de quatro acertos não identifica os 120 tipos em que somente um procedimento acerta.')
 rows=[]
 for x in j['one_dimensional_margins']['ending_group']:
  if not x['observations']:continue
  rows.append([ENDING[x['ending_group']],integer(x['observations']),integer(x['all_position_correct']),pt(100*x['all_position_correct']/x['observations'],2),integer(x['terminal5_correct']),pt(100*x['terminal5_correct']/x['observations'],2)])
 tables['ap_strata']=table('Recuperação antepenúltima por terminações gráficas.',
  ['Terminação','Tipos','Palavra','R (%)','Terminal5','R (%)'],rows,
  'Contagens de acertos e recuperação na condição sem marcas. Categorias gráficas mutuamente exclusivas; o restante não é uma classe morfológica.')
 coverage={label:{k:0 for k in COUNTS} for label in ['<5','5']}
 for x in j['one_dimensional_margins']['longest_training_terminal_feature_letters']:
  cell=coverage['5' if x['longest_training_terminal_feature_letters']==5 else '<5']
  for k in COUNTS:cell[k]+=x[k]
 tables['joint_coverage']=table('Cobertura terminal e resultados pareados na classe antepenúltima.',
  ['Cobertura','Tipos','Ambos','Só Palavra','Só Terminal5','Nenhum'],
  [[label,*[integer(coverage[label][k]) for k in COUNTS]] for label in ['<5','5']],
  'Cobertura: comprimento da maior terminação presente em pelo menos dois tipos de treino. Ambos/nenhum indicam acerto conjunto/erro conjunto. Todos esses tipos têm pelo menos cinco caracteres sem marcas.')
 rows=[]
 for ending in ENDING:
  for cov in ['<5','5']:
   x=next(x for x in j['binary_five_letter_coverage_crossings']['ending_group'] if x['ending_group']==ending and x['five_letter_coverage']==cov)
   rows.append([ENDING[ending],cov,integer(x['observations']),integer(x['all_position_correct']),integer(x['terminal5_correct']),integer(x['neither_correct'])])
 tables['ending_coverage']=table('Terminações e cobertura nos mesmos tipos antepenúltimos.',
  ['Terminação','Cobertura','Tipos','Palavra','Terminal5','Nenhum'],rows,
  'Palavra e Terminal5: acertos; nenhum: erros conjuntos. Cruzamento descritivo posterior aos resultados marginais; não separa efeitos causais das dimensões.')
 rows=[];classes={'antepenult':'AP','penult':'P','final':'O'}
 for cell in j['public_examples_selection']['cells']:
  x=cell['example'];assert x['recorded_label']=='antepenult'
  rows.append([x['canonical_public_spelling'],x['fully_folded_model_input'],x['recorded_syllable_count'],str(x['longest_training_terminal_feature_letters']),classes[x['all_position_prediction']],classes[x['terminal5_prediction']]])
 assert len(rows)==8
 tables['lexical_examples']=table('Exemplos determinados pelo critério de seleção do censo.',
  ['Grafia PSL','Entrada','Sílabas','Cobertura','Palavra','Terminal5'],rows,
  'Referência antepenúltima em todos os casos. AP: antepenúltima; P: penúltima. Um exemplo por resultado pareado × cobertura <5/5, escolhido pelo menor hash com prefixo fixo. Ilustrações não representativas; conservam a anotação do PSL.')
 figures={'terminal_coverage':'Recuperação dos rótulos antepenúltimos por cobertura terminal. Os números junto aos pontos são acertos/tipos. Cobertura mede disponibilidade de atributos no treino; os grupos não isolam causas e os pontos não representam respostas de falantes.'}
 return tables,figures,coverage

