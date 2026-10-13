"""Recompute all manuscript data from completed BOPTEST runs, without rerunning trials.

All 320 matched challenge blocks and all 16 coalitions are retained.
The 4000 bootstrap replicates resample entire blocks within each fixed reference
cell. All plots can subsequently be rebuilt from the small packaged source_data.
"""
from pathlib import Path
import pprint, hashlib, math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DEST=ROOT/'source_data'
RAW=ROOT / 'raw_data'
CFG=[f'{i:04b}' for i in range(16)]
ATK=['sensor_bias','sensor_dropout','command_injection','combined']
LEV=['weak','moderate','strong','extreme']; B=4000; SEED=20260922

def main():
    m=pd.read_csv(RAW/'challenge_manifest.csv').set_index('challenge_id')
    ids=[cid for a in ATK for l in LEV for cid in sorted(m[(m.attack_type==a)&(m.intensity_label==l)].index)]
    o=pd.read_csv(RAW/'paired_outcomes.csv',dtype={'mechanism_config':str})
    assert len(o)==5136 and len(ids)==320
    q=o.pivot(index='challenge_id',columns='mechanism_config',values='phi_exec_evaluation_mean').loc[ids,CFG].to_numpy()
    y=o.pivot(index='challenge_id',columns='mechanism_config',values='Y_security').loc[ids,CFG].to_numpy()
    z=[f'plant_post_hvac_reaZon{s}_TZon_y' for s in ['Cor','Eas','Nor','Sou','Wes']]
    cols=['trial_id','challenge_id','mechanism_config','attack_type_truth','intensity_label_truth','time_s','next_time_s','attack_start_s_truth','attack_end_s_truth','phi_exec_evaluation_end_s_truth','b_exec','a_exec','s_exec','L_exec_s','phi_exec','execution_state','active_safe_path','next_safe_path','protection_transient_kind','final_command_k','plant_diag_post_hvac_reaAhu_TSup_y']+z
    t=pd.read_csv(RAW/'physical_security_timeseries.csv',usecols=cols,dtype={'mechanism_config':str})
    assert len(t)==5136*72 and (t.next_time_s-t.time_s).eq(300).all()
    np.testing.assert_allclose(t.phi_exec,np.where(t.L_exec_s<=900,t[['b_exec','a_exec','s_exec']].min(axis=1),0),atol=1e-13)
    ref={c:g.set_index('next_time_s')[z] for c,g in t[t.attack_type_truth=='benign'].groupby('mechanism_config')}
    ev=t[t.challenge_id.isin(ids)&(t.time_s>=t.attack_start_s_truth)&(t.time_s<t.phi_exec_evaluation_end_s_truth)].copy()
    ev['k']=((ev.time_s-ev.attack_start_s_truth)//300).astype(int)
    idx=pd.MultiIndex.from_product([ids,CFG,range(54)],names=['challenge_id','mechanism_config','k'])
    ev=ev.set_index(['challenge_id','mechanism_config','k']).loc[idx].reset_index()
    phi=ev.phi_exec.to_numpy().reshape(320,16,54);dp=phi-phi[:,[0],:];delta=q-q[:,[0]]
    np.testing.assert_allclose(phi.mean(axis=2),q,atol=1e-12)
    gain=np.maximum(dp,0).mean(axis=2);harm=np.maximum(-dp,0).mean(axis=2);hn=np.maximum(-delta,0)
    np.testing.assert_allclose(gain-harm,delta,atol=1e-12)
    # Same 54 step endpoints as the execution-quality evaluation window.
    distances=[]
    for c in CFG:
        s=ev[ev.mechanism_config==c]
        v=np.abs(s[z].to_numpy()-ref[c].loc[s.next_time_s].to_numpy()).mean(axis=1).reshape(320,54).mean(axis=1)
        distances.append(v)
    j=np.column_stack(distances);improve=j[:,[0]]-j
    rng=np.random.default_rng(SEED);weights=np.zeros((B,320))
    for cell in range(16):
        draw=rng.integers(0,20,size=(B,20))
        for b in range(B):weights[b,cell*20:cell*20+20]=np.bincount(draw[b],minlength=20)/320
    boot=weights@delta
    variables={'E_Q':delta,'R_time':gain,'H_time':harm,'H_net':hn,'J':j,'physical_improvement':improve,'E_T':phi[:,:,-1]-phi[:,[0],-1],'tau_Y':y-y[:,[0]]}
    summary={'config':CFG,'Q_U':q[:,0].mean(),'Q_P':q.mean(axis=0)}
    for name,v in variables.items():
        draws=weights@v;lo,hi=np.quantile(draws,[.025,.975],axis=0)
        summary.update({name:v.mean(axis=0),name+'_low':lo,name+'_high':hi})
    s=pd.DataFrame(summary);s.to_csv(DEST/'configuration_results.csv',index=False)
    old=pd.read_csv(RAW/'analysis/BOPTEST_SECURITY_PROTECTION_EFFECTIVENESS.csv',dtype={'mechanism_config':str}).set_index('mechanism_config').loc[CFG]
    np.testing.assert_allclose(s.E_Q,old.E_sec,atol=1e-12)
    np.testing.assert_allclose(s.H_time,old.H_exec_time,atol=1e-12)
    cells=[]
    for cell in range(16):
        sl=slice(cell*20,(cell+1)*20);w=weights[:,sl]*16
        for k,c in enumerate(CFG):
            r={'attack':ATK[cell//4],'intensity':LEV[cell%4],'config':c}
            for name,v in [('E_Q',delta),('physical_improvement',improve)]:
                lo,hi=np.quantile(w@v[sl,k],[.025,.975]);r.update({name:v[sl,k].mean(),name+'_low':lo,name+'_high':hi})
            cells.append(r)
    pd.DataFrame(cells).to_csv(DEST/'condition_results.csv',index=False)
    dyn=[]
    for ai,a in enumerate(ATK):
        sl=slice(ai*80,(ai+1)*80);v=np.cumsum(dp[sl,15],axis=1)/np.arange(1,55);lo,hi=np.quantile((weights[:,sl]*4)@v,[.025,.975],axis=0)
        dyn.extend({'attack':a,'minute':(k+1)*5,'mean':v[:,k].mean(),'low':lo[k],'high':hi[k]} for k in range(54))
    pd.DataFrame(dyn).to_csv(DEST/'dynamics.csv',index=False)
    W=np.zeros((16,4))
    for k in range(4):
        bit=1<<(3-k)
        for c in range(16):
            if c&bit:continue
            n=c.bit_count();v=math.factorial(n)*math.factorial(3-n)/24;W[c|bit,k]+=v;W[c,k]-=v
    shap=delta@W;abl=delta[:,[15]]-delta[:,[7,11,13,14]]
    mech={'mechanism':list('RCIH')}
    for name,v in [('shapley',shap),('ablation',abl)]:
        lo,hi=np.quantile(weights@v,[.025,.975],axis=0);mech.update({name:v.mean(axis=0),name+'_low':lo,name+'_high':hi})
    pd.DataFrame(mech).to_csv(DEST/'mechanism_results.csv',index=False)
    pd.DataFrame({'contrast_RCIH_RCI':boot[:,15]-boot[:,14]}).to_csv(DEST/'bootstrap_contrast.csv',index=False)
    # Deterministic exemplar selection near the median within each moderate cell.
    reps=[]
    for a in ATK:
        ix=[i for i,cid in enumerate(ids) if m.loc[cid,'attack_type']==a and m.loc[cid,'intensity_label']=='moderate']
        med=np.median(delta[ix,15]);best=min(ix,key=lambda i:(round(abs(delta[i,15]-med),12),ids[i]));reps.append(ids[best])
    # Fixed exemplar matches the requested reference PDF's transient example.
    example='v2_formal_peak_heat_day_combined_extreme_020'
    traces=[]
    for cid in reps+[example]:
        for c in ['0000','1111']:
            g=t[(t.challenge_id==cid)&(t.mechanism_config==c)].sort_values('time_s').copy()
            baseline=ref[c].loc[g.next_time_s].to_numpy()
            g['zone_mean_C']=g[z].mean(axis=1)-273.15;g['benign_zone_mean_C']=baseline.mean(axis=1)-273.15
            g['zone_distance_K']=np.abs(g[z].to_numpy()-baseline).mean(axis=1)
            g['minute']=(g.time_s-g.attack_start_s_truth)/60;traces.append(g)
    pd.concat(traces).to_csv(DEST/'traces.csv',index=False)
    m.loc[ids].to_csv(DEST/'challenge_design.csv')
    evsmall=ev[['challenge_id','mechanism_config','k','phi_exec']]
    evsmall.to_csv(DEST/'evaluation_phi.csv.gz',index=False,compression='gzip')
    pairs=[]
    for i,cid in enumerate(ids):
        for k,c in enumerate(CFG):pairs.append({'challenge_id':cid,'config':c,'attack':m.loc[cid,'attack_type'],'intensity':m.loc[cid,'intensity_label'],'Q':q[i,k],'E_Q':delta[i,k],'R_time':gain[i,k],'H_time':harm[i,k],'J':j[i,k],'physical_improvement':improve[i,k]})
    pd.DataFrame(pairs).to_csv(DEST/'paired_results.csv',index=False)
    audits={'all_trials':len(o),'attacked_challenges':len(ids),'configurations':16,'time_samples_per_evaluation':54,'replicates_per_cell':20,'bootstrap':B,'seed':SEED,'max_error':float(np.max(np.abs(s.E_Q-old.E_sec.to_numpy()))),'representative_ids':reps,'transient_id':example,'negative_net_challenges':int((delta[:,15]<0).sum()),'any_time_harmed_challenges':int((harm[:,15]>0).sum()),'time_harmed_fraction':float((dp[:,15]<0).mean()),'paired_contrast_ci':np.quantile(boot[:,15]-boot[:,14],[.025,.975]).tolist(),'physical_J_U':float(j[:,0].mean()),'physical_J_P':float(j[:,15].mean()),'benign_between_config_difference':float(max(np.max(np.abs(ref[c].to_numpy()-ref['0000'].to_numpy())) for c in CFG)),'source_hashes':{f:hashlib.sha256((RAW/f).read_bytes()).hexdigest() for f in ['paired_outcomes.csv','challenge_manifest.csv']}}
    (DEST/'calculation_audit.txt').write_text(pprint.pformat(audits, sort_dicts=False),encoding='utf-8')
    print(pprint.pformat(audits, sort_dicts=False));print(s[s.config.isin(['1111','1110','0001'])][['config','E_Q','E_Q_low','E_Q_high','E_T','tau_Y','J','physical_improvement']].to_string(index=False))
    print(pd.DataFrame(cells).query("config == '1111' and attack == 'command_injection'").to_string(index=False))
    g=pd.concat(traces);g=g[(g.challenge_id==example)&(g.mechanism_config=='1111')]
    print(g.loc[g.execution_state.ne(g.execution_state.shift()),['minute','time_s','execution_state','active_safe_path','phi_exec']].to_string(index=False))
if __name__=='__main__':main()
