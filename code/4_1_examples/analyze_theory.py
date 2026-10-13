from pathlib import Path
import sys,pprint,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import caculate_th as m
OUT=ROOT/'source_data'

def main():
    m.self_checks()
    raw=pd.read_csv(ROOT/'source/theoretical_case_main.csv')
    calc=m.main_result_table()
    assert raw.configuration.tolist()==calc.configuration.tolist()
    np.testing.assert_allclose(raw.iloc[:,1:],calc.iloc[:,1:],atol=1e-12,rtol=0)
    calc.to_csv(OUT/'all_48_configuration_results.csv',index=False)
    rows=[]; shap=[]; ab=[]; trace=[]; decomp=[]; states=[]
    for topo in m.TOPOLOGIES:
        sv=m.shapley(topo)
        for mech in m.MECHS:
            shap.append({'topology':topo,'mechanism':mech,'shapley':sv[mech]})
            ab.append({'topology':topo,'removed_mechanism':mech,'loss_on_removal':m.e_q(m.MECHS,topo)-m.e_q(set(m.MECHS)-{mech},topo)})
        for conf in m.CONFIGS:
            r,h=m.r_q_h_q(conf,topo)
            vals=[m.q_chain(c,conf,topo)-m.q_chain(c,(),topo) for c in m.SCENARIOS]
            rows.append({'topology':topo,'configuration':m.config_name(conf),'E_Q':m.e_q(conf,topo),'R_Q':r,'H_Q':h,'R_net':sum(max(v,0) for v in vals)/2,'H_net':sum(max(-v,0) for v in vals)/2,'tau_Y':m.tau_y(conf,topo)})
        for c in m.SCENARIOS:
            pts=m.change_points(m.MECHS)
            for l,r in zip(pts[:-1],pts[1:]):
                mid=(l+r)/2
                trace.append({'topology':topo,'scenario':c,'left':l,'right':r,'phi_U':m.phi_exec(mid,c,(),topo),'phi_RCIH':m.phi_exec(mid,c,m.MECHS,topo)})
        r,h=m.r_q_h_q(m.MECHS,topo)
        decomp.append({'topology':topo,'R_Q':r,'H_Q':h,'E_Q':r-h})
    for name,st in [('Normal',m.NORMAL_STATE),('F damaged',m.F_DAMAGED_STATE),('C compromised',m.C_COMPROMISED_STATE)]:
        states.append({'state':name,**st})
    for topo in ['G_D','G_E']:
        for mech in ['R','C']:states.append({'state':topo+' '+mech,**m.PATH_STATE[topo][mech]})
    for name,rr in [('configuration_long',rows),('shapley',shap),('ablation',ab),('exact_trajectories',trace),('full_decomposition',decomp),('raw_state_mapping',states)]:
        pd.DataFrame(rr).to_csv(OUT/(name+'.csv'),index=False)
    # Same generator, draw order, and common p-vectors as the original implementation.
    rng=np.random.default_rng(20260914);urows=[];ranking=[]
    masks=np.array([[mech in subset for mech in m.MECHS] for subset in m.ALL_SUBSETS])
    for pbar in [.70,.85,.95]:
        for kappa in [50,100,200]:
            draws=rng.beta(pbar*kappa,(1-pbar)*kappa,size=(20000,4))
            weights=np.prod(np.where(masks[None,:,:],draws[:,None,:],1-draws[:,None,:]),axis=2)
            np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-12,rtol=0)
            vals={t:weights@np.array([m.e_q(c,t) for c in m.ALL_SUBSETS]) for t in m.TOPOLOGIES}
            row={'p_bar':pbar,'kappa':kappa}
            for t in m.TOPOLOGIES:
                row.update({t+'_mean':vals[t].mean(),t+'_q025':np.quantile(vals[t],.025),t+'_q975':np.quantile(vals[t],.975)})
            urows.append(row)
            ranking.append({'p_bar':pbar,'kappa':kappa,'draws':20000,'strict_order_count':int(((vals['G_E']>vals['G_D'])&(vals['G_D']>vals['G_B'])).sum())})
    unc=pd.DataFrame(urows)
    original=pd.read_csv(ROOT/'source/theoretical_case_uncertainty.csv')
    np.testing.assert_allclose(unc,original,atol=1e-12,rtol=0)
    unc.to_csv(OUT/'uncertainty_all_27_conditions.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'uncertainty_order_audit.csv',index=False)
    pd.DataFrame([{'topology':t,'p_bar':p,'expected_E_Q':m.expected_e_q_given_p(m.MECHS,t,dict.fromkeys(m.MECHS,p))} for t in m.TOPOLOGIES for p in [.7,.85,.95]]).to_csv(OUT/'exact_expected_means.csv',index=False)
    audit={'status':'PASS','deterministic_objects':48,'scenarios_per_object':2,'scenario_weights':[.5,.5],'exact_segments':len(trace),'original_main_csv_match_atol':1e-12,'original_uncertainty_csv_match_atol':1e-12,'uncertainty_draws_per_parameter_setting':20000,'parameter_settings':9,'seed':20260914,'full_configuration':decomp,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').glob('*') if p.is_file()},'scope':'Analytical model; raw state quantities are specified model inputs, not observed engineering data.'}
    (OUT/'calculation_audit.txt').write_text(pprint.pformat(audit, sort_dicts=False),encoding='utf-8')
    print(pprint.pformat({'status':'PASS','full_configuration':decomp,'uncertainty_max_abs_difference':float(np.max(np.abs(unc.to_numpy()-original.to_numpy())))}))

if __name__=='__main__':main()
