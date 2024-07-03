import csp_2 as csp
import options_dat as od
import deck as deck
import pandas as pd

# print(pd.read_pickle("assets/COST_cached_chain.pkl"))

# k = csp.CspClient(client_name="Carrier, James", RIC="AMD.O", shares=11000 ,green_light='n',cost_basis_per_share=85)
# k.get_sym()
# k.create_gap(shares_to_hedge=int(k.shares/2))
# k.gap.create_gap_plots()
# k.create_cov_call(shares_to_hedge=int(k.shares/2))
# k.cov_call.create_cc_plots()
# k.create_sef(shares_to_hedge=int(k.shares/3))
# k.sef.create_sef_plots()
# a = deck.CspDeck(csp_client=k)
# a.create_csp_report_ex_sef()
# a.create_csp_report()

reqs = pd.read_csv('upload_template.csv')

for i in range(0, len(reqs)):
    rq = reqs.iloc[i]
    name  = rq['name']
    ric = rq['ric']
    shares = rq['shares']
    cb = rq['cbps']
    k = csp.CspClient(client_name=name, RIC=ric, shares=shares ,green_light='n',cost_basis_per_share=cb)
    k.get_sym()
    k.create_gap(shares_to_hedge=int(rq['gap_shares']))
    k.gap.create_gap_plots()
    k.create_cov_call(shares_to_hedge=int(rq['cc_shares']))
    k.cov_call.create_cc_plots()
    k.create_sef(shares_to_hedge=int(rq['sef_shares']))
    k.sef.create_sef_plots()
    a = deck.CspDeck(csp_client=k)
    if rq['full_deck']=='n':
        a.create_csp_report_ex_sef()
    else:
        a.create_csp_report()