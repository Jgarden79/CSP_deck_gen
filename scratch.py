import csp_2 as csp
import options_dat as od
import deck as deck
import pandas as pd

# print(pd.read_pickle("assets/COST_cached_chain.pkl"))

k = csp.CspClient(client_name="Chustz, Randy", RIC="MPC", shares=5428 ,green_light='n',cost_basis_per_share=36.20)
k.get_sym()
k.create_gap(shares_to_hedge=int(k.shares/2))
k.gap.create_gap_plots()
k.create_cov_call(shares_to_hedge=int(k.shares/2))
k.cov_call.create_cc_plots()
k.create_sef(shares_to_hedge=int(k.shares/3))
k.sef.create_sef_plots()
a = deck.CspDeck(csp_client=k)
a.create_csp_report_ex_sef()
# a.create_csp_report()
