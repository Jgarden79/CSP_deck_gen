import pandas as pd
import numpy as np
import os
import time
import pandas as pd
import datetime as dt
import eikon as ek
from pathlib import Path
import sef as sef
import csp_gap as gp
import cov_call as ccl
import options_dat as od

today = dt.date.today()
eik_api = os.getenv('eikon')
ek.set_app_key(eik_api)

loc_path = Path(__file__).parent

#################REMOVE ONCE DEVELOPMENT IS COMPLETE#################
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.options.display.float_format = '{:.4f}'.format


#####################################################################

class CspClient:
    def __init__(self, client_name: str, RIC: str, shares: float, sef: str = 'y', cov_call: str = 'y', gap: str = 'y',
                 green_light: str = 'n', cost_basis_per_share: float = 0.0):
        self.last = None
        self.sym = None
        self.client_name = client_name
        self.RIC = RIC
        self.shares = shares
        self.sef_approved = sef
        self.sef = None
        self.cov_call_approved = cov_call
        self.cov_call = None
        self.gap_approved = gap
        self.gap = None
        self.green_light = green_light.lower()
        self.cost_basis_per_share = cost_basis_per_share

    def get_sym(self):
        age = od.check_data(self.RIC)
        if age > 99:
            od.get_options([self.RIC])
        self.sym = pd.read_pickle('assets/{}_sym.pkl'.format(self.RIC)).iloc[0]['ticker']  # import the symbol

    def create_sef(self, shares_to_hedge: int):
        if self.sef_approved == 'n':
            print("Client has not approved SEF")
            return
        else:
            age = od.check_data(self.RIC)
            if age > 99:
                chain = od.get_options([self.RIC])
            else:
                chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.sym))
            dates = list(chain['EXPIR_DATE'].dropna().unique())
            dates = pd.to_datetime(dates)
            dates_series = pd.to_datetime(dates)

            # Get date
            exp_date = pd.to_datetime(dt.datetime.now().date() + dt.timedelta(days=185))

            # Find the date closest to today
            sef_date = dates_series[np.abs(dates_series - exp_date).argmin()].strftime('%Y-%m-%d')
            synt = sef.Sef(self.RIC, expiration=sef_date, shares=shares_to_hedge, asset_risk=0.1, spy_risk=0.1)
            synt.generate_sef()
            self.sef = synt

    def create_cov_call(self, shares_to_hedge: int):
        if self.cov_call_approved == 'n':
            print("Client has not approved Cov Call")
            return
        else:
            age = od.check_data(self.RIC)
            if age > 99:
                chain = od.get_options([self.RIC])
            else:
                chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.sym))
            dates = list(chain['EXPIR_DATE'].dropna().unique())
            dates = pd.to_datetime(dates)
            dates_series = pd.to_datetime(dates)

            # Get today's date
            exp_date = pd.to_datetime(dt.datetime.now().date() + dt.timedelta(days=90))

            # Find the date closest to today
            cc_date = dates_series[np.abs(dates_series - exp_date).argmin()].strftime('%Y-%m-%d')
            cov_call = ccl.CoverdCalls(self.RIC, expiration=cc_date, shares=shares_to_hedge,
                                       cost_basis_per_share=self.cost_basis_per_share)
            cov_call.generate_covered_call()
            self.cov_call = cov_call

    def create_gap(self, shares_to_hedge: int):
        if self.gap_approved == 'n':
            print("Client has not approved Gap trade")
            return
        else:
            age = od.check_data(self.RIC)
            if age > 99:
                chain = od.get_options([self.RIC])
            else:
                chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.sym))
            dates = list(chain['EXPIR_DATE'].dropna().unique())
            dates = pd.to_datetime(dates)
            dates_series = pd.to_datetime(dates)

            # Get today's date
            exp_date = pd.to_datetime(dt.datetime.now().date() + dt.timedelta(days=395))
            gap_date = dates_series[np.abs(dates_series - exp_date).argmin()].strftime('%Y-%m-%d')
            gap_trade = gp.CspGap(self.RIC, get_data='n', expiry=gap_date, shares=shares_to_hedge, gap=5, prot=20,
                                  exclude_div='y')
            gap_trade.calc_trade()
            self.gap = gap_trade


j = CspClient(client_name="ABC", RIC='NVDA.O', shares=39600, cost_basis_per_share=25.0)
j.get_sym()
j.create_sef(13200)
j.sef.SEF_payoff_plots()
j.create_cov_call(13200)
j.cov_call.create_cc_plots()
j.create_gap(13200)
j.gap.create_gap_plots()
j.gap.data_tab()
print(j.cov_call.S_CALL)
print(j.gap.net_opt_cost)
print(j.gap.adj_time_left)
print(j.gap.trade_delta)
print(j.gap.data_table)
print(j.sef.collar)
print(j.sef.synthetic)

