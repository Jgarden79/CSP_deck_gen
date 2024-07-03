import csp_2 as csp
import options_dat as od
import deck as deck
import pandas as pd
import os
import datetime as dt
import shutil

reqs = 'upload_template.csv'

def generate_decks(requests_path:str):
    reqs = pd.read_csv(requests_path)
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
        return

def create_master_upload():
    files = [pd.read_csv(f'trading_files/{i}') for i in os.listdir('trading_files')]
    master_upload = pd.concat(files, ignore_index=True)
    master_upload = master_upload.drop_duplicates()
    [shutil.move(f'trading_files/{i}', 'archive/trading_files') for i in os.listdir('trading_files')]
    date = dt.date.today().strftime('%Y%m%d')
    master_upload.to_csv(f'trading_files/master_upload_{date}.csv', index=False)
