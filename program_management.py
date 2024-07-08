import csp_2 as csp
import options_dat as od
import deck as deck
import pandas as pd
import datetime as dt
import shutil
from pathlib import Path
import re
import refinitiv.data as rd
import refinitiv.data.session as sess
import refinitiv.data.content.symbol_conversion as sc
import account as act
import os
import pickle

app_key = os.getenv('rdp')
pd.options.mode.chained_assignment = None  # default='warn'

reqs = 'upload_template.csv'
current_loc = Path(__file__).parent
pst_trad = current_loc / 'post_trade'
pst_trad.mkdir(exist_ok=True, parents=True)
valid = current_loc / 'validation_files'

pre_trd = current_loc / 'pre_trade'
pre_trd.mkdir(exist_ok=True, parents=True)
today = dt.date.today().strftime('%Y-%m-%d')
pre_trade_file = pre_trd / f'pre_trade_{today}.csv'
gap_rec = current_loc / 'pm_records' / 'gap_records'
gap_rec.mkdir(exist_ok=True, parents=True)
trade_dat = current_loc / 'pm_records' / 'trade_data'
trade_dat.mkdir(exist_ok=True, parents=True)
pm_recs = current_loc / 'pm_records'
pm_recs.mkdir(exist_ok=True, parents=True)
acts = pm_recs / 'accounts'
acts.mkdir(exist_ok=True, parents=True)


def generate_decks(requests_path: str):
    reqs = pd.read_csv(requests_path)
    for i in range(0, len(reqs)):
        rq = reqs.iloc[i]
        name = rq['name']
        ric = rq['ric']
        shares = rq['shares']
        cb = rq['cbps']
        k = csp.CspClient(client_name=name, RIC=ric, shares=shares, green_light='n', cost_basis_per_share=cb)
        k.get_sym()
        k.create_gap(shares_to_hedge=int(rq['gap_shares']))
        k.gap.create_gap_plots()
        k.create_cov_call(shares_to_hedge=int(rq['cc_shares']))
        k.cov_call.create_cc_plots()
        k.create_sef(shares_to_hedge=int(rq['sef_shares']))
        k.sef.create_sef_plots()
        a = deck.CspDeck(csp_client=k)
        if rq['full_deck'] == 'n':
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


def tick_to_ric(ticker: str):
    sess.desktop.Definition(app_key=app_key).get_session()
    rd.open_session(app_key=app_key)
    a = sc.Definition(symbols=ticker.upper(), from_symbol_type=sc.SymbolTypes.TICKER_SYMBOL,
                      to_symbol_types=sc.SymbolTypes.RIC, preferred_country_code="G:6J").get_data()
    ric = a.data.df['RIC'].iloc[0]
    rd.close_session()
    return ric


def generate_trade(act_no: str, name: str, ticker: str, gap: int, sef: int, cov_call: int, cb: float, custodian: str,
                   exclude_div: str = 'y'):
    ric = tick_to_ric(ticker)
    tot_shares = gap + sef + cov_call
    if gap > 0:
        g = 'y'
    else:
        g = 'n'
    if sef > 0:
        s = 'y'
    else:
        s = 'n'
    if cov_call > 0:
        c = 'y'
    else:
        c = 'n'
    cbps = cb / tot_shares
    j = act.CspAccount(account_number=act_no, client_name=name, RIC=ric, shares=tot_shares, green_light='y',
                       cost_basis=cbps, gap_shares=gap, sef_shares=sef, cov_call_shares=cov_call, gap=g, sef=s,
                       cov_call=c,
                       exclude_div=exclude_div, custodian=custodian.upper())
    j.get_sym()
    j.execute_trades()
    save_to_pickle(j)
    print(f"Trades created for {j.account_number} - {ticker}: gap: {gap}, sef: {sef}, cov_call: {cov_call}")


def run_pretrade():
    file_path = pre_trade_file
    df = pd.read_csv(file_path)
    for i in range(0, len(df)):
        order = df.iloc[i]
        generate_trade(act_no=order['act_no'], name=order['name'], ticker=order['ticker'], gap=order['gap_shares'],
                       sef=order['sef_shares'], cov_call=order['cov_call_shares'], custodian=order['cust'],
                       exclude_div=order['exclude'], cb=order['cb'])
    create_master_upload()
    return


def load_post_trade():
    """Load post trade file"""
    date = dt.date.today().strftime('%Y%m%d')
    file_name = None
    for i in os.listdir(pst_trad):
        if date in i:
            file_name = i
            break
    post_trade = pd.read_csv(pst_trad / f'{file_name}')
    return post_trade


def save_to_pickle(account: act.CspAccount):
    with open(acts / f'{account.account_number}.pkl', 'wb') as f:
        pickle.dump(account, f)
    print(f'Saved {account.account_number} to pickle file')
    return


def load_from_pickle(act_no: str):
    with open(acts / f'{act_no}.pkl', 'rb') as f:
        j = pickle.load(f)
    return j
