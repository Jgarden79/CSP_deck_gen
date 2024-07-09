import pathlib
import time
import eikon as ek
import numpy as np
import pandas as pd
import os

import configparser as cp
cfg=cp.ConfigParser()
ek.set_app_key(os.getenv('eikon'))

import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
path = 'assets'
isExist = os.path.exists(path) # check if images exists
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)

# for f in os.listdir(path):
#     os.remove(os.path.join(path,f))

def get_options(RICLST = list):
    """
    gets options data from eikon based on rics in list
    :param RICLST: list of rics to get options data form
    :return: stores all data pulled in a pkl file
    """
    for RIC in RICLST: # go through list of rics
        x = ek.get_symbology(RIC, from_symbol_type='RIC', to_symbol_type='ticker') # go from Ric to ticker
        x.iloc[0]['ticker'] = x.iloc[0]['ticker'].replace(".","") # ensure ticker format
        x.to_pickle('assets/{}_sym.pkl'.format(RIC)) # save the ticker to pkl
        name_d = ek.get_data(RIC, fields=['TR.CommonName'])[0] # get the asset name
        name_d.to_pickle('assets/{}_name.pkl'.format(RIC)) # save name to pkl
        sym = x.iloc[0]['ticker'] # save tikcer as sym
        und = ek.get_data(RIC, fields = ['CF_LAST', 'YIELD'])[0] ###Gets Underlying Price###
        und = und.fillna(0)
        und.to_pickle('assets/{}_cached_und.pkl'.format(sym)) # save unerlying data to pkl
        sym_2 = RIC.split('.')[0] # save ric
        sym_2 = sym_2.upper() # make sure its in upper cas
        Request = '0#'+sym_2+'*.U' # set up request for options chain
        fields = ['PUTCALLIND', 'EXPIR_DATE', 'STRIKE_PRC', 'CF_BID', 'CF_ASK', 'IMP_VOLT', 'DELTA'] # list of fields
        chain = ek.get_data(Request, fields = fields)[0] # get the option data
        chain.to_pickle('assets/{}_cached_chain.pkl'.format(sym)) # store options chains in pkl
    return

def get_spx_options():
    sym = '.SPX'
    und = ek.get_data(sym, fields = ['CF_LAST', 'YIELD'])[0] ###Gets SPX Price###
    und.to_pickle('assets/spx_cached_und.pkl') # save unerlying data to pkl
    fields = ['PUTCALLIND', 'EXPIR_DATE', 'STRIKE_PRC', 'CF_BID', 'CF_ASK', 'IMP_VOLT', 'DELTA'] # list of fields
    Request = '0#SPX*.U' # set up request for options chain
    chain = ek.get_data(Request, fields = fields)[0] # get the option data
    chain.to_pickle('assets/spx_cached_chain.pkl') # store options chains in pkl
    
def get_xsp_options():
    count = 0
    while count < 5:
        try:
            sym = '.XSP'
            und = ek.get_data(sym, fields = ['CF_LAST', 'YIELD'])[0] ###Gets XSP Price###
            und.to_pickle('assets/xsp_cached_und.pkl') # save unerlying data to pkl
            fields = ['PUTCALLIND', 'EXPIR_DATE', 'STRIKE_PRC', 'CF_BID', 'CF_ASK', 'IMP_VOLT', 'DELTA'] # list of fields
            Request = '0#XSP*.U' # set up request for options chain
            chain = ek.get_data(Request, fields = fields)[0] # get the option data
            name_d = ek.get_data('.XSP', fields=['TR.CommonName'])[0]  # get the asset name
            name_d.to_pickle('assets/xsp_name.pkl')  # save name to pkl
            chain.to_pickle('assets/xsp_cached_chain.pkl') # store options chains in pkl
            break
        except Exception as e:
            count+=1
            time.sleep(5)


def get_bonds():
    """
    gets ust data for use with options trades
    :return: stores bond data as pkl
    """
    bnd_lst = ek.get_data('0#USTSY=', fields = 'MATUR_DATE')[0] # get all us treasuries
    bnd_lst.to_pickle('assets/cached_bndlst.pkl') # store in lst
    bnd_cus = [x for x in bnd_lst['Instrument']] # put cusips in a list
    bnd_dat = ek.get_data(bnd_cus,fields = ['SETTLEDATE','DIRTY_PRC', 'COUPN_RATE', 'PAR_AMT', 'MATUR_DATE',
                                            'TR.FiMaturityYearsToRedem', 'SEC_YLD_1'])[0] # list of fields
    bnd_dat.to_pickle('assets/bond_data.pkl') # store bond dat
    return

def calc_credit(ident, get_data = 'n'):
    """
    calculates yields for BSC based trades
    :param ident: <str> the idendifier for the Bond ETF in use
    :param get_data: <str> "y" to refresh data
    :return: <flt> ytm for use with Options trades
    """
    if get_data.lower() == 'y':
        url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker={}".format(ident)
        hold = pd.read_csv(url)  # get holdings
        hold = hold.dropna(subset=[" MaturityDate"]) # remove cash
        hold = hold.filter(['Security Identifier', ' PercentageOfFund']) # keep cusips and weights
        try:
            ytw = ek.get_data(list(hold['Security Identifier']), fields =['TR.FiWorstCorpYield'])[0] # import ytws
        except:
            time.sleep(3) # if there is an exception wait three secs and try again
            ytw = ek.get_data(list(hold['Security Identifier']), fields=['TR.FiWorstCorpYield'])[0]  # import ytws
        hold['YTW'] = list(ytw['Worst Corporate Yield']/100) # add to df
        hold = hold.dropna()
        hold[' PercentageOfFund'] = hold[' PercentageOfFund']/100 # convert weights to sub 1
        ytw_agg = np.dot(hold['YTW'], hold[' PercentageOfFund'])/hold[' PercentageOfFund'].sum() # calculate fund wieght
        pickle.dump(ytw_agg, open("assets/{}_ytw.pkl".format(ident), "wb"))
    else:
        ytw_agg = pickle.load(open("assets/{}_ytw.pkl".format(ident), 'rb')) # call from file if no update needed
    print("YTW for {} is {}".format(ident, ytw_agg))

    return ytw_agg # return weight for use

def get_zeros():
    '''
    gets UST PO Strips data for use with options trades
    :return: stores bond data as pkl
    '''

    zro_lst=ek.get_data('0#USTPO=',fields='MATUR_DATE')[0] #gets all US Treasuries
    zro_lst.to_pickle('assets/cached_zrolst.pkl') # store in lst
    zro_cus=[x for x in zro_lst['Instrument']] #puts CUSIPs in a list
    zro_dat=ek.get_data(zro_cus,fields=['SETTLEDATE','DIRTY_PRC','COUPN_RATE','PAR_AMT','MATUR_DATE',
                                        'TR.FiMaturityYearsToRedem','SEC_YLD_1'])[0] #list of fields
    zro_dat.to_pickle('assets/zero_data.pkl') # store bond dat
    zeros=pd.read_pickle('assets/zero_data.pkl')
    print(zeros.dropna())
    return

def check_data(RIC:str):
    try:
        # Get the file's modification time
        file_path = pathlib.Path(__file__).parent / 'assets' / f"{RIC}_name.pkl"
        file_mod_time = os.path.getmtime(file_path)
        # Get the current time
        current_time = time.time()
        # Calculate the age of the file in seconds
        file_age_seconds = current_time - file_mod_time
        # Convert the age to hours
        file_age_hours = file_age_seconds / 3600
        return file_age_hours
    except:
        return 999