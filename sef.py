import math

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import eikon as ek
from pathlib import Path
import options_dat as od
import seaborn as sns

today = dt.date.today()
eik_api = os.getenv('eikon')
ek.set_app_key(eik_api)

loc_path = Path(__file__).parent
img_path = loc_path / 'assets' / 'images'
img_path.mkdir(exist_ok=True)

#################REMOVE ONCE DEVELOPMENT IS COMPLETE#################
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.options.display.float_format = '{:.4f}'.format
#####################################################################



class Sef:

    def __init__(self, RIC:str, expiration:str, shares:int,spy_risk:float, asset_risk:float, custom_div='n',fee=0.0125,xsp='n'):
        self.collar_performace = None
        self.spy_contracts = None
        self.collar = None
        self.port_val = None
        self.last = None
        self.sym = None
        self.mkt_sym = None
        self.RIC = RIC
        self.shares = shares
        self.expiration = expiration
        self.spy_risk = spy_risk
        self.asset_risk = asset_risk
        self.custom_div = custom_div
        self.fee = fee
        self.xsp = None

    def _load_asset_data(self):
        age = od.check_data(self.RIC)
        if age >=2: # change back to 2 after dev
            od.get_options([f'{self.RIC}'])
        self.sym = pd.read_pickle('assets/{}_sym.pkl'.format(self.RIC)).iloc[0]['ticker']  # import the symbol
        chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.sym))  # read in the options chaing to pandas
        chain['MID'] = (chain['CF_BID'] + chain["CF_ASK"]) / 2
        trade_chain = chain[chain['EXPIR_DATE'] == self.expiration]  # isolate the chain we want
        sorted_trade_chain = trade_chain.sort_values(['PUTCALLIND', 'STRIKE_PRC'],
                                                 axis=0, )  # sort the chain by type and strike
        und = pd.read_pickle('assets/{}_cached_und.pkl'.format(self.sym))  ##Gets Underlying Price###
        self.last = und.iloc[0]['CF_LAST']  # get most recent price
        self.sorted_trade_chain = sorted_trade_chain.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    def _load_mkt_data(self):
        if self.xsp == 'n':
            mkt = 'SPY'
            age = od.check_data(mkt)
            if age >= 2:
                od.get_options(['SPY'])
        else:
            mkt = 'xsp'
            age = od.check_data('xsp')
            if age >= 2:
                od.get_xsp_options()
        self.mkt_sym = mkt# import the symbol
        chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.mkt_sym))  # read in the options chaing to pandas
        chain['MID'] = (chain['CF_BID'] + chain["CF_ASK"]) / 2
        trade_chain = chain[chain['EXPIR_DATE'] == self.expiration]  # isolate the chain we want
        sorted_trade_chain = trade_chain.sort_values(['PUTCALLIND', 'STRIKE_PRC'],
                                                 axis=0, )  # sort the chain by type and strike
        und = pd.read_pickle('assets/{}_cached_und.pkl'.format(self.mkt_sym))  ##Gets Underlying Price###
        self.last_mkt = und.iloc[0]['CF_LAST']  # get most recent price
        self.sorted_trade_chain_mkt = sorted_trade_chain.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def _gen_collar(self):
        self._load_asset_data()
        self.port_val = self.shares * self.last
        tgt_long_put = self.last * (1 - self.asset_risk)
        puts = self.sorted_trade_chain[self.sorted_trade_chain['PUTCALLIND'] == 'PUT']
        calls = self.sorted_trade_chain[self.sorted_trade_chain['PUTCALLIND'] == 'CALL']
        puts['DIF'] = (puts['STRIKE_PRC'] - tgt_long_put).abs()
        long_put = puts.loc[puts['DIF'].idxmin()]
        tgt_call = long_put['MID']
        calls['DIF'] = (calls['MID'] - tgt_call).abs()
        short_call = calls.loc[calls['DIF'].idxmin()]
        collar = pd.concat([long_put.to_frame().T,short_call.to_frame().T], axis=0)
        collar = collar.drop(['DIF'], axis=1)
        collar['Trade'] = ['BPO', 'SCO']
        self.collar = collar
        self.collar_cost = (long_put['MID'] - short_call['MID']) * (self.shares//100) * 100

    def _gen_synthetic(self):
        # SPY OTM Put
        self._load_mkt_data()
        puts = self.sorted_trade_chain_mkt[self.sorted_trade_chain_mkt['PUTCALLIND'] == 'PUT']
        calls = self.sorted_trade_chain_mkt[self.sorted_trade_chain_mkt['PUTCALLIND'] == 'CALL']
        SPUT = puts.loc[
                puts['STRIKE_PRC'].sub(self.last_mkt * (1 - self.spy_risk)).abs().idxmin()]  # get SPY otm put
        LCALL = calls.loc[
                calls['MID'].sub(SPUT['MID']).abs().idxmin()] # SPY long call
        synthetic = pd.concat([SPUT.to_frame().T, LCALL.to_frame().T], axis=0)
        synthetic['Trade'] = ['SPO', 'BCO']
        self.spy_contracts = math.floor((self.port_val / self.last_mkt) / 100)
        self.synthetic = synthetic
        self.synthetic_cost = (LCALL['MID'] - SPUT['MID']) * self.spy_contracts * 100

    def generate_sef(self):
        self._gen_collar()
        self._gen_synthetic()
        self.sef = pd.concat([self.collar, self.synthetic], ignore_index=True)

    def _calc_profit_loss(self, price_range, long_put, short_call):
        # Calculate profit/loss for collar strategy
        stock_profit_loss = (price_range - self.last) * self.shares
        long_put_profit_loss = np.maximum(long_put['STRIKE_PRC'] - price_range, 0) * self.shares - long_put['MID'] * self.shares
        short_call_profit_loss = np.minimum(price_range - short_call['STRIKE_PRC'], 0) * self.shares + short_call['MID'] * self.shares
        collar_profit_loss = stock_profit_loss + long_put_profit_loss + short_call_profit_loss
        return collar_profit_loss

    def SEF_payoff_plots(self):

        '''Generates strategy payoff diagram at expiration for presentation decks'''

        rng = 0.5
        date = dt.datetime.strptime(self.expiration, '%Y-%m-%d').date()  # convert date to datetime
        time_left = date - today  # days left
        adj_time_left = time_left / dt.timedelta(days=1)  # convert to flt
        dte = adj_time_left

        # Functions to calculate options payoffs at expiry
        def call_payoff(stock_range,strike,premium):
            return np.where(stock_range>strike,stock_range-strike,0)-premium
        def put_payoff(stock_range,strike,premium):
            return np.where(stock_range<strike,strike-stock_range,0)-premium

        ########################################
        # CONCENTRATED STOCK #
        # Define stock price range at expiration
        stock_range=np.arange((1-rng)*self.last,(1+rng)*self.last,1)
        mgt_fee = self.fee * (dte / 365) * self.port_val

        # Calculate payoffs for individual legs

        long_put=self.collar[self.collar['Trade'] == 'BPO'].iloc[0]['STRIKE_PRC']
        long_put_prem=self.collar[self.collar['Trade'] == 'BPO'].iloc[0]['MID']
        short_call=self.collar[self.collar['Trade'] == 'SCO'].iloc[0]['STRIKE_PRC']
        short_call_prem=self.collar[self.collar['Trade'] == 'SCO'].iloc[0]['MID']

        payoff_long_put=put_payoff(stock_range,long_put,long_put_prem) * self.shares
        payoff_short_call=call_payoff(stock_range,short_call,short_call_prem) * self.shares * -1

        # Calculate Strategy Payoff
        stock_pl=(stock_range-self.last)*self.shares
        strategy_pl=payoff_long_put+payoff_short_call+stock_pl
        strategy_pl_net=payoff_long_put+payoff_short_call+stock_pl+mgt_fee

        # Create DataFrame of Stock Prices every 5%
        pct_range=1+np.arange(-0.30,0.31,0.05)
        percent_range=pct_range*self.last
        # Caluclate P&L from stock, short call and covered call
        payoff_lp=put_payoff(percent_range,long_put,long_put_prem) * self.shares
        payoff_sc=call_payoff(percent_range,short_call,short_call_prem) * self.shares * -1
        # Calculate Strategy Payoff
        stock_pl_5=(percent_range-self.last)*self.shares
        strat_pl_5=payoff_lp+payoff_sc+stock_pl_5
        strat_pl_5_net=payoff_lp+payoff_sc+stock_pl_5+mgt_fee
        ret=pct_range-1
        df=pd.DataFrame({'return' : ret,'collar' : strat_pl_5/self.port_val,'collar (net)' : strat_pl_5_net/self.port_val,'stock' : stock_pl_5/self.port_val})
        df.set_index('return',inplace=True)
        df=df.T
        self.collar_performace = df
        ### Create Visualization
        plt.style.use('fivethirtyeight')
        fig,ax=plt.subplots(figsize=(14,6))
        plt.plot(stock_range,stock_pl,c='grey',lw=3,ls='dashed',label='{} only'.format(self.sym))
        plt.plot(stock_range,strategy_pl,c='green',lw=3,label='{} with Collar (gross)'.format(self.sym))
        plt.plot(stock_range,strategy_pl_net,lw=2,label='{} with Collar (net)'.format(self.sym))
        plt.vlines(x=self.last,ymin=stock_pl.min(),ymax=stock_pl.max(),linestyle='dashed',color='grey',lw=2)
        plt.annotate('Current Stock Price',
                        xy=(self.last,(stock_pl.max()-0)*0.5),
                        fontsize=12,
                        rotation=90,
                        horizontalalignment='right',
                        verticalalignment='center')
        plt.hlines(y=0,xmin=stock_range.min(),xmax=stock_range.max(),color='gray')
        plt.ylabel('Profit / Loss',fontsize=16,fontweight='bold')
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        plt.xlabel('{} Price at Expiration'.format(self.sym),fontsize=16,fontweight='bold')
        plt.xticks(fontsize=14)
        plt.gca().xaxis.set_major_formatter('${x:,.0f}')
        plt.suptitle('Stock: {}     Current Price: ${:.2f}     Option Moneyness: {:.1%} | {:.1%}'.format(self.sym,
                                                                                                         self.last,
                                                                                                         long_put / self.last,
                                                                                                         short_call/self.last),
              fontsize=16)
        plt.title('Options expiration: {}'.format(self.expiration),fontsize=14)
        plt.legend(loc='best',fontsize=14)

        plt.tight_layout()
        plt.savefig(img_path / '{}_collar_payoff'.format(self.sym))


        # SYNTHETIC EXPOSURE TO SPY #
        # Define price range at expiration
        spy_range=np.arange((1-rng)*self.last_mkt,(1+rng)*self.last_mkt,1)
        # Calculate payoffs for individual legs
        short_put=self.synthetic[self.synthetic['Trade']=='SPO'].iloc[0]['STRIKE_PRC']
        short_put_prem=self.synthetic[self.synthetic['Trade']=='SPO'].iloc[0]['MID']
        long_call=self.synthetic[self.synthetic['Trade']=='BCO'].iloc[0]['STRIKE_PRC']
        long_call_prem=self.synthetic[self.synthetic['Trade']=='BCO'].iloc[0]['MID']
        payoff_short_put=put_payoff(spy_range,short_put,short_put_prem) * self.spy_contracts * -100
        payoff_long_call=call_payoff(spy_range,long_call,long_call_prem) * self.spy_contracts * 100
        # Calculate Strategy Payoff
        spy_pl = (spy_range - self.last_mkt) * self.spy_contracts * 100
        synthetic_pl = payoff_short_put + payoff_long_call
        synthetic_pl_net = synthetic_pl - self.fee
        ### Create Visualization
        plt.style.use('fivethirtyeight')
        fig,ax=plt.subplots(figsize=(14,6))
        plt.plot(spy_range,spy_pl,c='grey',lw=3,ls='dashed',label='{} only'.format('SPY'))
        plt.plot(spy_range,synthetic_pl,c='green',lw=3,label='{} Synthetic (gross)'.format('SPY'))
        plt.plot(spy_range,synthetic_pl_net,lw=2,label='{} Synthetic (net)'.format('SPY'))
        plt.vlines(x=self.last_mkt,ymin=spy_pl.min(),ymax=spy_pl.max(),linestyle='dashed',color='grey',lw=2)
        plt.annotate('Current Stock Price',
                        xy=(self.last_mkt,(spy_pl.max()-0)*0.5),
                        fontsize=12,
                        rotation=90,
                        horizontalalignment='right',
                        verticalalignment='center')
        plt.hlines(y=0,xmin=spy_range.min(),xmax=spy_range.max(),color='gray')
        plt.ylabel('Profit / Loss',fontsize=16,fontweight='bold')
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        plt.xlabel('{} Price at Expiration'.format('SPY'),fontsize=16,fontweight='bold')
        plt.xticks(fontsize=14)
        plt.gca().xaxis.set_major_formatter('${x:,.0f}')
        plt.suptitle('Stock: {}     Current Price: ${:.2f}     Option Moneyness: {:.1%} | {:.1%}'.format('SPY',
                                                                                                         self.last_mkt,
                                                                                                         short_put / self.last_mkt,
                                                                                                         long_call/self.last_mkt),
              fontsize=16)
        plt.title('Options expiration: {}'.format(self.expiration),fontsize=14)
        plt.legend(loc='best',fontsize=14)

        plt.tight_layout()
        plt.savefig(img_path / 'SPY_synthetic_payoff_{}'.format(self.sym))

        ## sensetivity
        spy_rng_pct = pct_range * self.last_mkt
        payoff_short_put=put_payoff(spy_rng_pct,short_put,short_put_prem) * self.spy_contracts * -100
        payoff_long_call=call_payoff(spy_rng_pct,long_call,long_call_prem) * self.spy_contracts * 100
        synthetic_pl = payoff_short_put + payoff_long_call #13
        a = np.array(strat_pl_5)
        b = np.array(synthetic_pl)
        result_matrix = a[:, np.newaxis] + b
        starting = (self.last * self.shares)
        result_matrix = result_matrix - mgt_fee
        result_matrix = result_matrix/starting
        adj_pct = pct_range -1
        adj_pct = [f"{i:.0%}" for i in adj_pct]
        df = pd.DataFrame(result_matrix, columns=adj_pct, index=adj_pct)
        df = df.T * 100
        annot = df.applymap(lambda x: f"{x:.1f}%")


        # Plotting the heatmap

        plt.figure(figsize=(14, 10))
        heatmap = sns.heatmap(df, annot=annot, cmap="RdYlGn", fmt="", center=0, linewidths=.5, cbar=False)


        # Customize the plot to match the provided image
        plt.xlabel(f'{self.sym} Price Return', fontsize=14)
        plt.ylabel('SPY Price Return', fontsize=14)

        # Rotate the x-axis labels
        plt.xticks(rotation=45)

        # Show the plot
        plt.tight_layout()
        plt.savefig(img_path / 'heatmap_{}'.format(self.sym))

    def create_sef_plots(self):
        self.SEF_payoff_plots()





