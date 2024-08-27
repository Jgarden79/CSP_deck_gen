import math
import yfinance as yf
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
from matplotlib import colors

today = dt.date.today()
eik_api = os.getenv('EIKON_API.KEY')
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


class CspGap:

    def __init__(self, RIC:str, expiry:str, gap:int, prot:int, shares:float, exclude_div, custom_div='n', tgt_yld=0,
                 fee=0.0125, get_data = 'n'):
        self.div_dat = None
        self.RIC=RIC
        self.get_data=get_data
        self.expiry=expiry
        self.gap=gap/100
        self.prot=prot/100
        self.exclude_div=exclude_div
        self.custom_div=custom_div
        self.tgt_yld=tgt_yld/100
        self.fee=fee
        self.shares=shares
        self.rng=0.5 # range for payoff chart (+/- %)

        if self.get_data == 'y':  # get data if necessary
            od.get_options([self.RIC])
        else:  # otherwise pass
            pass
        self.sym = pd.read_pickle('assets/{}_sym.pkl'.format(self.RIC)).iloc[0]['ticker']  # import the symbol
        self.und = pd.read_pickle('assets/{}_cached_und.pkl'.format(self.sym))  ##Gets Underlying Price###
        self.last = self.und.iloc[0]['CF_LAST']  # get most recent price
        self.contracts = self.shares / 100
        self.port_val = self.shares * self.last



    def calc_trade(self):
        '''Calculates strategy trades'''
        # get div data
        if self.custom_div == 'n':
            if self.exclude_div == 'n':
                self.div_yld = self.und.iloc[0]['YIELD']
                if type(self.div_yld) == np.float64:
                    self.QTR_div_yld = self.div_yld / 400
                else:
                    self.div_yld = 0
                    self.QTR_div_yld = 0
            else:
                self.div_yld = 0
                self.QTR_div_yld = 0
        else:
            self.div_yld = input('Enter Annual Dividend Yield:')
            self.div_yld = float(self.div_yld)
            self.QTR_div_yld = float(self.div_yld) / 4

        self.ann_div_yld = self.div_yld / 100

        # options chain set up
        chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.RIC))  # read in the options chaing to pandas
        chain['MID'] = (chain['CF_BID'] + chain["CF_ASK"]) / 2
        trade_chain = chain[chain['EXPIR_DATE'] == self.expiry]  # isolate the chain we want
        sorted_trade_chain = trade_chain.sort_values(['PUTCALLIND', 'STRIKE_PRC'],
                                                 axis=0, )  # sort the chain by type and strike
        self.sorted_trade_chain_calls = sorted_trade_chain[sorted_trade_chain['PUTCALLIND'] == 'CALL']  # create a call df
        self.sorted_trade_chain_puts = sorted_trade_chain[sorted_trade_chain['PUTCALLIND'] == 'PUT ']  # create a put df

        # ATM CALL
        self.ATM = self.sorted_trade_chain_calls.loc[
                self.sorted_trade_chain_calls['STRIKE_PRC'].sub(self.last).abs().idxmin()]  # get atm call
        self.notional = self.last  # trade's notional is the last traded price of the underlying

        # Solve for Put option legs
        self.LPUT = self.sorted_trade_chain_puts.loc[
                self.sorted_trade_chain_puts['STRIKE_PRC'].sub(self.last * (1 - self.gap)).abs().idxmin()]  # get long put
        self.SPUT = self.sorted_trade_chain_puts.loc[
                self.sorted_trade_chain_puts['STRIKE_PRC'].sub(self.LPUT['STRIKE_PRC'] * (1 - self.prot)).abs().idxmin()] # get short put

        date = dt.datetime.strptime(self.expiry, '%Y-%m-%d').date()  # convert date to datetime
        today = dt.datetime.now().date()  # get today's date
        time_left = date - today  # days left
        self.adj_time_left = time_left / dt.timedelta(days=1)  # convert to flt
        adj_time_left_div = np.floor(time_left / dt.timedelta(days=1) / 365 * 4)
        self.plug = np.round(adj_time_left_div * self.QTR_div_yld * self.last, 2)  # total divs expected
        self.plug = np.nan_to_num(self.plug, 0)  # if no divs replace with 0
        self.inflows = self.SPUT['MID'] + self.plug  # calculate all cash inflows
        self.tgt_call_val = self.LPUT['MID'] - self.inflows  # calculate target call value

        # self.SCALL = self.sorted_trade_chain_calls.loc[
        #         self.sorted_trade_chain_calls['STRIKE_PRC'].sub(50).abs().idxmin()]
        self.SCALL = self.sorted_trade_chain_calls.loc[
                self.sorted_trade_chain_calls['MID'].sub(self.tgt_call_val + (self.last*self.tgt_yld)).abs().idxmin()]  # get short call

        self.net_opt_cost=self.LPUT['MID'] - self.SPUT['MID'] - self.SCALL['MID'] #net cost of options (total)
        self.net_opt_cost_dlrs=self.net_opt_cost*100
        self.drag = self.net_opt_cost - self.plug
        self.drag_pct = self.drag / self.last
        self.trade_delta = -self.LPUT['DELTA']-self.SPUT['DELTA']-self.SCALL['DELTA']

        #caclualte management fee
        self.mgt_fee_pct = self.fee * (self.adj_time_left / 365)

        self.SPUT['Trans'] = 'SPO'  # add trade type
        self.LPUT['Trans'] = 'BPO'  # add trade type
        self.SCALL['Trans'] = 'SCO'  # add trade type

        Trade = pd.concat([self.SCALL.to_frame().T,
                           self.LPUT.to_frame().T,
                           self.SPUT.to_frame().T])  # create Trade df
        exp_date = pd.to_datetime(Trade['EXPIR_DATE'].iloc[0]).strftime('%y%m%d')  # get options formated date
        option_type = [s[0] for s in Trade["PUTCALLIND"].to_list()]  # isolate first letter of option type
        strikes = Trade['STRIKE_PRC'].to_list()  # # isolate strikes in list
        option_sym = ['{}{}{}{}'.format(self.sym, self.expiry,
                                        option_type[i],
                                        int(strikes[i])) for i in range(0, len(Trade))]  # create symbols
        Trade['Symbol'] = option_sym  # add to df
        trade_details = Trade.filter(['Trans', 'Symbol', 'STRIKE_PRC', "MID"])  # create trade_det df
        self.upside = (self.SCALL['STRIKE_PRC'] / self.last) - 1  # upside calculation
        self.annual_up = self.upside * (365 / self.adj_time_left)
        self.protection = np.abs((self.LPUT['STRIKE_PRC'] - self.SPUT['STRIKE_PRC']) / self.last)  # protection calculation
        self.ds_before = 1 - (self.LPUT['STRIKE_PRC'] / self.last)
        self.annual_p = self.protection * (365 / self.adj_time_left)
        self.months_left = np.round((self.adj_time_left / 365) * 12, 1)
        chrt_rng = np.linspace(self.ATM['STRIKE_PRC'] * (1 - self.rng), self.ATM['STRIKE_PRC'] * (1 + self.rng), 50,
                               dtype=float)  # set chart range
        chrt_rng = np.round(chrt_rng, 2)
        scall_ev = [np.maximum(p - self.SCALL['STRIKE_PRC'], 0) * -1 for p in chrt_rng]  # calc scall end val
        lput_ev = [np.maximum(self.LPUT['STRIKE_PRC'] - p, 0) for p in chrt_rng]  # calc lcall end val
        sput_ev = [np.maximum(self.SPUT['STRIKE_PRC'] - p, 0) * -1 for p in chrt_rng]  # calc sput end val
        perf_df = pd.DataFrame({"{} Price".format(self.sym): chrt_rng,
                                "SCALL": scall_ev,
                                "LPUT": lput_ev,
                                "SPUT": sput_ev,
                                "UND": chrt_rng,
                                "DIVS": [self.plug for i in range(0, len(chrt_rng))]})  # Create the df
        perf_df = perf_df.set_index("{} Price".format(self.sym))  # Set the mkt px as the index
        perf_df['Trade'] = perf_df.sum(axis=1)  # calculate total value
        self.cost = self.LPUT['MID'] - self.SCALL['MID'] - self.SPUT[
            'MID'] + self.last  # total trade cost including underlying
        perf_df['Trade Return'] = (perf_df['Trade'] / (self.cost)) - 1  # trade return
        perf_df['Trade Return - Net'] = perf_df['Trade Return'] - self.mgt_fee_pct
        perf_df = perf_df.sort_index(ascending=False)  # reorder in descending
        perf_df['{} Price Return'.format(self.sym)] = [(p / self.last) - 1 for p in
                                                       perf_df.index]  # add underlying performance
        rets_tab = perf_df.filter(perf_df.columns[-3:]).reset_index()  # reset index
        rets_tab['Trade Return Net'] = [i - float(self.fee * self.adj_time_left) for i in
                                        perf_df['Trade Return'].to_list()]
        fc_date = pd.to_datetime(Trade['EXPIR_DATE'].iloc[0]).strftime("%m-%d-%Y")

        if self.exclude_div == 'n':  # use div data for chart
            self.div_dat = " "
        else:
            self.div_dat = " Excluding Dividends "  # insert if no divs


    def _payoff_plot(self):

        '''Prints strategy payodd diagram at expiration for presentation decks'''

        # Functions to calculate options payoffs at expiration
        def call_payoff(stock_range,strike,premium):
            return np.where(stock_range>strike,stock_range-strike,0)-premium

        def put_payoff(stock_range,strike,premium):
            return np.where(stock_range<strike,strike-stock_range,0)-premium

        # Define stock price range at expiration
        up_dn = 0.5
        stock_range=np.arange((1-up_dn)*self.last,(1+up_dn)*self.last,1)

        # Calculate payoffs for individual legs
        long_put=self.LPUT['STRIKE_PRC']
        long_put_prem=self.LPUT['MID']
        short_put=self.SPUT['STRIKE_PRC']
        short_put_prem=self.SPUT['MID']
        short_call=self.SCALL['STRIKE_PRC']
        short_call_prem=self.SCALL['MID']

        payoff_long_put=put_payoff(stock_range,long_put,long_put_prem)
        payoff_short_put=put_payoff(stock_range,short_put,short_put_prem)*-1
        payoff_short_call=call_payoff(stock_range,short_call,short_call_prem)*-1

        # Calculate Strategy Payoff
        strategy=(((payoff_short_put+payoff_long_put+payoff_short_call+stock_range+self.plug)/self.last)-1)
        strategy_net=(((payoff_short_put+payoff_long_put+payoff_short_call+stock_range+self.plug)/self.last)-1) - self.mgt_fee_pct
        buy_hold_ret=((stock_range/(self.last))-1)

        self.cap = (self.SCALL['STRIKE_PRC'] / self.last) - 1
        protection = np.abs(((self.LPUT['STRIKE_PRC'] - self.SPUT['STRIKE_PRC'])  / self.last))  # protection calculation
        gap = (self.LPUT['STRIKE_PRC'] / self.last) - 1 #calcualte gap

        #Create Visualization
        plt.style.use('fivethirtyeight')
        fig,ax=plt.subplots(figsize=(14,6))

        from matplotlib.ticker import PercentFormatter
        plt.yticks(np.arange(-up_dn,up_dn+0.01,0.2),fontsize=16)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xticks(fontsize=16)
        plt.gca().xaxis.set_major_formatter('${x:1.0f}')

        plt.plot(stock_range,buy_hold_ret,c='grey',lw=3,ls='dashed',label='{} Price Return'.format(self.sym))
        plt.plot(stock_range,strategy,c='green',lw=3,label='Strategy Return (gross)')
        plt.plot(stock_range,strategy_net,lw=2,label='Strategy Return (net)')

        plt.vlines(x=self.last,ymin=buy_hold_ret.min(),ymax=buy_hold_ret.max(),linestyle='dashed',color='grey',lw=2)
        plt.annotate('Current Price: ${:.2f}'.format(self.last),
                xy=(self.last,(buy_hold_ret.max()-0)*0.5),fontsize=12,
                rotation=90,
                horizontalalignment='right',verticalalignment='center')
        plt.hlines(y=0,xmin=stock_range.min(),xmax=stock_range.max(),color='gray')

        plt.legend(loc='upper left',fontsize=14)
        plt.ylabel('Return at Expiration',fontsize=16,fontweight='bold')
        plt.xlabel('{} Price at Expiration'.format(self.sym),fontsize=16,fontweight='bold')
        plt.suptitle('Stock: {}     Current Price: ${:.2f}     Cap & Cushion Strategy'.format(
                                    self.sym,
                                    self.last),
                     fontsize=16)
        plt.title('Options expiration: {}'.format(self.expiry),fontsize=14)

        plt.tight_layout()
        plt.savefig(img_path / '{}_gap_trade_payoff_plot.png'.format(self.sym))
        # plt.savefig('images/{}_gap_trade_payoff_plot.png'.format('BRK-B'))

    def data_tab(self):
        '''Creates strategy data table for use in presentations'''
        # Functions to calculate options payoffs at EXPIRY
        def call_payoff(stock_range,strike,premium):
            return np.where(stock_range>strike,stock_range-strike,0)-premium

        def put_payoff(stock_range,strike,premium):
            return np.where(stock_range<strike,strike-stock_range,0)-premium
        # Calculate payoffs for individual legs
        long_put=self.LPUT['STRIKE_PRC']
        long_put_prem=self.LPUT['MID']
        short_put=self.SPUT['STRIKE_PRC']
        short_put_prem=self.SPUT['MID']
        short_call=self.SCALL['STRIKE_PRC']
        short_call_prem=self.SCALL['MID']
        # Create DataFrame of Stock Prices every 5%
        pct_range=1+np.arange(-0.30,0.31,0.05)
        percent_range=pct_range*self.last
        payoff_long_put=put_payoff(percent_range,long_put,long_put_prem) * self.shares
        payoff_short_put=put_payoff(percent_range,short_put,short_put_prem) * self.shares * -1
        payoff_short_call = call_payoff(percent_range,short_call,short_call_prem) * self.shares * -1
        # Calculate Strategy Payoff
        stock_pl_5=(percent_range-self.last)*self.shares
        strat_pl_5=payoff_long_put + payoff_short_put + payoff_short_call + stock_pl_5
        strat_pl_5_net=strat_pl_5 - (self.fee * (self.adj_time_left)/365 * self.port_val)
        ret=pct_range-1
        ret_idx = [f'{x:.0%}' for x in ret ]
        df=pd.DataFrame({'return':ret_idx,f'{self.sym} with Hedge':strat_pl_5_net/self.port_val, f'{self.sym}':stock_pl_5/self.port_val})
        df.set_index('return',inplace=True)
        df=df.T
        self.data_table=df
        df = df.applymap(lambda x: f'{x:.1%}')
        # Plotting the table
        fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size to fit your data
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

        # Customizing header color
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#639DD9')  # Light blue color
            cell.set_edgecolor('black')  # Set gridline color
            cell.set_linewidth(1)  # Set gridline width
            cell.set_text_props(ha='center', va='center', fontsize=14)  # Center text and set font size
            # cell.set_aa(0.2)  # Set padding
            cell.PAD = 0.1
            cell.set_height(0.12)  # Adjust cell height
            cell.set_width(0.2)

        # Save the table as an image
        plt.savefig(img_path / f"return_at_expiration_table_{self.sym}_gap.png", bbox_inches='tight', pad_inches=0.1)



    def _historical_chart(self,period='2y'):
        '''
        Creates historical daily price chart of underlying stock or ETF using non-adjusted close prices from YFinance.
        Intended for use in presentation decks

        :param period: <str> The lookback period of daily prices ('1y', '3y', '5y', '10y')

        '''
        if self.sym == 'BRKB':
            self.sym = 'BRK-B'
        else:
            pass
        df=yf.download('{}'.format(self.sym),period=period,)['Close'].to_frame()
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(14,8))
        plt.plot(df['Close'],label=self.sym)
        plt.tick_params(labeltop=False,labelright=True)

        plt.hlines(y=self.SCALL['STRIKE_PRC'],xmin=df.index[0],xmax=df.index[-1],
                      linestyle='dashed',color='red',alpha=0.5,
                      label='Cap: ${} (+{:.1%})'.format(self.SCALL['STRIKE_PRC'],(self.SCALL['STRIKE_PRC']/self.last)-1))
        plt.hlines(y=self.last,xmin=df.index[0],xmax=df.index[-1],
                      linestyle='dashed',color='gray',alpha=0.5,
                      label='Current Price: ${}'.format(self.last))
        plt.hlines(y=self.LPUT['STRIKE_PRC'],xmin=df.index[0],xmax=df.index[-1],
                      linestyle='dashed',color='green',alpha=0.5,
                      label='Protection begins: ${} ({:.1%})'.format(self.LPUT['STRIKE_PRC'],(self.LPUT['STRIKE_PRC']/self.last)-1))
        plt.hlines(y=self.SPUT['STRIKE_PRC'],xmin=df.index[0],xmax=df.index[-1],
                      linestyle='dashed',color='green',alpha=0.5,
                      label='Protection ends: ${} ({:.1%})'.format(self.SPUT['STRIKE_PRC'],(self.SPUT['STRIKE_PRC']/self.last)-1))

        plt.title('Stock: {}     |     Trade Duration: {:.1f}yrs     |     Last: ${:.2f}'.format(self.sym,self.adj_time_left/365,self.last),
                    fontsize=16,fontweight='bold')
        plt.legend(loc='upper left',fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        plt.xlabel('Date',fontsize=16,fontweight='bold')
        # if self.sym == 'BRK.B':
        plt.ylabel(f'{self.sym} Price',fontsize=16,fontweight='bold')
        plt.tight_layout()
        plt.savefig(img_path / '{}_{}r_price_chart_CapCush.png'.format(self.sym,period))


    def create_gap_plots(self):
        self._payoff_plot()
        self.data_tab()
        self._historical_chart()


# a = CspGap(RIC='AAPL.OQ', get_data='n', expiry='2025-06-20', gap=5, prot=15, exclude_div='n', shares=1500)
# a.calc_trade()
# a.payoff_plot()
# a.data_tab()
# a.historical_chart()