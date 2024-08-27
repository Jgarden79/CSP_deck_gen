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



class CoverdCalls:
    def __init__(self, RIC:str, expiration:str, shares:int,cost_basis_per_share:float,fee=0.0125):
        self.premium_percent = None
        self.premium = None
        self.share_value = None
        self.contracts = None
        self.RIC = RIC
        self.expiration = expiration
        self.shares = shares
        self.fee = fee
        self.cost_basis_per_share = cost_basis_per_share
        self.sorted_trade_chain = None

    def _load_asset_data(self):
        age = od.check_data(self.RIC)
        if age >=2: # change back to 2 after dev
            od.get_options([f'{self.RIC}'])
        self.sym = pd.read_pickle('assets/{}_sym.pkl'.format(self.RIC)).iloc[0]['ticker']  # import the symbol
        chain = pd.read_pickle('assets/{}_cached_chain.pkl'.format(self.RIC))  # read in the options chaing to pandas
        chain['MID'] = (chain['CF_BID'] + chain["CF_ASK"]) / 2
        trade_chain = chain[chain['EXPIR_DATE'] == self.expiration]  # isolate the chain we want
        sorted_trade_chain = trade_chain.sort_values(['PUTCALLIND', 'STRIKE_PRC'],
                                                 axis=0, )  # sort the chain by type and strike
        und = pd.read_pickle('assets/{}_cached_und.pkl'.format(self.sym))  ##Gets Underlying Price###
        self.last = und.iloc[0]['CF_LAST']  # get most recent price
        self.sorted_trade_chain = sorted_trade_chain.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def _identify_call(self, hi_vol = 'n'):
        age = od.check_data(self.RIC)
        if age >=2:
            od.get_options([f'{self.RIC}'])
        call_options = self.sorted_trade_chain[self.sorted_trade_chain['PUTCALLIND'] == 'CALL']
        call_options = call_options[call_options['EXPIR_DATE'] == self.expiration]
        if hi_vol == 'y':
            call_options = call_options[call_options['DELTA'] <=0.2]
        else:
            call_options = call_options[call_options['DELTA'] <= 0.3]
        self.S_CALL = call_options.iloc[0]

    def generate_covered_call(self, hi_vol = 'n'):
        self._load_asset_data()
        self._identify_call(hi_vol=hi_vol)
        self.contracts = self.shares//100
        self.share_value = self.shares * self.last
        self.premium = self.contracts * self.S_CALL['MID'] * 100
        self.premium_percent = self.S_CALL['MID'] / self.S_CALL['STRIKE_PRC']

    def _cov_call_plots(self):
        def call_payoff(stock_range,strike,premium):
            return np.where(stock_range>strike,stock_range-strike,0)-premium

        date = dt.datetime.strptime(self.expiration, '%Y-%m-%d').date()  # convert date to datetime
        time_left = date - today  # days left
        adj_time_left = time_left / dt.timedelta(days=1)  # convert to flt
        self.adj_time_left = adj_time_left
        dte = adj_time_left
        rng = 0.5
        # Define stock price range at expiration
        stock_range=np.arange((1 - rng) * self.last, (1 + rng) * self.last, 1)
        self.cc_mgt_fee = self.fee * (dte / 365) * self.shares

        # Calculate payoffs for individual legs
        short_call=self.S_CALL['STRIKE_PRC']
        short_call_prem=self.S_CALL['MID']

        payoff_short_call = call_payoff(stock_range,short_call,short_call_prem) * self.shares * -1
        self.payoff_short_call = payoff_short_call

        # Calculate Strategy Payoff
        stock_pl = (stock_range - self.last) * self.shares
        strategy_pl = payoff_short_call + stock_pl
        strategy_pl_net = strategy_pl - self.cc_mgt_fee

        # Create DataFrame of Stock Prices every 5%
        pct_range=1+np.arange(-0.30,0.31,0.05)
        percent_range=pct_range*self.last
        # Caluclate P&L from stock, short call and covered call
        payoff_sc=call_payoff(percent_range,short_call,short_call_prem) * self.shares * -1

        # Calculate Strategy Payoff
        stock_pl_5=(percent_range-self.last)*self.shares
        strat_pl_5=payoff_sc+stock_pl_5
        strat_pl_5_net = strat_pl_5 - self.cc_mgt_fee
        ret=pct_range-1
        ret_idx = [f'{x:.0%}' for x in ret]
        df=pd.DataFrame({'return':ret_idx,f'{self.sym} Covered Call':strat_pl_5_net/self.share_value,f'{self.sym}':stock_pl_5/self.share_value})
        df.set_index('return',inplace=True)
        df=df.T
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
        plt.savefig(img_path / f"return_at_expiration_table_cov_{self.sym}.png", bbox_inches='tight', pad_inches=0.1)

        # create visualization
        plt.style.use('fivethirtyeight')
        fig,ax=plt.subplots(figsize=(14,6))
        plt.plot(stock_range,stock_pl,c='grey',lw=3,ls='dashed',label='{} only'.format(self.sym))
        plt.plot(stock_range,strategy_pl,c='green',lw=3,label='{} with Calls (gross)'.format(self.sym))
        plt.plot(stock_range,strategy_pl_net,lw=2,label='{} with Calls (net)'.format(self.sym))
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
        plt.suptitle('Stock: {}     Current Price: ${:.2f}     Option Moneyness: {:.1%}'.format(
                                    self.sym,
                                    self.last,
                                    self.S_CALL['STRIKE_PRC']/self.last),
                     fontsize=16)
        plt.title('Options expiration: {}'.format(self.expiration),fontsize=14)
        plt.legend(loc='best',fontsize=14)

        plt.tight_layout()
        plt.savefig(img_path / '{}_covd_call_payoff'.format(self.sym))
        # plt.savefig('images/{}_covd_call_payoff'.format('BRK-B'))

    def _strat_exit(self):
        rng = 0.5
        st_rate = 0.37
        lt_rate = 0.238
        cost_basis = self.cost_basis_per_share
        # Define stock price range at expiration
        stock_range = np.arange((1 - rng) * self.last, (1 + rng) * self.last, 1)
        options_df = pd.DataFrame(
            {'stock': stock_range, 'stock_pct': (stock_range / self.last) - 1, 'options_pl': self.payoff_short_call})
        options_df.set_index('stock_pct', inplace=True)
        options_df['options_pl_net'] = options_df['options_pl'].mul(1 - st_rate)

        options_df['shrs_to_sell'] = np.where(options_df['options_pl'] < 0, (0 - options_df['options_pl']).div(
            (options_df['stock'] - cost_basis)),
                                              (options_df['options_pl_net'] / lt_rate) / (
                                                          options_df['stock'] - cost_basis))

        options_df['shrs_to_sell'][options_df['shrs_to_sell'] >= self.shares] = self.shares

        options_df['stock_gain'] = (options_df['stock'].sub(cost_basis)).mul(options_df['shrs_to_sell'])
        options_df['stock_taxes_due'] = options_df['stock_gain'].mul(lt_rate)
        options_df['pct_shares'] = options_df['shrs_to_sell'].div(self.shares)

        ### Strategic Exit Plot ###
        plt.figure(figsize=(12, 8))
        plt.style.use('fivethirtyeight')
        plt.title(f'Amount of Liquidation - {self.sym} Tax Neutral Share Sale*', fontsize=18, fontweight='bold')
        # plt.title('Amount of Portfolio Liquidation - Tax Neutral Sale*',fontsize=18,fontweight='bold')

        plt.ylabel('Tax-Neutral Liquidation %', fontsize=16, fontweight='bold')
        plt.ylim(0, options_df['pct_shares'].max() + 0.1)
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter('{x:,.0%}')
        plt.xlabel('Stock Price at Expiration', fontsize=16, fontweight='bold')
        plt.xlim(options_df['stock'].min()*1.1, options_df['stock'].max())
        plt.xticks(fontsize=14)
        plt.gca().xaxis.set_major_formatter('${x:,.0f}')

        plt.plot(options_df['stock'], options_df['pct_shares'], label='Pct. of Shares to  \nSell "Tax-Neutral"')

        plt.vlines(x=self.last, ymin=0, ymax=options_df['pct_shares'].max() + 0.1, linestyle='dashed', color='grey',
                   lw=2)
        # plt.vlines(x=325,ymin=0,ymax=options_df['pct_shares'].max()+0.1,linestyle='dashed',color='red',lw=1)
        # plt.vlines(x=233,ymin=0,ymax=options_df['pct_shares'].max()+0.1,linestyle='dashed',color='red',lw=1)

        plt.annotate('Current Stock Price',
                     xy=(self.last, options_df['pct_shares'].max() * 0.7),
                     fontsize=12,
                     rotation=90,
                     horizontalalignment='right', verticalalignment='center')

        x1 = np.arange(options_df['stock'].min()*1.1, self.S_CALL['STRIKE_PRC'] + self.S_CALL['MID'],
                       0.01)  # Below strike
        plt.fill_between(x1, y1=0, y2=options_df['pct_shares'].max() + 0.1, color='green', alpha=0.05)

        x2 = np.arange(self.S_CALL['STRIKE_PRC'] + self.S_CALL['MID'], options_df['stock'].max(),
                       0.01)  # Above Strike
        plt.fill_between(x2, y1=0, y2=options_df['pct_shares'].max() + 0.1, color='blue', alpha=0.05)

        plt.legend(loc='upper right', fontsize=16)

        plt.annotate('Stock Flat to Lower',
                     xy=(self.last * 0.65, options_df['pct_shares'].max()),
                     fontweight='bold', fontsize=16,
                     horizontalalignment='left',
                     verticalalignment='top')
        plt.annotate('Options make money, \nuse premium to pay tax bill',
                     xy=(self.last * 0.65, options_df['pct_shares'].max() * 0.925),
                     fontsize=14,
                     horizontalalignment='left',
                     verticalalignment='top')
        plt.annotate('Stock Higher',
                     xy=(self.S_CALL['STRIKE_PRC'] * 1.05, options_df['pct_shares'].max()),
                     fontweight='bold', fontsize=16,
                     horizontalalignment='left',
                     verticalalignment='top')
        plt.annotate('Options lose money, \nuse losses to offset gains',
                     xy=(self.S_CALL['STRIKE_PRC'] * 1.05, options_df['pct_shares'].max() * 0.925),
                     fontsize=14,
                     horizontalalignment='left',
                     verticalalignment='top')
        plt.tight_layout()
        plt.savefig(img_path / '{}_strategic_exit.png'.format(self.sym))
        # plt.savefig('images/{}_strategic_exit.png'.format('BRK-B'))

    def _historical_chart(self, period='2y'):

        '''
        Creates historical daily price chart of underlying stock using non-adjusted close prices from YFinance.
        Intended for use in presentation decks

        :param period: <str> The lookback period of daily prices ('1y', '3y', '5y', '10y')

        '''
        today = dt.datetime.now().date()  # get today's date
        today_str = today_str = today.strftime('%Y-%m-%d')

        if self.sym == 'BRKB':
            self.sym = 'BRK-B'
        else:
            pass
        date = dt.datetime.strptime(self.expiration, '%Y-%m-%d').date()  # convert date to datetime
        time_left = date - today  # days left
        adj_time_left = time_left / dt.timedelta(days=1)  # convert to flt
        dte = adj_time_left
        # Underlier Chart
        df = yf.download('{}'.format(self.sym), period=period, progress=False)['Close'].to_frame()
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(14, 8), linewidth=5, edgecolor='black')
        plt.plot(df['Close'], label=self.sym)
        plt.tick_params(labeltop=False, labelright=True)
        plt.hlines(y=self.S_CALL['STRIKE_PRC'], xmin=df.index[0], xmax=df.index[-1],
                   linestyle='dashed', color='red', alpha=0.5,
                   label='Short Call Strike: ${} (+{:.1%})'.format(self.S_CALL['STRIKE_PRC'],
                                                                   (self.S_CALL['STRIKE_PRC'] / self.last) - 1))
        plt.hlines(y=self.last, xmin=df.index[0], xmax=df.index[-1],
                   linestyle='dashed', color='gray', alpha=0.5,
                   label='Current Price: ${}'.format(self.last))
        plt.title(
            'Stock: {}     |     Last: ${:.2f}     |     Time to First Exp.: {:.0f} days'.format(self.sym, self.last,
                                                                                                 dte),
            fontsize=16, fontweight='bold')
        plt.legend(loc='upper left', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        plt.xlabel('Date (through {})'.format(today_str), fontsize=14, fontweight='bold')
        plt.ylabel(f'{self.sym} Price', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(img_path / '{}_{}r_price_chart_CovdCall.png'.format(self.sym, period))

    def create_cc_plots(self):
        self._cov_call_plots()
        self._strat_exit()
        self._historical_chart()

