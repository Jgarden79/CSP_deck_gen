import pandas as pd
import numpy as np
import csp_2 as csp
import datetime as dt


class CspAccount(csp.CspClient):
    def __init__(self, account_number: str, client_name: str, custodian: str, RIC: str, shares: int, sef: str,
                 cov_call: str, gap: str,
                 green_light: str, cost_basis, gap_shares: int, sef_shares: int, cov_call_shares: int, exclude_div):
        super().__init__(client_name=client_name, RIC=RIC, shares=shares, sef=sef, cov_call=cov_call, gap=gap,
                         green_light=green_light, cost_basis_per_share=cost_basis / shares, exclude_div=exclude_div)
        self.sef_trade_sheet = None
        self.cov_call_sheet = None
        self.gap_trade_sheet = None
        self.account_number = account_number
        self.gap_shares = gap_shares
        self.sef_shares = sef_shares
        self.cov_call_shares = cov_call_shares
        self.custodian = custodian
        self.get_sym()

    def _generate_gap(self):
        self.create_gap(shares_to_hedge=self.gap_shares)

    def _generate_covered_calls(self):
        self.create_cov_call(shares_to_hedge=self.cov_call_shares)

    def _generate_sef(self):
        self.create_sef(shares_to_hedge=self.sef_shares)

    def stage_gap_trade(self):
        trade_mat_date = self.gap.expiry.replace('-', '')
        strikes = [self.gap.SCALL['STRIKE_PRC'], self.gap.LPUT['STRIKE_PRC'], self.gap.SPUT['STRIKE_PRC']]
        sides = [2, 1, 2]
        ex_dt = [trade_mat_date for i in range(len(strikes))]
        act_no = [self.account_number for i in range(len(strikes))]
        cust = [self.custodian for i in range(len(strikes))]
        sec_type = ["OPT" for i in range(len(strikes))]
        qty = [int(self.gap.contracts) for i in range(len(strikes))]
        syms = [self.sym for i in range(len(strikes))]
        p_c = ["CALL", "PUT", "PUT"]
        open_close = ["O" for i in range(len(strikes))]
        type = [2 for i in range(len(strikes))]
        prices = [self.gap.SCALL['MID'], self.gap.LPUT['MID'], self.gap.SPUT['MID']]
        prices = [np.round(i, 2) for i in prices]
        amt = ['' for i in range(len(strikes))]
        tif = ['Day' for i in range(len(strikes))]
        df = pd.DataFrame({"Account": act_no, "SecurityType": sec_type, "Destination": cust, "OmnibusAccount": cust,
                           "Side": sides, "Qty": qty, "Amount": amt, "Symbol": syms, "OrdType": type, "Price": prices,
                           'StopPrice': amt, "TimeInForce": tif, "ExpirationDate": ex_dt, "ExecType": amt,
                           "PutOrCall": p_c,
                           "SellALL": amt, "StrikePrice": strikes, "MaturityDate": ex_dt, "OpenClose": open_close})
        self.gap_trade_sheet = df

    def stage_cov_call(self):
        df = pd.DataFrame({"Account": self.account_number, "SecurityType": 'OPT', "Destination": self.custodian,
                           "OmnibusAccount": self.custodian, "Side": 2, "Qty": self.cov_call.contracts, "Amount": '',
                           "Symbol": self.sym, "OrdType": 2, "Price": self.cov_call.S_CALL['MID'].round(2), 'StopPrice': '',
                           "TimeInForce": 'Day', "ExpirationDate": self.cov_call.expiration.replace('-', ''),
                           "ExecType": '',
                           "PutOrCall": 'CALL', "SellALL": '', "StrikePrice": self.cov_call.S_CALL['STRIKE_PRC'],
                           "MaturityDate": self.cov_call.expiration.replace('-', ''), "OpenClose": 'O'}, index=[0])
        self.cov_call_sheet = df

    def stage_sef_trade(self):
        ex_dt = self.sef.sef['EXPIR_DATE'].str.replace('-', '').to_list()
        strikes = self.sef.sef['STRIKE_PRC'].to_list()
        sides = [1, 2, 1, 2]
        act_no = [self.account_number for i in range(len(strikes))]
        cust = [self.custodian for i in range(len(strikes))]
        sec_type = ["OPT" for i in range(len(strikes))]
        qty = [int(self.sef.shares // 100), int(self.sef.shares // 100), self.sef.spy_contracts, self.sef.spy_contracts]
        syms = [self.sym, self.sym, self.sef.mkt_sym.upper(), self.sef.mkt_sym.upper()]
        p_c = ["PUT", "CALL", "PUT", "CALL"]
        open_close = ["O" for i in range(len(strikes))]
        type = [2 for i in range(len(strikes))]
        prices = self.sef.sef['MID'].to_list()
        prices = [np.round(i,2) for i in prices]
        amt = ['' for i in range(len(strikes))]
        tif = ['Day' for i in range(len(strikes))]
        df = pd.DataFrame({"Account": act_no, "SecurityType": sec_type, "Destination": cust, "OmnibusAccount": cust,
                           "Side": sides, "Qty": qty, "Amount": amt, "Symbol": syms, "OrdType": type, "Price": prices,
                           'StopPrice': amt, "TimeInForce": tif, "ExpirationDate": ex_dt, "ExecType": amt,
                           "PutOrCall": p_c,
                           "SellALL": amt, "StrikePrice": strikes, "MaturityDate": ex_dt, "OpenClose": open_close})

        self.sef_trade_sheet = df


    def execute_trades(self):
        df = pd.DataFrame()
        if self.gap_approved.lower() == 'y':
            self._generate_gap()
            self.stage_gap_trade()
            df = pd.concat([df, self.gap_trade_sheet], ignore_index=True)
        if self.sef_approved.lower() == 'y':
            self._generate_sef()
            self.stage_sef_trade()
            df = pd.concat([df, self.sef_trade_sheet], ignore_index=True)
        if self.cov_call_approved.lower() == 'y':
            self._generate_covered_calls()
            self.stage_cov_call()
            df = pd.concat([df, self.cov_call_sheet], ignore_index=True)
        date = dt.datetime.today().strftime('%Y%m%d')
        df.to_csv(f"trading_files/{self.account_number}_{date}_{self.sym}.csv", index=False)



