import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from scipy.integrate import odeint

class NavierStokesMarketMaker:
    def __init__(self, data, lookback=20):
        self.data = data
        self.lookback = lookback
        self.prepare_data()
        
    def prepare_data(self):
        self.data['Momentum'] = self.calculate_momentum()
        self.data['WR'] = self.calculate_williams_r()
        self.data['CMF'] = self.calculate_cmf()
        self.data['Volatility'] = self.calculate_volatility()
    
    def calculate_momentum(self):
        return self.data['Close'] - self.data['Close'].shift(5)
    
    def calculate_williams_r(self):
        high = self.data['High'].rolling(14).max()
        low = self.data['Low'].rolling(14).min()
        close = self.data['Close']
        return (high - close) / (high - low)
    
    def calculate_cmf(self):
        money_flow_multiplier = ((self.data['Close'] - self.data['Low']) - 
                                 (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        money_flow_volume = money_flow_multiplier * self.data['Volume']
        return money_flow_volume.rolling(21).sum() / self.data['Volume'].rolling(21).sum()
    
    def calculate_density(self, promoter_holding, paid_up_capital):
        return self.data['Volume'] / ((1 - promoter_holding) * paid_up_capital)
    
    def calculate_volatility(self):
        return self.data['Close'].pct_change().rolling(20).std()
    
    def navier_stokes_model(self, y, t, params):
        momentum, wr, cmf, density, volatility = y
        a, b, c, d, e = params
        
        dmdt = a * momentum + b * wr + c * cmf + d * density + e * volatility
        dwrdt = -momentum  
        dcmfdt = -momentum 
        ddendt = 0  
        dvoldt = 0  
        
        return [dmdt, dwrdt, dcmfdt, ddendt, dvoldt]
    
    def solve_navier_stokes(self, initial_conditions, params, t):
        solution = odeint(self.navier_stokes_model, initial_conditions, t, args=(params,))
        return solution[:, 0]  # return momentum value
    
    def calculate_fair_price(self, row, promoter_holding, paid_up_capital):
        momentum = row['Momentum']
        wr = row['WR']
        cmf = row['CMF']
        density = self.calculate_density(promoter_holding, paid_up_capital).loc[row.name]
        volatility = row['Volatility']
        
        # Parameters for the Navier-Stokes model
        params = (0.1, -0.2, 0.3, -0.1, 0.2)
    
        t = np.linspace(0, 1, 2) 
        initial_conditions = [momentum, wr, cmf, density, volatility]
        
        predicted_momentum = self.solve_navier_stokes(initial_conditions, params, t)
        fair_price = row['Close'] + predicted_momentum[-1]
        
        return fair_price

class BacktestMarketMaker(NavierStokesMarketMaker):
    def __init__(self, data, lookback=20, spread=0.0025, promoter_holding=0.75, paid_up_capital=1000000):
        super().__init__(data, lookback)
        self.spread = self.data['Close'].mean()*spread
        self.promoter_holding = promoter_holding
        self.paid_up_capital = paid_up_capital
        self.trades = []
        self.data['Fair_Price'] = np.nan
        self.data['Buy_Price'] = np.nan
        self.data['Sell_Price'] = np.nan
    
    def run_backtest(self):
    
        for i in tqdm(range(len(self.data))):
            row = self.data.iloc[i]
            fair_price = self.calculate_fair_price(row, self.promoter_holding, self.paid_up_capital)
            self.data.at[row.name, 'Fair_Price'] = fair_price

        self.data['VWAP_Fair_Price'] = (
            self.data['Fair_Price'] * self.data['Volume']
        ).rolling(window=5).sum() / self.data['Volume'].rolling(window=5).sum()

        self.data['Buy_Price'] = self.data['VWAP_Fair_Price'] - self.spread
        self.data['Sell_Price'] = self.data['VWAP_Fair_Price'] + self.spread
        self.data.to_csv("result_prices.csv")
        i=1
        
        while (i<=(len(self.data)-1)):  
            prev_row = self.data.iloc[i-1]
            row = self.data.iloc[i]

            buy_price = prev_row['Buy_Price']
            sell_price = prev_row['Sell_Price']

            # Long 
            if row['Low']<buy_price:
                entry_price = buy_price
                
                j=i+1
                if(j==len(self.data)):
                    exit_price = buy_price
                    self.trades.append({
                        'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                        'PnL': entry_price - exit_price, 'Fair_Price': prev_row['Fair_Price'], 
                        'Close_Price': row['Close']
                    })  
                    break
                while(j<=min(len(self.data)-1,i+15+1)):
                    exit_row = self.data.iloc[j]
                    sell_price=exit_row['Sell_Price']
                    if  sell_price< exit_row['High']:
                        exit_price = sell_price
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': exit_price - entry_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })
                        
                        i = j+1 
                        print(f"Buy Trade done at {i}")
                        break
                    elif(j==(i+15+1)):
                        j+=1
                        exit_price = sell_price
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': exit_price - entry_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })   
                        i=j+1
                        break    
                                            
                    elif(j==len(self.data)-1):
                        # Exit Irrespective of Anything
                        exit_price = sell_price
                        j+=1
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': exit_price - entry_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })
                        i=j
                        break
                    else:
                        j+=1
                     

            # Short
            elif row['High']>sell_price:
                entry_price = sell_price 
                
                j=i+1
                if(j>=len(self.data)):
                    exit_price = buy_price
                    self.trades.append({
                        'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                        'PnL': entry_price - exit_price, 'Fair_Price': prev_row['Fair_Price'], 
                        'Close_Price': row['Close']
                    })  
                    break
                while(j<=min(len(self.data)-1,i+15+1)):
                    exit_row = self.data.iloc[j]
                    buy_price=exit_row['Buy_Price']
                    if  buy_price> exit_row['Low']:
                        exit_price = buy_price
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': entry_price - exit_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })
                        
                        i = j+1 
                        print(f"Sell Trade done at {i}")
                        break
                    elif(j==(i+15+1)):
                        j+=1
                        exit_price = buy_price
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': entry_price - exit_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })   
                        i=j+1
                        break    
                                            
                    elif(j==len(self.data)-1):
                        # Exit Irrespective of Anything
                        j+=1
                        exit_price = buy_price
                        self.trades.append({
                            'Trade': 'Long', 'Entry': entry_price, 'Exit': exit_price,
                            'PnL': entry_price - exit_price, 'Fair_Price': prev_row['Fair_Price'], 
                            'Close_Price': row['Close']
                        })
                        i=j
                        break
                    else:
                        j+=1
            else:
                i+=1

    def calculate_sharpe_ratio(self, returns, risk_free_rate = 6):
        # Taking a 6% annnualized risk free rate
        rfr = risk_free_rate/ 252

        excess_returns = returns - rfr

        return np.mean(excess_returns) * math.sqrt(252) / np.std(excess_returns)

    def calculate_sortino_ratio(self, returns, rfr = 6):
        # Taking a 6% annnualized risk free rate
        risk_free_rate = rfr / 252

        downside_risk = np.std([r for r in returns if r < risk_free_rate])
        return np.mean(returns - risk_free_rate) * math.sqrt(252) / downside_risk if downside_risk != 0 else np.inf

    def calculate_max_drawdown(self, returns):

        cumulative_pnl = pd.Series(np.cumsum(returns))
        dd = (cumulative_pnl.cummax() - cumulative_pnl) / (cumulative_pnl.cummax() + self.initial_capital)
        max_drawdown = dd.max() * 100
        
        return max_drawdown

    def calculate_pnl(self, initial_capital):
        pnl = pd.DataFrame(self.trades)
        pnl.dropna(inplace=True)

        self.initial_capital = initial_capital
        capital = initial_capital
        compounded_pnl = []
        percentage_returns = []
        kelly_fractions = []

        # Calculate PnL with compounding and Kelly Criterion
        for i, trade in pnl.iterrows():
            entry_price = trade['Entry']
            trade_pnl = trade['PnL']
            
            if i >= 10:
                winning_trades = len(pnl[pnl['PnL'] > 0])
                losing_trades = len(pnl[pnl['PnL'] < 0])
                
                if winning_trades + losing_trades > 0:
                    prob_win = winning_trades / (winning_trades + losing_trades)
                    win_loss_ratio = pnl[pnl['PnL'] > 0]['PnL'].mean() / abs(pnl[pnl['PnL'] < 0]['PnL'].mean())
                    
                    kelly_fraction = prob_win - (1 - prob_win) / win_loss_ratio
                else:
                    kelly_fraction = 0
            else:
                kelly_fraction = 0
            
            kelly_fractions.append(kelly_fraction)
            position_size = capital * kelly_fraction / entry_price if kelly_fraction > 0 else 0

            trade_return = position_size * trade_pnl
            compounded_pnl.append(trade_return)
            capital += trade_return
            percentage_return = (trade_return / capital) * 100 if capital > 0 else 0
            percentage_returns.append(percentage_return)

        pnl['Compounded_PnL'] = compounded_pnl
        pnl['Percentage_Return'] = percentage_returns
        pnl['Kelly_Fraction'] = kelly_fractions

        total_pnl = sum(compounded_pnl)
        winning_trades = len(pnl[pnl['Compounded_PnL'] > 0])
        losing_trades = len(pnl[pnl['Compounded_PnL'] < 0])

        sharpe_ratio = self.calculate_sharpe_ratio(pnl['Percentage_Return'].values)
        sortino_ratio = self.calculate_sortino_ratio(pnl['Percentage_Return'].values)
        max_drawdown = self.calculate_max_drawdown(pnl['Compounded_PnL'].values)

        return total_pnl, winning_trades, losing_trades, sharpe_ratio, sortino_ratio, max_drawdown



df = pd.read_csv(r"C:\Users\choud\OneDrive\Desktop\Coding\NIFTY_cash.csv")
df.columns = df.columns.str.capitalize()
backtest = BacktestMarketMaker(df)

initial_capital = 1000000  
backtest.run_backtest()
total_pnl, winning_trades, losing_trades, sharpe_ratio, sortino_ratio, max_drawdown = backtest.calculate_pnl(initial_capital)

print(f"Total PnL: {total_pnl*100/initial_capital :.2f} %")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Sharpe Ratio: {sharpe_ratio :.3f}")
print(f"Sortino Ratio: {sortino_ratio :.3f}")
print(f"Max Drawdown: {max_drawdown :.2f} %")
