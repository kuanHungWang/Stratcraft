import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Metrics:
    def __init__(self, trades: pd.DataFrame, pl_history: pd.DataFrame):
        """
        Initialize the metrics with trades and profit/loss history.
        
        Args:
            trades (pd.DataFrame): DataFrame containing trade details
            must has columns:
                direction: int, 1 for long, -1 for short,
                symbol: str, quantity: int, entry_price: float, entry_date: datetime, closed: bool, win: bool, close_date: datetime, p/l: float
            pl_history (pd.DataFrame): DataFrame containing profit/loss history
            must has columns:
                date: datetime, cash: float, equity: float
        """
        self.trades = trades
        self.pl_history = pl_history

    def metrics(self, concise=False):
        """
        Calculate various performance metrics based on trade and P/L history.
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        results = {}
        
        # Ensure pl_history has datetime index
        if not isinstance(self.pl_history.index, pd.DatetimeIndex):
            self.pl_history['date'] = pd.to_datetime(self.pl_history['date'])
            self.pl_history.set_index('date', inplace=True)
        
        # Calculate daily returns
        daily_returns = self.pl_history['equity'].pct_change().dropna()
        
        # Initial and final equity
        initial_equity = self.pl_history['equity'].iloc[0]
        final_equity = self.pl_history['equity'].iloc[-1]
        
        # Total return
        total_return = (final_equity / initial_equity) - 1
        results['Cumulative returns'] = total_return
        
        # Trading period in years
        start_date = self.pl_history.index[0]
        end_date = self.pl_history.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        results['Trading period years'] = years
        
        # Annual return
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
            results['Annual return'] = annual_return
        else:
            results['Annual return'] = 0
        
        # Win rate
        if len(self.trades) > 0:
            win_rate = self.trades['win'].sum() / len(self.trades)
            results['Win rate'] = win_rate
        else:
            results['Win rate'] = 0
        
        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        # results['cumulative_returns'] = cumulative_returns
        
        # Annual volatility
        if len(daily_returns) > 1:
            annual_volatility = daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
            results['Annual volatility'] = annual_volatility
        else:
            results['Annual volatility'] = 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if results['Annual volatility'] > 0:
            sharpe_ratio = results['Annual return'] / results['Annual volatility']
            results['Sharpe ratio'] = sharpe_ratio
        else:
            results['Sharpe ratio'] = 0
        
        # Max drawdown
        if not cumulative_returns.empty:
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
            max_drawdown = drawdown.min()
            results['Max drawdown'] = max_drawdown
        else:
            results['Max drawdown'] = 0
        
        # Calmar ratio
        if results['Max drawdown'] != 0:
            calmar_ratio = results['Annual return'] / abs(results['Max drawdown'])
            results['Calmar ratio'] = calmar_ratio
        else:
            results['Calmar ratio'] = 0
        
        # Stability (R-squared of linear regression of cumulative returns)
        if len(cumulative_returns) > 1:
            x = np.arange(len(cumulative_returns))
            y = cumulative_returns.values
            slope, intercept = np.polyfit(x, y, 1)
            line = slope * x + intercept
            r_squared = 1 - (np.sum((y - line) ** 2) / np.sum((y - np.mean(y)) ** 2))
            results['Stability'] = r_squared
        else:
            results['Stability'] = 0
        
        # Sortino ratio (downside deviation)
        if len(daily_returns) > 1:
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                if downside_deviation > 0:
                    sortino_ratio = results['Annual return'] / downside_deviation
                    results['Sortino ratio'] = sortino_ratio
                else:
                    results['Sortino ratio'] = 0
            else:
                results['Sortino ratio'] = 0
        else:
            results['Sortino ratio'] = 0
        
        # Omega ratio
        if len(daily_returns) > 1:
            threshold = 0  # Can be adjusted
            upside = daily_returns[daily_returns > threshold]
            downside = daily_returns[daily_returns < threshold]
            
            if len(downside) > 0 and downside.sum() != 0:
                omega_ratio = upside.sum() / abs(downside.sum())
                results['Omega ratio'] = omega_ratio
            else:
                results['Omega ratio'] = float('inf') if len(upside) > 0 else 0
        else:
            results['Omega ratio'] = 0
        
        # Skew and Kurtosis
        if len(daily_returns) > 1:
            results['Skew'] = daily_returns.skew()
            results['Kurtosis'] = daily_returns.kurtosis()
        else:
            results['Skew'] = 0
            results['Kurtosis'] = 0
        
        # Tail ratio
        if len(daily_returns) > 1:
            quantile05 = daily_returns.quantile(0.05)
            if quantile05 == 0:
                results['Tail ratio'] = 'N/A'
            else:
                tail_ratio = abs(daily_returns.quantile(0.95)) / abs(quantile05)
                results['Tail ratio'] = tail_ratio
        else:
            results['Tail ratio'] = 0
        
        # Average profit per trade
        if len(self.trades) > 0:
            avg_profit = self.trades['p/l'].mean()
            results['Avg profit per trade'] = avg_profit
        else:
            results['Avg profit per trade'] = 0
        
        # Average win and loss
        winning_trades = self.trades[self.trades['win'] == True]
        losing_trades = self.trades[self.trades['win'] == False]
        
        if len(winning_trades) > 0:
            avg_win = winning_trades['p/l'].mean()
            results['Avg win trade p/l'] = avg_win
        else:
            results['Avg win trade p/l'] = 0
            
        if len(losing_trades) > 0:
            avg_loss = losing_trades['p/l'].mean()
            results['Avg loss trade p/l'] = avg_loss
        else:
            results['Avg loss trade p/l'] = 0
        
        # Profit factor
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            total_win = winning_trades['p/l'].sum()
            total_loss = abs(losing_trades['p/l'].sum())
            
            if total_loss > 0:
                profit_factor = total_win / total_loss
                results['Profit factor'] = profit_factor
            else:
                results['Profit factor'] = float('inf')
        else:
            results['Profit factor'] = 0
        
        # number of trades
        results['Number of trades'] = len(self.trades)
        concise_items = ['Cumulative returns', 'Annual return', 'Win rate', 'Annual volatility', 'Sharpe ratio', 'Max drawdown','Number of trades']
        if concise:
            results = {k: v for k, v in results.items() if k in concise_items}
        return results

    @classmethod
    def pretty_print(cls, results):
        """
        Print the metrics in a human-readable format.
        
        Args:
            results (dict): Dictionary of metrics
        """
        percentage_keys = ['Total return','Annual return','Win rate','Annual volatility']  # print percentage format like 2.34%
        
        # Find the longest key for alignment
        max_key_length = max(len(key) for key in results.keys())
        
        for key, value in results.items():
            # Add padding to align all values
            padding = ' ' * (max_key_length - len(key) + 2)
            
            if isinstance(value, float):
                if key in percentage_keys:
                    # Format as percentage for percentage keys
                    print(f"{key}:{padding}{value*100:.3f}%")
                else:
                    print(f"{key}:{padding}{value:.3f}")
            else:
                print(f"{key}:{padding}{value}")

    def chart_history(self, additional_chart_data=None):
        """
        Generate an interactive visualization of trading performance history.
        
        Creates a multi-panel chart with the equity curve and optional additional data.
        The main panel displays the equity curve with triangular markers indicating trades
        (upward triangles for long positions, downward for short positions).
        
        Additional panels can display various data types such as price charts,
        technical indicators, cash/equity ratios, etc., with automatic dual y-axis
        scaling for data with different ranges.
        
        Interactive features include tooltips on hover showing detailed information
        about trades and data points.
        
        Parameters
        ----------
        additional_chart_data : dict, optional
            Dictionary of additional data to display in separate panels.
            Keys become panel titles (converted from snake_case to Title Case).
            Values can be dictionaries, DataFrames, or Series.
            Example: {'price_data': price_dict, 'indicators': indicators_dict}
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure object that can be:
            - Displayed directly in notebook: fig.show()
            - Exported to HTML: fig.write_html("chart.html")
            - Converted to JSON: fig.to_json()
            - Embedded in web apps: plotly.offline.plot(fig, include_plotlyjs=True, output_type='div')
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Determine how many subplots we need based on additional_chart_data
        num_rows = 1  # At minimum, we'll have the equity curve
        subplot_titles = ["Equity Curve"]
        
        # Create a list to store the order of subplots
        subplot_order = []
        
        if additional_chart_data is not None:
            # Use the keys of additional_chart_data to determine subplot titles and order
            for key in additional_chart_data.keys():
                num_rows += 1
                # Convert key from snake_case to Title Case for subplot title
                title = " ".join(word.capitalize() for word in key.split('_'))
                subplot_titles.append(title)
                subplot_order.append(key)
        
        # Create subplot structure
        fig = make_subplots(rows=num_rows, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.15,  # Increase spacing between subplots
                           subplot_titles=subplot_titles,
                           specs=[[{"secondary_y": True}] for _ in range(num_rows)])  # Enable secondary y-axis for each subplot
        
        # Ensure pl_history has datetime index
        pl_history = self.pl_history.copy()
        if not isinstance(pl_history.index, pd.DatetimeIndex):
            pl_history['date'] = pd.to_datetime(pl_history['date'])
            pl_history.set_index('date', inplace=True)
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=pl_history.index,
                y=pl_history['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Equity: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Process trades to add markers for entry points
        if not self.trades.empty:
            # Convert entry_date to datetime if it's not already
            trades = self.trades.copy()
            if not pd.api.types.is_datetime64_any_dtype(trades['entry_date']):
                trades['entry_date'] = pd.to_datetime(trades['entry_date'])
            
            # Get equity values at trade entry dates
            entry_equity_values = []
            for entry_date in trades['entry_date']:
                # Find the closest equity value for this date
                closest_date = pl_history.index[pl_history.index.get_indexer([entry_date], method='nearest')[0]]
                entry_equity_values.append(pl_history.loc[closest_date, 'equity'])
            
            # Create hover text for trade markers
            hover_texts = []
            for i, row in trades.iterrows():
                hover_text = f"Symbol: {row['symbol']}<br>"
                hover_text += f"Direction: {'Long' if row['direction'] == 1 else 'Short'}<br>"
                hover_text += f"Entry Date: {row['entry_date']}<br>"
                hover_text += f"Entry Price: ${row['entry_price']:.2f}<br>"
                hover_text += f"Quantity: {row['quantity']}<br>"
                
                if row['closed']:
                    hover_text += f"Close Date: {row['close_date']}<br>"
                    hover_text += f"P/L: ${row['p/l']:.2f}<br>"
                    hover_text += f"Result: {'Win' if row['win'] else 'Loss'}"
                else:
                    hover_text += "Status: Open"
                
                hover_texts.append(hover_text)
            
            # Add long trade markers (upward triangles)
            long_trades = trades[trades['direction'] == 1]
            if not long_trades.empty:
                long_indices = long_trades.index
                long_dates = long_trades['entry_date']
                long_equity_values = [entry_equity_values[i] for i in long_indices]
                long_hover_texts = [hover_texts[i] for i in long_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=long_dates,
                        y=long_equity_values,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        ),
                        name='Long Entries',
                        hoverinfo='text',
                        hovertext=long_hover_texts,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add short trade markers (downward triangles)
            short_trades = trades[trades['direction'] == -1]
            if not short_trades.empty:
                short_indices = short_trades.index
                short_dates = short_trades['entry_date']
                short_equity_values = [entry_equity_values[i] for i in short_indices]
                short_hover_texts = [hover_texts[i] for i in short_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=short_dates,
                        y=short_equity_values,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=1, color='darkred')
                        ),
                        name='Short Entries',
                        hoverinfo='text',
                        hovertext=short_hover_texts,
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add additional charts if provided
        if additional_chart_data is not None:
            # Process each key in the order they were provided
            for i, key in enumerate(subplot_order):
                current_row = i + 2  # Start from the second row for additional charts
                
                if key == 'price_data':
                    # Add price data for each symbol
                    for j, (symbol, data) in enumerate(additional_chart_data[key].items()):
                        # Alternate between primary and secondary y-axis based on data scale
                        use_secondary_y = j % 2 == 1  # First item on primary, second on secondary, etc.
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['close'],
                                mode='lines',
                                name=f'{symbol}',
                                line=dict(width=1.5),
                                hovertemplate=f'{symbol}: $%{{y:.2f}}<extra></extra>'
                            ),
                            row=current_row, col=1, secondary_y=use_secondary_y
                        )
                    
                    # Set y-axis titles
                    fig.update_yaxes(title_text="Price ($) - Primary", row=current_row, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="Price ($) - Secondary", row=current_row, col=1, secondary_y=True)
                
                elif key == 'indicators':
                    # Group indicators by scale range to determine which y-axis to use
                    indicators_by_scale = {'primary': [], 'secondary': []}
                    
                    # First pass: determine the scale ranges
                    min_values = {}
                    max_values = {}
                    for name, series in additional_chart_data[key].items():
                        min_values[name] = series.min()
                        max_values[name] = series.max()
                    
                    # Simple heuristic: sort by range magnitude
                    ranges = {name: max_values[name] - min_values[name] for name in additional_chart_data[key].keys()}
                    sorted_indicators = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
                    
                    # Assign to primary or secondary axis
                    for j, (name, _) in enumerate(sorted_indicators):
                        if j % 2 == 0:
                            indicators_by_scale['primary'].append(name)
                        else:
                            indicators_by_scale['secondary'].append(name)
                    
                    # Add indicators to the appropriate y-axis
                    for name, series in additional_chart_data[key].items():
                        use_secondary_y = name in indicators_by_scale['secondary']
                        
                        fig.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode='lines',
                                name=name,
                                line=dict(width=1.5),
                                hovertemplate=f'{name}: %{{y:.4f}}<extra></extra>'
                            ),
                            row=current_row, col=1, secondary_y=use_secondary_y
                        )
                    
                    # Set y-axis titles
                    if indicators_by_scale['primary']:
                        primary_names = ", ".join(indicators_by_scale['primary'])
                        fig.update_yaxes(title_text=f"{primary_names}", row=current_row, col=1, secondary_y=False)
                    
                    if indicators_by_scale['secondary']:
                        secondary_names = ", ".join(indicators_by_scale['secondary'])
                        fig.update_yaxes(title_text=f"{secondary_names}", row=current_row, col=1, secondary_y=True)
                
                elif key == 'cash_ratio':
                    # Add cash/equity ratio
                    cash_ratio = additional_chart_data[key]
                    fig.add_trace(
                        go.Scatter(
                            x=cash_ratio.index,
                            y=cash_ratio.values,
                            mode='lines',
                            name='Cash/Equity Ratio',
                            line=dict(color='purple', width=1.5),
                            hovertemplate='Cash/Equity: %{y:.2f}<extra></extra>'
                        ),
                        row=current_row, col=1, secondary_y=False
                    )
                    # Set y-axis title
                    fig.update_yaxes(title_text="Ratio", row=current_row, col=1, secondary_y=False)
                
                else:
                    # Handle any other type of data
                    data = additional_chart_data[key]
                    
                    # Check if it's a dictionary
                    if isinstance(data, dict):
                        # Group items by scale range to determine which y-axis to use
                        items_by_scale = {'primary': [], 'secondary': []}
                        
                        # First pass: determine the scale ranges
                        min_values = {}
                        max_values = {}
                        for name, series in data.items():
                            if isinstance(series, (pd.Series, np.ndarray)):
                                min_values[name] = np.min(series)
                                max_values[name] = np.max(series)
                        
                        # Simple heuristic: sort by range magnitude
                        ranges = {name: max_values[name] - min_values[name] for name in min_values.keys()}
                        sorted_items = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
                        
                        # Assign to primary or secondary axis
                        for i, (name, _) in enumerate(sorted_items):
                            if i % 2 == 0:
                                items_by_scale['primary'].append(name)
                            else:
                                items_by_scale['secondary'].append(name)
                        
                        # Plot each item in the dictionary
                        for name, series in data.items():
                            if isinstance(series, (pd.Series, np.ndarray)):
                                use_secondary_y = name in items_by_scale['secondary']
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=pd.Series(series).index if isinstance(series, pd.Series) else np.arange(len(series)),
                                        y=series.values if isinstance(series, pd.Series) else series,
                                        mode='lines',
                                        name=name,
                                        line=dict(width=1.5),
                                        hovertemplate=f'{name}: %{{y:.4f}}<extra></extra>'
                                    ),
                                    row=current_row, col=1, secondary_y=use_secondary_y
                                )
                        
                        # Set y-axis titles
                        if items_by_scale['primary']:
                            primary_names = ", ".join(items_by_scale['primary'])
                            fig.update_yaxes(title_text=f"{primary_names}", row=current_row, col=1, secondary_y=False)
                        
                        if items_by_scale['secondary']:
                            secondary_names = ", ".join(items_by_scale['secondary'])
                            fig.update_yaxes(title_text=f"{secondary_names}", row=current_row, col=1, secondary_y=True)
                    
                    # Check if it's a Series or DataFrame
                    elif isinstance(data, (pd.Series, pd.DataFrame)):
                        if isinstance(data, pd.DataFrame):
                            # Group columns by scale range
                            columns_by_scale = {'primary': [], 'secondary': []}
                            
                            # Determine scale ranges
                            min_values = data.min()
                            max_values = data.max()
                            ranges = max_values - min_values
                            
                            # Sort columns by range magnitude
                            sorted_columns = ranges.sort_values(ascending=False).index
                            
                            # Assign to primary or secondary axis
                            for i, column in enumerate(sorted_columns):
                                if i % 2 == 0:
                                    columns_by_scale['primary'].append(column)
                                else:
                                    columns_by_scale['secondary'].append(column)
                            
                            # Plot each column
                            for column in data.columns:
                                use_secondary_y = column in columns_by_scale['secondary']
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data.index,
                                        y=data[column],
                                        mode='lines',
                                        name=column,
                                        line=dict(width=1.5),
                                        hovertemplate=f'{column}: %{{y:.4f}}<extra></extra>'
                                    ),
                                    row=current_row, col=1, secondary_y=use_secondary_y
                                )
                            
                            # Set y-axis titles
                            if columns_by_scale['primary']:
                                primary_names = ", ".join(columns_by_scale['primary'])
                                fig.update_yaxes(title_text=f"{primary_names}", row=current_row, col=1, secondary_y=False)
                            
                            if columns_by_scale['secondary']:
                                secondary_names = ", ".join(columns_by_scale['secondary'])
                                fig.update_yaxes(title_text=f"{secondary_names}", row=current_row, col=1, secondary_y=True)
                        else:
                            # Plot the Series
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=data.values,
                                    mode='lines',
                                    name=key,
                                    line=dict(width=1.5),
                                    hovertemplate=f'{key}: %{{y:.4f}}<extra></extra>'
                                ),
                                row=current_row, col=1, secondary_y=False
                            )
                            
                            # Set y-axis title
                            fig.update_yaxes(title_text=key, row=current_row, col=1, secondary_y=False)
        
        # Update layout for better visualization
        fig.update_layout(
            title='Trading Performance History',
            height=300 * num_rows,  # Increase height per subplot
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',  # Show hover information for all traces at the same x-coordinate
            margin=dict(l=50, r=50, t=80, b=50),
            autosize=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Update y-axis title for equity curve
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        
        return fig

    def pie_chart_number_trades(self):
        """
        pie chart showing distribution of trades by symbol.
        Returns:
            chart object
        """
        pass

    def pie_chart_profit_contribution(self):
        """
        pie chart showing distribution of profit contribution by symbol(for win trade only)
        Returns:
            chart object
        """
        pass
