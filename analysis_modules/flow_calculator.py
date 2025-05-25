# flow_calculator.py
# Contains functions for calculating core flow metrics.

import pandas as pd
import numpy as np

def hf_calculate_ofi_agg(trades_in_window: pd.DataFrame) -> pd.Series:
    """
    Calculates Order Flow Imbalance (OFI) and Aggressor Quantities for a given window of trades.
    This function is used by both High-Frequency analysis and binned OFI summaries.
    """
    if trades_in_window.empty:
        return pd.Series({
            'NetDollarOFI': 0.0, 'BuyDollarVolume': 0.0, 'SellDollarVolume': 0.0,
            'NetAggressorQty': 0.0, 'BuyAggressorQty': 0.0, 'SellAggressorQty': 0.0,
            'TotalTrades': 0, 'TotalQuantity': 0.0
        })

    # Ensure relevant columns are numeric
    trades_in_window['TradeQuantity'] = pd.to_numeric(trades_in_window['TradeQuantity'], errors='coerce').fillna(0)
    trades_in_window['Trade_Price'] = pd.to_numeric(trades_in_window['Trade_Price'], errors='coerce').fillna(0)

    buy_mask = trades_in_window['Aggressor'] == 'Buyer (At Ask)'
    sell_mask = trades_in_window['Aggressor'] == 'Seller (At Bid)'

    buy_dollar_volume = (trades_in_window.loc[buy_mask, 'TradeQuantity'] * trades_in_window.loc[buy_mask, 'Trade_Price']).sum()
    sell_dollar_volume = (trades_in_window.loc[sell_mask, 'TradeQuantity'] * trades_in_window.loc[sell_mask, 'Trade_Price']).sum()
    net_dollar_ofi = buy_dollar_volume - sell_dollar_volume

    buy_aggressor_qty = trades_in_window.loc[buy_mask, 'TradeQuantity'].sum()
    sell_aggressor_qty = trades_in_window.loc[sell_mask, 'TradeQuantity'].sum()
    net_aggressor_qty = buy_aggressor_qty - sell_aggressor_qty

    return pd.Series({
        'NetDollarOFI': net_dollar_ofi, 
        'BuyDollarVolume': buy_dollar_volume, 
        'SellDollarVolume': sell_dollar_volume,
        'NetAggressorQty': net_aggressor_qty, 
        'BuyAggressorQty': buy_aggressor_qty, 
        'SellAggressorQty': sell_aggressor_qty,
        'TotalTrades': len(trades_in_window), 
        'TotalQuantity': trades_in_window['TradeQuantity'].sum()
    })

# Add other core flow calculation functions here in the future,
# e.g., detailed OFI binned summary generation if it becomes more complex.
# For now, the main binned summary logic can remain in the orchestrator
# if it's straightforward resampling + call to hf_calculate_ofi_agg.
