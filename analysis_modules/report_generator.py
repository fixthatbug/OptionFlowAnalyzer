# analysis_modules/report_generator.py
"""
Report generation for options flow analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import  Any
import json

def generate_detailed_txt_report(analysis_results: dict, ticker: str, 
                               source_description: str = "Options Flow Analysis") -> str:
    """Generate comprehensive text report"""
    
    report = f"""
{'=' * 80}
COMPREHENSIVE OPTIONS FLOW ANALYSIS REPORT
{'=' * 80}

TICKER: {ticker}
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SOURCE: {source_description}

{'=' * 80}
EXECUTIVE SUMMARY
{'=' * 80}
"""
    
    # Add executive summary
    report += _generate_executive_summary(analysis_results, ticker)
    
    # Data Summary
    report += f"""
{'=' * 80}
DATA SUMMARY
{'=' * 80}
"""
    data_summary = analysis_results.get('data_summary', {})
    report += f"Total Trades Analyzed: {data_summary.get('total_trades', 0):,}\n"
    report += f"Unique Options: {data_summary.get('unique_options', 0):,}\n"
    report += f"Total Volume: {data_summary.get('total_volume', 0):,} contracts\n"
    report += f"Total Notional: ${data_summary.get('total_notional', 0):,.0f}\n"
    
    if data_summary.get('time_range'):
        time_range = data_summary['time_range']
        report += f"Time Range: {time_range.get('start', 'N/A')} to {time_range.get('end', 'N/A')}\n"
    
    # Alpha Signals Section
    report += f"""
{'=' * 80}
ALPHA SIGNALS ANALYSIS
{'=' * 80}
"""
    report += _generate_alpha_signals_section(analysis_results)
    
    # Flow Analysis Section
    report += f"""
{'=' * 80}
FLOW ANALYSIS
{'=' * 80}
"""
    report += _generate_flow_analysis_section(analysis_results)
    
    # Pattern Analysis Section
    report += f"""
{'=' * 80}
PATTERN ANALYSIS
{'=' * 80}
"""
    report += _generate_pattern_analysis_section(analysis_results)
    
    # Unusual Activity Section
    report += f"""
{'=' * 80}
UNUSUAL ACTIVITY DETECTION
{'=' * 80}
"""
    report += _generate_unusual_activity_section(analysis_results)
    
    # Volatility Analysis Section
    report += f"""
{'=' * 80}
VOLATILITY ANALYSIS
{'=' * 80}
"""
    report += _generate_volatility_analysis_section(analysis_results)
    
    # Greek Analysis Section
    report += f"""
{'=' * 80}
GREEK FLOW ANALYSIS
{'=' * 80}
"""
    report += _generate_greek_analysis_section(analysis_results)
    
    # Strategy Analysis Section
    report += f"""
{'=' * 80}
STRATEGY IDENTIFICATION
{'=' * 80}
"""
    report += _generate_strategy_analysis_section(analysis_results)
    
    # Options of Interest Section
    report += f"""
{'=' * 80}
OPTIONS OF INTEREST
{'=' * 80}
"""
    report += _generate_options_of_interest_section(analysis_results)
    
    # Risk Assessment Section
    report += f"""
{'=' * 80}
RISK ASSESSMENT & ALERTS
{'=' * 80}
"""
    report += _generate_risk_assessment_section(analysis_results)
    
    # Trade Recommendations Section
    report += f"""
{'=' * 80}
TRADE RECOMMENDATIONS
{'=' * 80}
"""
    report += _generate_trade_recommendations_section(analysis_results)
    
    # Key Insights Section
    report += f"""
{'=' * 80}
KEY INSIGHTS
{'=' * 80}
"""
    report += _generate_key_insights_section(analysis_results)
    
    # Footer
    report += f"""
{'=' * 80}
END OF REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
    
    return report

def _generate_executive_summary(analysis_results: dict, ticker: str) -> str:
    """Generate executive summary"""
    
    summary = ""
    
    # Alpha signals summary
    alpha_signals = analysis_results.get('alpha_signals', [])
    urgent_signals = [s for s in alpha_signals if hasattr(s, 'urgency_score') and s.urgency_score >= 85]
    
    if alpha_signals:
        summary += f"🎯 ALPHA SIGNALS: {len(alpha_signals)} signals detected ({len(urgent_signals)} urgent)\n"
        
        if urgent_signals:
            top_signal = max(urgent_signals, key=lambda x: x.urgency_score)
            summary += f"   🚨 MOST URGENT: {top_signal.signal_type.value} in {top_signal.option_symbol}\n"
            summary += f"      Confidence: {top_signal.confidence:.0f}% | Urgency: {top_signal.urgency_score:.0f}\n"
    
    # Flow summary
    flow_analysis = analysis_results.get('flow_analysis', {})
    if flow_analysis:
        basic_metrics = flow_analysis.get('basic_metrics', {})
        directional_flow = flow_analysis.get('directional_flow', {})
        
        summary += f"\n📊 FLOW OVERVIEW:\n"
        summary += f"   Total Premium: ${basic_metrics.get('total_premium', 0):,.0f}\n"
        summary += f"   Call/Put Ratio: {directional_flow.get('call_put_volume_ratio', 1):.2f}\n"
        
        # Determine bias
        cp_ratio = directional_flow.get('call_put_volume_ratio', 1)
        if cp_ratio > 1.2:
            summary += f"   📈 BIAS: BULLISH (Call dominant)\n"
        elif cp_ratio < 0.8:
            summary += f"   📉 BIAS: BEARISH (Put dominant)\n"
        else:
            summary += f"   ➡️ BIAS: NEUTRAL\n"
    
    # Composite scores
    composite_scores = analysis_results.get('composite_scores', {})
    if composite_scores:
        summary += f"\n🎲 MARKET CONVICTION:\n"
        summary += f"   Overall Score: {composite_scores.get('overall_conviction', 0):.0f}/100\n"
        summary += f"   Institutional Activity: {composite_scores.get('institutional_activity', 0):.0f}/100\n"
    
    return summary + "\n"

def _generate_alpha_signals_section(analysis_results: dict) -> str:
    """Generate alpha signals section"""
    
    section = ""
    alpha_signals = analysis_results.get('alpha_signals', [])
    
    if not alpha_signals:
        section += "No alpha signals detected in the analyzed data.\n"
    else:
        # Add code here to format the alpha_signals into the section string
        # For example:
        section += "ALPHA SIGNALS:\n" + "-"*20 + "\n"
        for i, signal in enumerate(alpha_signals[:5], 1): # Example: show top 5
            section += f"{i}. Symbol: {getattr(signal, 'option_symbol', 'N/A')}\n" # Use getattr for safety
            section += f"   Type: {getattr(getattr(signal, 'signal_type', None), 'value', 'N/A')}\n"
            section += f"   Direction: {getattr(signal, 'direction', 'N/A')}\n"
            section += f"   Confidence: {getattr(signal, 'confidence', 0):.1f}%\n\n"
            
    return section # Make sure to return the constructed section

# This function definition should start at the same indentation level
# as the _generate_alpha_signals_section above.
def _generate_unusual_activity_section(analysis_results: dict) -> str:
    """Generate unusual activity section"""
    
    section = "" # Initialize section for this function
    unusual_activity = analysis_results.get('unusual_activity', {})
    
    if not unusual_activity:
        section += "No unusual activity detected.\n"
        return section
    
    # Volume unusual
    volume_unusual = unusual_activity.get('volume_unusual', [])
    if volume_unusual:
        section += f"UNUSUAL VOLUME: {len(volume_unusual)} options detected\n"
        for i, item in enumerate(volume_unusual[:5], 1):
            section += f"{i}. {item.get('symbol', 'N/A')}\n"
            section += f"   Current: {item.get('current_volume', 0):,} vs Avg: {item.get('average_volume', 0):,}\n"
            section += f"   Ratio: {item.get('volume_ratio', 1):.1f}x | Notional: ${item.get('notional_value', 0):,.0f}\n"
        section += "\n"
    
    # Size unusual
    size_unusual = unusual_activity.get('size_unusual', [])
    if size_unusual:
        section += f"UNUSUAL TRADE SIZES: {len(size_unusual)} large trades\n"
        for i, item in enumerate(size_unusual[:5], 1):
            section += f"{i}. {item.get('symbol', 'N/A')} - {item.get('size', 0):,} contracts\n"
            section += f"   Percentile: {item.get('percentile', 0):.1f}% | Notional: ${item.get('notional', 0):,.0f}\n"
        section += "\n"
    
    # Premium unusual
    premium_unusual = unusual_activity.get('premium_unusual', [])
    if premium_unusual:
        section += f"UNUSUAL PREMIUM: {len(premium_unusual)} high-value trades\n"
        for i, item in enumerate(premium_unusual[:3], 1):
            section += f"{i}. {item.get('symbol', 'N/A')} - ${item.get('premium', 0):,.0f}\n"
            section += f"   Contracts: {item.get('contracts', 0):,} | Side: {item.get('primary_side', 'N/A')}\n"
        section += "\n"
    
    # IV unusual
    iv_unusual = unusual_activity.get('iv_unusual', [])
    if iv_unusual:
        section += f"UNUSUAL IV ACTIVITY: {len(iv_unusual)} options with IV changes\n"
        for i, item in enumerate(iv_unusual[:3], 1):
            section += f"{i}. {item.get('symbol', 'N/A')}\n"
            section += f"   IV Change: {item.get('iv_change_pct', 0):+.1f}% ({item.get('iv_start', 0):.1%} → {item.get('iv_end', 0):.1%})\n"
        section += "\n"
    
    # Summary
    summary = unusual_activity.get('summary', {})
    if summary: # Check if summary dict itself is not empty
        section += "UNUSUAL ACTIVITY SUMMARY:\n"
        section += f"• Alert Level: {summary.get('alert_level', 'Normal')}\n"
        section += f"• Total Detections: {summary.get('total_detections', 0)}\n"
        section += f"• High Priority Items: {summary.get('high_priority_count', 0)}\n"
        
        top_symbols = summary.get('top_symbols', [])
        if top_symbols: # Check if top_symbols list is not empty
            section += "• Most Active Symbols: " + ", ".join([f"{sym}({count})" for sym, count in top_symbols[:5]]) + "\n"
        section += "\n"
    
    return section

def _generate_volatility_analysis_section(analysis_results: dict) -> str:
    """Generate volatility analysis section"""
    
    section = ""
    vol_analysis = analysis_results.get('volatility_analysis', {})
    
    if not vol_analysis:
        section += "No volatility analysis data available.\n"
        return section
    
    # Basic IV stats
    basic_iv = vol_analysis.get('basic_iv_stats', {})
    if basic_iv:
        section += "IMPLIED VOLATILITY OVERVIEW:\n"
        section += f"• Average IV: {basic_iv.get('mean_iv', 0):.1%}\n"
        section += f"• Median IV: {basic_iv.get('median_iv', 0):.1%}\n"
        section += f"• IV Range: {basic_iv.get('min_iv', 0):.1%} - {basic_iv.get('max_iv', 0):.1%}\n"
        section += f"• IV Std Dev: {basic_iv.get('std_iv', 0):.1%}\n"
        
        if basic_iv.get('volume_weighted_iv'):
            section += f"• Volume-Weighted IV: {basic_iv['volume_weighted_iv']:.1%}\n"
        section += "\n"
    
    # Vol skew
    vol_skew = vol_analysis.get('vol_skew', {})
    if vol_skew:
        put_skew = vol_skew.get('put_skew', {})
        call_skew = vol_skew.get('call_skew', {})
        
        section += "VOLATILITY SKEW:\n"
        if put_skew.get('skew_slope'):
            section += f"• Put Skew Slope: {put_skew['skew_slope']:.3f}\n"
        if call_skew.get('skew_slope'):
            section += f"• Call Skew Slope: {call_skew['skew_slope']:.3f}\n"
        
        overall_slope = vol_skew.get('skew_slope', 0)
        if overall_slope != 0:
            skew_desc = "Negative (Put bias)" if overall_slope < 0 else "Positive (Call bias)"
            section += f"• Overall Skew: {overall_slope:.3f} ({skew_desc})\n"
        section += "\n"
    
    # Term structure
    term_structure = vol_analysis.get('term_structure', {})
    if term_structure:
        section += "VOLATILITY TERM STRUCTURE:\n"
        slope = term_structure.get('term_structure_slope', 0)
        if slope != 0:
            structure_type = term_structure.get('contango_backwardation', 'flat')
            section += f"• Structure: {structure_type.title()}\n"
            section += f"• Slope: {slope:.4f}\n"
        
        opportunities = term_structure.get('term_structure_opportunities', [])
        if opportunities:
            section += "• Calendar Opportunities:\n"
            for opp in opportunities[:3]:
                section += f"  - {opp.get('near_dte', 0)}DTE vs {opp.get('far_dte', 0)}DTE: {opp.get('iv_difference', 0):.1%} diff\n"
        section += "\n"
    
    # IV rank/percentile
    iv_rank = vol_analysis.get('iv_rank_percentile', {})
    if iv_rank:
        section += "IV REGIME ANALYSIS:\n"
        section += f"• Current Regime: {iv_rank.get('iv_regime', 'unknown').replace('_', ' ').title()}\n"
        section += f"• Current Avg IV: {iv_rank.get('current_avg_iv', 0):.1%}\n"
        section += f"• IV Range (Session): {iv_rank.get('iv_range', 0):.1%}\n\n"
    
    # Trading opportunities
    vol_opportunities = vol_analysis.get('vol_trading_opportunities', {})
    if vol_opportunities:
        cheap_vol = vol_opportunities.get('cheap_vol', [])
        expensive_vol = vol_opportunities.get('expensive_vol', [])
        
        if cheap_vol:
            section += f"CHEAP VOLATILITY OPPORTUNITIES ({len(cheap_vol)}):\n"
            for opp in cheap_vol[:3]:
                section += f"• {opp.get('symbol', 'N/A')}: {opp.get('current_iv', 0):.1%} IV\n"
                section += f"  Confidence: {opp.get('confidence', 0):.0f}% | Entry: ${opp.get('vwap', 0):.2f}\n"
        
        if expensive_vol:
            section += f"EXPENSIVE VOLATILITY OPPORTUNITIES ({len(expensive_vol)}):\n"
            for opp in expensive_vol[:3]:
                section += f"• {opp.get('symbol', 'N/A')}: {opp.get('current_iv', 0):.1%} IV\n"
                section += f"  Confidence: {opp.get('confidence', 0):.0f}% | Entry: ${opp.get('vwap', 0):.2f}\n"
        section += "\n"
    
    return section

def _generate_greek_analysis_section(analysis_results: dict) -> str:
    """Generate Greek analysis section"""
    
    section = ""
    greek_analysis = analysis_results.get('greek_analysis', {})
    
    if not greek_analysis:
        section += "No Greek analysis data available.\n"
        return section
    
    # Greek exposures
    greek_exposures = greek_analysis.get('greek_exposures', {})
    if greek_exposures:
        section += "GREEK EXPOSURES:\n"
        section += f"• Net Delta: {greek_exposures.get('net_delta', 0):,.0f} shares\n"
        section += f"• Net Gamma: ${greek_exposures.get('net_gamma', 0):,.0f}\n"
        section += f"• Net Vega: ${greek_exposures.get('net_vega', 0):,.0f}\n"
        section += f"• Net Theta: ${greek_exposures.get('net_theta', 0):,.0f}/day\n\n"
    
    # Market maker positioning
    mm_position = greek_analysis.get('market_maker_position', {})
    if mm_position:
        section += "ESTIMATED MARKET MAKER POSITIONING:\n"
        section += f"• Position: {mm_position.get('estimated_position', 'Neutral')}\n"
        section += f"• Hedging Pressure: {mm_position.get('hedging_pressure', 0):,.0f}\n"
        section += f"• Gamma Imbalance: ${mm_position.get('gamma_imbalance', 0):,.0f}\n"
        
        pinning_strikes = mm_position.get('pinning_strikes', [])
        if pinning_strikes:
            section += "• Potential Pin Points:\n"
            for pin in pinning_strikes[:3]:
                section += f"  - ${pin.get('strike', 0):.0f}: {pin.get('pin_strength', 'Unknown')} gamma\n"
        section += "\n"
    
    # Hedging flows
    hedging_flows = greek_analysis.get('hedging_flows', [])
    if hedging_flows:
        section += f"HEDGING FLOW DETECTION ({len(hedging_flows)}):\n"
        for i, flow in enumerate(hedging_flows[:3], 1):
            section += f"{i}. {flow.get('type', 'Unknown')} at {flow.get('time', 'N/A')}\n"
            if flow.get('trades'):
                section += f"   Instruments: {len(flow['trades'])} legs\n"
        section += "\n"
    
    return section

def _generate_strategy_analysis_section(analysis_results: dict) -> str:
    """Generate strategy analysis section"""
    
    section = ""
    strategy_analysis = analysis_results.get('strategy_analysis', {})
    
    if not strategy_analysis:
        section += "No strategy analysis data available.\n"
        return section
    
    identified_strategies = strategy_analysis.get('identified_strategies', [])
    if identified_strategies:
        section += f"IDENTIFIED STRATEGIES ({len(identified_strategies)}):\n\n"
        
        for i, strategy in enumerate(identified_strategies[:5], 1):
            section += f"{i}. {strategy.get('strategy_name', 'Unknown Strategy')}\n"
            section += f"   Legs: {strategy.get('num_legs', 0)} | Notional: ${strategy.get('total_notional', 0):,.0f}\n"
            section += f"   Net Premium: ${strategy.get('net_debit_credit', 0):,.0f}\n"
            
            if strategy.get('strikes'):
                strikes_str = " / ".join([f"${s:.0f}" for s in strategy['strikes']])
                section += f"   Strikes: {strikes_str}\n"
            
            if strategy.get('expiration'):
                section += f"   Expiration: {strategy['expiration']}\n"
            
            # Quality metrics
            quality = strategy.get('quality_metrics', {})
            if quality:
                section += f"   Outlook: {quality.get('market_outlook', 'Neutral')}\n"
                section += f"   Complexity: {quality.get('complexity_score', 0)}/100\n"
            
            section += "\n"
    
    # Strategy summary
    strategy_summary = strategy_analysis.get('strategy_summary', {})
    if strategy_summary:
        section += "STRATEGY BREAKDOWN:\n"
        for strat_type, count in strategy_summary.items():
            section += f"• {strat_type.replace('_', ' ').title()}: {count}\n"
        section += "\n"
    
    # Institutional vs retail strategies
    institutional_strategies = strategy_analysis.get('institutional_strategies', [])
    retail_strategies = strategy_analysis.get('retail_strategies', [])
    
    section += f"STRATEGY CLASSIFICATION:\n"
    section += f"• Institutional Strategies: {len(institutional_strategies)}\n"
    section += f"• Retail Strategies: {len(retail_strategies)}\n"
    section += f"• Complex Positions (3+ legs): {len(strategy_analysis.get('complex_positions', []))}\n\n"
    
    return section

def _generate_options_of_interest_section(analysis_results: dict) -> str:
    """Generate options of interest section"""
    
    section = ""
    options_of_interest = analysis_results.get('options_of_interest_details', [])
    
    if not options_of_interest:
        section += "No specific options of interest identified.\n"
        return section
    
    section += f"TOP OPTIONS OF INTEREST ({len(options_of_interest)}):\n\n"
    
    for i, option in enumerate(options_of_interest[:15], 1):
        section += f"{i:2d}. {option.get('symbol', 'N/A')}\n"
        section += f"    Score: {option.get('score', 0):.1f} | Direction: {option.get('direction', 'NEUTRAL')}\n"
        section += f"    Confidence: {option.get('confidence', 0):.0f}% | Source: {option.get('source', 'Unknown')}\n"
        section += f"    Reason: {option.get('reason', 'N/A')}\n\n"
    
    return section

def _generate_risk_assessment_section(analysis_results: dict) -> str:
    """Generate risk assessment section"""
    
    section = ""
    risk_alerts = analysis_results.get('risk_alerts', [])
    
    if risk_alerts:
        section += f"RISK ALERTS ({len(risk_alerts)}):\n"
        for alert in risk_alerts:
            section += f"⚠️  {alert}\n"
        section += "\n"
    else:
        section += "No significant risk alerts identified.\n\n"
    
    # Market condition assessment
    composite_scores = analysis_results.get('composite_scores', {})
    if composite_scores:
        section += "MARKET CONDITION ASSESSMENT:\n"
        section += f"• Overall Conviction: {composite_scores.get('overall_conviction', 0):.0f}/100\n"
        section += f"• Volatility Expectation: {composite_scores.get('volatility_expectation', 0):.0f}/100\n"
        section += f"• Institutional Activity: {composite_scores.get('institutional_activity', 0):.0f}/100\n"
        
        # Risk level determination
        conviction = composite_scores.get('overall_conviction', 0)
        if conviction >= 80:
            risk_level = "HIGH CONVICTION - Strong directional bias"
        elif conviction >= 60:
            risk_level = "MODERATE CONVICTION - Some directional bias"
        elif conviction >= 40:
            risk_level = "LOW CONVICTION - Mixed signals"
        else:
            risk_level = "VERY LOW CONVICTION - Unclear direction"
        
        section += f"• Risk Assessment: {risk_level}\n\n"
    
    return section

def _generate_trade_recommendations_section(analysis_results: dict) -> str:
    """Generate trade recommendations section"""
    
    section = ""
    recommendations = analysis_results.get('trade_recommendations', [])
    
    if not recommendations:
        section += "No specific trade recommendations generated.\n"
        return section
    
    section += f"TRADE RECOMMENDATIONS ({len(recommendations)}):\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        section += f"{i:2d}. {rec}\n\n"
    
    # Risk management notes
    section += "RISK MANAGEMENT NOTES:\n"
    section += "• Always use appropriate position sizing\n"
    section += "• Set stop losses based on your risk tolerance\n"
    section += "• Monitor implied volatility changes\n"
    section += "• Be aware of upcoming earnings/events\n"
    section += "• Consider time decay for option positions\n\n"
    
    return section

def _generate_key_insights_section(analysis_results: dict) -> str:
    """Generate key insights section"""
    
    section = ""
    key_insights = analysis_results.get('key_insights', [])
    
    if not key_insights:
        section += "No key insights generated.\n"
        return section
    
    section += "KEY INSIGHTS:\n\n"
    
    for i, insight in enumerate(key_insights, 1):
        section += f"{i}. {insight}\n\n"
    
    return section

def generate_trade_briefing(analysis_results: dict, ticker: str) -> str:
    """Generate concise trade briefing"""
    
    briefing = f"""
TRADE BRIEFING - {ticker}
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

"""
    
    # Quick stats
    data_summary = analysis_results.get('data_summary', {})
    briefing += f"📊 SNAPSHOT:\n"
    briefing += f"Trades: {data_summary.get('total_trades', 0):,} | "
    briefing += f"Volume: {data_summary.get('total_volume', 0):,} | "
    briefing += f"Premium: ${data_summary.get('total_notional', 0):,.0f}\n\n"
    
    # Top alpha signals
    alpha_signals = analysis_results.get('alpha_signals', [])
    if alpha_signals:
        urgent_signals = [s for s in alpha_signals if hasattr(s, 'urgency_score') and s.urgency_score >= 85]
        briefing += f"🎯 ALPHA SIGNALS: {len(alpha_signals)} total ({len(urgent_signals)} urgent)\n"
        
        if urgent_signals:
            top_signal = max(urgent_signals, key=lambda x: x.urgency_score)
            briefing += f"🚨 MOST URGENT: {top_signal.trade_recommendation}\n"
        briefing += "\n"
    
    # Flow bias
    flow_analysis = analysis_results.get('flow_analysis', {})
    if flow_analysis:
        directional_flow = flow_analysis.get('directional_flow', {})
        cp_ratio = directional_flow.get('call_put_volume_ratio', 1)
        
        briefing += f"📈 FLOW BIAS: "
        if cp_ratio > 1.2:
            briefing += f"BULLISH (C/P: {cp_ratio:.2f})\n"
        elif cp_ratio < 0.8:
            briefing += f"BEARISH (C/P: {cp_ratio:.2f})\n"
        else:
            briefing += f"NEUTRAL (C/P: {cp_ratio:.2f})\n"
        briefing += "\n"
    
    # Top opportunities
    options_of_interest = analysis_results.get('options_of_interest_details', [])
    if options_of_interest:
        briefing += f"🔥 TOP OPPORTUNITIES:\n"
        for i, option in enumerate(options_of_interest[:3], 1):
            briefing += f"{i}. {option.get('symbol', 'N/A')} - {option.get('direction', 'NEUTRAL')}\n"
            briefing += f"   {option.get('reason', 'N/A')}\n"
        briefing += "\n"
    
    # Risk alerts
    risk_alerts = analysis_results.get('risk_alerts', [])
    if risk_alerts:
        briefing += f"⚠️  RISK ALERTS:\n"
        for alert in risk_alerts[:3]:
            briefing += f"• {alert}\n"
        briefing += "\n"
    
    # Quick recommendations
    recommendations = analysis_results.get('trade_recommendations', [])
    if recommendations:
        briefing += f"💡 QUICK RECS:\n"
        for rec in recommendations[:3]:
            briefing += f"• {rec}\n"
    
    briefing += f"\n{'=' * 50}\n"
    
    return briefing

def export_analysis_to_json(analysis_results: dict, filepath: str) -> bool:
    """Export analysis results to JSON"""
    
    try:
        # Convert any non-serializable objects
        serializable_results = _make_serializable(analysis_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False

def _make_serializable(obj):
    """Convert objects to JSON-serializable format"""
    
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj
    
    section += f"Total Alpha Signals: {len(alpha_signals)}\n\n"
    
    # Group by signal type
    signal_types = {}
    for signal in alpha_signals:
        sig_type = signal.signal_type.value
        if sig_type not in signal_types:
            signal_types[sig_type] = []
        signal_types[sig_type].append(signal)
    
    section += "SIGNAL TYPE BREAKDOWN:\n"
    for sig_type, signals in signal_types.items():
        section += f"• {sig_type}: {len(signals)} signals\n"
    
    section += "\nTOP ALPHA SIGNALS:\n"
    section += "-" * 50 + "\n"
    
    # Sort by composite score (confidence * urgency)
    sorted_signals = sorted(alpha_signals, 
                          key=lambda x: x.confidence * x.urgency_score, 
                          reverse=True)
    
    for i, signal in enumerate(sorted_signals[:10], 1):
        section += f"{i:2d}. {signal.signal_type.value} - {signal.direction}\n"
        section += f"    Symbol: {signal.option_symbol}\n"
        section += f"    Entry: ${signal.entry_price:.2f} | Notional: ${signal.notional_value:,.0f}\n"
        section += f"    Confidence: {signal.confidence:.0f}% | Urgency: {signal.urgency_score:.0f} | Smart Money: {signal.smart_money_score:.0f}\n"
        
        if signal.target_price:
            section += f"    Target: ${signal.target_price:.2f}"
        if signal.stop_price:
            section += f" | Stop: ${signal.stop_price:.2f}"
        section += "\n"
        
        section += f"    📋 {signal.trade_recommendation}\n\n"
    
    return section

def _generate_flow_analysis_section(analysis_results: dict) -> str:
    """Generate flow analysis section"""
    
    section = ""
    flow_analysis = analysis_results.get('flow_analysis', {})
    
    if not flow_analysis:
        section += "No flow analysis data available.\n"
        return section
    
    # Basic metrics
    basic_metrics = flow_analysis.get('basic_metrics', {})
    section += "BASIC FLOW METRICS:\n"
    section += f"• Total Trades: {basic_metrics.get('total_trades', 0):,}\n"
    section += f"• Total Contracts: {basic_metrics.get('total_contracts', 0):,}\n"
    section += f"• Total Premium: ${basic_metrics.get('total_premium', 0):,.0f}\n"
    section += f"• Average Trade Size: {basic_metrics.get('avg_trade_size', 0):.1f}\n"
    section += f"• Unique Options: {basic_metrics.get('unique_options', 0):,}\n\n"
    
    # Directional flow
    directional_flow = flow_analysis.get('directional_flow', {})
    if directional_flow:
        section += "DIRECTIONAL FLOW:\n"
        section += f"• Call Volume: {directional_flow.get('call_volume', 0):,}\n"
        section += f"• Put Volume: {directional_flow.get('put_volume', 0):,}\n"
        section += f"• Call/Put Ratio: {directional_flow.get('call_put_volume_ratio', 1):.2f}\n"
        section += f"• Net Flow: {directional_flow.get('net_volume', 0):+,} contracts\n\n"
    
    # Size analysis
    size_analysis = flow_analysis.get('size_analysis', {})
    if size_analysis:
        section += "TRADE SIZE DISTRIBUTION:\n"
        section += f"• Small Trades (≤10): {size_analysis.get('small_trades_pct', 0):.1f}%\n"
        section += f"• Medium Trades (11-50): {size_analysis.get('medium_trades_pct', 0):.1f}%\n"
        section += f"• Large Trades (51-100): {size_analysis.get('large_trades_pct', 0):.1f}%\n"
        section += f"• Block Trades (>100): {size_analysis.get('block_trades_pct', 0):.1f}%\n\n"
    
    # Significant flows
    significant_flows = flow_analysis.get('significant_flows', [])
    if significant_flows:
        section += "SIGNIFICANT FLOWS:\n"
        for i, flow in enumerate(significant_flows[:5], 1):
            section += f"{i}. {flow.get('symbol', 'N/A')} - {flow.get('type', 'N/A')}\n"
            section += f"   Volume: {flow.get('volume', 0):,} | Notional: ${flow.get('notional', 0):,.0f}\n"
            section += f"   Significance Score: {flow.get('significance_score', 0):.1f}\n\n"
    
    return section

def _generate_pattern_analysis_section(analysis_results: dict) -> str:
    """Generate pattern analysis section"""
    
    section = ""
    pattern_analysis = analysis_results.get('pattern_analysis', {})
    
    if not pattern_analysis:
        section += "No pattern analysis data available.\n"
        return section
    
    detected_patterns = pattern_analysis.get('detected_patterns', {})
    
    # Block trades
    block_trades = detected_patterns.get('block_trades', {})
    if block_trades.get('count', 0) > 0:
        section += f"BLOCK TRADES: {block_trades['count']} detected\n"
        section += f"• Total Volume: {block_trades.get('total_volume', 0):,} contracts\n"
        section += f"• Average Size: {block_trades.get('avg_size', 0):.1f} contracts\n"
        section += f"• Largest Trade: {block_trades.get('max_size', 0):,} contracts\n\n"
    
    # Sweep orders
    sweep_orders = detected_patterns.get('sweep_orders', {})
    if sweep_orders.get('count', 0) > 0:
        section += f"SWEEP ORDERS: {sweep_orders['count']} detected\n"
        sweeps = sweep_orders.get('sweeps', [])
        if sweeps:
            section += "Top Sweeps:\n"
            for i, sweep in enumerate(sweeps[:3], 1):
                section += f"{i}. {sweep.get('symbol', 'N/A')} - {sweep.get('direction', 'N/A')}\n"
                section += f"   Legs: {sweep.get('trade_count', 0)} | Urgency: {sweep.get('urgency_score', 0):.0f}\n"
        section += "\n"
    
    # Pattern summary
    pattern_summary = pattern_analysis.get('pattern_summary', {})
    if pattern_summary:
        section += "PATTERN SUMMARY:\n"
        section += f"• Total Patterns: {pattern_summary.get('total_patterns', 0)}\n"
        if pattern_summary.get('highest_confidence_pattern'):
            section += f"• Highest Confidence: {pattern_summary['highest_confidence_pattern']}\n"
        section += f"• Pattern Diversity: {pattern_summary.get('pattern_diversity', 0)}\n\n"
    
    # Institutional vs Retail indicators
    institutional = pattern_analysis.get('institutional_indicators', [])
    retail = pattern_analysis.get('retail_indicators', [])
    
    if institutional:
        section += "INSTITUTIONAL INDICATORS:\n"
        for indicator in institutional[:3]:
            section += f"• {indicator.get('indicator', 'N/A')}: {indicator.get('count', 0)} occurrences\n"
            section += f"  Confidence: {indicator.get('confidence', 0):.0f}%\n"
        section += "\n"
    
    if retail:
        section += "RETAIL INDICATORS:\n"
        for indicator in retail[:3]:
            section += f"• {indicator.get('indicator', 'N/A')}: {indicator.get('count', 0)} occurrences\n"
            section += f"  Confidence: {indicator.get('confidence', 0):.0f}%\n"
        section += "\n"
    
    return section