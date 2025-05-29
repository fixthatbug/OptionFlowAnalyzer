# analysis_engine.py
"""
Core analysis engine that orchestrates all analysis modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from analysis_modules.flow_calculator import FlowCalculator
from analysis_modules.trade_pattern_detector import TradePatternDetector
from analysis_modules.volatility_analyzer import VolatilityAnalyzer
from analysis_modules.unusual_activity_detector import UnusualActivityDetector
from analysis_modules.greek_flow_analyzer import GreekFlowAnalyzer
from analysis_modules.strategy_identifier import StrategyIdentifier
from analysis_modules.report_generator import generate_detailed_txt_report, generate_trade_briefing
from alpha_extractor import AlphaExtractor, generate_signal_report
from market_microstructure_analyzer import MarketMicrostructureAnalyzer

class AnalysisEngine:
    """Central analysis engine coordinating all analysis modules"""
    
    def __init__(self, ticker: str, log_callback: Optional[Callable] = None):
        self.ticker = ticker
        self.log = log_callback or self._default_log
        
        # Initialize analysis modules
        self.flow_calculator = FlowCalculator()
        self.pattern_detector = TradePatternDetector()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.unusual_activity_detector = UnusualActivityDetector()
        self.greek_analyzer = GreekFlowAnalyzer()
        self.strategy_identifier = StrategyIdentifier()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer(ticker)
        
        # Alpha extraction
        self.alpha_extractor = None
        
        # Analysis cache
        self.analysis_cache = {}
        self.last_analysis_time = None
        
    def _default_log(self, message: str, ticker: Optional[str] = None, is_error: bool = False):
        """Default logging function"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_holistic_analysis(self, df: pd.DataFrame, ticker: str = '', log_func: Optional[Callable] = None) -> dict:
        """
        Main analysis function that orchestrates all analysis modules
        """
        if ticker:
            self.ticker = ticker
        if log_func:
            self.log = log_func
            
        self.log(f"Starting holistic analysis for {self.ticker} with {len(df)} trades")
        
        if df.empty:
            return {'error': 'No data provided for analysis'}
        
        # Initialize results container
        analysis_results = {
            'ticker': self.ticker,
            'analysis_timestamp': datetime.now(),
            'total_trades': len(df),
            'data_summary': self._generate_data_summary(df),
            'flow_analysis': {},
            'pattern_analysis': {},
            'volatility_analysis': {},
            'unusual_activity': {},
            'greek_analysis': {},
            'strategy_analysis': {},
            'microstructure_analysis': {},
            'alpha_signals': [],
            'alpha_report': '',
            'options_of_interest': [],
            'options_of_interest_details': [],
            'key_insights': [],
            'risk_alerts': [],
            'trade_recommendations': []
        }
        
        try:
            # 1. Basic Flow Analysis
            self.log("Running flow analysis...")
            analysis_results['flow_analysis'] = self._run_flow_analysis(df)
            
            # 2. Trade Pattern Detection
            self.log("Analyzing trade patterns...")
            analysis_results['pattern_analysis'] = self._run_pattern_analysis(df)
            
            # 3. Volatility Analysis
            self.log("Performing volatility analysis...")
            analysis_results['volatility_analysis'] = self._run_volatility_analysis(df)
            
            # 4. Unusual Activity Detection
            self.log("Detecting unusual activity...")
            analysis_results['unusual_activity'] = self._run_unusual_activity_analysis(df)
            
            # 5. Greek Flow Analysis
            self.log("Analyzing Greeks flow...")
            analysis_results['greek_analysis'] = self._run_greek_analysis(df)
            
            # 6. Strategy Identification
            self.log("Identifying complex strategies...")
            analysis_results['strategy_analysis'] = self._run_strategy_analysis(df)
            
            # 7. Market Microstructure Analysis
            self.log("Analyzing market microstructure...")
            analysis_results['microstructure_analysis'] = self._run_microstructure_analysis(df)
            
            # 8. Alpha Extraction
            self.log("Extracting alpha signals...")
            analysis_results['alpha_signals'], analysis_results['alpha_report'] = self._run_alpha_extraction(df)
            
            # 9. Synthesize Results
            self.log("Synthesizing analysis results...")
            analysis_results = self._synthesize_results(analysis_results, df)
            
            # 10. Generate Key Insights
            analysis_results['key_insights'] = self._generate_key_insights(analysis_results)
            
            # 11. Risk Assessment
            analysis_results['risk_alerts'] = self._assess_risks(analysis_results, df)
            
            # 12. Trade Recommendations
            analysis_results['trade_recommendations'] = self._generate_recommendations(analysis_results)
            
            self.log(f"Analysis complete for {self.ticker}")
            
        except Exception as e:
            self.log(f"Error in analysis: {e}", is_error=True)
            analysis_results['error'] = str(e)
        
        # Cache results
        self.analysis_cache[self.ticker] = analysis_results
        self.last_analysis_time = datetime.now()
        
        return analysis_results
    
    def _generate_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics of the dataset"""
        summary = {
            'total_trades': len(df),
            'unique_options': df['StandardOptionSymbol'].nunique() if 'StandardOptionSymbol' in df.columns else 0,
            'total_volume': df['TradeQuantity'].sum() if 'TradeQuantity' in df.columns else 0,
            'total_notional': df['NotionalValue'].sum() if 'NotionalValue' in df.columns else 0,
            'time_range': {
                'start': df['Time_dt'].min() if 'Time_dt' in df.columns else None,
                'end': df['Time_dt'].max() if 'Time_dt' in df.columns else None
            },
            'price_range': {
                'min': df['Trade_Price'].min() if 'Trade_Price' in df.columns else None,
                'max': df['Trade_Price'].max() if 'Trade_Price' in df.columns else None
            },
            'exchanges': df['Exchange'].unique().tolist() if 'Exchange' in df.columns else [],
            'option_types': df['Option_Type_calc'].value_counts().to_dict() if 'Option_Type_calc' in df.columns else {}
        }
        
        return summary
    
    def _run_flow_analysis(self, df: pd.DataFrame) -> dict:
        """Run comprehensive flow analysis"""
        try:
            flow_results = self.flow_calculator.calculate_comprehensive_flow_metrics(df)
            
            # Add flow direction analysis
            flow_results['directional_bias'] = self._analyze_directional_bias(df)
            flow_results['flow_concentration'] = self._analyze_flow_concentration(df)
            flow_results['momentum_indicators'] = self._calculate_momentum_indicators(df)
            
            return flow_results
            
        except Exception as e:
            self.log(f"Error in flow analysis: {e}", is_error=True)
            return {'error': str(e)}
    
    def _run_pattern_analysis(self, df: pd.DataFrame) -> dict:
        """Run trade pattern detection"""
        try:
            patterns = self.pattern_detector.detect_all_patterns(df)
            
            # Enhance with timing analysis
            patterns['timing_patterns'] = self._analyze_timing_patterns(df)
            patterns['size_patterns'] = self._analyze_size_patterns(df)
            
            return patterns
            
        except Exception as e:
            self.log(f"Error in pattern analysis: {e}", is_error=True)
            return {'error': str(e)}

    def _analyze_timing_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze timing patterns in trade data"""
        if df.empty or 'Time_dt' not in df.columns:
            return {}
        df_sorted = df.sort_values('Time_dt')
        time_diffs = df_sorted['Time_dt'].diff().dt.total_seconds().dropna()
        if time_diffs.empty:
            return {}
        avg_time_between_trades = time_diffs.mean()
        min_time_between_trades = time_diffs.min()
        max_time_between_trades = time_diffs.max()
        bursty_periods = (time_diffs < avg_time_between_trades * 0.5).sum()
        return {
            'avg_time_between_trades': avg_time_between_trades,
            'min_time_between_trades': min_time_between_trades,
            'max_time_between_trades': max_time_between_trades,
            'bursty_periods_count': int(bursty_periods)
        }

    def _analyze_size_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze trade size patterns in the data"""
        if df.empty or 'TradeQuantity' not in df.columns:
            return {}
        size_stats = df['TradeQuantity'].describe()
        large_trade_threshold = size_stats['75%'] + 1.5 * (size_stats['75%'] - size_stats['25%'])
        large_trades = df[df['TradeQuantity'] > large_trade_threshold]
        return {
            'mean_trade_size': size_stats['mean'],
            'median_trade_size': size_stats['50%'],
            'max_trade_size': size_stats['max'],
            'min_trade_size': size_stats['min'],
            'large_trade_count': len(large_trades),
            'large_trade_threshold': large_trade_threshold
        }
    
    def _run_volatility_analysis(self, df: pd.DataFrame) -> dict:
        """Run volatility analysis"""
        try:
            vol_analysis = self.volatility_analyzer.analyze_volatility_dynamics(df)
            
            # Add term structure analysis if multiple expirations
            if 'Expiration_Date_calc' in df.columns:
                vol_analysis['term_structure'] = self._analyze_vol_term_structure(df)
            
            return vol_analysis
            
        except Exception as e:
            self.log(f"Error in volatility analysis: {e}", is_error=True)
            return {'error': str(e)}

    def _analyze_vol_term_structure(self, df: pd.DataFrame) -> dict:
        """Analyze implied volatility term structure across expirations"""
        if df.empty or 'Expiration_Date_calc' not in df.columns or 'ImpliedVolatility' not in df.columns:
            return {}
        term_structure = df.groupby('Expiration_Date_calc')['ImpliedVolatility'].mean().sort_index()
        term_slope = None
        if len(term_structure) > 1:
            expiries = np.arange(len(term_structure))
            vols = term_structure.values
            # Simple linear regression for slope
            slope = np.polyfit(expiries, np.array(vols), 1)[0]
            term_slope = slope
        return {
            'term_structure': term_structure.to_dict(),
            'term_slope': term_slope
        }
    
    def _run_unusual_activity_analysis(self, df: pd.DataFrame) -> dict:
        """Run unusual activity detection"""
        try:
            unusual = self.unusual_activity_detector.detect_unusual_activity(df)
            
            # Add statistical significance testing
            unusual['statistical_significance'] = self._test_statistical_significance(df)
            
            return unusual
            
        except Exception as e:
            self.log(f"Error in unusual activity analysis: {e}", is_error=True)
            return {'error': str(e)}

    def _test_statistical_significance(self, df: pd.DataFrame) -> dict:
        """
        Dummy statistical significance test for unusual activity.
        Replace with real statistical tests as needed.
        """
        if df.empty:
            return {'significant': False, 'p_value': 1.0}
        # Example: always return not significant
        return {'significant': False, 'p_value': 1.0}
    
    def _run_greek_analysis(self, df: pd.DataFrame) -> dict:
        """Run Greek flow analysis"""
        try:
            greek_analysis = self.greek_analyzer.analyze_greek_exposures(df)
            
            # Add hedging analysis
            greek_analysis['hedging_analysis'] = self._analyze_hedging_activity(df)
            
            return greek_analysis
            
        except Exception as e:
            self.log(f"Error in Greek analysis: {e}", is_error=True)
            return {'error': str(e)}

    def _analyze_hedging_activity(self, df: pd.DataFrame) -> dict:
        """Analyze potential hedging activity in the trade data"""
        if df.empty or 'Delta' not in df.columns:
            return {'hedging_detected': False, 'summary': 'Insufficient data for hedging analysis.'}
        # Example: Detect large delta-neutral trades as potential hedges
        delta_sum = df['Delta'].sum()
        notional = df['NotionalValue'].sum() if 'NotionalValue' in df.columns else 0
        hedging_detected = abs(delta_sum) < 0.1 * notional if notional > 0 else False
        return {
            'hedging_detected': hedging_detected,
            'total_delta': delta_sum,
            'total_notional': notional,
            'summary': 'Delta-neutral activity detected.' if hedging_detected else 'No significant hedging detected.'
        }
    
    def _run_strategy_analysis(self, df: pd.DataFrame) -> dict:
        """Run strategy identification"""
        try:
            strategies = self.strategy_identifier.identify_strategies(df)
            
            # Add strategy quality assessment
            for strategy in strategies.get('identified_strategies', []):
                strategy['quality_metrics'] = self.strategy_identifier.analyze_strategy_quality(strategy)
            
            return strategies
            
        except Exception as e:
            self.log(f"Error in strategy analysis: {e}", is_error=True)
            return {'error': str(e)}
    
    def _run_microstructure_analysis(self, df: pd.DataFrame) -> dict:
        """Run market microstructure analysis"""
        try:
            microstructure = self.microstructure_analyzer.analyze_microstructure(df)
            
            # Add liquidity stress indicators
            microstructure['liquidity_stress'] = self._assess_liquidity_stress(df)
            
            return microstructure
            
        except Exception as e:
            self.log(f"Error in microstructure analysis: {e}", is_error=True)
            return {'error': str(e)}

    def _assess_liquidity_stress(self, df: pd.DataFrame) -> dict:
        """Assess liquidity stress in the market microstructure"""
        if df.empty or 'Trade_Price' not in df.columns or 'Bid' not in df.columns or 'Ask' not in df.columns:
            return {'liquidity_stress': False, 'reason': 'Insufficient data for liquidity assessment.'}
        # Calculate effective spread
        effective_spreads = abs(df['Trade_Price'] - ((df['Bid'] + df['Ask']) / 2))
        avg_effective_spread = effective_spreads.mean() if not effective_spreads.empty else 0
        # Simple threshold for stress
        liquidity_stress = avg_effective_spread > 0.1
        return {
            'liquidity_stress': liquidity_stress,
            'avg_effective_spread': avg_effective_spread,
            'reason': 'High average effective spread.' if liquidity_stress else 'No significant liquidity stress detected.'
        }
    
    def _run_alpha_extraction(self, df: pd.DataFrame) -> tuple:
        """Run alpha signal extraction"""
        try:
            # Initialize alpha extractor if needed
            if not self.alpha_extractor:
                current_price = df['Underlying_Price'].iloc[-1] if 'Underlying_Price' in df.columns else 100
                self.alpha_extractor = AlphaExtractor(self.ticker, current_price)
            
            # Extract signals
            alpha_signals = self.alpha_extractor.process_trade_flow(df)
            
            # Generate report
            alpha_report = generate_signal_report(alpha_signals)
            
            return alpha_signals, alpha_report
            
        except Exception as e:
            self.log(f"Error in alpha extraction: {e}", is_error=True)
            return [], f"Alpha extraction error: {e}"
    
    def _synthesize_results(self, results: dict, df: pd.DataFrame) -> dict:
        """Synthesize all analysis results into actionable insights"""
        
        # Extract options of interest based on multiple criteria
        options_of_interest = self._identify_options_of_interest(results, df)
        results['options_of_interest'] = [opt['symbol'] for opt in options_of_interest]
        results['options_of_interest_details'] = options_of_interest
        
        # Cross-validate signals across modules
        results['cross_validated_signals'] = self._cross_validate_signals(results)
        
        # Calculate composite scores
        results['composite_scores'] = self._calculate_composite_scores(results, df)
        
        return results
    
    def _identify_options_of_interest(self, results: dict, df: pd.DataFrame) -> list[dict]:
        """Identify most interesting options based on multiple criteria"""
        options = []
        
        # From alpha signals
        for signal in results.get('alpha_signals', []):
            options.append({
                'symbol': signal.option_symbol,
                'score': signal.confidence * signal.urgency_score / 100,
                'reason': f"Alpha Signal: {signal.signal_type.value}",
                'direction': signal.direction,
                'confidence': signal.confidence,
                'source': 'alpha'
            })
        
        # From unusual activity
        unusual = results.get('unusual_activity', {})
        for activity in unusual.get('unusual_options', []):
            options.append({
                'symbol': activity.get('symbol', ''),
                'score': activity.get('score', 0),
                'reason': "Unusual Activity Detected",
                'direction': activity.get('direction', 'NEUTRAL'),
                'confidence': activity.get('confidence', 0),
                'source': 'unusual_activity'
            })
        
        # From flow analysis
        flow = results.get('flow_analysis', {})
        for flow_item in flow.get('significant_flows', []):
            options.append({
                'symbol': flow_item.get('symbol', ''),
                'score': flow_item.get('flow_score', 0),
                'reason': "Significant Flow Detected",
                'direction': flow_item.get('direction', 'NEUTRAL'),
                'confidence': flow_item.get('confidence', 0),
                'source': 'flow'
            })
        
        # Remove duplicates and sort by score
        unique_options = {}
        for opt in options:
            symbol = opt['symbol']
            if symbol not in unique_options or opt['score'] > unique_options[symbol]['score']:
                unique_options[symbol] = opt
        
        # Sort by score and return top options
        sorted_options = sorted(unique_options.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_options[:20]  # Top 20 options
    
    def _cross_validate_signals(self, results: dict) -> list[dict]:
        """Cross-validate signals across different analysis modules"""
        validated_signals = []
        
        # Get signals from different sources
        alpha_signals = results.get('alpha_signals', [])
        unusual_activities = results.get('unusual_activity', {}).get('unusual_options', [])
        pattern_signals = results.get('pattern_analysis', {}).get('significant_patterns', [])
        
        # Cross-reference alpha signals with other modules
        for signal in alpha_signals:
            validation_score = 1.0
            supporting_evidence = []
            
            # Check if supported by unusual activity
            for activity in unusual_activities:
                if activity.get('symbol') == signal.option_symbol:
                    validation_score += 0.5
                    supporting_evidence.append('unusual_activity')
            
            # Check if supported by pattern analysis
            for pattern in pattern_signals:
                if pattern.get('symbol') == signal.option_symbol:
                    validation_score += 0.3
                    supporting_evidence.append('pattern_analysis')
            
            validated_signals.append({
                'signal': signal,
                'validation_score': validation_score,
                'supporting_evidence': supporting_evidence,
                'confidence_boost': min(20, len(supporting_evidence) * 10)
            })
        
        return validated_signals
    
    def _calculate_composite_scores(self, results: dict, df: pd.DataFrame) -> dict:
        """Calculate composite scores for overall market assessment"""
        scores = {
            'bullish_sentiment': 0,
            'bearish_sentiment': 0,
            'volatility_expectation': 0,
            'institutional_activity': 0,
            'retail_activity': 0,
            'overall_conviction': 0
        }
        
        # Analyze alpha signals
        alpha_signals = results.get('alpha_signals', [])
        if alpha_signals:
            bullish_count = sum(1 for s in alpha_signals if s.direction == 'BULLISH')
            bearish_count = sum(1 for s in alpha_signals if s.direction == 'BEARISH')
            total_signals = len(alpha_signals)
            
            if total_signals > 0:
                scores['bullish_sentiment'] = int((bullish_count / total_signals) * 100)
                scores['bearish_sentiment'] = int((bearish_count / total_signals) * 100)
                scores['overall_conviction'] = int(sum(s.confidence for s in alpha_signals) / total_signals)
        
        # Analyze flow patterns
        flow = results.get('flow_analysis', {})
        if 'directional_bias' in flow:
            bias = flow['directional_bias']
            if bias.get('net_call_put_ratio', 1) > 1.2:
                scores['bullish_sentiment'] += 10
            elif bias.get('net_call_put_ratio', 1) < 0.8:
                scores['bearish_sentiment'] += 10
        
        # Analyze strategy complexity
        strategies = results.get('strategy_analysis', {})
        institutional_strategies = strategies.get('institutional_strategies', [])
        retail_strategies = strategies.get('retail_strategies', [])
        
        scores['institutional_activity'] = min(100, len(institutional_strategies) * 10)
        scores['retail_activity'] = min(100, len(retail_strategies) * 5)
        
        # Volatility expectation from options positioning
        vol_analysis = results.get('volatility_analysis', {})
        if 'skew_analysis' in vol_analysis:
            skew = vol_analysis['skew_analysis']
            scores['volatility_expectation'] = min(100, abs(skew.get('put_skew', 0)) * 50)
        
        return scores
    
    def _generate_key_insights(self, results: dict) -> list[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Alpha signal insights
        alpha_signals = results.get('alpha_signals', [])
        if alpha_signals:
            urgent_signals = [s for s in alpha_signals if s.urgency_score >= 85]
            if urgent_signals:
                insights.append(f"🚨 {len(urgent_signals)} urgent alpha signals detected requiring immediate attention")
            
            smart_money_signals = [s for s in alpha_signals if s.smart_money_score >= 80]
            if smart_money_signals:
                insights.append(f"🧠 {len(smart_money_signals)} smart money signals identified")
        
        # Flow insights
        flow = results.get('flow_analysis', {})
        if flow.get('total_premium_traded', 0) > 10000000:  # $10M+
            insights.append(f"💰 Heavy premium flow detected: ${flow['total_premium_traded']:,.0f}")
        
        # Strategy insights
        strategies = results.get('strategy_analysis', {})
        institutional_count = len(strategies.get('institutional_strategies', []))
        if institutional_count > 0:
            insights.append(f"🏢 {institutional_count} institutional strategy patterns identified")
        
        # Unusual activity insights
        unusual = results.get('unusual_activity', {})
        if unusual.get('unusual_volume_count', 0) > 5:
            insights.append("📊 Multiple unusual volume spikes detected across strikes")
        
        # Volatility insights
        vol_analysis = results.get('volatility_analysis', {})
        if vol_analysis.get('iv_percentile_avg', 50) > 80:
            insights.append("📈 Implied volatility at elevated levels - potential overvaluation")
        elif vol_analysis.get('iv_percentile_avg', 50) < 20:
            insights.append("📉 Implied volatility at depressed levels - potential opportunity")
        
        return insights
    
    def _assess_risks(self, results: dict, df: pd.DataFrame) -> list[str]:
        """Assess risks and generate alerts"""
        alerts = []
        
        # High concentration risk
        if df['StandardOptionSymbol'].nunique() < 5 and len(df) > 100:
            alerts.append("⚠️ High concentration risk - activity focused on few options")
        
        # Liquidity concerns
        microstructure = results.get('microstructure_analysis', {})
        if microstructure.get('liquidity_metrics', {}).get('avg_effective_spread', 0) > 0.1:
            alerts.append("⚠️ Wide spreads detected - liquidity may be limited")
        
        # Volatility spike warning
        vol_analysis = results.get('volatility_analysis', {})
        if vol_analysis.get('vol_spike_detected', False):
            alerts.append("⚠️ Volatility spike detected - increased risk environment")
        
        # Large position warnings
        flow = results.get('flow_analysis', {})
        if flow.get('max_single_trade_notional', 0) > 1000000:  # $1M+
            alerts.append("⚠️ Very large individual trades detected - market impact risk")
        
        return alerts
    
    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate actionable trade recommendations"""
        recommendations = []
        
        # From alpha signals
        alpha_signals = results.get('alpha_signals', [])
        top_signals = sorted(alpha_signals, key=lambda x: x.confidence * x.urgency_score, reverse=True)[:5]
        
        for signal in top_signals:
            recommendations.append(signal.trade_recommendation)
        
        # From strategy analysis
        strategies = results.get('strategy_analysis', {})
        for strategy in strategies.get('institutional_strategies', [])[:3]:
            recommendations.append(f"Consider following institutional {strategy['strategy_name']} pattern")
        
        # From flow analysis
        composite_scores = results.get('composite_scores', {})
        if composite_scores.get('bullish_sentiment', 0) > 70:
            recommendations.append("Consider bullish positioning based on strong positive flow")
        elif composite_scores.get('bearish_sentiment', 0) > 70:
            recommendations.append("Consider bearish positioning based on strong negative flow")
        
        return recommendations[:10]  # Top 10 recommendations
    
    # Helper methods for detailed analysis
    def _analyze_directional_bias(self, df: pd.DataFrame) -> dict:
        """Analyze directional bias in flow"""
        if df.empty:
            return {}
        
        # Call vs Put analysis
        calls = df[df['Option_Type_calc'] == 'Call'] if 'Option_Type_calc' in df.columns else pd.DataFrame()
        puts = df[df['Option_Type_calc'] == 'Put'] if 'Option_Type_calc' in df.columns else pd.DataFrame()
        
        call_volume = calls['TradeQuantity'].sum() if not calls.empty else 0
        put_volume = puts['TradeQuantity'].sum() if not puts.empty else 0
        
        call_premium = calls['NotionalValue'].sum() if not calls.empty else 0
        put_premium = puts['NotionalValue'].sum() if not puts.empty else 0
        
        return {
            'call_volume': call_volume,
            'put_volume': put_volume,
            'call_put_volume_ratio': call_volume / put_volume if put_volume > 0 else float('inf'),
            'call_premium': call_premium,
            'put_premium': put_premium,
            'call_put_premium_ratio': call_premium / put_premium if put_premium > 0 else float('inf'),
            'net_call_put_ratio': (call_volume - put_volume) / (call_volume + put_volume) if (call_volume + put_volume) > 0 else 0
        }
    
    def _analyze_flow_concentration(self, df: pd.DataFrame) -> dict:
        """Analyze concentration of flow"""
        if df.empty:
            return {}
        
        # By option symbol
        symbol_concentration = df.groupby('StandardOptionSymbol')['NotionalValue'].sum().sort_values(ascending=False)
        top_5_symbols = symbol_concentration.head(5)
        top_5_percentage = (top_5_symbols.sum() / symbol_concentration.sum()) * 100 if symbol_concentration.sum() > 0 else 0
        
        # By strike
        if 'Strike_Price_calc' in df.columns:
            strike_concentration = df.groupby('Strike_Price_calc')['NotionalValue'].sum().sort_values(ascending=False)
            top_strike_percentage = (strike_concentration.iloc[0] / strike_concentration.sum()) * 100 if len(strike_concentration) > 0 else 0
        else:
            top_strike_percentage = 0
        
        return {
            'top_5_symbols_percentage': top_5_percentage,
            'top_strike_percentage': top_strike_percentage,
            'herfindahl_index': self._calculate_herfindahl_index(symbol_concentration),
            'concentration_ratio': top_5_percentage
        }
    
    def _calculate_herfindahl_index(self, concentration_series: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        if concentration_series.empty or concentration_series.sum() == 0:
            return 0
        
        shares = concentration_series / concentration_series.sum()
        return (shares ** 2).sum()
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate momentum indicators"""
        if df.empty or len(df) < 10:
            return {}
        
        # Sort by time
        df_sorted = df.sort_values('Time_dt')
        
        # Volume momentum
        volume_ma_short = df_sorted['TradeQuantity'].rolling(10).mean()
        volume_ma_long = df_sorted['TradeQuantity'].rolling(30).mean()
        volume_momentum = volume_ma_short.iloc[-1] / volume_ma_long.iloc[-1] if volume_ma_long.iloc[-1] > 0 else 1
        
        # Price momentum (if we have price data)
        if 'Trade_Price' in df.columns:
            price_changes = df_sorted['Trade_Price'].pct_change()
            price_momentum = price_changes.tail(10).mean()
        else:
            price_momentum = 0
        
        return {
            'volume_momentum': volume_momentum,
            'price_momentum': price_momentum,
            'trade_frequency_acceleration': self._calculate_frequency_acceleration(df_sorted)
        }
    
    def _calculate_frequency_acceleration(self, df_sorted: pd.DataFrame) -> float:
        """Calculate trade frequency acceleration"""
        if len(df_sorted) < 20:
            return 0
        
        # Split into two halves
        mid_point = len(df_sorted) // 2
        first_half = df_sorted.iloc[:mid_point]
        second_half = df_sorted.iloc[mid_point:]
        
        # Calculate trade frequency in each half
        first_duration = (first_half['Time_dt'].max() - first_half['Time_dt'].min()).total_seconds()
        second_duration = (second_half['Time_dt'].max() - second_half['Time_dt'].min()).total_seconds()
        
        if first_duration > 0 and second_duration > 0:
            first_frequency = len(first_half) / first_duration
            second_frequency = len(second_half) / second_duration
            return (second_frequency - first_frequency) / first_frequency if first_frequency > 0 else 0
        
        return 0

def run_holistic_analysis(df: pd.DataFrame, ticker: str, log_func: Optional[Callable] = None) -> dict:
    """Main entry point for holistic analysis"""
    engine = AnalysisEngine(ticker, log_func)
    return engine.run_holistic_analysis(df, ticker, log_func)