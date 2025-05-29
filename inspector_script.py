# inspector_script.py
"""
Data inspection and quality assessment utilities
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataInspector:
    """Comprehensive data inspection and quality assessment"""
    
    def __init__(self):
        self.inspection_results = {}
        self.quality_thresholds = {
            'completeness': 0.8,  # 80% non-null
            'consistency': 0.9,   # 90% consistent
            'validity': 0.85      # 85% valid values
        }
    
    def inspect_options_data(self, df: pd.DataFrame, ticker: str) -> dict:
        """Comprehensive inspection of options data"""
        
        inspection = {
            'basic_info': self._get_basic_info(df),
            'data_quality': self._assess_data_quality(df),
            'completeness_analysis': self._analyze_completeness(df),
            'consistency_checks': self._check_consistency(df),
            'validity_tests': self._test_validity(df),
            'outlier_detection': self._detect_outliers(df),
            'time_analysis': self._analyze_time_patterns(df),
            'volume_patterns': self._analyze_volume_patterns(df),
            'price_analysis': self._analyze_price_patterns(df),
            'greek_validation': self._validate_greeks(df),
            'recommendations': []
        }
        
        # Generate recommendations
        inspection['recommendations'] = self._generate_recommendations(inspection)
        
        # Calculate overall quality score
        inspection['overall_quality_score'] = self._calculate_quality_score(inspection)
        
        return inspection
    
    def _get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset"""
        
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape
        }
        
        # Time range if available
        if 'Time_dt' in df.columns:
            time_col = pd.to_datetime(df['Time_dt'], errors='coerce')
            info['time_range'] = {
                'start': time_col.min(),
                'end': time_col.max(),
                'duration': time_col.max() - time_col.min()
            }
        
        return info
    
    def _assess_data_quality(self, df: pd.DataFrame) -> dict:
        """Assess overall data quality"""
        
        quality = {
            'completeness_score': 0,
            'consistency_score': 0,
            'validity_score': 0,
            'overall_score': 0,
            'issues_found': [],
            'critical_issues': []
        }
        
        if df.empty:
            quality['critical_issues'].append("Dataset is empty")
            return quality
        
        # Completeness
        non_null_pct = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality['completeness_score'] = non_null_pct * 100
        
        # Consistency checks
        consistency_issues = 0
        total_checks = 0
        
        # Check time ordering
        if 'Time_dt' in df.columns:
            total_checks += 1
            time_sorted = df['Time_dt'].is_monotonic_increasing
            if not time_sorted:
                consistency_issues += 1
                quality['issues_found'].append("Time column is not sorted")
        
        # Check positive quantities
        if 'TradeQuantity' in df.columns:
            total_checks += 1
            positive_qty = (df['TradeQuantity'] > 0).all()
            if not positive_qty:
                consistency_issues += 1
                quality['issues_found'].append("Negative trade quantities found")
        
        # Check positive prices
        if 'Trade_Price' in df.columns:
            total_checks += 1
            positive_price = (df['Trade_Price'] > 0).all()
            if not positive_price:
                consistency_issues += 1
                quality['issues_found'].append("Non-positive prices found")
        
        quality['consistency_score'] = ((total_checks - consistency_issues) / max(1, total_checks)) * 100
        
        # Validity checks
        validity_issues = 0
        validity_checks = 0
        
        # IV range check
        if 'IV' in df.columns:
            validity_checks += 1
            valid_iv = ((df['IV'] >= 0) & (df['IV'] <= 5)).sum() / len(df)
            if valid_iv < 0.95:
                validity_issues += 1
                quality['issues_found'].append(f"Invalid IV values: {(1-valid_iv)*100:.1f}% outside 0-500% range")
        
        # Delta range check
        if 'Delta' in df.columns:
            validity_checks += 1
            valid_delta = ((df['Delta'] >= -1) & (df['Delta'] <= 1)).sum() / len(df)
            if valid_delta < 0.95:
                validity_issues += 1
                quality['issues_found'].append(f"Invalid Delta values: {(1-valid_delta)*100:.1f}% outside -1 to 1 range")
        
        quality['validity_score'] = ((validity_checks - validity_issues) / max(1, validity_checks)) * 100
        
        # Overall score
        quality['overall_score'] = (quality['completeness_score'] + 
                                   quality['consistency_score'] + 
                                   quality['validity_score']) / 3
        
        return quality
    
    def _analyze_completeness(self, df: pd.DataFrame) -> dict:
        """Analyze data completeness by column"""
        
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness_pct = (non_null_count / total_count) * 100 if total_count > 0 else 0
            
            completeness[col] = {
                'non_null_count': non_null_count,
                'null_count': total_count - non_null_count,
                'completeness_pct': completeness_pct,
                'status': 'Good' if completeness_pct >= 90 else 'Poor' if completeness_pct < 70 else 'Fair'
            }
        
        return completeness
    
    def _check_consistency(self, df: pd.DataFrame) -> dict:
        """Check data consistency"""
        
        consistency = {
            'time_consistency': self._check_time_consistency(df),
            'price_consistency': self._check_price_consistency(df),
            'volume_consistency': self._check_volume_consistency(df),
            'greek_consistency': self._check_greek_consistency(df),
            'symbol_consistency': self._check_symbol_consistency(df)
        }
        
        return consistency
    
    def _check_time_consistency(self, df: pd.DataFrame) -> dict:
        """Check time data consistency"""
        
        result = {'status': 'Good', 'issues': [], 'score': 100}
        
        if 'Time_dt' not in df.columns:
            result['status'] = 'Missing'
            result['issues'].append('No time column found')
            result['score'] = 0
            return result
        
        time_col = pd.to_datetime(df['Time_dt'], errors='coerce')
        
        # Check for null timestamps
        null_count = time_col.isnull().sum()
        if null_count > 0:
            result['issues'].append(f'{null_count} invalid timestamps')
            result['score'] -= 20
        
        # Check chronological order
        if not time_col.is_monotonic_increasing:
            result['issues'].append('Timestamps not in chronological order')
            result['score'] -= 30
        
        # Check for duplicates
        duplicate_count = time_col.duplicated().sum()
        if duplicate_count > 0:
            result['issues'].append(f'{duplicate_count} duplicate timestamps')
            result['score'] -= 10
        
        # Check for large gaps
        if len(time_col) > 1:
            time_diffs = time_col.diff().dt.total_seconds()
            large_gaps = (time_diffs > 3600).sum()  # > 1 hour gaps
            if large_gaps > 0:
                result['issues'].append(f'{large_gaps} large time gaps (>1 hour)')
                result['score'] -= 10
        
        result['status'] = 'Good' if result['score'] >= 80 else 'Poor' if result['score'] < 50 else 'Fair'
        return result
    
    def _check_price_consistency(self, df: pd.DataFrame) -> dict:
        """Check price data consistency"""
        
        result = {'status': 'Good', 'issues': [], 'score': 100}
        
        price_columns = ['Trade_Price', 'Option_Bid', 'Option_Ask']
        available_price_cols = [col for col in price_columns if col in df.columns]
        
        if not available_price_cols:
            result['status'] = 'Missing'
            result['issues'].append('No price columns found')
            result['score'] = 0
            return result
        
        for col in available_price_cols:
            # Check for non-positive prices
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                result['issues'].append(f'{non_positive} non-positive values in {col}')
                result['score'] -= 15
            
            # Check for extreme values
            if col == 'Trade_Price':
                extreme_high = (df[col] > 1000).sum()  # > $1000 per contract
                if extreme_high > 0:
                    result['issues'].append(f'{extreme_high} extremely high prices in {col}')
                    result['score'] -= 10
        
        # Check bid-ask relationship
        if 'Option_Bid' in df.columns and 'Option_Ask' in df.columns:
            invalid_spreads = (df['Option_Bid'] > df['Option_Ask']).sum()
            if invalid_spreads > 0:
                result['issues'].append(f'{invalid_spreads} inverted bid-ask spreads')
                result['score'] -= 25
        
        result['status'] = 'Good' if result['score'] >= 80 else 'Poor' if result['score'] < 50 else 'Fair'
        return result
    
    def _check_volume_consistency(self, df: pd.DataFrame) -> dict:
        """Check volume data consistency"""
        
        result = {'status': 'Good', 'issues': [], 'score': 100}
        
        if 'TradeQuantity' not in df.columns:
            result['status'] = 'Missing'
            result['issues'].append('No volume column found')
            result['score'] = 0
            return result
        
        # Check for non-positive quantities
        non_positive = (df['TradeQuantity'] <= 0).sum()
        if non_positive > 0:
            result['issues'].append(f'{non_positive} non-positive trade quantities')
            result['score'] -= 30
        
        # Check for extremely large trades
        extreme_large = (df['TradeQuantity'] > 10000).sum()
        if extreme_large > 0:
            result['issues'].append(f'{extreme_large} extremely large trades (>10,000 contracts)')
            result['score'] -= 10
        
        result['status'] = 'Good' if result['score'] >= 80 else 'Poor' if result['score'] < 50 else 'Fair'
        return result
    
    def _check_greek_consistency(self, df: pd.DataFrame) -> dict:
        """Check Greek values consistency"""
        
        result = {'status': 'Good', 'issues': [], 'score': 100}
        
        greek_columns = ['Delta', 'Gamma', 'Theta', 'Vega', 'IV']
        available_greeks = [col for col in greek_columns if col in df.columns]
        
        if not available_greeks:
            result['status'] = 'Missing'
            result['issues'].append('No Greek columns found')
            result['score'] = 0
            return result
        
        # Delta checks
        if 'Delta' in df.columns:
            invalid_delta = ((df['Delta'] < -1) | (df['Delta'] > 1)).sum()
            if invalid_delta > 0:
                result['issues'].append(f'{invalid_delta} invalid Delta values (outside -1 to 1)')
                result['score'] -= 20
        
        # IV checks
        if 'IV' in df.columns:
            invalid_iv = ((df['IV'] < 0) | (df['IV'] > 5)).sum()
            if invalid_iv > 0:
                result['issues'].append(f'{invalid_iv} invalid IV values (outside 0% to 500%)')
                result['score'] -= 20
        
        # Gamma checks (should be positive for long options)
        if 'Gamma' in df.columns:
            negative_gamma = (df['Gamma'] < 0).sum()
            if negative_gamma > len(df) * 0.1:  # More than 10% negative
                result['issues'].append(f'Unusually high negative Gamma values: {negative_gamma}')
                result['score'] -= 10
        
        result['status'] = 'Good' if result['score'] >= 80 else 'Poor' if result['score'] < 50 else 'Fair'
        return result
    
    def _check_symbol_consistency(self, df: pd.DataFrame) -> dict:
        """Check option symbol consistency"""
        
        result = {'status': 'Good', 'issues': [], 'score': 100}
        
        if 'StandardOptionSymbol' not in df.columns:
            result['status'] = 'Missing'
            result['issues'].append('No option symbol column found')
            result['score'] = 0
            return result
        
        # Check for empty symbols
        empty_symbols = df['StandardOptionSymbol'].isnull().sum()
        if empty_symbols > 0:
            result['issues'].append(f'{empty_symbols} empty option symbols')
            result['score'] -= 20
        
        # Check symbol format consistency
        symbols = df['StandardOptionSymbol'].dropna()
        if not symbols.empty:
            # Basic format check (should contain letters and numbers)
            invalid_format = 0
            for symbol in symbols:
                if not any(c.isalpha() for c in str(symbol)) or not any(c.isdigit() for c in str(symbol)):
                    invalid_format += 1
            
            if invalid_format > 0:
                result['issues'].append(f'{invalid_format} symbols with invalid format')
                result['score'] -= 15
        
        result['status'] = 'Good' if result['score'] >= 80 else 'Poor' if result['score'] < 50 else 'Fair'
        return result
    
    def _test_validity(self, df: pd.DataFrame) -> dict:
        """Test data validity with business rules"""
        
        validity = {
            'business_rules': [],
            'statistical_tests': [],
            'validation_score': 100
        }
        
        # Business rule validations
        if 'TradeQuantity' in df.columns and 'Trade_Price' in df.columns:
            # Check if notional values are reasonable
            df['calc_notional'] = df['TradeQuantity'] * df['Trade_Price'] * 100
            extreme_notional = (df['calc_notional'] > 10000000).sum()  # > $10M per trade
            if extreme_notional > 0:
                validity['business_rules'].append({
                    'rule': 'Reasonable notional values',
                    'violations': extreme_notional,
                    'severity': 'High'
                })
                validity['validation_score'] -= 20
        
        # Greek relationships
        if all(col in df.columns for col in ['Delta', 'Strike_Price_calc', 'Option_Type_calc']):
            # For calls: Delta should be positive for ITM, negative for OTM relative to ATM
            # This is a simplified check
            call_options = df[df['Option_Type_calc'] == 'Call']
            if not call_options.empty:
                invalid_call_deltas = (call_options['Delta'] < 0).sum()
                if invalid_call_deltas > len(call_options) * 0.1:  # More than 10%
                    validity['business_rules'].append({
                        'rule': 'Call option Delta signs',
                        'violations': invalid_call_deltas,
                        'severity': 'Medium'
                    })
                    validity['validation_score'] -= 10
        
        return validity
    
    def _detect_outliers(self, df: pd.DataFrame) -> dict:
        """Detect outliers in numeric columns"""
        
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                values = df[col].dropna()
                
                # IQR method
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(values)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'extreme_values': values[outlier_mask].head(10).tolist()
                    }
        
        return outliers
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze temporal patterns in the data"""
        
        patterns = {}
        
        if 'Time_dt' not in df.columns:
            return patterns
        
        time_col = pd.to_datetime(df['Time_dt'], errors='coerce').dropna()
        
        if time_col.empty:
            return patterns
        
        # Trading hours analysis
        hours = time_col.dt.hour
        patterns['trading_hours'] = {
            'market_hours_pct': ((hours >= 9) & (hours <= 16)).sum() / len(hours) * 100,
            'after_hours_pct': ((hours < 9) | (hours > 16)).sum() / len(hours) * 100,
            'peak_hour': hours.mode().iloc[0] if not hours.mode().empty else None
        }
        
        # Day of week analysis
        if len(time_col) > 1:
            days = time_col.dt.dayofweek
            patterns['day_distribution'] = {
                'monday': (days == 0).sum(),
                'tuesday': (days == 1).sum(),
                'wednesday': (days == 2).sum(),
                'thursday': (days == 3).sum(),
                'friday': (days == 4).sum(),
                'weekend': ((days == 5) | (days == 6)).sum()
            }
        
        return patterns
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns"""
        
        patterns = {}
        
        if 'TradeQuantity' not in df.columns:
            return patterns
        
        volumes = df['TradeQuantity'].dropna()
        
        if volumes.empty:
            return patterns
        
        patterns['distribution'] = {
            'mean': volumes.mean(),
            'median': volumes.median(),
            'std': volumes.std(),
            'skewness': volumes.skew(),
            'kurtosis': volumes.kurtosis()
        }
        
        # Size categories
        patterns['size_categories'] = {
            'small_trades': (volumes <= 10).sum(),
            'medium_trades': ((volumes > 10) & (volumes <= 50)).sum(),
            'large_trades': ((volumes > 50) & (volumes <= 100)).sum(),
            'block_trades': (volumes > 100).sum()
        }
        
        return patterns
    
    def _analyze_price_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze price patterns"""
        
        patterns = {}
        
        if 'Trade_Price' not in df.columns:
            return patterns
        
        prices = df['Trade_Price'].dropna()
        
        if prices.empty:
            return patterns
        
        patterns['distribution'] = {
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'min': prices.min(),
            'max': prices.max(),
            'range': prices.max() - prices.min()
        }
        
        # Price categories
        patterns['price_categories'] = {
            'penny_options': (prices <= 0.05).sum(),
            'cheap_options': ((prices > 0.05) & (prices <= 1.0)).sum(),
            'moderate_options': ((prices > 1.0) & (prices <= 10.0)).sum(),
            'expensive_options': (prices > 10.0).sum()
        }
        
        return patterns
    
    def _validate_greeks(self, df: pd.DataFrame) -> dict:
        """Validate Greek values"""
        
        validation = {}
        greek_columns = ['Delta', 'Gamma', 'Theta', 'Vega', 'IV']
        
        for greek in greek_columns:
            if greek in df.columns:
                values = df[greek].dropna()
                
                if not values.empty:
                    validation[greek] = {
                        'count': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'valid_range_pct': self._check_greek_range(values, greek)
                    }
        
        return validation
    
    def _check_greek_range(self, values: pd.Series, greek: str) -> float:
        """Check if Greek values are in valid ranges"""
        
        valid_ranges = {
            'Delta': (-1, 1),
            'Gamma': (0, 1),
            'Theta': (-10, 0),
            'Vega': (0, 10),
            'IV': (0, 5)
        }
        
        if greek not in valid_ranges:
            return 100.0
        
        min_val, max_val = valid_ranges[greek]
        valid_count = ((values >= min_val) & (values <= max_val)).sum()
        
        return (valid_count / len(values)) * 100
    
    def _generate_recommendations(self, inspection: dict) -> list[str]:
        """Generate data quality recommendations"""
        
        recommendations = []
        
        # Check overall quality score
        overall_score = inspection.get('overall_quality_score', 0)
        
        if overall_score < 70:
            recommendations.append("Overall data quality is poor. Consider data cleaning before analysis.")
        
        # Check completeness
        completeness = inspection.get('completeness_analysis', {})
        for col, info in completeness.items():
            if info.get('completeness_pct', 100) < 80:
                recommendations.append(f"Column '{col}' has low completeness ({info['completeness_pct']:.1f}%). Consider imputation or removal.")
        
        # Check consistency issues
        quality = inspection.get('data_quality', {})
        if quality.get('issues_found'):
            recommendations.append("Address consistency issues: " + "; ".join(quality['issues_found'][:3]))
        
        # Check outliers
        outliers = inspection.get('outlier_detection', {})
        if outliers:
            recommendations.append(f"Review outliers in {len(outliers)} columns before analysis.")
        
        return recommendations
    
    def _calculate_quality_score(self, inspection: dict) -> float:
        """Calculate overall quality score"""
        
        quality_metrics = inspection.get('data_quality', {})
        
        completeness = quality_metrics.get('completeness_score', 0)
        consistency = quality_metrics.get('consistency_score', 0)
        validity = quality_metrics.get('validity_score', 0)
        
        return (completeness + consistency + validity) / 3
    
    def generate_inspection_report(self, inspection: dict, ticker: str) -> str:
        """Generate comprehensive inspection report"""
        
        report = f"DATA INSPECTION REPORT - {ticker}\n"
        report += "=" * 50 + "\n\n"
        
        # Basic info
        basic_info = inspection.get('basic_info', {})
        report += f"Dataset Size: {basic_info.get('total_rows', 0):,} rows × {basic_info.get('total_columns', 0)} columns\n"
        report += f"Memory Usage: {basic_info.get('memory_usage_mb', 0):.2f} MB\n\n"
        
        # Quality summary
        quality = inspection.get('data_quality', {})
        report += "QUALITY ASSESSMENT:\n"
        report += f"Overall Score: {inspection.get('overall_quality_score', 0):.1f}/100\n"
        report += f"Completeness: {quality.get('completeness_score', 0):.1f}%\n"
        report += f"Consistency: {quality.get('consistency_score', 0):.1f}%\n"
        report += f"Validity: {quality.get('validity_score', 0):.1f}%\n\n"
        
        # Issues
        if quality.get('issues_found'):
            report += "ISSUES FOUND:\n"
            for issue in quality['issues_found']:
                report += f"• {issue}\n"
            report += "\n"
        
        # Recommendations
        recommendations = inspection.get('recommendations', [])
        if recommendations:
            report += "RECOMMENDATIONS:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        # Outliers summary
        outliers = inspection.get('outlier_detection', {})
        if outliers:
            report += "OUTLIER SUMMARY:\n"
            for col, info in outliers.items():
                report += f"• {col}: {info['count']} outliers ({info['percentage']:.1f}%)\n"
        
        return report


def inspect_options_data(df: pd.DataFrame, ticker: str) -> dict:
    """Main function to inspect options data"""
    inspector = DataInspector()
    return inspector.inspect_options_data(df, ticker)


def generate_inspection_report(df: pd.DataFrame, ticker: str) -> str:
    """Generate inspection report"""
    inspector = DataInspector()
    inspection = inspector.inspect_options_data(df, ticker)
    return inspector.generate_inspection_report(inspection, ticker)