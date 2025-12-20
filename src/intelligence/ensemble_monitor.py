"""
EnsembleMonitor - Task 8.2: Monitoring and Analytics Dashboard

Real-time ensemble performance monitoring and model contribution analysis.
Provides comprehensive analytics for multi-model AI trading system performance,
cost tracking, and decision quality analysis.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import sqlite3
import numpy as np

from src.utils.database import DatabaseManager, ModelPerformance, EnsembleDecision
from src.utils.logging_setup import TradingLoggerMixin


class EnsembleMonitor(TradingLoggerMixin):
    """
    Comprehensive monitoring and analytics system for ensemble AI trading.

    Tracks real-time performance, analyzes model contributions, monitors costs,
    and provides insights for ensemble optimization and decision quality.
    """

    def __init__(self, db_path: str = None):
        """Initialize ensemble monitor with database connection."""
        self.db = DatabaseManager(db_path or "trading_system.db")
        self.logger.info("EnsembleMonitor initialized")

        # Performance tracking thresholds
        self.ALERT_THRESHOLDS = {
            'low_success_rate': 0.3,      # Alert if success rate below 30%
            'high_cost_ratio': 0.5,       # Alert if costs exceed 50% of PnL
            'high_disagreement': 0.7,     # Alert for high disagreement
            'consecutive_failures': 10,   # Alert after 10 consecutive failures
            'slow_response': 5000,        # Alert if response time > 5 seconds
            'budget_usage': 0.9           # Alert if >90% of budget used
        }

        # Performance windows for analysis
        self.TIME_WINDOWS = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '24h': timedelta(days=1),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }

    async def initialize(self) -> None:
        """Initialize database connection and create monitoring tables."""
        await self.db.initialize()
        self.logger.info("EnsembleMonitor database initialized")

    async def track_performance(self, decision_data: Dict[str, Any]) -> None:
        """
        Track ensemble decision performance data.

        Args:
            decision_data: Dictionary containing decision details and outcomes
        """
        try:
            # Create ensemble decision record
            ensemble_decision = EnsembleDecision(
                market_id=decision_data['market_id'],
                models_consulted=decision_data.get('models_consulted', []),
                final_decision=decision_data['final_decision'],
                disagreement_level=decision_data.get('disagreement_level', 0.0),
                selected_model=decision_data.get('selected_model', 'unknown'),
                reasoning=decision_data.get('reasoning', ''),
                timestamp=decision_data.get('timestamp', datetime.now())
            )

            # Store ensemble decision
            await self.db.create_ensemble_decision(ensemble_decision)

            # Track model performance if outcome is known
            if 'outcome' in decision_data:
                await self._track_model_performance(decision_data)

                # Check for performance alerts
                await self._check_performance_alerts(decision_data)

            self.logger.debug(
                "Performance tracked",
                market_id=decision_data['market_id'],
                selected_model=decision_data.get('selected_model'),
                outcome=decision_data.get('outcome')
            )

        except Exception as e:
            self.logger.error(
                "Failed to track performance",
                error=str(e),
                decision_data=decision_data
            )
            raise

    async def _track_model_performance(self, decision_data: Dict[str, Any]) -> None:
        """Track individual model performance from ensemble decision."""
        selected_model = decision_data.get('selected_model')
        if not selected_model:
            return

        # Calculate performance metrics
        success = decision_data['outcome'] == 'success'
        accuracy_score = 1.0 if success else 0.0

        # Confidence calibration (simplified)
        confidence = decision_data.get('confidence_extracted', 0.5)
        confidence_calibration = 1.0 - abs(confidence - accuracy_score)

        # Response time (from data or default)
        response_time_ms = decision_data.get('response_time_ms', 1000)

        # Cost information
        cost_usd = decision_data.get('cost_usd', 0.0)

        # Decision quality (combination of accuracy and PnL efficiency)
        pnl = decision_data.get('pnl', 0.0)
        decision_quality = min(1.0, max(0.0, (pnl / 100.0) if success else 0.0))

        performance = ModelPerformance(
            model_name=selected_model,
            timestamp=decision_data.get('timestamp', datetime.now()),
            market_category=decision_data.get('market_category', 'unknown'),
            accuracy_score=accuracy_score,
            confidence_calibration=confidence_calibration,
            response_time_ms=response_time_ms,
            cost_usd=cost_usd,
            decision_quality=decision_quality
        )

        await self.db.create_model_performance(performance)

    async def track_model_cost(self, cost_data: Dict[str, Any]) -> None:
        """
        Track model usage cost data.

        Args:
            cost_data: Dictionary containing cost information
        """
        try:
            # This would integrate with existing LLM query logging
            # For now, we'll store cost information in a simple structure
            cost_record = {
                'model_name': cost_data.get('model_name'),
                'timestamp': cost_data.get('timestamp', datetime.now()),
                'cost_usd': cost_data.get('cost_usd', 0.0),
                'market_category': cost_data.get('market_category', 'unknown'),
                'tokens_used': cost_data.get('tokens_used', 0),
                'market_id': cost_data.get('market_id')
            }

            # Store cost data (this could be enhanced with proper database table)
            await self._store_cost_record(cost_record)

            # Check cost alerts
            await self._check_cost_alerts(cost_data)

        except Exception as e:
            self.logger.error(
                "Failed to track model cost",
                error=str(e),
                cost_data=cost_data
            )

    async def _store_cost_record(self, cost_record: Dict[str, Any]) -> None:
        """Store cost record in database (simplified implementation)."""
        # This would ideally use a dedicated cost tracking table
        # For now, we'll store it as JSON in an existing table or log
        cost_json = json.dumps(cost_record)
        self.logger.info("Model cost tracked", cost_data=cost_json)

    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """
        Get real-time ensemble performance metrics.

        Returns:
            Dictionary containing current performance metrics
        """
        try:
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)

            # Get recent decisions
            recent_decisions = await self.db.get_ensemble_decisions(
                start_time=one_hour_ago
            )

            if not recent_decisions:
                return {
                    'total_decisions': 0,
                    'success_rate': 0.0,
                    'avg_disagreement': 0.0,
                    'total_pnl': 0.0,
                    'recent_trend': 'no_data'
                }

            # Calculate metrics
            total_decisions = len(recent_decisions)
            successful_decisions = sum(1 for d in recent_decisions
                                     if hasattr(d, 'outcome') and d.outcome == 'success')
            success_rate = successful_decisions / total_decisions

            avg_disagreement = sum(d.disagreement_level for d in recent_decisions) / total_decisions

            # Calculate PnL (this would come from actual trade results)
            total_pnl = 0.0  # This would be calculated from actual position outcomes

            # Determine trend
            recent_trend = self._calculate_trend(recent_decisions)

            return {
                'total_decisions': total_decisions,
                'success_rate': success_rate,
                'avg_disagreement': avg_disagreement,
                'total_pnl': total_pnl,
                'recent_trend': recent_trend,
                'timestamp': now.isoformat()
            }

        except Exception as e:
            self.logger.error("Failed to get realtime metrics", error=str(e))
            return {}

    def _calculate_trend(self, decisions: List[Any]) -> str:
        """Calculate recent performance trend from decisions."""
        if len(decisions) < 5:
            return 'insufficient_data'

        # Simple trend calculation based on recent vs older performance
        recent_half = decisions[:len(decisions)//2]
        older_half = decisions[len(decisions)//2:]

        # This is simplified - would use actual success/PnL data
        return 'stable'  # Could be 'improving', 'declining', 'stable'

    async def get_performance_windows(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics across different time windows.

        Returns:
            Dictionary with metrics for each time window
        """
        try:
            windows = {}
            now = datetime.now()

            for window_name, window_delta in self.TIME_WINDOWS.items():
                start_time = now - window_delta

                # Get decisions in this window
                decisions = await self.db.get_ensemble_decisions(
                    start_time=start_time
                )

                # Calculate window metrics
                window_metrics = self._calculate_window_metrics(decisions)
                windows[window_name] = window_metrics

            return windows

        except Exception as e:
            self.logger.error("Failed to get performance windows", error=str(e))
            return {}

    def _calculate_window_metrics(self, decisions: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for a specific time window."""
        if not decisions:
            return {
                'decisions': 0,
                'success_rate': 0.0,
                'avg_disagreement': 0.0,
                'model_distribution': {}
            }

        total_decisions = len(decisions)

        # Calculate model distribution
        model_count = {}
        for decision in decisions:
            model = decision.selected_model
            model_count[model] = model_count.get(model, 0) + 1

        model_distribution = {
            model: count / total_decisions
            for model, count in model_count.items()
        }

        return {
            'decisions': total_decisions,
            'success_rate': 0.0,  # Would calculate from actual outcomes
            'avg_disagreement': sum(d.disagreement_level for d in decisions) / total_decisions,
            'model_distribution': model_distribution
        }

    async def analyze_model_contributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze which models drive successful trades across different conditions.

        Returns:
            Dictionary with contribution analysis for each model
        """
        try:
            contributions = {}

            # Get all model performance records
            performances = await self.db.get_model_performances()

            # Group by model
            model_performances = {}
            for perf in performances:
                if perf.model_name not in model_performances:
                    model_performances[perf.model_name] = []
                model_performances[perf.model_name].append(perf)

            # Analyze each model's contributions
            for model_name, model_data in model_performances.items():
                contribution = self._analyze_model_contribution(model_name, model_data)
                contributions[model_name] = contribution

            return contributions

        except Exception as e:
            self.logger.error("Failed to analyze model contributions", error=str(e))
            return {}

    def _analyze_model_contribution(self, model_name: str, performances: List[Any]) -> Dict[str, Any]:
        """Analyze individual model's contribution."""
        if not performances:
            return {}

        # Calculate basic metrics
        total_decisions = len(performances)
        successful_decisions = sum(p.accuracy_score for p in performances)
        total_pnl = sum(p.decision_quality * 100 for p in performances)  # Simplified PnL calculation

        success_rate = successful_decisions / total_decisions
        avg_quality = sum(p.decision_quality for p in performances) / total_decisions

        # Group by market category
        category_performance = {}
        for perf in performances:
            category = perf.market_category
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(perf)

        by_category = {}
        for category, cat_perfs in category_performance.items():
            cat_success = sum(p.accuracy_score for p in cat_perfs)
            cat_count = len(cat_perfs)
            cat_pnl = sum(p.decision_quality * 100 for p in cat_perfs)

            by_category[category] = {
                'success_rate': cat_success / cat_count,
                'total_pnl': cat_pnl,
                'decisions': cat_count,
                'avg_quality': sum(p.decision_quality for p in cat_perfs) / cat_count
            }

        return {
            'total_decisions': total_decisions,
            'total_successes': successful_decisions,
            'total_failures': total_decisions - successful_decisions,
            'total_pnl': total_pnl,
            'success_rate': success_rate,
            'avg_decision_quality': avg_quality,
            'by_category': by_category,
            'avg_response_time': sum(p.response_time_ms for p in performances) / total_decisions,
            'avg_cost_per_decision': sum(p.cost_usd for p in performances) / total_decisions
        }

    async def identify_model_strengths(self) -> Dict[str, Dict[str, Any]]:
        """
        Identify model strengths in specific market conditions.

        Returns:
            Dictionary with strength analysis for each model
        """
        try:
            strengths = {}

            # Get recent performance data
            recent_performances = await self.db.get_model_performances(
                start_time=datetime.now() - timedelta(days=7)
            )

            # Group by model
            model_data = {}
            for perf in recent_performances:
                if perf.model_name not in model_data:
                    model_data[perf.model_name] = []
                model_data[perf.model_name].append(perf)

            # Analyze strengths for each model
            for model_name, performances in model_data.items():
                model_strengths = self._analyze_model_strengths(model_name, performances)
                strengths[model_name] = model_strengths

            return strengths

        except Exception as e:
            self.logger.error("Failed to identify model strengths", error=str(e))
            return {}

    def _analyze_model_strengths(self, model_name: str, performances: List[Any]) -> Dict[str, Any]:
        """Analyze individual model's strengths."""
        if not performances:
            return {}

        # Analyze performance by market category
        category_stats = {}
        for perf in performances:
            category = perf.market_category
            if category not in category_stats:
                category_stats[category] = {
                    'successes': 0,
                    'total': 0,
                    'total_quality': 0.0,
                    'total_response_time': 0,
                    'total_cost': 0.0
                }

            stats = category_stats[category]
            stats['total'] += 1
            stats['successes'] += perf.accuracy_score
            stats['total_quality'] += perf.decision_quality
            stats['total_response_time'] += perf.response_time_ms
            stats['total_cost'] += perf.cost_usd

        # Calculate category metrics
        top_categories = {}
        for category, stats in category_stats.items():
            if stats['total'] >= 3:  # Only include categories with sufficient data
                top_categories[category] = {
                    'success_rate': stats['successes'] / stats['total'],
                    'avg_pnl': (stats['total_quality'] / stats['total']) * 100,  # Simplified
                    'avg_response_time': stats['total_response_time'] / stats['total'],
                    'avg_cost': stats['total_cost'] / stats['total'],
                    'sample_size': stats['total']
                }

        # Sort categories by success rate
        sorted_categories = sorted(
            top_categories.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )

        # Identify top performing categories
        top_performers = dict(sorted_categories[:3])

        # Calculate overall model metrics
        overall_success = sum(perf.accuracy_score for perf in performances) / len(performances)
        overall_quality = sum(perf.decision_quality for perf in performances) / len(performances)
        overall_response_time = sum(perf.response_time_ms for perf in performances) / len(performances)

        return {
            'top_categories': top_performers,
            'overall_success_rate': overall_success,
            'overall_quality_score': overall_quality,
            'avg_response_time_ms': overall_response_time,
            'total_decisions_analyzed': len(performances),
            'strength_score': self._calculate_strength_score(top_categories, len(performances))
        }

    def _calculate_strength_score(self, categories: Dict[str, Any], total_decisions: int) -> float:
        """Calculate a strength score for the model based on category performance."""
        if not categories or total_decisions < 10:
            return 0.0

        # Weight categories by sample size and success rate
        weighted_score = 0.0
        total_weight = 0.0

        for category, stats in categories.items():
            weight = min(stats['sample_size'] / 10, 1.0)  # Cap weight at 1.0
            category_score = stats['success_rate']

            weighted_score += weight * category_score
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    async def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        Get comprehensive cost breakdown by model and category.

        Returns:
            Dictionary with detailed cost analysis
        """
        try:
            # This would ideally query from dedicated cost tracking tables
            # For now, we'll use model performance data for cost analysis

            performances = await self.db.get_model_performances()

            # Group costs by model
            model_costs = {}
            for perf in performances:
                model = perf.model_name
                if model not in model_costs:
                    model_costs[model] = {
                        'total_cost': 0.0,
                        'total_requests': 0,
                        'total_tokens': 0,
                        'total_pnl': 0.0
                    }

                model_costs[model]['total_cost'] += perf.cost_usd
                model_costs[model]['total_requests'] += 1
                model_costs[model]['total_tokens'] += getattr(perf, 'tokens_used', 0)
                model_costs[model]['total_pnl'] += perf.decision_quality * 100  # Simplified PnL

            # Calculate per-model metrics
            by_model = {}
            for model, costs in model_costs.items():
                by_model[model] = {
                    'total_cost': round(costs['total_cost'], 4),
                    'total_requests': costs['total_requests'],
                    'avg_cost_per_request': round(costs['total_cost'] / costs['total_requests'], 4),
                    'total_tokens': costs['total_tokens'],
                    'avg_tokens_per_request': costs['total_tokens'] / costs['total_requests'],
                    'total_pnl': round(costs['total_pnl'], 2),
                    'cost_per_pnl': costs['total_cost'] / abs(costs['total_pnl']) if costs['total_pnl'] != 0 else 0,
                    'roi_ratio': costs['total_pnl'] / costs['total_cost'] if costs['total_cost'] > 0 else 0
                }

            # Calculate category breakdown (simplified)
            category_costs = await self.get_cost_breakdown_by_category()

            return {
                'by_model': by_model,
                'by_category': category_costs,
                'summary': {
                    'total_cost': sum(costs['total_cost'] for costs in model_costs.values()),
                    'total_requests': sum(costs['total_requests'] for costs in model_costs.values()),
                    'total_pnl': sum(costs['total_pnl'] for costs in model_costs.values())
                }
            }

        except Exception as e:
            self.logger.error("Failed to get cost breakdown", error=str(e))
            return {}

    async def get_cost_breakdown_by_category(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by market category.

        Returns:
            Dictionary with cost analysis by category
        """
        try:
            performances = await self.db.get_model_performances()

            # Group by category
            category_costs = {}
            for perf in performances:
                category = perf.market_category
                if category not in category_costs:
                    category_costs[category] = {
                        'total_cost': 0.0,
                        'total_requests': 0,
                        'models_used': set(),
                        'total_pnl': 0.0
                    }

                category_costs[category]['total_cost'] += perf.cost_usd
                category_costs[category]['total_requests'] += 1
                category_costs[category]['models_used'].add(perf.model_name)
                category_costs[category]['total_pnl'] += perf.decision_quality * 100

            # Format output
            by_category = {}
            for category, costs in category_costs.items():
                by_category[category] = {
                    'total_cost': round(costs['total_cost'], 4),
                    'total_requests': costs['total_requests'],
                    'avg_cost_per_request': round(costs['total_cost'] / costs['total_requests'], 4),
                    'models_used': list(costs['models_used']),
                    'model_count': len(costs['models_used']),
                    'total_pnl': round(costs['total_pnl'], 2),
                    'cost_efficiency': costs['total_pnl'] / costs['total_cost'] if costs['total_cost'] > 0 else 0
                }

            return by_category

        except Exception as e:
            self.logger.error("Failed to get cost breakdown by category", error=str(e))
            return {}

    async def analyze_agreement_patterns(self) -> Dict[str, Any]:
        """
        Analyze ensemble agreement/disagreement patterns with decision quality correlation.

        Returns:
            Dictionary with agreement pattern analysis
        """
        try:
            # Get recent ensemble decisions
            recent_decisions = await self.db.get_ensemble_decisions(
                start_time=datetime.now() - timedelta(days=7)
            )

            if not recent_decisions:
                return {}

            # Group decisions by disagreement level
            agreement_groups = {
                'high_agreement': [],      # disagreement < 0.33
                'moderate_agreement': [],  # 0.33 <= disagreement < 0.67
                'high_disagreement': []    # disagreement >= 0.67
            }

            for decision in recent_decisions:
                disagreement = decision.disagreement_level
                if disagreement < 0.33:
                    agreement_groups['high_agreement'].append(decision)
                elif disagreement < 0.67:
                    agreement_groups['moderate_agreement'].append(decision)
                else:
                    agreement_groups['high_disagreement'].append(decision)

            # Analyze each group
            analysis = {}
            for group_name, group_decisions in agreement_groups.items():
                if group_decisions:
                    analysis[group_name] = self._analyze_agreement_group(group_decisions)
                else:
                    analysis[group_name] = {
                        'count': 0,
                        'avg_disagreement': 0.0,
                        'success_rate': 0.0,
                        'avg_pnl': 0.0
                    }

            # Calculate correlation
            correlation = self._calculate_disagreement_correlation(recent_decisions)
            analysis['correlation'] = correlation

            return analysis

        except Exception as e:
            self.logger.error("Failed to analyze agreement patterns", error=str(e))
            return {}

    def _analyze_agreement_group(self, decisions: List[Any]) -> Dict[str, Any]:
        """Analyze decisions within a specific agreement group."""
        if not decisions:
            return {}

        # Calculate basic metrics
        count = len(decisions)
        avg_disagreement = sum(d.disagreement_level for d in decisions) / count

        # This would use actual success data from trade outcomes
        # For now, we'll use placeholder calculations
        success_rate = 0.0  # Would calculate from actual outcomes
        avg_pnl = 0.0       # Would calculate from actual PnL

        return {
            'count': count,
            'avg_disagreement': round(avg_disagreement, 3),
            'success_rate': round(success_rate, 3),
            'avg_pnl': round(avg_pnl, 2),
            'model_distribution': self._get_model_distribution(decisions)
        }

    def _get_model_distribution(self, decisions: List[Any]) -> Dict[str, float]:
        """Get distribution of selected models in decisions."""
        model_counts = {}
        for decision in decisions:
            model = decision.selected_model
            model_counts[model] = model_counts.get(model, 0) + 1

        total = len(decisions)
        return {model: count / total for model, count in model_counts.items()}

    def _calculate_disagreement_correlation(self, decisions: List[Any]) -> Dict[str, Any]:
        """Calculate correlation between disagreement and decision quality."""
        if len(decisions) < 10:
            return {
                'correlation_coefficient': 0.0,
                'sample_size': len(decisions),
                'significance': 'insufficient_data'
            }

        # This would use actual decision quality/outcome data
        # For now, we'll provide a placeholder implementation
        correlation_coeff = 0.0  # Would calculate actual correlation

        significance = 'weak'
        if abs(correlation_coeff) > 0.3:
            significance = 'moderate'
        if abs(correlation_coeff) > 0.5:
            significance = 'strong'

        return {
            'correlation_coefficient': round(correlation_coeff, 3),
            'sample_size': len(decisions),
            'significance': significance
        }

    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data for monitoring and analytics.

        Returns:
            Dictionary with all dashboard components
        """
        try:
            # Get all data components
            realtime_metrics = await self.get_realtime_metrics()
            performance_windows = await self.get_performance_windows()
            model_contributions = await self.analyze_model_contributions()
            cost_breakdown = await self.get_cost_breakdown()
            agreement_analysis = await self.analyze_agreement_patterns()
            time_series = await self.generate_time_series_data()

            dashboard_data = {
                'last_updated': datetime.now().isoformat(),
                'summary_metrics': realtime_metrics,
                'performance_windows': performance_windows,
                'model_performance': model_contributions,
                'cost_analysis': cost_breakdown,
                'agreement_analysis': agreement_analysis,
                'time_series_data': time_series,
                'alerts': await self.generate_dashboard_alerts()
            }

            self.logger.info("Dashboard data generated", data_size=len(str(dashboard_data)))
            return dashboard_data

        except Exception as e:
            self.logger.error("Failed to generate dashboard data", error=str(e))
            return {}

    async def generate_time_series_data(self) -> Dict[str, Any]:
        """
        Generate time series data for dashboard charts.

        Returns:
            Dictionary with time series data points
        """
        try:
            # Get hourly data for last 48 hours
            now = datetime.now()
            hourly_metrics = []

            for hours_ago in range(48, -1, -1):
                timestamp = now - timedelta(hours=hours_ago)
                hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
                hour_end = hour_start + timedelta(hours=1)

                # Get decisions for this hour
                hour_decisions = await self.db.get_ensemble_decisions(
                    start_time=hour_start,
                    end_time=hour_end
                )

                hour_metrics = self._calculate_hourly_metrics(hour_decisions, hour_start)
                hourly_metrics.append(hour_metrics)

            # Calculate trends
            performance_trend = self._calculate_performance_trend(hourly_metrics)

            return {
                'hourly_metrics': hourly_metrics,
                'daily_metrics': self._aggregate_daily_metrics(hourly_metrics),
                'performance_trend': performance_trend
            }

        except Exception as e:
            self.logger.error("Failed to generate time series data", error=str(e))
            return {}

    def _calculate_hourly_metrics(self, decisions: List[Any], timestamp: datetime) -> Dict[str, Any]:
        """Calculate metrics for a specific hour."""
        if not decisions:
            return {
                'timestamp': timestamp.isoformat(),
                'decisions_count': 0,
                'success_rate': 0.0,
                'avg_disagreement': 0.0,
                'total_pnl': 0.0,
                'total_cost': 0.0
            }

        count = len(decisions)
        avg_disagreement = sum(d.disagreement_level for d in decisions) / count

        return {
            'timestamp': timestamp.isoformat(),
            'decisions_count': count,
            'success_rate': 0.0,  # Would calculate from actual outcomes
            'avg_disagreement': round(avg_disagreement, 3),
            'total_pnl': 0.0,       # Would calculate from actual PnL
            'total_cost': 0.0,      # Would calculate from actual costs
            'model_distribution': self._get_model_distribution(decisions)
        }

    def _aggregate_daily_metrics(self, hourly_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate hourly metrics into daily metrics."""
        if not hourly_metrics:
            return []

        # Group by day
        daily_data = {}
        for hour_metric in hourly_metrics:
            date_str = hour_metric['timestamp'][:10]  # Extract date
            if date_str not in daily_data:
                daily_data[date_str] = {
                    'date': date_str,
                    'total_decisions': 0,
                    'total_pnl': 0.0,
                    'total_cost': 0.0,
                    'success_rate_sum': 0.0,
                    'disagreement_sum': 0.0,
                    'hours_with_data': 0
                }

            day = daily_data[date_str]
            day['total_decisions'] += hour_metric['decisions_count']
            day['total_pnl'] += hour_metric['total_pnl']
            day['total_cost'] += hour_metric['total_cost']
            day['success_rate_sum'] += hour_metric['success_rate']
            day['disagreement_sum'] += hour_metric['avg_disagreement']
            if hour_metric['decisions_count'] > 0:
                day['hours_with_data'] += 1

        # Calculate daily averages
        daily_metrics = []
        for day in daily_data.values():
            avg_success_rate = day['success_rate_sum'] / day['hours_with_data'] if day['hours_with_data'] > 0 else 0
            avg_disagreement = day['disagreement_sum'] / day['hours_with_data'] if day['hours_with_data'] > 0 else 0

            daily_metrics.append({
                'date': day['date'],
                'total_decisions': day['total_decisions'],
                'avg_success_rate': round(avg_success_rate, 3),
                'avg_disagreement': round(avg_disagreement, 3),
                'total_pnl': round(day['total_pnl'], 2),
                'total_cost': round(day['total_cost'], 4)
            })

        return daily_metrics

    def _calculate_performance_trend(self, hourly_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trend from hourly metrics."""
        if len(hourly_metrics) < 24:  # Need at least 24 hours for trend
            return {
                'overall_trend': 'insufficient_data',
                'recent_performance': 'unknown',
                'volatility': 'unknown'
            }

        # Get last 24 hours vs previous 24 hours
        recent_24h = hourly_metrics[:24]
        previous_24h = hourly_metrics[24:48] if len(hourly_metrics) >= 48 else []

        # Calculate recent performance
        recent_hours_with_decisions = [h for h in recent_24h if h['decisions_count'] > 0]
        recent_success_rate = sum(h['success_rate'] for h in recent_hours_with_decisions) / len(recent_hours_with_decisions) if recent_hours_with_decisions else 0.0

        # Calculate trend
        if previous_24h:
            previous_success_rate = sum(h['success_rate'] for h in previous_24h if h['decisions_count'] > 0) / len([h for h in previous_24h if h['decisions_count'] > 0])

            if recent_success_rate > previous_success_rate + 0.05:
                trend = 'improving'
            elif recent_success_rate < previous_success_rate - 0.05:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_historical_data'

        # Determine recent performance level
        if recent_success_rate >= 0.6:
            recent_perf = 'excellent'
        elif recent_success_rate >= 0.4:
            recent_perf = 'good'
        elif recent_success_rate >= 0.2:
            recent_perf = 'poor'
        else:
            recent_perf = 'very_poor'

        return {
            'overall_trend': trend,
            'recent_performance': recent_perf,
            'recent_success_rate': round(recent_success_rate, 3)
        }

    async def generate_dashboard_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate alerts for dashboard display.

        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []

            # Get current metrics for alert checking
            realtime_metrics = await self.get_realtime_metrics()
            cost_breakdown = await self.get_cost_breakdown()

            # Check performance alerts
            if realtime_metrics.get('success_rate', 1.0) < self.ALERT_THRESHOLDS['low_success_rate']:
                alerts.append({
                    'type': 'low_success_rate',
                    'severity': 'high',
                    'message': f"Success rate ({realtime_metrics['success_rate']:.1%}) below threshold",
                    'timestamp': datetime.now().isoformat(),
                    'current_value': realtime_metrics['success_rate'],
                    'threshold': self.ALERT_THRESHOLDS['low_success_rate']
                })

            # Check cost alerts
            if cost_breakdown.get('summary', {}):
                total_cost = cost_breakdown['summary']['total_cost']
                total_pnl = cost_breakdown['summary']['total_pnl']

                if total_pnl > 0 and total_cost / total_pnl > self.ALERT_THRESHOLDS['high_cost_ratio']:
                    alerts.append({
                        'type': 'high_cost_ratio',
                        'severity': 'medium',
                        'message': f"Cost ratio ({total_cost/total_pnl:.1%}) exceeds threshold",
                        'timestamp': datetime.now().isoformat(),
                        'current_value': total_cost / total_pnl,
                        'threshold': self.ALERT_THRESHOLDS['high_cost_ratio']
                    })

            # Check disagreement alerts
            if realtime_metrics.get('avg_disagreement', 0.0) > self.ALERT_THRESHOLDS['high_disagreement']:
                alerts.append({
                    'type': 'high_disagreement',
                    'severity': 'medium',
                    'message': f"High disagreement level ({realtime_metrics['avg_disagreement']:.2f}) detected",
                    'timestamp': datetime.now().isoformat(),
                    'current_value': realtime_metrics['avg_disagreement'],
                    'threshold': self.ALERT_THRESHOLDS['high_disagreement']
                })

            return alerts

        except Exception as e:
            self.logger.error("Failed to generate dashboard alerts", error=str(e))
            return []

    async def _check_performance_alerts(self, decision_data: Dict[str, Any]) -> None:
        """Check for performance-related alerts and log them."""
        try:
            # Check for consecutive failures
            if decision_data.get('outcome') == 'failure':
                consecutive_failures = await self._get_consecutive_failures()
                if consecutive_failures >= self.ALERT_THRESHOLDS['consecutive_failures']:
                    self.logger.warning(
                        "Consecutive failures alert",
                        count=consecutive_failures,
                        threshold=self.ALERT_THRESHOLDS['consecutive_failures']
                    )
        except Exception as e:
            self.logger.error("Failed to check performance alerts", error=str(e))

    async def _check_cost_alerts(self, cost_data: Dict[str, Any]) -> None:
        """Check for cost-related alerts and log them."""
        try:
            cost_usd = cost_data.get('cost_usd', 0.0)
            if cost_usd > 1.0:  # Alert if single decision costs more than $1
                self.logger.warning(
                    "High cost decision detected",
                    cost=cost_usd,
                    model=cost_data.get('model_name', 'unknown')
                )
        except Exception as e:
            self.logger.error("Failed to check cost alerts", error=str(e))

    async def _get_consecutive_failures(self) -> int:
        """Get number of consecutive failures."""
        try:
            # Get recent decisions
            recent_decisions = await self.db.get_ensemble_decisions(
                start_time=datetime.now() - timedelta(hours=24)
            )

            consecutive = 0
            for decision in recent_decisions:
                # This would check actual outcomes
                # For now, we'll return 0 as placeholder
                break

            return consecutive
        except Exception as e:
            self.logger.error("Failed to get consecutive failures", error=str(e))
            return 0

    # Additional methods for advanced analytics
    async def calculate_contribution_metrics(self) -> Dict[str, Any]:
        """Calculate detailed contribution metrics for all models."""
        try:
            contributions = await self.analyze_model_contributions()

            total_pnl = sum(contrib.get('total_pnl', 0) for contrib in contributions.values())
            total_cost = sum(contrib.get('avg_cost_per_decision', 0) * contrib.get('total_decisions', 0)
                           for contrib in contributions.values())

            metrics = {
                'total_decisions': sum(contrib.get('total_decisions', 0) for contrib in contributions.values()),
                'total_pnl': total_pnl,
                'total_cost': total_cost,
                'overall_success_rate': 0.0,  # Would calculate from actual outcomes
                'by_model': {}
            }

            # Calculate per-model metrics
            for model, contrib in contributions.items():
                model_total = contrib.get('total_pnl', 0)
                model_cost = contrib.get('avg_cost_per_decision', 0) * contrib.get('total_decisions', 0)

                metrics['by_model'][model] = {
                    'success_rate': contrib.get('success_rate', 0.0),
                    'avg_pnl_per_decision': model_total / contrib.get('total_decisions', 1),
                    'contribution_percentage': (model_total / total_pnl * 100) if total_pnl != 0 else 0,
                    'risk_adjusted_return': model_total / (contrib.get('total_decisions', 1)),
                    'cost_efficiency': model_total / model_cost if model_cost > 0 else 0
                }

            return metrics

        except Exception as e:
            self.logger.error("Failed to calculate contribution metrics", error=str(e))
            return {}

    async def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance degradation and generate alerts."""
        try:
            alerts = []
            metrics = await self.get_realtime_metrics()

            # Success rate alert
            if metrics.get('success_rate', 1.0) < self.ALERT_THRESHOLDS['low_success_rate']:
                alerts.append({
                    'type': 'low_success_rate',
                    'severity': 'high',
                    'message': f"Success rate ({metrics['success_rate']:.1%}) below threshold",
                    'current_value': metrics['success_rate'],
                    'threshold': self.ALERT_THRESHOLDS['low_success_rate'],
                    'timestamp': datetime.now().isoformat()
                })

            # Consecutive failures alert
            consecutive_failures = await self._get_consecutive_failures()
            if consecutive_failures >= self.ALERT_THRESHOLDS['consecutive_failures']:
                alerts.append({
                    'type': 'consecutive_failures',
                    'severity': 'critical',
                    'message': f"{consecutive_failures} consecutive failures detected",
                    'current_value': consecutive_failures,
                    'threshold': self.ALERT_THRESHOLDS['consecutive_failures'],
                    'timestamp': datetime.now().isoformat()
                })

            return alerts

        except Exception as e:
            self.logger.error("Failed to check performance alerts", error=str(e))
            return []

    async def analyze_disagreement_correlation(self) -> Dict[str, Any]:
        """Detailed analysis of disagreement correlation with decision quality."""
        try:
            decisions = await self.db.get_ensemble_decisions(
                start_time=datetime.now() - timedelta(days=30)
            )

            if len(decisions) < 20:
                return {
                    'correlation_coefficient': 0.0,
                    'sample_size': len(decisions),
                    'significance': 'insufficient_data',
                    'agreement_buckets': {}
                }

            # Bucket by disagreement level
            buckets = {
                'low_disagreement': [],    # < 0.33
                'medium_disagreement': [], # 0.33 - 0.67
                'high_disagreement': []     # > 0.67
            }

            for decision in decisions:
                level = decision.disagreement_level
                if level < 0.33:
                    buckets['low_disagreement'].append(decision)
                elif level <= 0.67:
                    buckets['medium_disagreement'].append(decision)
                else:
                    buckets['high_disagreement'].append(decision)

            # Calculate metrics for each bucket
            bucket_metrics = {}
            for bucket_name, bucket_decisions in buckets.items():
                if bucket_decisions:
                    bucket_metrics[bucket_name] = self._calculate_bucket_metrics(bucket_decisions)
                else:
                    bucket_metrics[bucket_name] = {
                        'count': 0,
                        'success_rate': 0.0,
                        'avg_pnl': 0.0
                    }

            return {
                'correlation_coefficient': 0.0,  # Would calculate actual correlation
                'sample_size': len(decisions),
                'significance': 'calculated',
                'agreement_buckets': bucket_metrics,
                'recommendations': self._generate_disagreement_recommendations(bucket_metrics)
            }

        except Exception as e:
            self.logger.error("Failed to analyze disagreement correlation", error=str(e))
            return {}

    def _calculate_bucket_metrics(self, decisions: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for a disagreement bucket."""
        count = len(decisions)
        if count == 0:
            return {'count': 0, 'success_rate': 0.0, 'avg_pnl': 0.0}

        # This would use actual outcome and PnL data
        # For now, returning placeholder values
        return {
            'count': count,
            'success_rate': 0.0,  # Would calculate from actual outcomes
            'avg_pnl': 0.0,       # Would calculate from actual PnL
            'avg_disagreement': sum(d.disagreement_level for d in decisions) / count
        }

    def _generate_disagreement_recommendations(self, bucket_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on disagreement analysis."""
        recommendations = []

        low_disag = bucket_metrics.get('low_disagreement', {})
        high_disag = bucket_metrics.get('high_disagreement', {})

        if low_disag.get('success_rate', 0) > high_disag.get('success_rate', 0):
            recommendations.append("Consider prioritizing consensus trades for better success rates")

        if high_disag.get('count', 0) > 0:
            recommendations.append("Review high-disagreement cases for uncertainty reduction opportunities")

        return recommendations