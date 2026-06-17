"""
Auto Scaler Module
Automatically scale pipeline resources based on demand and predictions.
Optimizes compute resources and prevents bottlenecks.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


class AutoScaler:
    """
    Automatically scale pipeline resources.
    
    Key Features:
    - Demand prediction
    - Resource optimization
    - Cost awareness
    - Auto-scaling recommendations
    """
    
    def __init__(self, min_workers: int = 1, max_workers: int = 10,
                 cost_per_worker_hour: float = 1.0):
        """
        Initialize auto scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            cost_per_worker_hour: Cost per worker per hour
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cost_per_worker_hour = cost_per_worker_hour
        self.scaling_history = []
        self.thresholds = {
            'cpu_utilization': 80,  # Scale up if CPU > 80%
            'memory_utilization': 85,  # Scale up if memory > 85%
            'queue_depth': 100  # Scale up if queued items > 100
        }
    
    def predict_demand(self, historical_data: pd.DataFrame,
                      forecast_hours: int = 24) -> List[float]:
        """
        Predict resource demand for next N hours.
        
        Args:
            historical_data: Historical load data
            forecast_hours: Number of hours to forecast
            
        Returns:
            List of predicted demands
        """
        if len(historical_data) < 2:
            return [self.min_workers] * forecast_hours
        
        # Simple moving average forecast
        avg_load = historical_data['load'].mean()
        std_load = historical_data['load'].std()
        
        # Generate forecast with trend
        trend = (historical_data['load'].iloc[-1] - historical_data['load'].iloc[0]) / len(historical_data)
        
        forecast = []
        for i in range(forecast_hours):
            predicted_load = avg_load + (trend * i)
            forecast.append(max(self.min_workers, min(self.max_workers, predicted_load)))
        
        return forecast
    
    def calculate_optimal_workers(self, metrics: Dict[str, float]) -> int:
        """
        Calculate optimal number of workers based on current metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Recommended number of workers
        """
        workers = self.min_workers
        
        # Scale based on CPU utilization
        if metrics.get('cpu_utilization', 0) > self.thresholds['cpu_utilization']:
            cpu_scale = metrics['cpu_utilization'] / 100
            workers = max(workers, int(self.max_workers * cpu_scale))
        
        # Scale based on memory utilization
        if metrics.get('memory_utilization', 0) > self.thresholds['memory_utilization']:
            mem_scale = metrics['memory_utilization'] / 100
            workers = max(workers, int(self.max_workers * mem_scale))
        
        # Scale based on queue depth
        if metrics.get('queue_depth', 0) > self.thresholds['queue_depth']:
            queue_scale = min(1.0, metrics['queue_depth'] / 1000)
            workers = max(workers, int(self.max_workers * queue_scale))
        
        # Ensure within bounds
        workers = max(self.min_workers, min(self.max_workers, workers))
        
        return workers
    
    def get_scaling_recommendation(self, current_workers: int,
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get scaling recommendation based on metrics.
        
        Args:
            current_workers: Current number of workers
            metrics: Performance metrics
            
        Returns:
            Scaling recommendation
        """
        optimal_workers = self.calculate_optimal_workers(metrics)
        
        recommendation = {
            'timestamp': datetime.now(),
            'current_workers': current_workers,
            'recommended_workers': optimal_workers,
            'action': 'none',
            'reason': 'Resource utilization is optimal',
            'estimated_cost_per_hour': optimal_workers * self.cost_per_worker_hour
        }
        
        if optimal_workers > current_workers:
            workers_to_add = optimal_workers - current_workers
            recommendation['action'] = 'scale_up'
            recommendation['reason'] = f'High resource utilization detected'
            recommendation['workers_to_add'] = workers_to_add
        elif optimal_workers < current_workers:
            workers_to_remove = current_workers - optimal_workers
            recommendation['action'] = 'scale_down'
            recommendation['reason'] = f'Low resource utilization detected'
            recommendation['workers_to_remove'] = workers_to_remove
        
        # Calculate cost savings
        if recommendation['action'] == 'scale_down':
            current_cost = current_workers * self.cost_per_worker_hour
            recommended_cost = optimal_workers * self.cost_per_worker_hour
            recommendation['cost_savings_per_hour'] = current_cost - recommended_cost
        
        self.scaling_history.append(recommendation)
        return recommendation
    
    def estimate_cost(self, workers: int, duration_hours: float) -> float:
        """
        Estimate cost for given worker count and duration.
        
        Args:
            workers: Number of workers
            duration_hours: Duration in hours
            
        Returns:
            Estimated cost
        """
        return workers * self.cost_per_worker_hour * duration_hours
    
    def get_cost_optimization_report(self) -> Dict[str, Any]:
        """
        Generate cost optimization report.
        
        Returns:
            Cost analysis report
        """
        if not self.scaling_history:
            return {'report': 'No scaling history available'}
        
        df = pd.DataFrame(self.scaling_history)
        
        total_current_cost = (df['current_workers'] * self.cost_per_worker_hour).sum()
        total_recommended_cost = (df['recommended_workers'] * self.cost_per_worker_hour).sum()
        
        report = {
            'total_scaling_actions': len(df),
            'scale_up_actions': len(df[df['action'] == 'scale_up']),
            'scale_down_actions': len(df[df['action'] == 'scale_down']),
            'avg_workers_current': round(df['current_workers'].mean(), 2),
            'avg_workers_recommended': round(df['recommended_workers'].mean(), 2),
            'total_cost_current': round(total_current_cost, 2),
            'total_cost_recommended': round(total_recommended_cost, 2),
            'potential_savings': round(total_current_cost - total_recommended_cost, 2),
            'savings_percentage': round(((total_current_cost - total_recommended_cost) / total_current_cost * 100), 2) if total_current_cost > 0 else 0
        }
        
        return report
    
    def print_scaling_report(self, recommendation: Dict[str, Any] = None) -> None:
        """
        Print scaling recommendation report.
        
        Args:
            recommendation: Scaling recommendation (optional)
        """
        print("\n" + "="*80)
        print("AUTO-SCALING RECOMMENDATION REPORT")
        print("="*80 + "\n")
        
        if recommendation:
            print(f"Timestamp: {recommendation['timestamp']}")
            print(f"Current Workers: {recommendation['current_workers']}")
            print(f"Recommended Workers: {recommendation['recommended_workers']}")
            print(f"Action: {recommendation['action'].upper()}")
            print(f"Reason: {recommendation['reason']}")
            print(f"Estimated Cost/Hour: ${recommendation['estimated_cost_per_hour']:.2f}")
            
            if 'workers_to_add' in recommendation:
                print(f"Workers to Add: {recommendation['workers_to_add']}")
            elif 'workers_to_remove' in recommendation:
                print(f"Workers to Remove: {recommendation['workers_to_remove']}")
            
            if 'cost_savings_per_hour' in recommendation:
                print(f"Cost Savings/Hour: ${recommendation['cost_savings_per_hour']:.2f}")
        
        cost_report = self.get_cost_optimization_report()
        if cost_report.get('report'):
            print(f"\n{cost_report['report']}")
        else:
            print("\nCost Optimization Summary:")
            print(f"  Total Scaling Actions: {cost_report['total_scaling_actions']}")
            print(f"  Scale Up: {cost_report['scale_up_actions']}, Scale Down: {cost_report['scale_down_actions']}")
            print(f"  Potential Savings: ${cost_report['potential_savings']:.2f} ({cost_report['savings_percentage']:.1f}%)")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    scaler = AutoScaler(min_workers=2, max_workers=20, cost_per_worker_hour=2.0)
    
    # Get scaling recommendation
    metrics = {
        'cpu_utilization': 85,
        'memory_utilization': 75,
        'queue_depth': 150
    }
    
    recommendation = scaler.get_scaling_recommendation(current_workers=5, metrics=metrics)
    scaler.print_scaling_report(recommendation)
    
    # Estimate cost
    cost = scaler.estimate_cost(workers=10, duration_hours=24)
    print(f"\nEstimated cost for 10 workers over 24 hours: ${cost:.2f}")
