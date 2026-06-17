"""
Pipeline Monitor Module
Monitor data pipelines and track performance metrics.
Provides real-time monitoring and alerting for pipeline health.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum
import time


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"


class PipelineMonitor:
    """
    Monitor data pipeline jobs and track performance.
    
    Key Features:
    - Job execution tracking
    - Performance metrics collection
    - SLA monitoring
    - Alert generation
    """
    
    def __init__(self):
        """Initialize pipeline monitor."""
        self.jobs = {}
        self.metrics = []
        self.alerts = []
        self.sla_thresholds = {}
    
    def register_job(self, job_id: str, job_name: str, expected_duration_minutes: int = 60) -> None:
        """
        Register a pipeline job for monitoring.
        
        Args:
            job_id: Unique job identifier
            job_name: Human-readable job name
            expected_duration_minutes: Expected job duration in minutes
        """
        self.jobs[job_id] = {
            'job_name': job_name,
            'status': JobStatus.PENDING.value,
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'expected_duration_minutes': expected_duration_minutes,
            'records_processed': 0,
            'records_failed': 0,
            'error_message': None
        }
        
        self.sla_thresholds[job_id] = expected_duration_minutes * 60
    
    def start_job(self, job_id: str) -> None:
        """
        Mark job as started.
        
        Args:
            job_id: Job identifier
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not registered")
        
        self.jobs[job_id]['status'] = JobStatus.RUNNING.value
        self.jobs[job_id]['start_time'] = datetime.now()
    
    def complete_job(self, job_id: str, records_processed: int = 0,
                    records_failed: int = 0, success: bool = True) -> None:
        """
        Mark job as completed.
        
        Args:
            job_id: Job identifier
            records_processed: Number of records processed
            records_failed: Number of records that failed
            success: Whether job succeeded
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not registered")
        
        job = self.jobs[job_id]
        job['end_time'] = datetime.now()
        job['records_processed'] = records_processed
        job['records_failed'] = records_failed
        
        if job['start_time']:
            job['duration_seconds'] = (job['end_time'] - job['start_time']).total_seconds()
        
        if success:
            job['status'] = JobStatus.COMPLETED.value
        else:
            job['status'] = JobStatus.FAILED.value
        
        # Check SLA
        self._check_sla(job_id)
        
        # Record metrics
        self._record_metrics(job_id)
    
    def log_error(self, job_id: str, error_message: str) -> None:
        """
        Log an error for a job.
        
        Args:
            job_id: Job identifier
            error_message: Error message
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not registered")
        
        self.jobs[job_id]['error_message'] = error_message
        self.jobs[job_id]['status'] = JobStatus.FAILED.value
    
    def _check_sla(self, job_id: str) -> None:
        """
        Check if job meets SLA requirements.
        """
        job = self.jobs[job_id]
        sla_threshold = self.sla_thresholds[job_id]
        
        if job['duration_seconds'] > sla_threshold:
            alert = {
                'timestamp': datetime.now(),
                'job_id': job_id,
                'alert_type': 'SLA_BREACH',
                'message': f"Job {job_id} exceeded SLA by {job['duration_seconds'] - sla_threshold:.0f} seconds"
            }
            self.alerts.append(alert)
            job['status'] = JobStatus.WARNING.value
    
    def _record_metrics(self, job_id: str) -> None:
        """
        Record job metrics.
        """
        job = self.jobs[job_id]
        
        metric = {
            'timestamp': datetime.now(),
            'job_id': job_id,
            'job_name': job['job_name'],
            'status': job['status'],
            'duration_seconds': job['duration_seconds'],
            'records_processed': job['records_processed'],
            'records_failed': job['records_failed'],
            'success_rate': self._calculate_success_rate(job)
        }
        
        self.metrics.append(metric)
    
    def _calculate_success_rate(self, job: Dict[str, Any]) -> float:
        """
        Calculate job success rate.
        """
        total = job['records_processed'] + job['records_failed']
        if total == 0:
            return 100.0
        return round((job['records_processed'] / total) * 100, 2)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not registered")
        
        return self.jobs[job_id]
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """
        Get overall pipeline health metrics.
        
        Returns:
            Pipeline health report
        """
        if not self.metrics:
            return {'status': 'no_data'}
        
        df = pd.DataFrame(self.metrics)
        
        health = {
            'total_jobs': len(df),
            'completed_jobs': len(df[df['status'] == JobStatus.COMPLETED.value]),
            'failed_jobs': len(df[df['status'] == JobStatus.FAILED.value]),
            'warning_jobs': len(df[df['status'] == JobStatus.WARNING.value]),
            'overall_success_rate': round(df['success_rate'].mean(), 2),
            'avg_duration_seconds': round(df['duration_seconds'].mean(), 2),
            'total_records_processed': int(df['records_processed'].sum()),
            'total_records_failed': int(df['records_failed'].sum()),
            'alerts_count': len(self.alerts)
        }
        
        return health
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
        return recent_alerts
    
    def print_pipeline_report(self) -> None:
        """
        Print pipeline monitoring report.
        """
        health = self.get_pipeline_health()
        
        if health.get('status') == 'no_data':
            print("No pipeline metrics available.")
            return
        
        print("\n" + "="*80)
        print("PIPELINE MONITORING REPORT")
        print("="*80 + "\n")
        
        print(f"Total Jobs: {health['total_jobs']}")
        print(f"Completed: {health['completed_jobs']}")
        print(f"Failed: {health['failed_jobs']}")
        print(f"Warnings: {health['warning_jobs']}")
        print(f"\nOverall Success Rate: {health['overall_success_rate']}%")
        print(f"Average Job Duration: {health['avg_duration_seconds']:.0f} seconds")
        print(f"\nRecords Processed: {health['total_records_processed']}")
        print(f"Records Failed: {health['total_records_failed']}")
        print(f"Alerts: {health['alerts_count']}")
        
        if health['alerts_count'] > 0:
            print("\nRecent Alerts:")
            for alert in self.get_alerts(hours=24):
                print(f"  - {alert['alert_type']}: {alert['message']}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    monitor = PipelineMonitor()
    
    # Register jobs
    monitor.register_job('etl_001', 'Daily ETL Job', expected_duration_minutes=30)
    monitor.register_job('dq_check_001', 'Data Quality Check', expected_duration_minutes=15)
    
    # Simulate job execution
    monitor.start_job('etl_001')
    time.sleep(2)  # Simulate processing
    monitor.complete_job('etl_001', records_processed=10000, records_failed=5, success=True)
    
    monitor.start_job('dq_check_001')
    time.sleep(1)
    monitor.complete_job('dq_check_001', records_processed=10005, records_failed=0, success=True)
    
    # Print report
    monitor.print_pipeline_report()
