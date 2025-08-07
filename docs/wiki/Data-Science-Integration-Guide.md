# 📊 Data Science Integration Guide

Welcome, Data Scientists! This guide will help you integrate with SAMO Brain's AI capabilities for advanced analytics, model monitoring, and research collaboration.

## 🚀 **Quick Start (5 minutes)**

### **1. Test the API**
```python
import requests
import pandas as pd

# Test emotion detection
response = requests.post('http://localhost:8000/predict', 
    json={'text': 'I am feeling happy today!'})
result = response.json()

print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### **2. Get Model Metrics**
```python
# Get detailed metrics
metrics = requests.get('http://localhost:8000/metrics').json()
print(f"Success Rate: {metrics['server_metrics']['success_rate']}")
print(f"Average Response Time: {metrics['server_metrics']['average_response_time_ms']}ms")
```

---

## 📈 **Model Performance Analytics**

### **Performance Monitoring Dashboard**

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

class SAMOBrainAnalytics:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return None
    
    def analyze_emotion_distribution(self, texts):
        """Analyze emotion distribution across a dataset."""
        results = []
        
        for text in texts:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json={'text': text}
                )
                response.raise_for_status()
                result = response.json()
                results.append({
                    'text': text,
                    'emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
            except Exception as e:
                print(f"Error analyzing text: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_performance_report(self):
        """Generate comprehensive performance report."""
        metrics = self.get_performance_metrics()
        if not metrics:
            return None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'server_metrics': metrics['server_metrics'],
            'emotion_distribution': metrics['emotion_distribution'],
            'error_analysis': metrics.get('error_counts', {}),
            'rate_limiting': metrics.get('rate_limiting', {})
        }
        
        return report
    
    def plot_emotion_distribution(self, df):
        """Plot emotion distribution from analysis results."""
        plt.figure(figsize=(12, 6))
        
        # Emotion frequency
        emotion_counts = df['emotion'].value_counts()
        
        plt.subplot(1, 2, 1)
        emotion_counts.plot(kind='bar', color='skyblue')
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Confidence distribution
        plt.subplot(1, 2, 2)
        plt.hist(df['confidence'], bins=20, alpha=0.7, color='lightgreen')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_confidence_trends(self, df):
        """Analyze confidence trends by emotion."""
        confidence_by_emotion = df.groupby('emotion')['confidence'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).round(3)
        
        print("Confidence Analysis by Emotion:")
        print(confidence_by_emotion)
        
        return confidence_by_emotion

# Usage example
analytics = SAMOBrainAnalytics()

# Get performance metrics
metrics = analytics.get_performance_metrics()
if metrics:
    print(f"Success Rate: {metrics['server_metrics']['success_rate']}")
    print(f"Average Response Time: {metrics['server_metrics']['average_response_time_ms']}ms")

# Analyze sample dataset
sample_texts = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party",
    "I feel anxious about the presentation",
    "I am grateful for your help"
]

df = analytics.analyze_emotion_distribution(sample_texts)
analytics.plot_emotion_distribution(df)
analytics.analyze_confidence_trends(df)
```

---

## 🔍 **Model Drift Detection**

### **Performance Monitoring System**

```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ModelDriftDetector:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.baseline_metrics = None
        self.drift_threshold = 0.1  # 10% change threshold
    
    def establish_baseline(self, test_texts, n_runs=10):
        """Establish baseline performance metrics."""
        print("Establishing baseline metrics...")
        
        baseline_results = []
        for _ in range(n_runs):
            run_results = []
            for text in test_texts:
                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json={'text': text}
                    )
                    response.raise_for_status()
                    result = response.json()
                    run_results.append({
                        'emotion': result['predicted_emotion'],
                        'confidence': result['confidence'],
                        'response_time': result['prediction_time_ms']
                    })
                except Exception as e:
                    print(f"Error in baseline run: {e}")
                    continue
            
            if run_results:
                baseline_results.append(run_results)
        
        # Calculate baseline metrics
        self.baseline_metrics = {
            'avg_confidence': np.mean([r['confidence'] for run in baseline_results for r in run]),
            'confidence_std': np.std([r['confidence'] for run in baseline_results for r in run]),
            'avg_response_time': np.mean([r['response_time'] for run in baseline_results for r in run]),
            'emotion_distribution': pd.DataFrame([r['emotion'] for run in baseline_results for r in run]).value_counts().to_dict()
        }
        
        print(f"Baseline established:")
        print(f"  Average Confidence: {self.baseline_metrics['avg_confidence']:.3f}")
        print(f"  Average Response Time: {self.baseline_metrics['avg_response_time']:.1f}ms")
        
        return self.baseline_metrics
    
    def detect_drift(self, test_texts, n_runs=5):
        """Detect model drift by comparing current performance to baseline."""
        if self.baseline_metrics is None:
            print("Please establish baseline first using establish_baseline()")
            return None
        
        print("Detecting model drift...")
        
        current_results = []
        for _ in range(n_runs):
            run_results = []
            for text in test_texts:
                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json={'text': text}
                    )
                    response.raise_for_status()
                    result = response.json()
                    run_results.append({
                        'emotion': result['predicted_emotion'],
                        'confidence': result['confidence'],
                        'response_time': result['prediction_time_ms']
                    })
                except Exception as e:
                    print(f"Error in drift detection run: {e}")
                    continue
            
            if run_results:
                current_results.append(run_results)
        
        # Calculate current metrics
        current_metrics = {
            'avg_confidence': np.mean([r['confidence'] for run in current_results for r in run]),
            'confidence_std': np.std([r['confidence'] for run in current_results for r in run]),
            'avg_response_time': np.mean([r['response_time'] for run in current_results for r in run]),
            'emotion_distribution': pd.DataFrame([r['emotion'] for run in current_results for r in run]).value_counts().to_dict()
        }
        
        # Detect drift
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'confidence_drift': (current_metrics['avg_confidence'] - self.baseline_metrics['avg_confidence']) / self.baseline_metrics['avg_confidence'],
            'response_time_drift': (current_metrics['avg_response_time'] - self.baseline_metrics['avg_response_time']) / self.baseline_metrics['avg_response_time'],
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': current_metrics,
            'drift_detected': False,
            'drift_severity': 'none'
        }
        
        # Check for significant drift
        confidence_drift_abs = abs(drift_report['confidence_drift'])
        response_time_drift_abs = abs(drift_report['response_time_drift'])
        
        if confidence_drift_abs > self.drift_threshold or response_time_drift_abs > self.drift_threshold:
            drift_report['drift_detected'] = True
            
            if max(confidence_drift_abs, response_time_drift_abs) > 0.2:
                drift_report['drift_severity'] = 'high'
            elif max(confidence_drift_abs, response_time_drift_abs) > 0.1:
                drift_report['drift_severity'] = 'medium'
            else:
                drift_report['drift_severity'] = 'low'
        
        return drift_report
    
    def generate_drift_alert(self, drift_report):
        """Generate alert based on drift detection."""
        if not drift_report['drift_detected']:
            print("✅ No significant drift detected")
            return
        
        print(f"🚨 MODEL DRIFT DETECTED - Severity: {drift_report['drift_severity'].upper()}")
        print(f"   Confidence Drift: {drift_report['confidence_drift']:.1%}")
        print(f"   Response Time Drift: {drift_report['response_time_drift']:.1%}")
        print(f"   Timestamp: {drift_report['timestamp']}")
        
        if drift_report['drift_severity'] == 'high':
            print("   🔴 IMMEDIATE ACTION REQUIRED: Consider model retraining")
        elif drift_report['drift_severity'] == 'medium':
            print("   🟡 MONITOR CLOSELY: Investigate potential causes")
        else:
            print("   🟢 MINOR DRIFT: Continue monitoring")

# Usage example
detector = ModelDriftDetector()

# Test texts for drift detection
test_texts = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party",
    "I feel anxious about the presentation",
    "I am grateful for your help",
    "I am proud of my achievements",
    "I feel overwhelmed with work",
    "I am hopeful about the future"
]

# Establish baseline
baseline = detector.establish_baseline(test_texts)

# Detect drift
drift_report = detector.detect_drift(test_texts)
detector.generate_drift_alert(drift_report)
```

---

## 📊 **Advanced Analytics**

### **Emotion Pattern Analysis**

```python
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionPatternAnalyzer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_emotion_patterns(self, texts, labels=None):
        """Analyze emotion patterns in a dataset."""
        results = []
        
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={'text': text}
                )
                response.raise_for_status()
                result = response.json()
                
                results.append({
                    'text': text,
                    'emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'label': labels[i] if labels else None
                })
            except Exception as e:
                print(f"Error analyzing text {i}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def extract_emotion_features(self, df):
        """Extract numerical features from emotion probabilities."""
        emotion_cols = ['anxious', 'calm', 'content', 'excited', 'frustrated', 
                       'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        
        features = []
        for _, row in df.iterrows():
            feature_vector = [row['probabilities'].get(emotion, 0) for emotion in emotion_cols]
            features.append(feature_vector)
        
        return np.array(features), emotion_cols
    
    def cluster_emotion_patterns(self, df, n_clusters=3):
        """Cluster texts based on emotion patterns."""
        features, emotion_cols = self.extract_emotion_features(df)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # PCA plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis')
        plt.title('Emotion Pattern Clusters (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter)
        
        # Cluster analysis
        plt.subplot(2, 2, 2)
        cluster_emotions = []
        for cluster_id in range(n_clusters):
            cluster_texts = df[clusters == cluster_id]
            cluster_emotions.append(cluster_texts['emotion'].value_counts())
        
        cluster_df = pd.DataFrame(cluster_emotions).fillna(0)
        cluster_df.plot(kind='bar', ax=plt.gca())
        plt.title('Emotion Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Confidence distribution
        plt.subplot(2, 2, 3)
        for cluster_id in range(n_clusters):
            cluster_confidences = df[clusters == cluster_id]['confidence']
            plt.hist(cluster_confidences, alpha=0.7, label=f'Cluster {cluster_id}')
        plt.title('Confidence Distribution by Cluster')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Emotion correlation heatmap
        plt.subplot(2, 2, 4)
        features_df = pd.DataFrame(features, columns=emotion_cols)
        correlation_matrix = features_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Emotion Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        return clusters, features_2d
    
    def analyze_emotion_transitions(self, emotion_sequence):
        """Analyze emotion transitions in a sequence."""
        if len(emotion_sequence) < 2:
            return None
        
        transitions = []
        for i in range(len(emotion_sequence) - 1):
            transitions.append((emotion_sequence[i], emotion_sequence[i + 1]))
        
        transition_matrix = pd.crosstab(
            pd.Series([t[0] for t in transitions]),
            pd.Series([t[1] for t in transitions]),
            normalize='index'
        )
        
        # Visualize transition matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Emotion Transition Matrix')
        plt.xlabel('Next Emotion')
        plt.ylabel('Current Emotion')
        plt.show()
        
        return transition_matrix

# Usage example
analyzer = EmotionPatternAnalyzer()

# Sample dataset
sample_texts = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party",
    "I feel anxious about the presentation",
    "I am grateful for your help",
    "I am proud of my achievements",
    "I feel overwhelmed with work",
    "I am hopeful about the future",
    "I am tired after a long day",
    "I feel content with my life"
]

# Analyze patterns
df = analyzer.analyze_emotion_patterns(sample_texts)
clusters, features_2d = analyzer.cluster_emotion_patterns(df, n_clusters=3)

# Analyze transitions
emotion_sequence = df['emotion'].tolist()
transition_matrix = analyzer.analyze_emotion_transitions(emotion_sequence)
```

---

## 🔬 **Research Collaboration**

### **Experimental Model Testing**

```python
import requests
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ExperimentalModelTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def a_b_test_models(self, test_texts, model_a_config, model_b_config, n_runs=10):
        """Perform A/B testing between different model configurations."""
        print("Running A/B test...")
        
        results_a = []
        results_b = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            
            # Test Model A
            for text in test_texts:
                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json={'text': text, **model_a_config}
                    )
                    response.raise_for_status()
                    result = response.json()
                    results_a.append({
                        'emotion': result['predicted_emotion'],
                        'confidence': result['confidence'],
                        'response_time': result['prediction_time_ms']
                    })
                except Exception as e:
                    print(f"Error testing Model A: {e}")
            
            # Test Model B
            for text in test_texts:
                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json={'text': text, **model_b_config}
                    )
                    response.raise_for_status()
                    result = response.json()
                    results_b.append({
                        'emotion': result['predicted_emotion'],
                        'confidence': result['confidence'],
                        'response_time': result['prediction_time_ms']
                    })
                except Exception as e:
                    print(f"Error testing Model B: {e}")
        
        return pd.DataFrame(results_a), pd.DataFrame(results_b)
    
    def statistical_comparison(self, df_a, df_b):
        """Perform statistical comparison between two models."""
        print("Performing statistical comparison...")
        
        # Confidence comparison
        confidence_a = df_a['confidence']
        confidence_b = df_b['confidence']
        
        # T-test for confidence
        t_stat, p_value = stats.ttest_ind(confidence_a, confidence_b)
        
        # Response time comparison
        response_time_a = df_a['response_time']
        response_time_b = df_b['response_time']
        
        # Mann-Whitney U test for response time (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(response_time_a, response_time_b, alternative='two-sided')
        
        # Create comparison report
        comparison_report = {
            'confidence_comparison': {
                'model_a_mean': confidence_a.mean(),
                'model_b_mean': confidence_b.mean(),
                'difference': confidence_b.mean() - confidence_a.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'response_time_comparison': {
                'model_a_mean': response_time_a.mean(),
                'model_b_mean': response_time_b.mean(),
                'difference': response_time_b.mean() - response_time_a.mean(),
                'u_statistic': u_stat,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }
        }
        
        return comparison_report
    
    def visualize_comparison(self, df_a, df_b, comparison_report):
        """Visualize A/B test results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence comparison
        axes[0, 0].hist(df_a['confidence'], alpha=0.7, label='Model A', bins=20)
        axes[0, 0].hist(df_b['confidence'], alpha=0.7, label='Model B', bins=20)
        axes[0, 0].set_title('Confidence Distribution Comparison')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Response time comparison
        axes[0, 1].hist(df_a['response_time'], alpha=0.7, label='Model A', bins=20)
        axes[0, 1].hist(df_b['response_time'], alpha=0.7, label='Model B', bins=20)
        axes[0, 1].set_title('Response Time Distribution Comparison')
        axes[0, 1].set_xlabel('Response Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Box plots
        axes[1, 0].boxplot([df_a['confidence'], df_b['confidence']], labels=['Model A', 'Model B'])
        axes[1, 0].set_title('Confidence Box Plot')
        axes[1, 0].set_ylabel('Confidence')
        
        axes[1, 1].boxplot([df_a['response_time'], df_b['response_time']], labels=['Model A', 'Model B'])
        axes[1, 1].set_title('Response Time Box Plot')
        axes[1, 1].set_ylabel('Response Time (ms)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical results
        print("\n=== STATISTICAL COMPARISON RESULTS ===")
        print(f"Confidence Comparison:")
        print(f"  Model A Mean: {comparison_report['confidence_comparison']['model_a_mean']:.3f}")
        print(f"  Model B Mean: {comparison_report['confidence_comparison']['model_b_mean']:.3f}")
        print(f"  Difference: {comparison_report['confidence_comparison']['difference']:.3f}")
        print(f"  P-value: {comparison_report['confidence_comparison']['p_value']:.4f}")
        print(f"  Significant: {comparison_report['confidence_comparison']['significant']}")
        
        print(f"\nResponse Time Comparison:")
        print(f"  Model A Mean: {comparison_report['response_time_comparison']['model_a_mean']:.1f}ms")
        print(f"  Model B Mean: {comparison_report['response_time_comparison']['model_b_mean']:.1f}ms")
        print(f"  Difference: {comparison_report['response_time_comparison']['difference']:.1f}ms")
        print(f"  P-value: {comparison_report['response_time_comparison']['p_value']:.4f}")
        print(f"  Significant: {comparison_report['response_time_comparison']['significant']}")

# Usage example
tester = ExperimentalModelTester()

# Test configurations
model_a_config = {'threshold': 0.5}
model_b_config = {'threshold': 0.7}

# Test texts
test_texts = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party",
    "I feel anxious about the presentation",
    "I am grateful for your help"
]

# Run A/B test
df_a, df_b = tester.a_b_test_models(test_texts, model_a_config, model_b_config, n_runs=5)

# Compare results
comparison_report = tester.statistical_comparison(df_a, df_b)
tester.visualize_comparison(df_a, df_b, comparison_report)
```

---

## 📊 **Data Export & Integration**

### **Comprehensive Data Export**

```python
import requests
import pandas as pd
import json
from datetime import datetime
import sqlite3
import sqlalchemy

class SAMOBrainDataExporter:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def export_emotion_data(self, texts, labels=None, metadata=None):
        """Export comprehensive emotion analysis data."""
        results = []
        
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={'text': text}
                )
                response.raise_for_status()
                result = response.json()
                
                # Flatten probabilities
                probabilities = result['probabilities']
                
                export_row = {
                    'text_id': i,
                    'text': text,
                    'predicted_emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'prediction_time_ms': result['prediction_time_ms'],
                    'model_version': result['model_version'],
                    'timestamp': datetime.now().isoformat(),
                    **probabilities
                }
                
                if labels:
                    export_row['true_label'] = labels[i]
                
                if metadata:
                    export_row.update(metadata[i] if isinstance(metadata, list) else metadata)
                
                results.append(export_row)
                
            except Exception as e:
                print(f"Error exporting data for text {i}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def export_to_csv(self, df, filename):
        """Export data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
    
    def export_to_json(self, df, filename):
        """Export data to JSON file."""
        df.to_json(filename, orient='records', indent=2)
        print(f"Data exported to {filename}")
    
    def export_to_sqlite(self, df, db_path, table_name='emotion_analysis'):
        """Export data to SQLite database."""
        engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Data exported to SQLite table '{table_name}' in {db_path}")
    
    def export_to_postgres(self, df, connection_string, table_name='emotion_analysis'):
        """Export data to PostgreSQL database."""
        engine = sqlalchemy.create_engine(connection_string)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Data exported to PostgreSQL table '{table_name}'")
    
    def get_model_metrics_export(self):
        """Export model performance metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            metrics = response.json()
            
            # Flatten metrics for export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': metrics['server_metrics']['uptime_seconds'],
                'total_requests': metrics['server_metrics']['total_requests'],
                'successful_requests': metrics['server_metrics']['successful_requests'],
                'failed_requests': metrics['server_metrics']['failed_requests'],
                'success_rate': metrics['server_metrics']['success_rate'],
                'average_response_time_ms': metrics['server_metrics']['average_response_time_ms'],
                'requests_per_minute': metrics['server_metrics']['requests_per_minute'],
                **metrics['emotion_distribution']
            }
            
            return pd.DataFrame([export_data])
            
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return None

# Usage example
exporter = SAMOBrainDataExporter()

# Sample data
texts = [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party"
]

labels = ['happy', 'sad', 'excited']

metadata = [
    {'user_id': 'user1', 'session_id': 'session1'},
    {'user_id': 'user2', 'session_id': 'session1'},
    {'user_id': 'user1', 'session_id': 'session2'}
]

# Export data
df = exporter.export_emotion_data(texts, labels, metadata)

# Export to different formats
exporter.export_to_csv(df, 'emotion_analysis.csv')
exporter.export_to_json(df, 'emotion_analysis.json')
exporter.export_to_sqlite(df, 'emotion_data.db')

# Export metrics
metrics_df = exporter.get_model_metrics_export()
if metrics_df is not None:
    exporter.export_to_csv(metrics_df, 'model_metrics.csv')
```

---

## 🔄 **Feedback Loop Integration**

### **Model Feedback System**

```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json

class ModelFeedbackSystem:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.feedback_data = []
    
    def collect_feedback(self, text, predicted_emotion, user_feedback, confidence=None):
        """Collect user feedback on model predictions."""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'predicted_emotion': predicted_emotion,
            'user_feedback': user_feedback,
            'confidence': confidence,
            'feedback_type': 'user_correction'
        }
        
        self.feedback_data.append(feedback_entry)
        print(f"Feedback collected: {predicted_emotion} → {user_feedback}")
        
        return feedback_entry
    
    def analyze_feedback_patterns(self):
        """Analyze patterns in user feedback."""
        if not self.feedback_data:
            print("No feedback data available")
            return None
        
        df = pd.DataFrame(self.feedback_data)
        
        # Calculate accuracy metrics
        correct_predictions = df[df['predicted_emotion'] == df['user_feedback']]
        accuracy = len(correct_predictions) / len(df)
        
        # Analyze by emotion
        emotion_accuracy = df.groupby('predicted_emotion').apply(
            lambda x: (x['predicted_emotion'] == x['user_feedback']).mean()
        )
        
        # Analyze by confidence level
        if 'confidence' in df.columns:
            df['confidence_bin'] = pd.cut(df['confidence'], bins=5)
            confidence_accuracy = df.groupby('confidence_bin').apply(
                lambda x: (x['predicted_emotion'] == x['user_feedback']).mean()
            )
        else:
            confidence_accuracy = None
        
        analysis_report = {
            'total_feedback': len(df),
            'overall_accuracy': accuracy,
            'emotion_accuracy': emotion_accuracy.to_dict(),
            'confidence_accuracy': confidence_accuracy.to_dict() if confidence_accuracy is not None else None,
            'most_corrected_emotions': df[df['predicted_emotion'] != df['user_feedback']]['predicted_emotion'].value_counts().to_dict(),
            'most_common_corrections': df[df['predicted_emotion'] != df['user_feedback']]['user_feedback'].value_counts().to_dict()
        }
        
        return analysis_report
    
    def generate_retraining_recommendations(self, analysis_report):
        """Generate recommendations for model retraining."""
        recommendations = []
        
        if analysis_report['overall_accuracy'] < 0.8:
            recommendations.append({
                'type': 'retrain',
                'priority': 'high',
                'reason': f"Overall accuracy ({analysis_report['overall_accuracy']:.1%}) below 80% threshold"
            })
        
        # Check for specific emotion issues
        for emotion, accuracy in analysis_report['emotion_accuracy'].items():
            if accuracy < 0.7:
                recommendations.append({
                    'type': 'emotion_specific',
                    'priority': 'medium',
                    'reason': f"Low accuracy for '{emotion}' emotion ({accuracy:.1%})"
                })
        
        # Check for confidence issues
        if analysis_report['confidence_accuracy']:
            low_confidence_accuracy = min(analysis_report['confidence_accuracy'].values())
            if low_confidence_accuracy < 0.6:
                recommendations.append({
                    'type': 'confidence_calibration',
                    'priority': 'medium',
                    'reason': f"Low accuracy for high-confidence predictions ({low_confidence_accuracy:.1%})"
                })
        
        return recommendations
    
    def export_feedback_data(self, filename):
        """Export feedback data for analysis."""
        df = pd.DataFrame(self.feedback_data)
        df.to_csv(filename, index=False)
        print(f"Feedback data exported to {filename}")
    
    def load_feedback_data(self, filename):
        """Load feedback data from file."""
        df = pd.read_csv(filename)
        self.feedback_data = df.to_dict('records')
        print(f"Loaded {len(self.feedback_data)} feedback entries")

# Usage example
feedback_system = ModelFeedbackSystem()

# Collect some sample feedback
feedback_system.collect_feedback(
    "I am feeling happy today!",
    "happy",
    "happy",
    confidence=0.95
)

feedback_system.collect_feedback(
    "I feel a bit anxious about the meeting",
    "calm",
    "anxious",
    confidence=0.75
)

feedback_system.collect_feedback(
    "I'm excited for the weekend",
    "excited",
    "excited",
    confidence=0.88
)

# Analyze feedback patterns
analysis_report = feedback_system.analyze_feedback_patterns()
if analysis_report:
    print(f"Overall Accuracy: {analysis_report['overall_accuracy']:.1%}")
    print(f"Emotion Accuracy: {analysis_report['emotion_accuracy']}")

# Generate recommendations
recommendations = feedback_system.generate_retraining_recommendations(analysis_report)
for rec in recommendations:
    print(f"Recommendation: {rec['reason']} (Priority: {rec['priority']})")

# Export feedback data
feedback_system.export_feedback_data('user_feedback.csv')
```

---

## 📞 **Support & Resources**

- **API Documentation**: [Complete API Reference](API-Reference)
- **Model Monitoring**: [Performance Monitoring Guide](Model-Monitoring-Guide)
- **Research Tools**: [Experimental Testing Framework](Research-Collaboration-Guide)
- **GitHub Issues**: [Report Issues](https://github.com/your-org/SAMO--DL/issues)

---

**Ready to dive deep into emotion analytics?** Start with the [Quick Start](#-quick-start-5-minutes) section above! 📊🔬 