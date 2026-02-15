

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from carbon_capture_optimized import CarbonCapturePipeline, visualize_results
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Pipeline not found. Make sure carbon_capture_optimized.py is in the same directory.")
    PIPELINE_AVAILABLE = False

class SensorSimulator:

    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.time_step = 0
    
    def generate_industrial_exhaust(self, n_samples=100, start_time=None):
        """
        Generate Industrial Exhaust scenario data
        
        Characteristics:
        - High Temperature: 60-90°C
        - High CO2: 600-1200 ppm (polluted)
        - Moderate Humidity: 30-50% (dry from heat)
        - High Flow: 500-800 L/min (turbulent)
        - Variable Weight: 15-25 g (high capture)
        - High Fan Speed: 1600-1900 RPM
        
        Behavior:
        - Random spikes and volatility
        - Occasional sensor anomalies
        - Correlated changes (temp ↑ → CO2 ↑)
        """
        if start_time is None:
            start_time = datetime.now()
        
        timestamps = [start_time + timedelta(seconds=i*2) for i in range(n_samples)]

        t = np.linspace(0, 10, n_samples)
        
        # Temperature: 60-90°C with industrial fluctuations
        temp_base = 75 + 10 * np.sin(0.3 * t)  # Slow oscillation
        temp_noise = np.random.normal(0, 3, n_samples)  # Significant noise
        temp_spikes = np.random.choice([0, 0, 0, 0, 15], n_samples)  # Occasional spikes
        temperature = temp_base + temp_noise + temp_spikes
        temperature = np.clip(temperature, 60, 95)
        
        # CO2: 600-1200 ppm (correlated with temperature)
        co2_base = 800 + 200 * np.sin(0.3 * t + 0.5)
        co2_temp_correlation = (temperature - 75) * 3  # Temp affects CO2
        co2_noise = np.random.normal(0, 50, n_samples)
        co2_spikes = np.random.choice([0, 0, 0, 200], n_samples)  # Pollution spikes
        co2 = co2_base + co2_temp_correlation + co2_noise + co2_spikes
        co2 = np.clip(co2, 500, 1300)
        
        # Humidity: 30-50% (low due to heat)
        humidity_base = 40 + 8 * np.sin(0.2 * t)
        humidity_noise = np.random.normal(0, 3, n_samples)
        humidity = humidity_base + humidity_noise
        humidity = np.clip(humidity, 25, 55)
        
        # Flow Rate: 500-800 L/min (turbulent industrial exhaust)
        flow_base = 650 + 100 * np.sin(0.4 * t)
        flow_noise = np.random.normal(0, 30, n_samples)
        flow_turbulence = np.random.choice([-50, 0, 0, 50], n_samples)
        flow = flow_base + flow_noise + flow_turbulence
        flow = np.clip(flow, 450, 850)
        
        # Weight Captured: 15-25 g (high due to pollution)
        weight_base = 20 + 3 * np.sin(0.3 * t)
        weight_noise = np.random.normal(0, 1.5, n_samples)
        weight_co2_correlation = (co2 - 800) * 0.008  # More CO2 → more capture
        weight = weight_base + weight_noise + weight_co2_correlation
        weight = np.clip(weight, 12, 28)
        
        # Fan Speed: 1600-1900 RPM (high for industrial load)
        fan_base = 1750 + 100 * np.sin(0.35 * t)
        fan_noise = np.random.normal(0, 30, n_samples)
        fan = fan_base + fan_noise
        fan = np.clip(fan, 1500, 1950)
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'CO2_ppm': co2,
            'Temperature_C': temperature,
            'Humidity_Percent': humidity,
            'Gas_Flow_L_per_min': flow,
            'Weight_Change_g': weight,
            'Fan_Speed_RPM': fan.astype(int),
            'Scenario': 'Industrial_Exhaust'
        })
        
        return df
    
    def generate_ambient_air(self, n_samples=100, start_time=None):
        """
        Generate Ambient Air scenario data
        
        Characteristics:
        - Room Temperature: 20-30°C
        - Low CO2: 350-450 ppm (clean)
        - Normal Humidity: 45-65%
        - Low Flow: 300-500 L/min (steady)
        - Low Weight: 8-12 g (minimal capture)
        - Moderate Fan Speed: 1200-1500 RPM
        
        Behavior:
        - Stable, predictable patterns
        - Minimal noise
        - Gradual changes only
        """
        if start_time is None:
            start_time = datetime.now()
        
        timestamps = [start_time + timedelta(seconds=i*2) for i in range(n_samples)]
        
        t = np.linspace(0, 10, n_samples)
        
        # Temperature: 20-30°C (room temperature, stable)
        temp_base = 25 + 3 * np.sin(0.15 * t)  # Gentle variation
        temp_noise = np.random.normal(0, 0.5, n_samples)  # Minimal noise
        temperature = temp_base + temp_noise
        temperature = np.clip(temperature, 20, 32)
        
        # CO2: 350-450 ppm (clean ambient air)
        co2_base = 400 + 30 * np.sin(0.2 * t)
        co2_noise = np.random.normal(0, 10, n_samples)  # Low noise
        co2 = co2_base + co2_noise
        co2 = np.clip(co2, 340, 470)
        
        # Humidity: 45-65% (comfortable indoor range)
        humidity_base = 55 + 8 * np.sin(0.18 * t)
        humidity_noise = np.random.normal(0, 2, n_samples)
        humidity = humidity_base + humidity_noise
        humidity = np.clip(humidity, 42, 68)
        
        # Flow Rate: 300-500 L/min (gentle, steady)
        flow_base = 400 + 50 * np.sin(0.25 * t)
        flow_noise = np.random.normal(0, 15, n_samples)
        flow = flow_base + flow_noise
        flow = np.clip(flow, 280, 520)
        
        # Weight Captured: 8-12 g (low capture from clean air)
        weight_base = 10 + 1.5 * np.sin(0.22 * t)
        weight_noise = np.random.normal(0, 0.5, n_samples)
        weight = weight_base + weight_noise
        weight = np.clip(weight, 7, 13)
        
        # Fan Speed: 1200-1500 RPM (moderate)
        fan_base = 1350 + 80 * np.sin(0.28 * t)
        fan_noise = np.random.normal(0, 20, n_samples)
        fan = fan_base + fan_noise
        fan = np.clip(fan, 1150, 1550)
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'CO2_ppm': co2,
            'Temperature_C': temperature,
            'Humidity_Percent': humidity,
            'Gas_Flow_L_per_min': flow,
            'Weight_Change_g': weight,
            'Fan_Speed_RPM': fan.astype(int),
            'Scenario': 'Ambient_Air'
        })
        
        return df
    
    def generate_mixed_scenario(self, n_samples=100, transition_point=50):
        """
        Generate data that transitions from Industrial to Ambient
        Simulates a factory that stops production
        """
        industrial = self.generate_industrial_exhaust(transition_point)
        ambient = self.generate_ambient_air(n_samples - transition_point, 
                                           start_time=industrial['Timestamp'].iloc[-1] + timedelta(seconds=2))
        
        # Smooth transition
        transition_window = 10
        for i in range(transition_window):
            blend_factor = i / transition_window
            for col in ['CO2_ppm', 'Temperature_C', 'Humidity_Percent', 
                       'Gas_Flow_L_per_min', 'Weight_Change_g', 'Fan_Speed_RPM']:
                if i < len(ambient):
                    ambient.loc[i, col] = (industrial[col].iloc[-1] * (1 - blend_factor) + 
                                          ambient[col].iloc[i] * blend_factor)
        
        combined = pd.concat([industrial, ambient], ignore_index=True)
        combined['Scenario'] = ['Industrial→Ambient'] * len(combined)
        
        return combined

class ScenarioComparator:
    """Compare pipeline performance across scenarios"""
    
    def __init__(self):
        self.results = {}
        
    def run_scenario(self, scenario_name, data_df, enterprise_id="SIM_001"):
        """Run pipeline on a scenario"""
        print(f"\n{'='*70}")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"{'='*70}")
        
        # Split data for training and testing
        split_idx = int(len(data_df) * 0.3)
        train_data = data_df.iloc[:split_idx].copy()
        test_data = data_df.iloc[split_idx:].copy()
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples:  {len(test_data)}")
        
        # Initialize pipeline
        pipeline = CarbonCapturePipeline(f"{enterprise_id}_{scenario_name}", is_msme=True)
        pipeline.anomaly_detector.set_lenient_mode(True)
        
        # Train
        print("\nTraining models...")
        pipeline.train(train_data)
        
        # Process
        print("\nProcessing test data...")
        results_df = pipeline.process_batch(test_data)
        
        # Get portfolio
        portfolio = pipeline.report()
        
        # Store results
        self.results[scenario_name] = {
            'pipeline': pipeline,
            'train_data': train_data,
            'test_data': test_data,
            'results': results_df,
            'portfolio': portfolio,
            'stats': self._calculate_stats(results_df)
        }
        
        return results_df, portfolio
    
    def _calculate_stats(self, results_df):
        """Calculate scenario statistics"""
        return {
            'total_readings': len(results_df),
            'anomalies': results_df['Is_Anomaly'].sum(),
            'anomaly_rate': results_df['Is_Anomaly'].mean() * 100,
            'avg_fvalid': results_df['Fvalid'].mean(),
            'avg_co2': results_df['CO2_ppm'].mean(),
            'avg_temp': results_df.get('Temperature_C', results_df.get('Temp_C', [0])).mean() if 'Temperature_C' in results_df.columns or 'Temp_C' in results_df.columns else 0,
            'avg_flow': results_df['Flow_L_min'].mean(),
            'total_credits': results_df['Credits'].sum(),
            'total_tokens': results_df['Tokens'].sum(),
            'total_value_inr': results_df['Value_INR'].sum(),
            'avg_fan_rpm': results_df['Fan_RPM'].mean()
        }
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*70)
        print("SCENARIO COMPARISON REPORT")
        print("="*70)
        
        # Table header
        print(f"\n{'Metric':<30} {'Industrial':<18} {'Ambient':<18} {'Difference'}")
        print("-" * 70)
        
        if 'Industrial_Exhaust' in self.results and 'Ambient_Air' in self.results:
            ind_stats = self.results['Industrial_Exhaust']['stats']
            amb_stats = self.results['Ambient_Air']['stats']
            
            metrics = [
                ('Total Readings', 'total_readings', ''),
                ('Anomalies Detected', 'anomalies', ''),
                ('Anomaly Rate', 'anomaly_rate', '%'),
                ('Avg Fvalid Score', 'avg_fvalid', ''),
                ('Avg CO2 Level', 'avg_co2', ' ppm'),
                ('Avg Temperature', 'avg_temp', '°C'),
                ('Avg Flow Rate', 'avg_flow', ' L/min'),
                ('Avg Fan Speed', 'avg_fan_rpm', ' RPM'),
                ('Total Credits', 'total_credits', ' tonnes'),
                ('Total CUBEX Tokens', 'total_tokens', ''),
                ('Total Value', 'total_value_inr', ' INR')
            ]
            
            for label, key, unit in metrics:
                ind_val = ind_stats[key]
                amb_val = amb_stats[key]
                
                if key in ['total_credits', 'total_tokens', 'avg_fvalid']:
                    diff = ind_val - amb_val
                    print(f"{label:<30} {ind_val:>12.4f}{unit:<5} {amb_val:>12.4f}{unit:<5} {diff:>+12.4f}")
                elif key in ['total_value_inr']:
                    diff = ind_val - amb_val
                    print(f"{label:<30} ₹{ind_val:>11,.2f}{unit:<4} ₹{amb_val:>11,.2f}{unit:<4} ₹{diff:>+11,.2f}")
                elif key == 'anomaly_rate':
                    diff = ind_val - amb_val
                    print(f"{label:<30} {ind_val:>12.2f}{unit:<5} {amb_val:>12.2f}{unit:<5} {diff:>+12.2f}")
                else:
                    diff = ind_val - amb_val
                    print(f"{label:<30} {ind_val:>12.1f}{unit:<5} {amb_val:>12.1f}{unit:<5} {diff:>+12.1f}")
            
            print("\n" + "="*70)
            print("KEY INSIGHTS:")
            print("="*70)
            
            # Generate insights
            if ind_stats['total_credits'] > amb_stats['total_credits']:
                improvement = ((ind_stats['total_credits'] - amb_stats['total_credits']) / 
                              amb_stats['total_credits'] * 100)
                print(f"✅ Industrial scenario captured {improvement:.1f}% more CO2")
            
            if ind_stats['anomaly_rate'] > amb_stats['anomaly_rate']:
                print(f"⚠️ Industrial air has {ind_stats['anomaly_rate']:.1f}% anomaly rate vs {amb_stats['anomaly_rate']:.1f}% ambient")
            
            if ind_stats['avg_fan_rpm'] > amb_stats['avg_fan_rpm']:
                fan_increase = ((ind_stats['avg_fan_rpm'] - amb_stats['avg_fan_rpm']) / 
                               amb_stats['avg_fan_rpm'] * 100)
                print(f"🔄 Q-learning increased fan speed by {fan_increase:.1f}% for industrial load")
            
            value_diff = ind_stats['total_value_inr'] - amb_stats['total_value_inr']
            print(f"💰 Industrial scenario generates ₹{value_diff:,.2f} more value")
            
            print("\n" + "="*70)
    
    def visualize_comparison(self):
        """Create visualization comparing scenarios"""
        if 'Industrial_Exhaust' not in self.results or 'Ambient_Air' not in self.results:
            print("⚠️ Need both scenarios to compare")
            return
        
        ind_results = self.results['Industrial_Exhaust']['results']
        amb_results = self.results['Ambient_Air']['results']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. CO2 Levels
        axes[0, 0].plot(ind_results['Record_ID'], ind_results['CO2_ppm'], 
                       'r-', linewidth=2, label='Industrial', alpha=0.7)
        axes[0, 0].plot(amb_results['Record_ID'], amb_results['CO2_ppm'], 
                       'b-', linewidth=2, label='Ambient', alpha=0.7)
        axes[0, 0].set_title('CO2 Levels Comparison', fontweight='bold', fontsize=13)
        axes[0, 0].set_ylabel('CO2 (ppm)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Fvalid Score
        axes[0, 1].plot(ind_results['Record_ID'], ind_results['Fvalid'], 
                       'r-', linewidth=2, label='Industrial', alpha=0.7)
        axes[0, 1].plot(amb_results['Record_ID'], amb_results['Fvalid'], 
                       'b-', linewidth=2, label='Ambient', alpha=0.7)
        axes[0, 1].axhline(y=0.87, color='gray', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Data Validity (Fvalid)', fontweight='bold', fontsize=13)
        axes[0, 1].set_ylabel('Fvalid Score')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Fan Speed Control
        axes[1, 0].plot(ind_results['Record_ID'], ind_results['Fan_RPM'], 
                       'r-', linewidth=2, label='Industrial', alpha=0.7)
        axes[1, 0].plot(amb_results['Record_ID'], amb_results['Fan_RPM'], 
                       'b-', linewidth=2, label='Ambient', alpha=0.7)
        axes[1, 0].set_title('Q-Learning Fan Optimization', fontweight='bold', fontsize=13)
        axes[1, 0].set_ylabel('Fan Speed (RPM)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Flow Rate
        axes[1, 1].plot(ind_results['Record_ID'], ind_results['Flow_L_min'], 
                       'r-', linewidth=2, label='Industrial', alpha=0.7)
        axes[1, 1].plot(amb_results['Record_ID'], amb_results['Flow_L_min'], 
                       'b-', linewidth=2, label='Ambient', alpha=0.7)
        axes[1, 1].set_title('Gas Flow Rate', fontweight='bold', fontsize=13)
        axes[1, 1].set_ylabel('Flow (L/min)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 5. Carbon Credits Comparison
        axes[2, 0].bar(ind_results['Record_ID'], ind_results['Credits']*1000, 
                      color='red', alpha=0.6, label='Industrial', width=0.8)
        axes[2, 0].bar(amb_results['Record_ID'], amb_results['Credits']*1000, 
                      color='blue', alpha=0.6, label='Ambient', width=0.8)
        axes[2, 0].set_title('Carbon Credits Earned', fontweight='bold', fontsize=13)
        axes[2, 0].set_ylabel('Credits (mg)')
        axes[2, 0].set_xlabel('Record ID')
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3, axis='y')
        
        # 6. Cumulative Value
        ind_cumulative = (ind_results['Value_INR'].cumsum())
        amb_cumulative = (amb_results['Value_INR'].cumsum())
        axes[2, 1].plot(ind_results['Record_ID'], ind_cumulative, 
                       'r-', linewidth=3, label='Industrial', alpha=0.7)
        axes[2, 1].plot(amb_results['Record_ID'], amb_cumulative, 
                       'b-', linewidth=3, label='Ambient', alpha=0.7)
        axes[2, 1].set_title('Cumulative Token Value', fontweight='bold', fontsize=13)
        axes[2, 1].set_ylabel('Value (₹)')
        axes[2, 1].set_xlabel('Record ID')
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)
        
        plt.suptitle('Industrial Exhaust vs Ambient Air - Complete Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main simulation workflow"""
    print("="*70)
    print("CARBON CAPTURE DATA SIMULATION")
    print("="*70)
    print("\nGenerating synthetic sensor data for two scenarios:")
    print("  1. Industrial Exhaust (High pollution, high temp)")
    print("  2. Ambient Air (Clean, room temperature)")
    print("="*70)
    
    # Initialize simulator
    simulator = SensorSimulator(seed=42)
    
    # Generate scenarios
    print("\n📊 Generating Industrial Exhaust data...")
    industrial_data = simulator.generate_industrial_exhaust(n_samples=100)
    print(f"✅ Generated {len(industrial_data)} samples")
    print(f"   CO2 range: {industrial_data['CO2_ppm'].min():.0f}-{industrial_data['CO2_ppm'].max():.0f} ppm")
    print(f"   Temp range: {industrial_data['Temperature_C'].min():.1f}-{industrial_data['Temperature_C'].max():.1f}°C")
    
    print("\n📊 Generating Ambient Air data...")
    ambient_data = simulator.generate_ambient_air(n_samples=100)
    print(f"✅ Generated {len(ambient_data)} samples")
    print(f"   CO2 range: {ambient_data['CO2_ppm'].min():.0f}-{ambient_data['CO2_ppm'].max():.0f} ppm")
    print(f"   Temp range: {ambient_data['Temperature_C'].min():.1f}-{ambient_data['Temperature_C'].max():.1f}°C")
    
    if not PIPELINE_AVAILABLE:
        print("\n⚠️ Pipeline not available. Install carbon_capture_optimized.py")
        return
    
    # Run comparisons
    comparator = ScenarioComparator()
    
    # Run Industrial scenario
    ind_results, ind_portfolio = comparator.run_scenario('Industrial_Exhaust', industrial_data)
    
    # Run Ambient scenario
    amb_results, amb_portfolio = comparator.run_scenario('Ambient_Air', ambient_data)
    
    # Generate comparison report
    comparator.generate_comparison_report()
    
    # Visualize
    print("\n📈 Generating comparison visualizations...")
    comparator.visualize_comparison()
    
    print("\n✅ Simulation complete!")
    print("\n💡 Key Takeaways:")
    print("  • Industrial exhaust requires higher fan speeds (Q-learning adaptation)")
    print("  • More CO2 captured = more carbon credits = more CUBEX tokens")
    print("  • Anomaly detection flags industrial volatility")
    print("  • System autonomously optimizes for each environment")


if __name__ == "__main__":
    main()