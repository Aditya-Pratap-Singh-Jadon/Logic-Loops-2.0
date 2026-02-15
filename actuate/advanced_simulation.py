"""
Advanced Simulation Utilities
Custom scenario builder, real-time mode, and Q-table visualization

Usage:
    python advanced_simulation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json

try:
    from carbon_capture_optimized import CarbonCapturePipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


# ============================================================================
# CUSTOM SCENARIO BUILDER
# ============================================================================

class CustomScenarioBuilder:
    """Build custom sensor scenarios with specific characteristics"""
    
    def __init__(self):
        self.scenarios = {}
    
    def create_scenario(self, name, config):
        """
        Create custom scenario from configuration
        
        Example config:
        {
            'co2': {'mean': 500, 'std': 50, 'min': 400, 'max': 600},
            'temperature': {'mean': 45, 'std': 5, 'min': 35, 'max': 55},
            'humidity': {'mean': 50, 'std': 5, 'min': 40, 'max': 60},
            'flow': {'mean': 500, 'std': 30, 'min': 400, 'max': 600},
            'weight': {'mean': 15, 'std': 2, 'min': 10, 'max': 20},
            'fan_rpm': {'mean': 1500, 'std': 100, 'min': 1200, 'max': 1800},
            'n_samples': 100,
            'anomaly_rate': 0.05  # 5% anomalies
        }
        """
        n_samples = config.get('n_samples', 100)
        anomaly_rate = config.get('anomaly_rate', 0.0)
        
        # Generate base data
        data = {}
        
        for sensor in ['co2', 'temperature', 'humidity', 'flow', 'weight', 'fan_rpm']:
            if sensor in config:
                cfg = config[sensor]
                values = np.random.normal(cfg['mean'], cfg['std'], n_samples)
                values = np.clip(values, cfg['min'], cfg['max'])
                
                # Inject anomalies
                if anomaly_rate > 0:
                    n_anomalies = int(n_samples * anomaly_rate)
                    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
                    
                    for idx in anomaly_indices:
                        # Create anomaly (spike or drop)
                        if np.random.random() > 0.5:
                            values[idx] = cfg['max'] * 1.2  # Spike
                        else:
                            values[idx] = cfg['min'] * 0.8  # Drop
                
                data[sensor] = values
        
        # Create DataFrame
        timestamps = [datetime.now() + timedelta(seconds=i*2) for i in range(n_samples)]
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'CO2_ppm': data.get('co2', np.ones(n_samples) * 400),
            'Temperature_C': data.get('temperature', np.ones(n_samples) * 25),
            'Humidity_Percent': data.get('humidity', np.ones(n_samples) * 50),
            'Gas_Flow_L_per_min': data.get('flow', np.ones(n_samples) * 400),
            'Weight_Change_g': data.get('weight', np.ones(n_samples) * 10),
            'Fan_Speed_RPM': data.get('fan_rpm', np.ones(n_samples) * 1500).astype(int),
            'Scenario': name
        })
        
        self.scenarios[name] = df
        return df
    
    def save_scenario(self, name, filename):
        """Save scenario to CSV"""
        if name in self.scenarios:
            self.scenarios[name].to_csv(filename, index=False)
            print(f"✅ Saved {name} to {filename}")
        else:
            print(f"❌ Scenario {name} not found")
    
    def load_scenario(self, filename, name=None):
        """Load scenario from CSV"""
        df = pd.read_csv(filename)
        
        if name is None:
            name = filename.replace('.csv', '')
        
        self.scenarios[name] = df
        print(f"✅ Loaded {name} from {filename}")
        return df


# ============================================================================
# REAL-TIME SIMULATOR
# ============================================================================

class RealTimeSimulator:
    """Simulate sensors in real-time (like hardware)"""
    
    def __init__(self, scenario_data, speed_factor=1.0):
        """
        Args:
            scenario_data: DataFrame with sensor data
            speed_factor: Speed multiplier (2.0 = 2x faster, 0.5 = 2x slower)
        """
        self.data = scenario_data
        self.current_idx = 0
        self.speed_factor = speed_factor
        self.is_running = False
    
    def start(self, pipeline=None, interval=2.0):
        """
        Start real-time simulation
        
        Args:
            pipeline: CarbonCapturePipeline instance
            interval: Seconds between readings (default 2.0)
        """
        self.is_running = True
        actual_interval = interval / self.speed_factor
        
        print(f"\n🚀 Starting real-time simulation...")
        print(f"   Speed: {self.speed_factor}x")
        print(f"   Interval: {actual_interval:.2f} seconds")
        print(f"   Total samples: {len(self.data)}")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while self.is_running and self.current_idx < len(self.data):
                # Get current reading
                reading = self.data.iloc[self.current_idx]
                
                # Display
                self._display_reading(reading, self.current_idx)
                
                # Process through pipeline if provided
                if pipeline:
                    record = pd.Series({
                        'CO2_ppm': reading['CO2_ppm'],
                        'Temperature_C': reading['Temperature_C'],
                        'Humidity_Percent': reading['Humidity_Percent'],
                        'Gas_Flow_L_per_min': reading['Gas_Flow_L_per_min'],
                        'Weight_Change_g': reading['Weight_Change_g'],
                        'Fan_Speed_RPM': reading['Fan_Speed_RPM']
                    })
                    
                    result = pipeline.process_record(record)
                    self._display_result(result)
                
                # Wait
                time.sleep(actual_interval)
                
                self.current_idx += 1
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Simulation stopped by user")
        
        print(f"\n✅ Simulation complete! Processed {self.current_idx} readings")
    
    def _display_reading(self, reading, idx):
        """Display current sensor reading"""
        print(f"{'='*60}")
        print(f"📊 READING #{idx+1} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"🌡️  Temperature:  {reading['Temperature_C']:.1f}°C")
        print(f"💨 CO2:          {reading['CO2_ppm']:.0f} ppm")
        print(f"💧 Humidity:     {reading['Humidity_Percent']:.1f}%")
        print(f"🌊 Flow:         {reading['Gas_Flow_L_per_min']:.1f} L/min")
        print(f"⚖️  Weight:       {reading['Weight_Change_g']:.2f} g")
        print(f"🔄 Fan:          {reading['Fan_Speed_RPM']} RPM")
    
    def _display_result(self, result):
        """Display pipeline processing result"""
        if result and 'credit' in result:
            print(f"\n🔍 Anomaly:      {'⚠️ YES' if result['anomaly']['is_anomaly'] else '✅ NO'}")
            print(f"✓  Fvalid:       {result['anomaly']['fvalid']:.4f}")
            print(f"💰 Credits:      {result['credit']['credits']:.6f} tonnes")
            print(f"🪙 Tokens:       {result['token']['tokens_minted']:.6f} CUBEX")
            
            if result['token']['tokens_minted'] > 0:
                print(f"💵 Value:        ₹{result['token']['total_value_inr']:.2f}")
        print(f"{'='*60}\n")
    
    def stop(self):
        """Stop simulation"""
        self.is_running = False


# ============================================================================
# Q-TABLE ANALYZER
# ============================================================================

class QTableAnalyzer:
    """Analyze and visualize Q-learning behavior"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.rl_controller = pipeline.rl_controller
    
    def visualize_q_table(self, top_n=20):
        """Visualize top Q-table states"""
        if not self.rl_controller.q_table:
            print("❌ Q-table is empty. Train the model first.")
            return
        
        print(f"\n{'='*70}")
        print("Q-TABLE ANALYSIS")
        print(f"{'='*70}")
        print(f"Total states explored: {len(self.rl_controller.q_table)}")
        print(f"Actions per state: {len(self.rl_controller.actions)}")
        print(f"{'='*70}\n")
        
        # Find top states by max Q-value
        state_values = []
        for state, q_values in self.rl_controller.q_table.items():
            max_q = np.max(q_values)
            best_action = self.rl_controller.actions[np.argmax(q_values)]
            state_values.append((state, max_q, best_action, q_values))
        
        # Sort by Q-value
        state_values.sort(key=lambda x: x[1], reverse=True)
        
        # Display top states
        print(f"TOP {min(top_n, len(state_values))} STATES:\n")
        print(f"{'State (CO2,T,Fan,Flow)':<30} {'Best Action':<15} {'Max Q-Value':<15} {'All Q-Values'}")
        print("-" * 90)
        
        for i, (state, max_q, best_action, q_vals) in enumerate(state_values[:top_n], 1):
            q_str = f"[{q_vals[0]:.3f}, {q_vals[1]:.3f}, {q_vals[2]:.3f}]"
            action_str = f"{best_action:+d} RPM"
            print(f"{str(state):<30} {action_str:<15} {max_q:>12.4f}   {q_str}")
        
        # Statistics
        all_q_values = [q for _, q_values in self.rl_controller.q_table.items() for q in q_values]
        print(f"\n{'='*70}")
        print("Q-VALUE STATISTICS:")
        print(f"{'='*70}")
        print(f"Mean Q-value:   {np.mean(all_q_values):.4f}")
        print(f"Max Q-value:    {np.max(all_q_values):.4f}")
        print(f"Min Q-value:    {np.min(all_q_values):.4f}")
        print(f"Std Q-value:    {np.std(all_q_values):.4f}")
        
        # Visualize Q-value distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_q_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel('Q-Value')
        plt.ylabel('Frequency')
        plt.title('Q-Value Distribution', fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        
        plt.subplot(1, 2, 2)
        action_counts = {action: 0 for action in self.rl_controller.actions}
        for _, q_values in self.rl_controller.q_table.items():
            best_action_idx = np.argmax(q_values)
            best_action = self.rl_controller.actions[best_action_idx]
            action_counts[best_action] += 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        colors = ['red' if a < 0 else 'gray' if a == 0 else 'green' for a in actions]
        
        plt.bar(range(len(actions)), counts, color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(actions)), [f"{a:+d}" for a in actions])
        plt.xlabel('Action (RPM change)')
        plt.ylabel('States preferring this action')
        plt.title('Learned Action Preferences', fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def export_q_table(self, filename='q_table.json'):
        """Export Q-table to JSON"""
        q_table_export = {}
        
        for state, q_values in self.rl_controller.q_table.items():
            state_str = str(state)
            q_table_export[state_str] = {
                'q_values': q_values.tolist(),
                'best_action': self.rl_controller.actions[np.argmax(q_values)]
            }
        
        with open(filename, 'w') as f:
            json.dump(q_table_export, f, indent=2)
        
        print(f"✅ Q-table exported to {filename}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def demo_custom_scenarios():
    """Demonstrate custom scenario creation"""
    print("\n" + "="*70)
    print("DEMO: CUSTOM SCENARIO BUILDER")
    print("="*70)
    
    builder = CustomScenarioBuilder()
    
    # Create a "Factory Startup" scenario
    factory_startup = builder.create_scenario('Factory_Startup', {
        'co2': {'mean': 700, 'std': 100, 'min': 500, 'max': 1000},
        'temperature': {'mean': 70, 'std': 10, 'min': 60, 'max': 85},
        'humidity': {'mean': 35, 'std': 5, 'min': 25, 'max': 45},
        'flow': {'mean': 700, 'std': 50, 'min': 600, 'max': 800},
        'weight': {'mean': 22, 'std': 3, 'min': 15, 'max': 28},
        'fan_rpm': {'mean': 1800, 'std': 100, 'min': 1600, 'max': 1950},
        'n_samples': 50,
        'anomaly_rate': 0.10  # 10% anomalies during startup
    })
    
    print(f"\n✅ Created 'Factory_Startup' scenario")
    print(f"   Samples: {len(factory_startup)}")
    print(f"   Avg CO2: {factory_startup['CO2_ppm'].mean():.1f} ppm")
    print(f"   Avg Temp: {factory_startup['Temperature_C'].mean():.1f}°C")
    
    # Save scenario
    builder.save_scenario('Factory_Startup', 'factory_startup.csv')
    
    return factory_startup


def demo_real_time_simulation():
    """Demonstrate real-time simulation"""
    if not PIPELINE_AVAILABLE:
        print("⚠️ Pipeline not available")
        return
    
    print("\n" + "="*70)
    print("DEMO: REAL-TIME SIMULATION")
    print("="*70)
    
    # Create simple test data
    builder = CustomScenarioBuilder()
    test_data = builder.create_scenario('Real_Time_Test', {
        'co2': {'mean': 450, 'std': 30, 'min': 400, 'max': 500},
        'temperature': {'mean': 42, 'std': 3, 'min': 38, 'max': 48},
        'humidity': {'mean': 50, 'std': 5, 'min': 45, 'max': 55},
        'flow': {'mean': 450, 'std': 30, 'min': 400, 'max': 500},
        'weight': {'mean': 12, 'std': 2, 'min': 8, 'max': 15},
        'fan_rpm': {'mean': 1500, 'std': 50, 'min': 1400, 'max': 1600},
        'n_samples': 20,  # Short demo
        'anomaly_rate': 0.05
    })
    
    # Initialize pipeline
    pipeline = CarbonCapturePipeline("REALTIME_001", is_msme=True)
    pipeline.anomaly_detector.set_lenient_mode(True)
    
    # Quick training
    print("\n🎓 Quick training...")
    pipeline.train(test_data.iloc[:10])
    
    # Real-time simulation
    simulator = RealTimeSimulator(test_data.iloc[10:], speed_factor=2.0)
    simulator.start(pipeline=pipeline, interval=1.0)


def demo_q_table_analysis():
    """Demonstrate Q-table analysis"""
    if not PIPELINE_AVAILABLE:
        print("⚠️ Pipeline not available")
        return
    
    print("\n" + "="*70)
    print("DEMO: Q-TABLE ANALYSIS")
    print("="*70)
    
    # Create and train pipeline
    builder = CustomScenarioBuilder()
    data = builder.create_scenario('Q_Analysis', {
        'co2': {'mean': 500, 'std': 50, 'min': 400, 'max': 600},
        'temperature': {'mean': 45, 'std': 5, 'min': 40, 'max': 50},
        'humidity': {'mean': 50, 'std': 5, 'min': 45, 'max': 55},
        'flow': {'mean': 500, 'std': 30, 'min': 450, 'max': 550},
        'weight': {'mean': 15, 'std': 2, 'min': 12, 'max': 18},
        'fan_rpm': {'mean': 1500, 'std': 50, 'min': 1400, 'max': 1600},
        'n_samples': 100
    })
    
    pipeline = CarbonCapturePipeline("Q_DEMO", is_msme=True)
    print("\n🎓 Training...")
    pipeline.train(data)
    
    # Analyze Q-table
    analyzer = QTableAnalyzer(pipeline)
    analyzer.visualize_q_table(top_n=15)
    analyzer.export_q_table('demo_q_table.json')


if __name__ == "__main__":
    print("\n" + "🌟"*35)
    print("ADVANCED SIMULATION UTILITIES")
    print("🌟"*35)
    
    print("\nAvailable Demos:")
    print("  1. Custom Scenario Builder")
    print("  2. Real-Time Simulation")
    print("  3. Q-Table Analysis")
    
    choice = input("\nSelect demo (1/2/3) or Enter to skip: ").strip()
    
    if choice == '1':
        demo_custom_scenarios()
    elif choice == '2':
        demo_real_time_simulation()
    elif choice == '3':
        demo_q_table_analysis()
    else:
        print("\n💡 Import these utilities in your own scripts!")
        print("\nExample:")
        print("  from advanced_simulation import CustomScenarioBuilder")
        print("  builder = CustomScenarioBuilder()")
        print("  data = builder.create_scenario('MyScenario', config)")