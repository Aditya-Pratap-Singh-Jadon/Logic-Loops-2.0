"""
Carbon Capture System - Python Brain
Integrates Arduino sensors with Q-learning and CUBEX token minting

Requirements:
    pip install pyserial numpy pandas scikit-learn matplotlib

Hardware:
    - Arduino with DS18B20 + MQ135
    - L298N Motor Driver
    - USB Serial connection
"""

import serial
import serial.tools.list_ports
import time
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# Import the optimized pipeline
# Make sure carbon_capture_optimized.py is in the same directory
try:
    from carbon_capture_optimized import (
        AnomalyDetector, RLFanController, CreditCalculator,
        BlockchainTokenizer, CarbonCapturePipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    print("⚠️ Optimized pipeline not found. Running in standalone mode.")
    PIPELINE_AVAILABLE = False


# ============================================================================
# FALLBACK DATASET (Used when Serial fails)
# ============================================================================

FALLBACK_DATA = {
    'temperature': np.array([42.5, 43.1, 41.8, 44.2, 42.9, 43.5, 42.1, 43.8, 42.6, 43.2]),
    'co2_ppm': np.array([420, 435, 410, 445, 425, 440, 415, 450, 430, 438]),
    'humidity': np.array([55.0] * 10),
    'flow': np.array([450.0] * 10),
    'weight': np.array([12.5] * 10),
    'fan_rpm': np.array([1500] * 10)
}

FALLBACK_INDEX = 0


# ============================================================================
# ARDUINO SERIAL INTERFACE
# ============================================================================

class ArduinoInterface:
    """Manages Serial communication with Arduino"""
    
    def __init__(self, port=None, baudrate=9600, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.use_fallback = False
        self.fallback_index = 0
        
    def find_arduino(self):
        """Auto-detect Arduino port"""
        print("\n🔍 Scanning for Arduino...")
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("❌ No serial ports found")
            return None
        
        print("\n📡 Available ports:")
        for i, port in enumerate(ports, 1):
            print(f"  {i}. {port.device} - {port.description}")
        
        # Try to auto-detect Arduino
        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                print(f"\n✅ Auto-detected Arduino: {port.device}")
                return port.device
        
        # Manual selection
        if len(ports) == 1:
            return ports[0].device
        
        choice = input(f"\nSelect port (1-{len(ports)}) or Enter for fallback mode: ").strip()
        
        if choice == '':
            return None
        
        try:
            return ports[int(choice) - 1].device
        except (ValueError, IndexError):
            return None
    
    def connect(self):
        """Establish Serial connection"""
        if self.port is None:
            self.port = self.find_arduino()
        
        if self.port is None:
            print("\n⚠️ No port selected - Entering FALLBACK MODE")
            self.use_fallback = True
            return False
        
        try:
            print(f"\n📡 Connecting to {self.port}...")
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for Arduino to reset
            
            # Clear buffer
            self.serial.reset_input_buffer()
            
            print("✅ Connected to Arduino")
            self.connected = True
            return True
            
        except serial.SerialException as e:
            print(f"❌ Connection failed: {e}")
            print("⚠️ Entering FALLBACK MODE")
            self.use_fallback = True
            return False
    
    def read_sensor_data(self):
        """Read data from Arduino or fallback dataset"""
        if self.use_fallback:
            return self._get_fallback_data()
        
        try:
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                
                # Skip diagnostic messages
                if line.startswith("---") or line.startswith("CMD:") or "Initialized" in line:
                    return None
                
                # Parse CSV data: temp,co2,humidity,flow,weight,fan_rpm
                parts = line.split(',')
                
                if len(parts) == 6:
                    data = {
                        'temperature': float(parts[0]),
                        'co2_ppm': float(parts[1]),
                        'humidity': float(parts[2]),
                        'flow': float(parts[3]),
                        'weight': float(parts[4]),
                        'fan_rpm': int(parts[5]),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'arduino'
                    }
                    return data
                else:
                    print(f"⚠️ Invalid data format: {line}")
                    return None
            
            return None
            
        except (serial.SerialException, ValueError, UnicodeDecodeError) as e:
            print(f"❌ Serial error: {e}")
            print("⚠️ Switching to FALLBACK MODE")
            self.use_fallback = True
            self.connected = False
            return self._get_fallback_data()
    
    def send_fan_command(self, state):
        """Send fan control command to Arduino"""
        if self.use_fallback:
            print(f"[FALLBACK] Fan command: {'ON' if state else 'OFF'}")
            return True
        
        try:
            command = '1' if state else '0'
            self.serial.write(command.encode())
            self.serial.flush()
            return True
        except serial.SerialException as e:
            print(f"❌ Send error: {e}")
            return False
    
    def send_fan_speed(self, speed_level):
        """Send PWM speed command (2-9)"""
        if self.use_fallback:
            print(f"[FALLBACK] Fan speed: {speed_level}")
            return True
        
        try:
            speed_level = max(2, min(9, speed_level))
            self.serial.write(str(speed_level).encode())
            self.serial.flush()
            return True
        except serial.SerialException as e:
            print(f"❌ Send error: {e}")
            return False
    
    def _get_fallback_data(self):
        """Get data from fallback dataset"""
        global FALLBACK_INDEX
        
        data = {
            'temperature': FALLBACK_DATA['temperature'][FALLBACK_INDEX % 10],
            'co2_ppm': FALLBACK_DATA['co2_ppm'][FALLBACK_INDEX % 10],
            'humidity': FALLBACK_DATA['humidity'][FALLBACK_INDEX % 10],
            'flow': FALLBACK_DATA['flow'][FALLBACK_INDEX % 10],
            'weight': FALLBACK_DATA['weight'][FALLBACK_INDEX % 10],
            'fan_rpm': FALLBACK_DATA['fan_rpm'][FALLBACK_INDEX % 10],
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
        
        FALLBACK_INDEX += 1
        return data
    
    def close(self):
        """Close Serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial connection closed")


# ============================================================================
# CARBON CAPTURE BRAIN
# ============================================================================

class CarbonCaptureBrain:
    """Main control system integrating sensors, Q-learning, and credits"""
    
    def __init__(self, enterprise_id="ARDUINO_001", port=None):
        self.enterprise_id = enterprise_id
        
        # Initialize Arduino interface
        self.arduino = ArduinoInterface(port=port)
        
        # Initialize pipeline components
        if PIPELINE_AVAILABLE:
            self.pipeline = CarbonCapturePipeline(enterprise_id, is_msme=True)
            self.pipeline.anomaly_detector.set_lenient_mode(True)
            self.trained = False
        else:
            self.pipeline = None
            self.trained = False
        
        # Data collection
        self.data_history = []
        self.max_history = 100
        
        # Statistics
        self.total_credits = 0.0
        self.total_tokens = 0.0
        self.readings_count = 0
        self.anomalies_count = 0
    
    def connect(self):
        """Connect to Arduino"""
        return self.arduino.connect()
    
    def train_models(self, num_samples=20):
        """Collect data and train models"""
        if not PIPELINE_AVAILABLE:
            print("⚠️ Pipeline not available - skipping training")
            return
        
        print(f"\n🎓 Collecting {num_samples} samples for training...")
        training_data = []
        
        for i in range(num_samples):
            print(f"  Sample {i+1}/{num_samples}...", end='\r')
            
            data = self.arduino.read_sensor_data()
            if data:
                training_data.append(data)
            
            time.sleep(1)
        
        if len(training_data) < 10:
            print("\n⚠️ Not enough data for training")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        df.rename(columns={
            'temperature': 'Temperature_C',
            'co2_ppm': 'CO2_ppm',
            'humidity': 'Humidity_Percent',
            'flow': 'Gas_Flow_L_per_min',
            'weight': 'Weight_Change_g',
            'fan_rpm': 'Fan_Speed_RPM'
        }, inplace=True)
        
        # Train pipeline
        print(f"\n\n🎓 Training models on {len(df)} samples...")
        self.pipeline.train(df)
        self.trained = True
        print("✅ Training complete!")
    
    def process_sensor_reading(self, data):
        """Process one sensor reading"""
        if not data:
            return None
        
        self.readings_count += 1
        
        # Store in history
        self.data_history.append(data)
        if len(self.data_history) > self.max_history:
            self.data_history.pop(0)
        
        # If pipeline available and trained, process through it
        if PIPELINE_AVAILABLE and self.trained:
            # Convert to record format
            record = pd.Series({
                'CO2_ppm': data['co2_ppm'],
                'Temperature_C': data['temperature'],
                'Humidity_Percent': data['humidity'],
                'Gas_Flow_L_per_min': data['flow'],
                'Weight_Change_g': data['weight'],
                'Fan_Speed_RPM': data['fan_rpm']
            })
            
            # Process through pipeline
            result = self.pipeline.process_record(record)
            
            # Update statistics
            if result['anomaly']['is_anomaly']:
                self.anomalies_count += 1
            
            if result['credit']['is_valid']:
                self.total_credits += result['credit']['credits']
                self.total_tokens += result['token']['tokens_minted']
            
            # Determine fan action based on Q-learning
            recommended_fan_rpm = result['fan']
            
            # Convert RPM to ON/OFF (simple threshold)
            if recommended_fan_rpm > 1400:
                self.arduino.send_fan_command(True)
            else:
                self.arduino.send_fan_command(False)
            
            return result
        
        else:
            # Simple logic without pipeline
            # Turn fan ON if CO2 > 430 ppm or temp > 43°C
            if data['co2_ppm'] > 430 or data['temperature'] > 43:
                self.arduino.send_fan_command(True)
            else:
                self.arduino.send_fan_command(False)
            
            return {
                'data': data,
                'fan_action': 'ON' if (data['co2_ppm'] > 430 or data['temperature'] > 43) else 'OFF',
                'credits': 0,
                'tokens': 0
            }
    
    def print_status(self, data, result):
        """Print current status"""
        print("\n" + "="*70)
        print(f"📊 READING #{self.readings_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        # Sensor data
        print(f"🌡️  Temperature: {data['temperature']:.1f}°C")
        print(f"💨 CO2 Level:   {data['co2_ppm']:.0f} ppm")
        print(f"💧 Humidity:    {data['humidity']:.1f}%")
        print(f"🌊 Flow Rate:   {data['flow']:.1f} L/min")
        print(f"⚖️  Weight:      {data['weight']:.2f} g")
        print(f"🔄 Fan Speed:   {data['fan_rpm']} RPM")
        print(f"📡 Source:      {data['source']}")
        
        if PIPELINE_AVAILABLE and self.trained and isinstance(result, dict):
            print(f"\n🔍 Anomaly:     {'⚠️ YES' if result['anomaly']['is_anomaly'] else '✅ NO'}")
            print(f"✓  Fvalid:      {result['anomaly']['fvalid']:.4f}")
            print(f"💰 Credits:     {result['credit']['credits']:.6f} tonnes CO2")
            print(f"🪙 Tokens:      {result['token']['tokens_minted']:.6f} CUBEX")
            
            if result['token']['tokens_minted'] > 0:
                print(f"💵 Value:       ₹{result['token']['total_value_inr']:.2f}")
        
        print(f"\n📈 TOTALS:")
        print(f"   Readings: {self.readings_count}")
        print(f"   Anomalies: {self.anomalies_count}")
        print(f"   Credits: {self.total_credits:.6f} tonnes")
        print(f"   Tokens: {self.total_tokens:.6f} CUBEX")
        print("="*70)
    
    def run(self, duration_seconds=None, interval=2):
        """Main execution loop"""
        print("\n" + "🌟"*35)
        print("CARBON CAPTURE BRAIN - RUNNING")
        print("🌟"*35)
        
        start_time = time.time()
        
        try:
            while True:
                # Read sensor data
                data = self.arduino.read_sensor_data()
                
                if data:
                    # Process data
                    result = self.process_sensor_reading(data)
                    
                    # Print status
                    self.print_status(data, result)
                
                # Check duration
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Wait
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n" + "="*70)
        print("SHUTTING DOWN")
        print("="*70)
        
        # Final report
        if PIPELINE_AVAILABLE and self.trained:
            print("\n📊 FINAL REPORT:")
            self.pipeline.report()
        
        # Close Arduino connection
        self.arduino.close()
        
        print("\n✅ Shutdown complete")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    print("="*70)
    print("CARBON CAPTURE SYSTEM - PYTHON BRAIN")
    print("="*70)
    print("\nThis system will:")
    print("  1. Connect to Arduino (DS18B20 + MQ135)")
    print("  2. Collect sensor data")
    print("  3. Train Q-learning model")
    print("  4. Calculate carbon credits")
    print("  5. Mint CUBEX tokens")
    print("  6. Control fan via L298N")
    print("="*70)
    
    # Create brain instance
    brain = CarbonCaptureBrain(enterprise_id="ARDUINO_001")
    
    # Connect to Arduino
    brain.connect()
    
    # Training phase
    if PIPELINE_AVAILABLE:
        choice = input("\nTrain models? [Y/n]: ").strip().lower()
        if choice != 'n':
            brain.train_models(num_samples=20)
    
    # Run main loop
    print("\n🚀 Starting main loop...")
    print("Press Ctrl+C to stop\n")
    
    brain.run(duration_seconds=None, interval=2)


if __name__ == "__main__":
    main()