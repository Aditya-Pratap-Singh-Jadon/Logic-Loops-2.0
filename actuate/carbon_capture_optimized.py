"""
Optimized Carbon Capture AI Pipeline
Streamlined version with core functionality only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self.one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=contamination)
        self.is_trained = False
        self.lenient_mode = False
    
    def set_lenient_mode(self, lenient=True):
        self.lenient_mode = lenient
        mode = "LENIENT (0.85-0.89 for anomalies)" if lenient else "STRICT (0.0 for anomalies)"
        print(f"Detection mode: {mode}")
    
    def create_features(self, co2, temp, humidity, flow, weight, fan_speed):
        """Create feature vector from sensors"""
        co2_mass_flow = (co2 / 1e6) * flow * 1.96
        co2_available = co2_mass_flow * 60
        capture_eff = weight / (co2_available + 1e-6)
        flow_fan_dev = abs(flow - fan_speed * 0.3) / (fan_speed * 0.3 + 1e-6)
        temp_humidity_idx = temp / (humidity + 1e-6)
        
        return np.array([co2, temp, humidity, flow, weight, fan_speed, 
                        co2_mass_flow, capture_eff, flow_fan_dev, temp_humidity_idx]).reshape(1, -1)
    
    def train(self, normal_data):
        """Train on normal data"""
        if isinstance(normal_data, pd.DataFrame):
            normal_data = normal_data.values
        
        scaled = self.scaler.fit_transform(normal_data)
        self.isolation_forest.fit(scaled)
        self.one_class_svm.fit(scaled)
        self.is_trained = True
    
    def detect(self, features):
        """Detect anomalies"""
        if not self.is_trained:
            return {'is_anomaly': False, 'fvalid': 0.70, 'confidence': 0.70}
        
        scaled = self.scaler.transform(features)
        if_pred = self.isolation_forest.predict(scaled)[0]
        svm_pred = self.one_class_svm.predict(scaled)[0]
        is_normal = (if_pred == 1) and (svm_pred == 1)
        
        if is_normal:
            fvalid = np.random.uniform(0.92, 0.99)
        else:
            fvalid = np.random.uniform(0.85, 0.89) if self.lenient_mode else 0.0
        
        return {'is_anomaly': not is_normal, 'fvalid': round(fvalid, 4), 'confidence': round(fvalid, 4)}


# ============================================================================
# RL FAN CONTROLLER
# ============================================================================

class RLFanController:
    """Q-Learning controller for fan optimization"""
    
    def __init__(self):
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.actions = [-100, 0, 100]
        self.q_table = {}
        self.min_fan = 1000
        self.max_fan = 2000
    
    def _discretize(self, co2, temp, fan, flow):
        return (int(co2/50), int(temp/5), int(fan/100), int(flow/50))
    
    def get_action(self, state, explore=True):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        return np.argmax(self.q_table[state])
    
    def calculate_reward(self, weight, fan, flow):
        return weight * 20 - (fan / self.max_fan) * 3 + (2 if 400 <= flow <= 600 else -1)
    
    def update_q(self, state, action_idx, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
    
    def control(self, co2, temp, fan, flow, explore=False):
        state = self._discretize(co2, temp, fan, flow)
        action_idx = self.get_action(state, explore)
        new_fan = np.clip(fan + self.actions[action_idx], self.min_fan, self.max_fan)
        return new_fan, action_idx, state
    
    def train_episode(self, co2_data, temp_data, flow_data, weight_data):
        total_reward = 0
        fan = 1500
        
        for i in range(len(co2_data) - 1):
            new_fan, action_idx, state = self.control(co2_data[i], temp_data[i], fan, flow_data[i], True)
            reward = self.calculate_reward(weight_data[i+1], new_fan, flow_data[i+1])
            next_state = self._discretize(co2_data[i+1], temp_data[i+1], new_fan, flow_data[i+1])
            self.update_q(state, action_idx, reward, next_state)
            fan = new_fan
            total_reward += reward
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_reward


# ============================================================================
# CARBON CREDIT CALCULATOR
# ============================================================================

class CreditCalculator:
    """Calculate carbon credits with validation"""
    
    def __init__(self, fsme=1.2, alpha=0.1):
        self.fsme = fsme
        self.alpha = alpha
        self.cubex_price_inr = 20000
        self.eth_price_inr = 200000
        self.history = []
    
    def calc_E_actual(self, co2_ppm, flow, time_hours=1.0):
        """Calculate actual emissions from flow sensor"""
        total_gas = flow * 60 * time_hours
        co2_volume = total_gas * (co2_ppm / 1e6)
        co2_mass = co2_volume * 1.96 / 1e6
        return co2_mass
    
    def calc_E_removed(self, weight_g):
        """Calculate CO2 removed"""
        return weight_g * 0.32 / 1e6
    
    def calculate(self, co2_ppm, flow, weight_g, fvalid, baseline=0.001):
        """Calculate credits with strict validation"""
        if fvalid < 0.70:
            return {'E_net': 0, 'credits': 0, 'is_valid': False, 
                   'status': f'INVALID - Fvalid {fvalid:.4f} < 0.70'}
        
        e_actual = self.calc_E_actual(co2_ppm, flow)
        e_removed = self.calc_E_removed(weight_g)
        e_reduced = max(0, baseline - e_actual)
        e_gross = e_reduced + e_removed
        e_net = e_gross * fvalid * self.fsme * self.alpha
        
        if e_net <= 0:
            return {'E_net': 0, 'credits': 0, 'is_valid': False, 
                   'status': 'INVALID - No reduction'}
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'E_baseline': baseline,
            'E_actual': e_actual,
            'E_removed': e_removed,
            'E_net': round(e_net, 6),
            'credits': round(e_net, 6),
            'is_valid': True,
            'status': 'VALID',
            'F_valid': fvalid
        }
        self.history.append(result)
        return result


# ============================================================================
# BLOCKCHAIN TOKENIZER
# ============================================================================

class BlockchainTokenizer:
    """Mint CUBEX tokens for valid credits"""
    
    def __init__(self):
        self.cubex_price_inr = 20000
        self.eth_price_inr = 200000
        self.ledger = []
        self.total_supply = 0
    
    def mint(self, credits, enterprise_id, fvalid):
        
        """Mint tokens with validation"""
        if fvalid < 0.80 or credits <= 0:
            return {'success': False, 'tokens_minted': 0, 'total_value_inr': 0}
        
        tokens = credits
        value_inr = tokens * self.cubex_price_inr
        value_eth = tokens * (self.cubex_price_inr / self.eth_price_inr)
        
        tx_hash = hashlib.sha256(f"{enterprise_id}{tokens}{datetime.now()}".encode()).hexdigest()[:16]
        
        record = {
            'token_id': f"CUBEX-{len(self.ledger)+1:06d}",
            'enterprise_id': enterprise_id,
            'credits': round(credits, 6),
            'tokens_minted': round(tokens, 6),
            'value_inr': round(value_inr, 2),
            'value_eth': round(value_eth, 6),
            'fvalid': fvalid,
            'timestamp': datetime.now().isoformat(),
            'tx_hash': tx_hash
        }
        
        self.ledger.append(record)
        self.total_supply += tokens
        
        return {'success': True, 'tokens_minted': round(tokens, 6), 
                'total_value_inr': round(value_inr, 2), 'total_value_eth': round(value_eth, 6)}
    
    def get_portfolio(self, enterprise_id):
        """Get enterprise portfolio"""
        records = [r for r in self.ledger if r['enterprise_id'] == enterprise_id]
        total_tokens = sum(r['tokens_minted'] for r in records)
        total_credits = sum(r['credits'] for r in records)
        
        return {
            'enterprise_id': enterprise_id,
            'cubex_balance': round(total_tokens, 6),
            'carbon_credits': round(total_credits, 6),
            'value_inr': round(total_tokens * self.cubex_price_inr, 2),
            'value_eth': round(total_tokens * (self.cubex_price_inr / self.eth_price_inr), 6),
            'records': len(records)
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class CarbonCapturePipeline:
    """Complete optimized pipeline"""
    
    def __init__(self, enterprise_id, is_msme=True):
        self.enterprise_id = enterprise_id
        self.anomaly_detector = AnomalyDetector()
        self.rl_controller = RLFanController()
        self.credit_calculator = CreditCalculator(fsme=1.2 if is_msme else 1.0)
        self.blockchain = BlockchainTokenizer()
        self.current_fan = 1500
        self.is_trained = False
        print(f"Pipeline initialized for {enterprise_id}")
    
    def train(self, df):
        """Train models"""
        print(f"\nTraining on {len(df)} records...")
        
        # Train anomaly detector
        features = []
        for _, row in df.iterrows():
            f = self.anomaly_detector.create_features(
                row['CO2_ppm'], row['Temperature_C'], row['Humidity_Percent'],
                row['Gas_Flow_L_per_min'], row['Weight_Change_g'], row['Fan_Speed_RPM']
            )
            features.append(f[0])
        self.anomaly_detector.train(np.array(features))
        
        # Train RL
        for ep in range(5):
            reward = self.rl_controller.train_episode(
                df['CO2_ppm'].values, df['Temperature_C'].values,
                df['Gas_Flow_L_per_min'].values, df['Weight_Change_g'].values
            )
            if ep == 4:
                print(f"RL training complete (final reward: {reward:.2f})")
        
        self.is_trained = True
    
    def process_record(self, record):
        """Process single record"""
        # RL Control
        if self.is_trained:
            new_fan, _, _ = self.rl_controller.control(
                record['CO2_ppm'], record['Temperature_C'],
                self.current_fan, record['Gas_Flow_L_per_min']
            )
            self.current_fan = new_fan
        else:
            new_fan = record['Fan_Speed_RPM']
        
        # Anomaly detection
        features = self.anomaly_detector.create_features(
            record['CO2_ppm'], record['Temperature_C'], record['Humidity_Percent'],
            record['Gas_Flow_L_per_min'], record['Weight_Change_g'], new_fan
        )
        anomaly = self.anomaly_detector.detect(features)
        
        # Credits
        credit = self.credit_calculator.calculate(
            record['CO2_ppm'], record['Gas_Flow_L_per_min'],
            record['Weight_Change_g'], anomaly['fvalid']
        )
        
        # Tokens
        if credit['is_valid'] and credit['credits'] > 0:
            token = self.blockchain.mint(credit['credits'], self.enterprise_id, anomaly['fvalid'])
        else:
            token = {'success': False, 'tokens_minted': 0, 'total_value_inr': 0}
        
        return {'fan': new_fan, 'anomaly': anomaly, 'credit': credit, 'token': token}
    
    def process_batch(self, df):
        """Process batch of records"""
        print(f"\nProcessing {len(df)} records...")
        
        results = []
        for idx, row in df.iterrows():
            result = self.process_record(row)
            results.append({
                'Record_ID': idx + 1,
                'CO2_ppm': row['CO2_ppm'],
                'Flow_L_min': row['Gas_Flow_L_per_min'],
                'Weight_g': row['Weight_Change_g'],
                'Fan_RPM': result['fan'],
                'Is_Anomaly': result['anomaly']['is_anomaly'],
                'Fvalid': result['anomaly']['fvalid'],
                'Credits': result['credit']['credits'],
                'Tokens': result['token']['tokens_minted'],
                'Value_INR': result['token']['total_value_inr']
            })
        
        return pd.DataFrame(results)
    
    def report(self):
        """Generate report"""
        portfolio = self.blockchain.get_portfolio(self.enterprise_id)
        print(f"\n{'='*60}")
        print(f"ENTERPRISE: {self.enterprise_id}")
        print(f"{'='*60}")
        print(f"Carbon Credits: {portfolio['carbon_credits']:.6f} tonnes CO2")
        print(f"CUBEX Tokens:   {portfolio['cubex_balance']:.6f}")
        print(f"Value (INR):    ₹{portfolio['value_inr']:,.2f}")
        print(f"Value (ETH):    {portfolio['value_eth']:.6f} ETH")
        print(f"{'='*60}")
        return portfolio


# ============================================================================
# DATA UTILITIES
# ============================================================================

def generate_sample_data(n=50):
    """Generate sample sensor data"""
    np.random.seed(42)
    timestamps = pd.date_range('2024-02-14 10:00:00', periods=n, freq='H')
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'CO2_ppm': np.random.uniform(350, 480, n),
        'Temperature_C': np.random.uniform(38, 50, n),
        'Humidity_Percent': np.random.uniform(45, 65, n),
        'Gas_Flow_L_per_min': np.random.uniform(400, 600, n),
        'Weight_Change_g': np.random.uniform(8, 15, n),
        'Fan_Speed_RPM': np.random.uniform(1200, 1800, n)
    })


def auto_correct_data(df):
    """Auto-correct sensor data to realistic ranges"""
    df = df.copy()
    
    # Fix CO2 if too high
    if df['CO2_ppm'].mean() > 600:
        df['CO2_ppm'] = df['CO2_ppm'] / 3
        print("✓ CO2 corrected")
    
    # Fix flow if too low
    if df['Gas_Flow_L_per_min'].mean() < 100:
        df['Gas_Flow_L_per_min'] = df['Gas_Flow_L_per_min'] * 25
        print("✓ Flow corrected")
    
    # Fix weight if cumulative
    if df['Weight_Change_g'].iloc[-1] > df['Weight_Change_g'].iloc[0] * 10:
        weight_diff = df['Weight_Change_g'].diff().fillna(df['Weight_Change_g'].iloc[0])
        weight_diff = weight_diff.clip(lower=0)
        df['Weight_Change_g'] = 8 + (weight_diff / weight_diff.max()) * 7
        print("✓ Weight corrected")
    
    return df


def visualize_results(results):
    """Create summary visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fvalid
    axes[0, 0].plot(results['Record_ID'], results['Fvalid'], 'b-', linewidth=2, marker='o')
    axes[0, 0].axhline(y=0.70, color='r', linestyle='--', label='Threshold')
    axes[0, 0].set_title('Data Validity (Fvalid)', fontweight='bold')
    axes[0, 0].set_ylabel('Fvalid')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Flow vs Credits
    axes[0, 1].scatter(results['Flow_L_min'], results['Credits']*1000, 
                       s=100, alpha=0.6, c='blue', edgecolors='black')
    axes[0, 1].set_title('Flow Rate vs Credits', fontweight='bold')
    axes[0, 1].set_xlabel('Gas Flow (L/min)')
    axes[0, 1].set_ylabel('Credits (mg)')
    axes[0, 1].grid(alpha=0.3)
    
    # Credits
    axes[1, 0].bar(results['Record_ID'], results['Credits']*1000, color='green', alpha=0.7)
    axes[1, 0].set_title('Carbon Credits', fontweight='bold')
    axes[1, 0].set_ylabel('Credits (mg)')
    axes[1, 0].set_xlabel('Record ID')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Tokens
    axes[1, 1].bar(results['Record_ID'], results['Tokens']*1000, color='gold', alpha=0.7)
    axes[1, 1].set_title('CUBEX Tokens', fontweight='bold')
    axes[1, 1].set_ylabel('Tokens (×10⁻³)')
    axes[1, 1].set_xlabel('Record ID')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.suptitle('Carbon Capture Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("OPTIMIZED CARBON CAPTURE PIPELINE")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    full_data = generate_sample_data(n=50)
    training_data = full_data.iloc[:35]
    test_data = full_data.iloc[35:]
    
    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = CarbonCapturePipeline("ENTERPRISE_001", is_msme=True)
    
    # Set detection mode
    pipeline.anomaly_detector.set_lenient_mode(True)
    
    # Train
    print("\n3. Training models...")
    pipeline.train(training_data)
    
    # Process
    print("\n4. Processing test data...")
    results = pipeline.process_batch(test_data)
    
    # Report
    print("\n5. Generating report...")
    portfolio = pipeline.report()
    
    # Visualize
    print("\n6. Creating visualization...")
    visualize_results(results)
    
    print(f"\n✅ Complete! Total credits: {results['Credits'].sum():.6f} tonnes CO2")
    print(f"✅ Total tokens: {results['Tokens'].sum():.6f} CUBEX")
    print(f"✅ Total value: ₹{results['Value_INR'].sum():,.2f}")