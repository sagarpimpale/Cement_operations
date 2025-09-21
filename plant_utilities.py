import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import google.generativeai as genai
import json
import os
import warnings
warnings.filterwarnings('ignore')

class PlantUtilitiesOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.scaler = StandardScaler()
        self.power_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.load_optimizer = KMeans(n_clusters=4, random_state=42)
        
    def load_and_process_data(self, csv_path):
        """Load and process plant utilities data"""
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=600):
        """Create realistic plant utilities operational data"""
        np.random.seed(42)
        
        # Production load factor (affects all utilities)
        production_load = np.random.normal(85, 12, n_samples)
        production_load = np.clip(production_load, 60, 100)
        
        # Raw Mill System
        raw_mill_load = production_load + np.random.normal(0, 8, n_samples)
        raw_mill_load = np.clip(raw_mill_load, 50, 100)
        
        raw_mill_power = (2800 * (raw_mill_load / 100) ** 1.2 + 
                         np.random.normal(0, 150, n_samples))
        raw_mill_power = np.clip(raw_mill_power, 1800, 3500)  # kW
        
        # Coal Mill System
        coal_mill_load = production_load * 0.9 + np.random.normal(0, 6, n_samples)
        coal_mill_load = np.clip(coal_mill_load, 45, 100)
        
        coal_mill_power = (1200 * (coal_mill_load / 100) ** 1.15 + 
                          np.random.normal(0, 80, n_samples))
        coal_mill_power = np.clip(coal_mill_power, 700, 1500)  # kW
        
        # Kiln System (main drive, ID fan, etc.)
        kiln_drive_power = (850 * (production_load / 100) ** 1.1 + 
                           np.random.normal(0, 50, n_samples))
        kiln_drive_power = np.clip(kiln_drive_power, 600, 1000)  # kW
        
        # ID Fan (Induced Draft Fan)
        id_fan_load = production_load + np.random.normal(0, 5, n_samples)
        id_fan_load = np.clip(id_fan_load, 60, 100)
        
        id_fan_power = (2200 * (id_fan_load / 100) ** 2.8 + 
                       np.random.normal(0, 120, n_samples))
        id_fan_power = np.clip(id_fan_power, 1200, 3200)  # kW
        
        # Cement Mill System
        cement_mill_load = production_load + np.random.normal(0, 10, n_samples)
        cement_mill_load = np.clip(cement_mill_load, 50, 100)
        
        cement_mill_power = (4200 * (cement_mill_load / 100) ** 1.25 + 
                            np.random.normal(0, 200, n_samples))
        cement_mill_power = np.clip(cement_mill_power, 2500, 5000)  # kW
        
        # Bag Filters / ESP
        baghouse_power = (320 * (production_load / 100) ** 1.1 + 
                         np.random.normal(0, 20, n_samples))
        baghouse_power = np.clip(baghouse_power, 200, 400)  # kW
        
        # Compressor Systems
        compressor_load = np.random.normal(75, 15, n_samples)
        compressor_load = np.clip(compressor_load, 40, 100)
        
        compressor_power = (650 * (compressor_load / 100) ** 1.3 + 
                           np.random.normal(0, 40, n_samples))
        compressor_power = np.clip(compressor_power, 300, 800)  # kW
        
        # Conveyor Systems
        conveyor_power = (180 * (production_load / 100) + 
                         np.random.normal(0, 15, n_samples))
        conveyor_power = np.clip(conveyor_power, 120, 220)  # kW
        
        # Auxiliary Systems (lighting, cooling, misc)
        auxiliary_power = np.random.normal(450, 80, n_samples)
        auxiliary_power = np.clip(auxiliary_power, 300, 650)  # kW
        
        # Total Power Consumption
        total_power = (raw_mill_power + coal_mill_power + kiln_drive_power + 
                      id_fan_power + cement_mill_power + baghouse_power + 
                      compressor_power + conveyor_power + auxiliary_power)
        
        # Power Factor
        power_factor = np.random.normal(0.88, 0.06, n_samples)
        power_factor = np.clip(power_factor, 0.75, 0.95)
        
        # Specific Power Consumption (kWh/ton clinker)
        clinker_production = production_load * 2.5  # tons/hour approximate
        specific_power = total_power / clinker_production
        specific_power = np.clip(specific_power, 85, 130)
        
        # Cooling Water System
        cooling_water_flow = (1200 * (production_load / 100) + 
                             np.random.normal(0, 80, n_samples))
        cooling_water_flow = np.clip(cooling_water_flow, 800, 1500)  # m³/hr
        
        cooling_water_temp_in = np.random.normal(28, 4, n_samples)
        cooling_water_temp_in = np.clip(cooling_water_temp_in, 20, 35)  # °C
        
        cooling_water_temp_out = cooling_water_temp_in + np.random.normal(12, 3, n_samples)
        cooling_water_temp_out = np.clip(cooling_water_temp_out, 30, 45)  # °C
        
        # Compressed Air System
        air_pressure = np.random.normal(7.5, 0.8, n_samples)
        air_pressure = np.clip(air_pressure, 6.0, 9.0)  # bar
        
        air_consumption = (35 * (production_load / 100) + 
                          np.random.normal(0, 5, n_samples))
        air_consumption = np.clip(air_consumption, 20, 45)  # m³/min
        
        # Steam System (if applicable)
        steam_pressure = np.random.normal(12, 2, n_samples)
        steam_pressure = np.clip(steam_pressure, 8, 16)  # bar
        
        steam_consumption = (2.5 * (production_load / 100) + 
                            np.random.normal(0, 0.5, n_samples))
        steam_consumption = np.clip(steam_consumption, 1.5, 3.5)  # ton/hr
        
        # Energy Efficiency Metrics
        thermal_efficiency = (75 + (production_load - 85) * 0.15 + 
                             np.random.normal(0, 4, n_samples))
        thermal_efficiency = np.clip(thermal_efficiency, 65, 85)  # %
        
        # Power Quality Metrics
        voltage_variation = np.random.normal(0, 2.5, n_samples)  # % deviation from nominal
        voltage_variation = np.clip(voltage_variation, -5, 5)
        
        frequency_variation = np.random.normal(0, 0.2, n_samples)  # Hz deviation
        frequency_variation = np.clip(frequency_variation, -0.5, 0.5)
        
        # Maintenance Indicators
        vibration_raw_mill = np.random.normal(4.5, 1.2, n_samples)
        vibration_raw_mill = np.clip(vibration_raw_mill, 2, 8)  # mm/s RMS
        
        vibration_cement_mill = np.random.normal(5.2, 1.5, n_samples)
        vibration_cement_mill = np.clip(vibration_cement_mill, 2.5, 9)  # mm/s RMS
        
        bearing_temp_raw_mill = np.random.normal(65, 8, n_samples)
        bearing_temp_raw_mill = np.clip(bearing_temp_raw_mill, 45, 85)  # °C
        
        bearing_temp_cement_mill = np.random.normal(68, 10, n_samples)
        bearing_temp_cement_mill = np.clip(bearing_temp_cement_mill, 45, 90)  # °C
        
        # Cost Calculations
        electricity_cost_inr_kwh = np.random.normal(4.8, 0.5, n_samples)  # Industrial tariff
        electricity_cost_inr_kwh = np.clip(electricity_cost_inr_kwh, 3.5, 6.5)
        
        hourly_power_cost = total_power * electricity_cost_inr_kwh
        power_cost_per_ton = hourly_power_cost / clinker_production
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'production_load_pct': production_load,
            'raw_mill_load_pct': raw_mill_load,
            'raw_mill_power_kw': raw_mill_power,
            'coal_mill_load_pct': coal_mill_load,
            'coal_mill_power_kw': coal_mill_power,
            'kiln_drive_power_kw': kiln_drive_power,
            'id_fan_load_pct': id_fan_load,
            'id_fan_power_kw': id_fan_power,
            'cement_mill_load_pct': cement_mill_load,
            'cement_mill_power_kw': cement_mill_power,
            'baghouse_power_kw': baghouse_power,
            'compressor_load_pct': compressor_load,
            'compressor_power_kw': compressor_power,
            'conveyor_power_kw': conveyor_power,
            'auxiliary_power_kw': auxiliary_power,
            'total_power_kw': total_power,
            'power_factor': power_factor,
            'specific_power_kwh_ton': specific_power,
            'cooling_water_flow_m3_hr': cooling_water_flow,
            'cooling_water_temp_in_c': cooling_water_temp_in,
            'cooling_water_temp_out_c': cooling_water_temp_out,
            'compressed_air_pressure_bar': air_pressure,
            'compressed_air_consumption_m3_min': air_consumption,
            'steam_pressure_bar': steam_pressure,
            'steam_consumption_ton_hr': steam_consumption,
            'thermal_efficiency_pct': thermal_efficiency,
            'voltage_variation_pct': voltage_variation,
            'frequency_variation_hz': frequency_variation,
            'vibration_raw_mill_mm_s': vibration_raw_mill,
            'vibration_cement_mill_mm_s': vibration_cement_mill,
            'bearing_temp_raw_mill_c': bearing_temp_raw_mill,
            'bearing_temp_cement_mill_c': bearing_temp_cement_mill,
            'electricity_cost_inr_kwh': electricity_cost_inr_kwh,
            'power_cost_inr_ton': power_cost_per_ton
        })
    
    def train_models(self, df):
        """Train models for power consumption and efficiency prediction"""
        try:
            # Features for prediction
            features = ['production_load_pct', 'raw_mill_load_pct', 'coal_mill_load_pct',
                       'id_fan_load_pct', 'cement_mill_load_pct', 'compressor_load_pct',
                       'power_factor', 'thermal_efficiency_pct']
            
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 5:
                return False
                
            X = df[available_features].fillna(df[available_features].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Train total power predictor
            if 'total_power_kw' in df.columns:
                y_power = df['total_power_kw'].fillna(df['total_power_kw'].mean())
                self.power_predictor.fit(X_scaled, y_power)
            
            # Train specific power efficiency model
            if 'specific_power_kwh_ton' in df.columns:
                y_efficiency = df['specific_power_kwh_ton'].fillna(df['specific_power_kwh_ton'].mean())
                self.efficiency_model.fit(X_scaled, y_efficiency)
            
            # Train load pattern optimizer
            load_features = ['raw_mill_load_pct', 'cement_mill_load_pct', 'id_fan_load_pct', 'compressor_load_pct']
            available_load_features = [f for f in load_features if f in df.columns]
            
            if len(available_load_features) >= 3:
                load_data = df[available_load_features].fillna(df[available_load_features].mean())
                self.load_optimizer.fit(load_data)
            
            return True
            
        except Exception as e:
            print(f"Utilities model training error: {e}")
            return False
    
    def predict_utilities_performance(self, input_data):
        """Predict power consumption and efficiency metrics"""
        try:
            features = ['production_load_pct', 'raw_mill_load_pct', 'coal_mill_load_pct',
                       'id_fan_load_pct', 'cement_mill_load_pct', 'compressor_load_pct',
                       'power_factor', 'thermal_efficiency_pct']
            
            X = []
            for feature in features:
                X.append(input_data.get(feature, 0))
            
            X = np.array(X).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            total_power_pred = self.power_predictor.predict(X_scaled)[0]
            specific_power_pred = self.efficiency_model.predict(X_scaled)[0]
            
            # Calculate additional metrics
            production_load = input_data.get('production_load_pct', 85)
            power_factor = input_data.get('power_factor', 0.88)
            
            # Reactive power calculation
            apparent_power = total_power_pred / power_factor
            reactive_power = np.sqrt(apparent_power**2 - total_power_pred**2)
            
            # Efficiency score (0-100)
            efficiency_score = min(100, (110 - specific_power_pred) * 4)  # Lower specific power = higher efficiency
            
            return {
                'predicted_total_power_kw': float(total_power_pred),
                'predicted_specific_power_kwh_ton': float(specific_power_pred),
                'apparent_power_kva': float(apparent_power),
                'reactive_power_kvar': float(reactive_power),
                'efficiency_score': float(efficiency_score)
            }
            
        except Exception as e:
            print(f"Utilities prediction error: {e}")
            return {
                'predicted_total_power_kw': 12000.0,
                'predicted_specific_power_kwh_ton': 105.0,
                'apparent_power_kva': 13636.0,
                'reactive_power_kvar': 6000.0,
                'efficiency_score': 75.0
            }

    def generate_ai_recommendations(self, current_data, prediction_results):
        """Generate AI-powered utilities optimization recommendations"""
        if not self.api_key:
            return self._rule_based_utilities_recommendations(current_data, prediction_results)
        
        try:
            # Prepare comprehensive utilities data context
            utilities_context = {
                'power_consumption': {
                    'total_power': current_data.get('total_power_kw', 12000),
                    'raw_mill_power': current_data.get('raw_mill_power_kw', 2800),
                    'cement_mill_power': current_data.get('cement_mill_power_kw', 4200),
                    'id_fan_power': current_data.get('id_fan_power_kw', 2200),
                    'predicted_total_power': prediction_results.get('predicted_total_power_kw', 12000),
                    'specific_power': prediction_results.get('predicted_specific_power_kwh_ton', 105)
                },
                'system_loads': {
                    'production_load': current_data.get('production_load_pct', 85),
                    'raw_mill_load': current_data.get('raw_mill_load_pct', 85),
                    'cement_mill_load': current_data.get('cement_mill_load_pct', 85),
                    'id_fan_load': current_data.get('id_fan_load_pct', 85),
                    'compressor_load': current_data.get('compressor_load_pct', 75)
                },
                'power_quality': {
                    'power_factor': current_data.get('power_factor', 0.88),
                    'voltage_variation': current_data.get('voltage_variation_pct', 0),
                    'frequency_variation': current_data.get('frequency_variation_hz', 0),
                    'reactive_power': prediction_results.get('reactive_power_kvar', 6000)
                },
                'auxiliary_systems': {
                    'cooling_water_flow': current_data.get('cooling_water_flow_m3_hr', 1200),
                    'compressed_air_pressure': current_data.get('compressed_air_pressure_bar', 7.5),
                    'steam_consumption': current_data.get('steam_consumption_ton_hr', 2.5)
                },
                'targets': {
                    'specific_power_target': 95,  # kWh/ton clinker
                    'power_factor_target': 0.95,
                    'energy_efficiency_target': 85,
                    'cost_reduction_target': 15  # % reduction
                }
            }
            
            prompt = f"""
            You are an expert AI system for cement plant utilities optimization with deep knowledge of 
            industrial power systems, energy efficiency, and plant automation.

            CURRENT UTILITIES PERFORMANCE:
            Power Consumption Analysis:
            - Total Plant Power: {utilities_context['power_consumption']['total_power']:.0f} kW
            - Raw Mill Power: {utilities_context['power_consumption']['raw_mill_power']:.0f} kW
            - Cement Mill Power: {utilities_context['power_consumption']['cement_mill_power']:.0f} kW  
            - ID Fan Power: {utilities_context['power_consumption']['id_fan_power']:.0f} kW
            - Predicted Total Power: {utilities_context['power_consumption']['predicted_total_power']:.0f} kW
            - Specific Power Consumption: {utilities_context['power_consumption']['specific_power']:.1f} kWh/ton (target: <95)

            System Load Analysis:
            - Production Load: {utilities_context['system_loads']['production_load']:.1f}%
            - Raw Mill Load: {utilities_context['system_loads']['raw_mill_load']:.1f}%
            - Cement Mill Load: {utilities_context['system_loads']['cement_mill_load']:.1f}%
            - ID Fan Load: {utilities_context['system_loads']['id_fan_load']:.1f}%
            - Compressor Load: {utilities_context['system_loads']['compressor_load']:.1f}%

            Power Quality Metrics:
            - Power Factor: {utilities_context['power_quality']['power_factor']:.3f} (target: >0.95)
            - Voltage Variation: {utilities_context['power_quality']['voltage_variation']:.1f}% (limit: ±5%)
            - Frequency Variation: {utilities_context['power_quality']['frequency_variation']:.2f} Hz (limit: ±0.5Hz)
            - Reactive Power: {utilities_context['power_quality']['reactive_power']:.0f} kVAR

            Auxiliary Systems:
            - Cooling Water Flow: {utilities_context['auxiliary_systems']['cooling_water_flow']:.0f} m³/hr
            - Compressed Air Pressure: {utilities_context['auxiliary_systems']['compressed_air_pressure']:.1f} bar
            - Steam Consumption: {utilities_context['auxiliary_systems']['steam_consumption']:.1f} ton/hr

            OPTIMIZATION TARGETS:
            - Specific Power: <95 kWh/ton clinker (current: {utilities_context['power_consumption']['specific_power']:.1f})
            - Power Factor: >0.95 (current: {utilities_context['power_quality']['power_factor']:.3f})
            - Energy Cost Reduction: 15% target
            - System Reliability: 99.5% uptime

            INDUSTRIAL BEST PRACTICES:
            - Load factor optimization for maximum efficiency
            - Power factor correction for cost savings
            - Variable frequency drives for energy conservation
            - Waste heat recovery integration
            - Predictive maintenance for optimal performance

            Provide specific, technical recommendations in these categories:
            1. ENERGY_EFFICIENCY: Power consumption optimization strategies
            2. LOAD_MANAGEMENT: System load balancing and optimization
            3. POWER_QUALITY: Power factor correction and electrical optimization
            4. AUXILIARY_SYSTEMS: Cooling, compressed air, and steam optimization
            5. MAINTENANCE_OPTIMIZATION: Predictive maintenance and reliability
            6. COST_REDUCTION: Energy cost savings and demand management

            Each recommendation should include:
            - Specific technical parameters and targets
            - Expected power savings and cost reduction
            - Implementation priority (IMMEDIATE/HIGH/MEDIUM/LOW)
            - ROI estimation and payback period
            - Operational impact and risk assessment

            Format response as JSON with detailed technical recommendations.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    recommendations = json.loads(response_text[json_start:json_end])
                    
                    # Add utilities optimization metrics
                    recommendations['optimization_metrics'] = {
                        'energy_efficiency_gap_pct': max(0, utilities_context['power_consumption']['specific_power'] - 95),
                        'power_factor_improvement_needed': max(0, 0.95 - utilities_context['power_quality']['power_factor']),
                        'power_savings_potential_kw': max(0, utilities_context['power_consumption']['total_power'] * 0.15),
                        'cost_savings_potential_inr_day': max(0, utilities_context['power_consumption']['total_power'] * 24 * 4.8 * 0.15)
                    }
                    
                    return recommendations
            except json.JSONDecodeError:
                pass
                
            return {
                'ENERGY_EFFICIENCY': f"Optimize specific power from {utilities_context['power_consumption']['specific_power']:.1f} to 95 kWh/ton",
                'POWER_QUALITY': f"Improve power factor from {utilities_context['power_quality']['power_factor']:.3f} to 0.95",
                'LOAD_MANAGEMENT': "Balance mill loads for optimal efficiency",
                'ai_insights': response.text[:400] + "..." if len(response.text) > 400 else response.text
            }
            
        except Exception as e:
            print(f"AI utilities recommendation error: {e}")
            return self._rule_based_utilities_recommendations(current_data, prediction_results)
    
    def _rule_based_utilities_recommendations(self, current_data, prediction_results):
        """Expert rule-based utilities optimization recommendations"""
        recommendations = {
            'ENERGY_EFFICIENCY': [],
            'LOAD_MANAGEMENT': [],
            'POWER_QUALITY': [],
            'AUXILIARY_SYSTEMS': [],
            'MAINTENANCE_OPTIMIZATION': [],
            'COST_REDUCTION': []
        }
        
        # Extract values
        total_power = current_data.get('total_power_kw', 12000)
        specific_power = prediction_results.get('predicted_specific_power_kwh_ton', 105)
        power_factor = current_data.get('power_factor', 0.88)
        raw_mill_load = current_data.get('raw_mill_load_pct', 85)
        cement_mill_load = current_data.get('cement_mill_load_pct', 85)
        id_fan_load = current_data.get('id_fan_load_pct', 85)
        compressor_load = current_data.get('compressor_load_pct', 75)
        vibration_raw_mill = current_data.get('vibration_raw_mill_mm_s', 4.5)
        vibration_cement_mill = current_data.get('vibration_cement_mill_mm_s', 5.2)
        
        # Energy efficiency recommendations
        if specific_power > 105:
            power_gap = specific_power - 95
            recommendations['ENERGY_EFFICIENCY'].append(
                f"⚡ HIGH: Specific power at {specific_power:.1f} kWh/ton - optimize to <95 (gap: {power_gap:.1f})"
            )
            recommendations['ENERGY_EFFICIENCY'].append(
                f"🔧 Install VFDs on major motors for 8-12% energy savings"
            )
        elif specific_power <= 100:
            recommendations['ENERGY_EFFICIENCY'].append(
                f"🌟 EXCELLENT: Specific power at {specific_power:.1f} kWh/ton - maintain efficiency"
            )
        
        if total_power > 13000:
            recommendations['ENERGY_EFFICIENCY'].append(
                f"📊 HIGH: Total power consumption ({total_power:.0f} kW) - implement load shedding strategies"
            )
            recommendations['ENERGY_EFFICIENCY'].append(
                f"🔄 Consider waste heat recovery for auxiliary power generation"
            )
        
        # Load management
        if raw_mill_load > 95:
            recommendations['LOAD_MANAGEMENT'].append(
                f"⚙️ HIGH: Raw mill overloaded ({raw_mill_load:.1f}%) - optimize to 85-90% for efficiency"
            )
        elif raw_mill_load < 70:
            recommendations['LOAD_MANAGEMENT'].append(
                f"📈 MEDIUM: Raw mill underutilized ({raw_mill_load:.1f}%) - increase throughput"
            )
        
        if cement_mill_load > 95:
            recommendations['LOAD_MANAGEMENT'].append(
                f"🏭 HIGH: Cement mill overloaded ({cement_mill_load:.1f}%) - risk of motor overheating"
            )
        
        if id_fan_load > 90:
            recommendations['LOAD_MANAGEMENT'].append(
                f"💨 MEDIUM: ID fan high load ({id_fan_load:.1f}%) - check system resistance"
            )
        
        if compressor_load < 60:
            recommendations['LOAD_MANAGEMENT'].append(
                f"🔧 LOW: Compressor underloaded ({compressor_load:.1f}%) - optimize air demand"
            )
        
        # Power quality improvements
        if power_factor < 0.90:
            pf_gap = 0.95 - power_factor
            recommendations['POWER_QUALITY'].append(
                f"⚡ HIGH: Power factor at {power_factor:.3f} - install capacitors for >0.95 (gap: {pf_gap:.3f})"
            )
            recommendations['COST_REDUCTION'].append(
                f"💰 PF correction can save ₹{total_power * 24 * 30 * 1.2:.0f}/month in penalty charges"
            )
        elif power_factor >= 0.95:
            recommendations['POWER_QUALITY'].append(
                f"✅ EXCELLENT: Power factor at {power_factor:.3f} - optimal electrical efficiency"
            )
        
        recommendations['POWER_QUALITY'].extend([
            f"📊 Implement power quality monitoring for voltage/frequency stability",
            f"🔧 Install harmonic filters if THD >5% on major motors",
            f"⚡ Consider active power factor correction for dynamic loads"
        ])
        
        # Auxiliary systems optimization
        cooling_water_flow = current_data.get('cooling_water_flow_m3_hr', 1200)
        air_pressure = current_data.get('compressed_air_pressure_bar', 7.5)
        
        if cooling_water_flow > 1400:
            recommendations['AUXILIARY_SYSTEMS'].append(
                f"💧 MEDIUM: High cooling water flow ({cooling_water_flow:.0f} m³/hr) - optimize pump operation"
            )
        
        if air_pressure > 8.0:
            recommendations['AUXILIARY_SYSTEMS'].append(
                f"💨 MEDIUM: High air pressure ({air_pressure:.1f} bar) - reduce to 7.0-7.5 bar for energy savings"
            )
            recommendations['COST_REDUCTION'].append(
                f"🔧 Every 1 bar pressure reduction saves ~7% compressor energy"
            )
        
        recommendations['AUXILIARY_SYSTEMS'].extend([
            f"🌡️ Implement cooling tower optimization for water temperature control",
            f"💧 Install air leak detection system - 30% savings potential",
            f"⚙️ Optimize steam system pressure for minimum energy consumption"
        ])
        
        # Maintenance optimization
        maintenance_actions = []
        if vibration_raw_mill > 6.0:
            maintenance_actions.append(
                f"🔧 HIGH: Raw mill vibration ({vibration_raw_mill:.1f} mm/s) - check bearing alignment"
            )
        
        if vibration_cement_mill > 7.0:
            maintenance_actions.append(
                f"⚙️ HIGH: Cement mill vibration ({vibration_cement_mill:.1f} mm/s) - inspect mill internals"
            )
        
        bearing_temp_raw = current_data.get('bearing_temp_raw_mill_c', 65)
        bearing_temp_cement = current_data.get('bearing_temp_cement_mill_c', 68)
        
        if bearing_temp_raw > 75:
            maintenance_actions.append(
                f"🌡️ HIGH: Raw mill bearing temp ({bearing_temp_raw:.0f}°C) - check lubrication"
            )
        
        if bearing_temp_cement > 80:
            maintenance_actions.append(
                f"🔥 CRITICAL: Cement mill bearing temp ({bearing_temp_cement:.0f}°C) - immediate attention required"
            )
        
        maintenance_actions.extend([
            f"📱 Implement vibration monitoring with 4-20mA transmitters",
            f"🛠️ Schedule quarterly thermography inspections",
            f"⚙️ Oil analysis program for critical equipment",
            f"📊 Motor current signature analysis for early fault detection"
        ])
        
        recommendations['MAINTENANCE_OPTIMIZATION'] = maintenance_actions
        
        # Cost reduction strategies
        cost_actions = [
            f"⏰ Implement time-of-use tariff optimization (30% demand charge savings)",
            f"📊 Power factor improvement: ₹{total_power * 0.5 * 24 * 30:.0f}/month savings potential",
            f"🔋 Consider battery storage for peak shaving (15% demand cost reduction)",
            f"💡 LED lighting retrofit: 60% lighting energy savings"
        ]
        
        if specific_power > 100:
            energy_savings_potential = (specific_power - 95) / specific_power * 100
            cost_savings_daily = total_power * 24 * 4.8 * (energy_savings_potential / 100)
            cost_actions.append(
                f"⚡ Energy efficiency: ₹{cost_savings_daily:.0f}/day savings potential ({energy_savings_potential:.1f}%)"
            )
        
        recommendations['COST_REDUCTION'] = cost_actions
        
        return recommendations
    
    def calculate_utilities_performance(self, df):
        """Calculate comprehensive utilities performance metrics"""
        try:
            # Power consumption analysis
            avg_total_power = df['total_power_kw'].mean() if 'total_power_kw' in df.columns else 12000
            avg_specific_power = df['specific_power_kwh_ton'].mean() if 'specific_power_kwh_ton' in df.columns else 105
            power_variability = df['total_power_kw'].std() if 'total_power_kw' in df.columns else 800
            
            # Power quality metrics
            avg_power_factor = df['power_factor'].mean() if 'power_factor' in df.columns else 0.88
            pf_consistency = df['power_factor'].std() if 'power_factor' in df.columns else 0.03
            
            # System efficiency
            if 'thermal_efficiency_pct' in df.columns:
                avg_thermal_efficiency = df['thermal_efficiency_pct'].mean()
            else:
                avg_thermal_efficiency = 75
            
            # Load factor analysis
            load_columns = ['raw_mill_load_pct', 'cement_mill_load_pct', 'id_fan_load_pct']
            available_loads = [col for col in load_columns if col in df.columns]
            
            if available_loads:
                avg_load_factor = df[available_loads].mean().mean()
                load_balance = df[available_loads].std().mean()
            else:
                avg_load_factor, load_balance = 85, 8
            
            # Cost analysis
            if 'power_cost_inr_ton' in df.columns:
                avg_power_cost = df['power_cost_inr_ton'].mean()
            else:
                avg_power_cost = avg_total_power * 4.8 / (avg_load_factor * 2.5)  # Estimated
            
            # Maintenance indicators
            maintenance_score = 100
            if 'vibration_raw_mill_mm_s' in df.columns:
                avg_vibration = df['vibration_raw_mill_mm_s'].mean()
                if avg_vibration > 6:
                    maintenance_score -= 15
                elif avg_vibration > 4.5:
                    maintenance_score -= 5
            
            # Performance targets achievement
            targets = {
                'specific_power_target': 95,
                'power_factor_target': 0.95,
                'thermal_efficiency_target': 80,
                'load_factor_target': 85
            }
            
            target_achievement = {
                'specific_power': (95 / max(avg_specific_power, 95)) * 100,
                'power_factor': (avg_power_factor / 0.95) * 100,
                'thermal_efficiency': (avg_thermal_efficiency / 80) * 100,
                'load_optimization': (avg_load_factor / 85) * 100
            }
            
            overall_performance = sum(target_achievement.values()) / len(target_achievement)
            
            # Economic impact calculations
            annual_energy_mwh = avg_total_power * 8760 / 1000  # Assuming 100% uptime
            annual_energy_cost_lakhs = annual_energy_mwh * 1000 * 4.8 / 100000
            
            # Improvement potential
            energy_savings_potential = max(0, avg_specific_power - 95)
            pf_improvement_potential = max(0, 0.95 - avg_power_factor)
            cost_savings_potential = energy_savings_potential / avg_specific_power * annual_energy_cost_lakhs
            
            return {
                'power_performance': {
                    'avg_total_power_kw': avg_total_power,
                    'avg_specific_power_kwh_ton': avg_specific_power,
                    'power_variability_kw': power_variability,
                    'avg_power_cost_inr_ton': avg_power_cost
                },
                'power_quality': {
                    'avg_power_factor': avg_power_factor,
                    'power_factor_consistency': pf_consistency,
                    'power_quality_score': min(100, avg_power_factor / 0.95 * 100)
                },
                'system_efficiency': {
                    'avg_thermal_efficiency_pct': avg_thermal_efficiency,
                    'avg_load_factor_pct': avg_load_factor,
                    'load_balance_score': max(0, 100 - load_balance * 5)
                },
                'target_achievement': target_achievement,
                'overall_performance_score': overall_performance,
                'maintenance_health_score': maintenance_score,
                'economic_metrics': {
                    'annual_energy_consumption_mwh': annual_energy_mwh,
                    'annual_energy_cost_inr_lakhs': annual_energy_cost_lakhs,
                    'energy_savings_potential_kwh_ton': energy_savings_potential,
                    'cost_savings_potential_inr_lakhs': cost_savings_potential
                },
                'performance_rating': (
                    'EXCELLENT' if overall_performance >= 95 else
                    'GOOD' if overall_performance >= 85 else
                    'SATISFACTORY' if overall_performance >= 75 else
                    'NEEDS_IMPROVEMENT'
                )
            }
            
        except Exception as e:
            print(f"Utilities performance calculation error: {e}")
            return {
                'power_performance': {'error': 'Calculation failed'},
                'power_quality': {},
                'system_efficiency': {},
                'target_achievement': {},
                'overall_performance_score': 0,
                'maintenance_health_score': 0,
                'economic_metrics': {},
                'performance_rating': 'UNKNOWN'
            }