import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import google.generativeai as genai
import json
import os
import warnings
warnings.filterwarnings('ignore')

class RawMaterialGrindingOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.scaler = StandardScaler()
        self.grinding_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def load_and_process_data(self, csv_path):
        """Load and process raw material and grinding data"""
        try:
            df = pd.read_csv(csv_path)
            # Convert timestamp if exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=500):
        """Create realistic sample data for raw materials and grinding"""
        np.random.seed(42)
        
        # Generate correlated data that mimics real cement plant operations
        limestone_purity = np.random.normal(88, 3, n_samples)
        limestone_purity = np.clip(limestone_purity, 80, 95)
        
        # Moisture affects grinding efficiency
        moisture = np.random.normal(4.5, 1.2, n_samples)
        moisture = np.clip(moisture, 1, 8)
        
        # Material hardness varies with geological source
        hardness = np.random.normal(10, 2.5, n_samples) 
        hardness = np.clip(hardness, 6, 18)
        
        # Clay reactivity affects clinker formation
        clay_alumina = np.random.normal(15, 2, n_samples)
        clay_alumina = np.clip(clay_alumina, 10, 20)
        
        # Iron ore content
        iron_content = np.random.normal(3.2, 0.8, n_samples)
        iron_content = np.clip(iron_content, 2, 5)
        
        # Silica affects strength development
        silica_content = np.random.normal(12.5, 1.5, n_samples)
        silica_content = np.clip(silica_content, 8, 16)
        
        # Feed rate affects grinding efficiency
        feed_rate = np.random.normal(52, 8, n_samples)
        feed_rate = np.clip(feed_rate, 35, 70)
        
        # Mill load depends on material properties
        mill_load = 70 + (hardness - 10) * 2 + (moisture - 4.5) * 3 + np.random.normal(0, 5, n_samples)
        mill_load = np.clip(mill_load, 60, 95)
        
        # Grinding energy is function of material properties
        grinding_energy = (75 + (hardness - 10) * 3.5 + 
                          (moisture - 4.5) * 2.8 + 
                          (feed_rate - 52) * 0.4 +
                          np.random.normal(0, 8, n_samples))
        grinding_energy = np.clip(grinding_energy, 45, 120)
        
        # Blaine fineness (cement fineness)
        blaine_fineness = (3400 - (grinding_energy - 85) * 15 + 
                          np.random.normal(0, 200, n_samples))
        blaine_fineness = np.clip(blaine_fineness, 2800, 4200)
        
        # Particle size distribution
        particle_size_90 = np.random.normal(18, 3, n_samples)
        particle_size_90 = np.clip(particle_size_90, 12, 28)
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='2H')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'limestone_purity_pct': limestone_purity,
            'moisture_content_pct': moisture,
            'hardness_index': hardness,
            'clay_alumina_pct': clay_alumina,
            'iron_ore_fe2o3_pct': iron_content,
            'silica_sand_sio2_pct': silica_content,
            'feed_rate_tons_hr': feed_rate,
            'mill_load_pct': mill_load,
            'grinding_energy_kwh_ton': grinding_energy,
            'blaine_fineness_cm2_g': blaine_fineness,
            'particle_size_d90_micron': particle_size_90
        })
    
    def train_models(self, df):
        """Train predictive models for grinding optimization"""
        # Features for prediction
        features = ['limestone_purity_pct', 'moisture_content_pct', 'hardness_index',
                   'clay_alumina_pct', 'iron_ore_fe2o3_pct', 'silica_sand_sio2_pct',
                   'feed_rate_tons_hr', 'mill_load_pct']
        
        # Target variables
        targets = ['grinding_energy_kwh_ton', 'blaine_fineness_cm2_g']
        
        available_features = [f for f in features if f in df.columns]
        available_targets = [t for t in targets if t in df.columns]
        
        if not available_features or not available_targets:
            return False
        
        X = df[available_features].fillna(df[available_features].mean())
        
        # Train grinding energy predictor
        if 'grinding_energy_kwh_ton' in available_targets:
            y_energy = df['grinding_energy_kwh_ton'].fillna(df['grinding_energy_kwh_ton'].mean())
            self.grinding_model.fit(X, y_energy)
        
        # Train anomaly detector
        X_scaled = self.scaler.fit_transform(X)
        self.anomaly_detector.fit(X_scaled)
        
        return True
    
    def predict_grinding_performance(self, input_data):
        """Predict grinding energy and detect anomalies"""
        try:
            features = ['limestone_purity_pct', 'moisture_content_pct', 'hardness_index',
                       'clay_alumina_pct', 'iron_ore_fe2o3_pct', 'silica_sand_sio2_pct',
                       'feed_rate_tons_hr', 'mill_load_pct']
            
            # Prepare input
            X = []
            for feature in features:
                X.append(input_data.get(feature, 0))
            
            X = np.array(X).reshape(1, -1)
            
            # Predictions
            energy_pred = self.grinding_model.predict(X)[0]
            X_scaled = self.scaler.transform(X)
            anomaly_score = self.anomaly_detector.score_samples(X_scaled)[0]
            
            return {
                'predicted_energy': float(energy_pred),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': anomaly_score < -0.5
            }
        except:
            return {
                'predicted_energy': 85.0,
                'anomaly_score': 0.0,
                'is_anomaly': False
            }
    
    def generate_ai_recommendations(self, current_data, prediction_results):
        """Generate AI-powered recommendations for raw material and grinding optimization"""
        if not self.api_key:
            return self._rule_based_recommendations(current_data, prediction_results)
        
        try:
            # Prepare comprehensive data summary
            data_context = {
                'current_conditions': {
                    'limestone_purity': current_data.get('limestone_purity_pct', 88),
                    'moisture_content': current_data.get('moisture_content_pct', 5),
                    'hardness_index': current_data.get('hardness_index', 10),
                    'feed_rate': current_data.get('feed_rate_tons_hr', 50),
                    'mill_load': current_data.get('mill_load_pct', 75),
                    'predicted_energy': prediction_results['predicted_energy'],
                    'anomaly_detected': prediction_results['is_anomaly']
                },
                'industry_benchmarks': {
                    'target_energy': 78,  # kWh/ton for raw meal preparation
                    'limestone_purity_min': 85,
                    'moisture_max': 4,
                    'optimal_feed_rate': '45-55 tons/hr',
                    'mill_load_optimal': '75-85%'
                }
            }
            
            prompt = f"""
            You are an expert AI system for cement plant raw material and grinding optimization. 
            Analyze the current operational data and provide specific, actionable recommendations.

            CURRENT OPERATIONAL DATA:
            - Raw Material Quality:
              * Limestone Purity: {data_context['current_conditions']['limestone_purity']:.1f}% (target: >85%)
              * Moisture Content: {data_context['current_conditions']['moisture_content']:.1f}% (target: <4%)
              * Material Hardness: {data_context['current_conditions']['hardness_index']:.1f} HI
            
            - Grinding Operations:
              * Feed Rate: {data_context['current_conditions']['feed_rate']:.1f} tons/hr
              * Mill Load: {data_context['current_conditions']['mill_load']:.1f}%
              * Predicted Energy: {data_context['current_conditions']['predicted_energy']:.1f} kWh/ton
              * Anomaly Status: {"DETECTED" if data_context['current_conditions']['anomaly_detected'] else "NORMAL"}

            OPTIMIZATION TARGETS:
            - Energy consumption: <78 kWh/ton for raw meal preparation
            - Limestone purity: >85% for consistent clinker quality  
            - Moisture control: <4% to minimize energy losses
            - Mill efficiency: 75-85% load for optimal grinding

            Provide specific recommendations in these categories:
            1. RAW_MATERIAL_SOURCING: Immediate actions for material quality
            2. MOISTURE_MANAGEMENT: Drying and handling optimization
            3. GRINDING_EFFICIENCY: Mill operation and circuit optimization
            4. ENERGY_REDUCTION: Specific energy-saving measures
            5. QUALITY_CONTROL: Monitoring and adjustment protocols
            6. PREDICTIVE_ACTIONS: Preventive measures based on current trends

            Format response as JSON with actionable recommendations for each category.
            Each recommendation should include specific parameters, expected impact, and implementation priority.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse AI response
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    recommendations = json.loads(response_text[json_start:json_end])
                    
                    # Add performance metrics
                    recommendations['performance_metrics'] = {
                        'current_energy_efficiency': 78 / data_context['current_conditions']['predicted_energy'] * 100,
                        'material_quality_score': min(100, data_context['current_conditions']['limestone_purity'] / 85 * 100),
                        'optimization_potential': max(0, data_context['current_conditions']['predicted_energy'] - 78)
                    }
                    
                    return recommendations
            except json.JSONDecodeError:
                pass
            
            # Fallback structured response
            return {
                'RAW_MATERIAL_SOURCING': f"Optimize limestone purity to >85% (current: {data_context['current_conditions']['limestone_purity']:.1f}%)",
                'GRINDING_EFFICIENCY': f"Adjust mill load to 80% (current: {data_context['current_conditions']['mill_load']:.1f}%)",
                'ENERGY_REDUCTION': f"Target energy reduction to 78 kWh/ton (current prediction: {data_context['current_conditions']['predicted_energy']:.1f})",
                'ai_insights': response.text[:300] + "..." if len(response.text) > 300 else response.text
            }
            
        except Exception as e:
            print(f"AI generation error: {e}")
            return self._rule_based_recommendations(current_data, prediction_results)
    
    def _rule_based_recommendations(self, current_data, prediction_results):
        """Comprehensive rule-based recommendations"""
        recommendations = {
            'RAW_MATERIAL_SOURCING': [],
            'MOISTURE_MANAGEMENT': [],
            'GRINDING_EFFICIENCY': [],
            'ENERGY_REDUCTION': [],
            'QUALITY_CONTROL': [],
            'PREDICTIVE_ACTIONS': []
        }
        
        limestone_purity = current_data.get('limestone_purity_pct', 88)
        moisture = current_data.get('moisture_content_pct', 5)
        hardness = current_data.get('hardness_index', 10)
        feed_rate = current_data.get('feed_rate_tons_hr', 50)
        mill_load = current_data.get('mill_load_pct', 75)
        predicted_energy = prediction_results['predicted_energy']
        
        # Raw material sourcing
        if limestone_purity < 85:
            recommendations['RAW_MATERIAL_SOURCING'].append(
                f"🔍 URGENT: Limestone purity at {limestone_purity:.1f}% - source higher grade limestone (target: >85%)"
            )
            recommendations['RAW_MATERIAL_SOURCING'].append(
                "🏗️ Implement limestone screening and beneficiation to remove impurities"
            )
        elif limestone_purity > 92:
            recommendations['RAW_MATERIAL_SOURCING'].append(
                "✅ Excellent limestone quality - maintain current quarry operations"
            )
        
        # Moisture management
        if moisture > 5:
            recommendations['MOISTURE_MANAGEMENT'].append(
                f"🔥 High moisture content ({moisture:.1f}%) - implement pre-heating to <4%"
            )
            recommendations['ENERGY_REDUCTION'].append(
                f"⚡ Moisture reduction could save {(moisture - 4) * 2.8:.1f} kWh/ton in grinding energy"
            )
        elif moisture < 2:
            recommendations['MOISTURE_MANAGEMENT'].append(
                "⚠️ Very low moisture may cause dust issues - monitor material handling systems"
            )
        
        # Grinding efficiency
        if mill_load > 85:
            recommendations['GRINDING_EFFICIENCY'].append(
                f"⚙️ Mill overloaded at {mill_load:.1f}% - reduce to 80-85% for optimal efficiency"
            )
        elif mill_load < 70:
            recommendations['GRINDING_EFFICIENCY'].append(
                f"📈 Mill underutilized at {mill_load:.1f}% - increase load to 75-80%"
            )
        
        if hardness > 13:
            recommendations['GRINDING_EFFICIENCY'].append(
                f"🔨 Hard material detected ({hardness:.1f} HI) - consider grinding aids or pre-conditioning"
            )
        
        if feed_rate > 58:
            recommendations['GRINDING_EFFICIENCY'].append(
                f"🔄 High feed rate ({feed_rate:.1f} t/hr) may cause incomplete grinding - optimize to 45-55 t/hr"
            )
        
        # Energy reduction
        if predicted_energy > 85:
            recommendations['ENERGY_REDUCTION'].append(
                f"⚡ HIGH ENERGY PREDICTION ({predicted_energy:.1f} kWh/ton) - target <78 kWh/ton"
            )
            recommendations['ENERGY_REDUCTION'].append(
                "🔧 Optimize separator efficiency and mill ventilation for energy savings"
            )
        
        # Quality control
        recommendations['QUALITY_CONTROL'].extend([
            "📊 Monitor Blaine fineness every hour (target: 3400-3800 cm²/g)",
            "🧪 Track raw meal LSF (Lime Saturation Factor): 0.92-0.98",
            "📈 Maintain SiO2 modulus: 2.0-3.0 for optimal burnability"
        ])
        
        # Predictive actions
        if prediction_results['is_anomaly']:
            recommendations['PREDICTIVE_ACTIONS'].append(
                "🚨 ANOMALY DETECTED - immediate investigation of mill condition and material properties"
            )
            recommendations['PREDICTIVE_ACTIONS'].append(
                "🔍 Check mill liner condition and ball charge distribution"
            )
        
        recommendations['PREDICTIVE_ACTIONS'].extend([
            "🔮 Implement 2-hour ahead energy prediction for proactive adjustments",
            "📱 Set up automated alerts for material quality deviations"
        ])
        
        return recommendations

    def calculate_optimization_potential(self, df):
        """Calculate potential energy and cost savings"""
        if 'grinding_energy_kwh_ton' not in df.columns:
            return {}
        
        current_avg_energy = df['grinding_energy_kwh_ton'].mean()
        target_energy = 78  # Industry best practice
        
        energy_savings_kwh_ton = max(0, current_avg_energy - target_energy)
        
        # Assume 50 tons/hour average production
        annual_production_tons = 50 * 24 * 365  # 438,000 tons/year
        annual_energy_savings_mwh = energy_savings_kwh_ton * annual_production_tons / 1000
        
        # Energy cost: ₹4.5/kWh average industrial rate in India
        annual_cost_savings_inr = annual_energy_savings_mwh * 1000 * 4.5
        
        return {
            'current_energy_kwh_ton': current_avg_energy,
            'target_energy_kwh_ton': target_energy,
            'potential_savings_kwh_ton': energy_savings_kwh_ton,
            'annual_energy_savings_mwh': annual_energy_savings_mwh,
            'annual_cost_savings_inr_lakhs': annual_cost_savings_inr / 100000,  # Convert to lakhs
            'co2_reduction_tons_year': annual_energy_savings_mwh * 0.82  # CO2 factor for Indian grid
        }