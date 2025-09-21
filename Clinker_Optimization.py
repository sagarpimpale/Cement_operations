import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ClinkerOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.scaler = StandardScaler()
        self.fuel_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def load_and_process_data(self, csv_path):
        """Load and process clinker production data"""
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=400):
        """Create realistic clinker production data"""
        np.random.seed(42)
        
        # Kiln operating conditions
        kiln_temp = np.random.normal(1450, 20, n_samples)
        kiln_temp = np.clip(kiln_temp, 1420, 1480)
        
        # Oxygen levels affect combustion efficiency
        o2_level = np.random.normal(3.5, 0.8, n_samples)
        o2_level = np.clip(o2_level, 2.0, 6.0)
        
        # Feed rate affects residence time and heat transfer
        feed_rate = np.random.normal(45, 6, n_samples)
        feed_rate = np.clip(feed_rate, 35, 58)
        
        # Alternative fuel ratio (sustainability metric)
        alt_fuel_ratio = np.random.normal(20, 8, n_samples)
        alt_fuel_ratio = np.clip(alt_fuel_ratio, 5, 40)
        
        # Raw meal fineness affects burnability
        raw_meal_fineness = np.random.normal(3600, 300, n_samples)
        raw_meal_fineness = np.clip(raw_meal_fineness, 3000, 4200)
        
        # LSF (Lime Saturation Factor) - critical for clinker quality
        lsf = np.random.normal(0.95, 0.03, n_samples)
        lsf = np.clip(lsf, 0.88, 1.02)
        
        # Silica Modulus
        sm = np.random.normal(2.4, 0.3, n_samples)
        sm = np.clip(sm, 1.8, 3.2)
        
        # Alumina Modulus
        am = np.random.normal(1.4, 0.2, n_samples)
        am = np.clip(am, 1.0, 2.0)
        
        # Fuel consumption based on kiln conditions and efficiency
        base_fuel = 750  # kcal/kg clinker
        fuel_consumption = (base_fuel + 
                           (kiln_temp - 1450) * 0.8 +  # Higher temp = more fuel
                           (o2_level - 3.5) * 15 +      # Excess O2 = heat loss
                           (feed_rate - 45) * -2 +      # Higher feed = better efficiency
                           (alt_fuel_ratio - 20) * -1.5 + # Alt fuel slightly less efficient
                           np.random.normal(0, 40, n_samples))
        fuel_consumption = np.clip(fuel_consumption, 680, 850)
        
        # 28-day strength depends on clinker mineralogy and burning
        c3s_content = 55 + (lsf - 0.95) * 30 + np.random.normal(0, 3, n_samples)
        c3s_content = np.clip(c3s_content, 45, 68)
        
        strength_28_day = (45 + c3s_content * 0.8 + 
                          (kiln_temp - 1450) * 0.15 +
                          (raw_meal_fineness - 3600) * 0.003 +
                          np.random.normal(0, 4, n_samples))
        strength_28_day = np.clip(strength_28_day, 48, 75)
        
        # Free lime content (quality indicator)
        free_lime = (2.5 + (kiln_temp - 1450) * -0.03 +
                     (feed_rate - 45) * 0.08 +
                     (raw_meal_fineness - 3600) * -0.001 +
                     np.random.normal(0, 0.6, n_samples))
        free_lime = np.clip(free_lime, 0.5, 4.5)
        
        # CO2 emissions
        co2_emissions = (850 + fuel_consumption * 0.2 +  # Process + fuel CO2
                        (100 - alt_fuel_ratio) * 2 +     # More fossil fuel = more CO2
                        np.random.normal(0, 20, n_samples))
        co2_emissions = np.clip(co2_emissions, 800, 950)
        
        # NOx emissions
        nox_emissions = (600 + (kiln_temp - 1450) * 8 +
                        (o2_level - 3.5) * 30 +
                        np.random.normal(0, 80, n_samples))
        nox_emissions = np.clip(nox_emissions, 400, 900)
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='3H')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'kiln_temperature_celsius': kiln_temp,
            'oxygen_level_pct': o2_level,
            'feed_rate_tons_hr': feed_rate,
            'alt_fuel_ratio_pct': alt_fuel_ratio,
            'raw_meal_fineness_cm2_g': raw_meal_fineness,
            'lsf': lsf,
            'silica_modulus': sm,
            'alumina_modulus': am,
            'fuel_consumption_kcal_kg': fuel_consumption,
            'c3s_content_pct': c3s_content,
            'strength_28_day_mpa': strength_28_day,
            'free_lime_pct': free_lime,
            'co2_emissions_kg_ton': co2_emissions,
            'nox_emissions_mg_nm3': nox_emissions
        })
    
    def train_models(self, df):
        """Train models for fuel consumption and quality prediction"""
        try:
            # Features for prediction
            features = ['kiln_temperature_celsius', 'oxygen_level_pct', 'feed_rate_tons_hr',
                       'alt_fuel_ratio_pct', 'raw_meal_fineness_cm2_g', 'lsf',
                       'silica_modulus', 'alumina_modulus']
            
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 4:
                return False
                
            X = df[available_features].fillna(df[available_features].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Train fuel consumption model
            if 'fuel_consumption_kcal_kg' in df.columns:
                y_fuel = df['fuel_consumption_kcal_kg'].fillna(df['fuel_consumption_kcal_kg'].mean())
                self.fuel_model.fit(X_scaled, y_fuel)
            
            # Train strength prediction model  
            if 'strength_28_day_mpa' in df.columns:
                y_quality = df['strength_28_day_mpa'].fillna(df['strength_28_day_mpa'].mean())
                self.quality_model.fit(X_scaled, y_quality)
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def predict_clinker_performance(self, input_data):
        """Predict fuel consumption and clinker quality"""
        try:
            features = ['kiln_temperature_celsius', 'oxygen_level_pct', 'feed_rate_tons_hr',
                       'alt_fuel_ratio_pct', 'raw_meal_fineness_cm2_g', 'lsf',
                       'silica_modulus', 'alumina_modulus']
            
            X = []
            for feature in features:
                X.append(input_data.get(feature, 0))
            
            X = np.array(X).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            fuel_pred = self.fuel_model.predict(X_scaled)[0]
            strength_pred = self.quality_model.predict(X_scaled)[0]
            
            # Calculate efficiency metrics
            thermal_efficiency = 1000000 / fuel_pred  # kcal/kg to efficiency
            
            return {
                'predicted_fuel_consumption': float(fuel_pred),
                'predicted_strength_28_day': float(strength_pred),
                'thermal_efficiency': float(thermal_efficiency),
                'estimated_co2_emissions': float(850 + fuel_pred * 0.2)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'predicted_fuel_consumption': 750.0,
                'predicted_strength_28_day': 55.0,
                'thermal_efficiency': 1333.0,
                'estimated_co2_emissions': 900.0
            }
    
    def generate_ai_recommendations(self, current_data, prediction_results):
        """Generate AI-powered clinker optimization recommendations"""
        if not self.api_key:
            return self._rule_based_recommendations(current_data, prediction_results)
        
        try:
            data_context = {
                'current_conditions': {
                    'kiln_temperature': current_data.get('kiln_temperature_celsius', 1450),
                    'oxygen_level': current_data.get('oxygen_level_pct', 3.5),
                    'feed_rate': current_data.get('feed_rate_tons_hr', 45),
                    'alt_fuel_ratio': current_data.get('alt_fuel_ratio_pct', 20),
                    'lsf': current_data.get('lsf', 0.95),
                    'predicted_fuel': prediction_results['predicted_fuel_consumption'],
                    'predicted_strength': prediction_results['predicted_strength_28_day'],
                    'estimated_co2': prediction_results['estimated_co2_emissions']
                },
                'targets': {
                    'fuel_consumption_target': 720,  # kcal/kg clinker
                    'strength_target': 55,           # MPa minimum
                    'alt_fuel_target': 30,           # % alternative fuels
                    'co2_target': 850,               # kg/ton clinker
                    'oxygen_optimal': '2.5-4.0',    # % for efficient combustion
                    'temperature_range': '1440-1460' # °C optimal range
                }
            }
            
            prompt = f"""
            You are an expert AI system for cement clinker production optimization with deep knowledge of 
            pyroprocessing, combustion chemistry, and alternative fuel integration.

            CURRENT CLINKER PRODUCTION DATA:
            - Kiln Operating Conditions:
              * Temperature: {data_context['current_conditions']['kiln_temperature']:.0f}°C (optimal: 1440-1460°C)
              * Oxygen Level: {data_context['current_conditions']['oxygen_level']:.1f}% (optimal: 2.5-4.0%)
              * Feed Rate: {data_context['current_conditions']['feed_rate']:.1f} tons/hr
              * Alternative Fuel: {data_context['current_conditions']['alt_fuel_ratio']:.1f}% (target: ≥30%)
            
            - Chemistry & Quality:
              * LSF (Lime Saturation Factor): {data_context['current_conditions']['lsf']:.3f} (optimal: 0.92-0.98)
              * Predicted 28-day Strength: {data_context['current_conditions']['predicted_strength']:.1f} MPa (min: 55 MPa)
            
            - Performance Metrics:
              * Predicted Fuel Consumption: {data_context['current_conditions']['predicted_fuel']:.0f} kcal/kg (target: <720)
              * Estimated CO2 Emissions: {data_context['current_conditions']['estimated_co2']:.0f} kg/ton (target: <850)

            INDUSTRY BENCHMARKS & SUSTAINABILITY GOALS:
            - Thermal energy: <720 kcal/kg clinker (best practice)
            - Alternative fuel substitution: >30% for carbon reduction
            - NOx emissions: <500 mg/Nm³ at 10% O2
            - Clinker factor optimization: maximize strength while minimizing fuel

            Provide specific, technical recommendations in these categories:
            1. COMBUSTION_OPTIMIZATION: Temperature, oxygen, and flame management
            2. ALTERNATIVE_FUELS: Sustainable fuel integration and co-processing
            3. RAW_MEAL_CHEMISTRY: LSF, SM, AM adjustments for optimal burnability
            4. THERMAL_EFFICIENCY: Heat recovery and energy optimization
            5. EMISSIONS_CONTROL: NOx, CO2, and particulate reduction strategies
            6. QUALITY_ASSURANCE: Clinker mineralogy and cement strength optimization

            Each recommendation should include:
            - Specific parameter targets and adjustments
            - Expected impact on fuel consumption and emissions
            - Implementation priority (HIGH/MEDIUM/LOW)
            - Technical rationale based on cement chemistry

            Format as JSON with detailed technical recommendations.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    recommendations = json.loads(response_text[json_start:json_end])
                    
                    # Add performance analysis
                    recommendations['performance_analysis'] = {
                        'fuel_efficiency_vs_target': (720 / data_context['current_conditions']['predicted_fuel']) * 100,
                        'alt_fuel_gap_pct': max(0, 30 - data_context['current_conditions']['alt_fuel_ratio']),
                        'co2_reduction_potential': max(0, data_context['current_conditions']['estimated_co2'] - 850),
                        'optimization_priority': 'HIGH' if data_context['current_conditions']['predicted_fuel'] > 750 else 'MEDIUM'
                    }
                    
                    return recommendations
            except json.JSONDecodeError:
                pass
                
            return {
                'COMBUSTION_OPTIMIZATION': f"Optimize temperature to 1450°C and O2 to 3.0%",
                'ALTERNATIVE_FUELS': f"Increase alt fuel from {data_context['current_conditions']['alt_fuel_ratio']:.1f}% to 30%",
                'THERMAL_EFFICIENCY': f"Target fuel reduction to 720 kcal/kg (current pred: {data_context['current_conditions']['predicted_fuel']:.0f})",
                'ai_insights': response.text[:400] + "..." if len(response.text) > 400 else response.text
            }
            
        except Exception as e:
            print(f"AI generation error: {e}")
            return self._rule_based_recommendations(current_data, prediction_results)
    
    def _rule_based_recommendations(self, current_data, prediction_results):
        """Expert rule-based clinker optimization recommendations"""
        recommendations = {
            'COMBUSTION_OPTIMIZATION': [],
            'ALTERNATIVE_FUELS': [],
            'RAW_MEAL_CHEMISTRY': [],
            'THERMAL_EFFICIENCY': [],
            'EMISSIONS_CONTROL': [],
            'QUALITY_ASSURANCE': []
        }
        
        # Extract values
        temp = current_data.get('kiln_temperature_celsius', 1450)
        o2 = current_data.get('oxygen_level_pct', 3.5)
        feed_rate = current_data.get('feed_rate_tons_hr', 45)
        alt_fuel = current_data.get('alt_fuel_ratio_pct', 20)
        lsf = current_data.get('lsf', 0.95)
        pred_fuel = prediction_results['predicted_fuel_consumption']
        pred_strength = prediction_results['predicted_strength_28_day']
        
        # Combustion optimization
        if temp > 1465:
            recommendations['COMBUSTION_OPTIMIZATION'].append(
                f"🔥 HIGH: Reduce kiln temperature from {temp:.0f}°C to 1450-1460°C to save ~{(temp-1460)*0.8:.0f} kcal/kg"
            )
            recommendations['EMISSIONS_CONTROL'].append(
                f"🌡️ Temperature reduction will decrease NOx emissions by ~{(temp-1460)*8:.0f} mg/Nm³"
            )
        elif temp < 1440:
            recommendations['COMBUSTION_OPTIMIZATION'].append(
                f"⚠️ MEDIUM: Low temperature ({temp:.0f}°C) may cause incomplete burning - increase to 1445-1455°C"
            )
            recommendations['QUALITY_ASSURANCE'].append(
                "🔍 Monitor free lime content closely due to low burning temperature"
            )
        
        if o2 > 4.5:
            recommendations['COMBUSTION_OPTIMIZATION'].append(
                f"💨 HIGH: Excess oxygen ({o2:.1f}%) causing heat loss - optimize to 2.8-3.5% to save ~{(o2-3.5)*15:.0f} kcal/kg"
            )
        elif o2 < 2.5:
            recommendations['COMBUSTION_OPTIMIZATION'].append(
                f"⚠️ HIGH: Low oxygen ({o2:.1f}%) risks incomplete combustion - increase to 2.8-3.2%"
            )
        
        # Alternative fuels recommendations
        if alt_fuel < 25:
            fuel_gap = 30 - alt_fuel
            co2_reduction = fuel_gap * 2  # Approximate CO2 reduction per % alt fuel
            recommendations['ALTERNATIVE_FUELS'].append(
                f"🌱 HIGH: Increase alternative fuels from {alt_fuel:.1f}% to 30% (gap: {fuel_gap:.1f}%)"
            )
            recommendations['ALTERNATIVE_FUELS'].append(
                f"♻️ Target fuels: RDF (40%), biomass (35%), used tires (15%), waste oils (10%)"
            )
            recommendations['EMISSIONS_CONTROL'].append(
                f"📉 Alt fuel increase could reduce CO2 by ~{co2_reduction:.0f} kg/ton clinker"
            )
        elif alt_fuel > 40:
            recommendations['ALTERNATIVE_FUELS'].append(
                f"⚠️ MEDIUM: High alt fuel ratio ({alt_fuel:.1f}%) - monitor flame stability and clinker quality"
            )
        
        # Raw meal chemistry optimization
        if lsf > 0.98:
            recommendations['RAW_MEAL_CHEMISTRY'].append(
                f"⚗️ HIGH: LSF too high ({lsf:.3f}) - reduce limestone addition to 0.92-0.96 for better burnability"
            )
            recommendations['THERMAL_EFFICIENCY'].append(
                f"🔥 LSF optimization could save 15-25 kcal/kg fuel consumption"
            )
        elif lsf < 0.90:
            recommendations['RAW_MEAL_CHEMISTRY'].append(
                f"⚗️ MEDIUM: LSF too low ({lsf:.3f}) - increase to 0.92-0.96 for adequate C3S formation"
            )
            recommendations['QUALITY_ASSURANCE'].append(
                "💪 Low LSF will reduce 28-day strength development"
            )
        
        # Thermal efficiency
        if pred_fuel > 750:
            savings_potential = pred_fuel - 720
            recommendations['THERMAL_EFFICIENCY'].append(
                f"⚡ HIGH: Fuel consumption {pred_fuel:.0f} kcal/kg - optimize to <720 (potential: {savings_potential:.0f} kcal/kg)"
            )
            recommendations['THERMAL_EFFICIENCY'].append(
                "🔄 Implement waste heat recovery from clinker cooler and preheater"
            )
            recommendations['THERMAL_EFFICIENCY'].append(
                "🏭 Optimize kiln feed preparation and improve thermal insulation"
            )
        
        if feed_rate < 40:
            recommendations['THERMAL_EFFICIENCY'].append(
                f"📈 MEDIUM: Low feed rate ({feed_rate:.1f} t/hr) reduces thermal efficiency - optimize to 42-48 t/hr"
            )
        
        # Emissions control
        recommendations['EMISSIONS_CONTROL'].extend([
            "🌍 Implement SNCR system for NOx reduction to <500 mg/Nm³",
            "💨 Optimize combustion air distribution for uniform temperature profile",
            "🔬 Use low-NOx burners and staged combustion technology"
        ])
        
        # Quality assurance
        quality_actions = [
            "📊 Monitor clinker microscopy: C3S (50-65%), C2S (15-30%), C3A (5-12%)",
            "🧪 Track free lime every 2 hours (target: <1.5%)",
            "💎 Maintain Blaine fineness 3200-3800 cm²/g for optimal strength development"
        ]
        
        if pred_strength < 55:
            quality_actions.append(
                f"⚠️ LOW STRENGTH PREDICTION ({pred_strength:.1f} MPa) - adjust raw meal chemistry and burning profile"
            )
        
        recommendations['QUALITY_ASSURANCE'] = quality_actions
        
        return recommendations
    
    def calculate_sustainability_metrics(self, df):
        """Calculate sustainability and performance metrics"""
        try:
            # Energy and fuel metrics
            avg_fuel = df['fuel_consumption_kcal_kg'].mean() if 'fuel_consumption_kcal_kg' in df.columns else 750
            avg_alt_fuel = df['alt_fuel_ratio_pct'].mean() if 'alt_fuel_ratio_pct' in df.columns else 20
            avg_co2 = df['co2_emissions_kg_ton'].mean() if 'co2_emissions_kg_ton' in df.columns else 900
            
            # Quality metrics
            avg_strength = df['strength_28_day_mpa'].mean() if 'strength_28_day_mpa' in df.columns else 55
            
            # Calculate potential improvements
            fuel_reduction_potential = max(0, avg_fuel - 720)
            alt_fuel_increase_potential = max(0, 30 - avg_alt_fuel)
            co2_reduction_potential = max(0, avg_co2 - 850)
            
            # Economic calculations (Indian cement industry)
            annual_clinker_tons = 300000  # Typical plant capacity
            
            # Fuel cost savings (₹3000/ton of fuel equivalent)
            fuel_cost_savings = fuel_reduction_potential * 0.001 * annual_clinker_tons * 3000  # Convert kcal to tons
            
            # Alternative fuel cost benefit (₹500/ton saved vs coal)
            alt_fuel_savings = alt_fuel_increase_potential * 0.01 * annual_clinker_tons * 500
            
            # Carbon credit potential (₹1500/ton CO2)
            carbon_credits = co2_reduction_potential * annual_clinker_tons * 1500 / 1000  # Convert kg to tons
            
            return {
                'current_metrics': {
                    'avg_fuel_consumption_kcal_kg': avg_fuel,
                    'avg_alt_fuel_ratio_pct': avg_alt_fuel,
                    'avg_co2_emissions_kg_ton': avg_co2,
                    'avg_strength_28_day_mpa': avg_strength
                },
                'improvement_potential': {
                    'fuel_reduction_kcal_kg': fuel_reduction_potential,
                    'alt_fuel_increase_pct': alt_fuel_increase_potential,
                    'co2_reduction_kg_ton': co2_reduction_potential
                },
                'economic_impact_inr_lakhs': {
                    'fuel_cost_savings': fuel_cost_savings / 100000,
                    'alt_fuel_benefits': alt_fuel_savings / 100000,
                    'carbon_credits': carbon_credits / 100000,
                    'total_annual_savings': (fuel_cost_savings + alt_fuel_savings + carbon_credits) / 100000
                },
                'sustainability_score': min(100, (30/max(1,avg_alt_fuel)) * 25 + (720/max(1,avg_fuel)) * 25 + 
                                           (850/max(1,avg_co2)) * 25 + (avg_strength/55) * 25)
            }
            
        except Exception as e:
            print(f"Sustainability calculation error: {e}")
            return {
                'current_metrics': {'error': 'Calculation failed'},
                'improvement_potential': {},
                'economic_impact_inr_lakhs': {},
                'sustainability_score': 0
            }