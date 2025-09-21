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

class AlternativeFuelsOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.scaler = StandardScaler()
        self.fuel_efficiency_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.emissions_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.fuel_optimizer = KMeans(n_clusters=5, random_state=42)
        
    def load_and_process_data(self, csv_path):
        """Load and process alternative fuels data"""
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=500):
        """Create realistic alternative fuels operational data"""
        np.random.seed(42)
        
        # Base fuel composition (percentages)
        coal_pct = np.random.normal(65, 15, n_samples)
        coal_pct = np.clip(coal_pct, 35, 85)
        
        # Alternative fuels
        rdf_pct = np.random.normal(15, 8, n_samples)  # Refuse Derived Fuel
        rdf_pct = np.clip(rdf_pct, 2, 35)
        
        biomass_pct = np.random.normal(12, 6, n_samples)  # Wood waste, agricultural residue
        biomass_pct = np.clip(biomass_pct, 0, 25)
        
        tire_pct = np.random.normal(5, 3, n_samples)   # Waste tires
        tire_pct = np.clip(tire_pct, 0, 15)
        
        waste_oil_pct = np.random.normal(3, 2, n_samples)  # Waste oils
        waste_oil_pct = np.clip(waste_oil_pct, 0, 10)
        
        # Normalize to 100%
        total_fuel = coal_pct + rdf_pct + biomass_pct + tire_pct + waste_oil_pct
        coal_pct = coal_pct / total_fuel * 100
        rdf_pct = rdf_pct / total_fuel * 100
        biomass_pct = biomass_pct / total_fuel * 100
        tire_pct = tire_pct / total_fuel * 100
        waste_oil_pct = waste_oil_pct / total_fuel * 100
        
        # Alternative fuel ratio (non-coal fuels)
        alt_fuel_ratio = 100 - coal_pct
        
        # Fuel properties based on composition
        # Calorific value (kcal/kg) - weighted average
        cv_coal = 6000
        cv_rdf = 4200
        cv_biomass = 4500
        cv_tire = 8500
        cv_waste_oil = 9200
        
        calorific_value = ((coal_pct * cv_coal + rdf_pct * cv_rdf + 
                           biomass_pct * cv_biomass + tire_pct * cv_tire + 
                           waste_oil_pct * cv_waste_oil) / 100)
        calorific_value += np.random.normal(0, 150, n_samples)
        calorific_value = np.clip(calorific_value, 4000, 8000)
        
        # Moisture content affects combustion
        moisture_content = (coal_pct * 0.08 + rdf_pct * 0.15 + 
                           biomass_pct * 0.25 + tire_pct * 0.02 + 
                           waste_oil_pct * 0.01) / 100
        moisture_content += np.random.normal(0, 0.03, n_samples)
        moisture_content = np.clip(moisture_content, 0.02, 0.35)
        
        # Ash content
        ash_content = (coal_pct * 0.12 + rdf_pct * 0.18 + 
                      biomass_pct * 0.04 + tire_pct * 0.15 + 
                      waste_oil_pct * 0.02) / 100
        ash_content += np.random.normal(0, 0.02, n_samples)
        ash_content = np.clip(ash_content, 0.03, 0.25)
        
        # Sulfur content (environmental concern)
        sulfur_content = (coal_pct * 0.008 + rdf_pct * 0.003 + 
                         biomass_pct * 0.001 + tire_pct * 0.012 + 
                         waste_oil_pct * 0.015) / 100
        sulfur_content += np.random.normal(0, 0.002, n_samples)
        sulfur_content = np.clip(sulfur_content, 0.001, 0.025)
        
        # Chlorine content (affects emissions and refractory)
        chlorine_content = (coal_pct * 0.001 + rdf_pct * 0.008 + 
                           biomass_pct * 0.002 + tire_pct * 0.001 + 
                           waste_oil_pct * 0.003) / 100
        chlorine_content += np.random.normal(0, 0.001, n_samples)
        chlorine_content = np.clip(chlorine_content, 0.0005, 0.015)
        
        # Kiln operational parameters
        kiln_temp = np.random.normal(1450, 20, n_samples)
        kiln_temp = np.clip(kiln_temp, 1420, 1480)
        
        # Oxygen level affects combustion efficiency
        oxygen_level = np.random.normal(3.5, 0.8, n_samples)
        oxygen_level = np.clip(oxygen_level, 2.2, 5.5)
        
        # Feed rate
        feed_rate = np.random.normal(48, 7, n_samples)
        feed_rate = np.clip(feed_rate, 35, 65)
        
        # Fuel consumption based on calorific value and efficiency
        base_fuel_consumption = 750  # kcal/kg clinker baseline
        
        # Efficiency factor based on alt fuel ratio and properties
        efficiency_factor = (1 + (alt_fuel_ratio - 35) * 0.002 +  # Alt fuels slightly less efficient
                            (moisture_content - 0.1) * -2 +       # Moisture reduces efficiency
                            (calorific_value - 5500) * 0.00008)   # Higher CV improves efficiency
        
        fuel_consumption = (base_fuel_consumption / efficiency_factor + 
                           np.random.normal(0, 30, n_samples))
        fuel_consumption = np.clip(fuel_consumption, 650, 900)
        
        # CO2 emissions calculation
        # Coal: ~94 kg CO2/GJ, RDF: ~85 kg CO2/GJ, Biomass: ~0 kg CO2/GJ (carbon neutral)
        # Tires: ~85 kg CO2/GJ, Waste oil: ~75 kg CO2/GJ
        
        co2_factor_coal = 94
        co2_factor_rdf = 85
        co2_factor_biomass = 0  # Carbon neutral
        co2_factor_tire = 85
        co2_factor_waste_oil = 75
        
        # Calculate weighted CO2 factor
        weighted_co2_factor = ((coal_pct * co2_factor_coal + 
                               rdf_pct * co2_factor_rdf + 
                               biomass_pct * co2_factor_biomass + 
                               tire_pct * co2_factor_tire + 
                               waste_oil_pct * co2_factor_waste_oil) / 100)
        
        # CO2 emissions (kg/ton clinker)
        co2_emissions = ((fuel_consumption * calorific_value * 4.184 / 1000000) * 
                        weighted_co2_factor + 525)  # 525 kg/ton from calcination
        co2_emissions += np.random.normal(0, 25, n_samples)
        co2_emissions = np.clip(co2_emissions, 750, 1000)
        
        # NOx emissions (affected by fuel type and combustion)
        nox_base = 600  # mg/Nm3
        nox_emissions = (nox_base + 
                        (oxygen_level - 3.5) * 40 +     # Excess O2 increases NOx
                        alt_fuel_ratio * -3 +            # Alt fuels generally lower NOx
                        (kiln_temp - 1450) * 8 +         # Higher temp increases NOx
                        np.random.normal(0, 80, n_samples))
        nox_emissions = np.clip(nox_emissions, 300, 1000)
        
        # SO2 emissions (mainly from sulfur in fuels)
        so2_emissions = (sulfur_content * 100000 * 2 +  # Convert to mg/Nm3 and double for SO2
                        np.random.normal(0, 50, n_samples))
        so2_emissions = np.clip(so2_emissions, 50, 800)
        
        # Fuel cost (₹/ton clinker)
        cost_coal = 4500  # ₹/ton
        cost_rdf = 2800
        cost_biomass = 3200
        cost_tire = 1500  # Often free or paid to accept
        cost_waste_oil = 2000
        
        weighted_fuel_cost = ((coal_pct * cost_coal + 
                              rdf_pct * cost_rdf + 
                              biomass_pct * cost_biomass + 
                              tire_pct * cost_tire + 
                              waste_oil_pct * cost_waste_oil) / 100)
        
        # Fuel cost per ton clinker based on consumption
        fuel_cost_per_ton = (weighted_fuel_cost * fuel_consumption / calorific_value * 
                            calorific_value / 1000000 * 1000)  # Convert and scale
        fuel_cost_per_ton += np.random.normal(0, 200, n_samples)
        fuel_cost_per_ton = np.clip(fuel_cost_per_ton, 800, 2500)
        
        # Flame stability index (0-100, higher is better)
        flame_stability = (85 + 
                          (calorific_value - 5500) * 0.002 +
                          (100 - alt_fuel_ratio) * 0.1 +  # Lower alt fuel = more stable
                          (moisture_content - 0.1) * -30 +
                          np.random.normal(0, 8, n_samples))
        flame_stability = np.clip(flame_stability, 40, 100)
        
        # Thermal substitution rate (TSR) - key sustainability metric
        thermal_substitution = alt_fuel_ratio * (calorific_value / 6000)  # Adjusted for CV
        thermal_substitution = np.clip(thermal_substitution, 5, 60)
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='2H')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'coal_pct': coal_pct,
            'rdf_pct': rdf_pct,
            'biomass_pct': biomass_pct,
            'tire_pct': tire_pct,
            'waste_oil_pct': waste_oil_pct,
            'alt_fuel_ratio_pct': alt_fuel_ratio,
            'calorific_value_kcal_kg': calorific_value,
            'moisture_content_pct': moisture_content * 100,
            'ash_content_pct': ash_content * 100,
            'sulfur_content_pct': sulfur_content * 100,
            'chlorine_content_ppm': chlorine_content * 10000,
            'kiln_temperature_celsius': kiln_temp,
            'oxygen_level_pct': oxygen_level,
            'feed_rate_tons_hr': feed_rate,
            'fuel_consumption_kcal_kg': fuel_consumption,
            'co2_emissions_kg_ton': co2_emissions,
            'nox_emissions_mg_nm3': nox_emissions,
            'so2_emissions_mg_nm3': so2_emissions,
            'fuel_cost_inr_ton': fuel_cost_per_ton,
            'flame_stability_index': flame_stability,
            'thermal_substitution_rate_pct': thermal_substitution
        })
    
    def train_models(self, df):
        """Train models for fuel efficiency and emissions prediction"""
        try:
            # Features for prediction
            features = ['alt_fuel_ratio_pct', 'calorific_value_kcal_kg', 'moisture_content_pct',
                       'ash_content_pct', 'sulfur_content_pct', 'kiln_temperature_celsius',
                       'oxygen_level_pct', 'feed_rate_tons_hr']
            
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 5:
                return False
                
            X = df[available_features].fillna(df[available_features].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Train fuel efficiency model
            if 'fuel_consumption_kcal_kg' in df.columns:
                y_fuel = df['fuel_consumption_kcal_kg'].fillna(df['fuel_consumption_kcal_kg'].mean())
                self.fuel_efficiency_model.fit(X_scaled, y_fuel)
            
            # Train emissions model
            if 'co2_emissions_kg_ton' in df.columns:
                y_emissions = df['co2_emissions_kg_ton'].fillna(df['co2_emissions_kg_ton'].mean())
                self.emissions_model.fit(X_scaled, y_emissions)
            
            # Train fuel mix optimizer
            fuel_features = ['rdf_pct', 'biomass_pct', 'tire_pct', 'waste_oil_pct']
            available_fuel_features = [f for f in fuel_features if f in df.columns]
            
            if len(available_fuel_features) >= 3:
                fuel_data = df[available_fuel_features].fillna(0)
                self.fuel_optimizer.fit(fuel_data)
            
            return True
            
        except Exception as e:
            print(f"Alternative fuels model training error: {e}")
            return False
    
    def predict_fuel_performance(self, input_data):
        """Predict fuel consumption and emissions for given fuel mix"""
        try:
            features = ['alt_fuel_ratio_pct', 'calorific_value_kcal_kg', 'moisture_content_pct',
                       'ash_content_pct', 'sulfur_content_pct', 'kiln_temperature_celsius',
                       'oxygen_level_pct', 'feed_rate_tons_hr']
            
            X = []
            for feature in features:
                X.append(input_data.get(feature, 0))
            
            X = np.array(X).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            fuel_consumption_pred = self.fuel_efficiency_model.predict(X_scaled)[0]
            co2_emissions_pred = self.emissions_model.predict(X_scaled)[0]
            
            # Calculate additional metrics
            alt_fuel_ratio = input_data.get('alt_fuel_ratio_pct', 35)
            calorific_value = input_data.get('calorific_value_kcal_kg', 5500)
            
            # Thermal substitution rate
            tsr = alt_fuel_ratio * (calorific_value / 6000)
            
            # Cost estimation
            estimated_cost = self._estimate_fuel_cost(input_data)
            
            return {
                'predicted_fuel_consumption': float(fuel_consumption_pred),
                'predicted_co2_emissions': float(co2_emissions_pred),
                'thermal_substitution_rate': float(tsr),
                'estimated_fuel_cost': float(estimated_cost),
                'sustainability_score': min(100, alt_fuel_ratio * 2.5)  # Simple sustainability metric
            }
            
        except Exception as e:
            print(f"Fuel performance prediction error: {e}")
            return {
                'predicted_fuel_consumption': 750.0,
                'predicted_co2_emissions': 850.0,
                'thermal_substitution_rate': 35.0,
                'estimated_fuel_cost': 1500.0,
                'sustainability_score': 50.0
            }
    
    def _estimate_fuel_cost(self, fuel_data):
        """Estimate fuel cost based on composition"""
        coal_pct = fuel_data.get('coal_pct', 65)
        rdf_pct = fuel_data.get('rdf_pct', 15)
        biomass_pct = fuel_data.get('biomass_pct', 12)
        tire_pct = fuel_data.get('tire_pct', 5)
        waste_oil_pct = fuel_data.get('waste_oil_pct', 3)
        
        # Cost per ton of each fuel type (₹/ton)
        costs = {
            'coal': 4500,
            'rdf': 2800,
            'biomass': 3200,
            'tire': 1500,
            'waste_oil': 2000
        }
        
        weighted_cost = ((coal_pct * costs['coal'] + 
                         rdf_pct * costs['rdf'] + 
                         biomass_pct * costs['biomass'] + 
                         tire_pct * costs['tire'] + 
                         waste_oil_pct * costs['waste_oil']) / 100)
        
        # Convert to cost per ton clinker
        fuel_consumption = fuel_data.get('predicted_fuel_consumption', 750)  # kcal/kg
        calorific_value = fuel_data.get('calorific_value_kcal_kg', 5500)
        
        fuel_required_tons = fuel_consumption / calorific_value  # tons fuel per kg clinker
        cost_per_ton_clinker = fuel_required_tons * weighted_cost * 1000  # Convert to per ton clinker
        
        return cost_per_ton_clinker
    
    def optimize_fuel_mix(self, constraints=None, objectives=None):
        """Optimize fuel mix based on constraints and objectives"""
        if constraints is None:
            constraints = {
                'max_alt_fuel_ratio': 45,  # Maximum alternative fuel percentage
                'min_calorific_value': 4800,  # Minimum CV requirement
                'max_chlorine_ppm': 800,   # Environmental limit
                'max_moisture_pct': 20,    # Combustion efficiency limit
                'min_flame_stability': 70  # Operational stability
            }
        
        if objectives is None:
            objectives = {
                'minimize_co2': 0.4,      # Weight for CO2 reduction
                'minimize_cost': 0.3,     # Weight for cost reduction
                'maximize_tsr': 0.3       # Weight for thermal substitution
            }
        
        # Generate optimized fuel mix scenarios
        scenarios = []
        
        # Scenario 1: Maximum sustainability (high alt fuel, low CO2)
        scenario1 = {
            'name': 'Maximum Sustainability',
            'coal_pct': 55,
            'rdf_pct': 25,
            'biomass_pct': 15,
            'tire_pct': 3,
            'waste_oil_pct': 2,
            'focus': 'Environmental impact reduction'
        }
        
        # Scenario 2: Cost optimization
        scenario2 = {
            'name': 'Cost Optimized',
            'coal_pct': 60,
            'rdf_pct': 20,
            'biomass_pct': 5,
            'tire_pct': 12,
            'waste_oil_pct': 3,
            'focus': 'Minimum fuel cost'
        }
        
        # Scenario 3: Balanced approach
        scenario3 = {
            'name': 'Balanced Performance',
            'coal_pct': 65,
            'rdf_pct': 18,
            'biomass_pct': 10,
            'tire_pct': 5,
            'waste_oil_pct': 2,
            'focus': 'Balance of cost, emissions, and stability'
        }
        
        # Scenario 4: High thermal substitution
        scenario4 = {
            'name': 'High TSR Target',
            'coal_pct': 50,
            'rdf_pct': 22,
            'biomass_pct': 20,
            'tire_pct': 6,
            'waste_oil_pct': 2,
            'focus': 'Maximum thermal substitution rate'
        }
        
        scenarios = [scenario1, scenario2, scenario3, scenario4]
        
        # Calculate metrics for each scenario
        optimized_scenarios = []
        for scenario in scenarios:
            # Calculate derived properties
            alt_fuel_ratio = 100 - scenario['coal_pct']
            
            # Estimate calorific value
            cv = ((scenario['coal_pct'] * 6000 + 
                  scenario['rdf_pct'] * 4200 + 
                  scenario['biomass_pct'] * 4500 + 
                  scenario['tire_pct'] * 8500 + 
                  scenario['waste_oil_pct'] * 9200) / 100)
            
            # Estimate properties
            moisture = ((scenario['coal_pct'] * 8 + 
                        scenario['rdf_pct'] * 15 + 
                        scenario['biomass_pct'] * 25 + 
                        scenario['tire_pct'] * 2 + 
                        scenario['waste_oil_pct'] * 1) / 100)
            
            ash_content = ((scenario['coal_pct'] * 12 + 
                           scenario['rdf_pct'] * 18 + 
                           scenario['biomass_pct'] * 4 + 
                           scenario['tire_pct'] * 15 + 
                           scenario['waste_oil_pct'] * 2) / 100)
            
            # Create input data for prediction
            fuel_input = {
                'alt_fuel_ratio_pct': alt_fuel_ratio,
                'calorific_value_kcal_kg': cv,
                'moisture_content_pct': moisture,
                'ash_content_pct': ash_content,
                'sulfur_content_pct': 0.6,  # Average
                'kiln_temperature_celsius': 1450,
                'oxygen_level_pct': 3.5,
                'feed_rate_tons_hr': 48,
                **scenario
            }
            
            # Predict performance
            performance = self.predict_fuel_performance(fuel_input)
            
            optimized_scenarios.append({
                **scenario,
                'alt_fuel_ratio_pct': alt_fuel_ratio,
                'calorific_value_kcal_kg': cv,
                'moisture_content_pct': moisture,
                'performance_metrics': performance,
                'constraints_met': self._check_constraints(fuel_input, constraints)
            })
        
        return optimized_scenarios
    
    def _check_constraints(self, fuel_data, constraints):
        """Check if fuel mix meets operational constraints"""
        checks = {}
        
        alt_fuel_ratio = fuel_data.get('alt_fuel_ratio_pct', 35)
        checks['alt_fuel_ratio'] = alt_fuel_ratio <= constraints['max_alt_fuel_ratio']
        
        cv = fuel_data.get('calorific_value_kcal_kg', 5500)
        checks['calorific_value'] = cv >= constraints['min_calorific_value']
        
        moisture = fuel_data.get('moisture_content_pct', 10)
        checks['moisture_content'] = moisture <= constraints['max_moisture_pct']
        
        # Overall compliance
        checks['all_constraints_met'] = all(checks.values())
        
        return checks
    
    def generate_ai_recommendations(self, current_data, prediction_results, optimization_scenarios):
        """Generate AI-powered alternative fuels recommendations"""
        if not self.api_key:
            return self._rule_based_fuel_recommendations(current_data, prediction_results, optimization_scenarios)
        
        try:
            # Prepare comprehensive fuel data context
            fuel_context = {
                'current_fuel_mix': {
                    'coal_pct': current_data.get('coal_pct', 65),
                    'rdf_pct': current_data.get('rdf_pct', 15),
                    'biomass_pct': current_data.get('biomass_pct', 12),
                    'tire_pct': current_data.get('tire_pct', 5),
                    'waste_oil_pct': current_data.get('waste_oil_pct', 3),
                    'alt_fuel_ratio': current_data.get('alt_fuel_ratio_pct', 35),
                    'thermal_substitution_rate': prediction_results.get('thermal_substitution_rate', 35)
                },
                'fuel_properties': {
                    'calorific_value': current_data.get('calorific_value_kcal_kg', 5500),
                    'moisture_content': current_data.get('moisture_content_pct', 10),
                    'ash_content': current_data.get('ash_content_pct', 12),
                    'sulfur_content': current_data.get('sulfur_content_pct', 0.6),
                    'chlorine_content': current_data.get('chlorine_content_ppm', 400)
                },
                'performance_metrics': {
                    'predicted_fuel_consumption': prediction_results.get('predicted_fuel_consumption', 750),
                    'predicted_co2_emissions': prediction_results.get('predicted_co2_emissions', 850),
                    'estimated_fuel_cost': prediction_results.get('estimated_fuel_cost', 1500),
                    'sustainability_score': prediction_results.get('sustainability_score', 50)
                },
                'sustainability_targets': {
                    'tsr_target': 40,  # 40% thermal substitution rate
                    'co2_reduction_target': 800,  # kg/ton clinker
                    'cost_optimization_target': 1200,  # ₹/ton clinker
                    'renewable_fuel_target': 20  # % biomass content
                }
            }
            
            prompt = f"""
            You are an expert AI system for cement plant alternative fuels optimization with deep knowledge 
            of co-processing, thermal substitution, waste-to-energy, and sustainable fuel management.

            CURRENT ALTERNATIVE FUELS STATUS:
            Fuel Mix Composition:
            - Coal: {fuel_context['current_fuel_mix']['coal_pct']:.1f}% (baseline fossil fuel)
            - RDF (Refuse Derived Fuel): {fuel_context['current_fuel_mix']['rdf_pct']:.1f}% (municipal waste)
            - Biomass: {fuel_context['current_fuel_mix']['biomass_pct']:.1f}% (carbon neutral)
            - Waste Tires: {fuel_context['current_fuel_mix']['tire_pct']:.1f}% (high calorific value)
            - Waste Oil: {fuel_context['current_fuel_mix']['waste_oil_pct']:.1f}% (liquid fuel)
            - Total Alternative Fuel Ratio: {fuel_context['current_fuel_mix']['alt_fuel_ratio']:.1f}%
            - Thermal Substitution Rate: {fuel_context['current_fuel_mix']['thermal_substitution_rate']:.1f}%

            Fuel Quality Properties:
            - Calorific Value: {fuel_context['fuel_properties']['calorific_value']:.0f} kcal/kg (target: >4800)
            - Moisture Content: {fuel_context['fuel_properties']['moisture_content']:.1f}% (limit: <20%)
            - Ash Content: {fuel_context['fuel_properties']['ash_content']:.1f}% (affects clinker quality)
            - Sulfur Content: {fuel_context['fuel_properties']['sulfur_content']:.2f}% (emission concern)
            - Chlorine Content: {fuel_context['fuel_properties']['chlorine_content']:.0f} ppm (limit: <800 ppm)

            Performance Metrics:
            - Predicted Fuel Consumption: {fuel_context['performance_metrics']['predicted_fuel_consumption']:.0f} kcal/kg
            - Predicted CO2 Emissions: {fuel_context['performance_metrics']['predicted_co2_emissions']:.0f} kg/ton
            - Estimated Fuel Cost: ₹{fuel_context['performance_metrics']['estimated_fuel_cost']:.0f}/ton
            - Sustainability Score: {fuel_context['performance_metrics']['sustainability_score']:.0f}/100

            SUSTAINABILITY TARGETS:
            - Thermal Substitution Rate: ≥40% (current: {fuel_context['current_fuel_mix']['thermal_substitution_rate']:.1f}%)
            - CO2 Emissions: <800 kg/ton (current pred: {fuel_context['performance_metrics']['predicted_co2_emissions']:.0f})
            - Fuel Cost Optimization: <₹1200/ton (current: ₹{fuel_context['performance_metrics']['estimated_fuel_cost']:.0f})
            - Biomass Content: ≥20% for carbon neutrality (current: {fuel_context['current_fuel_mix']['biomass_pct']:.1f}%)

            REGULATORY & ENVIRONMENTAL CONSIDERATIONS:
            - Pollution Control Board approvals for waste co-processing
            - Flame temperature and stability requirements
            - Heavy metals and toxic emissions compliance
            - Circular economy and waste diversion goals

            Provide specific, technical recommendations in these categories:
            1. FUEL_MIX_OPTIMIZATION: Specific percentage adjustments for each fuel type
            2. THERMAL_SUBSTITUTION: Strategies to achieve 40%+ TSR safely
            3. BIOMASS_INTEGRATION: Carbon-neutral fuel scaling and sourcing
            4. COST_OPTIMIZATION: Economic fuel procurement and processing strategies
            5. EMISSIONS_CONTROL: Environmental compliance and monitoring
            6. OPERATIONAL_STABILITY: Flame management and combustion optimization

            Each recommendation should include:
            - Specific fuel percentage targets and gradual implementation steps
            - Expected impact on TSR, CO2 emissions, and costs
            - Implementation timeline and priority level (IMMEDIATE/HIGH/MEDIUM/LOW)
            - Risk mitigation measures for operational stability
            - Regulatory compliance considerations

            Format response as JSON with detailed technical recommendations.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    recommendations = json.loads(response_text[json_start:json_end])
                    
                    # Add fuel optimization metrics
                    recommendations['optimization_potential'] = {
                        'tsr_improvement_pct': max(0, 40 - fuel_context['current_fuel_mix']['thermal_substitution_rate']),
                        'co2_reduction_potential': max(0, fuel_context['performance_metrics']['predicted_co2_emissions'] - 800),
                        'cost_savings_potential': max(0, fuel_context['performance_metrics']['estimated_fuel_cost'] - 1200),
                        'biomass_increase_needed': max(0, 20 - fuel_context['current_fuel_mix']['biomass_pct'])
                    }
                    
                    return recommendations
            except json.JSONDecodeError:
                pass
                
            return {
                'FUEL_MIX_OPTIMIZATION': f"Increase alt fuel ratio from {fuel_context['current_fuel_mix']['alt_fuel_ratio']:.1f}% to 40%",
                'THERMAL_SUBSTITUTION': f"Target TSR improvement from {fuel_context['current_fuel_mix']['thermal_substitution_rate']:.1f}% to 40%",
                'BIOMASS_INTEGRATION': f"Scale biomass from {fuel_context['current_fuel_mix']['biomass_pct']:.1f}% to 20%",
                'ai_insights': response.text[:400] + "..." if len(response.text) > 400 else response.text
            }
            
        except Exception as e:
            print(f"AI fuel recommendation error: {e}")
            return self._rule_based_fuel_recommendations(current_data, prediction_results, optimization_scenarios)
    
    def _rule_based_fuel_recommendations(self, current_data, prediction_results, optimization_scenarios):
        """Expert rule-based alternative fuels recommendations"""
        recommendations = {
            'FUEL_MIX_OPTIMIZATION': [],
            'THERMAL_SUBSTITUTION': [],
            'BIOMASS_INTEGRATION': [],
            'COST_OPTIMIZATION': [],
            'EMISSIONS_CONTROL': [],
            'OPERATIONAL_STABILITY': []
        }
        
        # Extract values
        coal_pct = current_data.get('coal_pct', 65)
        rdf_pct = current_data.get('rdf_pct', 15)
        biomass_pct = current_data.get('biomass_pct', 12)
        tire_pct = current_data.get('tire_pct', 5)
        waste_oil_pct = current_data.get('waste_oil_pct', 3)
        alt_fuel_ratio = current_data.get('alt_fuel_ratio_pct', 35)
        tsr = prediction_results.get('thermal_substitution_rate', 35)
        co2_pred = prediction_results.get('predicted_co2_emissions', 850)
        cost_pred = prediction_results.get('estimated_fuel_cost', 1500)
        
        # Fuel mix optimization
        if alt_fuel_ratio < 35:
            gap = 35 - alt_fuel_ratio
            recommendations['FUEL_MIX_OPTIMIZATION'].append(
                f"🎯 HIGH: Increase alternative fuel ratio from {alt_fuel_ratio:.1f}% to 35% (gap: {gap:.1f}%)"
            )
            recommendations['FUEL_MIX_OPTIMIZATION'].append(
                f"📈 Suggested increment: +2% RDF, +1.5% biomass, +0.5% tire"
            )
        elif alt_fuel_ratio >= 40:
            recommendations['FUEL_MIX_OPTIMIZATION'].append(
                f"🌟 EXCELLENT: Alt fuel ratio at {alt_fuel_ratio:.1f}% - maintain current performance"
            )
        
        if coal_pct > 70:
            recommendations['FUEL_MIX_OPTIMIZATION'].append(
                f"⚡ MEDIUM: High coal dependency ({coal_pct:.1f}%) - diversify fuel portfolio"
            )
        
        # Thermal substitution recommendations
        if tsr < 35:
            tsr_gap = 40 - tsr
            recommendations['THERMAL_SUBSTITUTION'].append(
                f"🔥 HIGH: TSR at {tsr:.1f}% - target 40% for sustainability (gap: {tsr_gap:.1f}%)"
            )
            recommendations['THERMAL_SUBSTITUTION'].append(
                f"🎯 Strategy: Increase high-CV fuels (tires, waste oil) by {tsr_gap/2:.1f}%"
            )
        elif tsr >= 40:
            recommendations['THERMAL_SUBSTITUTION'].append(
                f"✅ TARGET ACHIEVED: TSR at {tsr:.1f}% - excellent sustainability performance"
            )
        
        # Biomass integration
        if biomass_pct < 15:
            biomass_gap = 20 - biomass_pct
            recommendations['BIOMASS_INTEGRATION'].append(
                f"🌱 HIGH: Biomass at {biomass_pct:.1f}% - scale to 20% for carbon neutrality (gap: {biomass_gap:.1f}%)"
            )
            recommendations['BIOMASS_INTEGRATION'].append(
                f"♻️ Sources: Agricultural residue (40%), wood waste (35%), energy crops (25%)"
            )
        elif biomass_pct >= 20:
            recommendations['BIOMASS_INTEGRATION'].append(
                f"🌿 EXCELLENT: Biomass at {biomass_pct:.1f}% - significant carbon footprint reduction"
            )
        
        if biomass_pct > 25:
            recommendations['BIOMASS_INTEGRATION'].append(
                f"⚠️ CAUTION: High biomass ({biomass_pct:.1f}%) - monitor ash content and alkali levels"
            )
        
        # Cost optimization
        if cost_pred > 1400:
            cost_gap = cost_pred - 1200
            recommendations['COST_OPTIMIZATION'].append(
                f"💰 HIGH: Fuel cost at ₹{cost_pred:.0f}/ton - optimize to <₹1200 (potential: ₹{cost_gap:.0f})"
            )
            recommendations['COST_OPTIMIZATION'].append(
                f"💡 Strategy: Increase tire usage (+{max(0, 10-tire_pct):.1f}%) and negotiate RDF rates"
            )
        
        if tire_pct < 8:
            recommendations['COST_OPTIMIZATION'].append(
                f"🚗 MEDIUM: Low tire usage ({tire_pct:.1f}%) - scale to 8-12% for cost savings"
            )
        
        # Emissions control
        if co2_pred > 850:
            co2_gap = co2_pred - 800
            recommendations['EMISSIONS_CONTROL'].append(
                f"🌍 HIGH: CO2 at {co2_pred:.0f} kg/ton - target <800 (reduction needed: {co2_gap:.0f} kg/ton)"
            )
            recommendations['EMISSIONS_CONTROL'].append(
                f"📉 Action: Increase biomass to 20% for {biomass_pct * 10:.0f} kg/ton CO2 reduction"
            )
        
        recommendations['EMISSIONS_CONTROL'].extend([
            f"🔬 Monitor SO2: Target <300 mg/Nm³ with low-sulfur fuel selection",
            f"⚗️ Control NOx: Optimize combustion air staging for <500 mg/Nm³",
            f"🧪 Track heavy metals: Monthly stack testing for regulatory compliance"
        ])
        
        # Operational stability
        stability_actions = [
            f"🔥 Maintain flame temperature >1450°C for complete combustion",
            f"📊 Monitor thermal shock index for refractory protection",
            f"🎛️ Implement gradual fuel switching protocols (±2% per shift)",
            f"⚙️ Optimize secondary air temperature for stable ignition"
        ]
        
        if alt_fuel_ratio > 35:
            stability_actions.append(
                f"🔍 Enhanced monitoring: Flame imagery and combustion analytics"
            )
        
        if biomass_pct > 15:
            stability_actions.append(
                f"🌾 Biomass handling: Maintain <20% moisture and uniform sizing"
            )
        
        recommendations['OPERATIONAL_STABILITY'] = stability_actions
        
        return recommendations
    
    def calculate_sustainability_impact(self, df):
        """Calculate sustainability metrics and environmental impact"""
        try:
            # Current performance
            avg_alt_fuel = df['alt_fuel_ratio_pct'].mean() if 'alt_fuel_ratio_pct' in df.columns else 35
            avg_tsr = df['thermal_substitution_rate_pct'].mean() if 'thermal_substitution_rate_pct' in df.columns else 35
            avg_biomass = df['biomass_pct'].mean() if 'biomass_pct' in df.columns else 12
            avg_co2 = df['co2_emissions_kg_ton'].mean() if 'co2_emissions_kg_ton' in df.columns else 850
            avg_fuel_cost = df['fuel_cost_inr_ton'].mean() if 'fuel_cost_inr_ton' in df.columns else 1500
            
            # Improvement potential
            tsr_improvement_potential = max(0, 40 - avg_tsr)
            co2_reduction_potential = max(0, avg_co2 - 800)
            cost_savings_potential = max(0, avg_fuel_cost - 1200)
            biomass_scale_potential = max(0, 20 - avg_biomass)
            
            # Annual impact calculations (assume 300,000 tons clinker/year)
            annual_clinker_tons = 300000
            
            # CO2 reduction impact
            annual_co2_reduction_tons = co2_reduction_potential * annual_clinker_tons / 1000
            
            # Cost savings impact
            annual_cost_savings_inr = cost_savings_potential * annual_clinker_tons
            
            # Waste diversion calculation
            total_alt_fuel_tons = avg_alt_fuel / 100 * annual_clinker_tons * 0.75  # Approximate fuel/clinker ratio
            
            # Carbon credits potential (₹1500/ton CO2)
            carbon_credits_value = annual_co2_reduction_tons * 1500
            
            # Waste processing fees earned (₹500/ton waste)
            waste_processing_revenue = total_alt_fuel_tons * 500
            
            return {
                'current_performance': {
                    'avg_alternative_fuel_pct': avg_alt_fuel,
                    'avg_thermal_substitution_pct': avg_tsr,
                    'avg_biomass_content_pct': avg_biomass,
                    'avg_co2_emissions_kg_ton': avg_co2,
                    'avg_fuel_cost_inr_ton': avg_fuel_cost
                },
                'improvement_potential': {
                    'tsr_increase_pct': tsr_improvement_potential,
                    'co2_reduction_kg_ton': co2_reduction_potential,
                    'cost_reduction_inr_ton': cost_savings_potential,
                    'biomass_scaling_pct': biomass_scale_potential
                },
                'annual_environmental_impact': {
                    'co2_reduction_tons_year': annual_co2_reduction_tons,
                    'waste_diverted_tons_year': total_alt_fuel_tons,
                    'fossil_fuel_displacement_tons_year': total_alt_fuel_tons * 0.65,  # Equivalent coal displacement
                    'renewable_energy_equivalent_mwh': total_alt_fuel_tons * avg_biomass/100 * 4.5  # Biomass energy equivalent
                },
                'economic_benefits_inr_lakhs': {
                    'annual_fuel_cost_savings': annual_cost_savings_inr / 100000,
                    'carbon_credits_revenue': carbon_credits_value / 100000,
                    'waste_processing_fees': waste_processing_revenue / 100000,
                    'total_economic_benefit': (annual_cost_savings_inr + carbon_credits_value + waste_processing_revenue) / 100000
                },
                'sustainability_score': min(100, 
                    (avg_tsr/40 * 25) + (avg_biomass/20 * 25) + 
                    ((1000-avg_co2)/200 * 25) + (avg_alt_fuel/50 * 25)),
                'sustainability_rating': (
                    'EXCELLENT' if avg_tsr >= 35 and avg_biomass >= 15 and avg_co2 <= 850 else
                    'GOOD' if avg_tsr >= 30 and avg_biomass >= 10 and avg_co2 <= 900 else
                    'MODERATE' if avg_tsr >= 25 and avg_biomass >= 8 else
                    'NEEDS_IMPROVEMENT'
                ),
                'regulatory_compliance': {
                    'pollution_control_board': avg_alt_fuel <= 60,  # Typical approval limit
                    'waste_management_rules': True,  # Assuming compliance
                    'emission_norms': avg_co2 <= 950,  # Environmental clearance
                    'thermal_substitution_target': avg_tsr >= 25  # Industry target
                }
            }
            
        except Exception as e:
            print(f"Sustainability impact calculation error: {e}")
            return {
                'current_performance': {'error': 'Calculation failed'},
                'improvement_potential': {},
                'annual_environmental_impact': {},
                'economic_benefits_inr_lakhs': {},
                'sustainability_score': 0,
                'sustainability_rating': 'UNKNOWN',
                'regulatory_compliance': {}
            }
            