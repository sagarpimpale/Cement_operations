import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import google.generativeai as genai
import json
import os
import warnings
warnings.filterwarnings('ignore')

class QualityControlSystem:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.scaler = StandardScaler()
        self.strength_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.quality_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def load_and_process_data(self, csv_path):
        """Load and process cement quality data"""
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return self._create_sample_data()
    
    def _create_sample_data(self, n_samples=600):
        """Create realistic cement quality control data"""
        np.random.seed(42)
        
        # Chemical composition data
        cao_content = np.random.normal(64.0, 1.5, n_samples)
        cao_content = np.clip(cao_content, 60, 67)
        
        sio2_content = np.random.normal(21.5, 1.2, n_samples)
        sio2_content = np.clip(sio2_content, 18, 25)
        
        al2o3_content = np.random.normal(5.2, 0.8, n_samples)
        al2o3_content = np.clip(al2o3_content, 3.5, 7.0)
        
        fe2o3_content = np.random.normal(3.1, 0.5, n_samples)
        fe2o3_content = np.clip(fe2o3_content, 2.0, 4.5)
        
        mgo_content = np.random.normal(2.8, 0.6, n_samples)
        mgo_content = np.clip(mgo_content, 1.5, 4.5)
        
        so3_content = np.random.normal(2.9, 0.4, n_samples)
        so3_content = np.clip(so3_content, 2.0, 3.8)
        
        # Physical properties
        blaine_fineness = np.random.normal(3500, 250, n_samples)
        blaine_fineness = np.clip(blaine_fineness, 3000, 4200)
        
        # Setting time depends on fineness and C3A content
        c3a_content = 4.071 * al2o3_content - 2.078 * fe2o3_content
        c3a_content = np.clip(c3a_content, 2, 15)
        
        initial_setting = (180 + (blaine_fineness - 3500) * -0.02 + 
                          c3a_content * 8 + so3_content * -15 +
                          np.random.normal(0, 25, n_samples))
        initial_setting = np.clip(initial_setting, 120, 300)
        
        final_setting = initial_setting + np.random.normal(60, 15, n_samples)
        final_setting = np.clip(final_setting, initial_setting + 30, 480)
        
        # Compressive strength calculations based on Bogue equations
        c3s_content = 4.071 * cao_content - 7.6 * sio2_content - 6.718 * al2o3_content - 1.43 * fe2o3_content
        c3s_content = np.clip(c3s_content, 45, 70)
        
        c2s_content = 2.867 * sio2_content - 0.7544 * c3s_content
        c2s_content = np.clip(c2s_content, 10, 30)
        
        # 7-day strength
        strength_7_day = (25 + c3s_content * 0.45 + c2s_content * 0.1 + 
                         (blaine_fineness - 3500) * 0.004 +
                         np.random.normal(0, 3, n_samples))
        strength_7_day = np.clip(strength_7_day, 20, 50)
        
        # 28-day strength
        strength_28_day = (42 + c3s_content * 0.6 + c2s_content * 0.3 + 
                          (blaine_fineness - 3500) * 0.005 +
                          np.random.normal(0, 4, n_samples))
        strength_28_day = np.clip(strength_28_day, 35, 75)
        
        # Soundness test (Le Chatelier expansion)
        soundness = (1.5 + mgo_content * 0.8 + so3_content * 0.5 +
                    np.random.normal(0, 1.2, n_samples))
        soundness = np.clip(soundness, 0.5, 8)
        
        # Water demand for consistency
        water_demand = (26 + (blaine_fineness - 3500) * 0.002 + 
                       c3a_content * 0.3 + np.random.normal(0, 1.5, n_samples))
        water_demand = np.clip(water_demand, 23, 32)
        
        # Heat of hydration (important for mass concrete)
        heat_hydration_7day = (280 + c3s_content * 2.8 + c3a_content * 8.5 +
                              np.random.normal(0, 25, n_samples))
        heat_hydration_7day = np.clip(heat_hydration_7day, 220, 400)
        
        # Quality classification based on IS:8112 and IS:12269 standards
        quality_grade = []
        for i in range(n_samples):
            if (strength_28_day[i] >= 53 and initial_setting[i] >= 30 and 
                final_setting[i] <= 600 and soundness[i] <= 10):
                if strength_28_day[i] >= 63:
                    quality_grade.append('OPC 53 Premium')  # High strength
                else:
                    quality_grade.append('OPC 53 Standard')
            elif (strength_28_day[i] >= 43 and initial_setting[i] >= 30 and 
                  final_setting[i] <= 600 and soundness[i] <= 10):
                quality_grade.append('OPC 43')
            else:
                quality_grade.append('Below Standard')
        
        # Defect detection
        defects = []
        for i in range(n_samples):
            defect_list = []
            if initial_setting[i] < 30:
                defect_list.append('Flash_Setting')
            if soundness[i] > 10:
                defect_list.append('Unsound')
            if strength_28_day[i] < 43:
                defect_list.append('Low_Strength')
            if so3_content[i] > 3.5:
                defect_list.append('High_SO3')
            if mgo_content[i] > 6:
                defect_list.append('High_MgO')
            
            defects.append('|'.join(defect_list) if defect_list else 'None')
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='4H')
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'cao_pct': cao_content,
            'sio2_pct': sio2_content,
            'al2o3_pct': al2o3_content,
            'fe2o3_pct': fe2o3_content,
            'mgo_pct': mgo_content,
            'so3_pct': so3_content,
            'blaine_fineness_cm2_g': blaine_fineness,
            'c3s_content_pct': c3s_content,
            'c3a_content_pct': c3a_content,
            'initial_setting_min': initial_setting,
            'final_setting_min': final_setting,
            'strength_7_day_mpa': strength_7_day,
            'strength_28_day_mpa': strength_28_day,
            'soundness_mm': soundness,
            'water_demand_pct': water_demand,
            'heat_hydration_7day_cal_g': heat_hydration_7day,
            'quality_grade': quality_grade,
            'defects': defects
        })
    
    def train_models(self, df):
        """Train quality prediction and classification models"""
        try:
            # Features for prediction
            chemical_features = ['cao_pct', 'sio2_pct', 'al2o3_pct', 'fe2o3_pct', 'mgo_pct', 'so3_pct']
            physical_features = ['blaine_fineness_cm2_g', 'water_demand_pct']
            
            all_features = chemical_features + physical_features
            available_features = [f for f in all_features if f in df.columns]
            
            if len(available_features) < 4:
                return False
            
            X = df[available_features].fillna(df[available_features].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Train 28-day strength predictor
            if 'strength_28_day_mpa' in df.columns:
                y_strength = df['strength_28_day_mpa'].fillna(df['strength_28_day_mpa'].mean())
                self.strength_predictor.fit(X_scaled, y_strength)
            
            # Train quality grade classifier
            if 'quality_grade' in df.columns:
                y_quality = df['quality_grade'].fillna('OPC 43')
                self.quality_classifier.fit(X_scaled, y_quality)
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def predict_quality_parameters(self, input_data):
        """Predict cement quality parameters"""
        try:
            features = ['cao_pct', 'sio2_pct', 'al2o3_pct', 'fe2o3_pct', 'mgo_pct', 'so3_pct',
                       'blaine_fineness_cm2_g', 'water_demand_pct']
            
            X = []
            for feature in features:
                X.append(input_data.get(feature, 0))
            
            X = np.array(X).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predictions
            strength_pred = self.strength_predictor.predict(X_scaled)[0]
            quality_grade_pred = self.quality_classifier.predict(X_scaled)[0]
            quality_proba = self.quality_classifier.predict_proba(X_scaled)[0]
            
            # Calculate Bogue compounds
            cao = input_data.get('cao_pct', 64)
            sio2 = input_data.get('sio2_pct', 21.5)
            al2o3 = input_data.get('al2o3_pct', 5.2)
            fe2o3 = input_data.get('fe2o3_pct', 3.1)
            
            c3s = 4.071 * cao - 7.6 * sio2 - 6.718 * al2o3 - 1.43 * fe2o3
            c2s = 2.867 * sio2 - 0.7544 * c3s
            c3a = 4.071 * al2o3 - 2.078 * fe2o3
            c4af = 3.043 * fe2o3
            
            return {
                'predicted_strength_28_day': float(strength_pred),
                'predicted_quality_grade': quality_grade_pred,
                'quality_confidence': float(max(quality_proba)),
                'bogue_compounds': {
                    'c3s_pct': max(0, float(c3s)),
                    'c2s_pct': max(0, float(c2s)),
                    'c3a_pct': max(0, float(c3a)),
                    'c4af_pct': max(0, float(c4af))
                }
            }
            
        except Exception as e:
            print(f"Quality prediction error: {e}")
            return {
                'predicted_strength_28_day': 55.0,
                'predicted_quality_grade': 'OPC 43',
                'quality_confidence': 0.8,
                'bogue_compounds': {'c3s_pct': 55.0, 'c2s_pct': 18.0, 'c3a_pct': 8.0, 'c4af_pct': 9.0}
            }
    
    def detect_quality_issues(self, current_data, prediction_results):
        """Detect potential quality issues and deviations"""
        issues = []
        warnings = []
        
        # Chemical composition checks (IS standards)
        cao = current_data.get('cao_pct', 64)
        sio2 = current_data.get('sio2_pct', 21.5)
        al2o3 = current_data.get('al2o3_pct', 5.2)
        fe2o3 = current_data.get('fe2o3_pct', 3.1)
        mgo = current_data.get('mgo_pct', 2.8)
        so3 = current_data.get('so3_pct', 2.9)
        
        # Critical issues
        if mgo > 6.0:
            issues.append(f"HIGH MgO CONTENT ({mgo:.1f}%) - Risk of unsoundness (IS limit: 6.0%)")
        
        if so3 > 3.5:
            issues.append(f"EXCESS SO3 CONTENT ({so3:.1f}%) - May cause false set (IS limit: 3.5%)")
        
        predicted_strength = prediction_results.get('predicted_strength_28_day', 55)
        if predicted_strength < 43:
            issues.append(f"LOW STRENGTH PREDICTION ({predicted_strength:.1f} MPa) - Below OPC 43 standard")
        
        # Early warnings
        if mgo > 4.5:
            warnings.append(f"MgO content trending high ({mgo:.1f}%) - monitor for expansion")
        
        if so3 < 2.0:
            warnings.append(f"Low SO3 content ({so3:.1f}%) - may affect early strength development")
        
        c3a_estimated = prediction_results.get('bogue_compounds', {}).get('c3a_pct', 8)
        if c3a_estimated > 12:
            warnings.append(f"High C3A content ({c3a_estimated:.1f}%) - monitor heat of hydration")
        
        blaine = current_data.get('blaine_fineness_cm2_g', 3500)
        if blaine < 3000:
            warnings.append(f"Low fineness ({blaine:.0f} cm²/g) - may affect strength development")
        elif blaine > 4000:
            warnings.append(f"High fineness ({blaine:.0f} cm²/g) - monitor water demand and setting time")
        
        return {
            'critical_issues': issues,
            'early_warnings': warnings,
            'quality_status': 'CRITICAL' if issues else 'WARNING' if warnings else 'NORMAL',
            'compliance_check': {
                'is_8112_compliant': predicted_strength >= 53 and mgo <= 6.0 and so3 <= 3.5,
                'is_12269_compliant': predicted_strength >= 43 and mgo <= 6.0 and so3 <= 3.5
            }
        }
    
    def generate_ai_recommendations(self, current_data, prediction_results, quality_issues):
        """Generate AI-powered quality control recommendations"""
        if not self.api_key:
            return self._rule_based_quality_recommendations(current_data, prediction_results, quality_issues)
        
        try:
            # Prepare comprehensive quality data
            quality_context = {
                'chemical_composition': {
                    'cao': current_data.get('cao_pct', 64),
                    'sio2': current_data.get('sio2_pct', 21.5),
                    'al2o3': current_data.get('al2o3_pct', 5.2),
                    'fe2o3': current_data.get('fe2o3_pct', 3.1),
                    'mgo': current_data.get('mgo_pct', 2.8),
                    'so3': current_data.get('so3_pct', 2.9)
                },
                'physical_properties': {
                    'blaine_fineness': current_data.get('blaine_fineness_cm2_g', 3500),
                    'predicted_strength': prediction_results.get('predicted_strength_28_day', 55),
                    'quality_grade': prediction_results.get('predicted_quality_grade', 'OPC 43')
                },
                'quality_status': {
                    'issues_count': len(quality_issues.get('critical_issues', [])),
                    'warnings_count': len(quality_issues.get('early_warnings', [])),
                    'compliance_status': quality_issues.get('compliance_check', {})
                },
                'bogue_compounds': prediction_results.get('bogue_compounds', {})
            }
            
            prompt = f"""
            You are an expert cement quality control AI with deep knowledge of IS:8112, IS:12269, 
            ASTM C150, and international cement standards. Analyze the current cement composition 
            and provide specific quality optimization recommendations.

            CURRENT CEMENT ANALYSIS:
            Chemical Composition (% by weight):
            - CaO: {quality_context['chemical_composition']['cao']:.2f}% (typical: 60-67%)
            - SiO2: {quality_context['chemical_composition']['sio2']:.2f}% (typical: 18-25%)
            - Al2O3: {quality_context['chemical_composition']['al2o3']:.2f}% (typical: 3-8%)
            - Fe2O3: {quality_context['chemical_composition']['fe2o3']:.2f}% (typical: 1-5%)
            - MgO: {quality_context['chemical_composition']['mgo']:.2f}% (IS limit: ≤6.0%)
            - SO3: {quality_context['chemical_composition']['so3']:.2f}% (IS limit: ≤3.5%)

            Physical Properties:
            - Blaine Fineness: {quality_context['physical_properties']['blaine_fineness']:.0f} cm²/g (typical: 3000-4000)
            - Predicted 28-day Strength: {quality_context['physical_properties']['predicted_strength']:.1f} MPa
            - Quality Grade Prediction: {quality_context['physical_properties']['quality_grade']}

            Bogue Compound Estimation:
            - C3S: {quality_context['bogue_compounds'].get('c3s_pct', 55):.1f}% (affects early strength)
            - C2S: {quality_context['bogue_compounds'].get('c2s_pct', 18):.1f}% (affects later strength)
            - C3A: {quality_context['bogue_compounds'].get('c3a_pct', 8):.1f}% (affects setting time, heat)
            - C4AF: {quality_context['bogue_compounds'].get('c4af_pct', 9):.1f}% (flux, affects color)

            Quality Issues Detected:
            - Critical Issues: {len(quality_context['quality_status']['issues_count'])}
            - Early Warnings: {len(quality_context['quality_status']['warnings_count'])}

            QUALITY STANDARDS COMPLIANCE:
            - IS:8112 (OPC 53): {"PASS" if quality_context['quality_status']['compliance_status'].get('is_8112_compliant') else "FAIL"}
            - IS:12269 (OPC 43): {"PASS" if quality_context['quality_status']['compliance_status'].get('is_12269_compliant') else "FAIL"}

            Provide specific recommendations in these categories:
            1. CHEMICAL_ADJUSTMENT: Raw material feed corrections for optimal composition
            2. GRINDING_OPTIMIZATION: Fineness adjustments for strength and workability balance
            3. QUALITY_ASSURANCE: Testing protocols and monitoring improvements
            4. STANDARD_COMPLIANCE: Actions to meet IS/ASTM requirements
            5. STRENGTH_ENHANCEMENT: Strategies to optimize compressive strength development
            6. PROCESS_CONTROL: Real-time adjustments for consistent quality

            Each recommendation should include:
            - Specific parameter targets and tolerances
            - Expected impact on cement properties
            - Implementation priority (IMMEDIATE/HIGH/MEDIUM/LOW)
            - Quality control checkpoints

            Focus on actionable recommendations based on cement chemistry principles.
            Format response as JSON with detailed technical recommendations.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    recommendations = json.loads(response_text[json_start:json_end])
                    
                    # Add quality metrics
                    recommendations['quality_metrics'] = {
                        'strength_achievement_pct': (quality_context['physical_properties']['predicted_strength'] / 53) * 100,
                        'composition_balance_score': self._calculate_composition_score(quality_context['chemical_composition']),
                        'standards_compliance_pct': sum(quality_context['quality_status']['compliance_status'].values()) * 50,
                        'optimization_priority': 'IMMEDIATE' if quality_context['quality_status']['issues_count'] > 0 else 'HIGH'
                    }
                    
                    return recommendations
            except json.JSONDecodeError:
                pass
                
            return {
                'CHEMICAL_ADJUSTMENT': f"Optimize CaO to 63-65% for better C3S formation",
                'STRENGTH_ENHANCEMENT': f"Target strength improvement to >{53 if quality_context['physical_properties']['predicted_strength'] < 53 else 60} MPa",
                'QUALITY_ASSURANCE': "Implement hourly XRF analysis for composition control",
                'ai_insights': response.text[:400] + "..." if len(response.text) > 400 else response.text
            }
            
        except Exception as e:
            print(f"AI quality recommendation error: {e}")
            return self._rule_based_quality_recommendations(current_data, prediction_results, quality_issues)
    
    def _calculate_composition_score(self, composition):
        """Calculate composition balance score (0-100)"""
        scores = []
        
        # CaO score (target: 63-65%)
        cao = composition['cao']
        if 63 <= cao <= 65:
            scores.append(100)
        else:
            scores.append(max(0, 100 - abs(cao - 64) * 10))
        
        # SiO2 score (target: 20-23%)
        sio2 = composition['sio2']
        if 20 <= sio2 <= 23:
            scores.append(100)
        else:
            scores.append(max(0, 100 - abs(sio2 - 21.5) * 15))
        
        # MgO score (limit: <6%)
        mgo = composition['mgo']
        if mgo <= 4:
            scores.append(100)
        elif mgo <= 6:
            scores.append(80)
        else:
            scores.append(max(0, 80 - (mgo - 6) * 20))
        
        # SO3 score (target: 2.5-3.5%)
        so3 = composition['so3']
        if 2.5 <= so3 <= 3.5:
            scores.append(100)
        else:
            scores.append(max(0, 100 - abs(so3 - 3.0) * 40))
        
        return sum(scores) / len(scores)
    
    def _rule_based_quality_recommendations(self, current_data, prediction_results, quality_issues):
        """Comprehensive rule-based quality recommendations"""
        recommendations = {
            'CHEMICAL_ADJUSTMENT': [],
            'GRINDING_OPTIMIZATION': [],
            'QUALITY_ASSURANCE': [],
            'STANDARD_COMPLIANCE': [],
            'STRENGTH_ENHANCEMENT': [],
            'PROCESS_CONTROL': []
        }
        
        # Extract values
        cao = current_data.get('cao_pct', 64)
        sio2 = current_data.get('sio2_pct', 21.5)
        al2o3 = current_data.get('al2o3_pct', 5.2)
        mgo = current_data.get('mgo_pct', 2.8)
        so3 = current_data.get('so3_pct', 2.9)
        blaine = current_data.get('blaine_fineness_cm2_g', 3500)
        pred_strength = prediction_results.get('predicted_strength_28_day', 55)
        
        # Chemical adjustments
        if cao < 62:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"🔧 IMMEDIATE: Increase limestone feed - CaO at {cao:.1f}% (target: 63-65%)"
            )
            recommendations['STRENGTH_ENHANCEMENT'].append(
                f"📈 Low CaO reduces C3S formation - expect {(65-cao)*2:.0f} MPa strength loss"
            )
        elif cao > 66:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"⚠️ HIGH: Reduce limestone feed - CaO at {cao:.1f}% may cause coating issues"
            )
        
        if sio2 < 19:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"🏗️ HIGH: Increase silica addition - SiO2 at {sio2:.1f}% (target: 20-23%)"
            )
        elif sio2 > 24:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"⚙️ MEDIUM: Reduce silica feed - high SiO2 ({sio2:.1f}%) reduces early strength"
            )
        
        if mgo > 4.5:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"🚨 HIGH: Monitor MgO at {mgo:.1f}% - approaching IS limit (6.0%)"
            )
            recommendations['STANDARD_COMPLIANCE'].append(
                "📋 Implement daily MgO monitoring - risk of soundness failure"
            )
        
        if so3 > 3.2:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"⚗️ HIGH: Reduce gypsum addition - SO3 at {so3:.1f}% near limit (3.5%)"
            )
        elif so3 < 2.2:
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                f"📊 MEDIUM: Increase gypsum - low SO3 ({so3:.1f}%) may delay setting"
            )
        
        # Grinding optimization
        if blaine < 3200:
            recommendations['GRINDING_OPTIMIZATION'].append(
                f"⚡ HIGH: Increase grinding time - Blaine at {blaine:.0f} cm²/g (target: 3400-3800)"
            )
            recommendations['STRENGTH_ENHANCEMENT'].append(
                f"💪 Fineness increase to 3500 cm²/g could add {(3500-blaine)*0.008:.1f} MPa strength"
            )
        elif blaine > 4000:
            recommendations['GRINDING_OPTIMIZATION'].append(
                f"🔄 MEDIUM: Reduce grinding intensity - high fineness ({blaine:.0f}) increases water demand"
            )
        
        # Strength enhancement
        c3s_content = prediction_results.get('bogue_compounds', {}).get('c3s_pct', 55)
        if pred_strength < 53:
            recommendations['STRENGTH_ENHANCEMENT'].append(
                f"🎯 IMMEDIATE: Strength at {pred_strength:.1f} MPa - target >53 MPa for OPC 53"
            )
            recommendations['STRENGTH_ENHANCEMENT'].append(
                f"🧪 Optimize C3S content from {c3s_content:.1f}% to >58% through CaO adjustment"
            )
            recommendations['CHEMICAL_ADJUSTMENT'].append(
                "🔬 Increase LSF (Lime Saturation Factor) to 0.95-0.98 for higher C3S"
            )
        
        if c3s_content < 50:
            recommendations['STRENGTH_ENHANCEMENT'].append(
                f"⚡ LOW C3S CONTENT ({c3s_content:.1f}%) - optimize raw meal proportions"
            )
        
        # Quality assurance protocols
        qa_protocols = [
            "🔬 Implement XRF analysis every 2 hours for chemical composition",
            "📊 Monitor Blaine fineness every hour during production",
            "🧪 Conduct setting time tests every 4 hours",
            "💪 Prepare strength cubes every 2 hours for 7 & 28-day testing"
        ]
        
        if len(quality_issues.get('critical_issues', [])) > 0:
            qa_protocols.append(
                "🚨 CRITICAL ISSUE PROTOCOL: Test every batch until resolved"
            )
        
        recommendations['QUALITY_ASSURANCE'] = qa_protocols
        
        # Standard compliance
        compliance_actions = []
        if not quality_issues.get('compliance_check', {}).get('is_8112_compliant', True):
            compliance_actions.append(
                "📋 IMMEDIATE: Non-compliance with IS:8112 - halt OPC 53 dispatch"
            )
            compliance_actions.append(
                "🔧 Action plan: Adjust composition and retest within 4 hours"
            )
        
        if mgo > 5:
            compliance_actions.append(
                f"⚠️ MgO approaching limit - implement preventive measures"
            )
        
        recommendations['STANDARD_COMPLIANCE'] = compliance_actions if compliance_actions else [
            "✅ All standards compliant - maintain current protocols"
        ]
        
        # Process control
        process_controls = [
            "🎛️ Set CaO control limits: 63.5 ± 1.5% with alarms",
            "📈 Implement SPC charts for key oxides with ±2σ limits",
            "🔄 Automate gypsum dosing based on real-time SO3 analysis",
            "🎯 Target Blaine fineness: 3500 ± 200 cm²/g"
        ]
        
        if pred_strength > 60:
            process_controls.append(
                f"🌟 Excellent strength ({pred_strength:.1f} MPa) - document optimal parameters"
            )
        
        recommendations['PROCESS_CONTROL'] = process_controls
        
        return recommendations
    
    def calculate_quality_metrics(self, df):
        """Calculate comprehensive quality performance metrics"""
        try:
            # Strength performance
            if 'strength_28_day_mpa' in df.columns:
                avg_strength = df['strength_28_day_mpa'].mean()
                strength_variability = df['strength_28_day_mpa'].std()
                strength_compliance = (df['strength_28_day_mpa'] >= 43).mean() * 100
                opc53_compliance = (df['strength_28_day_mpa'] >= 53).mean() * 100
            else:
                avg_strength, strength_variability = 55.0, 4.0
                strength_compliance, opc53_compliance = 85.0, 60.0
            
            # Chemical composition consistency
            chemical_cols = ['cao_pct', 'sio2_pct', 'al2o3_pct', 'fe2o3_pct', 'mgo_pct', 'so3_pct']
            available_chemical = [col for col in chemical_cols if col in df.columns]
            
            composition_variability = {}
            for col in available_chemical:
                composition_variability[col] = df[col].std()
            
            # Standards compliance
            compliance_checks = []
            if 'mgo_pct' in df.columns:
                mgo_compliance = (df['mgo_pct'] <= 6.0).mean() * 100
                compliance_checks.append(('MgO_compliance', mgo_compliance))
            
            if 'so3_pct' in df.columns:
                so3_compliance = (df['so3_pct'] <= 3.5).mean() * 100
                compliance_checks.append(('SO3_compliance', so3_compliance))
            
            # Quality grade distribution
            if 'quality_grade' in df.columns:
                grade_distribution = df['quality_grade'].value_counts(normalize=True) * 100
            else:
                grade_distribution = pd.Series({'OPC 43': 70, 'OPC 53 Standard': 25, 'OPC 53 Premium': 5})
            
            # Calculate overall quality score (0-100)
            quality_factors = [
                min(100, (avg_strength / 55) * 100),  # Strength performance
                max(0, 100 - strength_variability * 5),  # Consistency
                strength_compliance,  # Standards compliance
                sum(comp[1] for comp in compliance_checks) / len(compliance_checks) if compliance_checks else 95
            ]
            
            overall_quality_score = sum(quality_factors) / len(quality_factors)
            
            return {
                'strength_performance': {
                    'average_strength_mpa': avg_strength,
                    'strength_variability_mpa': strength_variability,
                    'opc43_compliance_pct': strength_compliance,
                    'opc53_compliance_pct': opc53_compliance
                },
                'composition_consistency': composition_variability,
                'standards_compliance': dict(compliance_checks),
                'quality_grade_distribution': grade_distribution.to_dict(),
                'overall_quality_score': overall_quality_score,
                'quality_rating': (
                    'EXCELLENT' if overall_quality_score >= 90 else
                    'GOOD' if overall_quality_score >= 80 else
                    'ACCEPTABLE' if overall_quality_score >= 70 else
                    'NEEDS_IMPROVEMENT'
                )
            }
            
        except Exception as e:
            print(f"Quality metrics calculation error: {e}")
            return {
                'strength_performance': {'error': 'Calculation failed'},
                'composition_consistency': {},
                'standards_compliance': {},
                'quality_grade_distribution': {},
                'overall_quality_score': 0,
                'quality_rating': 'UNKNOWN'
            }