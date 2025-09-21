import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json
import os
import warnings
warnings.filterwarnings('ignore')

class CrossProcessOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.scaler = StandardScaler()
        self.integrated_model = RandomForestRegressor(n_estimators=150, random_state=42)
        
    def integrate_plant_data(self, raw_materials_data, clinker_data, quality_data, 
                           alt_fuels_data, utilities_data):
        """Integrate data from all plant processes"""
        
        integrated_metrics = {
            # Energy integration
            'total_energy_consumption': (
                raw_materials_data.get('predicted_energy', 85) +
                (clinker_data.get('predicted_fuel_consumption', 750) * 0.01) +  # Convert kcal to kWh
                utilities_data.get('predicted_specific_power_kwh_ton', 105)
            ),
            
            # Quality integration
            'overall_quality_score': (
                (raw_materials_data.get('fineness_score', 85) * 0.2) +
                (clinker_data.get('predicted_strength_28_day', 55) * 0.4) +
                (quality_data.get('predicted_strength_28_day', 55) * 0.4)
            ),
            
            # Sustainability integration
            'sustainability_index': (
                (alt_fuels_data.get('thermal_substitution_rate', 35) * 0.4) +
                (100 - alt_fuels_data.get('predicted_co2_emissions', 850) / 10 * 0.3) +
                (utilities_data.get('efficiency_score', 75) * 0.3)
            ),
            
            # Cost integration
            'total_production_cost': (
                raw_materials_data.get('energy_cost_inr_ton', 400) +
                alt_fuels_data.get('estimated_fuel_cost', 1500) +
                utilities_data.get('power_cost_inr_ton', 500) +
                200  # Other costs
            ),
            
            # Process stability
            'process_stability_score': (
                (100 if not raw_materials_data.get('is_anomaly', False) else 70) * 0.25 +
                (clinker_data.get('thermal_efficiency', 75)) * 0.25 +
                (quality_data.get('quality_confidence', 0.8) * 100) * 0.25 +
                (utilities_data.get('efficiency_score', 75)) * 0.25
            )
        }
        
        return integrated_metrics
    
    def identify_cross_process_synergies(self, integrated_data):
        """Identify optimization synergies across processes"""
        
        synergies = []
        
        # Energy synergies
        if integrated_data['total_energy_consumption'] > 200:
            synergies.append({
                'type': 'Energy Integration',
                'opportunity': 'Waste Heat Recovery Network',
                'description': 'Utilize cement mill exhaust heat for raw material drying',
                'impact': 'potential_energy_savings_pct',
                'savings': 12,
                'implementation': 'Install heat exchanger network between mills and dryers'
            })
        
        # Quality-Energy synergy
        if integrated_data['overall_quality_score'] > 60 and integrated_data['total_energy_consumption'] > 180:
            synergies.append({
                'type': 'Quality-Energy Optimization',
                'opportunity': 'Optimal Fineness Control',
                'description': 'Balance cement fineness for strength while minimizing grinding energy',
                'impact': 'quality_energy_balance',
                'savings': 8,
                'implementation': 'AI-controlled separator optimization'
            })
        
        # Fuel-Quality synergy
        sustainability_score = integrated_data['sustainability_index']
        if sustainability_score < 75:
            synergies.append({
                'type': 'Fuel-Quality Integration',
                'opportunity': 'Alternative Fuel Quality Control',
                'description': 'Optimize alt fuel mix while maintaining clinker quality',
                'impact': 'sustainability_improvement',
                'savings': 15,
                'implementation': 'Real-time fuel composition adjustment based on clinker quality'
            })
        
        # Utilities-Production synergy
        if integrated_data['process_stability_score'] < 85:
            synergies.append({
                'type': 'Production-Utilities Optimization',
                'opportunity': 'Load Management Optimization',
                'description': 'Coordinate mill operations for optimal power consumption',
                'impact': 'power_optimization',
                'savings': 10,
                'implementation': 'Intelligent load scheduling and demand response'
            })
        
        return synergies
    
    def generate_master_optimization_plan(self, integrated_data, synergies):
        """Generate comprehensive master optimization plan"""
        
        if not self.api_key:
            return self._rule_based_master_plan(integrated_data, synergies)
        
        try:
            # Prepare comprehensive context
            optimization_context = {
                'current_performance': {
                    'total_energy_consumption': integrated_data['total_energy_consumption'],
                    'overall_quality_score': integrated_data['overall_quality_score'],
                    'sustainability_index': integrated_data['sustainability_index'],
                    'total_production_cost': integrated_data['total_production_cost'],
                    'process_stability_score': integrated_data['process_stability_score']
                },
                'identified_synergies': len(synergies),
                'improvement_potential': {
                    'energy_savings': sum([s['savings'] for s in synergies if 'energy' in s['type'].lower()]),
                    'sustainability_improvement': sum([s['savings'] for s in synergies if 'sustainability' in s['impact']]),
                    'cost_optimization': sum([s['savings'] for s in synergies if 'cost' in s['impact']])
                },
                'industry_benchmarks': {
                    'energy_target': 180,  # kWh/ton total
                    'quality_target': 85,  # Score out of 100
                    'sustainability_target': 85,  # Score out of 100
                    'cost_target': 2000,  # ₹/ton total production cost
                    'stability_target': 95  # Score out of 100
                }
            }
            
            prompt = f"""
            You are the Master AI Optimization System for a cement plant with comprehensive knowledge 
            of integrated process optimization, cross-functional synergies, and holistic plant management.

            CURRENT INTEGRATED PLANT PERFORMANCE:
            - Total Energy Consumption: {optimization_context['current_performance']['total_energy_consumption']:.1f} kWh/ton (target: <180)
            - Overall Quality Score: {optimization_context['current_performance']['overall_quality_score']:.1f}/100 (target: >85)
            - Sustainability Index: {optimization_context['current_performance']['sustainability_index']:.1f}/100 (target: >85)
            - Total Production Cost: ₹{optimization_context['current_performance']['total_production_cost']:.0f}/ton (target: <₹2000)
            - Process Stability Score: {optimization_context['current_performance']['process_stability_score']:.1f}/100 (target: >95)

            CROSS-PROCESS SYNERGIES IDENTIFIED: {optimization_context['identified_synergies']}
            - Energy Integration Opportunities: {optimization_context['improvement_potential']['energy_savings']}% savings potential
            - Sustainability Enhancement: {optimization_context['improvement_potential']['sustainability_improvement']}% improvement potential
            - Cost Optimization: {optimization_context['improvement_potential']['cost_optimization']}% reduction potential

            STRATEGIC OPTIMIZATION OBJECTIVES:
            1. Energy Efficiency: Achieve <180 kWh/ton total energy consumption
            2. Quality Excellence: Maintain >85/100 quality score with consistency
            3. Sustainability Leadership: Reach >85/100 sustainability index
            4. Cost Competitiveness: Target <₹2000/ton total production cost
            5. Operational Excellence: Achieve >95/100 process stability

            INTEGRATION FOCUS AREAS:
            - Raw Material → Grinding → Clinker production energy cascading
            - Alternative fuels impact on clinker quality and emissions
            - Utilities optimization across all process stages
            - Quality control integration with process parameters
            - Waste heat recovery and energy recycling

            Provide a comprehensive Master Optimization Strategy with:
            1. STRATEGIC_PRIORITIES: Top 3 critical optimization areas with maximum ROI
            2. PHASE_1_ACTIONS: Immediate actions (0-3 months) with specific targets
            3. PHASE_2_INTEGRATION: Medium-term integration projects (3-9 months)
            4. PHASE_3_TRANSFORMATION: Long-term transformation initiatives (9-24 months)
            5. PERFORMANCE_TARGETS: Quantified improvement targets for each phase
            6. RISK_MITIGATION: Operational risks and mitigation strategies
            7. ECONOMIC_IMPACT: Detailed ROI analysis and payback periods

            Each recommendation should include:
            - Specific technical implementation details
            - Expected quantitative improvements
            - Cross-process dependencies and coordination requirements
            - Investment requirements and payback analysis
            - Key performance indicators for monitoring success

            Format response as JSON with comprehensive technical recommendations.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    master_plan = json.loads(response_text[json_start:json_end])
                    
                    # Add quantitative projections
                    master_plan['quantitative_projections'] = {
                        'energy_reduction_kwh_ton': max(0, optimization_context['current_performance']['total_energy_consumption'] - 180),
                        'quality_improvement_points': max(0, 85 - optimization_context['current_performance']['overall_quality_score']),
                        'sustainability_enhancement_points': max(0, 85 - optimization_context['current_performance']['sustainability_index']),
                        'cost_savings_inr_ton': max(0, optimization_context['current_performance']['total_production_cost'] - 2000),
                        'annual_savings_inr_crores': self._calculate_annual_savings(optimization_context)
                    }
                    
                    return master_plan
            except json.JSONDecodeError:
                pass
                
            return {
                'STRATEGIC_PRIORITIES': ['Energy Integration', 'Alternative Fuels Scaling', 'Quality Optimization'],
                'PHASE_1_ACTIONS': ['VFD Installation', 'Power Factor Correction', 'Biomass Scaling'],
                'PHASE_2_INTEGRATION': ['Waste Heat Recovery', 'AI Process Control', 'Quality Automation'],
                'ai_insights': response.text[:500] + "..." if len(response.text) > 500 else response.text
            }
            
        except Exception as e:
            print(f"AI master plan generation error: {e}")
            return self._rule_based_master_plan(integrated_data, synergies)
    
    def _calculate_annual_savings(self, context):
        """Calculate projected annual savings in crores"""
        current_cost = context['current_performance']['total_production_cost']
        target_cost = context['industry_benchmarks']['cost_target']
        savings_per_ton = max(0, current_cost - target_cost)
        
        # Assume 300,000 tons annual production
        annual_savings = savings_per_ton * 300000 / 10000000  # Convert to crores
        return annual_savings
    
    def _rule_based_master_plan(self, integrated_data, synergies):
        """Generate rule-based master optimization plan"""
        
        master_plan = {
            'STRATEGIC_PRIORITIES': [],
            'PHASE_1_ACTIONS': [],
            'PHASE_2_INTEGRATION': [],
            'PHASE_3_TRANSFORMATION': [],
            'PERFORMANCE_TARGETS': {},
            'ECONOMIC_IMPACT': {}
        }
        
        # Strategic priorities based on gaps
        energy_gap = integrated_data['total_energy_consumption'] - 180
        quality_gap = 85 - integrated_data['overall_quality_score']
        sustainability_gap = 85 - integrated_data['sustainability_index']
        cost_gap = integrated_data['total_production_cost'] - 2000
        
        priorities = []
        if energy_gap > 20:
            priorities.append({
                'priority': 'Energy Optimization',
                'gap': f'{energy_gap:.1f} kWh/ton excess',
                'impact': 'HIGH',
                'savings_potential': f'₹{energy_gap * 4.8 * 300000 / 100000:.1f}L annually'
            })
        
        if sustainability_gap > 10:
            priorities.append({
                'priority': 'Sustainability Enhancement',
                'gap': f'{sustainability_gap:.1f} points below target',
                'impact': 'HIGH',
                'savings_potential': 'Carbon credits + regulatory compliance'
            })
        
        if cost_gap > 500:
            priorities.append({
                'priority': 'Cost Optimization',
                'gap': f'₹{cost_gap:.0f}/ton excess cost',
                'impact': 'CRITICAL',
                'savings_potential': f'₹{cost_gap * 300000 / 100000:.1f}L annually'
            })
        
        master_plan['STRATEGIC_PRIORITIES'] = priorities[:3]
        
        # Phase 1 actions (0-3 months)
        phase1_actions = [
            {
                'action': '⚡ Variable Frequency Drive Installation',
                'target': '12% energy reduction in major motors',
                'timeline': '8 weeks',
                'investment': '₹80L',
                'roi': '18 months'
            },
            {
                'action': '🌱 Alternative Fuel Scale-up',
                'target': 'Increase biomass to 20%, alt fuel ratio to 40%',
                'timeline': '10 weeks',
                'investment': '₹50L',
                'roi': '14 months'
            },
            {
                'action': '🔧 Power Factor Correction',
                'target': 'Improve PF from 0.88 to 0.95',
                'timeline': '6 weeks',
                'investment': '₹35L',
                'roi': '12 months'
            },
            {
                'action': '📊 Real-time Quality Monitoring',
                'target': '95% deviation prevention',
                'timeline': '12 weeks',
                'investment': '₹45L',
                'roi': '16 months'
            }
        ]
        
        master_plan['PHASE_1_ACTIONS'] = phase1_actions
        
        # Phase 2 integration (3-9 months)
        phase2_integration = [
            {
                'project': '🔥 Waste Heat Recovery Network',
                'description': 'Integrate heat recovery across all processes',
                'target': '15% total energy reduction',
                'timeline': '24 weeks',
                'investment': '₹2.5Cr',
                'roi': '30 months'
            },
            {
                'project': '🤖 AI Process Control Integration',
                'description': 'Unified AI control across all processes',
                'target': '8% efficiency improvement',
                'timeline': '20 weeks',
                'investment': '₹1.8Cr',
                'roi': '28 months'
            },
            {
                'project': '⚙️ Advanced Process Analytics',
                'description': 'Predictive maintenance and optimization',
                'target': '5% cost reduction',
                'timeline': '16 weeks',
                'investment': '₹1.2Cr',
                'roi': '24 months'
            }
        ]
        
        master_plan['PHASE_2_INTEGRATION'] = phase2_integration
        
        # Phase 3 transformation (9-24 months)
        phase3_transformation = [
            {
                'initiative': '🌍 Carbon Neutral Operations',
                'description': 'Carbon capture, renewable energy, circular economy',
                'target': 'Net zero emissions by 2030',
                'timeline': '18 months',
                'investment': '₹15Cr',
                'roi': 'Strategic/Regulatory'
            },
            {
                'initiative': '🏭 Fully Autonomous Plant',
                'description': 'AI-driven autonomous operations',
                'target': '25% labor cost reduction, 99.5% uptime',
                'timeline': '24 months',
                'investment': '₹8Cr',
                'roi': '48 months'
            },
            {
                'initiative': '♻️ Circular Economy Integration',
                'description': 'Waste-to-resource conversion ecosystem',
                'target': '60% alternative raw materials',
                'timeline': '20 months',
                'investment': '₹12Cr',
                'roi': '42 months'
            }
        ]
        
        master_plan['PHASE_3_TRANSFORMATION'] = phase3_transformation
        
        # Performance targets
        master_plan['PERFORMANCE_TARGETS'] = {
            'energy_efficiency': {
                'current': f"{integrated_data['total_energy_consumption']:.1f} kWh/ton",
                'phase1_target': f"{max(180, integrated_data['total_energy_consumption'] * 0.88):.1f} kWh/ton",
                'phase2_target': f"{max(170, integrated_data['total_energy_consumption'] * 0.75):.1f} kWh/ton",
                'phase3_target': "160 kWh/ton (World class)"
            },
            'sustainability_score': {
                'current': f"{integrated_data['sustainability_index']:.1f}/100",
                'phase1_target': f"{min(100, integrated_data['sustainability_index'] + 10):.1f}/100",
                'phase2_target': f"{min(100, integrated_data['sustainability_index'] + 20):.1f}/100",
                'phase3_target': "95/100 (Industry leader)"
            },
            'cost_reduction': {
                'current': f"₹{integrated_data['total_production_cost']:.0f}/ton",
                'phase1_target': f"₹{integrated_data['total_production_cost'] * 0.92:.0f}/ton",
                'phase2_target': f"₹{integrated_data['total_production_cost'] * 0.85:.0f}/ton",
                'phase3_target': "₹1800/ton (Benchmark)"
            }
        }
        
        # Economic impact
        total_investment = 80 + 50 + 35 + 45 + 250 + 180 + 120 + 1500 + 800 + 1200  # Lakhs
        annual_savings = max(0, cost_gap) * 300000 / 100000  # Lakhs per year
        
        master_plan['ECONOMIC_IMPACT'] = {
            'total_investment_cr': total_investment / 100,
            'annual_savings_cr': annual_savings / 100,
            'payback_period_months': (total_investment / annual_savings * 12) if annual_savings > 0 else 60,
            'roi_5_year_cr': (annual_savings * 5 - total_investment) / 100,
            'breakeven_analysis': f"Breakeven in {(total_investment / annual_savings * 12):.0f} months" if annual_savings > 0 else "Review required"
        }
        
        return master_plan
    
    def monitor_integrated_performance(self, historical_data):
        """Monitor integrated performance across all processes"""
        
        if len(historical_data) < 10:
            return {"error": "Insufficient historical data for monitoring"}
        
        try:
            # Calculate performance trends
            performance_metrics = {
                'energy_trend': self._calculate_trend(historical_data, 'total_energy_consumption'),
                'quality_trend': self._calculate_trend(historical_data, 'overall_quality_score'),
                'sustainability_trend': self._calculate_trend(historical_data, 'sustainability_index'),
                'cost_trend': self._calculate_trend(historical_data, 'total_production_cost'),
                'stability_trend': self._calculate_trend(historical_data, 'process_stability_score')
            }
            
            # Alert generation
            alerts = []
            
            if performance_metrics['energy_trend'] > 5:  # Energy increasing
                alerts.append({
                    'type': 'WARNING',
                    'area': 'Energy Consumption',
                    'message': f"Energy consumption trending up by {performance_metrics['energy_trend']:.1f}%"
                })
            
            if performance_metrics['quality_trend'] < -3:  # Quality declining
                alerts.append({
                    'type': 'CRITICAL',
                    'area': 'Quality Control',
                    'message': f"Quality score declining by {abs(performance_metrics['quality_trend']):.1f}%"
                })
            
            if performance_metrics['stability_trend'] < -5:  # Stability issues
                alerts.append({
                    'type': 'HIGH',
                    'area': 'Process Stability',
                    'message': f"Process stability decreasing by {abs(performance_metrics['stability_trend']):.1f}%"
                })
            
            return {
                'performance_trends': performance_metrics,
                'alerts': alerts,
                'overall_health_score': np.mean([
                    100 - abs(performance_metrics['energy_trend']),
                    100 + performance_metrics['quality_trend'],
                    100 + performance_metrics['sustainability_trend'],
                    100 - abs(performance_metrics['cost_trend']),
                    100 + performance_metrics['stability_trend']
                ]),
                'recommendations': self._generate_trend_recommendations(performance_metrics)
            }
            
        except Exception as e:
            return {"error": f"Performance monitoring failed: {str(e)}"}
    
    def _calculate_trend(self, data, metric):
        """Calculate trend percentage for a metric"""
        if len(data) < 2:
            return 0
        
        recent_avg = np.mean([d[metric] for d in data[-5:]])  # Last 5 points
        baseline_avg = np.mean([d[metric] for d in data[:5]])  # First 5 points
        
        if baseline_avg == 0:
            return 0
            
        trend_pct = ((recent_avg - baseline_avg) / baseline_avg) * 100
        return trend_pct
    
    def _generate_trend_recommendations(self, trends):
        """Generate recommendations based on performance trends"""
        recommendations = []
        
        if trends['energy_trend'] > 3:
            recommendations.append("🔋 Energy consumption increasing - review VFD performance and load optimization")
        
        if trends['quality_trend'] < -2:
            recommendations.append("🧪 Quality declining - increase quality monitoring frequency and review raw material specs")
        
        if trends['sustainability_trend'] < -3:
            recommendations.append("🌱 Sustainability score dropping - accelerate alternative fuel implementation")
        
        if trends['cost_trend'] > 5:
            recommendations.append("💰 Costs rising - implement immediate cost control measures")
        
        if trends['stability_trend'] < -4:
            recommendations.append("⚙️ Process instability - review maintenance schedules and control parameters")
        
        return recommendations