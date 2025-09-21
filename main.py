import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the optimization modules
from raw_materials_grinding import RawMaterialGrindingOptimizer
from Clinker_Optimization import ClinkerOptimizer
from Quality_Control import QualityControlSystem
from Alternative_Fuels_Optimization import AlternativeFuelsOptimizer
from plant_utilities import PlantUtilitiesOptimizer

# Page configuration
st.set_page_config(
    page_title="Cement Plant AI Optimization Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .alert-success {
        padding: 0.75rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .alert-warning {
        padding: 0.75rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .alert-danger {
        padding: 0.75rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        color: #1f77b4;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimizers_initialized' not in st.session_state:
    st.session_state.optimizers_initialized = False
    st.session_state.optimizers = {}

def initialize_optimizers():
    """Initialize all optimization modules"""
    if not st.session_state.optimizers_initialized:
        with st.spinner("Initializing AI Optimization Modules..."):
            api_key = os.getenv('GEMINI_API_KEY')
            
            st.session_state.optimizers = {
                'raw_materials': RawMaterialGrindingOptimizer(api_key),
                'clinker': ClinkerOptimizer(api_key),
                'quality': QualityControlSystem(api_key),
                'alt_fuels': AlternativeFuelsOptimizer(api_key),
                'utilities': PlantUtilitiesOptimizer(api_key)
            }
            st.session_state.optimizers_initialized = True

def create_dashboard_overview():
    """Create main dashboard overview"""
    st.markdown('<div class="main-header">🏭 Cement Plant AI Optimization Platform</div>', unsafe_allow_html=True)
    
    # Key Performance Indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🔥 Energy Efficiency",
            value="92 kWh/t",
            delta="-8 kWh/t",
            help="Raw meal grinding energy consumption"
        )
    
    with col2:
        st.metric(
            label="🌱 Alt Fuel Ratio",
            value="35%",
            delta="+5%",
            help="Alternative fuel substitution rate"
        )
    
    with col3:
        st.metric(
            label="💪 Cement Strength",
            value="58 MPa",
            delta="+3 MPa",
            help="28-day compressive strength"
        )
    
    with col4:
        st.metric(
            label="🌍 CO₂ Reduction",
            value="820 kg/t",
            delta="-30 kg/t",
            help="CO₂ emissions per ton clinker"
        )
    
    with col5:
        st.metric(
            label="⚡ Power Factor",
            value="0.92",
            delta="+0.04",
            help="Plant power factor"
        )

def render_raw_materials_tab():
    """Render raw materials and grinding optimization tab"""
    st.header("🔨 Raw Materials & Grinding Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Current Parameters")
        
        # Input parameters
        limestone_purity = st.slider("Limestone Purity (%)", 80.0, 95.0, 88.5, 0.1)
        moisture_content = st.slider("Moisture Content (%)", 1.0, 8.0, 4.2, 0.1)
        hardness_index = st.slider("Material Hardness Index", 6.0, 18.0, 10.5, 0.1)
        feed_rate = st.slider("Feed Rate (tons/hr)", 35.0, 70.0, 52.0, 0.5)
        mill_load = st.slider("Mill Load (%)", 60.0, 95.0, 78.0, 1.0)
        blaine_fineness = st.slider("Blaine Fineness (cm²/g)", 3000, 4200, 3500, 50)
        
        # Button to analyze
        if st.button("🔍 Analyze Raw Materials", type="primary"):
            optimizer = st.session_state.optimizers['raw_materials']
            
            # Create sample data and train model
            sample_data = optimizer._create_sample_data(100)
            optimizer.train_models(sample_data)
            
            # Prepare input data
            input_data = {
                'limestone_purity_pct': limestone_purity,
                'moisture_content_pct': moisture_content,
                'hardness_index': hardness_index,
                'feed_rate_tons_hr': feed_rate,
                'mill_load_pct': mill_load,
                'blaine_fineness_cm2_g': blaine_fineness,
                'clay_alumina_pct': 15.2,
                'iron_ore_fe2o3_pct': 3.1,
                'silica_sand_sio2_pct': 12.5
            }
            
            # Get predictions
            predictions = optimizer.predict_grinding_performance(input_data)
            
            # Generate recommendations
            recommendations = optimizer.generate_ai_recommendations(input_data, predictions)
            
            # Store in session state
            st.session_state.raw_materials_results = {
                'predictions': predictions,
                'recommendations': recommendations
            }
    
    with col2:
        st.subheader("📈 Analysis Results")
        
        if 'raw_materials_results' in st.session_state:
            results = st.session_state.raw_materials_results
            predictions = results['predictions']
            recommendations = results['recommendations']
            
            # Display predictions
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                energy_val = predictions['predicted_energy']
                energy_color = "normal" if energy_val <= 85 else "inverse"
                st.metric("🔋 Predicted Energy", f"{energy_val:.1f} kWh/t", 
                         delta=f"{energy_val-78:.1f} vs target", delta_color=energy_color)
            
            with pred_col2:
                anomaly_status = "🚨 ANOMALY" if predictions['is_anomaly'] else "✅ NORMAL"
                st.metric("🔍 Process Status", anomaly_status, 
                         delta=f"Score: {predictions['anomaly_score']:.3f}")
            
            with pred_col3:
                efficiency = 78 / energy_val * 100
                st.metric("⚡ Efficiency", f"{efficiency:.1f}%", 
                         delta=f"{efficiency-90:.1f}% vs benchmark")
            
            # Display recommendations
            st.subheader("🎯 AI Recommendations")
            
            if isinstance(recommendations, dict):
                for category, actions in recommendations.items():
                    if actions and category != 'performance_metrics':
                        with st.expander(f"📋 {category.replace('_', ' ').title()}", expanded=True):
                            if isinstance(actions, list):
                                for action in actions:
                                    if "HIGH:" in action:
                                        st.markdown(f'<div class="alert-danger">{action}</div>', unsafe_allow_html=True)
                                    elif "MEDIUM:" in action:
                                        st.markdown(f'<div class="alert-warning">{action}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="alert-success">{action}</div>', unsafe_allow_html=True)
                            else:
                                st.write(f"• {actions}")
        else:
            st.info("👆 Adjust parameters and click 'Analyze Raw Materials' to see results")
            
            # Show sample visualization
            sample_data = st.session_state.optimizers['raw_materials']._create_sample_data(50)
            
            fig = px.scatter(sample_data, 
                           x='hardness_index', 
                           y='grinding_energy_kwh_ton',
                           color='moisture_content_pct',
                           size='limestone_purity_pct',
                           title="Energy vs Material Properties",
                           labels={'hardness_index': 'Material Hardness Index',
                                  'grinding_energy_kwh_ton': 'Energy (kWh/ton)'})
            st.plotly_chart(fig, use_container_width=True)

def render_clinker_tab():
    """Render clinker optimization tab"""
    st.header("🔥 Clinker Production Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🌡️ Kiln Parameters")
        
        kiln_temp = st.slider("Kiln Temperature (°C)", 1420, 1480, 1450, 5)
        oxygen_level = st.slider("Oxygen Level (%)", 2.0, 6.0, 3.5, 0.1)
        feed_rate = st.slider("Feed Rate (t/hr)", 35, 58, 45, 1)
        alt_fuel_ratio = st.slider("Alternative Fuel Ratio (%)", 5, 40, 22, 1)
        lsf = st.slider("LSF (Lime Saturation Factor)", 0.88, 1.02, 0.95, 0.01)
        sm = st.slider("Silica Modulus", 1.8, 3.2, 2.4, 0.1)
        am = st.slider("Alumina Modulus", 1.0, 2.0, 1.4, 0.1)
        
        if st.button("🔥 Optimize Clinker", type="primary"):
            optimizer = st.session_state.optimizers['clinker']
            
            # Create and train model
            sample_data = optimizer._create_sample_data(100)
            optimizer.train_models(sample_data)
            
            input_data = {
                'kiln_temperature_celsius': kiln_temp,
                'oxygen_level_pct': oxygen_level,
                'feed_rate_tons_hr': feed_rate,
                'alt_fuel_ratio_pct': alt_fuel_ratio,
                'lsf': lsf,
                'silica_modulus': sm,
                'alumina_modulus': am,
                'raw_meal_fineness_cm2_g': 3600
            }
            
            predictions = optimizer.predict_clinker_performance(input_data)
            recommendations = optimizer.generate_ai_recommendations(input_data, predictions)
            
            st.session_state.clinker_results = {
                'predictions': predictions,
                'recommendations': recommendations
            }
    
    with col2:
        st.subheader("📊 Clinker Performance")
        
        if 'clinker_results' in st.session_state:
            results = st.session_state.clinker_results
            predictions = results['predictions']
            recommendations = results['recommendations']
            
            # Performance metrics
            met_col1, met_col2, met_col3 = st.columns(3)
            
            with met_col1:
                fuel_val = predictions['predicted_fuel_consumption']
                fuel_color = "normal" if fuel_val <= 720 else "inverse"
                st.metric("⛽ Fuel Consumption", f"{fuel_val:.0f} kcal/kg",
                         delta=f"{fuel_val-720:.0f} vs target", delta_color=fuel_color)
            
            with met_col2:
                strength_val = predictions['predicted_strength_28_day']
                strength_color = "normal" if strength_val >= 55 else "inverse"
                st.metric("💪 28-day Strength", f"{strength_val:.1f} MPa",
                         delta=f"{strength_val-55:.1f} vs min", delta_color=strength_color)
            
            with met_col3:
                co2_val = predictions['estimated_co2_emissions']
                co2_color = "normal" if co2_val <= 850 else "inverse"
                st.metric("🌍 CO₂ Emissions", f"{co2_val:.0f} kg/t",
                         delta=f"{co2_val-850:.0f} vs target", delta_color=co2_color)
            
            # Recommendations display
            st.subheader("🎯 Optimization Recommendations")
            
            if isinstance(recommendations, dict):
                for category, actions in recommendations.items():
                    if actions and category != 'performance_analysis':
                        with st.expander(f"🔧 {category.replace('_', ' ').title()}", expanded=True):
                            if isinstance(actions, list):
                                for action in actions:
                                    priority = "HIGH" if "HIGH:" in action else "MEDIUM" if "MEDIUM:" in action else "LOW"
                                    if priority == "HIGH":
                                        st.markdown(f'<div class="alert-danger">{action}</div>', unsafe_allow_html=True)
                                    elif priority == "MEDIUM":
                                        st.markdown(f'<div class="alert-warning">{action}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="alert-success">{action}</div>', unsafe_allow_html=True)
                            else:
                                st.write(f"• {actions}")
        else:
            st.info("👆 Set kiln parameters and click 'Optimize Clinker' to see results")
            
            # Sample chart
            sample_data = st.session_state.optimizers['clinker']._create_sample_data(50)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=sample_data['kiln_temperature_celsius'], 
                          y=sample_data['fuel_consumption_kcal_kg'],
                          name="Fuel Consumption", mode='markers'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=sample_data['kiln_temperature_celsius'], 
                          y=sample_data['strength_28_day_mpa'],
                          name="28-day Strength", mode='markers'),
                secondary_y=True
            )
            fig.update_xaxes(title_text="Kiln Temperature (°C)")
            fig.update_yaxes(title_text="Fuel Consumption (kcal/kg)", secondary_y=False)
            fig.update_yaxes(title_text="28-day Strength (MPa)", secondary_y=True)
            fig.update_layout(title="Temperature vs Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)

def render_quality_control_tab():
    """Render quality control tab"""
    st.header("🔬 Quality Control & Testing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧪 Quality Parameters")
        
        # Chemical composition inputs
        st.write("**Chemical Composition:**")
        cao = st.slider("CaO (%)", 60.0, 68.0, 64.2, 0.1)
        sio2 = st.slider("SiO₂ (%)", 19.0, 25.0, 21.8, 0.1)
        al2o3 = st.slider("Al₂O₃ (%)", 4.0, 8.0, 5.5, 0.1)
        fe2o3 = st.slider("Fe₂O₃ (%)", 2.0, 5.0, 3.2, 0.1)
        
        # Physical properties
        st.write("**Physical Properties:**")
        blaine_fineness = st.slider("Blaine Fineness (cm²/g)", 2800, 4500, 3200, 50)
        residue_45 = st.slider("Residue on 45μm (%)", 8.0, 18.0, 12.0, 0.5)
        initial_set = st.slider("Initial Setting (min)", 45, 180, 120, 5)
        
        # Strength parameters
        st.write("**Strength Tests:**")
        strength_1_day = st.slider("1-day Strength (MPa)", 8.0, 25.0, 15.0, 0.5)
        strength_3_day = st.slider("3-day Strength (MPa)", 20.0, 40.0, 28.0, 0.5)
        strength_7_day = st.slider("7-day Strength (MPa)", 30.0, 50.0, 38.0, 0.5)
        
        if st.button("🧪 Analyze Quality", type="primary"):
            optimizer = st.session_state.optimizers['quality']
            
            # Create and train model
            sample_data = optimizer._create_sample_data(100)
            optimizer.train_models(sample_data)
            
            input_data = {
                'cao_percent': cao,
                'sio2_percent': sio2,
                'al2o3_percent': al2o3,
                'fe2o3_percent': fe2o3,
                'blaine_fineness_cm2_g': blaine_fineness,
                'residue_45_micron_pct': residue_45,
                'initial_setting_min': initial_set,
                'strength_1_day_mpa': strength_1_day,
                'strength_3_day_mpa': strength_3_day,
                'strength_7_day_mpa': strength_7_day
            }
            
            predictions = optimizer.predict_quality_parameters(input_data)
            
            # Create quality_issues dictionary instead of passing boolean
            quality_issues = {
                'critical_issues': [],
                'warning_issues': [],
                'recommendations': []
            }
            
            # Add logic to populate quality_issues based on predictions if needed
            if predictions.get('predicted_28_day_strength', 0) < 52.5:
                quality_issues['critical_issues'].append("28-day strength below specification")
            if predictions.get('compliance_score', 1.0) < 0.9:
                quality_issues['warning_issues'].append("Compliance score below target")
            
            recommendations = optimizer.generate_ai_recommendations(input_data, predictions, quality_issues)
            
            st.session_state.quality_results = {
                'predictions': predictions,
                'recommendations': recommendations
            }
    
    with col2:
        st.subheader("📈 Quality Analysis")
        
        if 'quality_results' in st.session_state:
            results = st.session_state.quality_results
            predictions = results['predictions']
            recommendations = results['recommendations']
            
            # Display predictions
            qual_col1, qual_col2, qual_col3 = st.columns(3)
            
            with qual_col1:
                strength_28 = predictions['predicted_28_day_strength']
                strength_color = "normal" if strength_28 >= 52.5 else "inverse"
                st.metric("🎯 28-day Strength", f"{strength_28:.1f} MPa",
                         delta=f"{strength_28-52.5:.1f} vs spec", delta_color=strength_color)
            
            with qual_col2:
                compliance_score = predictions.get('compliance_score', 0.85) * 100
                compliance_color = "normal" if compliance_score >= 90 else "inverse"
                st.metric("✅ Compliance Score", f"{compliance_score:.1f}%",
                         delta=f"{compliance_score-90:.1f}% vs target", delta_color=compliance_color)
            
            with qual_col3:
                anomaly_status = "🚨 ISSUE" if predictions.get('is_anomaly', False) else "✅ OK"
                st.metric("🔍 Quality Status", anomaly_status,
                         delta=f"Score: {predictions.get('anomaly_score', 0.1):.3f}")
            
            # Quality recommendations
            st.subheader("🎯 Quality Improvement Recommendations")
            
            if isinstance(recommendations, dict):
                for category, actions in recommendations.items():
                    if actions and category not in ['performance_metrics', 'quality_analysis']:
                        with st.expander(f"🔬 {category.replace('_', ' ').title()}", expanded=True):
                            if isinstance(actions, list):
                                for action in actions:
                                    if "CRITICAL:" in action:
                                        st.markdown(f'<div class="alert-danger">{action}</div>', unsafe_allow_html=True)
                                    elif "WARNING:" in action:
                                        st.markdown(f'<div class="alert-warning">{action}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="alert-success">{action}</div>', unsafe_allow_html=True)
                            else:
                                st.write(f"• {actions}")
        else:
            st.info("👆 Set quality parameters and click 'Analyze Quality' to see results")
            
            # Sample quality trend chart
            sample_data = st.session_state.optimizers['quality']._create_sample_data(30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sample_data['strength_28_day_mpa'], 
                name='28-day Strength',
                line=dict(color='blue', width=2)
            ))
            fig.add_hline(y=52.5, line_dash="dash", line_color="red", 
                         annotation_text="Min Spec (52.5 MPa)")
            fig.update_layout(
                title="28-Day Strength Trend (Sample Data)",
                xaxis_title="Sample Number",
                yaxis_title="Strength (MPa)"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_alt_fuels_tab():
    """Render alternative fuels optimization tab"""
    st.header("🌱 Alternative Fuels Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚡ Fuel Mix Parameters")
        
        # Fuel composition
        st.write("**Current Fuel Mix:**")
        coal_ratio = st.slider("Coal (%)", 40, 80, 65, 1)
        petcoke_ratio = st.slider("Petcoke (%)", 0, 30, 15, 1)
        biomass_ratio = st.slider("Biomass (%)", 0, 25, 12, 1)
        waste_derived_ratio = st.slider("Waste-derived Fuel (%)", 0, 20, 8, 1)
        
        # Normalize to 100%
        total_ratio = coal_ratio + petcoke_ratio + biomass_ratio + waste_derived_ratio
        if total_ratio != 100:
            st.warning(f"Total fuel mix: {total_ratio}% (should be 100%)")
        
        # Operating conditions
        st.write("**Operating Conditions:**")
        kiln_temp = st.slider("Average Kiln Temp (°C)", 1420, 1480, 1450, 5)
        thermal_substitution = st.slider("Current TSR (%)", 10, 50, 25, 1)
        oxygen_content = st.slider("Oxygen in Flue Gas (%)", 2.5, 5.5, 3.8, 0.1)
        
        if st.button("🌱 Optimize Fuel Mix", type="primary"):
            optimizer = st.session_state.optimizers['alt_fuels']
            
            # Create and train model
            sample_data = optimizer._create_sample_data(100)
            optimizer.train_models(sample_data)
            
            input_data = {
                'coal_ratio_pct': coal_ratio,
                'petcoke_ratio_pct': petcoke_ratio,
                'biomass_ratio_pct': biomass_ratio,
                'waste_derived_ratio_pct': waste_derived_ratio,
                'kiln_temperature_celsius': kiln_temp,
                'thermal_substitution_rate_pct': thermal_substitution,
                'oxygen_content_pct': oxygen_content,
                'feed_rate_tons_hr': 45.0
            }
            
            predictions = optimizer.predict_fuel_performance(input_data)
            
            # Create optimization_scenarios dictionary
            optimization_scenarios = {
                'cost_optimization': {
                    'priority': 'high',
                    'target_savings': 10.0
                },
                'environmental_optimization': {
                    'priority': 'high',
                    'co2_reduction_target': 15.0
                },
                'stability_optimization': {
                    'priority': 'medium',
                    'min_stability_score': 0.8
                }
            }
            
            recommendations = optimizer.generate_ai_recommendations(input_data, predictions, optimization_scenarios)
            
            st.session_state.alt_fuels_results = {
                'predictions': predictions,
                'recommendations': recommendations
            }
    
    with col2:
        st.subheader("📊 Fuel Performance Analysis")
        
        if 'alt_fuels_results' in st.session_state:
            results = st.session_state.alt_fuels_results
            predictions = results['predictions']
            recommendations = results['recommendations']
            
            # Performance metrics
            fuel_col1, fuel_col2, fuel_col3 = st.columns(3)
            
            with fuel_col1:
                co2_reduction = predictions.get('predicted_co2_reduction', 15)
                co2_color = "normal" if co2_reduction >= 10 else "inverse"
                st.metric("🌍 CO₂ Reduction", f"{co2_reduction:.1f}%",
                         delta=f"{co2_reduction-10:.1f}% vs baseline", delta_color=co2_color)
            
            with fuel_col2:
                cost_saving = predictions.get('predicted_cost_saving', 8.5)
                cost_color = "normal" if cost_saving >= 5 else "inverse"
                st.metric("💰 Cost Saving", f"{cost_saving:.1f}%",
                         delta=f"{cost_saving-5:.1f}% vs target", delta_color=cost_color)
            
            with fuel_col3:
                stability_score = predictions.get('predicted_stability_score', 0.82) * 100
                stability_color = "normal" if stability_score >= 80 else "inverse"
                st.metric("⚖️ Process Stability", f"{stability_score:.0f}%",
                         delta=f"{stability_score-80:.0f}% vs min", delta_color=stability_color)
            
            # Fuel mix optimization chart
            fuel_types = ['Coal', 'Petcoke', 'Biomass', 'Waste-derived']
            current_mix = [coal_ratio, petcoke_ratio, biomass_ratio, waste_derived_ratio]
            
            # Generate optimized mix (example)
            optimized_mix = [
                max(50, coal_ratio - 5),
                petcoke_ratio,
                min(biomass_ratio + 3, 20),
                min(waste_derived_ratio + 2, 15)
            ]
            
            fig = go.Figure(data=[
                go.Bar(name='Current Mix', x=fuel_types, y=current_mix),
                go.Bar(name='Optimized Mix', x=fuel_types, y=optimized_mix)
            ])
            fig.update_layout(
                title="Current vs Optimized Fuel Mix",
                xaxis_title="Fuel Type",
                yaxis_title="Percentage (%)",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            st.subheader("🎯 Fuel Optimization Recommendations")
            
            if isinstance(recommendations, dict):
                for category, actions in recommendations.items():
                    if actions and category not in ['performance_analysis', 'fuel_analysis']:
                        with st.expander(f"🔥 {category.replace('_', ' ').title()}", expanded=True):
                            if isinstance(actions, list):
                                for action in actions:
                                    if "HIGH:" in action:
                                        st.markdown(f'<div class="alert-danger">{action}</div>', unsafe_allow_html=True)
                                    elif "MEDIUM:" in action:
                                        st.markdown(f'<div class="alert-warning">{action}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="alert-success">{action}</div>', unsafe_allow_html=True)
                            else:
                                st.write(f"• {actions}")
        else:
            st.info("👆 Set fuel parameters and click 'Optimize Fuel Mix' to see results")
            
def render_utilities_tab():
    """Render plant utilities optimization tab"""
    st.header("⚡ Plant Utilities Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔌 Electrical Parameters")
        
        # Power consumption
        st.write("**Power Consumption:**")
        total_power = st.slider("Total Power (MW)", 15.0, 35.0, 22.5, 0.5)
        grinding_power = st.slider("Grinding Power (MW)", 8.0, 18.0, 12.0, 0.5)
        kiln_power = st.slider("Kiln Drive Power (MW)", 2.0, 6.0, 3.5, 0.1)
        fan_power = st.slider("Fan Systems (MW)", 3.0, 8.0, 5.2, 0.1)
        
        # Power quality
        st.write("**Power Quality:**")
        power_factor = st.slider("Power Factor", 0.75, 0.98, 0.88, 0.01)
        voltage_stability = st.slider("Voltage Stability (%)", 95.0, 100.0, 98.2, 0.1)
        harmonics_thd = st.slider("Total Harmonic Distortion (%)", 2.0, 8.0, 4.5, 0.1)
        
        # Compressed air system
        st.write("**Compressed Air:**")
        compressor_load = st.slider("Compressor Load (%)", 60.0, 95.0, 78.0, 1.0)
        air_pressure = st.slider("System Pressure (bar)", 6.5, 8.5, 7.2, 0.1)
        air_leakage = st.slider("Estimated Leakage (%)", 15.0, 35.0, 22.0, 1.0)
        
        if st.button("⚡ Optimize Utilities", type="primary"):
            optimizer = st.session_state.optimizers['utilities']
            
            # Create and train model
            sample_data = optimizer._create_sample_data(100)
            optimizer.train_models(sample_data)
            
            input_data = {
                'total_power_consumption_mw': total_power,
                'grinding_power_mw': grinding_power,
                'kiln_power_mw': kiln_power,
                'fan_power_mw': fan_power,
                'power_factor': power_factor,
                'voltage_stability_pct': voltage_stability,
                'harmonics_thd_pct': harmonics_thd,
                'compressor_load_pct': compressor_load,
                'air_pressure_bar': air_pressure,
                'air_leakage_pct': air_leakage
            }
            
            predictions = optimizer.predict_utilities_performance(input_data)
            recommendations = optimizer.generate_ai_recommendations(input_data, predictions)
            
            st.session_state.utilities_results = {
                'predictions': predictions,
                'recommendations': recommendations
            }
    
    with col2:
        st.subheader("📊 Utilities Performance")
        
        if 'utilities_results' in st.session_state:
            results = st.session_state.utilities_results
            predictions = results['predictions']
            recommendations = results['recommendations']
            
            # Performance metrics
            util_col1, util_col2, util_col3 = st.columns(3)
            
            with util_col1:
                efficiency = predictions.get('predicted_efficiency', 85.5)
                efficiency_color = "normal" if efficiency >= 85 else "inverse"
                st.metric("⚡ Energy Efficiency", f"{efficiency:.1f}%",
                         delta=f"{efficiency-85:.1f}% vs target", delta_color=efficiency_color)
            
            with util_col2:
                cost_kwh = predictions.get('predicted_cost_per_kwh', 0.085)
                cost_color = "normal" if cost_kwh <= 0.09 else "inverse"
                st.metric("💰 Cost per kWh", f"${cost_kwh:.3f}",
                         delta=f"{(cost_kwh-0.09)*1000:.1f} vs budget", delta_color=cost_color)
            
            with util_col3:
                reliability = predictions.get('predicted_reliability', 94.2)
                reliability_color = "normal" if reliability >= 95 else "inverse"
                st.metric("🔧 System Reliability", f"{reliability:.1f}%",
                         delta=f"{reliability-95:.1f}% vs target", delta_color=reliability_color)
            
            # Power consumption breakdown
            power_data = {
                'System': ['Grinding', 'Kiln Drive', 'Fans', 'Other'],
                'Power (MW)': [grinding_power, kiln_power, fan_power, 
                              total_power - grinding_power - kiln_power - fan_power],
                'Percentage': [
                    grinding_power/total_power*100,
                    kiln_power/total_power*100,
                    fan_power/total_power*100,
                    (total_power - grinding_power - kiln_power - fan_power)/total_power*100
                ]
            }
            
            fig = px.pie(
                values=power_data['Power (MW)'],
                names=power_data['System'],
                title="Power Consumption Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            st.subheader("🎯 Utilities Optimization Recommendations")
            
            if isinstance(recommendations, dict):
                for category, actions in recommendations.items():
                    if actions and category not in ['performance_analysis', 'utilities_analysis']:
                        with st.expander(f"⚡ {category.replace('_', ' ').title()}", expanded=True):
                            if isinstance(actions, list):
                                for action in actions:
                                    if "URGENT:" in action:
                                        st.markdown(f'<div class="alert-danger">{action}</div>', unsafe_allow_html=True)
                                    elif "IMPORTANT:" in action:
                                        st.markdown(f'<div class="alert-warning">{action}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="alert-success">{action}</div>', unsafe_allow_html=True)
                            else:
                                st.write(f"• {actions}")
        else:
            st.info("👆 Set utilities parameters and click 'Optimize Utilities' to see results")
            
            # Sample power trend
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            power_trend = np.random.normal(22.5, 2.5, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=power_trend,
                mode='lines+markers',
                name='Total Power Consumption',
                line=dict(color='orange', width=2)
            ))
            fig.add_hline(y=25, line_dash="dash", line_color="red", 
                         annotation_text="Design Limit (25 MW)")
            fig.update_layout(
                title="Power Consumption Trend (Sample Data)",
                xaxis_title="Date",
                yaxis_title="Power (MW)"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_reports_tab():
    """Render comprehensive reports tab"""
    st.header("📋 Comprehensive Reports & Analytics")
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type:",
        ["Daily Performance Summary", "Weekly Optimization Report", 
         "Monthly Sustainability Report", "Annual Efficiency Analysis"]
    )
    
    if st.button("📊 Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            # Simulate report generation
            current_time = datetime.now()
            
            if report_type == "Daily Performance Summary":
                st.subheader(f"📅 Daily Performance Summary - {current_time.strftime('%Y-%m-%d')}")
                
                # Key metrics summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏭 Production", "2,850 tons", "+125 tons")
                    st.metric("⚡ Energy Consumption", "85.2 kWh/t", "-3.8 kWh/t")
                
                with col2:
                    st.metric("🌱 Alt Fuel Usage", "28.5%", "+2.3%")
                    st.metric("💪 Avg Strength", "56.8 MPa", "+1.2 MPa")
                
                with col3:
                    st.metric("🌍 CO₂ Emissions", "832 kg/t", "-18 kg/t")
                    st.metric("⚖️ Process Stability", "94.2%", "+1.8%")
                
                with col4:
                    st.metric("💰 Cost per Ton", "$78.50", "-$2.30")
                    st.metric("🔧 Equipment Uptime", "98.1%", "+0.5%")
                
                # Performance trends
                st.subheader("📈 Performance Trends")
                
                # Generate sample trend data
                hours = list(range(24))
                production_rate = [2800 + np.random.normal(0, 100) for _ in hours]
                energy_consumption = [85 + np.random.normal(0, 5) for _ in hours]
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=hours, y=production_rate, name="Production Rate (t/day)", 
                             line=dict(color='blue')),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Scatter(x=hours, y=energy_consumption, name="Energy Consumption (kWh/t)", 
                             line=dict(color='red')),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Hour of Day")
                fig.update_yaxes(title_text="Production Rate (t/day)", secondary_y=False)
                fig.update_yaxes(title_text="Energy Consumption (kWh/t)", secondary_y=True)
                fig.update_layout(title="Hourly Performance Trends")
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif report_type == "Monthly Sustainability Report":
                st.subheader(f"🌍 Monthly Sustainability Report - {current_time.strftime('%B %Y')}")
                
                # Sustainability metrics
                sust_col1, sust_col2 = st.columns(2)
                
                with sust_col1:
                    st.markdown("### 🌱 Environmental Impact")
                    
                    # CO2 emissions chart
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                    co2_emissions = [875, 860, 845, 838, 825, 810]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=months, y=co2_emissions,
                        mode='lines+markers',
                        name='CO₂ Emissions (kg/t)',
                        line=dict(color='green', width=3)
                    ))
                    fig.add_hline(y=850, line_dash="dash", line_color="orange", 
                                 annotation_text="Target (850 kg/t)")
                    fig.update_layout(title="CO₂ Emissions Trend")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Alternative fuels usage
                    alt_fuel_data = {
                        'Fuel Type': ['Biomass', 'Waste-derived', 'Used Tires', 'Other'],
                        'Usage (%)': [12.5, 8.3, 4.2, 3.8]
                    }
                    
                    fig = px.bar(alt_fuel_data, x='Fuel Type', y='Usage (%)', 
                               title="Alternative Fuels Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
                
                with sust_col2:
                    st.markdown("### ⚡ Resource Efficiency")
                    
                    # Energy efficiency metrics
                    efficiency_data = {
                        'Process': ['Raw Grinding', 'Kiln Operation', 'Cement Grinding', 'Utilities'],
                        'Current (kWh/t)': [32, 45, 28, 15],
                        'Benchmark (kWh/t)': [30, 42, 26, 14]
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Current', x=efficiency_data['Process'], 
                              y=efficiency_data['Current (kWh/t)']),
                        go.Bar(name='Benchmark', x=efficiency_data['Process'], 
                              y=efficiency_data['Benchmark (kWh/t)'])
                    ])
                    fig.update_layout(title="Energy Efficiency vs Benchmark", barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Water usage
                    st.metric("💧 Water Usage", "0.28 m³/t", "-0.03 m³/t vs last month")
                    st.metric("♻️ Waste Heat Recovery", "15.2%", "+2.1% improvement")
                    st.metric("🌿 Dust Emissions", "18 mg/Nm³", "-3 mg/Nm³ reduction")
                
                # Sustainability recommendations
                st.subheader("🎯 Sustainability Recommendations")
                
                recommendations = [
                    "✅ Achieved 28.8% alternative fuel substitution rate (target: 25%)",
                    "⚠️ Consider increasing biomass usage to further reduce CO₂ emissions",
                    "📈 Raw grinding efficiency improved by 4.2% through optimization",
                    "🔄 Implement waste heat recovery system for additional 3% efficiency gain",
                    "💡 LED lighting upgrade completed, saving 120 MWh annually"
                ]
                
                for rec in recommendations:
                    if "✅" in rec:
                        st.markdown(f'<div class="alert-success">{rec}</div>', unsafe_allow_html=True)
                    elif "⚠️" in rec:
                        st.markdown(f'<div class="alert-warning">{rec}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-success">{rec}</div>', unsafe_allow_html=True)
            
            # Export options
            st.subheader("📤 Export Options")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Export as PDF"):
                    st.success("PDF export initiated! Check your downloads folder.")
            
            with export_col2:
                if st.button("📈 Export as Excel"):
                    st.success("Excel export initiated! Check your downloads folder.")
            
            with export_col3:
                if st.button("📧 Email Report"):
                    st.success("Report scheduled for email delivery!")

def main():
    """Main application function"""
    
    # Initialize optimizers
    initialize_optimizers()
    
    # Sidebar navigation
    st.sidebar.title("🏭 Navigation")
    st.sidebar.markdown("---")
    
    # Main navigation
    nav_option = st.sidebar.radio(
        "Select Module:",
        ["🏠 Dashboard Overview", "🔨 Raw Materials", "🔥 Clinker Production", 
         "🔬 Quality Control", "🌱 Alternative Fuels", "⚡ Plant Utilities", 
         "📋 Reports & Analytics"],
        key="main_nav"
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    status_items = [
        ("🟢", "Raw Mill", "Online"),
        ("🟢", "Kiln", "Optimal"),
        ("🟡", "Cement Mill", "Warning"),
        ("🟢", "Power Systems", "Normal"),
        ("🟢", "Quality Lab", "Active")
    ]
    
    for indicator, system, status in status_items:
        st.sidebar.write(f"{indicator} **{system}**: {status}")
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("Production Today", "2,850 tons")
    st.sidebar.metric("Energy Efficiency", "85.2 kWh/t")
    st.sidebar.metric("Quality Score", "94.2%")
    
    # Main content area
    if nav_option == "🏠 Dashboard Overview":
        create_dashboard_overview()
        
        # Recent alerts and notifications
        st.subheader("🚨 Recent Alerts & Notifications")
        
        alerts = [
            ("🟡", "MEDIUM", "Raw mill vibration levels elevated - monitoring required"),
            ("🟢", "INFO", "Alternative fuel ratio target achieved (28.5%)"),
            ("🟡", "MEDIUM", "Cement mill power consumption 5% above normal"),
            ("🟢", "INFO", "Quality test results: All parameters within specification"),
            ("🔴", "HIGH", "Kiln oxygen level fluctuation detected - immediate attention needed")
        ]
        
        for indicator, priority, message in alerts:
            if priority == "HIGH":
                st.markdown(f'<div class="alert-danger">{indicator} **{priority}**: {message}</div>', 
                           unsafe_allow_html=True)
            elif priority == "MEDIUM":
                st.markdown(f'<div class="alert-warning">{indicator} **{priority}**: {message}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success">{indicator} **{priority}**: {message}</div>', 
                           unsafe_allow_html=True)
    
    elif nav_option == "🔨 Raw Materials":
        render_raw_materials_tab()
    
    elif nav_option == "🔥 Clinker Production":
        render_clinker_tab()
    
    elif nav_option == "🔬 Quality Control":
        render_quality_control_tab()
    
    elif nav_option == "🌱 Alternative Fuels":
        render_alt_fuels_tab()
    
    elif nav_option == "⚡ Plant Utilities":
        render_utilities_tab()
    
    elif nav_option == "📋 Reports & Analytics":
        render_reports_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏭 <strong>Cement Plant AI Optimization Platform</strong> v2.1.0</p>
        <p>Powered by Advanced Machine Learning & AI Analytics</p>
        <p><em>Optimizing cement production for efficiency, quality, and sustainability</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()