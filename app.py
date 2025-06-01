import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Boston Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load Boston Housing Dataset
@st.cache_data
def load_data():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    return df, boston

# Load data
df, boston = load_data()

# Title
st.title("🏠 Boston Housing Price Analysis Dashboard")
st.markdown("Interactive visualization and analysis of Boston Housing dataset")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Component 1: Selectbox for feature selection
    feature_x = st.selectbox(
        "Select X-axis feature",
        options=list(df.columns),
        index=5  # Default to RM
    )
    
    # Component 2: Selectbox for Y-axis
    feature_y = st.selectbox(
        "Select Y-axis feature",
        options=list(df.columns),
        index=13  # Default to PRICE
    )
    
    # Component 3: Multiselect for features to display
    selected_features = st.multiselect(
        "Select features for correlation analysis",
        options=list(df.columns),
        default=['RM', 'LSTAT', 'PTRATIO', 'PRICE']
    )
    
    # Component 4: Slider for sample size
    sample_size = st.slider(
        "Sample size for visualization",
        min_value=50,
        max_value=len(df),
        value=min(200, len(df)),
        step=50
    )
    
    # Component 5: Radio button for plot type
    plot_type = st.radio(
        "Select visualization type",
        ["Scatter Plot", "3D Scatter", "Box Plot", "Distribution"]
    )
    
    # Component 6: Checkbox for regression line
    show_regression = st.checkbox("Show regression line", value=True)
    
    # Component 7: Color picker
    plot_color = st.color_picker("Choose plot color", "#1f77b4")
    
    # Component 8: Number input for bins
    n_bins = st.number_input(
        "Number of bins (for distribution plot)",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )
    
    # Component 9: Toggle for standardization
    standardize = st.toggle("Standardize data", value=False)

# Data preprocessing
df_sample = df.sample(n=sample_size, random_state=42)

if standardize:
    scaler = StandardScaler()
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    df_sample[numeric_cols] = scaler.fit_transform(df_sample[numeric_cols])

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📈 Statistical Analysis", "🤖 Model Prediction", "📋 Data Overview"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if plot_type == "Scatter Plot":
            fig = px.scatter(df_sample, x=feature_x, y=feature_y, 
                            color='PRICE' if not standardize else None,
                            size='PRICE' if not standardize else None,
                            hover_data=df_sample.columns,
                            title=f"{feature_y} vs {feature_x}",
                            color_continuous_scale=[[0, 'white'], [1, plot_color]])
            
            if show_regression and feature_x != feature_y:
                # Add regression line
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                X = df_sample[feature_x].values.reshape(-1, 1)
                y = df_sample[feature_y].values
                lr.fit(X, y)
                y_pred = lr.predict(X)
                
                fig.add_trace(go.Scatter(
                    x=df_sample[feature_x],
                    y=y_pred,
                    mode='lines',
                    name='Regression Line',
                    line=dict(color='red', width=2)
                ))
                
        elif plot_type == "3D Scatter":
            # Component 10: Additional selectbox for Z-axis
            with st.sidebar:
                feature_z = st.selectbox(
                    "Select Z-axis feature",
                    options=list(df.columns),
                    index=0
                )
            
            fig = px.scatter_3d(df_sample, x=feature_x, y=feature_y, z=feature_z,
                              color='PRICE' if not standardize else None,
                              size='PRICE' if not standardize else None,
                              hover_data=df_sample.columns,
                              title=f"3D Plot: {feature_x} vs {feature_y} vs {feature_z}")
            
        elif plot_type == "Box Plot":
            fig = px.box(df_sample, y=feature_y, 
                        title=f"Distribution of {feature_y}",
                        color_discrete_sequence=[plot_color])
            
        else:  # Distribution
            fig = px.histogram(df_sample, x=feature_y, nbins=n_bins,
                             title=f"Distribution of {feature_y}",
                             color_discrete_sequence=[plot_color])
            fig.add_vline(x=df_sample[feature_y].mean(), 
                         line_dash="dash", 
                         annotation_text="Mean")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Quick Stats")
        if feature_y in df_sample.columns:
            st.metric("Mean", f"{df_sample[feature_y].mean():.2f}")
            st.metric("Median", f"{df_sample[feature_y].median():.2f}")
            st.metric("Std Dev", f"{df_sample[feature_y].std():.2f}")
            st.metric("Min", f"{df_sample[feature_y].min():.2f}")
            st.metric("Max", f"{df_sample[feature_y].max():.2f}")

with tab2:
    st.subheader("Correlation Analysis")
    
    if len(selected_features) > 1:
        # Correlation matrix
        corr_matrix = df[selected_features].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True,
                            color_continuous_scale='RdBu',
                            title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance for Price Prediction")
        if 'PRICE' in selected_features:
            from sklearn.ensemble import RandomForestRegressor
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            X = df[selected_features].drop('PRICE', axis=1)
            y = df['PRICE']
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h',
                           title="Feature Importance",
                           color='Importance',
                           color_continuous_scale=[[0, 'white'], [1, plot_color]])
            st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("🤖 Price Prediction Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Input Features")
        # Create input fields for prediction
        input_data = {}
        for feature in ['RM', 'LSTAT', 'PTRATIO', 'DIS', 'NOX']:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(df[feature].mean()),
                step=0.1
            )
    
    with col2:
        st.write("### Prediction Results")
        
        # Train simple model
        X = df[['RM', 'LSTAT', 'PTRATIO', 'DIS', 'NOX']]
        y = df['PRICE']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Make prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        
        st.metric("Predicted Price", f"${prediction*1000:.2f}")
        
        # Model accuracy
        from sklearn.metrics import r2_score
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        st.metric("Model R² Score", f"{r2:.3f}")

with tab4:
    st.subheader("📋 Data Overview")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_raw = st.checkbox("Show raw data", value=True)
    with col2:
        show_summary = st.checkbox("Show summary statistics", value=True)
    with col3:
        show_info = st.checkbox("Show dataset info", value=True)
    
    if show_info:
        st.write("### Dataset Information")
        st.write(f"- **Number of samples**: {len(df)}")
        st.write(f"- **Number of features**: {len(df.columns)-1}")
        st.write("- **Target variable**: PRICE (Median value of owner-occupied homes in $1000's)")
    
    if show_raw:
        st.write("### Raw Data")
        st.dataframe(df_sample, use_container_width=True)
    
    if show_summary:
        st.write("### Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Boston Housing Dataset")