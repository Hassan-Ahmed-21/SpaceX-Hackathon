

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

# Set up the app
st.set_page_config(page_title="Space Launch Analytics", layout="wide")

# Load sample data (replace with your actual data source)
@st.cache_data
def load_data():
    # Sample data - replace with real launch data
    data = pd.DataFrame({
        'date': pd.date_range(start='1950-01-01', end='2023-12-31', periods=500),
        'mission': [f'Mission {i}' for i in range(500)],
        'launch_site': ['Kennedy Space Center', 'Baikonur Cosmodrome', 'Vandenberg AFB', 
                        'Cape Canaveral', 'Jiuquan Satellite Launch Center'] * 100,
        'vehicle': ['Falcon 9', 'Soyuz', 'Ariane 5', 'Long March', 'Atlas V'] * 100,
        'outcome': ['Success', 'Failure', 'Partial Failure'] * 166 + ['Success', 'Failure'],
        'payload_mass_kg': [1000 + i*50 for i in range(500)],
        'lat': [28.5729, 45.965, 34.742, 28.4889, 40.957] * 100,
        'lon': [-80.6489, 63.305, -120.537, -80.5778, 100.291] * 100,
        'orbit': ['LEO', 'GEO', 'MEO', 'HEO'] * 125
    })
    data['year'] = data['date'].dt.year
    return data

df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2000, 2023)
)

launch_sites = st.sidebar.multiselect(
    "Select Launch Sites",
    options=df['launch_site'].unique(),
    default=df['launch_site'].unique()
)

outcomes = st.sidebar.multiselect(
    "Select Outcomes",
    options=df['outcome'].unique(),
    default=['Success', 'Partial Failure']
)

# Filter data based on selections
filtered_df = df[
    (df['year'].between(selected_years[0], selected_years[1])) &
    (df['launch_site'].isin(launch_sites)) &
    (df['outcome'].isin(outcomes))
]

# Main app
st.title("🚀 Space Launch Analytics Dashboard")

# Tab layout
tab1, tab2, tab3 = st.tabs(["Historical Data", "Launch Site Map", "Success Predictor"])

with tab1:
    st.header("Historical Launch Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Launches", len(filtered_df))
    
    with col2:
        success_rate = len(filtered_df[filtered_df['outcome'] == 'Success']) / len(filtered_df) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Launch trend chart
    fig = px.line(
        filtered_df.groupby('year').size().reset_index(name='count'),
        x='year',
        y='count',
        title='Launches by Year'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.dataframe(
        filtered_df[['date', 'mission', 'launch_site', 'vehicle', 'outcome', 'payload_mass_kg']],
        column_config={
            "date": "Date",
            "mission": "Mission",
            "launch_site": "Launch Site",
            "vehicle": "Vehicle",
            "outcome": "Outcome",
            "payload_mass_kg": "Payload Mass (kg)"
        },
        hide_index=True,
        use_container_width=True
    )

with tab2:
    st.header("Launch Site Map")
    
    # Create Folium map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Color mapping for outcomes
    color_map = {
        'Success': 'green',
        'Failure': 'red',
        'Partial Failure': 'orange'
    }
    
    # Add markers
    for idx, row in filtered_df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"""
            <b>{row['mission']}</b><br>
            Date: {row['date'].date()}<br>
            Vehicle: {row['vehicle']}<br>
            Outcome: <span style='color:{color_map[row['outcome']]}'>{row['outcome']}</span><br>
            Payload: {row['payload_mass_kg']} kg
            """,
            icon=folium.Icon(color=color_map[row['outcome']], icon='rocket', prefix='fa')
        ).add_to(m)
    
    # Display map
    st_folium(m, width=1200, height=600)

with tab3:
    st.header("Launch Success Predictor")
    
    # Train a simple model (in a real app, you'd load a pre-trained model)
    @st.cache_resource
    def train_model():
        # Simple binary classification (success/failure)
        X = df[['payload_mass_kg', 'year']]
        y = (df['outcome'] == 'Success').astype(int)
        model = RandomForestClassifier()
        model.fit(X, y)
        return model
    
    model = train_model()
    
    # Prediction form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=df['vehicle'].unique()
            )
            
            payload_mass = st.number_input(
                "Payload Mass (kg)",
                min_value=0,
                max_value=100000,
                value=5000
            )
        
        with col2:
            launch_site = st.selectbox(
                "Launch Site",
                options=df['launch_site'].unique()
            )
            
            orbit_type = st.selectbox(
                "Target Orbit",
                options=df['orbit'].unique()
            )
        
        submitted = st.form_submit_button("Calculate Success Probability")
    
    if submitted:
        # Make prediction (simplified - in reality you'd use more features)
        current_year = datetime.now().year
        proba = model.predict_proba([[payload_mass, current_year]])[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Success Probability", f"{proba*100:.1f}%")
            
            # Gauge visualization
            st.progress(int(proba*100))
        
        with col2:
            st.write("Key Factors:")
            st.write(f"- Payload mass: {'Within' if payload_mass < 10000 else 'Above'} typical range")
            st.write(f"- Vehicle type: {vehicle_type} historical success rate")
            st.write(f"- Launch site: {launch_site} track record")
            
            if proba > 0.7:
                st.success("High probability of success!")
            elif proba > 0.4:
                st.warning("Moderate probability of success")
            else:
                st.error("Low probability of success")

# Run with: streamlit run app.py