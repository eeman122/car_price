import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
# Configuration
st.set_page_config(page_title='Car Price Prediction & Search', layout='wide')

# # For Hugging Face Spaces deployment
# BACKEND_SPACE_NAME = "car-price-app-backend"  # Your backend space name
# HF_USERNAME = "annus-lums"  # Your actual Hugging Face username
# BASE_URL = f"https://{HF_USERNAME}-{BACKEND_SPACE_NAME}.hf.space"

# # Alternatively, for local development (will override if running locally)
# import os
# if os.environ.get("RUNNING_LOCALLY"):
#     BASE_URL = "http://localhost:8000"  # Local backend


# BASE_URL = os.getenv('BACKEND_URL', 'https://car-price-2.onrender.com')

if os.path.exists('/.dockerenv') and not os.getenv('RUNNING_LOCALLY'):
    # Hugging Face production
    BASE_URL = f"https://{os.getenv('HF_USERNAME', 'annus-lums')}-car-price-app-backend.hf.space"
else:
    # Local development
    BASE_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

# Optional: Verify URL (check container logs)
print(f"Using backend at: {BASE_URL}")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# Helper functions
def display_analytics():
    st.subheader("📊 Car Market Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Imported vs Local Cars")
        try:
            data = requests.get(f"{BASE_URL}/analytics/imported").json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
                if not df.empty:
                    fig = px.pie(df, values='count', names='assembly_type', 
                                title='Distribution by Assembly Type',
                                hover_data=['avg_price'])
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df.set_index('assembly_type'))
                else:
                    st.warning("No data available for assembly types.")
            else:
                st.error("Unexpected data format received from API")
        except Exception as e:
            st.error(f"Error loading assembly analytics: {e}")
    
    with col2:
        st.markdown("### Transmission Types")
        try:
            data = requests.get(f"{BASE_URL}/analytics/transmission").json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
                if not df.empty:
                    fig = px.bar(df, x='transmission', y='count', 
                                title='Count by Transmission Type',
                                hover_data=['avg_price'])
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df.set_index('transmission'))
                else:
                    st.warning("No data available for transmission types.")
            else:
                st.error("Unexpected data format received from API")
        except Exception as e:
            st.error(f"Error loading transmission analytics: {e}")

# Pages
def main_page():
    st.title("🚗 Car Price Prediction & Search System")
    st.write("Welcome! Choose an option below to continue:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔮 Predict Car Price", use_container_width=True, 
                    help="Get a price prediction for your car"):
            st.session_state.page = "predict"
            st.rerun()
    with col2:
        if st.button("🔍 Search Car Listings", use_container_width=True,
                    help="Search existing car listings in our database"):
            st.session_state.page = "search"
            st.rerun()

    # Add a visual separator
    st.markdown("---")
    # Display analytics on main page
    display_analytics()
    
COLORS = ['White', 'Black', 'Silver', 'Grey', 'Red', 'Blue', 'Brown', 'Golden', 'Green', 'Other']
BODY_TYPES = ['Sedan', 'Hatchback', 'SUV', 'Van', 'Crossover', 'Wagon']
CITIES = ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Peshawar', 'Multan', 'Other']


COMMON_MAKES = ['Toyota', 'Honda', 'Suzuki', 'Hyundai', 'Daihatsu']
MAKE_MODEL_MAPPING = {
    'Toyota': ['Corolla', 'Fortuner', 'Hilux', 'Land', 'Passo', 'Prado', 'Vitz', 'Yaris'],
    'Honda': ['City', 'Civic', 'Vezel'],
    'Suzuki': ['Alto', 'Bolan', 'Cultus', 'Mehran', 'Swift', 'Wagon'],
    'Daihatsu': ['Cuore', 'Mira'],
    'KIA': ['Sportage']
}

def predict_page():
    st.title("🔮 Car Price Prediction")
    
    if st.button("← Back to Main"):
        st.session_state.page = "main"
        st.rerun()
    
    # Initialize form data in session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'make': 'Toyota',
            'model': 'Corolla',
            'year': 2015,
            'engine': 1500,
            'fuel': 'Petrol',
            'transmission': 'Automatic',
            'mileage': 50000,
            'assembly': 'Local'
        }
    
    # --- Display Inputs Row-wise ---
    # Make and Model
    st.subheader("Car Details")
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox(
            "Make",
            list(MAKE_MODEL_MAPPING.keys()),
            index=list(MAKE_MODEL_MAPPING.keys()).index(st.session_state.form_data['make']),
            key='make'
        )

    with col2:
        # Automatically adjust model list based on selected make
        available_models = MAKE_MODEL_MAPPING[make]

        # If the selected model is not in the new make, set it to the first one
        if st.session_state.form_data['model'] not in available_models:
            st.session_state.form_data['model'] = available_models[0]

        model = st.selectbox(
            "Model",
            available_models,
            index=available_models.index(st.session_state.form_data['model']),
            key='model'
        )

    # Year and Engine
    col3, col4 = st.columns(2)
    with col3:
        year = st.number_input("Year", min_value=1990, max_value=2025, 
                               value=st.session_state.form_data['year'])
    with col4:
        engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, 
                                 value=st.session_state.form_data['engine'])
    
    # Fuel and Transmission
    col5, col6 = st.columns(2)
    with col5:
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"],
                            index=["Petrol", "Diesel", "Hybrid"].index(st.session_state.form_data['fuel']))
    with col6:
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"],
                                    index=["Automatic", "Manual"].index(st.session_state.form_data['transmission']))
    
    # Mileage and Assembly
    col7, col8 = st.columns(2)
    with col7:
        mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, 
                                  value=st.session_state.form_data['mileage'])
    with col8:
        assembly = st.selectbox("Assembly", ["Local", "Imported"],
                                index=["Local", "Imported"].index(st.session_state.form_data['assembly']))

    # Submit Button
    if st.button("💰 Predict Price"):
        # Update session state
        st.session_state.form_data.update({
            'make': make,
            'model': model,
            'year': year,
            'engine': engine,
            'fuel': fuel,
            'transmission': transmission,
            'mileage': mileage,
            'assembly': assembly
        })
        
        try:
            response = requests.post(f'{BASE_URL}/predict', json=st.session_state.form_data)
            if response.status_code == 200:
                prediction = response.json().get('prediction', 'N/A')
                st.success(f'### Predicted Price: PKR {prediction:,.2f}')
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def search_page():
    st.title("🔍 Search Car Listings")
    
    if st.button("← Back to Main"):
        st.session_state.page = "main"
        st.rerun()
    
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            color = st.selectbox("Color", ["Any"] + ["White", "Black", "Silver", "Red", "Blue", "Other"])
            make = st.selectbox("Make", ["Any"] + ["Toyota", "Honda", "Suzuki", "Hyundai", "Kia", "Other"])
        
        with col2:
            min_price = st.number_input("Minimum Price (PKR)", min_value=0, value=0)
            max_price = st.number_input("Maximum Price (PKR)", min_value=0, value=10000000)
        
        if st.form_submit_button("Search"):
            params = {
                "color": color if color != "Any" else None,
                "make": make if make != "Any" else None,
                "min_price": min_price if min_price > 0 else None,
                "max_price": max_price if max_price > 0 else None
            }
            
            try:
                response = requests.get(f"{BASE_URL}/search", params=params)
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Display summary stats
                        st.subheader(f"Found {len(df)} listings")
                        st.write(f"Average price: PKR {df['price'].mean():,.2f}")
                        st.write(f"Price range: PKR {df['price'].min():,.2f} - PKR {df['price'].max():,.2f}")
                        
                        # Show the data
                        st.dataframe(df)
                    else:
                        st.warning("No matching cars found. Try different search criteria.")
                else:
                    st.error(f"Search failed with status {response.status_code}")
            except Exception as e:
                st.error(f"Error searching listings: {e}")

# Page routing
if st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'predict':
    predict_page()
elif st.session_state.page == 'search':
    search_page()
