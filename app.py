import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load the model (or train it here if needed)
# lr_model = joblib.load('house_price_model.pkl')

# For demo purpose: define a dummy Linear Regression model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data and train model (quickly)
data = pd.read_csv("housing.csv")  # make sure you have housing.csv in the same folder

# Encode ocean_proximity
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['ocean_proximity'] = le.fit_transform(data['ocean_proximity'])

# Handle missing values
data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ---------- Streamlit App ----------
st.title("🏠 California House Price Predictor")

st.sidebar.header("Input Features")

longitude = st.sidebar.number_input("Longitude", -125.0, -114.0, step=0.01)
latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, step=0.01)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 52, 30)
total_rooms = st.sidebar.slider("Total Rooms", 100, 10000, 2000)
total_bedrooms = st.sidebar.slider("Total Bedrooms", 1, 3000, 500)
population = st.sidebar.slider("Population", 1, 5000, 1000)
households = st.sidebar.slider("Households", 1, 3000, 500)
median_income = st.sidebar.slider("Median Income", 0.5, 15.0, 3.0)

ocean_dict = {
    'INLAND': 0,
    '<1H OCEAN': 1,
    'NEAR BAY': 2,
    'NEAR OCEAN': 3,
    'ISLAND': 4
}
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", list(ocean_dict.keys()))
ocean_encoded = ocean_dict[ocean_proximity]

# Create input DataFrame
input_df = pd.DataFrame([{
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_encoded
}])

# Predict
if st.button("Predict House Price"):
    prediction = lr_model.predict(input_df)[0]
    st.success(f"🏡 Predicted House Value: ${prediction:,.2f}")
