import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Title of the app
st.title("Gradient Boost Model Application")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["Regression: Car Selling Price Prediction", "Classification: Predict Holiday Package"])

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@st.dialog("Predict car selling price")
def reg_predict(car_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, 
                    mileage, engine, max_power, seats):
    # Display input parameters
    st.write("Input Parameters:")
    st.write(f"Car Name: {car_name}")
    st.write(f"Vehicle Age: {vehicle_age} years")
    st.write(f"Kilometers Driven: {km_driven} km")
    st.write(f"Seller Type: {seller_type}")
    st.write(f"Fuel Type: {fuel_type}")
    st.write(f"Transmission Type: {transmission_type}")
    st.write(f"Mileage: {mileage} kmpl")
    st.write(f"Engine: {engine} cc")
    st.write(f"Max Power: {max_power} bhp")
    st.write(f"Seats: {seats}")

    # Load the trained model and preprocessor
    model = load_pickle('model/reg/model.pkl')
    label_encoder = load_pickle('model/reg/label_encoder.pkl')
    preprocessor = load_pickle('model/reg/preprocessor.pkl')

    # Encode the car name
    car_name_encoded = label_encoder.transform([car_name])[0]

    # Prepare the input data
    input_data = np.array([[car_name_encoded, vehicle_age, km_driven, seller_type,
                            fuel_type, transmission_type, mileage, engine, max_power, seats]])

    # Preprocess the input data
    input_df = pd.DataFrame(input_data, columns=['car_name', 'vehicle_age', 'km_driven', 
                                                'seller_type', 'fuel_type', 'transmission_type', 
                                                'mileage', 'engine', 'max_power', 'seats'])
    processed_input = preprocessor.transform(input_df)

    # Predict the selling price
    prediction = model.predict(processed_input)
    st.write(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")

@st.dialog("Predict holiday package purchase")
def clf_predict(CustomerID, Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender, NumberOfFollowups, ProductPitched, 
                PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, Designation, 
                MonthlyIncome, NumberOfPersonVisiting, NumberOfChildrenVisiting):
    # Display input parameters
    st.write("Input Parameters:")
    st.write(f"Customer ID: {CustomerID}")
    st.write(f"Age: {Age} years")
    st.write(f"Type of Contact: {TypeofContact}")
    st.write(f"City Tier: {CityTier}")
    st.write(f"Duration of Sales Pitch: {DurationOfPitch} minutes")
    st.write(f"Occupation: {Occupation}")
    st.write(f"Gender: {Gender}")
    st.write(f"Number of Follow-ups: {NumberOfFollowups}")
    st.write(f"Product Pitched: {ProductPitched}")
    st.write(f"Preferred Property Star Rating: {PreferredPropertyStar}")
    st.write(f"Marital Status: {MaritalStatus}")
    st.write(f"Number of Trips Taken: {NumberOfTrips}")
    st.write(f"Passport: {Passport}")
    st.write(f"Pitch Satisfaction Score: {PitchSatisfactionScore}")
    st.write(f"Own Car: {OwnCar}")
    st.write(f"Designation: {Designation}")
    st.write(f"Monthly Income: {MonthlyIncome} thousands")
    st.write(f"Number of Persons Visiting: {NumberOfPersonVisiting}")
    st.write(f"Number of Children Visiting: {NumberOfChildrenVisiting}")

    # Load the trained model and preprocessor
    model = load_pickle('model/clf/new_model.pkl')
    preprocessor = load_pickle('model/clf/new_preprocessor.pkl')

    # Prepare the input data by merging visit counts
    TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting
    Passport = 1 if Passport.strip().lower() == 'yes' else 0
    OwnCar = 1 if OwnCar.strip().lower() == 'yes' else 0
    # Create input data
    input_data = np.array([[Age, TypeofContact, CityTier, DurationOfPitch, 
                            Occupation, Gender, NumberOfFollowups, ProductPitched, 
                            PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, 
                            PitchSatisfactionScore, OwnCar, Designation, MonthlyIncome, 
                            TotalVisiting]])

    # Convert input data to DataFrame for preprocessing
    input_df = pd.DataFrame(input_data, columns=[
        'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
        'Occupation', 'Gender', 'NumberOfFollowups', 'ProductPitched',
        'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
        'PitchSatisfactionScore', 'OwnCar', 'Designation', 'MonthlyIncome', 'TotalVisiting'
    ])

    # Preprocess the input data
    processed_input = preprocessor.transform(input_df)

    # Predict the class
    prediction = model.predict(processed_input)

    st.write(f"Predicted Purchase Decision: **{'Yes' if prediction[0] == 1 else 'No'}**")

if app_mode == "Regression: Car Selling Price Prediction":
    # Regression Task Inputs
    st.header("Predict Car Selling Price")

    # Feature Inputs for Regression Model
    car_name = st.selectbox('Car Name', [
        'Hyundai i20', 'Maruti Swift Dzire', 'Maruti Swift', 'Maruti Alto', 'Honda City',
        'Maruti Wagon R', 'Hyundai Grand', 'Toyota Innova', 'Hyundai Verna', 'Hyundai i10',
        'Ford Ecosport', 'Volkswagen Polo', 'Maruti Baleno', 'Honda Amaze', 'Maruti Ciaz',
        'Maruti Ertiga', 'Hyundai Creta', 'Mahindra XUV500', 'Renault KWID', 'Maruti Vitara',
        'Mahindra Scorpio', 'Ford Figo', 'Volkswagen Vento', 'Maruti Celerio', 'Renault Duster',
        'Mahindra Bolero', 'Toyota Fortuner', 'Skoda Rapid', 'Honda Jazz', 'BMW 3',
        'Tata Tiago', 'Hyundai Santro', 'Others', 'Mercedes-Benz E-Class', 'Maruti Eeco',
        'Mercedes-Benz C-Class', 'BMW 5', 'Honda WR-V', 'Tata Safari', 'Audi A4',
        'Skoda Superb', 'Tata Nexon', 'Datsun GO', 'Datsun RediGO', 'Maruti Ignis',
        'Mahindra KUV', 'Ford Aspire', 'Audi A6', 'Mahindra Thar', 'Honda Civic',
        'Skoda Octavia', 'Hyundai Venue', 'BMW X1', 'Jaguar XF', 'Land Rover Rover',
        'Hyundai Elantra', 'Ford Endeavour', 'Tata Hexa', 'Jeep Compass', 'Mercedes-Benz S-Class',
        'Tata Tigor', 'BMW 7', 'Toyota Camry', 'Mercedes-Benz GL-Class', 'Ford Freestyle',
        'Honda CR-V', 'Kia Seltos', 'Mahindra KUV100', 'BMW X5', 'Mahindra Marazzo',
        'BMW X3', 'Audi Q7', 'Tata Harrier', 'MG Hector', 'BMW 6',
        'Mini Cooper', 'Maruti Dzire VXI', 'Toyota Yaris', 'Porsche Cayenne', 'Mahindra XUV300',
        'Maruti S-Presso', 'Renault Triber', 'Mercedes-Benz GLS', 'Hyundai Tucson', 'Datsun redi-GO'
    ])
    vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=50, value=5, step=1)
    km_driven = st.number_input('Kilometers Driven', min_value=0, value=50000, step=1000)
    seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
    mileage = st.number_input('Mileage (kmpl)', min_value=0.0, value=15.0, step=0.1)
    engine = st.number_input('Engine (cc)', min_value=500, max_value=5000, value=1500, step=10)
    max_power = st.number_input('Max Power (bhp)', min_value=20.0, max_value=500.0, value=100.0, step=1.0)
    seats = st.number_input('Seats', min_value=2, max_value=10, value=5, step=1)

    # Check if any input is missing
    inputs_filled = all([
        car_name, vehicle_age, km_driven, seller_type, fuel_type,
        transmission_type, mileage, engine, max_power, seats
    ])

    # Button to submit the form
    if st.button('Predict Selling Price', disabled=not inputs_filled):
        reg_predict(car_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, 
                    mileage, engine, max_power, seats)

    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")

elif app_mode == "Classification: Predict Holiday Package":
    # Classification Task Inputs
    st.header("Predict Holiday Package Purchase using AdaBoost")

    # Feature Inputs for Classification Model
    CustomerID = st.text_input('Customer ID', value='12345')
    Age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
    TypeofContact = st.selectbox('Type of Contact', ['Company Invited', 'Self Initiated'])
    CityTier = st.number_input('City Tier (1/2/3)', min_value=1, max_value=3, value=1, step=1)
    DurationOfPitch = st.number_input('Duration of Sales Pitch (in minutes)', min_value=0, value=10, step=1)
    Occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    NumberOfFollowups = st.number_input('Number of Follow-ups', min_value=0, value=1, step=1)
    ProductPitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Premium', 'Super Deluxe', 'King'])
    PreferredPropertyStar = st.number_input('Preferred Property Star', min_value=1, max_value=5, value=3, step=1)
    MaritalStatus = st.selectbox('Marital Status', ['Married', 'Unmarried', 'Divorced'])
    NumberOfTrips = st.number_input('Number of Trips', min_value=0, value=1, step=1)
    Passport = st.selectbox('Passport Ownership', ['Yes', 'No'])
    PitchSatisfactionScore = st.number_input('Pitch Satisfaction Score', min_value=1, max_value=5, value=3, step=1)
    OwnCar = st.selectbox('Own Car', ['Yes', 'No'])
    Designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    MonthlyIncome = st.number_input('Monthly Income', min_value=0, value=50000, step=1000)
    NumberOfPersonVisiting = st.number_input('Number of Persons Visiting', min_value=0, value=1, step=1)
    NumberOfChildrenVisiting = st.number_input('Number of Children Visiting', min_value=0, value=0, step=1)

    # Check if all inputs are filled
    inputs_filled = all([
        CustomerID,
        Age is not None,
        TypeofContact,
        CityTier is not None,
        DurationOfPitch is not None,
        Occupation,
        Gender,
        NumberOfFollowups is not None,
        ProductPitched,
        PreferredPropertyStar is not None,
        MaritalStatus,
        NumberOfTrips is not None,
        Passport,
        PitchSatisfactionScore is not None,
        OwnCar,
        Designation,
        MonthlyIncome is not None,
        NumberOfPersonVisiting is not None,
        NumberOfChildrenVisiting is not None
    ])

    # Button to submit the form
    if st.button('Predict Holiday Package Purchase', disabled=not inputs_filled):
        clf_predict(CustomerID, Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender, NumberOfFollowups, ProductPitched, 
                    PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, Designation, 
                    MonthlyIncome, NumberOfPersonVisiting, NumberOfChildrenVisiting)

    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")
