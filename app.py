import pandas as pd
import streamlit as st
import joblib

# Load the trained model
model_filename = r"C:\Users\Admin\Downloads\RMIPRMODEL\corrosion prediction of steel\random_forest_regressor_model.pkl"
try:
    loaded_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Model file not found at {model_filename}. Please ensure the file exists.")
    st.stop()

# Streamlit app
def main():
    st.title("Corrosion Rate Prediction")
    st.write("Enter the details to predict the corrosion rate:")

    # Input fields for user-provided data
    material_options = ['Material_A', 'Material_B', 'Material_C']  # Update with actual material names
    condition_mapping = {
        "non-aerated": 0,
        "aerated": 1,
        "slight to moderate aerated": 0.5,
        "moderate to aerated": 0.5,
        "With 0.04 mol/L fluoride": 2,
        "max corrosion rate": 3
    }

    material = st.selectbox('Material', material_options)
    condition = st.selectbox('Condition', list(condition_mapping.keys()))
    concentration = st.number_input('Concentration (Vol %)', min_value=0.0, max_value=100.0, value=50.0)
    temperature_degC = st.number_input('Temperature (°C)', min_value=-100.0, max_value=1000.0, value=25.0)
    duration = st.number_input('Duration (days)', min_value=0.0, value=30.0)

    # Placeholder for additional features
    additional_features = {}
    for i in range(1, 16):  # Assuming 15 more features are required
        additional_features[f'Feature_{i}'] = st.number_input(f'Feature_{i}', value=0.0)

    # Prediction button
    if st.button('Predict'):
        # Map categorical values to numerical
        condition_value = condition_mapping[condition]
        material_encoded = material_options.index(material)

        # Create input data with all required features
        input_data = pd.DataFrame({
            'Material': [material_encoded],
            'Condition': [condition_value],
            'Concentration': [concentration],
            'Temperature_degC': [temperature_degC],
            'Duration': [duration],
            **{f'Feature_{i}': [additional_features[f'Feature_{i}']] for i in range(1, 16)}
        })

        # Ensure the number of features matches the model's expectation
        if input_data.shape[1] != 20:
            st.error(f"Input data has {input_data.shape[1]} features, but the model expects 20.")
            return

        # Predict using the loaded model
        prediction = loaded_model.predict(input_data)

        # Display the prediction
        st.success(f'Predicted Corrosion Rate: {prediction[0]:.2f} mm/year')

if __name__ == '__main__':
    main()
