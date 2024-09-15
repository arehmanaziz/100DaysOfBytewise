import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd

# Load the models
weather_model = joblib.load('weather_model.pkl')
air_quality_model = joblib.load('air_quality_model.pkl')

# Function to handle weather prediction
def predict_weather():
    temperature = float(temperature_entry.get())
    humidity = float(humidity_entry.get())
    precipitation = float(precipitation_entry.get())
    wind_speed = float(wind_speed_entry.get())
    
    input_data = pd.DataFrame([[temperature, humidity, precipitation, wind_speed]],
                              columns=['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh'])
    prediction = weather_model.predict(input_data)[0]
    
    result_frame = tk.Frame(root)
    result_frame.pack(fill='both', expand=True)
    
    result_label = tk.Label(result_frame, text=f"Weather Condition: {prediction}")
    result_label.pack(pady=20)
    
    # Hide the input frame
    weather_frame.pack_forget()

def predict_air_quality():
    # Retrieve input values
    pt08_s1_co = float(pt08_s1_co_entry.get())
    nmhc_gt = float(nmhc_gt_entry.get())
    c6h6_gt = float(c6h6_gt_entry.get())
    pt08_s2_nm = float(pt08_s2_nm_entry.get())
    nox_gt = float(nox_gt_entry.get())
    pt08_s3_nox = float(pt08_s3_nox_entry.get())
    no2_gt = float(no2_gt_entry.get())
    pt08_s4_no2 = float(pt08_s4_no2_entry.get())
    pt08_s5_o3 = float(pt08_s5_o3_entry.get())
    temperature = float(temperature_entry_air_quality.get())
    humidity = float(humidity_entry_air_quality.get())
    ah = float(ah_entry_air_quality.get())
    
    input_data = pd.DataFrame([[pt08_s1_co, nmhc_gt, c6h6_gt, pt08_s2_nm, nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3, temperature, humidity, ah]],
                              columns=['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])
    prediction = air_quality_model.predict(input_data)[0]
    
    result_frame = tk.Frame(root)
    result_frame.pack(fill='both', expand=True)
    
    result_label = tk.Label(result_frame, text=f"CO Concentration: {prediction}")
    result_label.pack(pady=20)
    
    # Hide the input frame
    air_quality_frame.pack_forget()

# Function to show weather input frame
def show_weather_inputs():
    main_frame.pack_forget()
    
    global weather_frame
    weather_frame = tk.Frame(root)
    weather_frame.pack(fill='both', expand=True)
    
    tk.Label(weather_frame, text="Temperature (C):").pack(pady=5)
    global temperature_entry
    temperature_entry = tk.Entry(weather_frame)
    temperature_entry.pack(pady=5)
    
    tk.Label(weather_frame, text="Humidity (%):").pack(pady=5)
    global humidity_entry
    humidity_entry = tk.Entry(weather_frame)
    humidity_entry.pack(pady=5)
    
    tk.Label(weather_frame, text="Precipitation (mm):").pack(pady=5)
    global precipitation_entry
    precipitation_entry = tk.Entry(weather_frame)
    precipitation_entry.pack(pady=5)
    
    tk.Label(weather_frame, text="Wind Speed (km/h):").pack(pady=5)
    global wind_speed_entry
    wind_speed_entry = tk.Entry(weather_frame)
    wind_speed_entry.pack(pady=5)
    
    predict_button = tk.Button(weather_frame, text="Predict", command=predict_weather)
    predict_button.pack(pady=20)

# Function to show air quality input frame
def show_air_quality_inputs():
    main_frame.pack_forget()
    
    global air_quality_frame
    air_quality_frame = tk.Frame(root)
    air_quality_frame.pack(fill='both', expand=True)
    
    tk.Label(air_quality_frame, text="PT08.S1(CO):").pack(pady=5)
    global pt08_s1_co_entry
    pt08_s1_co_entry = tk.Entry(air_quality_frame)
    pt08_s1_co_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="NMHC(GT):").pack(pady=5)
    global nmhc_gt_entry
    nmhc_gt_entry = tk.Entry(air_quality_frame)
    nmhc_gt_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="C6H6(GT):").pack(pady=5)
    global c6h6_gt_entry
    c6h6_gt_entry = tk.Entry(air_quality_frame)
    c6h6_gt_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="PT08.S2(NMHC):").pack(pady=5)
    global pt08_s2_nm_entry
    pt08_s2_nm_entry = tk.Entry(air_quality_frame)
    pt08_s2_nm_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="NOx(GT):").pack(pady=5)
    global nox_gt_entry
    nox_gt_entry = tk.Entry(air_quality_frame)
    nox_gt_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="PT08.S3(NOx):").pack(pady=5)
    global pt08_s3_nox_entry
    pt08_s3_nox_entry = tk.Entry(air_quality_frame)
    pt08_s3_nox_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="NO2(GT):").pack(pady=5)
    global no2_gt_entry
    no2_gt_entry = tk.Entry(air_quality_frame)
    no2_gt_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="PT08.S4(NO2):").pack(pady=5)
    global pt08_s4_no2_entry
    pt08_s4_no2_entry = tk.Entry(air_quality_frame)
    pt08_s4_no2_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="PT08.S5(O3):").pack(pady=5)
    global pt08_s5_o3_entry
    pt08_s5_o3_entry = tk.Entry(air_quality_frame)
    pt08_s5_o3_entry.pack(pady=5)
    
    tk.Label(air_quality_frame, text="Temperature (C):").pack(pady=5)
    global temperature_entry_air_quality
    temperature_entry_air_quality = tk.Entry(air_quality_frame)
    temperature_entry_air_quality.pack(pady=5)
    
    tk.Label(air_quality_frame, text="Humidity (%):").pack(pady=5)
    global humidity_entry_air_quality
    humidity_entry_air_quality = tk.Entry(air_quality_frame)
    humidity_entry_air_quality.pack(pady=5)
    
    tk.Label(air_quality_frame, text="AH:").pack(pady=5)
    global ah_entry_air_quality
    ah_entry_air_quality = tk.Entry(air_quality_frame)
    ah_entry_air_quality.pack(pady=5)
    
    predict_button = tk.Button(air_quality_frame, text="Predict", command=predict_air_quality)
    predict_button.pack(pady=20)

# Create the main window
root = tk.Tk()
root.title("Predictor")

# Create the main frame with options
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True)

tk.Label(main_frame, text="Select Prediction Type").pack(pady=20)

weather_button = tk.Button(main_frame, text="Predict Weather", command=show_weather_inputs)
weather_button.pack(pady=10)

air_quality_button = tk.Button(main_frame, text="Predict Air Quality", command=show_air_quality_inputs)
air_quality_button.pack(pady=10)

root.mainloop()
