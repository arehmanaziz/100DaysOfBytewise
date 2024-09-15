Air Quality and Weather Prediction GUI

This project includes a graphical user interface (GUI) for predicting weather conditions and air quality index based on user inputs. The models are pre-trained and saved. Simply run the main.py file to use the GUI.

Project Files:
main.py: Main script to run the GUI.
weather_model.pkl: Trained model for predicting weather conditions.
air_quality_model.pkl: Trained model for predicting air quality index.

Requirements:
1. tkinter (usually pre-installed with Python)
2. pandas
3. joblib
4. scikit-learn

Install the required packages using pip:
pip install pandas joblib scikit-learn

How to Run:
1. Clone the repository or download the files.
2. Navigate to the project directory.
3. Run the GUI with the command 'python main.py'

Using the GUI:

Select Prediction Type: Choose "Predict Weather" or "Predict Air Quality".
Enter Inputs: Provide the required information based on your selection.
Get Prediction: Click "Predict" to see the results.
Model Details:

Weather Model: Predicts weather conditions based on temperature, humidity, precipitation, and wind speed.
Air Quality Model: Predicts CO concentration based on sensor readings and environmental factors.
