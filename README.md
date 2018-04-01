Required libraries:

OpenCV (pip install opencv-python).

scikit-multilearn (pip install scikit-multilearn)

Code exexecution:

First put all files in one directory and uncompress the .zip files.

Then run 'clean_data.py' on Spark to produce a single cleaned data file for ML. 

Run on command line: spark-submit data_clean.py yvr-weather(sample input)

Output a .csv file named 'weather_pd.csv'.

Lastly you can run either 'weather_predict_by_conditions.py' or 'weather_PredictByImages.py' in any order.

Run on command line with no additianal sys argument since it takes 'weather_pd.csv' and/or 'katkam-scaled'as the only input. (e.g. python3 weather_PredictByImages.py)

Output will be two .csv files 'weather_predicton_by_conditions.csv' and 'weather_prediction_by_images.csv', in which the last column titled 'predicted_weather' is the prediction.

(You should wait until the prompt print out a float as the model score because it might take quite some time to load the images for 'weather_PredictByImages.py'.
