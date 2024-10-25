from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = r'C:\Users\use\Downloads\cps\crop_health_detection_2\Crop-Disease-Detection\plant_disease_model.h5'

# Disease information database
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'causes': [
            'Fungus Venturia inaequalis',
            'Infected plant debris',
            'Moist, humid weather conditions',
            'Poor air circulation'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Remove infected leaves and debris',
            'Prune for better air circulation',
            'Avoid overhead irrigation'
        ],
        'pesticides': [
            {'name': 'Captan', 'application': 'Spray before flowering'},
            {'name': 'Mancozeb', 'application': 'Apply every 7-14 days during wet conditions'},
            {'name': 'Propiconazole', 'application': 'Use as a preventive treatment'}
        ],
        'weather_conditions': {
            'temperature': '60-70°F (15-21°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during spring',
            'seasonality': 'Spring through early summer'
        },
        'occurrence_pattern': 'Common in humid regions, particularly during wet springs',
        'severity_level': 'Moderate to High'
    },
    
    'Apple___Black_rot': {
        'causes': [
            'Fungus Diplocarpon mali',
            'Infected plant debris',
            'Wet and humid conditions',
            'Poor air circulation'
        ],
        'prevention': [
            'Remove fallen fruit and infected leaves',
            'Use resistant apple varieties',
            'Improve air circulation',
            'Avoid overhead irrigation'
        ],
        'pesticides': [
            {'name': 'Mancozeb', 'application': 'Use preventatively during wet weather'},
            {'name': 'Myclobutanil', 'application': 'Apply at early bloom'},
            {'name': 'Captan', 'application': 'Use during critical growth stages'}
        ],
        'weather_conditions': {
            'temperature': '60-75°F (15-24°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during spring',
            'seasonality': 'Spring through fall'
        },
        'occurrence_pattern': 'More severe in wet seasons, common in commercial orchards',
        'severity_level': 'Moderate to High'
    },
    
    'Apple___Cedar_apple_rust': {
        'causes': [
            'Fungus Gymnosporangium juniperi-virginianae',
            'Infection requires both apple and cedar trees',
            'Moisture and humidity',
            'Wind dispersal of spores'
        ],
        'prevention': [
            'Plant resistant apple varieties',
            'Remove cedar trees near apple orchards',
            'Improve air circulation',
            'Avoid overhead irrigation'
        ],
        'pesticides': [
            {'name': 'Chlorothalonil', 'application': 'Spray at bud break'},
            {'name': 'Mancozeb', 'application': 'Use every 7-14 days during spring'},
            {'name': 'Propiconazole', 'application': 'Use for control in early season'}
        ],
        'weather_conditions': {
            'temperature': '60-75°F (15-24°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rains during spring',
            'seasonality': 'Spring through summer'
        },
        'occurrence_pattern': 'Common in areas with both cedar and apple trees',
        'severity_level': 'Moderate to High'
    },
    
    'Apple___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '60-75°F (15-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Blueberry___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Cherry_(including_sour)___Powdery_mildew': {
        'causes': [
            'Fungus Podosphaera clandestina',
            'High humidity',
            'Poor air circulation',
            'Warm temperatures'
        ],
        'prevention': [
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove infected plant debris',
            'Use resistant cherry varieties'
        ],
        'pesticides': [
            {'name': 'Sulfur', 'application': 'Apply at first sign of infection'},
            {'name': 'Myclobutanil', 'application': 'Use preventatively every 10-14 days'},
            {'name': 'Trifloxystrobin', 'application': 'Use as needed'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Regular moisture required',
            'seasonality': 'Spring through summer'
        },
        'occurrence_pattern': 'Common in humid areas, particularly during warm weather',
        'severity_level': 'Moderate to High'
    },
    
    'Cherry_(including_sour)___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'causes': [
            'Fungus Cercospora zeae-maydis',
            'Warm, humid conditions',
            'Infected plant debris',
            'Poor air circulation'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected debris',
            'Improve air circulation',
            'Use resistant maize varieties'
        ],
        'pesticides': [
            {'name': 'Azoxystrobin', 'application': 'Use preventatively in susceptible fields'},
            {'name': 'Chlorothalonil', 'application': 'Spray every 10-14 days as needed'},
            {'name': 'Mancozeb', 'application': 'Use according to label instructions'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during summer',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in humid regions; can be severe during wet seasons',
        'severity_level': 'Moderate to High'
    },
    
    'Corn_(maize)___Common_rust_': {
        'causes': [
            'Fungus Puccinia sorghi',
            'Windborne spores',
            'High humidity',
            'Warm temperatures'
        ],
        'prevention': [
            'Plant resistant maize varieties',
            'Practice crop rotation',
            'Avoid overhead irrigation',
            'Remove volunteer maize plants'
        ],
        'pesticides': [
            {'name': 'Azoxystrobin', 'application': 'Use as a preventive spray'},
            {'name': 'Chlorothalonil', 'application': 'Apply during early infection stages'},
            {'name': 'Triazole fungicides', 'application': 'Use when rust is detected'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during summer',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in many regions; severity depends on weather',
        'severity_level': 'Moderate to High'
    },
    
    'Corn_(maize)___Northern_Leaf_Blight': {
        'causes': [
            'Fungus Exserohilum turcicum',
            'Infected plant debris',
            'Warm, wet conditions',
            'Poor air circulation'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected debris',
            'Improve air circulation',
            'Use resistant maize varieties'
        ],
        'pesticides': [
            {'name': 'Azoxystrobin', 'application': 'Use as needed during disease outbreaks'},
            {'name': 'Mancozeb', 'application': 'Use preventatively as needed'},
            {'name': 'Propiconazole', 'application': 'Apply at early signs of infection'}
        ],
        'weather_conditions': {
            'temperature': '70-80°F (21-27°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during summer',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in humid areas; severity varies by year',
        'severity_level': 'Moderate to High'
    },
    
    'Corn_(maize)___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Grape___Black_rot': {
        'causes': [
            'Fungus Guignardia bidwellii',
            'Warm, humid conditions',
            'Infected plant debris',
            'Poor air circulation'
        ],
        'prevention': [
            'Remove infected leaves and debris',
            'Use resistant grape varieties',
            'Improve air circulation in vineyards',
            'Avoid overhead irrigation'
        ],
        'pesticides': [
            {'name': 'Mancozeb', 'application': 'Use preventatively during wet conditions'},
            {'name': 'Myclobutanil', 'application': 'Spray every 7-10 days if needed'},
            {'name': 'Copper-based fungicides', 'application': 'Apply at bud break'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during summer',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in humid regions; severity increases with wet weather',
        'severity_level': 'Moderate to High'
    },
    
    'Grape___Esca_(Black_Measles)': {
        'causes': [
            'Fungus Phaeoacremonium spp. and other fungi',
            'Infection through pruning wounds',
            'Warm, humid weather',
            'Aging vines'
        ],
        'prevention': [
            'Prune in dry weather',
            'Use resistant grape varieties',
            'Improve vineyard sanitation',
            'Monitor vine health'
        ],
        'pesticides': [
            {'name': 'Copper-based fungicides', 'application': 'Use preventatively as needed'},
            {'name': 'Chlorothalonil', 'application': 'Apply at first sign of infection'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in vineyards, especially older vines',
        'severity_level': 'High'
    },
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'causes': [
            'Fungus Isariopsis spp.',
            'Warm, humid conditions',
            'Infected plant debris',
            'Poor air circulation'
        ],
        'prevention': [
            'Remove infected leaves and debris',
            'Practice crop rotation',
            'Improve air circulation in vineyards',
            'Use resistant grape varieties'
        ],
        'pesticides': [
            {'name': 'Mancozeb', 'application': 'Use preventatively during wet weather'},
            {'name': 'Chlorothalonil', 'application': 'Spray every 10-14 days if needed'},
            {'name': 'Copper-based fungicides', 'application': 'Use as per label instructions'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Late spring to early fall'
        },
        'occurrence_pattern': 'Common in humid regions; can be severe in wet years',
        'severity_level': 'Moderate to High'
    },
    
    'Grape___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Orange___Huanglongbing_(Citrus_greening)': {
        'causes': [
            'Bacteria Candidatus Liberibacter spp.',
            'Spread by Asian citrus psyllid',
            'Nutrient deficiencies',
            'Stress from drought or flooding'
        ],
        'prevention': [
            'Control psyllid populations',
            'Remove infected trees',
            'Maintain healthy soil',
            'Implement good cultural practices'
        ],
        'pesticides': [
            {'name': 'Imidacloprid', 'application': 'Use for psyllid control'},
            {'name': 'Thiamethoxam', 'application': 'Use as needed'},
            {'name': 'Bifenthrin', 'application': 'Use for psyllid control'}
        ],
        'weather_conditions': {
            'temperature': '65-85°F (18-29°C)',
            'humidity': 'Variable',
            'rainfall': 'Depends on region',
            'seasonality': 'Year-round threat'
        },
        'occurrence_pattern': 'Endemic in citrus-growing regions; severe impact on production',
        'severity_level': 'Very High'
    },
    
    'Peach___Bacterial_spot': {
        'causes': [
            'Bacteria Xanthomonas arboricola pv. pruni',
            'Wet weather conditions',
            'Infected plant debris',
            'Poor air circulation'
        ],
        'prevention': [
            'Remove infected plant debris',
            'Practice crop rotation',
            'Improve air circulation',
            'Use resistant peach varieties'
        ],
        'pesticides': [
            {'name': 'Copper-based bactericides', 'application': 'Spray preventatively before bloom'},
            {'name': 'Streptomycin', 'application': 'Use as needed for severe infections'}
        ],
        'weather_conditions': {
            'temperature': '60-80°F (15-27°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Spring through summer'
        },
        'occurrence_pattern': 'Common in humid regions; can be severe in wet years',
        'severity_level': 'Moderate to High'
    },
    
    'Peach___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Pepper,_bell___Bacterial_spot': {
        'causes': [
            'Bacteria Xanthomonas campestris pv. vesicatoria',
            'Wet weather conditions',
            'Infected plant debris',
            'Insect damage'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected plant debris',
            'Improve air circulation',
            'Use resistant pepper varieties'
        ],
        'pesticides': [
            {'name': 'Copper-based bactericides', 'application': 'Use preventatively before bloom'},
            {'name': 'Streptomycin', 'application': 'Use as needed for severe infections'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Spring through summer'
        },
        'occurrence_pattern': 'Common in humid regions; severity increases with wet weather',
        'severity_level': 'Moderate to High'
    },
    
    'Pepper,_bell___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Potato___Early_blight': {
        'causes': [
            'Fungus Alternaria solani',
            'Infected plant debris',
            'Warm, humid conditions',
            'Poor air circulation'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected plant debris',
            'Improve air circulation',
            'Use resistant potato varieties'
        ],
        'pesticides': [
            {'name': 'Chlorothalonil', 'application': 'Use preventatively during growing season'},
            {'name': 'Mancozeb', 'application': 'Apply every 7-10 days as needed'},
            {'name': 'Azoxystrobin', 'application': 'Use when conditions are favorable for blight'}
        ],
        'weather_conditions': {
            'temperature': '70-80°F (21-27°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Spring through fall'
        },
        'occurrence_pattern': 'Common in humid areas; severity increases with wet weather',
        'severity_level': 'Moderate to High'
    },
    
    'Potato___Late_blight': {
        'causes': [
            'Fungus Phytophthora infestans',
            'Infected plant debris',
            'Cool, moist conditions',
            'High humidity'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected plant debris',
            'Improve air circulation',
            'Use resistant potato varieties'
        ],
        'pesticides': [
            {'name': 'Metalaxyl', 'application': 'Use preventatively'},
            {'name': 'Chlorothalonil', 'application': 'Apply every 7-10 days as needed'},
            {'name': 'Mancozeb', 'application': 'Use as needed when conditions are favorable'}
        ],
        'weather_conditions': {
            'temperature': '60-70°F (15-21°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Spring through fall'
        },
        'occurrence_pattern': 'Common in humid and cool areas; can devastate crops',
        'severity_level': 'Very High'
    },
    
    'Potato___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Raspberry___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Soybean___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Squash___Powdery_mildew': {
        'causes': [
            'Fungus Podosphaera xanthii',
            'High humidity',
            'Poor air circulation',
            'Warm temperatures'
        ],
        'prevention': [
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove infected plant debris',
            'Use resistant squash varieties'
        ],
        'pesticides': [
            {'name': 'Sulfur', 'application': 'Apply at first sign of infection'},
            {'name': 'Myclobutanil', 'application': 'Use preventatively every 10-14 days'},
            {'name': 'Trifloxystrobin', 'application': 'Use as needed'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'High',
            'rainfall': 'Regular moisture required',
            'seasonality': 'Spring through summer'
        },
        'occurrence_pattern': 'Common in humid areas; particularly severe in late summer',
        'severity_level': 'Moderate to High'
    },
    
    'Strawberry___Leaf_scorch': {
        'causes': [
            'Environmental stress',
            'Excessive heat',
            'Drought conditions',
            'Fungal infections'
        ],
        'prevention': [
            'Provide adequate water',
            'Mulch to retain moisture',
            'Avoid planting in hot, dry areas',
            'Use resistant strawberry varieties'
        ],
        'pesticides': [
            {'name': 'Fungicides for associated fungal infections', 'application': 'Use as needed'}
        ],
        'weather_conditions': {
            'temperature': '70-80°F (21-27°C)',
            'humidity': 'Low',
            'rainfall': 'Infrequent',
            'seasonality': 'Summer through early fall'
        },
        'occurrence_pattern': 'Common during hot, dry summers; severity varies',
        'severity_level': 'Moderate'
    },
    
    'Strawberry___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
    
    'Tomato___Blossom_end_rot': {
        'causes': [
            'Calcium deficiency',
            'Irregular watering practices',
            'Poor soil drainage',
            'Excessive nitrogen'
        ],
        'prevention': [
            'Maintain consistent watering schedule',
            'Ensure adequate calcium in soil',
            'Improve drainage',
            'Avoid over-fertilization'
        ],
        'pesticides': [
            {'name': 'None required for nutrient deficiency', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '70-85°F (21-29°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Summer through fall'
        },
        'occurrence_pattern': 'Common in hot, dry conditions; severity varies',
        'severity_level': 'Moderate'
    },
    
    'Tomato___Late_blight': {
        'causes': [
            'Fungus Phytophthora infestans',
            'Infected plant debris',
            'Cool, moist conditions',
            'High humidity'
        ],
        'prevention': [
            'Practice crop rotation',
            'Remove infected plant debris',
            'Improve air circulation',
            'Use resistant tomato varieties'
        ],
        'pesticides': [
            {'name': 'Metalaxyl', 'application': 'Use preventatively'},
            {'name': 'Chlorothalonil', 'application': 'Apply every 7-10 days as needed'},
            {'name': 'Mancozeb', 'application': 'Use as needed when conditions are favorable'}
        ],
        'weather_conditions': {
            'temperature': '60-70°F (15-21°C)',
            'humidity': 'High',
            'rainfall': 'Frequent rain during growing season',
            'seasonality': 'Spring through fall'
        },
        'occurrence_pattern': 'Common in humid and cool areas; can devastate crops',
        'severity_level': 'Very High'
    },
    
    'Tomato___healthy': {
        'causes': [
            'Healthy plants with no visible diseases',
            'Proper care and maintenance',
            'Good soil health',
            'Appropriate watering practices'
        ],
        'prevention': [
            'Regular monitoring for pests',
            'Maintain proper spacing between plants',
            'Implement crop rotation',
            'Ensure good drainage'
        ],
        'pesticides': [
            {'name': 'None required for healthy plants', 'application': 'N/A'}
        ],
        'weather_conditions': {
            'temperature': '65-75°F (18-24°C)',
            'humidity': 'Moderate',
            'rainfall': 'Adequate but not excessive',
            'seasonality': 'Year-round care needed'
        },
        'occurrence_pattern': 'Consistent health with proper care',
        'severity_level': 'N/A'
    },
}


# Load your trained model
print(" ** Model Loading **")
try:
    model = load_model(MODEL_PATH)
    print(" ** Model Loaded **")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

def get_disease_info(disease_key):
    """Get detailed information about a disease"""
    print("-----disease_key---------------------",disease_key)
    if disease_key in DISEASE_INFO:
        return DISEASE_INFO[disease_key]
    else:
        return {
            "causes": ["Information not available"],
            "prevention": ["Information not available"],
            "pesticides": [{"name": "Consult local agricultural expert", "application": ""}],
            "weather_conditions": {
                "temperature": "Information not available",
                "humidity": "Information not available",
                "rainfall": "Information not available",
                "seasonality": "Information not available"
            },
            "occurrence_pattern": "Information not available",
            "severity_level": "Information not available"
        }

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0

        preds = model.predict(x)
        d = preds.flatten()
        j = d.max()
        
        li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']
        
        class_idx = np.argmax(d)
        disease_key = li[class_idx]
        class_name = disease_key.split('___')
        
        # Get detailed information about the disease
        disease_info = get_disease_info(disease_key)
        
        return {
            'crop': class_name[0],
            'disease': class_name[1].replace('_', ' '),
            'disease_info': disease_info
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'crop': 'Error',
            'disease': 'Failed to process image',
            'disease_info': None
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Get prediction and disease information
            result = model_predict(file_path, model)
            
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e)
            })
    return None

if __name__ == '__main__':
    app.run(debug=True)