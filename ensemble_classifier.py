import joblib
import numpy as np
import os
from pydantic import BaseModel
from typing import List

class ClassifierSymptomRequest(BaseModel):
    symptoms: List[str]

class ClassifierResponse(BaseModel):
    predicted_disease: str

ALL_POSSIBLE_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
    'spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings',
    'weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough',
    'high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache',
    'yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes',
    'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise',
    'blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
    'runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus',
    'neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
    'extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements',
    'loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
    'foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching',
    'toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium',
    'red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze'
]

SYMPTOM_TO_INDEX = {symptom: i for i, symptom in enumerate(ALL_POSSIBLE_SYMPTOMS)}

INDO_TO_ENG_SYMPTOM = {
    'gatal': 'itching',
    'ruam kulit': 'skin_rash',
    'benjolan di kulit': 'nodal_skin_eruptions',
    'bersin terus menerus': 'continuous_sneezing',
    'menggigil': 'shivering',
    'kedinginan': 'chills',
    'nyeri sendi': 'joint_pain',
    'sakit perut': 'stomach_pain',
    'asam lambung': 'acidity',
    'sariawan di lidah': 'ulcers_on_tongue',
    'penyusutan otot': 'muscle_wasting',
    'muntah': 'vomiting',
    'nyeri saat buang air kecil': 'burning_micturition',
    'bercak saat kencing': 'spotting_ urination',
    'kelelahan': 'fatigue',
    'kenaikan berat badan': 'weight_gain',
    'kecemasan': 'anxiety',
    'tangan dan kaki dingin': 'cold_hands_and_feets',
    'perubahan suasana hati': 'mood_swings',
    'penurunan berat badan': 'weight_loss',
    'kegelisahan': 'restlessness',
    'lesu': 'lethargy',
    'bercak di tenggorokan': 'patches_in_throat',
    'kadar gula tidak teratur': 'irregular_sugar_level',
    'batuk': 'cough',
    'demam tinggi': 'high_fever',
    'mata cekung': 'sunken_eyes',
    'sesak napas': 'breathlessness',
    'berkeringat': 'sweating',
    'dehidrasi': 'dehydration',
    'gangguan pencernaan': 'indigestion',
    'sakit kepala': 'headache',
    'kulit kekuningan': 'yellowish_skin',
    'urin berwarna gelap': 'dark_urine',
    'mual': 'nausea',
    'kehilangan selera makan': 'loss_of_appetite',
    'nyeri di belakang mata': 'pain_behind_the_eyes',
    'sakit punggung': 'back_pain',
    'sembelit': 'constipation',
    'nyeri perut': 'abdominal_pain',
    'diare': 'diarrhoea',
    'demam ringan': 'mild_fever',
    'urin berwarna kuning': 'yellow_urine',
    'mata menguning': 'yellowing_of_eyes',
    'gagal hati akut': 'acute_liver_failure',
    'kelebihan cairan': 'fluid_overload',
    'pembengkakan perut': 'swelling_of_stomach',
    'pembengkakan kelenjar getah bening': 'swelled_lymph_nodes',
    'malaise': 'malaise',
    'penglihatan kabur dan terdistorsi': 'blurred_and_distorted_vision',
    'dahak': 'phlegm',
    'iritasi tenggorokan': 'throat_irritation',
    'mata merah': 'redness_of_eyes',
    'tekanan sinus': 'sinus_pressure',
    'hidung meler': 'runny_nose',
    'hidung tersumbat': 'congestion',
    'nyeri dada': 'chest_pain',
    'kelemahan pada tungkai': 'weakness_in_limbs',
    'detak jantung cepat': 'fast_heart_rate',
    'nyeri saat buang air besar': 'pain_during_bowel_movements',
    'nyeri di daerah anus': 'pain_in_anal_region',
    'tinja berdarah': 'bloody_stool',
    'iritasi pada anus': 'irritation_in_anus',
    'nyeri leher': 'neck_pain',
    'pusing': 'dizziness',
    'kram': 'cramps',
    'memar': 'bruising',
    'obesitas': 'obesity',
    'kaki bengkak': 'swollen_legs',
    'pembuluh darah bengkak': 'swollen_blood_vessels',
    'wajah dan mata bengkak': 'puffy_face_and_eyes',
    'pembesaran tiroid': 'enlarged_thyroid',
    'kuku rapuh': 'brittle_nails',
    'pembengkakan ekstremitas': 'swollen_extremeties',
    'rasa lapar berlebihan': 'excessive_hunger',
    'kontak di luar nikah': 'extra_marital_contacts',
    'bibir kering dan kesemutan': 'drying_and_tingling_lips',
    'bicara cadel': 'slurred_speech',
    'nyeri lutut': 'knee_pain',
    'nyeri sendi panggul': 'hip_joint_pain',
    'kelemahan otot': 'muscle_weakness',
    'leher kaku': 'stiff_neck',
    'pembengkakan sendi': 'swelling_joints',
    'kekakuan gerakan': 'movement_stiffness',
    'gerakan berputar': 'spinning_movements',
    'kehilangan keseimbangan': 'loss_of_balance',
    'goyah': 'unsteadiness',
    'kelemahan satu sisi tubuh': 'weakness_of_one_body_side',
    'kehilangan indra penciuman': 'loss_of_smell',
    'ketidaknyamanan kandung kemih': 'bladder_discomfort',
    'bau urin tidak sedap': 'foul_smell_of urine',
    'rasa ingin kencing terus menerus': 'continuous_feel_of_urine',
    'buang angin': 'passage_of_gases',
    'gatal bagian dalam': 'internal_itching',
    'penampilan toksik (tifus)': 'toxic_look_(typhos)',
    'depresi': 'depression',
    'iritabilitas': 'irritability',
    'nyeri otot': 'muscle_pain',
    'perubahan kesadaran': 'altered_sensorium',
    'bintik merah di tubuh': 'red_spots_over_body',
    'sakit perut bagian bawah': 'belly_pain',
    'menstruasi tidak normal': 'abnormal_menstruation',
    'bercak perubahan warna kulit': 'dischromic _patches',
    'mata berair': 'watering_from_eyes',
    'peningkatan nafsu makan': 'increased_appetite',
    'poliuria': 'polyuria',
    'riwayat keluarga': 'family_history',
    'dahak berlendir': 'mucoid_sputum',
    'dahak berwarna karat': 'rusty_sputum',
    'kurang konsentrasi': 'lack_of_concentration',
    'gangguan penglihatan': 'visual_disturbances',
    'menerima transfusi darah': 'receiving_blood_transfusion',
    'menerima suntikan tidak steril': 'receiving_unsterile_injections',
    'koma': 'coma',
    'pendarahan lambung': 'stomach_bleeding',
    'perut kembung': 'distention_of_abdomen',
    'riwayat konsumsi alkohol': 'history_of_alcohol_consumption',
    'kelebihan cairan': 'fluid_overload', 
    'darah dalam dahak': 'blood_in_sputum',
    'urat nadi menonjol di betis': 'prominent_veins_on_calf',
    'jantung berdebar': 'palpitations',
    'sakit saat berjalan': 'painful_walking',
    'jerawat berisi nanah': 'pus_filled_pimples',
    'komedo': 'blackheads',
    'jaringan parut': 'scurring',
    'kulit mengelupas': 'skin_peeling',
    'debu seperti perak': 'silver_like_dusting',
    'lekukan kecil di kuku': 'small_dents_in_nails',
    'kuku meradang': 'inflammatory_nails',
    'melepuh': 'blister',
    'luka merah di sekitar hidung': 'red_sore_around_nose',
    'kerak kuning mengalir': 'yellow_crust_ooze'
}

ENG_TO_INDO_DISEASE = {
    'Fungal Infection': 'Infeksi Jamur',
    'Allergy': 'Alergi',
    'GERD': 'GERD',
    'Chronic Cholestasis': 'Kolestasis Kronis',
    'Drug Reaction': 'Reaksi Obat',
    'Peptic Ulcer Disease': 'Penyakit Tukak Peptik',
    'AIDS': 'AIDS',
    'Diabetes ': 'Diabetes',
    'Gastroenteritis': 'Gastroenteritis',
    'Bronchial Asthma': 'Asma Bronkial',
    'Hypertension ': 'Hipertensi',
    'Migraine': 'Migrain',
    'Cervical Spondylosis': 'Spondilosis Servikal',
    'Paralysis (brain hemorrhage)': 'Kelumpuhan (Pendarahan Otak)',
    'Jaundice': 'Penyakit Kuning',
    'Malaria': 'Malaria',
    'Chickenpox': 'Cacar Air',
    'Dengue': 'Demam Berdarah',
    'Typhoid': 'Tifus',
    'Hepatitis A': 'Hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'Alcoholic Hepatitis': 'Hepatitis Alkoholik',
    'Tuberculosis': 'Tuberkulosis',
    'Common Cold': 'Pilek',
    'Pneumonia': 'Pneumonia',
    'Dimorphic Hemmorhoids (piles)': 'Ambeien Dimorfik',
    'Heart Attack': 'Serangan Jantung',
    'Varicose Veins': 'Varises',
    'Hypothyroidism': 'Hipotiroidisme',
    'Hyperthyroidism': 'Hipertiroidisme',
    'Hypoglycemia': 'Hipoglikemia',
    'Osteoarthritis': 'Osteoartritis',
    'Arthritis': 'Artritis',
    'Vertigo': 'Vertigo',
    'Acne': 'Jerawat',
    'Urinary Tract Infection': 'Infeksi Saluran Kemih',
    'Psoriasis': 'Psoriasis',
    'Impetigo': 'Impetigo'
}

def translate_symptoms(indonesian_symptoms: List[str]) -> List[str]:
    translated = []
    for symp in indonesian_symptoms:
        key = symp.strip().lower()
        if key in INDO_TO_ENG_SYMPTOM:
            translated.append(INDO_TO_ENG_SYMPTOM[key])
        else:
            print(f"Perhatian: gejala '{symp}' tidak dikenali—diabaikan.")
    return translated

class SymptomClassifier:
    def __init__(self, classifier_path: str, label_encoder_path: str):
        print(f"DEBUG: SymptomClassifier init: classifier='{classifier_path}', label_encoder='{label_encoder_path}'")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found at: {classifier_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found at: {label_encoder_path}")

        self.classifier = joblib.load(classifier_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.all_symptoms_list = ALL_POSSIBLE_SYMPTOMS
        self.symptom_to_index = SYMPTOM_TO_INDEX
        print("DEBUG: SymptomClassifier loaded classifier and label encoder.")

    def _create_feature_vector(self, user_symptoms: List[str]) -> np.ndarray:
        feature_vector = np.zeros(len(self.all_symptoms_list), dtype=int)
        for symptom in user_symptoms:
            symptom_cleaned = symptom.strip().lower().replace(" ", "_")
            if symptom_cleaned in self.symptom_to_index:
                feature_vector[self.symptom_to_index[symptom_cleaned]] = 1
            else:
                print(f"WARNING: Gejala '{symptom}' (cleaned: '{symptom_cleaned}') tidak dikenali. Mengabaikan.")
        return feature_vector.reshape(1, -1)

    def predict_disease(self, user_symptoms_indonesia: List[str]) -> str:
        if not isinstance(user_symptoms_indonesia, list):
            raise ValueError("Input symptoms must be a list of strings.")
        if not user_symptoms_indonesia:
            raise ValueError("No symptoms provided for classification.")

        translated = translate_symptoms(user_symptoms_indonesia)
        if not translated:
            raise ValueError("Tidak ada gejala valid setelah penerjemahan.")

        feature_vector = self._create_feature_vector(translated)
        prediction_numeric = self.classifier.predict(feature_vector)
        predicted_english = self.label_encoder.inverse_transform(prediction_numeric)[0]

        predicted_indonesia = ENG_TO_INDO_DISEASE.get(predicted_english, predicted_english)
        return predicted_indonesia