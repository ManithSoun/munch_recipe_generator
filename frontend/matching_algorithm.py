import pandas as pd
import ast
import os
import random
import numpy as np
import tensorflow as tf 
import joblib
from tensorflow.keras.applications.efficientnet import preprocess_input

recipe_folder = "recipes_parts/"
num_random_file = 2

all_csv_files = [os.path.join(recipe_folder, f) for f in os.listdir(recipe_folder) if f.endswith('.csv')]
selected_files = random.sample(all_csv_files, min(num_random_file, len(all_csv_files)))

all_recipes = [pd.read_csv(file) for file in selected_files]
print(f"Randomly loaded {len(all_recipes)} recipe parts: {selected_files}")

@tf.keras.utils.register_keras_serializable()
def extract_bert_cls(inputs):
    input_ids, attention_mask = inputs
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0, :]
    return cls_token

text_model = tf.keras.models.load_model('models/text_bert_model.h5', custom_objects={'extract_bert_cls': extract_bert_cls}, compile=False)
image_model = tf.keras.models.load_model('models/best_model_v2.keras')
label_encoder = joblib.load('models/label_encoder.pkl')

def clean_ingredients(ingredients_str):
    try:
        ingredients = ast.literal_eval(ingredients_str)
    except (ValueError, SyntaxError):
        ingredients = ingredients_str.split(',')
    ingredients = [i.strip().lower() for i in ingredients if i.strip()]
    return ingredients

def clean_instruction(instruction):
    if isinstance(instruction, str):
        try:
            decoded = ast.literal_eval(instruction)
            if isinstance(decoded, list) and len(decoded) > 0:
                return " ".join(decoded) 
        except (ValueError, SyntaxError):
            return instruction
    return str(instruction)

def predict_ingredients_from_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(380, 380))  # Correct EfficientNetB4 size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)  # EfficientNet expects preprocessed input
    img_array = np.expand_dims(img_array, axis=0)

    preds = image_model.predict(img_array)
    top_indices = np.argsort(preds[0])[-3:]

    try:
        predicted_labels = label_encoder.inverse_transform(top_indices)
    except ValueError:
        return []

    return [label.lower() for label in predicted_labels]


def match_recipes(provided_ingredients, recipe_list):
    provided_set = set([i.strip().lower() for i in provided_ingredients])
    matching_recipes = []

    for df in recipe_list:
        for _, row in df.iterrows():
            recipe_ingredients = clean_ingredients(row['Ingredients'])
            matches = provided_set.intersection(recipe_ingredients)

            if matches:
                total = len(recipe_ingredients) if len(recipe_ingredients) > 0 else 1
                missing = set(recipe_ingredients) - matches

                matching_recipes.append({
                    'Recipe': row['Recipe'],
                    'Instruction': clean_instruction(row['Instruction']),
                    'Matching Ingredients': matches,
                    'Missing Ingredients': missing,
                    'Matching Count': len(matches),
                    'Total Ingredients': total,
                    'Match Percent': len(matches) / total
                })

    matching_recipes = sorted(matching_recipes, key=lambda x: (x['Match Percent'], x['Matching Count']), reverse=True)
    return matching_recipes

def find_recipe(text_input):
    if isinstance(text_input, str):
        ingredients = [i.strip().lower() for i in text_input.split(',')]
    else:
        ingredients = text_input

    matched = match_recipes(ingredients, all_recipes)

    if not matched:
        return "❌ No matching recipes found."

    result = f"🧾 Your available ingredient(s): {', '.join(ingredients)}\n\n"
    for idx, recipe in enumerate(matched[:5], start=1):
        result += f"""🍽️ Top {idx}: {recipe['Recipe']}
📝 Instructions: {recipe['Instruction']}
🥘 Matching Ingredients: {', '.join(recipe['Matching Ingredients'])}
❌ Missing Ingredients: {', '.join(recipe['Missing Ingredients']) if recipe['Missing Ingredients'] else 'None'}
✅ Match {recipe['Matching Count']} out of {recipe['Total Ingredients']} ingredients ({recipe['Match Percent']*100:.2f}% match)\n\n"""

    return result

def find_recipe_from_image(image_path):
    predicted_ingredients = predict_ingredients_from_image(image_path)

    if not predicted_ingredients:
        return "⚠️ Sorry, could not recognize ingredients clearly. Please upload a clearer image."

    return find_recipe(predicted_ingredients)
