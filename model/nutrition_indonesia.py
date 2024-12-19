import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the nutrition data
nutrition_df = pd.read_csv('nutrition.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        ingredients = ingredients.split(',')
        ingredients = [ingredient.strip() for ingredient in ingredients]

        # Filter the dataframe based on the ingredients
        filtered_df = nutrition_df[nutrition_df['Ingredients'].apply(lambda x: all(item in x.split(',') for item in ingredients))]

        # Get the recommended dishes
        recommended_dishes = filtered_df['Dish'].tolist()

        # Get the nutritional information for the recommended dishes
        nutrition_info = {}
        for dish in recommended_dishes:
            nutrition_info[dish] = {
                'Calories': filtered_df[filtered_df['Dish'] == dish]['Calories'].values[0],
                'Protein': filtered_df[filtered_df['Dish'] == dish]['Protein'].values[0],
                'Fat': filtered_df[filtered_df['Dish'] == dish]['Fat'].values[0],
                'Carbohydrates': filtered_df[filtered_df['Dish'] == dish]['Carbohydrates'].values[0]
            }

        return render_template('../index.html', recommended_dishes=recommended_dishes, nutrition_info=nutrition_info)

    else:
        return render_template('../index.html')

if __name__ == '__main__':
    app.run(debug=True)