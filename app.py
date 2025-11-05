import pandas as pd
import joblib
import gradio as gr
import os

# --- 1. LOAD MODELS AND DATA ---
print("Loading models and data...")
# Load the trained models
try:
    win_loss_model = joblib.load('models/win_loss_classifier.pkl')
    score_model = joblib.load('models/score_diff_regressor.pkl')
    print("✅ Models loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Model files not found. Please run the '03_model_training.ipynb' notebook.")
    exit()

# Load the final dataset to get the most recent team stats
try:
    df = pd.read_csv('data/final/final_dataset.csv')
    # Important: Convert the single date column for proper sorting
    df['home_gameDate'] = pd.to_datetime(df['home_gameDate'])
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Data file not found. Please run the '02_feature_engineering.ipynb' notebook.")
    exit()

# Get a sorted list of unique team names for the dropdown menu
team_names = sorted(df['home_teamName'].unique())
print(f"✅ Found {len(team_names)} teams.")

# --- 2. DEFINE THE PREDICTION FUNCTION ---
def predict_game(home_team, away_team):
    """
    Predicts the outcome of a game between two teams based on their most recent stats.
    """
    if not home_team or not away_team:
        return "Please select both a Home and an Away team.", ""
    if home_team == away_team:
        return "Please select two different teams.", ""

    # Find the most recent stats for the HOME team (when they were the home team)
    latest_home_stats_row = df[df['home_teamName'] == home_team].sort_values(by='home_gameDate', ascending=False)
    if latest_home_stats_row.empty:
        return f"No data available for {home_team} as a home team.", ""
    latest_home_stats = latest_home_stats_row.iloc[0]
    
    # Find the most recent stats for the AWAY team (when they were the away team)
    # **CORRECTED LINE BELOW**
    latest_away_stats_row = df[df['away_teamName'] == away_team].sort_values(by='home_gameDate', ascending=False)
    if latest_away_stats_row.empty:
        return f"No data available for {away_team} as an away team.", ""
    latest_away_stats = latest_away_stats_row.iloc[0]

    # The order of columns MUST be the same as when the model was trained
    feature_columns = [
        'home_points_rolling5', 'home_assists_rolling5', 'home_reboundsTotal_rolling5', 
        'home_fieldGoalsPercentage_rolling5', 'home_threePointersPercentage_rolling5', 'home_freeThrowsPercentage_rolling5',
        'away_points_rolling5', 'away_assists_rolling5', 'away_reboundsTotal_rolling5', 
        'away_fieldGoalsPercentage_rolling5', 'away_threePointersPercentage_rolling5', 'away_freeThrowsPercentage_rolling5'
    ]
    
    # Assemble the feature vector from the retrieved stats
    model_input_data = [
        latest_home_stats['home_points_rolling5'],
        latest_home_stats['home_assists_rolling5'],
        latest_home_stats['home_reboundsTotal_rolling5'],
        latest_home_stats['home_fieldGoalsPercentage_rolling5'],
        latest_home_stats['home_threePointersPercentage_rolling5'],
        latest_home_stats['home_freeThrowsPercentage_rolling5'],
        latest_away_stats['away_points_rolling5'],
        latest_away_stats['away_assists_rolling5'],
        latest_away_stats['away_reboundsTotal_rolling5'],
        latest_away_stats['away_fieldGoalsPercentage_rolling5'],
        latest_away_stats['away_threePointersPercentage_rolling5'],
        latest_away_stats['away_freeThrowsPercentage_rolling5']
    ]
    model_input = pd.DataFrame([model_input_data], columns=feature_columns)

    # --- Make Predictions ---
    win_probability = win_loss_model.predict_proba(model_input)[0]
    home_win_prob = win_probability[1]
    predicted_score_diff = score_model.predict(model_input)[0]

    # --- Format Output ---
    winner_text = f"Predicted Winner: **{home_team}** with **{home_win_prob:.1%}** probability."
    score_text = f"Predicted Score Difference: **{home_team} by {predicted_score_diff:.1f} points**."
    
    return winner_text, score_text

# --- 3. CREATE AND LAUNCH THE GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="NBA Game Predictor") as app:
    gr.Markdown("# 🏀 NBA Game Predictor")
    gr.Markdown("Select a home and away team to predict the game's outcome based on their recent performance.")
    
    with gr.Row():
        home_team_dropdown = gr.Dropdown(choices=team_names, label="Home Team", value="Golden State Warriors")
        away_team_dropdown = gr.Dropdown(choices=team_names, label="Away Team", value="Los Angeles Lakers")
        
    predict_button = gr.Button("Predict Game Outcome")
    
    with gr.Column():
        winner_output = gr.Markdown()
        score_output = gr.Markdown()

    predict_button.click(
        fn=predict_game,
        inputs=[home_team_dropdown, away_team_dropdown],
        outputs=[winner_output, score_output]
    )

print("Launching the application...")
app.launch()