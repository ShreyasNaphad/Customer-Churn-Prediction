import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set random seed for reproducibility
#np.random.seed(42)
def truncate_array(array, max_length=90):
  """Truncates an array to the specified maximum length.

  Args:
    array: The input array.
    max_length: The maximum length of the truncated array.

  Returns:
    The truncated array.
  """

  if len(array) > max_length:
    return array[:max_length]
  else:
    return array
# Sample dataset creation (replace with your own dataset)
data = {
    'Age': [18, 21, 22, 21, 22, 21, 22, 22, 21, 21, 20, 22, 21, 22, 22, 21, 21, 21, 22, 21, 22, 24, 22,
          22, 21, 22, 25, 21, 22, 21, 27, 22, 21, 22, 22, 22, 21, 21, 22, 20, 21, 21, 21, 21, 22, 22,
          21, 21, 21, 23, 18, 21, 22, 21, 22, 21, 22, 22, 21, 21, 20, 22, 21, 22, 22, 21, 21, 21, 22, 21, 22, 24, 22,
          22, 21, 22, 25, 21, 22, 21, 27, 22, 21, 22, 22],

    'Gender': ["Male", "Male", "Male", "Male", "Male", "Male", "Female", "Male", "Male", "Male", "Male", "Male",
               "Male", "Male", "Male", "Female", "Male", "Female", "Male", "Male", "Male", "Male", "Male",
               "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Female",
               "Male", "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Male",
               "Female", "Male", "Male", "Female", "Female", "Male",    "Female", "Male", "Male", "Male", "Male",
                "Male", "Male", "Male", "Female", "Male",
                "Female", "Male", "Female", "Female", "Female",
                "Female", "Male", "Male", "Female", "Female",
                "Female", "Female", "Female", "Female", "Female",
                "Male", "Male", "Male", "Female", "Female",
                "Male", "Female", "Female", "Female", "Male",],

    'Platform': ["Netflix", "Netflix", "HBO", "Amazon Prime Video", "Netflix", "Jio Cinema", "Disney+", "Disney+",
                 "Netflix", "Netflix", "Disney+", "Sony Liv", "Netflix", "Sony Liv", "Netflix", "Disney+", "Disney+",
                 "Netflix", "Netflix", "Netflix", "Jio Cinema", "Amazon Prime Video", "Netflix", "Jio Cinema",
                 "Disney+", "Netflix", "Netflix", "Netflix", "Jio Cinema", "Amazon Prime Video", "Netflix",
                 "Amazon Prime Video", "Amazon Prime Video", "HBO", "Sony Liv", "Amazon Prime Video", "Netflix",
                 "Disney+", "Amazon Prime Video", "Sony Liv", "Amazon Prime Video", "Amazon Prime Video", "Netflix",
                 "Netflix", "Netflix", "Netflix", "Netflix", "Netflix", "Netflix","Netflix",     "Disney+",
    "Netflix","Netflix","Amazon Prime Video","Netflix","Amazon Prime Video",
    "Disney+","Netflix","Netflix","Netflix","Netflix",
    "Amazon Prime Video","Netflix","Disney+",
    "Netflix","Jio Cinema","Jio Cinema","Netflix","Netflix",
    "Amazon Prime Video","Netflix","Netflix", "Netflix", "Disney+",
    "Jio Cinema",
    "Netflix","Netflix","Netflix",
    "Amazon Prime Video",
    "Zee5","Disney+","Zee5",
    "Netflix",
    "YouTube",
    "Netflix",],

    'Content_Satisfaction': ["Satisfied", "Very Satisfied", "Satisfied", "Satisfied", "Satisfied", "Very Unsatisfied",
                             "Satisfied", "Satisfied", "Satisfied", "Satisfied", "Very Unsatisfied", "Very Satisfied",
                             "Satisfied", "Unsatisfied", "Very Satisfied", "Satisfied", "Very Satisfied", "Satisfied",
                             "Very Satisfied", "Satisfied", "Satisfied", "Very Unsatisfied", "Satisfied", "Neutral",
                             "Satisfied", "Neutral", "Very Satisfied", "Very Satisfied", "Satisfied", "Satisfied",
                             "Satisfied", "Satisfied", "Satisfied", "Very Satisfied", "Unsatisfied", "Neutral",
                             "Satisfied", "Neutral", "Satisfied", "Neutral", "Satisfied", "Neutral", "Very Satisfied",
                             "Satisfied", "Satisfied", "Satisfied", "Very Satisfied", "Satisfied", "Satisfied","Satisfied",
                             "Neutral", "Satisfied", "Satisfied", "Satisfied", "Satisfied",
                             "Very satisfied", "Satisfied", "Satisfied", "Very satisfied", "Satisfied",
                             "Satisfied", "Satisfied", "Satisfied", "Satisfied", "Satisfied",
                             "Satisfied", "Very satisfied", "Satisfied", "Satisfied", "Satisfied",
                             "Satisfied", "Very satisfied", "Very satisfied", "Very satisfied",
                             "Satisfied", "Neutral", "Satisfied", "Satisfied", "Satisfied",
                             "Neutral", "Satisfied", "Neutral", "Satisfied", "Very satisfied","Neutral"
                             ],

    'Recommendation_Quality': ["Excellent", "Excellent", "Good", "Excellent", "Good", "Good", "Good", "Excellent",
                               "Good", "Fair", "Good", "Good", "Poor", "Excellent", "Excellent", "Excellent",
                               "Good", "Excellent", "Excellent", "Good", "Very Poor", "Excellent", "Good", "Fair",
                               "Good", "Excellent", "Excellent", "Good", "Good", "Excellent", "Good", "Excellent",
                               "Excellent", "Fair", "Excellent", "Excellent", "Fair", "Good", "Poor", "Good",
                               "Fair", "Excellent", "Good", "Fair", "Excellent", "Excellent", "Good", "Good", "Good",
                               "Excellent", "Good",
    "Fair", "Excellent", "Excellent", "Good", "Good", "Excellent", "Fair", "Good",
    "Fair", "Good", "Fair", "Good", "Excellent", "Good", "Excellent", "Excellent",
    "Good", "Excellent", "Excellent", "Excellent", "Good", "Fair", "Excellent", "Good",
    "Excellent","Good", "Excellent", "Excellent", "Excellent", "Good", "Fair", "Excellent", "Good",
    "Excellent"],

    'Content_Viewing_Frequency': ["Daily", "Weekly", "Weekly", "Monthly", "Daily", "Weekly", "Weekly", "Daily",
                                  "Weekly", "Monthly", "Monthly", "Weekly", "Weekly", "Monthly", "Weekly", "Daily",
                                  "Daily", "Weekly", "Daily", "Weekly", "Weekly", "Weekly", "Weekly", "Weekly",
                                  "Daily", "Weekly", "Daily", "Weekly", "Monthly", "Weekly", "Daily", "Weekly",
                                  "Weekly", "Weekly", "Daily", "Monthly", "Weekly", "Weekly", "Daily", "Daily",
                                  "Weekly", "Monthly", "Weekly", "Weekly", "Monthly", "Daily", "Daily", "Weekly",
                                  "Weekly", "Monthly", "Weekly",
    "Weekly", "Weekly", "Weekly", "Daily",
    "Weekly", "Weekly", "Weekly", "Weekly",
    "Weekly", "Weekly", "Weekly", "Weekly",
    "Weekly", "Weekly", "Weekly", "Weekly",
    "Weekly", "Weekly", "Weekly", "Monthly",
    "Weekly", "Weekly", "Daily", "Weekly",
    "Weekly", "Weekly", "Daily", "Weekly",
    "Daily", "Weekly", "Daily", "Monthly",
    "Daily", "Weekly"
],

    'Subscription_Plan': ["Standard", "Standard", "Basic", "Premium", "Basic", "Basic", "Standard", "Premium",
                          "Basic", "Standard", "Basic", "Premium", "Premium", "Basic", "Basic", "Premium", "Standard",
                          "Premium", "Premium", "Premium", "Standard", "Premium", "Standard", "Basic", "Premium",
                          "Premium", "Premium", "Standard", "Basic", "Basic", "Basic", "Premium", "Premium", "Basic",
                          "Basic", "Premium", "Premium", "Basic", "Premium", "Premium", "Premium", "Premium",
                          "Standard", "Standard", "Standard", "Standard", "Standard", "Basic", "Premium", "Standard",
                          "Standard",
                          "Basic",
                          "Premium",
                          "Standard",
                          "Premium",
                          "Premium",
                          "Premium",
                          "Basic",
                          "Standard",
                          "Premium",
                          "Basic",
                          "Premium",
                          "Basic",
                          "Basic",
                          "Premium",
                          "Basic",
                          "Premium",
                          "Basic",
                          "Standard",
                          "Premium",
                          "Standard",
                          "Premium",
                          "Basic",
                          "Standard",
                          "Basic",
                          "Basic",
                          "Basic",
                          "Basic",
                          "Premium",
                          "Basic",
                          "Standard",
                          "Basic",
                          "Basic",
                          "Basic",
                          "Basic"
                          ],

    'Value_for_Money': ["Good", "Good", "Excellent", "Excellent", "Good", "Excellent", "Good", "Good", "Good", "Fair",
                        "Fair", "Good", "Good", "Poor", "Excellent", "Excellent", "Fair", "Fair", "Excellent", "Good",
                        "Fair", "Poor", "Good", "Good", "Good", "Fair", "Excellent", "Excellent", "Fair", "Fair",
                        "Good", "Excellent", "Excellent", "Excellent", "Poor", "Good", "Fair", "Poor", "Excellent",
                        "Very Poor", "Good", "Fair", "Good", "Good", "Good", "Good", "Good", "Excellent", "Good",
                        "Fair", "Good", "Good", "Good", "Poor", "Excellent",
    "Fair", "Excellent", "Good", "Fair", "Good", "Very poor", "Fair", "Good", "Fair", "Good",
    "Good", "Good", "Fair", "Good", "Good", "Good", "Excellent", "Excellent", "Good", "Good",
    "Good", "Fair", "Fair", "Good", "Fair", "Fair", "Fair", "Good", "Fair", "Excellent"],

    'Technical_Issues_Frequency': ["Never", "Never", "Never", "Rarely", "Rarely", "Rarely", "Rarely", "Rarely",
                                   "Rarely", "Never", "Rarely", "Never", "Rarely", "Sometimes", "Rarely", "Rarely",
                                   "Rarely", "Rarely", "Never", "Rarely", "Sometimes", "Rarely", "Sometimes", "Rarely",
                                   "Rarely", "Rarely", "Never", "Rarely", "Rarely", "Rarely", "Rarely", "Rarely",
                                   "Rarely", "Never", "Sometimes", "Rarely", "Rarely", "Sometimes", "Rarely", "Never",
                                   "Rarely", "Rarely", "Never", "Rarely", "Rarely", "Never", "Rarely", "Never", "Never",
                                   "Rarely", "Rarely", "Sometimes",
    "Rarely", "Rarely", "Rarely", "Rarely", "Sometimes",
    "Sometimes", "Rarely", "Never", "Rarely", "Never",
    "Never", "Never", "Rarely", "Rarely", "Never",
    "Never", "Rarely", "Sometimes", "Rarely", "Rarely",
    "Rarely", "Rarely", "Rarely", "Sometimes", "Never",
    "Sometimes", "Sometimes", "Rarely", "Sometimes", "Never",
    "Sometimes", "Sometimes", "Rarely"],

    'Likelihood_to_Cancel': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,
1, 0, 0, 0,
0, 0, 0, 0,
0, 0, 0, 0,
0, 0, 0, 0,
1, 0, 0, 0,
1, 0, 0, 0,
0, 0, 0, 0,
0, 0, 0, 0,
0, 0]
}
for key, value in data.items():
    print(f'{key}: {len(value)}')
# Convert to DataFrame
# Convert to DataFrame
df = pd.DataFrame(data)

# Label encoding for categorical columns
label_encoders = {}
for column in [
    'Gender', 'Content_Satisfaction', 'Recommendation_Quality',
    'Platform', 'Content_Viewing_Frequency', 'Subscription_Plan',
    'Value_for_Money', 'Technical_Issues_Frequency'
]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store the encoder for later use

# Scale the features
scaler = StandardScaler()
X = df.drop('Likelihood_to_Cancel', axis=1)
X_scaled = scaler.fit_transform(X)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Likelihood_to_Cancel'], test_size=0.2,
                                                    random_state=42)

# Using Random Forest for better model performance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate feature importance
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Function to handle unseen labels and make predictions

# Function to handle unseen labels and make predictions
def predict_likelihood(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Handle unseen labels
    for column, val in input_df.items():
        if column in label_encoders:
            le = label_encoders[column]
            # If the value is unseen, use the most frequent label (fallback)
            if val[0] not in le.classes_:
                fallback_value = le.classes_[0]  # Most common class
                input_df[column] = le.transform([fallback_value])
                #st.warning(f"Unseen value for '{column}' replaced with most frequent: '{fallback_value}'")
            else:
                input_df[column] = le.transform([val[0]])

    # Scale the input data using the same StandardScaler
    scaled_data = scaler.transform(input_df)

    # Predict probability of churn
    prediction_probability = model.predict_proba(scaled_data)[0][1]  # Probability of churn
    prediction_percentage = prediction_probability * 100  # Convert to percentage
    return prediction_percentage




# Streamlit UI for the app
st.set_page_config(page_title="OTT Churn Prediction App", page_icon="🎬", layout="centered")

# Dynamic color theme based on platform
platform_colors = {
    "Netflix": "#E50914",  # Red for Netflix
    "Hulu": "#3dbb3d",  # Green for Hulu
    "Amazon Prime": "#00A8E1",  # Blue for Amazon Prime
    "Disney+": "#113CCF",  # Blue for Disney+
    "Other": "#ffffff"  # Default white
}
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Feature Importance Graph"])
if page == "Prediction":
    # Churn Prediction Page
    st.title("OTT Churn Prediction")
# Get platform from user to change theme
    selected_platform = st.selectbox("Choose OTT Platform", list(platform_colors.keys()))

# CSS for OTT-inspired theme with dynamic color, background image, and animations
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');
        body {{
            background: url(https://path-to-your-background-image.jpg);
            background-size: cover;
            color: #c9d1d9;
            font-family: 'Montserrat', sans-serif;
            font-size: 1.1rem;
        }}
        h1 {{
            color: {platform_colors[selected_platform]};
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }}
        .container {{
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.3);
            background-color: rgba(26, 26, 46, 0.9);
            margin-bottom: 30px;
            animation: fadeIn 2s ease-in-out;
            }}
        .stButton button {{
            background-color: {platform_colors[selected_platform]};
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            transition: 0.3s;
            font-size: 1.2em;
        }}
        .stButton button:hover {{
            background-color: #1f6feb;
            transform: scale(1.05);
        }}
        .stSelectbox label, .stNumberInput label {{
            color: #f0f6fc;
        }}
        .result {{
            color: #76ff03;
            font-size: 1.7em;
            text-align: center;
            padding-top: 15px;
            animation: fadeIn 2s ease-in-out;
        }}
        .progress-bar {{
            width: 100%;
            background-color: #ccc;
            border-radius: 13px;
            margin-top: 20px;
        }}
        .progress-bar-fill {{
            height: 20px;
            border-radius: 13px;
            background-color: {platform_colors[selected_platform]};
            transition: width 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        .floating-elements {{
            position: fixed;
            top: -50px;
            width: 100%;
            height: 150px;
            background: rgba(255, 255, 255, 0.05);
            filter: blur(5px);
            animation: float 5s ease-in-out infinite;
        }}
        @keyframes float {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(20px); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # OTT logo or poster representation
    #st.image(f"https://path-to-ott-{selected_platform}-poster.jpg", caption=f"Experience {selected_platform}", use_column_width=True)

    # Page header with a cinematic feel 🎬
    st.title(f"🎬 {selected_platform} Churn Prediction App 🎥")
    st.write("Predict how likely a user is to cancel their subscription based on their platform usage. Let’s dive in!")

    # Input sections wrapped in card-like containers
    st.header("👤 User Information")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Viewing Preferences
    st.header("📺 Viewing Preferences")
    with st.container():
        col3, col4, col5 = st.columns(3)
        with col3:
            content_satisfaction = st.selectbox("Content Satisfaction", ["Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very Unsatisfied"])
        with col4:
            recommendation_quality = st.selectbox("Recommendation Quality", ["Excellent", "Good", "Average", "Poor", "Very Poor"])
        with col5:
            viewing_frequency = st.selectbox("Content Viewing Frequency", ["Daily", "Weekly", "Monthly", "Rarely", "Never"])

    # Subscription Details
    st.header("💳 Subscription Details")
    with st.container():
        col6, col7, col8 = st.columns(3)
        with col6:
            subscription_plan = st.selectbox("Subscription Plan", ["Basic", "Standard", "Premium"])
        with col7:
            value_for_money = st.selectbox("Value for Money", ["Excellent", "Good", "Fair", "Poor", "Very Poor"])
        with col8:
            technical_issues_frequency = st.selectbox("Technical Issues Frequency", ["Never", "Rarely", "Sometimes", "Often", "Always"])

    # Submit button to make predictions
    if st.button("🎯 Predict Likelihood to Cancel"):
        with st.spinner('Analyzing the data...'):
            time.sleep(2)  # Simulate delay

            # Prepare user input data
            user_input = {
                'Age': age,
                'Gender': gender,
                'Platform': selected_platform,
                'Content_Satisfaction': content_satisfaction,
                'Recommendation_Quality': recommendation_quality,
                'Content_Viewing_Frequency': viewing_frequency,
                'Subscription_Plan': subscription_plan,
                'Value_for_Money': value_for_money,
                'Technical_Issues_Frequency': technical_issues_frequency,
            }

            # Prediction output
            likelihood = predict_likelihood(user_input)

            # Displaying the prediction with an animated progress bar
            st.subheader("Likelihood to Cancel Subscription:")
            st.write(f"**{likelihood:.2f}%**")
            st.markdown(
                f"""
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {likelihood}%;"></div>
                </div>
                """, unsafe_allow_html=True
            )

    # Adding floating background elements for enhanced visuals
    st.markdown("<div class='floating-elements'></div>", unsafe_allow_html=True)
elif page == "Feature Importance Graph":
    # Feature Importance Graph Page
    st.title("Feature Importance for Churn Prediction")

    # Plot animated bar chart of feature importance
    fig, ax = plt.subplots()

    bars = ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel("Importance")
    plt.title("Animated Feature Importance")

    # Plot the feature importance
    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="Blues_d", ax=ax)
    ax.set_title("Feature Importance for Predicting Churn")
    st.pyplot(fig)