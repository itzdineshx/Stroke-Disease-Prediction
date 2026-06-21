import gradio as gr
import pandas as pd
import joblib

# Load the trained model pipeline
model_path = 'models/stroke_prediction_pipeline.pkl'
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}. Please ensure the model is trained and saved.")

def predict_stroke(gender, age, hypertension_str, heart_disease_str, ever_married, 
                   work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    if model is None:
        return "Model not loaded. Please train the model first."
    
    # Map Yes/No to 1/0 for hypertension and heart_disease
    hypertension = 1 if hypertension_str == "Yes" else 0
    heart_disease = 1 if heart_disease_str == "Yes" else 0
    
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })
    
    # Predict probabilities
    prob = model.predict_proba(input_data)[0][1]
    
    return f"Probability of Stroke: {prob:.2%}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Stroke Disease Prediction")
    gr.Markdown("Enter the patient's information to predict the probability of a stroke.")
    
    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Female")
            age = gr.Slider(minimum=0, maximum=120, step=1, label="Age", value=50)
            hypertension = gr.Radio(choices=["No", "Yes"], label="Hypertension", value="No")
            heart_disease = gr.Radio(choices=["No", "Yes"], label="Heart Disease", value="No")
            ever_married = gr.Radio(choices=["No", "Yes"], label="Ever Married", value="Yes")
            
        with gr.Column():
            work_type = gr.Dropdown(choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"], label="Work Type", value="Private")
            residence_type = gr.Radio(choices=["Urban", "Rural"], label="Residence Type", value="Urban")
            avg_glucose_level = gr.Slider(minimum=0.0, maximum=300.0, step=0.1, label="Average Glucose Level", value=100.0)
            bmi = gr.Slider(minimum=10.0, maximum=100.0, step=0.1, label="BMI", value=28.0)
            smoking_status = gr.Dropdown(choices=["formerly smoked", "never smoked", "smokes", "Unknown"], label="Smoking Status", value="never smoked")
            
    predict_btn = gr.Button("Predict Stroke Probability", variant="primary")
    output = gr.Textbox(label="Prediction Result", text_align="center")
    
    predict_btn.click(
        fn=predict_stroke,
        inputs=[gender, age, hypertension, heart_disease, ever_married, work_type, 
                residence_type, avg_glucose_level, bmi, smoking_status],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
