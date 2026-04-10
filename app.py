import gradio as gr
from predict import predict_category

def classify_paper(abstract):
    if not abstract.strip():
        return "⚠️ Please enter a valid abstract."
    result = predict_category(abstract)
    return f"🎯 Predicted Category: {result}"

# 🎨 Custom UI with styling
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📄 NLP Research Paper Classifier
    ### 🚀 Enter your abstract and get instant category prediction
    """)

    with gr.Row():
        with gr.Column():
            abstract_input = gr.Textbox(
                lines=6,
                placeholder="✍️ Paste your research abstract here...",
                label="Abstract Input"
            )

            predict_btn = gr.Button("🔍 Predict Category")

        with gr.Column():
            output = gr.Textbox(
                label="Prediction Output",
                lines=3
            )

    # 🔹 Example inputs (great for presentation)
    gr.Examples(
        examples=[
            ["This paper proposes a deep learning model using CNNs for image classification."],
            ["We analyze stock market trends using Bayesian statistics and time series models."],
            ["A new method to study prime numbers and number theory is presented."],
            ["Observations of black holes and gravitational waves using astrophysics models."]
        ],
        inputs=abstract_input
    )

    # Button click action
    predict_btn.click(
        fn=classify_paper,
        inputs=abstract_input,
        outputs=output
    )

    gr.Markdown("💡 Tip: Try different research abstracts to see how the model performs!")

# Launch app
demo.launch()