# Banana Leaf Doctor (deployment app)

A Streamlit web app: upload a banana leaf photo, get the predicted class (cordana, sigatoka,
pestalotiopsis or healthy), the confidence in %, a gauge and a probability chart.

## 1. Get the model files
Run the deployment notebook on Kaggle, then download these from the **Output** tab and put them in
this folder next to `app.py`:

- `banana_leaf_mobilenetv2.keras` (required)
- `class_names.json` (recommended)
- `model_config.json` (optional, shows the test accuracy in the sidebar)

## 2. Install and run
Use a Python version supported by TensorFlow (3.10 to 3.12 is a safe choice).

```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Folder layout
```
banana_leaf_app/
  app.py
  requirements.txt
  banana_leaf_mobilenetv2.keras
  class_names.json
  model_config.json
  .streamlit/config.toml
```

## Publishing online (optional)
Hugging Face Spaces (Streamlit) or Streamlit Community Cloud both work: upload this folder
(the model file is about 12 MB) and the app runs from `app.py`.
