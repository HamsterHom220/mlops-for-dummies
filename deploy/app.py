import streamlit as st
import os
import requests

# Default to localhost
UPLOAD_URL = os.getenv('UPLOAD_URL', "http://localhost:8000/upload")
PREDICT_URL = os.getenv('PREDICT_URL', "http://localhost:8000/predict/")

# Directory for images
IMAGES_DIR = os.path.join(os.getcwd(), 'imgs')

# Determine the environment (e.g., via an environment variable)
environment = os.getenv('ENV_TYPE', 'local')

st.session_state['selected_image'] = None

# Fetch available images from the directory
def get_available_images():
    return [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

# Upload image to FastAPI server
def upload_image(image):
    files = {'file': image}
    response = requests.post(UPLOAD_URL, files=files)
    return response.ok

# Predict image label
def predict_image(filename):
    response = requests.get(f"{PREDICT_URL}?filename={filename}")
    if response.ok:
        return response.json().get("label")
    return "Error"

# Initialize selected image in session state if not set
if 'selected_image' not in st.session_state:
    st.session_state['selected_image'] = None

# Sidebar: Available images
st.sidebar.title("Available images")
image_files = get_available_images()
selected_image = st.sidebar.selectbox("Choose an image", image_files)

# Set selected image in session state
if selected_image:
    st.session_state['selected_image'] = selected_image

# Upload section
uploaded_file = st.sidebar.file_uploader("Upload new image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    if upload_image(uploaded_file):
        st.sidebar.success("Image uploaded successfully!")
        # Set the uploaded file as the selected image
        st.session_state['selected_image'] = uploaded_file.name
    else:
        st.sidebar.error("Failed to upload image.")

# Main section: Show selected image
st.subheader("Selected image")
if st.session_state['selected_image']:
    if environment == 'docker':
        st.image(os.path.join(IMAGES_DIR, st.session_state['selected_image']),
                    caption=f"Image name: {st.session_state['selected_image']}",
                    use_container_width=True)
    else:
        st.image(os.path.join(IMAGES_DIR, st.session_state['selected_image']),
                    caption=f"Image name: {st.session_state['selected_image']}",
                    use_column_width=True)

# Prediction section
if st.button("Predict"):
    label = predict_image(st.session_state['selected_image'])
    st.write(f"Predicted label: {label}")
