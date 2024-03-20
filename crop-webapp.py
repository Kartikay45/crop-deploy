import streamlit as st
import pickle
from PIL import Image
import pyttsx3

DecisionTree_model = pickle.load(open('DecisionTree_model.pkl', 'rb'))

def classify(answer):
    return answer[0] + " is the best crop for cultivation here."

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("KrishiVeer (Crop Recommendation System)...")
    image = Image.open('cereal2.jpg')
    st.image(image, use_column_width=True)
    html_temp = """
    <div style="background-color:teal; padding:5px">
    <h2 style="color:white;text-align:center;">Find The Most Suitable Crop</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    activities = ['Decision Tree']
    option = st.sidebar.selectbox("One Stop Solutions To Farmers", activities)
    st.subheader(option)

    sn = st.slider('NITROGEN (N)', 0.0, 150.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 150.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 210.0)
    pt = st.slider('TEMPERATURE', 0.0, 50.0)
    phu = st.slider('HUMIDITY', 0.0, 100.0)
    pPh = st.slider('Ph', 0.0, 14.0)
    pr = st.slider('RAINFALL', 0.0, 300.0)

    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    if st.button('Classify'):
        if option == 'Decision Tree':
            prediction = classify(DecisionTree_model.predict(inputs))
            st.success(prediction)
            speak(prediction)  # Speak the prediction text using TTS

    # Display confusion matrix on a separate page
    if st.sidebar.checkbox("View Confusion Matrix", False):
        confusion_matrix_image = Image.open('output.png')
        st.image(confusion_matrix_image, caption='Confusion Matrix', use_column_width=True)

if __name__ == '__main__':
    main()
