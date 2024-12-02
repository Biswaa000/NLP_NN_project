import streamlit as st 
import pickle
import os
import tensorflow
import distutils
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences

# Check if you can access keras.utils
print(keras.utils)



def main():
    # Get the current directory (the folder where the main app resides)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        encoder = pickle.load(open(f'{current_dir}/tokenizer.pkl', 'rb'))
        model = tf.keras.models.load_model(f'{current_dir}/my_model.h5')

    except FileNotFoundError:
        st.error("Required files (`vectorizer.pkl` or `model.pkl`) are missing. Please add them to the working directory.")
        st.stop()









    # Streamlit app UI
    st.title("Sentiment Analysis App")
    st.write("Enter text below and click 'Analyze' to see the sentiment.")

    # Input text area
    input_text = st.text_area("Input Text", placeholder="Type your text here...")



    seq = encoder.texts_to_sequences([input_text])  # Convert the list of text to sequences

    # Pad the sequences to a max length of 50
    sequences = pad_sequences(seq, padding='post', maxlen=50, truncating='post')

    # Check the length of the padded sequence
    print(f"Length of sequences: {len(sequences)}")
    print(f"Padded sequences: {sequences[0]}")








    if st.button('Analyze 😊'):
        if input_text.strip():
            # Preprocess the input
            # transformed_sms = list(input_text)
            # Vectorize the input
            try:
                # vector_input = encoder.texts_to_sequences(transformed_sms)
                vector_input = encoder.texts_to_sequences([input_text])

                # Pad the sequences to a max length of 50
                sequences = pad_sequences(seq, padding='post', maxlen=50, truncating='post')

            except Exception as e:
                st.error(f"Error with encoder: {e}")
                st.stop()
            
            # Predict using the model
            try:
                # Make sure sequences[0] has the correct shape for prediction
                input_data = sequences[0].reshape(1, -1)
                result = model.predict(input_data)
            except Exception as e:
                st.error(f"Error with model prediction: {e}")
                st.stop()

            # Display the result
            if result > 0.5:
                st.header("Positive")
            else:
                st.header("Negative")
        else:
            st.warning("Please enter a valid message.")

if __name__== "__main__":
    main()
