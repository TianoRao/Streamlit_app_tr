import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def make_circle_mask(img):
    # Create a mask with the same size as the image
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw a white, filled circle on the mask image
    draw.ellipse((0, 0) + img.size, fill=255)

    # Create a new image with the same size and a transparent background
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Paste the original image onto the result image using the mask
    result.paste(img, mask=mask)
    return result


# Define your Streamlit app
def app():
    st.set_page_config(layout="wide")

    # Load and display an image
    try:
        img = Image.open("assets/images/logo.png")  # Updated path
    except FileNotFoundError:
        st.error("Error: Image file not found.")
        return

    new_size = (150, 150)
    img = img.resize(new_size)
    circular_img = make_circle_mask(img)

    st.image(circular_img)

    # Initial setup
    history = []
    st.title("Hi I'm your Chatbot, how can I help you?")
    st.subheader("This Chatbot is based on gemma model")

    st.write("This bot can answer questions and have conversation")
    st.write("Double check the bot's answer")

    # Create a text area for user input
    user_input = st.text_area("Let's start our conversation here:", height=5)

    # Process form submission
    if st.button("Submit"):
        if user_input:
            history.append("User: " + user_input)
            # Get the reply from the gemma model
            response = query({"inputs": user_input})
            if response:
                # Assuming the response is a list of dictionaries and we want the first one
                bot_response_dict = response[0] if response else {}
                bot_response = bot_response_dict.get(
                    "generated_text",
                    "Sorry, I couldn't get a response. Please try again.",
                )
                history.append("Weebsu: " + bot_response)
            else:
                history.append(
                    "Weebsu: I'm having trouble connecting. Please try again later."
                )

            # Combine the conversation history into a single string
            history_text = "\n".join(history)

            # Display the conversation history in a text area
            st.text_area(
                "Conversation history:",
                value=history_text,
                height=300,
                key="history_area",
                disabled=True,
            )

    # Additional information
    st.write("-----------\n\nThis project used Google Gemma Model from Huggingface")
    st.write("-----------\n\nProject by Tianji Rao")


# The following line ensures the app runs when the script is executed
if __name__ == "__main__":
    app()
