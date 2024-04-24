import streamlit as st
from openai import OpenAI

client = OpenAI()
import os
import datetime
import base64


@st.cache_data
def call_chat_completions_api(
    model,
    message,
    response_format="text",
    system_message=None,
    messages=None,
    stream=False,
    temperature=0,
):
    if not messages:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        response_format={
            "type": "text" if response_format == "text" else "json_object"
        },
        temperature=temperature,
        stream=stream,  # True,
        model=model,  # "gpt-4-0125-preview",
        messages=messages,  # [
        #     {"role": "system", "content": "You are a web scraping expert in Python and Beautiful Soup that provides error-free code to the user"},
        #     {"role": "user", "content": ''}
        # ]
    )
    return response


@st.cache_data
def call_gpt4(model, messages):
    return client.chat.completions.create(
        model=model,
        messages=messages,
    )


@st.cache_data
# Function to encode the image
def encode_image(image):
    return base64.b64encode(image.getbuffer()).decode("utf-8")


@st.cache_data
def make_message(message):
    return {"type": "text", "text": message}


@st.cache_data
def make_messages(message, uploaded_files):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(uploaded_file)}",
                    },
                }
                for uploaded_file in uploaded_files
            ]
            + [{"type": "text", "text": message}],
        }
    ]


@st.cache_data
def generate_file_name(uploaded_file_name):
    return f'{str(datetime.datetime.now()).replace(":", "-")}-{uploaded_file_name}'


@st.cache_data
def write_to_disk(file_path, open_type, _data):
    with open(file_path, open_type) as f:
        f.write(_data)


# Main Streamlit app
def main():
    st.title("Ask GPT 4")

    # Ask a question
    st.subheader("Ask a Question")
    message = st.text_area("Enter your question here")

    # Upload images (optional)
    st.subheader("Upload Images (optional)")
    uploaded_files = st.file_uploader(
        "Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if st.button("Send"):
        # Process question and provide response
        messages = make_messages(message, uploaded_files)

        response = call_gpt4("gpt-4-turbo", messages)
        st.subheader("Response")
        response = response.choices[0].message.content
        st.write(response)

        file_path = os.path.join("qa", f"qa-{hash(message)}.txt")
        write_to_disk(
            file_path,
            "w",
            f"""
            {message = }

            {response = }
            """,
        )


if __name__ == "__main__":
    main()
