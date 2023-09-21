# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)
# model = tf.keras.models.load_model('file from google drive')  # Load your .h5 model file

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         uploaded_file = request.files['file']
#         img = tf.image.decode_image(uploaded_file.read(), channels=3)
#         img = tf.image.resize(img, (224, 224))
#         img = img / 255.0  # Normalize the image
#         img = np.expand_dims(img, axis=0)

#         prediction = model.predict(img)
#         # Process the prediction as needed, e.g., get class labels, probabilities, etc.

#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import gdown  # Import the gdown library

app = Flask(__name__)

# Define the Google Drive file ID of your model
google_drive_file_id = "1Akizg9DC2Z_7AfYquSCMw9p6iqf2KRTv"

# Define the local file path where you want to save the model
model_file_path = "model.h5"

# Download the model from Google Drive
gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_file_path, quiet=False)

# Load your .h5 model file
model = tf.keras.models.load_model(model_file_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        img = tf.image.decode_image(uploaded_file.read(), channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        # Process the prediction as needed, e.g., get class labels, probabilities, etc.

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
