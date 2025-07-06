from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__, template_folder='Templates')

# Load trained poultry disease model
model = load_model("Poultry_Disease.h5")

# Verify model architecture
print("\n=== MODEL ARCHITECTURE ===")
model.summary()
print(f"Output shape: {model.output_shape}")

# Poultry disease classes - MUST match model's training classes
classes = [
    'Newcastle Disease',
    'Salmonella',
    'Avian Influenza',
    'Healthy'
]

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    try:
        print(f"\n🔍 Processing: {img_path}")

        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get model prediction
        predictions = model.predict(img_array, verbose=0)
        print(f"Raw model output: {predictions}")

        # Handle different model output types
        if len(predictions[0]) == 1:  # Binary classification
            confidence = float(predictions[0][0])
            class_idx = 0 if confidence > 0.5 else 1
            confidence = confidence if class_idx == 0 else 1 - confidence
            result = {
                'class': classes[class_idx] if class_idx < len(classes) else "Unknown",
                'confidence': confidence * 100,
                'all_predictions': {
                    classes[0] if len(classes) > 0 else "Class_0": confidence*100,
                    classes[1] if len(classes) > 1 else "Class_1": (1-confidence)*100
                }
            }
        else:  # Multi-class classification
            prediction = predictions[0]
            top_class = np.argmax(prediction)
            confidence = prediction[top_class]
            
            # Build results with safe class indexing
            all_preds = {}
            for i, prob in enumerate(prediction):
                class_name = classes[i] if i < len(classes) else f"Class_{i}"
                all_preds[class_name] = float(prob)
            
            result = {
                'class': classes[top_class] if top_class < len(classes) else "Unknown",
                'confidence': float(confidence*100),
                'all_predictions': all_preds
            }

        # Debug output
        print("\n✅ Prediction results:")
        for class_name, prob in result['all_predictions'].items():
            print(f"{class_name:20s}: {prob:.2f}%")
        print(f"🎯 Final prediction: {result['class']} ({result['confidence']:.2f}% confidence)")

        return result
    
    except Exception as e:
        print(f"\n❌ Prediction error: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}

# Flask routes
@app.route('/')
def home():  # Changed from 'index' to 'home' to match template url_for('home')
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('predict.html', error="No file selected")
            
        file = request.files['file']
        
        # Validate file
        if file.filename == '':
            return render_template('predict.html', error="No file selected")
            
        if not allowed_file(file.filename):
            return render_template('predict.html', error="Invalid file type (only JPG, PNG allowed)")

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            print("\n" + "="*50)
            print(f"🖼️ Processing new image: {filename}")
            print("="*50)

            # Get prediction
            result = predict_disease(save_path)
            
            if 'error' in result:
                return render_template('predict.html', error=result['error'])
                
            img_url = f"/static/uploads/{filename}"
            
            return render_template(
                'predict.html',
                prediction=result['class'],
                confidence=result['confidence'],
                img_path=img_url,
                all_predictions=result['all_predictions']
            )
            
        except Exception as e:
            print(f"Upload processing error: {str(e)}")
            return render_template('predict.html', error=f"Processing error: {str(e)}")

    return render_template('predict.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("\n✅ Poultry Disease Classifier starting...")
    print(f"Model expects {model.output_shape[-1]} output classes")
    print(f"Defined classes: {classes}")
    app.run(debug=True)
