# Waste Classification (CNN) - Vercel ready

Classes: cardboard, glass, metal, paper, plastic, trash.

## 1. Train locally (one time)
    pip install -r requirements-train.txt
    # put images in dataset/<class_name>/ folders
    python train.py            # creates models/waste_classifier.keras + class_names.json
    python convert_to_onnx.py  # creates models/waste_classifier.onnx

## 2. Test locally
    pip install -r requirements.txt
    python api/index.py        # http://127.0.0.1:5000

## 3. Deploy to Vercel
    npm i -g vercel
    vercel login
    vercel --prod
Or push to GitHub and import the repo in the Vercel dashboard.
Make sure models/waste_classifier.onnx and models/class_names.json are committed.
