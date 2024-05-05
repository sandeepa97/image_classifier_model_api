from django.http import JsonResponse
from django.shortcuts import render
from .ml_model import load_model, classify_image

def classify_image_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        image = request.FILES['image']

        # Load the machine learning model
        model = load_model()

        # Classify the image
        classification_report = classify_image(model, image)

        # Return the classification report as JSON
        return JsonResponse({'classification_report': classification_report})
    else:
        return render(request, 'upload_image.html')  # Render a form to upload an image

