from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedImage
from .models import UploadedImageSerializer
import ml_model

class UploadImageView(APIView):
    def post(self, request, format=None):
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            # Save the uploaded image
            uploaded_image = serializer.save()

            # Call ml_model.py with the file path
            file_path = uploaded_image.image.path
            X_train, y_train, X_test, y_test = ml_model.load_data(file_path)
            X_train, X_test = ml_model.preprocess_data(X_train, X_test)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(X_train)
            model = ml_model.create_model(X_train.shape[1:])
            history = ml_model.train_model(model, X_train, y_train, datagen, X_val, y_val)
            report, matrix = ml_model.evaluate_model(model, X_test, y_test)

            return Response({'file_path': file_path, 'classification_report': report, 'confusion_matrix': matrix}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
