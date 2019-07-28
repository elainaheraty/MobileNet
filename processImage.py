from IPython.display import Image
Image(filename='MobileNet-images/1.PNG', width=300, height=200)

preprocessed_image = prepare_image('1.PNG')
predictions = mobile.predict(preprocessed_image)

#imagenet utility function which returns top 5 imagenet class predictions + class ID, class label and probability
results = imagenet_utils.decode_preidctions(predictions)
results
