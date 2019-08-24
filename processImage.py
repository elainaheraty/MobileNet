from IPython.display import Image
Image(filename='Users/elainaheraty/pictures/1.PNG', width=300, height=200)

preprocessed_image = prepare_image('1.PNG')
predictions = mobile.predict(preprocessed_image)

#imagenet utility function which returns top 5 imagenet class predictions + class ID, class label and probability
results = imagenet_utils.decode_preidctions(predictions)
results
#
#test on jupyter notebooks
#next create mobilenet that works without pretrained bot yeahh
