import cv2
from utils.predict import Predict

"""
    A script to run train and test modules of project.
"""
######################### Evaluate train ##################################################

# train = Train()
# train.read_data_and_create_train_and_test()
# model = train.create_model()
# train.train_model(model)

######################### Evaluate predict #################################################

path = r''
image = cv2.imread(path)
predict = Predict()
predict.load_model()
result = predict.is_card_valid(image)
if result[0]:
    startX = int(result[1][0])
    startY = int(result[1][1])
    endX = int(result[1][2])
    endY = int(result[1][3])
    # draw the predicted bounding box and class label on the image
    cv2.rectangle(image, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    r = predict.crop_image(image, startX, startY, endX, endY)
    cv2.imshow("Output", r)
    cv2.waitKey(0)
else:
    print(result[0])
