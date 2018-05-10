import csv
import numpy as np
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow

def load_data(path, from_= 0):
    results = []
    inputs = []
    with open(path) as test_data_file:
        reader = csv.reader(test_data_file)
        next(reader)

        for row in reader:
            row = map(int, row)
            
            result = row[0]

            input = row[from_:]

            inputs.append(input)
            results.append(result)

    return [np.asarray(inputs), np.asarray(results)]

def printImgFromVector(img_vector):
    img_array = img_vector.reshape((28, 28)).astype('uint8')
    img = Image.fromarray(img_array).convert('LA')
    imshow(np.asarray(img))

def export_to_submission(network):
    with open('data/submission.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['ImageId', 'Label'])
        for i in range(28000):
            test_results = network.feedforward(test_data[i][0])
            digit_decision = net.test_mb_predictions(i/mini_batch_size)[i%mini_batch_size]
            csvwriter.writerow([i+1, digit_decision])