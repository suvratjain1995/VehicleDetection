def read_car(image_path):
    import glob
    import cv2
    images_png = glob.glob(image_path[0])
    print(len(images_png))
    image_list =[]
    for images in images_png:
        temp = cv2.imread(images)
        temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
        image_list.append(temp)
    
    PIK = 'car.pickle'
    import pickle
    with open(PIK,"wb") as f:
        pickle.dump(image_list,f)

def read_non_car(image_path):
    import glob
    import cv2
    images_png = glob.glob(image_path[0])
    print(len(images_png))
    image_list =[]
    for images in images_png:
        temp = cv2.imread(images)
        temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
        image_list.append(temp)
    
    PIK = 'nocar.pickle'
    import pickle
    with open(PIK,"wb") as f:
        pickle.dump(image_list,f)


car_image_path = ['vehicles/vehicles/*/*.png','vehicles/vehicles/*/*.jpg']
non_car_image_path = ['non-vehicles/non-vehicles/*/*.png','non-vehicles/non-vehicles/*/*.jpg']

def __init__():
    car_image_path = ['vehicles/vehicles/*/*.png','vehicles/vehicles/*/*.jpg']
    non_car_image_path = ['non-vehicles/non-vehicles/*/*.png','non-vehicles/non-vehicles/*/*.jpg']
    read_car(car_image_path)
    read_non_car(non_car_image_path)

read_car(car_image_path)
read_non_car(non_car_image_path)