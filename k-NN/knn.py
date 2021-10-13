# # coding='utf-8'
# __author__ = 'fang'
#
# # import the necessary packages
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
# from pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
# from imutils import paths
# import argparse
#
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(
#     r'C:\Users\Administrator.DESKTOP-DHPJ48Q\Desktop\Limingming\Deep-Learning-For-Computer-Vision-master\datasets\animals'))
# print(imagePaths)
# # initialize the image preprocessor, load the dataset from disk,  and reshape the data matrix #初始化图像预处理程序，从磁盘加载数据集，并重塑数据矩阵
# sp = SimplePreprocessor(32, 32)
# sdl = SimpleDatasetLoader(preprocessors=[sp])
# (data, labels) = sdl.load(imagePaths, verbose=500)
# data = data.reshape((data.shape[0], 3072))
# print("[INFO] features matrix: {:.1f}MB".format(
#     data.nbytes / (1024 * 1000.0)))
# le = LabelEncoder()
# labels = le.fit_transform(labels)
# print(labels)
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
#
# print("[INFO] evaluating k-NN classifier...")
# Knn = KNeighborsClassifier()
# Knn.fit(trainX, trainY)
# print('k-NN分类器')
# print(classification_report(testY, Knn.predict(testX), target_names=le.classes_))


# from xgboost import XGBClassifier
#
# xgb = XGBClassifier()
# xgb.fit(trainX, trainY)
# print('XGBOOST分类器')
# print(classification_report(testY, xgb.predict(testX),target_names=le.classes_))

# def var_name(var,all_var=locals()):
#     return [var_name for var_name in all_var if all_var[var_name] is var][0]
#
# 方轩豪 = 10
# print(locals())
# # print(var_name(方轩豪))

