# Real-Time Number Plate Detection"
## 📝Contents:
<details><summary>💡An Introduction about YOLO and darknet.</summary>

- Darknet is an open-source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. You can find the source on the GitHub of the developer Joseph Redmon.

- You only look once (YOLO) is a state-of-the-art, real-time object detection system built on top of Darknet. You can download all the resources at the official website here. YOLO is one the most interesting algorithms for real-time detection since it has only one forward propagation step to make the predictions. Post-Non-Max Suppression then outputs the recognized objects with bounding boxes.

### 🏗️Architecture:

- The Input is a batch of images each reshaped to (608,608,3).
- The image is then divided into 19 by 19 parts and then each of those parts is input into a Deep CNN, we use 5 anchor boxes here.
- We need 5 parameters to detect the object in the box and detect the bounding boxes(p,bx,by,bw,bh), we need 80 different classes to decide which object it is.
- The Output needed is a Numpy array of shape (19,19,5,85)
![Yolo working](https://github.com/Alwyn25/number-plate-detection-real-time/assets/99828232/f66b59a3-1470-46ad-aa6a-f8df84b303d4)

Once we get the output in the shape of (19,19,5,85). The last two dimensions are flattened, so the final output will have the shape of (19,19,425).
</details>


<details><summary>📝Annotation of Images for the training set.</summary>

Before you annotate your training set you need to gather it first. Here I gathered it from some Kaggle datasets, some own pictures, and Google. This same technique of annotating datasets can be used to detect any kind of custom object(although most are already pre-classified in Yolo and here number plate detection is a very specific case).

Once the data is gathered, we use a custom python program called [LabelImg](https://github.com/HumanSignal/labelImg) developed by GitHub user tzutalin. Clear instructions on how to use this are given in the above link of the GitHub page. This software helps you annotate, draw bounding boxes, save the annotation in XML files, which can be processed by our YOLO model, it even has a configuration named YOLO to make things easier. This software automatically creates a configuration file called classes.txt to denote the number of classes. Make sure to remove everything except the class you created for the number plate.

This process is tedious and **has to be done for every Image**. It is boring enough to push most people out of this project but I promise you, it will all be worth it when the final code works.
![labelimg](https://github.com/Alwyn25/number-plate-detection-real-time/assets/99828232/a55e3f82-db1c-4688-ae3a-4a9f02f02cfe)



</details>



<details><summary>⬇️Downloading all the required resources.</summary>

There are a lot of resources on training your Yolo model on a custom object, some of the best resources are [this](https://towardsdatascience.com/how-to-detect-license-plates-with-python-and-yolo-8842aa6d25f7) and [this](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208), the basic gist here is to install darknet and YOLO from pjreddie’s GitHub, build it, and then make changes in configuration and paths to point the model to the custom dataset and train it. I found the research process for this very tedious but thanks to the dark flow created by the GitHub user thtrieu, it all became just one command to implement darknet and Yolo. Although a bit more configuration needed to be done after that to run the number plate detection.

To simplify everything to implement this project, the easiest way would be to download or clone my GitHub repository on the same.
</details>

<details><summary>✔Implementing the code.</summary>

Once the cython extensions are in place and built using one of the above commands.

Please make sure to download the weights from [this link](https://pjreddie.com/media/files/yolov2-tiny.weights) which is taken from the official [yolo webiste](https://pjreddie.com/darknet/yolo/)
```
python -m venv env
```
```
.\env\Scripts\activate
```

```
cd root_directory
```

```
pip install ultralytics==8.0.20
```
```
python img.py
```
This will create images of the inputted videos. You can adjust the mumber of frames to be collected.
```
!pip install labelimg
```
Label all the images with clear numberplates, skip others and save it in yolo files from the application itself

```
python imgdeletetyolo.py
```
- Create a folder named annoted_images and subfolders as images and labels and add train, valid folders to it, then copy all the files from images and paste it in all the valid and train folders. Upload this annoted_images folder and yolov8_object_detection_on_custom_dataset.ipynb to the drive. Run the jupyter file and at the end doenload best.pt from it and add to the project director.
- Run the ipynb files. I had some dependency issues while running it on my personal machine, but it runs perfectly on Google Colab.
- or make use of pyenv python package manager
- Install tesseract, add the root location to web.txt file


```
python main1.py
```
or 

```
python main2.py
```
main1.py stores the number plate data to a txt file and main2.py will store it in sqlite db.

This could be used with yolo for commericial applications instead of tiny yolo and with a lot more training.

That is all
</details>

<details>
  <summary>📁Project Directory</summary>

My project directory is as follows

```
└── 📁carnumberplate
    └── .python-version
    └── 📁annotated_images
        └── 📁images
            └── 📁training
                └── classes copy.txt
                └── classes.txt
                └── numberplate_0.jpg
                └── numberplate_1.jpg
                └── numberplate_1.txt
                └── numberplate_10.jpg
                └── numberplate_100.jpg
                └── numberplate_101.jpg
                └── numberplate_102.jpg
                └── numberplate_103.jpg
                └── numberplate_104.jpg
                └── numberplate_105.jpg
                └── numberplate_11.jpg
                └── numberplate_12.jpg
                └── numberplate_13.jpg
                └── numberplate_14.jpg
                └── numberplate_15.jpg
                └── numberplate_16.jpg
                └── numberplate_17.jpg
                └── numberplate_18.jpg
                └── numberplate_19.jpg
                └── numberplate_2.jpg
                └── numberplate_2.txt
                └── numberplate_20.jpg
                └── numberplate_21.jpg
                └── numberplate_22.jpg
                └── numberplate_23.jpg
                └── numberplate_23.txt
                └── numberplate_24.jpg
                └── numberplate_24.txt
                └── numberplate_25.jpg
                └── numberplate_25.txt
                └── numberplate_26.jpg
                └── numberplate_26.txt
                └── numberplate_27.jpg
                └── numberplate_27.txt
                └── numberplate_28.jpg
                └── numberplate_28.txt
                └── numberplate_29.jpg
                └── numberplate_29.txt
                └── numberplate_3.jpg
                └── numberplate_3.txt
                └── numberplate_30.jpg
                └── numberplate_30.txt
                └── numberplate_31.jpg
                └── numberplate_31.txt
                └── numberplate_32.jpg
                └── numberplate_32.txt
                └── numberplate_33.jpg
                └── numberplate_33.txt
                └── numberplate_34.jpg
                └── numberplate_35.jpg
                └── numberplate_36.jpg
                └── numberplate_37.jpg
                └── numberplate_38.jpg
                └── numberplate_38.txt
                └── numberplate_39.jpg
                └── numberplate_39.txt
                └── numberplate_4.jpg
                └── numberplate_4.txt
                └── numberplate_40.jpg
                └── numberplate_40.txt
                └── numberplate_41.jpg
                └── numberplate_41.txt
                └── numberplate_42.jpg
                └── numberplate_42.txt
                └── numberplate_43.jpg
                └── numberplate_43.txt
                └── numberplate_44.jpg
                └── numberplate_44.txt
                └── numberplate_45.jpg
                └── numberplate_45.txt
                └── numberplate_46.jpg
                └── numberplate_46.txt
                └── numberplate_47.jpg
                └── numberplate_47.txt
                └── numberplate_48.jpg
                └── numberplate_48.txt
                └── numberplate_49.jpg
                └── numberplate_5.jpg
                └── numberplate_5.txt
                └── numberplate_50.jpg
                └── numberplate_51.jpg
                └── numberplate_52.jpg
                └── numberplate_53.jpg
                └── numberplate_54.jpg
                └── numberplate_55.jpg
                └── numberplate_56.jpg
                └── numberplate_57.jpg
                └── numberplate_58.jpg
                └── numberplate_59.jpg
                └── numberplate_59.txt
                └── numberplate_6.jpg
                └── numberplate_6.txt
                └── numberplate_60.jpg
                └── numberplate_60.txt
                └── numberplate_61.jpg
                └── numberplate_61.txt
                └── numberplate_62.jpg
                └── numberplate_62.txt
                └── numberplate_63.jpg
                └── numberplate_63.txt
                └── numberplate_64.jpg
                └── numberplate_64.txt
                └── numberplate_65.jpg
                └── numberplate_65.txt
                └── numberplate_66.jpg
                └── numberplate_66.txt
                └── numberplate_67.jpg
                └── numberplate_67.txt
                └── numberplate_68.jpg
                └── numberplate_68.txt
                └── numberplate_69.jpg
                └── numberplate_69.txt
                └── numberplate_7.jpg
                └── numberplate_7.txt
                └── numberplate_70 copy.txt
                └── numberplate_70.jpg
                └── numberplate_70.txt
                └── numberplate_71.jpg
                └── numberplate_71.txt
                └── numberplate_72.jpg
                └── numberplate_72.txt
                └── numberplate_73.jpg
                └── numberplate_73.txt
                └── numberplate_74.jpg
                └── numberplate_74.txt
                └── numberplate_75.jpg
                └── numberplate_75.txt
                └── numberplate_76.jpg
                └── numberplate_76.txt
                └── numberplate_77.jpg
                └── numberplate_78.jpg
                └── numberplate_79.jpg
                └── numberplate_8.jpg
                └── numberplate_8.txt
                └── numberplate_80.jpg
                └── numberplate_81.jpg
                └── numberplate_82.jpg
                └── numberplate_83.jpg
                └── numberplate_84.jpg
                └── numberplate_85.jpg
                └── numberplate_86.jpg
                └── numberplate_87.jpg
                └── numberplate_88.jpg
                └── numberplate_89.jpg
                └── numberplate_9.jpg
                └── numberplate_9.txt
                └── numberplate_90.jpg
                └── numberplate_91.jpg
                └── numberplate_92.jpg
                └── numberplate_93.jpg
                └── numberplate_94.jpg
                └── numberplate_95.jpg
                └── numberplate_96.jpg
                └── numberplate_97.jpg
                └── numberplate_98.jpg
                └── numberplate_99.jpg
            └── 📁validation
                └── classes copy.txt
                └── classes.txt
                └── numberplate_0.jpg
                └── numberplate_1.jpg
                └── numberplate_1.txt
                └── numberplate_10.jpg
                └── numberplate_100.jpg
                └── numberplate_101.jpg
                └── numberplate_102.jpg
                └── numberplate_103.jpg
                └── numberplate_104.jpg
                └── numberplate_105.jpg
                └── numberplate_11.jpg
                └── numberplate_12.jpg
                └── numberplate_13.jpg
                └── numberplate_14.jpg
                └── numberplate_15.jpg
                └── numberplate_16.jpg
                └── numberplate_17.jpg
                └── numberplate_18.jpg
                └── numberplate_19.jpg
                └── numberplate_2.jpg
                └── numberplate_2.txt
                └── numberplate_20.jpg
                └── numberplate_21.jpg
                └── numberplate_22.jpg
                └── numberplate_23.jpg
                └── numberplate_23.txt
                └── numberplate_24.jpg
                └── numberplate_24.txt
                └── numberplate_25.jpg
                └── numberplate_25.txt
                └── numberplate_26.jpg
                └── numberplate_26.txt
                └── numberplate_27.jpg
                └── numberplate_27.txt
                └── numberplate_28.jpg
                └── numberplate_28.txt
                └── numberplate_29.jpg
                └── numberplate_29.txt
                └── numberplate_3.jpg
                └── numberplate_3.txt
                └── numberplate_30.jpg
                └── numberplate_30.txt
                └── numberplate_31.jpg
                └── numberplate_31.txt
                └── numberplate_32.jpg
                └── numberplate_32.txt
                └── numberplate_33.jpg
                └── numberplate_33.txt
                └── numberplate_34.jpg
                └── numberplate_35.jpg
                └── numberplate_36.jpg
                └── numberplate_37.jpg
                └── numberplate_38.jpg
                └── numberplate_38.txt
                └── numberplate_39.jpg
                └── numberplate_39.txt
                └── numberplate_4.jpg
                └── numberplate_4.txt
                └── numberplate_40.jpg
                └── numberplate_40.txt
                └── numberplate_41.jpg
                └── numberplate_41.txt
                └── numberplate_42.jpg
                └── numberplate_42.txt
                └── numberplate_43.jpg
                └── numberplate_43.txt
                └── numberplate_44.jpg
                └── numberplate_44.txt
                └── numberplate_45.jpg
                └── numberplate_45.txt
                └── numberplate_46.jpg
                └── numberplate_46.txt
                └── numberplate_47.jpg
                └── numberplate_47.txt
                └── numberplate_48.jpg
                └── numberplate_48.txt
                └── numberplate_49.jpg
                └── numberplate_5.jpg
                └── numberplate_5.txt
                └── numberplate_50.jpg
                └── numberplate_51.jpg
                └── numberplate_52.jpg
                └── numberplate_53.jpg
                └── numberplate_54.jpg
                └── numberplate_55.jpg
                └── numberplate_56.jpg
                └── numberplate_57.jpg
                └── numberplate_58.jpg
                └── numberplate_59.jpg
                └── numberplate_59.txt
                └── numberplate_6.jpg
                └── numberplate_6.txt
                └── numberplate_60.jpg
                └── numberplate_60.txt
                └── numberplate_61.jpg
                └── numberplate_61.txt
                └── numberplate_62.jpg
                └── numberplate_62.txt
                └── numberplate_63.jpg
                └── numberplate_63.txt
                └── numberplate_64.jpg
                └── numberplate_64.txt
                └── numberplate_65.jpg
                └── numberplate_65.txt
                └── numberplate_66.jpg
                └── numberplate_66.txt
                └── numberplate_67.jpg
                └── numberplate_67.txt
                └── numberplate_68.jpg
                └── numberplate_68.txt
                └── numberplate_69.jpg
                └── numberplate_69.txt
                └── numberplate_7.jpg
                └── numberplate_7.txt
                └── numberplate_70 copy.txt
                └── numberplate_70.jpg
                └── numberplate_70.txt
                └── numberplate_71.jpg
                └── numberplate_71.txt
                └── numberplate_72.jpg
                └── numberplate_72.txt
                └── numberplate_73.jpg
                └── numberplate_73.txt
                └── numberplate_74.jpg
                └── numberplate_74.txt
                └── numberplate_75.jpg
                └── numberplate_75.txt
                └── numberplate_76.jpg
                └── numberplate_76.txt
                └── numberplate_77.jpg
                └── numberplate_78.jpg
                └── numberplate_79.jpg
                └── numberplate_8.jpg
                └── numberplate_8.txt
                └── numberplate_80.jpg
                └── numberplate_81.jpg
                └── numberplate_82.jpg
                └── numberplate_83.jpg
                └── numberplate_84.jpg
                └── numberplate_85.jpg
                └── numberplate_86.jpg
                └── numberplate_87.jpg
                └── numberplate_88.jpg
                └── numberplate_89.jpg
                └── numberplate_9.jpg
                └── numberplate_9.txt
                └── numberplate_90.jpg
                └── numberplate_91.jpg
                └── numberplate_92.jpg
                └── numberplate_93.jpg
                └── numberplate_94.jpg
                └── numberplate_95.jpg
                └── numberplate_96.jpg
                └── numberplate_97.jpg
                └── numberplate_98.jpg
                └── numberplate_99.jpg
        └── 📁labels
            └── 📁training
                └── classes copy.txt
                └── classes.txt
                └── numberplate_0.jpg
                └── numberplate_1.jpg
                └── numberplate_1.txt
                └── numberplate_10.jpg
                └── numberplate_100.jpg
                └── numberplate_101.jpg
                └── numberplate_102.jpg
                └── numberplate_103.jpg
                └── numberplate_104.jpg
                └── numberplate_105.jpg
                └── numberplate_11.jpg
                └── numberplate_12.jpg
                └── numberplate_13.jpg
                └── numberplate_14.jpg
                └── numberplate_15.jpg
                └── numberplate_16.jpg
                └── numberplate_17.jpg
                └── numberplate_18.jpg
                └── numberplate_19.jpg
                └── numberplate_2.jpg
                └── numberplate_2.txt
                └── numberplate_20.jpg
                └── numberplate_21.jpg
                └── numberplate_22.jpg
                └── numberplate_23.jpg
                └── numberplate_23.txt
                └── numberplate_24.jpg
                └── numberplate_24.txt
                └── numberplate_25.jpg
                └── numberplate_25.txt
                └── numberplate_26.jpg
                └── numberplate_26.txt
                └── numberplate_27.jpg
                └── numberplate_27.txt
                └── numberplate_28.jpg
                └── numberplate_28.txt
                └── numberplate_29.jpg
                └── numberplate_29.txt
                └── numberplate_3.jpg
                └── numberplate_3.txt
                └── numberplate_30.jpg
                └── numberplate_30.txt
                └── numberplate_31.jpg
                └── numberplate_31.txt
                └── numberplate_32.jpg
                └── numberplate_32.txt
                └── numberplate_33.jpg
                └── numberplate_33.txt
                └── numberplate_34.jpg
                └── numberplate_35.jpg
                └── numberplate_36.jpg
                └── numberplate_37.jpg
                └── numberplate_38.jpg
                └── numberplate_38.txt
                └── numberplate_39.jpg
                └── numberplate_39.txt
                └── numberplate_4.jpg
                └── numberplate_4.txt
                └── numberplate_40.jpg
                └── numberplate_40.txt
                └── numberplate_41.jpg
                └── numberplate_41.txt
                └── numberplate_42.jpg
                └── numberplate_42.txt
                └── numberplate_43.jpg
                └── numberplate_43.txt
                └── numberplate_44.jpg
                └── numberplate_44.txt
                └── numberplate_45.jpg
                └── numberplate_45.txt
                └── numberplate_46.jpg
                └── numberplate_46.txt
                └── numberplate_47.jpg
                └── numberplate_47.txt
                └── numberplate_48.jpg
                └── numberplate_48.txt
                └── numberplate_49.jpg
                └── numberplate_5.jpg
                └── numberplate_5.txt
                └── numberplate_50.jpg
                └── numberplate_51.jpg
                └── numberplate_52.jpg
                └── numberplate_53.jpg
                └── numberplate_54.jpg
                └── numberplate_55.jpg
                └── numberplate_56.jpg
                └── numberplate_57.jpg
                └── numberplate_58.jpg
                └── numberplate_59.jpg
                └── numberplate_59.txt
                └── numberplate_6.jpg
                └── numberplate_6.txt
                └── numberplate_60.jpg
                └── numberplate_60.txt
                └── numberplate_61.jpg
                └── numberplate_61.txt
                └── numberplate_62.jpg
                └── numberplate_62.txt
                └── numberplate_63.jpg
                └── numberplate_63.txt
                └── numberplate_64.jpg
                └── numberplate_64.txt
                └── numberplate_65.jpg
                └── numberplate_65.txt
                └── numberplate_66.jpg
                └── numberplate_66.txt
                └── numberplate_67.jpg
                └── numberplate_67.txt
                └── numberplate_68.jpg
                └── numberplate_68.txt
                └── numberplate_69.jpg
                └── numberplate_69.txt
                └── numberplate_7.jpg
                └── numberplate_7.txt
                └── numberplate_70 copy.txt
                └── numberplate_70.jpg
                └── numberplate_70.txt
                └── numberplate_71.jpg
                └── numberplate_71.txt
                └── numberplate_72.jpg
                └── numberplate_72.txt
                └── numberplate_73.jpg
                └── numberplate_73.txt
                └── numberplate_74.jpg
                └── numberplate_74.txt
                └── numberplate_75.jpg
                └── numberplate_75.txt
                └── numberplate_76.jpg
                └── numberplate_76.txt
                └── numberplate_77.jpg
                └── numberplate_78.jpg
                └── numberplate_79.jpg
                └── numberplate_8.jpg
                └── numberplate_8.txt
                └── numberplate_80.jpg
                └── numberplate_81.jpg
                └── numberplate_82.jpg
                └── numberplate_83.jpg
                └── numberplate_84.jpg
                └── numberplate_85.jpg
                └── numberplate_86.jpg
                └── numberplate_87.jpg
                └── numberplate_88.jpg
                └── numberplate_89.jpg
                └── numberplate_9.jpg
                └── numberplate_9.txt
                └── numberplate_90.jpg
                └── numberplate_91.jpg
                └── numberplate_92.jpg
                └── numberplate_93.jpg
                └── numberplate_94.jpg
                └── numberplate_95.jpg
                └── numberplate_96.jpg
                └── numberplate_97.jpg
                └── numberplate_98.jpg
                └── numberplate_99.jpg
            └── 📁validation
                └── classes copy.txt
                └── classes.txt
                └── numberplate_0.jpg
                └── numberplate_1.jpg
                └── numberplate_1.txt
                └── numberplate_10.jpg
                └── numberplate_100.jpg
                └── numberplate_101.jpg
                └── numberplate_102.jpg
                └── numberplate_103.jpg
                └── numberplate_104.jpg
                └── numberplate_105.jpg
                └── numberplate_11.jpg
                └── numberplate_12.jpg
                └── numberplate_13.jpg
                └── numberplate_14.jpg
                └── numberplate_15.jpg
                └── numberplate_16.jpg
                └── numberplate_17.jpg
                └── numberplate_18.jpg
                └── numberplate_19.jpg
                └── numberplate_2.jpg
                └── numberplate_2.txt
                └── numberplate_20.jpg
                └── numberplate_21.jpg
                └── numberplate_22.jpg
                └── numberplate_23.jpg
                └── numberplate_23.txt
                └── numberplate_24.jpg
                └── numberplate_24.txt
                └── numberplate_25.jpg
                └── numberplate_25.txt
                └── numberplate_26.jpg
                └── numberplate_26.txt
                └── numberplate_27.jpg
                └── numberplate_27.txt
                └── numberplate_28.jpg
                └── numberplate_28.txt
                └── numberplate_29.jpg
                └── numberplate_29.txt
                └── numberplate_3.jpg
                └── numberplate_3.txt
                └── numberplate_30.jpg
                └── numberplate_30.txt
                └── numberplate_31.jpg
                └── numberplate_31.txt
                └── numberplate_32.jpg
                └── numberplate_32.txt
                └── numberplate_33.jpg
                └── numberplate_33.txt
                └── numberplate_34.jpg
                └── numberplate_35.jpg
                └── numberplate_36.jpg
                └── numberplate_37.jpg
                └── numberplate_38.jpg
                └── numberplate_38.txt
                └── numberplate_39.jpg
                └── numberplate_39.txt
                └── numberplate_4.jpg
                └── numberplate_4.txt
                └── numberplate_40.jpg
                └── numberplate_40.txt
                └── numberplate_41.jpg
                └── numberplate_41.txt
                └── numberplate_42.jpg
                └── numberplate_42.txt
                └── numberplate_43.jpg
                └── numberplate_43.txt
                └── numberplate_44.jpg
                └── numberplate_44.txt
                └── numberplate_45.jpg
                └── numberplate_45.txt
                └── numberplate_46.jpg
                └── numberplate_46.txt
                └── numberplate_47.jpg
                └── numberplate_47.txt
                └── numberplate_48.jpg
                └── numberplate_48.txt
                └── numberplate_49.jpg
                └── numberplate_5.jpg
                └── numberplate_5.txt
                └── numberplate_50.jpg
                └── numberplate_51.jpg
                └── numberplate_52.jpg
                └── numberplate_53.jpg
                └── numberplate_54.jpg
                └── numberplate_55.jpg
                └── numberplate_56.jpg
                └── numberplate_57.jpg
                └── numberplate_58.jpg
                └── numberplate_59.jpg
                └── numberplate_59.txt
                └── numberplate_6.jpg
                └── numberplate_6.txt
                └── numberplate_60.jpg
                └── numberplate_60.txt
                └── numberplate_61.jpg
                └── numberplate_61.txt
                └── numberplate_62.jpg
                └── numberplate_62.txt
                └── numberplate_63.jpg
                └── numberplate_63.txt
                └── numberplate_64.jpg
                └── numberplate_64.txt
                └── numberplate_65.jpg
                └── numberplate_65.txt
                └── numberplate_66.jpg
                └── numberplate_66.txt
                └── numberplate_67.jpg
                └── numberplate_67.txt
                └── numberplate_68.jpg
                └── numberplate_68.txt
                └── numberplate_69.jpg
                └── numberplate_69.txt
                └── numberplate_7.jpg
                └── numberplate_7.txt
                └── numberplate_70 copy.txt
                └── numberplate_70.jpg
                └── numberplate_70.txt
                └── numberplate_71.jpg
                └── numberplate_71.txt
                └── numberplate_72.jpg
                └── numberplate_72.txt
                └── numberplate_73.jpg
                └── numberplate_73.txt
                └── numberplate_74.jpg
                └── numberplate_74.txt
                └── numberplate_75.jpg
                └── numberplate_75.txt
                └── numberplate_76.jpg
                └── numberplate_76.txt
                └── numberplate_77.jpg
                └── numberplate_78.jpg
                └── numberplate_79.jpg
                └── numberplate_8.jpg
                └── numberplate_8.txt
                └── numberplate_80.jpg
                └── numberplate_81.jpg
                └── numberplate_82.jpg
                └── numberplate_83.jpg
                └── numberplate_84.jpg
                └── numberplate_85.jpg
                └── numberplate_86.jpg
                └── numberplate_87.jpg
                └── numberplate_88.jpg
                └── numberplate_89.jpg
                └── numberplate_9.jpg
                └── numberplate_9.txt
                └── numberplate_90.jpg
                └── numberplate_91.jpg
                └── numberplate_92.jpg
                └── numberplate_93.jpg
                └── numberplate_94.jpg
                └── numberplate_95.jpg
                └── numberplate_96.jpg
                └── numberplate_97.jpg
                └── numberplate_98.jpg
                └── numberplate_99.jpg
    └── annotated_images.zip
    └── best.pt
    └── car_plate_data.db
    └── car_plate_data.txt
    └── car_plate_database.db
    └── coco1.txt
    └── data.txt
    └── 📁images
        └── classes copy.txt
        └── classes.txt
        └── numberplate_1.jpg
        └── numberplate_1.txt
        └── numberplate_2.jpg
        └── numberplate_2.txt
        └── numberplate_23.jpg
        └── numberplate_23.txt
        └── numberplate_24.jpg
        └── numberplate_24.txt
        └── numberplate_25.jpg
        └── numberplate_25.txt
        └── numberplate_26.jpg
        └── numberplate_26.txt
        └── numberplate_27.jpg
        └── numberplate_27.txt
        └── numberplate_28.jpg
        └── numberplate_28.txt
        └── numberplate_29.jpg
        └── numberplate_29.txt
        └── numberplate_3.jpg
        └── numberplate_3.txt
        └── numberplate_30.jpg
        └── numberplate_30.txt
        └── numberplate_31.jpg
        └── numberplate_31.txt
        └── numberplate_32.jpg
        └── numberplate_32.txt
        └── numberplate_33.jpg
        └── numberplate_33.txt
        └── numberplate_38.jpg
        └── numberplate_38.txt
        └── numberplate_39.jpg
        └── numberplate_39.txt
        └── numberplate_4.jpg
        └── numberplate_4.txt
        └── numberplate_40.jpg
        └── numberplate_40.txt
        └── numberplate_41.jpg
        └── numberplate_41.txt
        └── numberplate_42.jpg
        └── numberplate_42.txt
        └── numberplate_43.jpg
        └── numberplate_43.txt
        └── numberplate_44.jpg
        └── numberplate_44.txt
        └── numberplate_45.jpg
        └── numberplate_45.txt
        └── numberplate_46.jpg
        └── numberplate_46.txt
        └── numberplate_47.jpg
        └── numberplate_47.txt
        └── numberplate_48.jpg
        └── numberplate_48.txt
        └── numberplate_5.jpg
        └── numberplate_5.txt
        └── numberplate_59.jpg
        └── numberplate_59.txt
        └── numberplate_6.jpg
        └── numberplate_6.txt
        └── numberplate_60.jpg
        └── numberplate_60.txt
        └── numberplate_61.jpg
        └── numberplate_61.txt
        └── numberplate_62.jpg
        └── numberplate_62.txt
        └── numberplate_63.jpg
        └── numberplate_63.txt
        └── numberplate_64.jpg
        └── numberplate_64.txt
        └── numberplate_65.jpg
        └── numberplate_65.txt
        └── numberplate_66.jpg
        └── numberplate_66.txt
        └── numberplate_67.jpg
        └── numberplate_67.txt
        └── numberplate_68.jpg
        └── numberplate_68.txt
        └── numberplate_69.jpg
        └── numberplate_69.txt
        └── numberplate_7.jpg
        └── numberplate_7.txt
        └── numberplate_70 copy.txt
        └── numberplate_70.jpg
        └── numberplate_70.txt
        └── numberplate_71.jpg
        └── numberplate_71.txt
        └── numberplate_72.jpg
        └── numberplate_72.txt
        └── numberplate_73.jpg
        └── numberplate_73.txt
        └── numberplate_74.jpg
        └── numberplate_74.txt
        └── numberplate_75.jpg
        └── numberplate_75.txt
        └── numberplate_76.jpg
        └── numberplate_76.txt
        └── numberplate_8.jpg
        └── numberplate_8.txt
        └── numberplate_9.jpg
        └── numberplate_9.txt
    └── img.py
    └── imgdeletetyolo.py
    └── main1.py
    └── main2.py
    └── mycarplate.mp4
    └── README.md
    └── requirements.txt
    └── tesseract_path.txt
    └── test.py
    └── Test1.mp4
    └── web.txt
    └── yolov8n.pt
    └── yolov8s.pt
    └── yolov8_object_detection_on_custom_dataset.ipynb
    └── 📁__pycache__
        └── centroid_tracker.cpython-37.pyc
```
</details>

<details><summary>🏁 Conclusion and References.</summary>

  Using this method Basic Object detection and recognition will become a breeze, this particular application can be used to simplify tasks such as allowing vehicles in restricted areas. Feel free to train in more and optimize it further.
</details>

