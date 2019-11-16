# My solution and learning process

Both problems were quite challenging for me. Especially because it was only 7 days to solve both of them. Although I found the problems quite challenging I need to be frank and say that I loved the opportunity to have tried a solution to the problems.
Despite the fact I have experience with Machine Learning and Deep Learning, I had not yet worked with image processing. During these 7 days I was able to learn how to process image and I'm very excited to be part of the NUVEO team and improve my skills on those topics.

My name is **Jefferson Martins** and below you find more information about how I managed to solve both tasks and how I think it could be improved. I hope you enjoy. 


# Document Cleanup

During the week I tried to find image processing solutions to rotate images that were not well centered or horizontally. I also looked for solutions to detect text only and delete background that is not part of the text.

My solution for image rotation is 100% effective when the background doesn't contain much noise. However it loses efficiency in cases where the background has colors very similar to the text of the image. 

In order to eliminate the background, my solution was more efficient than trying to rotate all images. Here I can almost completely exclude all noise from images. The result can be seen in the folder **output cleanup**.

One of the techniques that is not in my solution but I also noticed a lot of efficiency was edge detection.  Although my edge detection result was quite efficient, I preferred not to include it in my solution because I realized that some more processing was still required to have the desired result.

I thought about using some machine learning technique to identify what is letters and what is background. Once I had identified the letters and background I could remove only the background and leave only the letters. I chose not to follow this methodology because I am not aware of the algorithms that perform what I had in mind and I was afraid to not find a good solution using ML until the deadline. Therefore,  I chose to continue doing image processing only, since I knew that at the end of the deadline I would have at least one good result to display.


# Fraud Detection

 - The dataset provided not include the **test set**.
By reading the problem description I easily identified that this problem would easily be solved using some ML technique. The first looked for ways to extract signature features from images. I read some articles that deal with the same problem I was trying to solve and selected some features that I wanted to extract from the images. The features are as follows:
	 - Number of holes
	 - Curvature of the signature
	 - Number of holes
	 - Number of connected points
	 - 	Position of barycenter

I searched for ways of how to extract the features mentioned above. It was a few hours of reading and some attempts at feature extraction. Due to the time restriction I decided not to extract those features and try a deep learning solution.
I opted to use deep learning because I knew that a neural network is capable of identifying important features automatically. Since I didn't have enough time to learn how to extract all the features I had proposed,  I used a neural network to automatically identify important features.

Because the dataset does not have the **test set**, the way I evaluated my solution was based on the reduction of  the loss in each batch. More information about the parameters I used and the NN I used are written in the python script. My model had a loss of 0.06 or less as it can be seen in the **result.png**.


# How to run the code
I have two python files: **signature_detection.py** and **clean_images.py**.
Download the dataset and put the Python scripts in the correct place as it is described below.
## Running 01-DocumentCleanup
The **clean_images.py** should be inside the **01-DocumentCleanup**.
See the image below:
![See](https://i.ibb.co/CmKKdGZ/image1.png)
Run in your terminal
``` Python clean_images.py```
One folder called  **cleaned_images** will be created with all the images cleaned.

## Running 02-FraudDetection
The **signature_detection.py** should be inside the **02-FraudDetection**.
See the image below:
![see](https://i.ibb.co/zmT0R96/image1.png)

Run in your terminal
``` python signature_detection.py```
On your terminal you'll see the loss and accuracy of the model as the image below.
![see](https://i.ibb.co/NLDwcWP/result.png)

## System setup
There is a file called **requirements.txt** with all the Python lib dependencies. 
run the following and install all the requirements:
```pip install -r requirements.txt```

Once you have all the requirements you are ready to run **signature_detection.py** and **clean_images.py** scripts.

## Final remarks
I wrote a lot but I still know that there are more to be said. Maybe in the next interview I have the opportunity to explain anything which is not clear and/or why I did not use this or that methodology. For any questions please send me an email (jefferson.sousa@ccc.ufcg.edu.br).


Thank you so much for the opportunity and finger crossed. I hope to be part of the NUVEO team soon. 
