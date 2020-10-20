# Document Cleanup

This is the algorithm to preprocessing a image before to use a OCR algorithm.


# How to run?  

1 - Install the Docker and run these commands:

- docker build . -t document_cleanup:v1
- docker run -it --name cont_1 document_cleanup:v1 

2 - **Run script**

- python main.py -p "noisy_data"


