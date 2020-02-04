# Document Cleanup package


The following code contains an approach to smoothing noise in documents, in which the Median Blur technique was used and the OPENcv library. This technique was the one that presented the best result in the space of time and taking into account my the current knowledge to solve this type of problem.


However, when investigating the state of the art, it was found that, with the use of Neural Networks, the quality of the result is much better, taking this into account, I decided to send my own code, even with a limited technique, considering the results of recent approaches that reference. For it to be possible to create an approach that presents better results than those of the state of the art, a longer period of time and an analysis of the others would be necessary so that the wheel was not reinvented. Therefore, I decided to send the references found in the brief bibliographic search. In addition, many variables and the training of the model must be carried out and tested later, so that it was possible to state that an approach made by me would have a real contribution to the scenario.

# Instructions

1 - Create virtual environment, activate and upgrade pip.
```
python3 -m venv venv_challenge
source .venv/bin/activate
pip install -U pip
```
2 - install requiriments and package inside this virtual environment.
```
pip install -r requirements.txt -U
pip install -e .
```
3 - Edit marked sections in /cleanup/main/main.py
4 - Run the main.py script
```
python3 main.py
```
### Requiriments


| Package | Version |
| ------ | ------ |
| opencv-python| 4.1.2.30  |
| numpy | 1.18.1 |


# References found:

https://medium.com/illuin/cleaning-up-dirty-scanned-documents-with-deep-learning-2e8e6de6cfa6


Some approaches: 

These links bring together different approaches with different techniques for solving the proposed problem.
1 - https://www.kaggle.com/c/denoising-dirty-documents/notebooks
2 - https://github.com/ZhiHaoSun/Document-Denoising-Net
