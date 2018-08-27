For each image I made a script, then just run the script and the output will be saved on a .txt.

1. TO RUN
    1.1 Dependencies:
            python 3.6
            tesseract 4.0 (windows version)
            run "pip install -r requeriments.txt" to install other dependencies

    1.2 Extract text
        python testx.py img_path txt_path tesseract_path
        Run: Example for test2
            python test2.py images\test2.jpg test2.txt "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

    1.3 Old names:
        coroatá-3.jpg -> images\test1.jpg
        emory1877_0002.jpg -> images\test2.jpg
        3320114_2_0002.jpg -> images\test3.jpg

