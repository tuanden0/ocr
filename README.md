## OCR

### How to run?
```bash
# Install required package
sudo apt-get update && sudo apt-get install tesseract-ocr

# Install python package
pip install -r requirements.txt

# Run action_ezocr.py
python action_ezocr.py

# Run action_tess.py
python action_tess.py
```

### Note

- This repo `does not contains` code for `generated pre-processed img`

### Problems

- Even data was pre-processed, `easyocr` and `pytesseract` can not return good results for Vietnamese subtitles.

### Dataset/Raw

- The data source in [dataset/raw](dataset/raw) was randomly selected and retrieved from [Youtube - MuseVN - Data source](https://www.youtube.com/watch?v=7s2j8fUaK04&list=PLdM751AKK4aPJHQ0Fgq__AMe_VmO7sDVE&index=1)

- This directory contains raw subtitle image from the data source and named base on the data source duration.

### Dataset/Pre-Process

- This directory contains processed image files that have had unrelated objects removed and have been checked for subtitles.

### Results/easyocr

- This directory contains all text results after run [action_ezocr.py](action_ezocr.py)

### Results/pytesseract

- This directory contains all text results after run [action_tess.py](action_tess.py)