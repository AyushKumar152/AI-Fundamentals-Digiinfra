# AI Fundamentals Challenge – DigiInfra

## Name and Details
Name: Ayush Kumar  
Program: AI Fundamentals & Python Task  
Organization: DigiInfra  

---

## Summary
This project is part of the AI Fundamentals Challenge.  
The goal was to practice Python basics, image processing, data analysis,
data augmentation, and annotation quality checking.

I focused on understanding the concepts clearly and implementing them
step by step rather than producing artificial or forced results.

---

## Task Progress

### Challenge 1: Python Fundamentals
In this challenge, I worked on basic matrix operations without using NumPy
to understand the logic behind array operations.

I also worked with NumPy arrays and loops and learned how Python handles
numerical data. Using OpenCV, I explored how images are represented as
arrays and how basic image processing works.

**What I learned:**
- Difference between Python lists and NumPy arrays  
- Importance of array shapes and data types  

**Difficulties faced:**
- Shape mismatch errors  
- Understanding NumPy broadcasting  

---

### Challenge 2: Data Analysis
In this challenge,
- Analyze the segmentation masks  
- Calculate the class distributions  
- Checked for missing and small objects  

**What I learned:**
- Dataset analysis is important before training any model  
- Clean and well-structured data simplifies later stages  

**Difficulties faced:**
- Converting `.txt` labels into mask images  
- Understanding how class IDs are stored in masks  

---

### Challenge 3: Data Augmentation
In this challenge,
- I implemented image and mask augmentations  
- Applied flip, rotation, crop, and noise  
- Visualized original and augmented images with masks  

**What I learned:**
- Image and mask must always be augmented together  
- Masks require nearest-neighbor interpolation  
- Visualization is essential to verify correctness  

**Difficulties faced:**
- Mask distortion during rotation  
- Saving and displaying correct outputs  

---

### Challenge 4: Annotation Quality Checker
In this challenge,
- Checked annotation boundary smoothness  
- Detected small isolated regions  
- Generated confidence heatmaps  
- Created an HTML report for the dataset  

**What I learned:**
- High-quality annotations can genuinely produce very high confidence scores  
- Artificial noise is not required to make results look realistic  
- Visual overlays are more informative than raw numbers  

**Important note:**  
Some images showed confidence values close to 1.0 because the annotations
were clean polygon-based masks. This is not fake and reflects the actual
quality of the data.

---

## Key Takeaways
- Learned how to work with NumPy and OpenCV  
- Understood segmentation masks and annotations  
- Learned how data quality impacts model performance  
- Gained confidence in reading and debugging real code  

---

## Questions & Reflections
**What was hardest for me?**  
Understanding why annotation confidence was very high and accepting that it
can be correct.

**What surprised me?**  
That clean annotations can naturally give near-perfect confidence values.

**How does this help in real work?**  
It shows the importance of checking annotation quality before training models.

**What do I want to learn next?**
- Training segmentation models  
- Model evaluation metrics  
- Improving annotation tools  

---

## Code Highlights
- `txt_to_mask.py` – Converts `.txt` labels into mask images  
- `seg_augmentation.py` – Performs image and mask augmentation  
- `annotation_quality_checker.py` – Evaluates annotation quality and generates HTML reports  

---

## Final Note
This project focuses on learning and correctness rather than artificial results.  
All outputs are based on actual data behavior.
