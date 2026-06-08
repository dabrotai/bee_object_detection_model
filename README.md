# bee_object_detection_model
This project uses Yolov8 Nano to train an object detection model to identify and classify Varroa mites, worker bees, drone bees, and queen bees from images. 

| Notebook Workflow | Quick Launch |
| :--- | :--- |
| **Model Training & Testing** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kDZoi9cbpZovVKPtpt-2K7FreXhrUEtz#scrollTo=JxTW8ysITaOF) |
| **Automated Data Labelling** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pRj0aQYxKxJJvLVgYuteawr8BfOor37L?usp=sharing) |
## Dataset
```python
from roboflow import Roboflow

rf = Roboflow(api_key="Aq******************")
project = rf.workspace("tainuis-workspace").project("bee-project-upcez")
version = project.version(1)
dataset = version.download("yolov8")
```
## Model training
```python
model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    batch=4,
    lr0=0.01,
    dropout=0.20,
    optimizer="RMSProp",
    visualize=True
)
```

## Optimizer Comparison Results

Below is the side-by-side training performance metrics for **AdamW**, **RMSProp**, and **SGD** across 50 epochs. 

## Model Performance & Optimizer Comparison

<table border="0" width="100%">
<!-- Row 1: Training Curves Images -->
<tr>
<td align="center" valign="top" width="33%">
<strong>AdamW Training Curves</strong><br><br>
<img src="models/AdamW/results.png" alt="AdamW Results" width="100%">
</td>
<td align="center" valign="top" width="33%">
<strong>RMSProp Training Curves</strong><br><br>
<img src="models/RMSProp/results.png" alt="RMSProp Results" width="100%">
</td>
<td align="center" valign="top" width="33%">
<strong>SGD Training Curves</strong><br><br>
<img src="models/SGD/results.png" alt="SGD Results" width="100%">
</td>
</tr>

<!-- Row 2: Numerical Metric Tables -->
<tr>
<td valign="top">


| Metric | Value |
| :--- | :---: |
| Precision | 0.8588 |
| Recall | 0.8513 |
| mAP@50 | 0.8948 |
| mAP@50-95 | 0.6118 |

</td>
<td valign="top">


| Metric | Value |
| :--- | :---: |
| Precision | 0.9138 |
| Recall | 0.8795 |
| mAP@50 | 0.9140 |
| mAP@50-95 | 0.6388 |

</td>
<td valign="top">


| Metric | Value |
| :--- | :---: |
| Precision | 0.9211 |
| Recall | 0.8782 |
| mAP@50 | 0.9135 |
| mAP@50-95 | 0.6429 |

</td>
</tr>
</table>


</th>
</tr>
</table>

## Object Counting (Cell 14)
This script processes the model's inference results across your test dataset split. It iterates through each image, extracts the predicted class IDs, and aggregates them to output a precise count of each object type detected 
````python
from collections import Counter

for r in results:
    print(f"\nImage: {r.path}")

    if r.boxes is not None and len(r.boxes) > 0:
        classes = r.boxes.cls.cpu().numpy().astype(int)

        counts = Counter(classes)

        for cls_id, count in counts.items():
            class_name = model.names[cls_id]
            print(f"{class_name}: {count}")
    else:
        print("No detections")
---------------------------------
Image: /content/bee_object_detection_model/bee-project-1/test/images/img_106_jpg.rf.240909ef60b7a641ce270c133e989b0f.jpg
worker: 1

Image: /content/bee_object_detection_model/bee-project-1/test/images/img_110_jpg.rf.5918bf33883495ca109dc9ed84f50f6f.jpg
drone: 1

Image: /content/bee_object_detection_model/bee-project-1/test/images/img_112_jpg.rf.31a7a3edc22bd8962688eae99c3e029b.jpg
```
