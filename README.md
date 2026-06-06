# bee_object_detection_model
This project uses Yolov8 Nano to train an object detection model to identify and classify Varroa mites, worker bees, drone bees, and queen bees from images. 

Dataset: 
```python
from roboflow import Roboflow

rf = Roboflow(api_key="Aq******************")
project = rf.workspace("tainuis-workspace").project("bee-project-upcez")
version = project.version(1)
dataset = version.download("yolov8")
```

## Optimizer Comparison Results

Below is the side-by-side training performance metrics for **AdamW**, **RMSProp**, and **SGD** across 50 epochs. 

| AdamW Results | RMSProp Results | SGD Results |
| :---: | :---: | :---: |
| ![](models/AdamW/results.png) | ![](models/RMSProp/results.png) | ![](models/SGD/results.png) |

### Model Performance 
<table border="0">
<tr>
<th valign="top">


| AdamW Results | Value |
| :--- | :---: |
| Precision | 0.8588 |
| Recall | 0.8513 |
| mAP@50 | 0.8948 |
| mAP@50-95 | 0.6118 |

</th>
<th valign="top">


| RMSProp Results | Value |
| :--- | :---: |
| Precision | 0.9138 |
| Recall | 0.8795 |
| mAP@50 | 0.9140 |
| mAP@50-95 | 0.6388 |

</th>
<th valign="top">


| SGD Results | Value |
| :--- | :---: |
| Precision | 0.9211 |
| Recall | 0.8782 |
| mAP@50 | 0.9135 |
| mAP@50-95 | 0.6429 |

</th>
</tr>
</table>

