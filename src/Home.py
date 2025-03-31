import streamlit as st
from PIL import Image
import cv2
import numpy as np
from roboflow import Roboflow
import io
from fpdf import FPDF
import base64
import os
from datetime import datetime

## store initial session state values
project_url_od, private_api_key, uploaded_file_od = ("", "", "")

if 'project_url_od' not in st.session_state:
    st.session_state['project_url_od'] = "https://app.roboflow.com/ai-eg-7bmff/gas-pipelines/2"
if 'private_api_key' not in st.session_state:
    st.session_state['private_api_key'] = "u8UOsCouQTPV12lxGfny"
if 'uploaded_file_od' not in st.session_state:
    st.session_state['uploaded_file_od'] = ""
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = "40"
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = "30"
if 'include_bbox' not in st.session_state:
    st.session_state['include_bbox'] = "Yes"
if 'show_class_label' not in st.session_state:
    st.session_state['show_class_label'] = 'Show Labels'
if 'box_type' not in st.session_state:
    st.session_state['box_type'] = "regular"

project_url_od = 'https://app.roboflow.com/ai-eg-7bmff/gas-pipelines/2'
private_api_key = 'u8UOsCouQTPV12lxGfny'
extracted_url = project_url_od.split("roboflow.com/")[1]
if "model" in project_url_od.split("roboflow.com/")[1]:
    workspace_id = extracted_url.split("/")[0]
    model_id = extracted_url.split("/")[1]
    version_number = extracted_url.split("/")[3]
elif "deploy" in project_url_od.split("roboflow.com/")[1]:
    workspace_id = extracted_url.split("/")[0]
    model_id = extracted_url.split("/")[1]
    version_number = extracted_url.split("/")[3]
else:
    workspace_id = extracted_url.split("/")[0]
    model_id = extracted_url.split("/")[1]
    version_number = extracted_url.split("/")[2]

def run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img):
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(workspace_id).project(model_id)
    project_metadata = project.get_version_information()
    version = project.version(version_number)
    model = version.model

    project_type = st.write(f"#### Project Type: {project.type}")

    for i in range(len(project_metadata)):
        if project_metadata[i]['id'] == extracted_url:
            st.write(f"#### Model: {model_id}")
            st.write(f"#### Version: {project_metadata[i]['name']}")
            st.write(f"Input image (px):")

            width_metric, height_metric = st.columns(2)
            width_metric.metric(label='Width (px)', value=project_metadata[i]['preprocessing']['resize']['width'])
            height_metric.metric(label='Height (px)', value=project_metadata[i]['preprocessing']['resize']['height'])

    st.write("#### Uploaded image")
    st.image(uploaded_img, caption="Uploaded image")

    predictions = model.predict(uploaded_img) # 'https://daanaea.github.io/i/assets/img/IMG_6905_pipe_with_corrosion.jpg'
    predictions.save("output.jpg")
    predictions_json = predictions.json()

    # drawing bounding boxes with the Pillow library
    collected_predictions = []
    
    for bounding_box in predictions:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']

        # st.write(class_name) # TODO

        confidence_score = bounding_box['confidence']

        box = (x0, x1, y0, y1)
        detected_x = int(bounding_box['x'] - bounding_box['width'] / 2)
        detected_y = int(bounding_box['y'] - bounding_box['height'] / 2)
        detected_width = int(bounding_box['width'])
        detected_height = int(bounding_box['height'])

        # ROI (Region of Interest), or detected bounding box area
        roi_bbox = [detected_y, detected_height, detected_x, detected_width]

        # st.write(roi_bbox) # TODO

        collected_predictions.append({
            "Class": class_name, 
            "Confidence": confidence_score,
            "x0,x1,y0,y1": [int(x0),int(x1),int(y0),int(y1)],
            "Width": int(bounding_box['width']),
            "Height": int(bounding_box['height']),
            "ROI, bbox (y+h, x+w)": roi_bbox,
            "Area, bbox (px)": abs(int(x0)-int(x1))*abs(int(y0)-int(y1))
        })

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        
        # draw/place FILLED half-transparent bounding boxes on image
        if class_name == 'gas-pipelines':
            bg_color = (255, 255, 0)
            alpha = 0.25 # transparency factor
            # add class name without background on the left bottom corner
            cv2.putText(inferenced_img,
                class_name + ' ' + str(round(confidence_score, 2)), #text to place on image
                (int(x0) + 5, int(y1) - 55), #location of text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, #font
                fontScale=3, #font scale
                color=(255, 255, 255), #text color
                thickness=3
            )
        else:
            bg_color = (255, 0, 0)
            alpha = 0.35 # transparency factor
            cv2.putText(inferenced_img,
                class_name + ' ' + str(round(confidence_score, 2)), #text to place on image
                (int(x0) + 5, int(y0) - 55), #location of text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, #font
                fontScale=3, #font scale
                color=(255, 0, 0), #text color
                thickness=3
            )
        overlay = inferenced_img.copy()
        cv2.rectangle(
            overlay,
            start_point,
            end_point,
            color=bg_color,
            thickness=-1
        )
        cv2.addWeighted(overlay, alpha, inferenced_img, 1 - alpha, 0, inferenced_img)


    ## Subtitle.
    st.write("### Detected defects")
    st.image(inferenced_img, caption="Processed image", use_container_width=True)

    results_tab, json_tab, project_tab = st.tabs(["Processing results", "Results in JSON format", "Model information"])

    with results_tab:
        ## Display results dataframe in main app.
        st.write('### Processing results')
        st.dataframe(collected_predictions)

    with json_tab:
        ## Display the JSON in main app.
        st.write('### Results in JSON format')
        st.write(predictions_json)

    with project_tab:
        st.write(f"Damage group: {project.annotation}")
        col1, col2, col3 = st.columns(3)
        col1.write(f'Total images in dataset: {version.images}')
        # col1.metric(label='Количество аугментированных изображений', value=version.splits['train'])
        
        for i in range(len(project_metadata)):
            if project_metadata[i]['id'] == extracted_url:
                col2.metric(label='mean Average Precision (mAP)', value=f"{float(project_metadata[i]['model']['map'])}%")
        
        col3.metric(label='Training dataset (train)', value=project.splits['train'])
        col3.metric(label='Validation dataset (validation)', value=project.splits['valid'])
        col3.metric(label='Test dataset (test)', value=project.splits['test'])

        col4, col5, col6 = st.columns(3)
        col4.write('Applied preprocessing steps:')
        col4.json(version.preprocessing)
    
    if st.button("Generate PDF report"):
        pdf_path = generate_pdf(predictions_json, "output.jpg")
        display_pdf_download_button(pdf_path)

# save results into PDF
def generate_pdf(result_json, image_path):
    pdf = FPDF()
    pdf.add_page()

    # Add project name in bold
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, txt="GPdefectscan AI", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Gas Pipeline Defect Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.image(image_path, x=10, y=None, w=180)
    pdf.ln(85)

    pdf.set_font("Arial", size=10)
    for item in result_json['predictions']:
        text = f"Class: {item['class']}, Confidence: {item['confidence']:.2f}, BBox: {item['x']}, {item['y']}, {item['width']}, {item['height']}"
        pdf.multi_cell(0, 10, txt=text)

    pdf.ln(10)
    pdf.set_font("Arial", size=8)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Generated on: {current_datetime}", ln=True, align='C')

    # Add project website link
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Website: https://pipes-defect-detection.streamlit.app/", ln=True, align='C')
    pdf.ln(10)

    pdf_output = "detection_report.pdf"
    pdf.output(pdf_output)
    return pdf_output

def display_pdf_download_button(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="detection_report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

st.write("# GPdefectscan AI: Intelligent Pipeline Defect Segmentation System")

with st.form("project_access"):
    st.write("#### Select an image for processing.")
    uploaded_file_od = st.file_uploader("Upload Image File",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)
    st.write("#### Click 'Find defects' after uploading the image!")
   
    submitted = st.form_submit_button("Find defects")
    
    if submitted:
        st.write("Loading model...")

if uploaded_file_od != None:
    # upload user image
    image = Image.open(uploaded_file_od)
    uploaded_img = np.array(image)
    inferenced_img = uploaded_img.copy()

    # convert image to bytes
    byte_io = io.BytesIO()
    image.save(byte_io, format='JPEG')
    byte_array = byte_io.getvalue()

    with open("test_image.jpg", "wb") as f:
        f.write(byte_array)
    
    workspace_id = 'ai-eg-7bmff'
    model_id = 'gas-pipelines'
    version_number = 2

    run_inference(workspace_id, model_id, version_number, "test_image.jpg", inferenced_img)


"""

### User guide

##### Website

Go to the link: [pipes-defect-detection.streamlit.app](https://pipes-defect-detection.streamlit.app).

---

##### Instructions

1. Uploading an umage:
    * Click the "Upload Image" button.
    * Select a file with an image of the pipeline section on your device.
    * Ensure the image is clear and well-lit.
2. Running analysis:
    * After uploading the file, click the "Analyze" or "Run Detection" button.
    * Wait for the analysis to complete (takes a few seconds).
3. Viewing results:
    * The screen will display the original image with highlighted boxes on areas where defects are detected.
    * The type of defect and its probability will be shown next to it.

---

##### Tips

* Image Quality: Upload high-resolution images to improve analysis accuracy.
* Types of Defects: The system detects visually noticeable defects such as deformation, corrosion, and paint damage.

---

##### FAQ

*What images are suitable?* Clear photos of above-ground pipeline sections taken in good lighting conditions.

*Can videos be analyzed?* Currently, only image uploads are supported.

*How to provide feedback?* Email us at: [eginovaa@gmail.com](mailto:eginovaa@gmail.com).

---

##### Limitations

The system does not replace physical pipeline inspections but serves as an additional diagnostic tool.

Defects hidden under insulation cannot be detected.

---

##### Recommendations

Use the analysis results for prompt response to detected defects and to prevent emergency situations.
"""