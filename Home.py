import streamlit as st
from PIL import Image
import cv2
import numpy as np
from roboflow import Roboflow
import io

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

def run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img):
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(workspace_id).project(model_id)
    project_metadata = project.get_version_information()
    version = project.version(version_number)
    model = version.model

    project_type = st.write(f"#### Тип проекта: {project.type}")

    for i in range(len(project_metadata)):
        if project_metadata[i]['id'] == extracted_url:
            st.write(f"#### Модель: {model_id}")
            st.write(f"#### Версия: {project_metadata[i]['name']}")
            st.write(f"Параметры входного изображения (пиксели, px):")

            width_metric, height_metric = st.columns(2)
            width_metric.metric(label='Ширина (px)', value=project_metadata[i]['preprocessing']['resize']['width'])
            height_metric.metric(label='Высота (px)', value=project_metadata[i]['preprocessing']['resize']['height'])

            # if project_metadata[i]['model']['fromScratch']:
            #     train_checkpoint = 'Scratch'
            #     st.write(f"#### Version trained from {train_checkpoint}")
            # elif project_metadata[i]['model']['fromScratch'] is False:
            #     train_checkpoint = 'Checkpoint'
            #     st.write(f"#### Version trained from {train_checkpoint}")
            # else:
            #     train_checkpoint = 'Not Yet Trained'
            #     st.write(f"#### Version is {train_checkpoint}")

    st.write("#### Загруженное изображение")
    st.image(uploaded_img, caption="Загруженное изображение")

    predictions = model.predict(uploaded_img) # 'https://daanaea.github.io/i/assets/img/IMG_6905_pipe_with_corrosion.jpg'
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
            "Класс": class_name, 
            "Точность": confidence_score,
            "x0,x1,y0,y1": [int(x0),int(x1),int(y0),int(y1)],
            "Ширина":int(bounding_box['width']),
            "Высота":int(bounding_box['height']),
            "ROI, bbox (y+h, x+w)": roi_bbox,
            "Площадь, bbox (px)": abs(int(x0)-int(x1))*abs(int(y0)-int(y1))
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
            # add class name with filled background
            # cv2.rectangle(
            #     inferenced_img,
            #     (int(x0), int(y0)), (int(x1), int(y0) + 85),
            #     color=(0, 255, 100),
            #     thickness=5
            # )
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
    st.write("### Найденные повреждения")
    st.image(inferenced_img, caption="Обработанное изображение", use_container_width=True)

    results_tab, json_tab, project_tab = st.tabs(["Результаты обработки", "Результаты в формате JSON", "Информация о модели"])

    with results_tab:
        ## Display results dataframe in main app.
        st.write('### Результаты обработки')
        st.dataframe(collected_predictions)

    with json_tab:
        ## Display the JSON in main app.
        st.write('### Результаты в формате JSON')
        st.write(predictions_json)

    with project_tab:
        st.write(f"Группа повреждений: {project.annotation}")
        col1, col2, col3 = st.columns(3)
        col1.write(f'Общее количество изображений в датасете: {version.images}')
        # col1.metric(label='Количество аугментированных изображений', value=version.splits['train'])
        
        for i in range(len(project_metadata)):
            if project_metadata[i]['id'] == extracted_url:
                col2.metric(label='mean Average Precision (mAP)', value=f"{float(project_metadata[i]['model']['map'])}%")
        
        col3.metric(label='Тренировочный датасет (train)', value=project.splits['train'])
        col3.metric(label='Проверочный датасет (validation)', value=project.splits['valid'])
        col3.metric(label='Тестовый датасет (test)', value=project.splits['test'])

        col4, col5, col6 = st.columns(3)
        col4.write('Примененные шаги предобработки:')
        col4.json(version.preprocessing)
        # col5.write('Augmentation steps applied:')
        # col5.json(version.augmentation)
        # col6.metric(label='Тренировочный датасет (train)', value=version.splits['train'], delta=f"Increased by Factor of {(version.splits['train'] / project.splits['train'])}")
        # col6.metric(label='Проверочный датасет (validation)', value=version.splits['valid'], delta="No Change")
        # col6.metric(label='Тестовый датасет (test)', value=version.splits['test'], delta="No Change")


# # Add in location to select image.
# with st.sidebar:
#     st.write("#### Выберите изображения для обработки.")
#     uploaded_file_od = st.file_uploader("Загрузка файла изображения",
#                                         type=["png", "jpg", "jpeg"],
#                                         accept_multiple_files=False)
    
st.write("# GPdefectscan AI: Интеллектуальная система сегментации дефектов трубопроводов")

with st.form("project_access"):
    st.write("#### Выберите изображения для обработки.")
    uploaded_file_od = st.file_uploader("Загрузка файла изображения",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)
    st.write("#### Нажмите на кнопку 'Найти повреждения' после загрузки изображения!")
    # project_url_od = st.text_input("Project URL", key="project_url_od",
    #                             help="Copy/Paste Your Project URL: https://docs.roboflow.com/python#finding-your-project-information-manually",
    #                             placeholder="https://app.roboflow.com/workspace-id/model-id/version")
    # private_api_key = st.text_input("Private API Key", key="private_api_key", type="password",placeholder="Input Private API Key")
    submitted = st.form_submit_button("Найти повреждения")
    
    if submitted:
        st.write("Загрузка модели...")
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

if uploaded_file_od != None:
    # User-selected image.
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

### Памятка пользователя

##### Открытие сайта

Перейдите по ссылке: [pipes-defect-detection.streamlit.app](https://pipes-defect-detection.streamlit.app).

---

##### Основные шаги работы

1. Загрузка изображения:
    * Нажмите кнопку «Upload Image».
    * Выберите файл с изображением участка газопровода на вашем устройстве.
    * Убедитесь, что изображение четкое и хорошо освещено.
2. Запуск анализа:
    * После загрузки файла нажмите кнопку «Analyze» или «Run Detection».
    * Дождитесь завершения анализа (займет несколько секунд).
3. Просмотр результата:
    * На экране отобразится исходное изображение с выделенными рамками на
участках, где обнаружены дефекты.
    * Рядом будут указаны тип дефекта и вероятность его наличия.

---

##### Полезные советы

* Качество изображения: Загружайте изображения с высоким разрешением
для повышения точности анализа.
* Типы дефектов: Система распознает визуально заметные дефекты, такие
как деформация, коррозия, разрушение окраски.

---

##### Часто задаваемые вопросы

*Какие изображения подходят?* Четкие фото надземных участков
газопровода, снятые при хорошем освещении.

*Можно ли анализировать видео?* На данный момент поддерживается
только загрузка изображений.

*Как связаться для обратной связи?* Пишите на email: [eginovaa@gmail.com](mailto:eginovaa@gmail.com).

---

##### Ограничения

Система не заменяет физический осмотр газопровода, а является
дополнительным инструментом для диагностики.

Дефекты, скрытые под изоляцией, не могут быть распознаны.

---

##### Рекомендации

Используйте результаты анализа для оперативного реагирования на
обнаруженные дефекты и предотвращения аварийных ситуаций.
"""