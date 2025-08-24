import streamlit as st
import joblib

# загрузка
obj = joblib.load("news_topic_svc.joblib")
pipeline = obj["pipeline"]
le = obj["label_encoder"]

st.title("Классификация темы новости по по ее тексту")

user_input = st.text_area("Введите текст новости:", "")

if st.button("Определить тему"):
    if user_input.strip():
        pred = pipeline.predict([user_input])
        topic = le.inverse_transform(pred)[0]
        st.success(f"Тема: **{topic}**")
    else:
        st.warning("Введите текст!")
