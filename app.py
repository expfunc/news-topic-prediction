from preprocessing import tokenize_sentence
import streamlit as st
import joblib

obj = joblib.load("news_topic_svc.joblib")
pipeline = obj["pipeline"]
le = obj["label_encoder"]

st.set_page_config(page_title="News Topic Classifier", layout="centered")

st.title("Классификация тем новостей по тексту")

# Описание
st.markdown("""
Эта модель позволяет по тексту новости определить её тему.  
Модель (LinearSVC) обучена на русскоязычных новостях Lenta.ru.  

**Доступные темы:**  
Россия, Мир, Экономика, Спорт, Культура, Бывший СССР,  
Наука и техника, Интернет и СМИ, Из жизни, Дом,  
Силовые структуры, Ценности, Бизнес, Путешествия, Прочее.
""")

# Загрузка примеров
examples = {}
for fname in ["sport.txt", "science.txt", "home.txt"]:
    with open(f"examples/{fname}", "r", encoding="utf-8") as f:
        examples[fname.replace(".txt", "")] = f.read().strip()

# Кнопка примеров
with st.expander("Примеры новостей"):
    st.subheader("Спорт")
    st.markdown("[Российские гребцы победили на чемпионате мира](https://lenta.ru/news/2025/08/24/rossiyskie-grebtsy-pobedili-na-chempionate-mira/)")
    st.code(examples["sport"], language="text")

    st.subheader("Наука и техника")
    st.markdown("[В Туле обсудят применение технологий для борьбы с борщевиком](https://ria.ru/20250822/tula-2037062159.html)")
    st.code(examples["science"], language="text")

    st.subheader("Дом")
    st.markdown("[Названы районы Москвы с наибольшим ростом цен на аренду жилья в июле](https://realty.rbc.ru/news/689b9b859a794779902a9375)")
    st.code(examples["home"], language="text")

# Поле ввода текста
user_input = st.text_area("Вставьте текст новости:", height=150)

if st.button("Определить тему"):
    if user_input.strip():
        pred = pipeline.predict([user_input])[0]
        topic = le.inverse_transform([pred])[0]
        st.success(f"**Тема новости:** {topic}")
    else:
        st.warning("Введите текст для определения.")



