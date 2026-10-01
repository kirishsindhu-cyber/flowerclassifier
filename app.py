import numpy as np, joblib, streamlit as st, pandas as pd
import tensorflow as tf

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier (ANN)")
st.caption("TensorFlow/Keras neural network + Streamlit  |  Kirish (RA2411056050002) & [Teammate]")

@st.cache_resource
def load():
    return tf.keras.models.load_model("iris_model.keras"), joblib.load("scaler.pkl")
model, scaler = load()
NAMES = ["setosa", "versicolor", "virginica"]

st.write("Set the flower's measurements (cm) and the network predicts its species.")
c1, c2 = st.columns(2)
sl = c1.slider("Sepal length", 4.0, 8.0, 5.8, 0.1)
sw = c1.slider("Sepal width", 2.0, 4.5, 3.0, 0.1)
pl = c2.slider("Petal length", 1.0, 7.0, 4.3, 0.1)
pw = c2.slider("Petal width", 0.1, 2.6, 1.3, 0.1)

p = model.predict(scaler.transform([[sl, sw, pl, pw]]), verbose=0)[0]
st.subheader(f"Prediction: **{NAMES[int(p.argmax())].capitalize()}**  ({p.max()*100:.1f}% confidence)")
st.bar_chart(pd.DataFrame({"probability": p}, index=NAMES))
with st.expander("Try sample flowers"):
    st.markdown("Setosa: 5.1, 3.5, 1.4, 0.2  \nVersicolor: 6.0, 2.9, 4.5, 1.5  \nVirginica: 6.7, 3.0, 5.2, 2.3")
