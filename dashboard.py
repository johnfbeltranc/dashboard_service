import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load & preprocess data
# -------------------------------
X_raw, y_raw = load_iris(return_X_y=True, as_frame=True)
df_raw = X_raw
df_raw["species"] = y_raw

df = df_raw.copy()
df.columns = df.columns.str.lower().str.replace("(","").str.replace(")","").str.replace(" ","_")
df["species"] = df["species"].map({0:'setosa', 1:'versicolor', 2:'virginica'}).astype("category")

# -------------------------------
# 2. Train/test split & model
# -------------------------------
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['species'], random_state=2025)
X_train, y_train = df_train.drop(columns=['species']), df_train['species']
X_test, y_test = df_test.drop(columns=['species']), df_test['species']

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

clf_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=2025))
])
clf_rf.fit(X_train, y_train_enc)
y_hat = clf_rf.predict(X_test)
acc = accuracy_score(y_test_enc, y_hat)

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🌸 Iris Dashboard")
st.write("Exploración, visualización y predicción con el dataset de Iris")

# Sidebar
st.sidebar.header("Opciones")
species_filter = st.sidebar.selectbox("Filtrar especie:", df["species"].unique())
st.sidebar.write(f"Exactitud del modelo: **{acc:.2f}**")

# Tabla filtrada
st.subheader("📊 Datos filtrados")
st.write(df[df["species"] == species_filter])

# Scatter Matrix con Plotly
st.subheader("🔎 Visualización multivariable")
fig = px.scatter_matrix(
    df,
    dimensions=['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm'],
    color="species"
)
st.plotly_chart(fig)

# Heatmap con Seaborn
st.subheader("🔥 Correlaciones")
fig2, ax = plt.subplots()
sns.heatmap(df.select_dtypes('number').corr(), vmin=-1, vmax=1, annot=True, cmap="RdBu", ax=ax)
st.pyplot(fig2)

# Predicción interactiva
st.subheader("🤖 Predicción con Random Forest")
sl = st.number_input("Sepal length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sw = st.number_input("Sepal width (cm)", min_value=2.0, max_value=4.5, step=0.1)
pl = st.number_input("Petal length (cm)", min_value=1.0, max_value=7.0, step=0.1)
pw = st.number_input("Petal width (cm)", min_value=0.1, max_value=2.5, step=0.1)

if st.button("Predecir especie"):
    sample = pd.DataFrame([[sl, sw, pl, pw]], columns=X_train.columns)
    pred = clf_rf.predict(sample)
    st.success(f"La especie predicha es: **{label_encoder.inverse_transform(pred)[0]}**")


