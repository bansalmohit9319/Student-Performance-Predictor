import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI Student Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton>button {
    background-color: #38bdf8;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stDownloadButton>button {
    background-color: green;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.title(" AI Student Performance Dashboard")
st.write("Smart ML-powered prediction system")


st.sidebar.header(" Dashboard Menu")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "study_hours": [1,2,3,4,5,6,7,8,2,5,6,1,7,8,3,4],
        "attendance": [40,50,60,65,70,80,85,90,45,75,82,35,88,92,55,67],
        "marks": [30,35,40,50,60,70,75,85,33,68,72,25,80,90,45,58],
        "result": [
            "Fail","Fail","Fail","Pass",
            "Pass","Pass","Pass","Pass",
            "Fail","Pass","Pass","Fail",
            "Pass","Pass","Fail","Pass"
        ]
    })


st.subheader(" Dataset Preview")
st.dataframe(df, use_container_width=True)


X = df[["study_hours", "attendance", "marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = (model, acc)


best_model_name = max(
    results,
    key=lambda x: results[x][1]
)

best_model = results[best_model_name][0]

best_accuracy = results[best_model_name][1]


st.subheader(" Model Performance")

col1, col2 = st.columns(2)

with col1:
    for name, (_, acc) in results.items():
        st.metric(
            label=name,
            value=f"{acc*100:.2f}%"
        )

with col2:
    st.success(
        f"Best Model: {best_model_name}"
    )

st.subheader("Student Prediction System")

student_name = st.text_input(
    "Student Name"
)

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.slider(
        "Study Hours",
        0, 12, 4
    )

with col2:
    attendance = st.slider(
        "Attendance %",
        0, 100, 75
    )

with col3:
    marks = st.slider(
        "Marks",
        0, 100, 60
    )

if st.button(" Predict Performance"):

    input_data = pd.DataFrame(
        [[study_hours, attendance, marks]],
        columns=[
            "study_hours",
            "attendance",
            "marks"
        ]
    )

    prediction = best_model.predict(input_data)

    probability = best_model.predict_proba(
        input_data
    )

    confidence = max(probability[0]) * 100

    st.subheader("Prediction Result")

    if prediction[0] == "Pass":
        st.success(
            f" {student_name} Will Likely PASS"
        )
    else:
        st.error(
            f" {student_name} Will Likely FAIL"
        )

    st.info(
        f"Prediction Confidence: {confidence:.2f}%"
    )

    
    st.progress(int(confidence))

st.subheader(" Feature Importance Analysis")

rf_model = results["Random Forest"][0]

importance = rf_model.feature_importances_

fig, ax = plt.subplots()

ax.bar(
    X.columns,
    importance,
    color=["red", "blue", "green"]
)

ax.set_title(
    "Which Factor Impacts Most?"
)

st.pyplot(fig)


st.subheader(" Dataset Analytics")

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(df["marks"])

with col2:
    st.line_chart(df["attendance"])

st.subheader("⬇ Export Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Dataset CSV",
    data=csv,
    file_name="student_dataset.csv",
    mime="text/csv"
)

