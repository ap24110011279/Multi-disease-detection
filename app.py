"""
Multi-disease prediction app.   Run:  python -m streamlit run app.py
"""
import glob
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Disease Detection", page_icon="🩺")
st.title("🩺 Multi-Disease Detection System")
st.caption("Logistic Regression models. For learning purposes only, not medical advice.")

files = sorted(glob.glob("models/*.joblib"))
if not files:
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

# ---------------------------------------------------------------
# 1. Patient details (same for every disease)
# ---------------------------------------------------------------
st.subheader("1. Patient details")
name = st.text_input("Patient name", placeholder="Enter full name")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
with c2:
    sex = st.selectbox("Sex", ["Female", "Male"])

# ---------------------------------------------------------------
# 2. Disease selection
# ---------------------------------------------------------------
st.subheader("2. Choose disease")
names = {os.path.basename(f)[:-7].replace("_", " ").title(): f for f in files}
choice = st.selectbox("Select disease", list(names))
bundle = joblib.load(names[choice])
model, meta = bundle["model"], bundle["meta"]

AGE_NAMES = {"age"}
SEX_NAMES = {"sex", "gender"}
age_feature = next((f for f in meta if f.strip().lower() in AGE_NAMES), None)
sex_feature = next((f for f in meta if f.strip().lower() in SEX_NAMES), None)

if age_feature is None or sex_feature is None:
    missing = [w for w, f in (("age", age_feature), ("sex", sex_feature)) if f is None]
    st.info(f"The {choice} model does not use {' or '.join(missing)} as an input, "
            f"so {'it is' if len(missing) == 1 else 'they are'} shown in the result only "
            "and do not change the prediction.")
if choice == "Diabetes" and sex == "Male":
    st.warning("The Diabetes model was trained only on female patients (Pima dataset), "
               "so treat this result with extra caution.")

# Friendly labels for yes/no style columns
FRIENDLY = {
    "fbs":   {0: "No (sugar 120 or below)", 1: "Yes (sugar above 120)"},
    "exang": {0: "No", 1: "Yes"},
}


def is_code_column(info):
    """A numeric column holding only a few whole numbers (0/1, 0-3...) is a code, not a measurement."""
    lo, hi = info["min"], info["max"]
    return float(lo).is_integer() and float(hi).is_integer() and (hi - lo) <= 5


def sex_value(info):
    """Convert the chosen Sex into whatever this model expects."""
    if info["type"] == "num":
        return 1 if sex == "Male" else 0           # 1 = male, 0 = female
    for option in info["options"]:
        if str(option).strip().lower() == sex.lower():
            return option
    for option in info["options"]:
        if str(option).strip().lower()[:1] == sex.lower()[:1]:
            return option
    return info["default"]


# ---------------------------------------------------------------
# 3. Medical values
# ---------------------------------------------------------------
st.subheader(f"3. Medical values for {choice}")
values = {}
cols = st.columns(2)
shown = 0
for feature, info in meta.items():
    if feature == age_feature:
        values[feature] = float(age)
        continue
    if feature == sex_feature:
        values[feature] = sex_value(info)
        continue
    with cols[shown % 2]:
        shown += 1
        if info["type"] == "num" and is_code_column(info):
            options = list(range(int(info["min"]), int(info["max"]) + 1))
            labels = FRIENDLY.get(feature, {})
            default = min(options, key=lambda o: abs(o - info["default"]))
            values[feature] = st.selectbox(
                feature, options, index=options.index(default),
                format_func=lambda o, labels=labels: labels.get(o, str(o)))
        elif info["type"] == "num":
            values[feature] = st.number_input(
                feature, min_value=info["min"], max_value=info["max"],
                value=min(max(info["default"], info["min"]), info["max"]))
        else:
            values[feature] = st.selectbox(feature, info["options"])

if age_feature is not None:
    lo, hi = meta[age_feature]["min"], meta[age_feature]["max"]
    if not lo <= age <= hi:
        st.warning(f"The {choice} model was trained on ages {int(lo)} to {int(hi)}. "
                   "The result may be less reliable for this age.")

# ---------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------
if st.button("Predict"):
    prob = model.predict_proba(pd.DataFrame([values]))[0, 1]
    patient = name.strip() or "Patient"
    st.subheader(f"Result for {patient}")
    st.table(pd.DataFrame([{"Name": patient, "Age": age, "Sex": sex, "Disease": choice,
                            "Risk": f"{prob * 100:.1f}%"}]))
    st.metric("Estimated disease probability", f"{prob * 100:.1f}%")
    if prob >= 0.5:
        st.error("⚠️ High risk detected. Please consult a doctor.")
    else:
        st.success("✅ Low risk predicted.")
    st.progress(float(prob))