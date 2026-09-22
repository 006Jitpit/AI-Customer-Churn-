# -*- coding: utf-8 -*-
"""Streamlit app — Telco Churn Predictor (Workshop 11 · ทีมรีเทค 006-022)

สร้างตามเนื้อหาเวิร์กชอปสัปดาห์ที่ 12 (steps 1–9):
  ขั้น 2 ตรวจไฟล์โมเดล → ขั้น 3–4 สร้าง app.py แล้วรัน → ขั้น 5 แก้ error → ขั้น 6 ทำความเข้าใจโค้ด
  → ขั้น 7 ปรับ UI → ขั้น 8 ใช้โมเดล/ฟีเจอร์จริงของทีม → ขั้น 9 เตรียมไฟล์ deploy

โมเดล: DecisionTreeClassifier(max_depth=3, random_state=42) → telco_churn.joblib
ฟีเจอร์: 12 คอลัมน์ตาม model.feature_names_in_ (ข้อมูลทีมไม่ได้สเกล)
"""
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Telco Churn Predictor — ทีมรีเทค", page_icon="📉", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "telco_churn.joblib"
INFO_PATH = BASE_DIR / "model_info.json"


@st.cache_resource
def load_model():
    """โหลดโมเดลครั้งเดียวแล้ว cache ไว้ (ขั้น 3)"""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    with open(INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


model = load_model()
info = load_info()

# ── ขั้น 2: ตรวจไฟล์โมเดล — ยึดรายชื่อฟีเจอร์จากตัวโมเดลจริง ไม่เดา ──
MODEL_FEATURES = list(getattr(model, "feature_names_in_", info["features"]))
KINDS = info["feature_kinds"]
RANGES = info["numeric_ranges"]

HELP = {
    "tenure": "จำนวนเดือนที่เป็นลูกค้าบริษัทนี้ (0–72) — ค่าน้อย = ลูกค้าใหม่ = มักเสี่ยงยกเลิกมากกว่า",
    "MonthlyCharges": "ค่าบริการรายเดือน (ดอลลาร์) — ค่าสูงมักมาคู่กับความเสี่ยงยกเลิกที่สูงขึ้น",
    "TotalCharges": "ยอดเงินสะสมที่จ่ายมาทั้งหมด (ดอลลาร์) — สัมพันธ์กับระยะเวลาที่เป็นลูกค้า",
    "SeniorCitizen": "ลูกค้าอายุ 65 ปีขึ้นไปหรือไม่ (1 = ใช่, 0 = ไม่ใช่)",
    "Contract_One year": "ทำสัญญาแบบรายปี (1 = ใช่) — มีสัญญายิ่งยกเลิกยากขึ้น",
    "Contract_Two year": "ทำสัญญาแบบ 2 ปี (1 = ใช่) — กลุ่มที่เสี่ยงยกเลิกน้อยที่สุด",
    "InternetService_Fiber optic": "ใช้บริการอินเทอร์เน็ตแบบไฟเบอร์ optic (1 = ใช่) — กลุ่มเสี่ยงยกเลิกสูง",
    "InternetService_No": "ไม่มีบริการอินเทอร์เน็ตกับบริษัท (1 = ไม่มี)",
    "PaperlessBilling_Yes": "ใช้ใบแจ้งหนี้แบบอิเล็กทรอนิกส์ (1 = ใช่)",
    "PaymentMethod_Credit card (automatic)": "จ่ายผ่านบัตรเครดิตแบบหักอัตโนมัติ (1 = ใช่)",
    "PaymentMethod_Electronic check": "จ่ายผ่านเช็คอิเล็กทรอนิกส์ (1 = ใช่) — กลุ่มที่เสี่ยงยกเลิกสูง",
    "PaymentMethod_Mailed check": "จ่ายผ่านเช็คทางไปรษณีย์ (1 = ใช่)",
}
LABELS = {
    "tenure": "tenure — ระยะเวลาที่เป็นลูกค้า (เดือน)",
    "MonthlyCharges": "MonthlyCharges — ค่าบริการรายเดือน",
    "TotalCharges": "TotalCharges — ยอดเงินสะสมที่จ่ายทั้งหมด",
    "SeniorCitizen": "SeniorCitizen — เป็นผู้สูงอายุ (65+)",
    "Contract_One year": "สัญญาแบบ 1 ปี",
    "Contract_Two year": "สัญญาแบบ 2 ปี",
    "InternetService_Fiber optic": "อินเทอร์เน็ตแบบไฟเบอร์ optic",
    "InternetService_No": "ไม่มีบริการอินเทอร์เน็ต",
    "PaperlessBilling_Yes": "ใบแจ้งหนี้แบบอิเล็กทรอนิกส์",
    "PaymentMethod_Credit card (automatic)": "จ่ายบัตรเครดิต (หักอัตโนมัติ)",
    "PaymentMethod_Electronic check": "จ่ายเช็คอิเล็กทรอนิกส์",
    "PaymentMethod_Mailed check": "จ่ายเช็คทางไปรษณีย์",
}


def build_input(values: dict) -> pd.DataFrame:
    """สร้าง DataFrame 1 แถว เรียงคอลัมน์ตรงกับตอนเทรนเป๊ะ (ข้อมูลทีมไม่ได้สเกล → ส่งค่าจริง)"""
    return pd.DataFrame([values], columns=MODEL_FEATURES)


st.title("📉 Telco Churn Predictor — ทีมรีเทค")
st.caption(
    "Workshop สัปดาห์ที่ 11 · 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · "
    f"โมเดล: `{info['model']}`"
)
st.write(
    "กรอกข้อมูลลูกค้าแล้วกด **ประเมินความเสี่ยง** — ระบบจะบอกว่าลูกค้ารายนี้มีแนวโน้ม"
    "**ยกเลิกบริการ (Churn)** หรือไม่ พร้อมข้อเสนอว่าควรทำอะไรต่อ"
)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.subheader("ข้อมูลลูกค้า")
    st.caption(f"ฟอร์มสร้างจากฟีเจอร์จริงของโมเดล {len(MODEL_FEATURES)} ตัว (ตรวจจาก `model.feature_names_in_`)")

    values = {}
    with st.form("churn_form"):
        # ตัวเลขต่อเนื่อง → number_input
        for f in [x for x in MODEL_FEATURES if KINDS.get(x) == "numeric"]:
            r = RANGES.get(f, {"min": 0.0, "max": 100.0})
            values[f] = st.number_input(
                LABELS.get(f, f), min_value=float(r["min"]), max_value=float(r["max"]),
                value=float(round((r["min"] + r["max"]) / 2, 2)), step=1.0,
                help=HELP.get(f, ""),
            )
        # ตัวแปร 0/1 → checkbox (แบ่งเป็น 2 คอลัมน์ให้อ่านง่าย)
        st.markdown("**ลักษณะบริการ/การชำระเงิน** (ติ๊ก = 1, ไม่ติ๊ก = 0)")
        cols = st.columns(2)
        for i, f in enumerate([x for x in MODEL_FEATURES if KINDS.get(x) != "numeric"]):
            with cols[i % 2]:
                values[f] = 1 if st.checkbox(LABELS.get(f, f), value=False, help=HELP.get(f, "")) else 0

        submitted = st.form_submit_button("ประเมินความเสี่ยง", use_container_width=True)

with right:
    st.subheader("ผลการประเมิน")
    if not submitted:
        st.info("กรอกข้อมูลด้านซ้าย แล้วกด **ประเมินความเสี่ยง** เพื่อดูผล")
    else:
        payload = build_input(values)
        pred = int(model.predict(payload)[0])
        proba = model.predict_proba(payload)[0]
        p_churn = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

        st.metric("โอกาสยกเลิกบริการ (Churn)", f"{p_churn:.1%}")
        st.progress(min(max(p_churn, 0.0), 1.0))

        if pred == 1:
            st.error(
                f"### ⚠️ เสี่ยง**ยกเลิกบริการ** ({p_churn:.1%})\n"
                "**สิ่งที่ควรทำ:** ติดต่อลูกค้าเชิงรุก — เสนอโปรโมชันต่อสัญญา/ส่วนลด หรือย้ายไปแพ็กเกจที่คุ้มกว่า "
                "และมอบหมายเจ้าหน้าที่ดูแลเฉพาะราย"
            )
        else:
            st.success(
                f"### ✅ ความเสี่ยง**ต่ำ** — น่าจะยังใช้บริการต่อ ({p_churn:.1%})\n"
                "**สิ่งที่ควรทำ:** ดูแลตามปกติ เก็บไว้ในกลุ่ม remarketing และติดตามผลอีกครั้งเมื่อพฤติกรรมเปลี่ยน"
            )

        with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล (12 ฟีเจอร์) และเหตุผล"):
            show = payload.copy()
            st.dataframe(show, use_container_width=True)
            st.caption("ข้อมูลของทีมไม่ได้สเกล → ส่งค่าจริงเข้าโมเดลได้เลย")
            if values.get("tenure", 99) <= 16.5 and values.get("InternetService_Fiber optic") == 1:
                st.warning("กฎสำคัญของโมเดล: **ลูกค้าใหม่ (tenure ≤ ~16 เดือน) + ใช้ไฟเบอร์** = กลุ่มเสี่ยงยกเลิกสูงสุด")
            elif values.get("Contract_Two year") == 1:
                st.info("ลูกค้าที่ทำสัญญา 2 ปี เป็นกลุ่มที่โมเดลมองว่าความเสี่ยงยกเลิกต่ำ")

with st.sidebar:
    st.header("ℹ️ ข้อมูลโมเดล (ขั้น 2)")
    st.write(f"**ทีม:** {info['team']} ({info['team_code']})")
    st.write(f"**ชนิดโมเดล:** `{type(model).__name__}`")
    st.write(f"**max_depth:** `{model.get_params().get('max_depth')}` · **random_state:** `{model.get_params().get('random_state')}`")
    st.write(f"**ความแม่น:** test {info['test_accuracy']:.4f} · CV {info['cv_accuracy']:.4f}")
    st.write(f"**ข้อมูลเทรน:** {info['n_rows']:,} แถว")
    st.write(f"**ฟีเจอร์จาก `feature_names_in_` ({len(MODEL_FEATURES)}):**")
    st.code("\n".join(MODEL_FEATURES), language="text")
    st.caption(
        "ผลลัพธ์: 1 = ยกเลิกบริการ · 0 = ยังใช้บริการอยู่ · เกณฑ์ 0.5\n\n"
        "ข้อมูลไม่ได้สเกล (ไม่ต้องใช้ MinMax)\n\n"
        "ขั้น 9: ไฟล์พร้อม deploy — requirements.txt · README.md · .gitignore"
    )
