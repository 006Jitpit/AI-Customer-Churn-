# -*- coding: utf-8 -*-
"""Streamlit app — Telco Churn Predictor (Workshop สัปดาห์ที่ 11 · ทีมรีเทค 006-022)

UI เวอร์ชันมืออาชีพ: ธีมสีเข้ม-ทีล, hero header พร้อมชื่อสมาชิก, การ์ดผลลัพธ์, แท็บข้อมูลโมเดล

โมเดล: DecisionTreeClassifier(max_depth=3, random_state=42) → telco_churn.joblib
ฟีเจอร์: อ่านจาก model.feature_names_in_ (12 คอลัมน์ · ข้อมูลทีมไม่ได้สเกล)
"""
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Telco Churn Predictor · ทีมรีเทค",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────  ธีม / CSS  ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
    font-family: 'IBM Plex Sans Thai', 'Sarabun', system-ui, sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, #0F2A43 0%, #124E66 55%, #1D7A8C 100%);
    border-radius: 22px; padding: 30px 34px 26px; color: #fff;
    box-shadow: 0 24px 48px -28px rgba(15, 42, 67, .75);
    position: relative; overflow: hidden;
}
.hero:after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,.16), transparent 65%); border-radius: 50%;
}
.hero .kicker { font-size: .82rem; letter-spacing: 2.4px; text-transform: uppercase; opacity: .8; margin-bottom: 6px; }
.hero h1 { font-size: 2.05rem; font-weight: 700; margin: 0 0 8px; line-height: 1.25; }
.hero p { margin: 0; opacity: .92; font-size: .98rem; }
.chips { margin-top: 16px; }
.chip {
    display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.30);
    color: #fff; padding: 6px 14px; border-radius: 999px; font-size: .84rem; margin: 0 8px 8px 0;
    backdrop-filter: blur(4px);
}
.chip.solid { background: #FFFFFF; color: #0F2A43; border-color: #FFFFFF; font-weight: 600; }

/* ── การ์ด ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important; border: 1px solid #E3E9EF !important;
    background: #FFFFFF !important; box-shadow: 0 12px 30px -24px rgba(15, 42, 67, .5);
    padding: 6px 4px;
}

/* ── ปุ่ม ── */
div.stFormSubmitButton > button, div.stButton > button {
    background: linear-gradient(135deg, #124E66, #1D7A8C) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: .62rem 1rem !important;
    box-shadow: 0 10px 24px -16px rgba(18, 78, 102, .9);
    transition: transform .12s ease, box-shadow .12s ease;
}
div.stFormSubmitButton > button:hover, div.stButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 14px 26px -14px rgba(18, 78, 102, .95);
}

/* ── metric ── */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #F7FAFC, #EEF4F8);
    border: 1px solid #E3E9EF; border-radius: 14px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { font-weight: 600; color: #4A5A6A; }

/* ── แบนเนอร์ผลลัพธ์ ── */
.badge { border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }
.badge h3 { margin: 0 0 6px; font-size: 1.18rem; font-weight: 700; }
.badge p { margin: 0; font-size: .95rem; line-height: 1.6; }
.badge.risk { background: #FDECEC; border: 1px solid #F5C6C0; color: #8F2C22; }
.badge.safe { background: #E9F7EF; border: 1px solid #BFE7D0; color: #1B6B42; }
.badge .tag {
    display: inline-block; font-size: .74rem; letter-spacing: 1.4px; text-transform: uppercase;
    font-weight: 700; opacity: .75; margin-bottom: 6px;
}

/* ── KPI เล็ก ๆ ใต้ผลลัพธ์ ── */
.mini { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 2px; }
.mini div {
    flex: 1 1 120px; background: #F5F8FB; border: 1px solid #E3E9EF; border-radius: 12px;
    padding: 10px 12px; text-align: center;
}
.mini span { display: block; font-size: .74rem; color: #6B7A8C; }
.mini strong { font-size: 1.02rem; color: #0F2A43; }

.footer { text-align: center; color: #8A96A6; font-size: .84rem; margin-top: 28px; line-height: 1.8; }
.section-note { color: #5A6B82; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────  โหลดโมเดล  ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "telco_churn.joblib")


@st.cache_data
def load_info():
    with open(BASE_DIR / "model_info.json", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
info = load_info()

MODEL_FEATURES = list(getattr(model, "feature_names_in_", info["features"]))
KINDS = info["feature_kinds"]
RANGES = info["numeric_ranges"]

TEAM = [("006", "จิตรพิชญา จันทรโชติ"), ("022", "ธีรพันธ์ โคตรวงษ์")]

HELP = {
    "tenure": "จำนวนเดือนที่เป็นลูกค้าบริษัทนี้ (0–72) — ค่าน้อย = ลูกค้าใหม่ = มักเสี่ยงยกเลิกมากกว่า",
    "MonthlyCharges": "ค่าบริการรายเดือน (ดอลลาร์) — ค่าสูงมักมาคู่กับความเสี่ยงยกเลิกที่สูงขึ้น",
    "TotalCharges": "ยอดเงินสะสมที่จ่ายมาทั้งหมด (ดอลลาร์) — สัมพันธ์กับระยะเวลาที่เป็นลูกค้า",
    "SeniorCitizen": "ลูกค้าอายุ 65 ปีขึ้นไปหรือไม่ (ติ๊ก = ใช่)",
    "Contract_One year": "ทำสัญญาแบบ 1 ปี — มีสัญญายิ่งยกเลิกยากขึ้น",
    "Contract_Two year": "ทำสัญญาแบบ 2 ปี — กลุ่มที่เสี่ยงยกเลิกน้อยที่สุด",
    "InternetService_Fiber optic": "ใช้บริการอินเทอร์เน็ตแบบไฟเบอร์ optic — กลุ่มเสี่ยงยกเลิกสูง",
    "InternetService_No": "ไม่มีบริการอินเทอร์เน็ตกับบริษัท",
    "PaperlessBilling_Yes": "ใช้ใบแจ้งหนี้แบบอิเล็กทรอนิกส์",
    "PaymentMethod_Credit card (automatic)": "จ่ายผ่านบัตรเครดิตแบบหักอัตโนมัติ",
    "PaymentMethod_Electronic check": "จ่ายผ่านเช็คอิเล็กทรอนิกส์ — กลุ่มที่เสี่ยงยกเลิกสูง",
    "PaymentMethod_Mailed check": "จ่ายผ่านเช็คทางไปรษณีย์",
}
LABELS = {
    "tenure": "ระยะเวลาที่เป็นลูกค้า (เดือน)",
    "MonthlyCharges": "ค่าบริการรายเดือน (USD)",
    "TotalCharges": "ยอดเงินสะสมที่จ่ายทั้งหมด (USD)",
    "SeniorCitizen": "ผู้สูงอายุ 65 ปีขึ้นไป",
    "Contract_One year": "สัญญาแบบ 1 ปี",
    "Contract_Two year": "สัญญาแบบ 2 ปี",
    "InternetService_Fiber optic": "อินเทอร์เน็ตแบบไฟเบอร์ optic",
    "InternetService_No": "ไม่มีบริการอินเทอร์เน็ต",
    "PaperlessBilling_Yes": "ใบแจ้งหนี้แบบอิเล็กทรอนิกส์",
    "PaymentMethod_Credit card (automatic)": "จ่ายบัตรเครดิต (หักอัตโนมัติ)",
    "PaymentMethod_Electronic check": "จ่ายเช็คอิเล็กทรอนิกส์",
    "PaymentMethod_Mailed check": "จ่ายเช็คทางไปรษณีย์",
}
GROUP_NUM = [f for f in MODEL_FEATURES if KINDS.get(f) == "numeric"]
GROUP_BIN = [f for f in MODEL_FEATURES if KINDS.get(f) != "numeric"]


def build_input(values: dict) -> pd.DataFrame:
    """สร้าง DataFrame 1 แถว เรียงคอลัมน์ตรงกับตอนเทรนเป๊ะ (ข้อมูลทีมไม่ได้สเกล)"""
    return pd.DataFrame([values], columns=MODEL_FEATURES)


# ─────────────────────────────  ส่วนหัว  ─────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="kicker">Workshop สัปดาห์ที่ 11 · Deploy</div>
  <h1>📉 Telco Churn Predictor</h1>
  <p>ระบบทำนายลูกค้าที่มีแนวโน้ม <b>ยกเลิกบริการ (Churn)</b> — ห่อโมเดลสุดท้ายของทีมเป็นแอปที่ใช้งานได้จริง<br>
     วิชา 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล</p>
  <div class="chips">
    <span class="chip solid">ทีม รีเทค · 006-022</span>
    {''.join(f'<span class="chip">{code} · {name}</span>' for code, name in TEAM)}
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ─────────────────────────────  แท็บหลัก  ─────────────────────────────
tab_predict, tab_model, tab_help = st.tabs(["🎯  ประเมินความเสี่ยง", "🧠  ข้อมูลโมเดล", "📘  วิธีใช้ & ข้อจำกัด"])

# ══════════════ แท็บ 1: ประเมิน ══════════════
with tab_predict:
    col_form, col_result = st.columns([1.08, 1], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown("#### ข้อมูลลูกค้า")
            st.markdown(
                f'<div class="section-note">ฟอร์มสร้างจากฟีเจอร์จริงของโมเดล {len(MODEL_FEATURES)} ตัว '
                f'(ตรวจจาก <code>model.feature_names_in_</code>) — ไม่ขาด ไม่เกิน</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            values = {}
            with st.form("churn_form"):
                st.markdown("**① ตัวเลขการใช้งาน**")
                c1, c2 = st.columns(2)
                for i, f in enumerate(GROUP_NUM):
                    r = RANGES.get(f, {"min": 0.0, "max": 100.0})
                    with (c1 if i % 2 == 0 else c2):
                        values[f] = st.number_input(
                            LABELS.get(f, f),
                            min_value=float(r["min"]), max_value=float(r["max"]),
                            value=float(round((r["min"] + r["max"]) / 2, 2)), step=1.0,
                            help=HELP.get(f, ""),
                        )
                st.divider()
                st.markdown("**② บริการและวิธีชำระเงิน** (ติ๊ก = ใช้บริการนั้น)")
                c3, c4 = st.columns(2)
                for i, f in enumerate(GROUP_BIN):
                    with (c3 if i % 2 == 0 else c4):
                        values[f] = 1 if st.checkbox(LABELS.get(f, f), value=False, help=HELP.get(f, "")) else 0

                st.write("")
                submitted = st.form_submit_button("ประเมินความเสี่ยง", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown("#### ผลการประเมิน")
            if not submitted:
                st.info("กรอกข้อมูลด้านซ้าย แล้วกด **ประเมินความเสี่ยง** เพื่อดูผล")
                st.markdown(
                    '<div class="section-note">ผลลัพธ์จะแสดงเป็น % ความน่าจะเป็นยกเลิกบริการ '
                    'พร้อมข้อเสนอเชิงธุรกิจที่ฝ่ายดูแลลูกค้านำไปใช้ได้ทันที</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("กำลังประมวลผลด้วยโมเดลของทีม..."):
                    payload = build_input(values)
                    pred = int(model.predict(payload)[0])
                    proba = model.predict_proba(payload)[0]
                    p = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

                st.metric("โอกาสยกเลิกบริการ (Churn)", f"{p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))

                if pred == 1:
                    st.markdown(f"""
                    <div class="badge risk">
                      <div class="tag">ผลการทำนาย</div>
                      <h3>⚠️ เสี่ยงยกเลิกบริการ</h3>
                      <p><b>สิ่งที่ควรทำ:</b> ติดต่อลูกค้าเชิงรุก — เสนอโปรโมชันต่อสัญญา/ส่วนลด
                      หรือย้ายไปแพ็กเกจที่คุ้มกว่า และมอบหมายเจ้าหน้าที่ดูแลเฉพาะราย</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="badge safe">
                      <div class="tag">ผลการทำนาย</div>
                      <h3>✅ ความเสี่ยงต่ำ — น่าจะยังใช้บริการต่อ</h3>
                      <p><b>สิ่งที่ควรทำ:</b> ดูแลตามปกติ เก็บไว้ในกลุ่ม remarketing
                      และติดตามผลอีกครั้งเมื่อพฤติกรรมการใช้งานเปลี่ยน</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="mini">
                  <div><span>ระดับความเสี่ยง</span><strong>{'สูง' if p >= 0.5 else 'ต่ำ'}</strong></div>
                  <div><span>เกณฑ์ตัดสินใจ</span><strong>0.50</strong></div>
                  <div><span>ความแม่นโมเดล</span><strong>{info['test_accuracy']:.4f}</strong></div>
                </div>""", unsafe_allow_html=True)

                with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล และเหตุผลประกอบ"):
                    st.dataframe(payload, use_container_width=True)
                    st.caption("ข้อมูลของทีมไม่ได้สเกล → ส่งค่าจริงเข้าโมเดลได้เลย")
                    if values.get("tenure", 99) <= 16.5 and values.get("InternetService_Fiber optic") == 1:
                        st.warning("กฎสำคัญของโมเดล: **ลูกค้าใหม่ (tenure ≤ ~16 เดือน) + ใช้ไฟเบอร์** = กลุ่มเสี่ยงยกเลิกสูงสุด")
                    elif values.get("Contract_Two year") == 1:
                        st.info("ลูกค้าที่ทำสัญญา 2 ปี เป็นกลุ่มที่โมเดลมองว่าความเสี่ยงยกเลิกต่ำ")

# ══════════════ แท็บ 2: ข้อมูลโมเดล ══════════════
with tab_model:
    st.markdown("#### สรุปโมเดลสุดท้ายของทีม")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ชนิดโมเดล", "Decision Tree")
    m2.metric("max_depth", str(model.get_params().get("max_depth")))
    m3.metric("ความแม่น (test)", f"{info['test_accuracy']:.4f}")
    m4.metric("ความแม่น (5-fold CV)", f"{info['cv_accuracy']:.4f}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**ฟีเจอร์ที่โมเดลใช้จริง** (จาก `model.feature_names_in_`)")
            st.code("\n".join(MODEL_FEATURES), language="text")
            st.caption(f"รวม {len(MODEL_FEATURES)} ฟีเจอร์ · ตัวเลขต่อเนื่อง {len(GROUP_NUM)} · ตัวแปร 0/1 {len(GROUP_BIN)}")
    with right:
        with st.container(border=True):
            st.markdown("**ข้อมูลและข้อกำหนด**")
            st.write(f"- ข้อมูลเทรน: **{info['n_rows']:,} แถว** (Telco Customer Churn · Kaggle)")
            st.write(f"- สเกลข้อมูล: **{'สเกลแล้ว' if info.get('scaled') else 'ไม่ได้สเกล'}** → ส่งค่าจริงเข้าโมเดลได้เลย")
            st.write("- ผลลัพธ์: `1` = ยกเลิกบริการ · `0` = ยังใช้บริการอยู่")
            st.write(f"- เกณฑ์ตัดสินใจ: **{info['threshold']}**")
            st.write(f"- ไฟล์โมเดล: `{info['model_file']}` (~3 KB)")

# ══════════════ แท็บ 3: วิธีใช้ ══════════════
with tab_help:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### วิธีใช้")
            st.markdown(
                "1. กรอกตัวเลขการใช้งานของลูกค้า (ระยะเวลาเป็นลูกค้า ค่าบริการรายเดือน ยอดสะสม)\n"
                "2. ติ๊กช่องบริการ/วิธีชำระเงินที่ลูกค้ารายนั้นใช้\n"
                "3. กด **ประเมินความเสี่ยง** → ระบบแสดง % ความน่าจะเป็นยกเลิกบริการและข้อเสนอที่ควรทำ\n"
                "4. ใช้ **ข้อมูลโมเดล** เพื่อดูว่าฟีเจอร์ใดถูกใช้จริง และดูความแม่นของโมเดล"
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### ข้อจำกัดของโมเดล")
            st.markdown(
                "1. **ข้อมูลไม่สมดุล** — ลูกค้าที่ Churn มีเพียง 26.5% (Dummy ได้ 0.7346) "
                "recall ของคลาส Churn อยู่ที่ 0.39 → ใช้เป็น *สัญญาณเตือน* ไม่ใช่คำตัดสินสุดท้าย\n"
                "2. **ฟีเจอร์เป็นค่า 0/1 จาก one-hot** — ควรอ่าน `help=` ประกอบทุกช่อง\n"
                "3. **ช่วงค่าที่รับ** จำกัดตามข้อมูลเทรน (tenure 0–72 เดือน) การกรอกนอกช่วงคือการ extrapolate\n"
                "4. ต้องการฟอร์มสั้นลง ใช้ผล GA สัปดาห์ที่ 10 (`tenure` + `InternetService_Fiber optic`) "
                "ได้ความแม่นเท่าเดิม (CV 0.7901)"
            )

st.markdown(f"""
<div class="footer">
  จัดทำโดย <b>ทีม รีเทค (006-022)</b> — {' · '.join(f'{code} {name}' for code, name in TEAM)}<br>
  วิชา 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · Workshop สัปดาห์ที่ 11 · ข้อมูล: Telco Customer Churn (Kaggle)
</div>
""", unsafe_allow_html=True)
