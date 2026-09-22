# 📉 Telco Churn Predictor — ทีมรีเทค (Workshop สัปดาห์ที่ 11)

ชุดไฟล์แอป Streamlit ที่สร้างตาม **เนื้อหาเวิร์กชอปสัปดาห์ที่ 12 (ขั้น 1–9)**
ห่อโมเดลสุดท้ายของทีมให้เป็นแอปที่คนอื่นกดใช้ได้จริง
ข้อมูล: Telco Customer Churn (Kaggle `blastchar/telco-customer-churn`) · โจทย์: ทำนายลูกค้าที่จะยกเลิกบริการ

## ไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่ | ขั้นในเวิร์กชอป |
|---|---|---|
| `app.py` | แอป Streamlit (โหลดโมเดล → ฟอร์ม 12 ฟีเจอร์ → ทำนาย → คำแนะนำธุรกิจ) | ขั้น 3, 4, 7, 8 |
| `telco_churn.joblib` | โมเดลสุดท้ายของทีม `DecisionTreeClassifier(max_depth=3, random_state=42)` | ขั้น 2 |
| `model_info.json` | ข้อมูลกำกับโมเดล: รายชื่อฟีเจอร์ · ชนิด (ตัวเลข/0-1) · ช่วงค่า · ความแม่น | ขั้น 2 |
| `requirements.txt` | ไลบรารี + เวอร์ชันที่ใช้จริง (ล็อกให้ตรงกับตอนบันทึกโมเดล) | ขั้น 9 |
| `README.md` | เอกสารนี้ (วิธีรัน + ข้อจำกัดโมเดล) | ขั้น 9 |
| `.gitignore` | กันไม่ให้ `venv/`, `__pycache__/`, ไฟล์ซ้ำขึ้น repo | ขั้น 9 |
| `PROMPT_LOG.md` | สรุป prompt ที่ใช้สั่ง Claude ตามขั้น 1–9 | ภาค C |
| `run.bat` | ดับเบิลคลิกเพื่อเปิดแอป | ขั้น 4 |

## ขั้น 2 — ผลตรวจไฟล์โมเดล (ยึดจากตัวโมเดลจริง)

```
โมเดล      : DecisionTreeClassifier(max_depth=3, random_state=42)
ฟีเจอร์     : 12 คอลัมน์ (model.feature_names_in_)
             tenure, MonthlyCharges, TotalCharges, SeniorCitizen,
             Contract_One year, Contract_Two year,
             InternetService_Fiber optic, InternetService_No,
             PaperlessBilling_Yes,
             PaymentMethod_Credit card (automatic),
             PaymentMethod_Electronic check, PaymentMethod_Mailed check
สเกลข้อมูล : ไม่สเกล → ส่งค่าจริงเข้าโมเดลได้เลย
ความแม่น   : test 0.7871 · 5-fold CV 0.7901 (ข้อมูล 7,043 แถว)
ผลลัพธ์     : 1 = ลูกค้ายกเลิกบริการ · 0 = ยังใช้บริการอยู่ · เกณฑ์ตัดสินใจ 0.5
```

> แอปสร้างฟอร์ม**อัตโนมัติจาก `model.feature_names_in_`** — ถ้าเปลี่ยนโมเดล ฟอร์มจะเปลี่ยนตามเอง
> ไม่มีทางเกิด error "คอลัมน์ไม่ตรง" (ปัญหาที่เจอบ่อยในขั้น 5 ของเวิร์กชอป)

## วิธีรันในเครื่อง (ขั้น 4)

```powershell
cd C:\Users\T3flx6\hee\workshop11_retech
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

เปิด http://localhost:8501

> ทางลัด: ดับเบิลคลิก `run.bat`
> ถ้า PowerShell บล็อก activate: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## ขั้น 9 — เช็กลิสต์ก่อน deploy (สัปดาห์ที่ 13)

- [x] พาธไฟล์โมเดลเป็นแบบ**สัมพัทธ์** (`Path(__file__).resolve().parent / "telco_churn.joblib"`)
- [x] ขนาดไฟล์โมเดลเล็ก (~3 KB) → push ขึ้น GitHub ได้สะดวก
- [x] ไม่มีข้อมูลส่วนบุคคล/ข้อมูลลูกค้าจริงในโฟลเดอร์ (ใช้ข้อมูลสาธารณะจาก Kaggle)
- [x] ไม่มีโฟลเดอร์ `venv/` ขึ้น repo (อยู่ใน `.gitignore`)
- [ ] ตัดสินใจว่า repository จะเป็นสาธารณะหรือส่วนตัว
- [ ] บน Streamlit Cloud เลือก main file = `app.py`

## ข้อจำกัดของโมเดล

1. **ข้อมูลไม่สมดุล** — ลูกค้าที่ Churn มีเพียง 26.5% (Dummy ทายคลาสใหญ่สุดได้ 0.7346)
   recall ของคลาส Churn อยู่ที่ 0.39 → ใช้เป็น "สัญญาณเตือนให้ติดต่อลูกค้า" ไม่ใช่ตัวตัดสินสุดท้าย
2. **ฟีเจอร์เป็นค่า 0/1 ที่มาจาก one-hot** — ผู้ใช้ต้องเข้าใจว่าการติ๊กแต่ละช่องหมายถึงอะไร (มี `help=` อธิบายทุกช่อง)
3. **ช่วงค่าที่รับ** — ฟอร์มจำกัดตามช่วงข้อมูลที่ใช้เทรน (tenure 0–72, MonthlyCharges 18.25–118.75, TotalCharges 18.8–8684.8)
4. ถ้าต้องการฟอร์มสั้นลง ใช้ผล GA สัปดาห์ที่ 10 (`tenure` + `InternetService_Fiber optic`) ได้ความแม่นเท่าเดิม (0.7901)

## ถ้าดาวน์โหลดไฟล์ซ้ำ

Windows อาจเซฟเป็น `app (1).py` แทนการแทนที่ไฟล์เดิม ทำให้รันไฟล์เก่าซ้ำ ๆ
ให้ลบไฟล์เก่าก่อนวางไฟล์ใหม่ และเช็กด้วย `dir *.py` ว่ามีไฟล์เดียว
