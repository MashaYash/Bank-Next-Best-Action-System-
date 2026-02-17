import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =========================
# PATHS (robust)
# =========================
BASE_DIR = str(Path(_file).resolve().parent) if "file_" in globals() else os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    alt = os.path.join(BASE_DIR, "Data")
    if os.path.exists(alt):
        DATA_DIR = alt

MODEL_PATH = os.path.join(BASE_DIR, "nbp_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "input_features.pkl")

# =========================
# HELPERS
# =========================
def find_file(patterns, required=True):
    for pat in patterns:
        matches = glob.glob(os.path.join(DATA_DIR, pat))
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    if required:
        raise FileNotFoundError(f"Missing file. Looked for {patterns} inside {DATA_DIR}")
    return None

def normalize_nic(x):
    return str(x).strip()

def pick_col(df, candidates):
    if df is None or df.empty:
        return None
    m = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None

def calc_emi(principal, annual_rate, months):
    # EMI = P*r*(1+r)^n / ((1+r)^n - 1)
    if principal <= 0 or months <= 0:
        return 0.0
    r = (annual_rate / 100.0) / 12.0
    if r == 0:
        return principal / months
    num = principal * r * (1 + r) ** months
    den = ((1 + r) ** months) - 1
    return (num / den) if den != 0 else 0.0

def soft_actions(product_label):
    p = str(product_label).lower()

    if "wealth" in p or "invest" in p:
        return [
            "Offer: Wealth/Investment plan (goal-based investing).",
            "Bundle: FD rollover + RM appointment + investment starter pack.",
            "Action: Invite to portfolio review + risk profiling."
        ]
    if "loan" in p:
        return [
            "Offer: Personal/SME loan option (based on profile).",
            "Bundle: FD-secured loan + insurance add-on (if applicable).",
            "Action: Run EMI simulation + share required documents."
        ]
    if "credit" in p or "card" in p:
        return [
            "Offer: Credit card with cashback/points.",
            "Bundle: Card + bill pay + alerts + insurance add-on.",
            "Action: Apply instantly + set spending limit."
        ]
    if "saving" in p or "savings" in p or "account" in p or "rd" in p:
        return [
            "Offer: Savings plan / recurring deposit (RD).",
            "Bundle: Auto-transfer + FD maturity sweep to savings.",
            "Action: Set monthly savings goal + reminders."
        ]
    if "fixed" in p or "deposit" in p or "fd" in p:
        return [
            "Offer: Fixed Deposit option (tenure/interest payout based).",
            "Bundle: FD rollover + maturity alerts + nominee update.",
            "Action: Explain FD benefits + confirm tenure + complete setup."
        ]

    return [
        "Offer: Best-matching product.",
        "Bundle: Related add-ons to increase value.",
        "Action: Schedule follow-up and explain benefits."
    ]

def call_script(name, nic, product, prob, purpose=None):
    nm = name if name else "Customer"
    purpose_txt = f" (purpose: {purpose})" if purpose and purpose != "Auto" else ""
    return (
        f"Hi {nm}, this is from the bank.\n\n"
        f"Based on your profile{purpose_txt}, we have a suitable offer for you: "
        f"{product} (confidence ~ {prob*100:.1f}%).\n\n"
        f"Would you like me to explain the benefits and help you get started today?\n\n"
        f"(Reference NIC: {nic})"
    )

# =========================
# MODEL / FEATURES
# =========================
@st.cache_resource
def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"nbp_model.pkl not found at: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"input_features.pkl not found at: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feat = joblib.load(FEATURES_PATH)
    return model, list(feat)

# =========================
# LOAD DATA (existing customers)
# =========================
@st.cache_data
def load_bank_data():
    cust_xlsx = find_file(["NICs and CustomerDetails*.xlsx", "NICs and CustomerDetails*.xls"])
    loan_xlsx = find_file(["Loan_Dataset_Synthetic*.xlsx", "Loan_Dataset*.xlsx"], required=False)

    xls = pd.ExcelFile(cust_xlsx)

    cust_sheet = None
    for s in xls.sheet_names:
        if "customer" in s.lower():
            cust_sheet = s
            break
    if cust_sheet is None:
        cust_sheet = xls.sheet_names[0]

    cust = pd.read_excel(xls, sheet_name=cust_sheet)
    loan_df = pd.read_excel(loan_xlsx) if loan_xlsx else pd.DataFrame()

    nic_col = pick_col(cust, ["NIC", "NationalID", "National ID"])
    if nic_col:
        cust[nic_col] = cust[nic_col].astype(str).map(normalize_nic)

    nic_col_loan = pick_col(loan_df, ["NIC", "NationalID", "National ID"])
    if nic_col_loan:
        loan_df[nic_col_loan] = loan_df[nic_col_loan].astype(str).map(normalize_nic)

    return cust, loan_df

def get_existing_customer_snapshot(nic, cust, loan_df):
    nic = normalize_nic(nic)
    nic_col = pick_col(cust, ["NIC", "NationalID", "National ID"])
    if not nic_col:
        return False, {"reason": "Customer NIC column not found in customer sheet."}

    c = cust[cust[nic_col].astype(str) == nic].head(1)
    if c.empty:
        return False, {"reason": "NIC not found in customer dataset."}

    row = c.iloc[0].to_dict()
    one = pd.DataFrame([row])

    name_col = pick_col(one, ["Name", "FullName", "CustomerName", "Customer Name"])
    age_col  = pick_col(one, ["Age", "CustomerAge", "Customer Age"])
    town_col = pick_col(one, ["Town", "City", "Location", "Address"])

    name = row.get(name_col) if name_col else None
    age = int(row.get(age_col)) if (age_col and pd.notna(row.get(age_col))) else 0
    town = str(row.get(town_col)) if (town_col and pd.notna(row.get(town_col))) else "Unknown"

    loan_total = 0.0
    loan_rate_avg = 0.0
    salary = 0.0

    if loan_df is not None and not loan_df.empty:
        nic_col_loan = pick_col(loan_df, ["NIC", "NationalID", "National ID"])
        amt_col = pick_col(loan_df, ["LoanAmount", "Loan Amount", "RequiredLoanAmount"])
        rate_col = pick_col(loan_df, ["InterestRate", "Interest Rate", "Rate"])
        salary_col = pick_col(loan_df, ["MonthlySalary", "Salary", "Monthly Salary"])

        if nic_col_loan:
            tmp = loan_df[loan_df[nic_col_loan].astype(str) == nic]
            if not tmp.empty and amt_col:
                loan_total = float(tmp[amt_col].fillna(0).sum())
            if not tmp.empty and rate_col:
                loan_rate_avg = float(tmp[rate_col].fillna(0).mean())
            if not tmp.empty and salary_col:
                salary = float(tmp[salary_col].fillna(0).max())

    return True, {
        "nic": nic,
        "name": str(name) if name else None,
        "age": age,
        "town": town,
        "loan_total": loan_total,
        "loan_avg_rate": loan_rate_avg,
        "salary": salary
    }

# =========================
# FEATURE ROW BUILDER (must match training)
# =========================
def build_feature_row(features, age=0, town="Unknown",
                      fd_total=0.0, fd_rate=0.0,
                      loan_total=0.0, loan_rate=0.0, salary=0.0):

    base = {
        "Age": age,
        "FD_Total_Amount": fd_total,
        "FD_Avg_Rate": fd_rate,
        "Loan_Total": loan_total,
        "Loan_Avg_Rate": loan_rate,
        "Salary": salary
    }

    df = pd.DataFrame([base])
    df["Town"] = str(town)

    df = pd.get_dummies(df, columns=["Town"], dummy_na=False)

    row = pd.DataFrame(columns=features)
    row.loc[0] = 0
    for col in df.columns:
        if col in row.columns:
            row.at[0, col] = df.at[0, col]
    return row

def predict_top3_and_all(model, X):
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    pairs = list(zip(classes, proba))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:3], pairs

# =========================
# PURPOSE OVERRIDE (IMPORTANT)
# =========================
def loan_purpose_recommendation(salary, loan_amount, loan_rate, tenure_months):
    """
    Purpose-based recommendation for Loan when ML model doesn't contain 'Loan' class.
    We create a realistic confidence using affordability ratio.
    """
    emi = calc_emi(loan_amount, loan_rate, tenure_months)
    if salary <= 0 or emi <= 0:
        return "Personal Loan", 0.55, emi, "Salary/EMI inputs are missing."

    ratio = emi / salary  # how much of salary goes to EMI

    # simple bank-style affordability scoring (heuristic)
    # <= 0.30 : very affordable
    # 0.30-0.45 : medium
    # >0.45 : risky
    if ratio <= 0.30:
        conf = 0.92
        note = "Affordable EMI (low EMI-to-salary ratio)."
    elif ratio <= 0.45:
        conf = 0.78
        note = "Moderate affordability (EMI-to-salary ratio is medium)."
    else:
        conf = 0.60
        note = "Higher risk (EMI-to-salary ratio is high). Consider lower amount/tenure."

    return "Personal Loan", conf, emi, note

# =========================
# UI
# =========================
st.set_page_config(page_title="Bank Next Best Action", page_icon="🏦", layout="centered")
st.title("🏦 Bank Next Best Action System")
st.caption("NIC-based recommendations with probabilities + Action Plan + Officer Call Script")

if not os.path.exists(DATA_DIR):
    st.error(f"Missing data folder. Create 'data' or 'Data' folder next to app.py.\n\nExpected: {DATA_DIR}")
    st.stop()

try:
    model, input_features = load_model_and_features()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.info("Fix: keep these files next to app.py:\n- nbp_model.pkl\n- input_features.pkl")
    st.stop()

try:
    cust, loan_df = load_bank_data()
except Exception as e:
    st.error(f"Dataset load error: {e}")
    st.stop()

# helpful (optional): show classes so you can explain to supervisor
with st.expander("Show model classes (what the ML model can predict)"):
    st.write(list(model.classes_))

nic = st.text_input("Enter NIC", placeholder="e.g., 200278501512")

c1, c2 = st.columns([1, 1])
with c1:
    run_btn = st.button("Get Recommendation")
with c2:
    clear_btn = st.button("Clear Results")

if "result" not in st.session_state:
    st.session_state.result = None

if clear_btn:
    st.session_state.result = None
    st.rerun()

if run_btn:
    if not nic.strip():
        st.warning("Please enter NIC.")
        st.stop()

    nic_norm = normalize_nic(nic)
    found, snap_or_reason = get_existing_customer_snapshot(nic_norm, cust, loan_df)

    st.session_state.result = {
        "nic": nic_norm,
        "found": found,
        "data": snap_or_reason
    }

# =========================
# DISPLAY FLOW
# =========================
if st.session_state.result is not None:
    nic_norm = st.session_state.result["nic"]
    found = st.session_state.result["found"]
    data = st.session_state.result["data"]

    # -------------------------
    # EXISTING CUSTOMER
    # -------------------------
    if found:
        st.success("✅ Existing customer found.")

        st.subheader("Customer Snapshot")
        st.write(f"Name: *{data.get('name') or 'N/A'}*")
        st.write(f"NIC: *{data.get('nic')}*")
        st.write(f"Town: *{data.get('town')}* | Age: *{data.get('age')}*")
        st.write(f"Loan Total: *{data.get('loan_total'):,.2f}* | Salary: *{data.get('salary'):,.2f}*")

        purpose = st.selectbox(
            "Purpose (optional) - what did the customer come for?",
            ["Auto", "FD", "Loan", "Savings"]
        )

        fd_amount = 0.0
        fd_rate = 0.0
        loan_amount = 0.0
        loan_rate = 0.0
        tenure = 36

        if purpose == "FD":
            fd_amount = st.number_input("FD Amount (LKR)", min_value=0.0, value=200000.0, step=10000.0)
            fd_rate = st.number_input("FD Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)

        if purpose == "Loan":
            loan_amount = st.number_input("Required Loan Amount (LKR)", min_value=0.0, value=500000.0, step=10000.0)
            loan_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=14.0, step=0.1)
            tenure = st.number_input("Tenure (months)", min_value=1, value=36, step=1)

        # ML features (still run ML for cross-sell suggestions)
        X = build_feature_row(
            input_features,
            age=data.get("age", 0),
            town=data.get("town", "Unknown"),
            fd_total=fd_amount if purpose == "FD" else 0.0,
            fd_rate=fd_rate if purpose == "FD" else 0.0,
            loan_total=loan_amount if purpose == "Loan" else data.get("loan_total", 0.0),
            loan_rate=loan_rate if purpose == "Loan" else data.get("loan_avg_rate", 0.0),
            salary=data.get("salary", 0.0),
        )

        top3, all_pairs = predict_top3_and_all(model, X)

        # ✅ PURPOSE LOGIC:
        # - Auto: use ML top
        # - FD/Savings: use ML top (because model has FD/Savings)
        # - Loan: OVERRIDE to Loan recommendation (because model doesn't contain Loan class)
        if purpose == "Loan":
            top_label, top_prob, emi, note = loan_purpose_recommendation(
                salary=data.get("salary", 0.0),
                loan_amount=float(loan_amount),
                loan_rate=float(loan_rate),
                tenure_months=int(tenure),
            )
            st.info(f"Loan purpose selected → showing Loan as top recommendation. ({note})")

        else:
            top_label, top_prob = top3[0]

        st.subheader("Top Recommendation (Purpose)")
        st.write(f"✅ *{top_label} — {top_prob*100:.2f}%*")

        st.subheader("Other High-Probability (Cross-sell) Recommendations (from ML)")
        for i, (lab, p) in enumerate(top3, start=1):
            st.write(f"{i}. *{lab} — {p*100:.2f}%*")

        st.subheader("Bank Action Plan")
        for a in soft_actions(top_label):
            st.write(f"✅ {a}")

        if purpose == "Loan":
            st.subheader("EMI Simulation")
            st.write(f"Monthly EMI (approx): *{emi:,.2f} LKR*")

        if purpose == "Savings":
            st.subheader("Savings Guidance")
            goal = st.selectbox("Savings Goal (optional)", ["None", "Children Education", "Vehicle", "House", "Medical", "Retirement"])
            if goal != "None":
                st.write(f"✅ Goal noted: *{goal}*")
                st.write("✅ Suggest: RD / Savings plan aligned to the goal.")
                st.write("✅ If relevant: Insurance / Education loan guidance.")

        st.subheader("Officer Call Script (Copy-ready)")
        st.text_area(
            "Script",
            value=call_script(data.get("name"), nic_norm, top_label, top_prob, purpose=purpose),
            height=180
        )

    # -------------------------
    # NEW CUSTOMER
    # -------------------------
    else:
        st.warning("⚠️ NIC not found. New customer scenario.")
        st.info("Select what customer came for → minimal inputs → recommendation + action plan + call script.")

        product_type = st.selectbox("Customer came for", ["FD", "Loan", "Savings"])

        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        town = st.text_input("Town", value="Colombo")

        fd_amount = 0.0
        fd_rate = 0.0
        loan_amount = 0.0
        loan_rate = 0.0
        tenure = 36
        salary = 0.0
        goal = None

        if product_type == "FD":
            fd_amount = st.number_input("FD Amount (LKR)", min_value=0.0, value=200000.0, step=10000.0)
            fd_rate = st.number_input("FD Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)

        elif product_type == "Loan":
            salary = st.number_input("Monthly Salary (LKR)", min_value=0.0, value=150000.0, step=10000.0)
            loan_amount = st.number_input("Required Loan Amount (LKR)", min_value=0.0, value=500000.0, step=10000.0)
            loan_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=14.0, step=0.1)
            tenure = st.number_input("Tenure (months)", min_value=1, value=36, step=1)

        elif product_type == "Savings":
            goal = st.selectbox("Savings Goal", ["Children Education", "Vehicle", "House", "Medical", "Retirement"])
            salary = st.number_input("Monthly Salary (optional)", min_value=0.0, value=120000.0, step=10000.0)

        gen = st.button("Generate Recommendation")

        if gen:
            # We still run ML to show cross-sell suggestions
            X = build_feature_row(
                input_features,
                age=int(age),
                town=str(town),
                fd_total=float(fd_amount),
                fd_rate=float(fd_rate),
                loan_total=float(loan_amount),
                loan_rate=float(loan_rate),
                salary=float(salary),
            )
            top3, _ = predict_top3_and_all(model, X)

            # ✅ PURPOSE LOGIC:
            if product_type == "Loan":
                top_label, top_prob, emi, note = loan_purpose_recommendation(
                    salary=float(salary),
                    loan_amount=float(loan_amount),
                    loan_rate=float(loan_rate),
                    tenure_months=int(tenure),
                )
                st.info(f"Loan purpose selected → showing Loan as top recommendation. ({note})")
            else:
                top_label, top_prob = top3[0]

            st.subheader("Top Recommendation (Purpose)")
            st.write(f"✅ *{top_label} — {top_prob*100:.2f}%*")

            st.subheader("Other High-Probability (Cross-sell) Recommendations (from ML)")
            for i, (lab, p) in enumerate(top3, start=1):
                st.write(f"{i}. *{lab} — {p*100:.2f}%*")

            st.subheader("Bank Action Plan")
            for a in soft_actions(top_label):
                st.write(f"✅ {a}")

            if product_type == "Loan":
                st.subheader("EMI Simulation")
                st.write(f"Monthly EMI (approx): *{emi:,.2f} LKR*")

            if product_type == "Savings" and goal:
                st.subheader("Savings Guidance")
                st.write(f"✅ Goal noted: *{goal}*")
                st.write("✅ Suggest: RD / Savings plan aligned to the goal.")
                st.write("✅ If relevant: Insurance / Education loan guidance.")

            st.subheader("Officer Call Script (Copy-ready)")
            st.text_area(
                "Script",
                value=call_script(None, nic_norm, top_label, top_prob, purpose=product_type),
                height=180
            )