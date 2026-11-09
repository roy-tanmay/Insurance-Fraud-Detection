import os

import requests
import streamlit as st

from options import OPTIONS

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# starting values for the form (the sample claim)
SAMPLE = {
    "base_policy": "Liability",
    "fault": "Policy Holder",
    "make": "Chevrolet",
    "past_number_of_claims": "none",
    "address_change_claim": "no change",
    "deductible": 400,
    "vehicle_price": "20000 to 29000",
    "month": "Jul",
    "month_claimed": "Jul",
    "age_of_policy_holder": "36 to 40",
    "number_of_suppliments": "none",
    "agent_type": "External",
    "age_of_vehicle": "7 years",
}

st.set_page_config(page_title="Insurance Fraud Check", page_icon="🔍")
st.title("Insurance claim fraud check")
st.caption("Enter the claim details and score it. A claim is flagged if it falls among the riskiest ~10%.")

claim = {}
cols = st.columns(2)
for i, (name, options) in enumerate(OPTIONS.items()):
    label = name.replace("_", " ").capitalize()
    claim[name] = cols[i % 2].selectbox(label, options, index=options.index(SAMPLE[name]))

if st.button("Score claim", type="primary"):
    try:
        resp = requests.post(f"{API_URL}/predict", json=claim, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
    else:
        st.metric("Fraud score", f"{result['score']:.3f}", help="Higher means more fraud-like")
        st.write(f"Flag cutoff: **{result['cutoff']:.3f}** (claims scoring above this are in the riskiest ~10%)")
        st.progress(min(result["score"] / result["cutoff"], 1.0) / 1.0)
        if result["flag"]:
            st.error("Flagged for review")
        else:
            st.success("Not flagged. Below the review cutoff.")