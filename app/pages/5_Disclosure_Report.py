"""The responsible-disclosure report rendered by notebook 09, with downloads."""
import streamlit as st

import common
from mia.config import REPORTS_DIR

common.setup("Disclosure report")
report, table = REPORTS_DIR / "disclosure_report.md", REPORTS_DIR / "results_tables.csv"

if not report.exists():
    st.warning("The report has not been rendered yet. Run notebooks/09_disclosure_report.ipynb "
               "(`jupyter lab`, then run all cells), then press Refresh data.")
    st.stop()

try:
    text = report.read_text(encoding="utf-8")
except OSError as e:
    common.fail(e, "read the report")

c1, c2 = st.columns(2)
c1.download_button("Download report (Markdown)", text.encode(), file_name="disclosure_report.md", mime="text/markdown")
if table.exists():
    c2.download_button("Download results_tables.csv", table.read_bytes(), file_name="results_tables.csv", mime="text/csv")
else:
    c2.info("results_tables.csv appears after notebook 08.")
st.divider()
st.markdown(text)
