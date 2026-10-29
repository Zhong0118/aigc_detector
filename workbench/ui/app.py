"""Isolated preview workbench entry: `streamlit run workbench/ui/app.py` (see scripts/workbench_preview.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from workbench.ui.common import inject_css  # noqa: E402
from workbench.ui.pages import experiments_page, history_page, image_page, text_page  # noqa: E402

st.set_page_config(page_title="AIGC 检测工作台（预览）", layout="wide")
inject_css()
st.navigation([
    st.Page(text_page, title="文本检测", icon=":material/article:", url_path="text", default=True),
    st.Page(image_page, title="图片检测", icon=":material/image_search:", url_path="image"),
    st.Page(experiments_page, title="实验对照", icon=":material/table_chart:", url_path="experiments"),
    st.Page(history_page, title="历史报告", icon=":material/history:", url_path="history"),
], position="top").run()
