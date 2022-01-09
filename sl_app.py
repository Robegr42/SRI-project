import os
from pathlib import Path

import switer_api as api
import streamlit as st

state_vars = [
    "docs",
    "results",
    "model",
    "database",
    "from_i",
    "to_i",
    "query",
    "score",
    "top",
]
for var in state_vars:
    if var not in st.session_state:
        st.session_state[var] = None


def reset_search():
    st.session_state.results = []


def reset():
    st.session_state.docs = None
    st.session_state.model = None
    st.session_state.score = 0.2
    st.session_state.top = 10
    reset_search()


def show_result(result: dict):
    doc_id = result["doc_index"]
    meta = result["doc_metadata"]
    pos = result["pos"]
    score = result["weight"]
    expander_header = "Document {}".format(doc_id)
    if "title" in meta:
        expander_header = meta["title"].capitalize()
    with st.expander(f"{pos}. {expander_header}"):
        st.caption(f"Score: {score}")
        for k, v in meta.items():
            if k == "title" or v == "":
                continue
            st.caption(f"**{k.upper()}:** {v}")
        if st.session_state.docs is not None:
            st.markdown(st.session_state.docs[doc_id].capitalize())


st.title("Switer")

# Aviable databases
databases = os.listdir("database")

if not databases:
    st.write("No databases found")

col1, col2 = st.columns([1, 3])

with col1:
    database = st.selectbox("Select a database", databases)
    if st.session_state.docs is None or st.session_state.database != database:
        reset()
        st.session_state.docs = api.get_docs(database)
        st.session_state.database = database

    db_folder = Path("database") / database
    db_model_path = db_folder / "model"

    if not db_model_path.exists():
        st.error("Database model not found. Do want to build it?")
        if st.button("Build"):
            api.build_database_model(database)
            st.success("Database model built")
        else:
            st.stop()

    if st.session_state.model is None or st.session_state.database != database:
        st.session_state.model = api.IRModel(str(db_folder))
        st.session_state.database = database

with col2:
    query = st.text_input("Enter a query", placeholder="Write your query here")
    if st.session_state.query != query:
        st.session_state.query = query
        st.session_state.results = []

cols = st.columns(2)

with cols[0]:
    score_slider = st.slider(
        "Score threshold",
        0.0,
        1.0,
        0.2,
        help="Minimum score to show results\n\nScore 0 docs are not shown",
    )
    if st.session_state.score != score_slider:
        reset_search()
        st.session_state.score = score_slider

with cols[1]:
    limit = st.slider(
        "Top", 1, len(st.session_state.docs), 10, help="Max number of results to show"
    )
    if st.session_state.top != limit:
        reset_search()
        st.session_state.top = limit

if st.session_state.query:
    if not st.session_state.results:
        st.session_state.results = list(
            api.single_query(st.session_state.query, st.session_state.model)
        )
    results = []
    for i, res in enumerate(st.session_state.results):
        if i >= limit:
            break
        if res["weight"] < score_slider:
            continue
        results.append(res)

    if results:
        st.write(f"Found {len(results)} results")
        for res in results:
            show_result(res)
    else:
        msg = "No results found."
        if score_slider > 0:
            msg += " Try a lower score threshold"
        st.caption(msg)
