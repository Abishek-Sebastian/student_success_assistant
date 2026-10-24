"""Minimal Streamlit demo for lesson explanation feedback."""
import sys
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'task2'))
from src.evaluator import load_knowledge,evaluate_one

st.set_page_config(page_title='Student Success Assistant | Explanation Coach',page_icon='🎓',layout='centered')
st.title('Student Success Assistant')
st.subheader('Lesson explanation feedback')
st.caption('Feedback is generated from the lesson rubric and retrieved development examples. Treat it as learning support, not a high-stakes decision.')
knowledge=load_knowledge()
lesson_id=st.selectbox('Lesson',options=list(knowledge),format_func=lambda x:f"{x} · {knowledge[x]['title']}")
explanation=st.text_area('Student explanation',height=160,placeholder='Explain the concept in your own words…')
if st.button('Evaluate explanation',type='primary',disabled=not explanation.strip()):
    with st.spinner('Evaluating against the lesson rubric…'):
        try:
            result=evaluate_one({'explanation_id':'STREAMLIT_DEMO','lesson_id':lesson_id,'explanation_text':explanation},knowledge)
        except Exception as exc:
            st.error(f'Could not reach the configured LLM API: {exc}')
        else:
            st.markdown(f"### Verdict: **{result['verdict'].title()}**")
            st.write('**What you got right**')
            st.write(result['got_right'])
            st.write('**What to revisit**')
            st.write(result['missed'])
            st.write('**Hint**')
            st.write(result['hint'])
            st.caption(f"Status: {result['status']} · Latency: {result.get('latency_seconds', 0):.2f}s · Retrieved examples: {result.get('retrieved_cases', 0)}")
            if result['status']=='failed_after_retries':
                st.warning('The model response failed validation. This is not a reliable grading result; retry the request.')
