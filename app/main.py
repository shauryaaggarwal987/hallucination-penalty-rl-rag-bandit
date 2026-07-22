import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import os

# Add repo root
sys.path.append(str(Path(__file__).parent.parent))

from config import ACTIONS, DEFAULT_EPSILON, DEFAULT_TOP_K, DEFAULT_SENT_SIM_THRESHOLD
from rag.retrieve import retrieve
from llm.generate import LLMClient, generate_all_candidates
from scoring.reward import score_answer
from rl.bandit import EpsilonGreedyBandit
from utils.io import read_jsonl
from config import env_loaded, env_path
from utils.env_loader import get_redacted_key, get_gemini_key

st.set_page_config(page_title="Hallucination Penalty RAG Bandit", layout="wide")

st.title("Hallucination Penalty RL-RAG Bandit")
st.markdown("""
A transparent RAG system that uses **Reinforcement Learning (MAB)** to select the best answering strategy 
based on a **Groundedness - Hallucination Penalty** reward.
""")

# Sidebar settings
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox("LLM Provider", ["dryrun", "gemini"], index=0)
os.environ["LLM_PROVIDER"] = provider

# Need to reload client if provider changes?
# But we init new client each time for simplicity or cache it?
# We'll re-init.

epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.0, 1.0, DEFAULT_EPSILON, 0.05)
top_k = st.sidebar.slider("Top K Chunks", 1, 10, DEFAULT_TOP_K)
sim_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, DEFAULT_SENT_SIM_THRESHOLD, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("System Health (Debug)")
st.sidebar.write(f"`.env` loaded: **{'✅ Yes' if env_loaded else '❌ No'}**")
api_key_present = bool(get_gemini_key())
st.sidebar.write(f"API Key: **{'✅ ' + get_redacted_key() if api_key_present else '❌ Missing'}**")
st.sidebar.write(f"Selected Provider: **{provider}**")

from config import METADATA_JSON
from utils.io import safe_read_json
meta = safe_read_json(METADATA_JSON, [])
st.sidebar.write(f"Indexed Chunks: **{len(meta)}**")
if meta:
    unique_srcs = set([c.get('doc_name', 'unknown') for c in meta])
    st.sidebar.write(f"Unique Sources: **{len(unique_srcs)}**")

st.sidebar.markdown("---")
if st.sidebar.button("Re-Ingest Docs"):
    import rag.ingest
    rag.ingest.main()
    st.sidebar.success("Ingestion Complete!")

if st.sidebar.button("Re-Build Index"):
    import rag.build_index
    from rag.retrieve import load_resources
    rag.build_index.build_index()
    load_resources(force_reload=True)
    st.sidebar.success("Index Built! Memory Cache Purged.")

if st.sidebar.button("Reset Bandit State"):
    import config
    if config.BANDIT_STATE_PATH.exists():
        config.BANDIT_STATE_PATH.unlink()
    st.sidebar.success("Bandit State Reset!")

st.sidebar.markdown("---")
st.sidebar.write("### Actions (Strategies)")
for a in ACTIONS:
    st.sidebar.code(a)

# Main Area
query = st.text_input("Ask a question:", "What is RAG?")

if st.button("Generate Answer") and query:
    with st.spinner("Retrieving and Generating..."):
        # 1. Retrieve
        contexts, sources, hits = retrieve(query, top_k=top_k)
        
        # 2. Display Context
        with st.expander("Retrieved Context & Sources", expanded=False):
            for i, (ctx, src) in enumerate(zip(contexts, sources)):
                st.markdown(f"**{src}**")
                st.text(ctx)
                st.divider()

        # 3. Generate Candidates
        client = LLMClient(provider=provider)
        
        if provider == "gemini" and client.provider == "dryrun":
            st.error(f"⚠️ **Fallback Warning:** You selected 'gemini', but the application silently fell back to 'dryrun'.\n\n**Reason:** {getattr(client, 'fallback_reason', 'Unknown Module Error')}")
            st.warning("Please check your terminal logs. Make sure you installed dependencies (`pip install google-genai`) in the same environment running this Streamlit app.")
            
        candidates = generate_all_candidates(client, query, hits)

        
        # Check if generation returned API errors rather than ignoring
        has_error = False
        for act, ans in candidates.items():
            if str(ans).startswith("Gemini Error:"):
                st.error(f"Generation failed for action {act}: {ans}")
                has_error = True
                
        if has_error:
            st.stop()
            
        # 4. Score Candidates
        all_context = "\n".join(contexts)
        scores = {}
        for act, ans in candidates.items():
            scores[act] = score_answer(ans, all_context, sim_threshold=sim_threshold)
            
        # 5. Bandit Selection
        bandit = EpsilonGreedyBandit(ACTIONS, epsilon=epsilon)
        bandit.load()
        action = bandit.choose_action()
        
        # Update Bandit (Single step online learning for demo)
        # Note: In real app, we might not update on inference unless we have ground truth or use the heuristic as ground truth.
        # Here we assume the heuristic IS the reward signal.
        reward = scores[action]["reward"]
        bandit.update(action, reward)
        bandit.save()
        
        # 6. Display Results
        st.subheader("Bandit Decision")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chosen Action", action)
        col1.markdown(f"*Strategy: {action}*")
        col2.metric("Reward", round(reward, 3))
        col3.metric("Exploration", f"{bandit.epsilon:.2f}")

        st.subheader("Final Answer")
        st.info(candidates[action])
        
        # 7. Candidate Table
        st.subheader("Candidate Analysis (Transparency)")
        
        data = []
        for a in ACTIONS:
            s = scores[a]
            data.append({
                "Action": a,
                "Reward": round(s["reward"], 3),
                "Groundedness": round(s["groundedness"], 3),
                "Penalty": round(s["hallucination_penalty"], 3),
                "Supported Sents": f"{s['supported']}/{s['n_sentences']}",
                "Answer Preview": candidates[a][:50] + "..."
            })
            
        df = pd.DataFrame(data)
        st.dataframe(df.style.highlight_max(axis=0, subset=["Reward"]), use_container_width=True)
        
        # 8. Detailed breakdown for chosen
        with st.expander(f"Detailed Scoring for Chosen Action ({action})"):
            details = scores[action]["per_sentence"]
            for d in details:
                color = "green" if d["supported"] else "red"
                icon = "✅" if d["supported"] else "❌"
                st.markdown(f"{icon} **[{d['sim']:.3f}]** :{color}[{d['sentence']}]")

st.markdown("---")
st.subheader("Offline Training Results")
if st.button("Run 200-Episode Training"):
    import experiments.run_eval
    import experiments.plots
    
    with st.spinner("Training... check terminal for progress..."):
        experiments.run_eval.run_training_loop(num_episodes=200)
        experiments.plots.generate_plots()
    st.success("Training Complete!")
    
    # Show plots
    if (Path("outputs") / "reward_plot.png").exists():
        st.image(str(Path("outputs") / "reward_plot.png"), caption="Reward Learning Curve")
    if (Path("outputs") / "action_counts.png").exists():
        st.image(str(Path("outputs") / "action_counts.png"), caption="Action Selection Distribution")

elif (Path("outputs") / "reward_plot.png").exists():
     st.image(str(Path("outputs") / "reward_plot.png"), caption="Last Training: Reward Curve")
