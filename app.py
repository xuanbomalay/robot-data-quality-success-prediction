"""Local recruiting demo. Every metric is read from completed experiment outputs."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from robot_audit.paths import Workspace
ws = Workspace(ROOT); ws.setup()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Robot Data Quality Lab', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container{padding-top:4.5rem;max-width:1400px}
h1{letter-spacing:-1.6px;font-weight:750!important}h2,h3{letter-spacing:-.5px}
[data-testid="stMetric"]{background:#f2f7f7;padding:18px 22px;border-radius:12px;border:1px solid #dbe7e7}
[data-testid="stMetricLabel"]{color:#34515b}[data-testid="stMetricValue"]{color:#103a46}
.eyebrow{color:#479aa7;letter-spacing:2px;font-size:12px;font-weight:700}
.intro{font-size:18px;color:inherit;opacity:.78;max-width:880px;line-height:1.6}
</style>''', unsafe_allow_html=True)


def read_json(name):
    path = ROOT/'reports'/f'{name}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@st.cache_data
def read_table(name):
    return pd.read_parquet(ROOT/'data/processed'/f'{name}.parquet')


audit = read_json('audit_summary'); prediction = read_json('prediction_metrics'); anomaly = read_json('anomaly_summary')
st.markdown('<div class="eyebrow">ROBOT DATA QUALITY LAB · INDEPENDENT PROJECT</div>', unsafe_allow_html=True)
st.title('From demonstrations to evidence.')
st.markdown('<p class="intro">Audit robot trajectories, identify review candidates, and test whether early motion predicts demonstration success.</p>', unsafe_allow_html=True)
if audit is None or prediction is None or anomaly is None:
    st.info('Real-data experiment outputs are not available in this checkout. Follow the README to fetch pinned data and run the pipeline. No sample scores are substituted.')
    st.stop()
selected = prediction['selected_model']; selected_metrics = prediction['models'][selected]['test']
video = read_json('video_summary'); policy = read_json('policy_results')
c1,c2,c3,c4 = st.columns(4)
c1.metric('Demonstrations audited', f'{audit["audited_episodes"]:,}')
c2.metric('Review candidates', f'{anomaly["review_flags"]:,}')
c3.metric('Held-out ROC-AUC', f'{selected_metrics["roc_auc_success"]:.3f}')
c4.metric('Video episodes checked', str(video['decoded_episodes']) if video else 'Pending')
st.caption('Real DROID teleoperation data · first 30 frames only · held-out collectors · anomaly ≠ corruption')
overview, quality, trajectories, success = st.tabs(['01  Overview', '02  Quality review', '03  Trajectories', '04  Success prediction'])

with overview:
    left,right=st.columns([1.3,1],gap='large')
    with left:
        st.subheader('A reproducible reliability workflow')
        st.markdown('**Pinned data → structural audit → anomaly review → grouped prediction → controlled policy test.**')
        st.markdown('The core contribution is an inspectable pipeline: source hashes, fixed-prefix features, leakage checks, and comparison experiments with honest boundaries.')
        counts=pd.DataFrame([{'split': k.title(), 'episodes':v} for k,v in audit['split_counts'].items()])
        fig=px.bar(counts,x='split',y='episodes',color='split',color_discrete_sequence=['#137c8b','#da6b46','#e0ad54'])
        fig.update_layout(showlegend=False,height=300,margin=dict(l=0,r=0,t=10,b=0),yaxis_title='Episodes',xaxis_title='')
        st.plotly_chart(fig,width='stretch')
    with right:
        st.subheader('What the evidence supports')
        st.markdown('- Quality checks on converted trajectories.\n- Success classification across held-out collectors.\n- Sampled video alignment checks.\n- A separate simulated cleaning experiment.')
        st.info('Outcome labels describe human demonstrations. They do not measure autonomous robot reliability. Converted timestamps cannot reveal original sensor-clock drift.')
    st.subheader('Controlled policy experiment')
    if policy and policy.get('status')=='complete':
        pframe=pd.DataFrame([{'condition':k,'success_rate':v['success_rate'],
                             'upper':v['ci95'][1]-v['success_rate'],
                             'lower':v['success_rate']-v['ci95'][0]} for k,v in policy['conditions'].items()])
        pfig=px.bar(pframe,x='condition',y='success_rate',error_y='upper',error_y_minus='lower',
                    color_discrete_sequence=['#137c8b'])
        seed_results=pd.DataFrame(policy['per_seed'])
        pfig.add_scatter(x=seed_results.condition,y=seed_results.success,mode='markers',
                         marker=dict(color='#da6b46',size=7),name='Training seeds')
        pfig.update_yaxes(tickformat='.0%',range=[0,1])
        pfig.update_layout(xaxis_title='',yaxis_title='Simulation success rate')
        st.plotly_chart(pfig,width='stretch')
        difference=policy['audit_minus_contaminated_percentage_points']
        interval=policy['audit_minus_contaminated_ci95_pp']
        st.markdown(f'**Audit minus contaminated:** {difference:+.1f} percentage points '
                    f'(95% CI {interval[0]:+.1f} to {interval[1]:+.1f}).')
        st.caption(f"Simulated Lift only · {policy['rollouts']} rollouts · injected action corruptions · paired initial states verified")
        st.caption('Bars: mean over three training seeds; dots: individual seeds; intervals: paired two-way cluster bootstrap over training seeds and shared initial states.')
        qualitative = read_json('policy_qualitative')
        if qualitative:
            st.markdown('**Prespecified qualitative rollouts**')
            st.caption('Training seed 17, shared evaluation start 20000; excluded from the 1,500 formal evaluations. These examples were fixed before viewing their outcomes.')
            for column, clip in zip(st.columns(2), qualitative['clips']):
                path = ws.path(f'reports/figures/{clip["file"]}')
                if path.exists():
                    with column:
                        st.markdown(clip['rollout']['condition'].replace('_', ' ').title())
                        st.video(str(path))
                        st.caption('Observed outcome: ' + ('success' if clip['rollout']['success'] else 'failure'))
            showcase = qualitative.get('showcase')
            if showcase:
                path = ws.path(f'reports/figures/{showcase["file"]}')
                if path.exists():
                    st.markdown('**Clean-policy success illustration**')
                    st.caption('Explicitly selected after formal evaluation: the first successful clean-policy seed-17 rollout above start 10000. This is an illustration, not a representative sample or an extra evaluation result.')
                    st.video(str(path))
    else:
        st.info('Policy comparison is still pending. No downstream improvement is claimed.')

with quality:
    st.subheader('Review candidates with traceable reasons')
    st.caption('Isolation Forest is fitted on training trajectories only. A review flag is a request for inspection, not a confirmed defect.')
    if not (ROOT/'data/processed/audit.parquet').exists():
        st.info('Download the pinned data and run the audit to inspect individual candidates. Summary reports remain available in this checkout.')
    else:
        data=read_table('audit')
        a,b=st.columns([1,3])
        with a:
            flagged=st.checkbox('Show review candidates only',True)
            outcome=st.selectbox('Demonstration outcome',['All','Success','Failure'])
        view=data[data.review_flag] if flagged else data
        if outcome!='All': view=view[view.success.eq(int(outcome=='Success'))]
        columns=['episode_id','actual_frames','success','isolation_score','reasons','split']
        with b:
            st.dataframe(view.sort_values('isolation_score',ascending=False)[columns].head(500),hide_index=True,width='stretch')
        fig=px.histogram(data,x='isolation_score',color=data.success.map({0:'Failure',1:'Success'}),barmode='overlay',nbins=50,
                         color_discrete_map={'Failure':'#da6b46','Success':'#137c8b'},labels={'color':'Outcome'})
        fig.update_layout(height=330,xaxis_title='Anomaly score',yaxis_title='Episodes')
        st.plotly_chart(fig,width='stretch')

with trajectories:
    st.subheader('Inspect a trajectory')
    local=(ROOT/'data/processed/selection.parquet').exists()
    if local:
        selection=read_table('selection')
        episode=st.selectbox('Episode identifier',selection.episode_id.tolist())
        meta=selection[selection.episode_id==episode].iloc[0]
        spec=read_json('source_versions')['sources'][meta.source]
        shard=ROOT/'data/raw'/spec['repo']/f'data/chunk-{int(meta["data/chunk_index"]):03d}/file_{int(meta["data/file_index"]):03d}.parquet'
        if shard.exists():
            frame=pd.read_parquet(shard,columns=['episode_index','timestamp','observation.joint_position','action.joint_velocity'])
            frame=frame[frame.episode_index.eq(meta.episode_index)]
            which=st.radio('Signal',['Joint position','Commanded joint velocity'],horizontal=True)
            signal='observation.joint_position' if which=='Joint position' else 'action.joint_velocity'
            values=np.stack(frame[signal])
            fig=go.Figure()
            for j in range(values.shape[1]): fig.add_scatter(x=frame.timestamp,y=values[:,j],name=f'Joint {j+1}',mode='lines')
            fig.add_vrect(x0=float(frame.timestamp.min()),x1=float(frame.timestamp.iloc[29]),fillcolor='#137c8b',opacity=.1,line_width=0,annotation_text='Prediction prefix')
            fig.update_layout(height=420,xaxis_title='Converted timestamp (s)',yaxis_title='Position (rad)' if which=='Joint position' else 'Commanded velocity (rad/s)')
            st.plotly_chart(fig,width='stretch')
        image=ROOT/'artifacts/video_frames'/f'{episode}.jpg'
        if image.exists():
            st.image(str(image),caption='Frame from the sampled left external camera; real dataset image',width=640)
        else:
            st.caption('This episode is outside the downloaded video sample. No image is fabricated.')
    else:
        st.info('Fetch the pinned control data to enable local trajectory inspection.')

with success:
    st.subheader('Predict eventual demonstration success')
    st.caption('All models use the same fixed prefix and collector split. The displayed model was selected on validation failure PR-AUC.')
    comparison=pd.DataFrame([{'model':k,**{q:v['test'][q] for q in ['roc_auc_success','pr_auc_failure','failure_recall','failure_precision']}} for k,v in prediction['models'].items()])
    st.dataframe(comparison,hide_index=True,width='stretch')
    ci=prediction['models'][selected]['test_ci95']['roc_auc_success']
    st.markdown(f'**Selected model:** {selected} · **ROC-AUC 95% CI:** {ci[0]:.3f}–{ci[1]:.3f}')
    if (ROOT/'data/processed/test_predictions.parquet').exists():
        records=read_table('test_predictions'); records=records[records.model.eq(selected)]
        case=st.selectbox('Held-out example',records.episode_id.tolist())
        row=records[records.episode_id.eq(case)].iloc[0]
        l,r=st.columns(2)
        l.metric('Predicted failure probability',f'{1-row.probability_success:.1%}')
        r.metric('Observed outcome','Success' if row.success else 'Failure')
        st.caption(f'Failure alert threshold: {row.failure_threshold:.2f}, selected using validation data only.')
    image=ROOT/'reports/figures/prediction.png'
    if image.exists(): st.image(str(image),caption='Held-out ROC and failure precision-recall curves')

st.divider()
st.caption('Evidence-first portfolio · No target-score tuning · Source and conversion attribution in README · No physical-robot deployment claim')
