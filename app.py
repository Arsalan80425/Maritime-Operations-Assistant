"""
Complete Enhanced Maritime Operations Assistant - Streamlit Dashboard
With workflow visualization, conversation memory, and all features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import graphviz

# Page config
st.set_page_config(
    page_title="Maritime Operations Assistant",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
try:
    from orchestrator import MaritimeOrchestrator
    from utils.data_loader import get_data_loader
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.info("Run: pip install -r requirements.txt")
    st.stop()

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .assistant-bg {
        background: linear-gradient(135deg, #f3f4f7 0%, #e8ecff 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .agent-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.4rem;
    }
    .tracking {background: #4CAF50; color: white;}
    .analytics {background: #2196F3; color: white;}
    .reporting {background: #FF9800; color: white;}
    .communication {background: #E91E63; color: white;}

    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin-bottom: 2rem;
    }
    .info-box h3 {
        color: #667eea;
        font-size: 1.6rem;
        margin-top: 0;
    }

    .workflow-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea10, #764ba210);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .stChatInput input {
        font-size: 1.3rem !important;
        padding: 1.5rem 1.5rem !important;
        border: 3px solid #667eea !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3) !important;
        min-height: 70px !important;
    }
    
    .stChatInput input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 12px 35px rgba(118,75,162,0.4) !important;
    }

    [data-testid="stChatMessage"][aria-label*="user"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        border-left: 4px solid #2196F3 !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%) !important;
        border-left: 4px solid #9C27B0 !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .high-confidence {background: #4CAF50; color: white;}
    .medium-confidence {background: #FF9800; color: white;}
    .low-confidence {background: #F44336; color: white;}
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Setup
if 'orchestrator' not in st.session_state:
    with st.spinner("🔄 Initializing Enhanced Maritime Operations Assistant..."):
        try:
            st.session_state.orchestrator = MaritimeOrchestrator()
            st.session_state.data_loader = get_data_loader()
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.session_state.workflow_history = []
            st.session_state.show_workflow_modal = False
            st.success("✅ System initialized with multi-agent collaboration!")
        except Exception as e:
            st.error(f"❌ Initialization Error: {str(e)}")
            st.info("Ensure Ollama is running and vector store is built")
            st.stop()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []
if 'show_workflow_modal' not in st.session_state:
    st.session_state.show_workflow_modal = False

# Sidebar
with st.sidebar:
    st.title("🚢 Maritime AI")
    st.markdown("**Enhanced Multi-Agent System by Mohammed Arsalan**")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "💬 AI Assistant", "📊 Analytics", "📋 Reports", "📦 Tracking"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System Status
    st.subheader("🤖 System Status")
    try:
        rag_status = st.session_state.orchestrator.get_rag_status()
        st.write("**RAG System:**", "✅ Active" if rag_status['available'] else "⚠️ Inactive")
        st.write("**Memory:**", "✅ Enabled" if rag_status.get('memory_enabled') else "❌ Disabled")
        st.write("**Conversations:**", rag_status.get('conversation_count', 0))
    except:
        st.write("**Status:** Loading...")
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    try:
        data_loader = st.session_state.data_loader
        shipments = data_loader.shipments
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", len(shipments))
            st.metric("In Transit", len(shipments[shipments['Status'] == 'In Transit']))
        with col2:
            st.metric("Delayed", len(shipments[shipments['Status'] == 'Delayed']))
            st.metric("Arrived", len(shipments[shipments['Status'] == 'Arrived']))
    except:
        pass
    
    st.markdown("---")
    
    if st.session_state.query_count > 0:
        st.subheader("🎯 Session Stats")
        st.info(f"Queries: {st.session_state.query_count}")
        
        # Agent usage stats
        if st.session_state.workflow_history:
            agents_used = {}
            for workflow in st.session_state.workflow_history:
                for agent in workflow.get('agent_chain', []):
                    agents_used[agent] = agents_used.get(agent, 0) + 1
            
            if agents_used:
                st.write("**Agent Usage:**")
                for agent, count in agents_used.items():
                    st.write(f"- {agent.title()}: {count}")
    
    st.markdown("---")
    st.caption("Powered by LangGraph & Ollama")

# ==============================
# 🏠 DASHBOARD PAGE
# ==============================
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🚢 Maritime Operations Dashboard</h1>', unsafe_allow_html=True)
    
    try:
        data_loader = st.session_state.data_loader
        
        # Key metrics with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(data_loader.shipments)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#667eea;">📦 Total Shipments</h3>
                <h1 style="margin:0.5rem 0;">{total:,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delayed = len(data_loader.shipments[data_loader.shipments['Status'] == 'Delayed'])
            delay_rate = (delayed / total * 100) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#FF5722;">⏰ Delayed</h3>
                <h1 style="margin:0.5rem 0;">{delayed}</h1>
                <p style="margin:0; color:#666;">{delay_rate:.1f}% of total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_delay = data_loader.shipments['Delay_Hours'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#FF9800;">⏱️ Avg Delay</h3>
                <h1 style="margin:0.5rem 0;">{avg_delay:.1f}h</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            active_ports = len(data_loader.port_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#4CAF50;">🗺️ Active Ports</h3>
                <h1 style="margin:0.5rem 0;">{active_ports}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Shipment Status Distribution")
            status_counts = data_loader.shipments['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values, 
                names=status_counts.index, 
                hole=0.4,
                color_discrete_sequence=['#4CAF50', '#2196F3', '#FF5722']
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📦 Cargo Type Distribution")
            cargo_counts = data_loader.shipments['Cargo_Type'].value_counts()
            fig = px.bar(
                x=cargo_counts.index, 
                y=cargo_counts.values,
                color=cargo_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="Cargo Type",
                yaxis_title="Count",
                showlegend=False,
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Port Performance
        st.subheader("🗺️ Top 10 Busiest Ports")
        top_ports = data_loader.port_data.nlargest(10, 'Vessels in Port')[
            ['Port Name', 'Country', 'Vessels in Port', 'Traffic Category']
        ]
        st.dataframe(top_ports, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

# ==============================
# 💬 AI ASSISTANT PAGE
# ==============================
elif page == "💬 AI Assistant":
    st.markdown('<h1 class="main-header">💬 Enhanced AI Assistant</h1>', unsafe_allow_html=True)

    # Info Box
    st.markdown("""
    <div class="info-box">
        <h3>🤖 Multi-Agent AI System with Advanced Features</h3>
        <p><strong>✨ New Features:</strong> Agent Collaboration • Workflow Visualization • Predictive Analytics • Conversation Memory • Auto-Alerts</p>
        <div>
            <span class="agent-badge tracking">📦 Tracking</span>
            <span class="agent-badge analytics">📊 Analytics + Predictions</span>
            <span class="agent-badge reporting">📋 Reports</span>
            <span class="agent-badge communication">✉️ Communication + Sending</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example Queries
    with st.expander("💡 Try These Advanced Queries", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔮 Predictive Analytics**
            - Predict delays at SHANGHAI for next week
            - Forecast congestion trends at NANTONG
            
            **🤝 Agent Collaboration**
            - Track CNT10000 (will trigger multi-agent flow)
            - Analyze high delays (triggers alerts)
            """)
        
        with col2:
            st.markdown("""
            **🧠 Context-Aware**
            - Follow-up questions use conversation memory
            - "What about that container?" (remembers context)
            
            **📧 Communication**
            - Draft delay alert for CNT10001
            - Generate weekly status update
            """)

    # Chat Section
    st.markdown("### 💭 Intelligent Conversation")

    # Display chat history with enhanced features
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
            
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    meta = message["metadata"]
                    
                    # Agent chain visualization
                    if meta.get("agent_chain"):
                        agents = " → ".join([a.title() for a in meta["agent_chain"]])
                        st.caption(f"🔄 **Agent Flow:** {agents}")
                    
                    # Confidence score
                    if meta.get("confidence", 0) > 0:
                        conf = meta["confidence"]
                        conf_class = "high-confidence" if conf > 0.7 else "medium-confidence" if conf > 0.4 else "low-confidence"
                        st.markdown(f'<span class="confidence-badge {conf_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                    
                    # RAG indicator
                    if meta.get("rag_used"):
                        st.caption("🔍 **RAG:** Enhanced with historical data")
                    
                    # Handoff indicator
                    if meta.get("handoff_reason"):
                        st.info(f"🔄 **Agent Handoff:** {meta['handoff_reason']}")
    else:
        st.info("👋 Start a conversation! Try asking about container tracking or port analytics.")

    # Chat Input
    query = st.chat_input("🚢 Ask me anything about maritime operations...")
    
    if query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.query_count += 1

        # Display user message
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(query)

        # Process with AI
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⚙️ Processing with enhanced multi-agent system..."):
                try:
                    response = st.session_state.orchestrator.process_query(query)
                    
                    assistant_text = response.get('response', 'No response generated.')
                    
                    # Display response
                    st.markdown(assistant_text)
                    
                    # Display agent chain
                    if response.get('agent_chain'):
                        agents = " → ".join([a.title() for a in response['agent_chain']])
                        st.caption(f"🔄 **Agent Flow:** {agents}")
                    
                    # Display confidence
                    if response.get('confidence', 0) > 0:
                        conf = response['confidence']
                        conf_class = "high-confidence" if conf > 0.7 else "medium-confidence" if conf > 0.4 else "low-confidence"
                        st.markdown(f'<span class="confidence-badge {conf_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                    
                    # Display RAG
                    if response.get('rag_used'):
                        st.caption("🔍 **RAG:** Enhanced with historical data")
                        if response.get('sources'):
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(response['sources'][:5], 1):
                                    st.write(f"{i}. {source}")
                    
                    # Display handoff
                    if response.get('handoff_reason'):
                        st.info(f"🔄 **Agent Collaboration:** {response['handoff_reason']}")
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_text,
                        "metadata": {
                            "intent": response.get('intent'),
                            "agent_chain": response.get('agent_chain', []),
                            "confidence": response.get('confidence', 0),
                            "rag_used": response.get('rag_used', False),
                            "handoff_reason": response.get('handoff_reason', '')
                        }
                    })
                    
                    # Save workflow
                    if response.get('workflow_steps'):
                        st.session_state.workflow_history.append(response)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {}
                    })

    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.session_state.orchestrator.clear_memory()
            st.rerun()
    
    with col2:
        if st.button("📊 View Workflow", use_container_width=True) and st.session_state.workflow_history:
            st.session_state.show_workflow_modal = True
    
    with col3:
        if st.session_state.chat_history:
            chat_text = "\n\n".join([
                f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" 
                for msg in st.session_state.chat_history
            ])
            st.download_button(
                label="💾 Export Chat",
                data=chat_text,
                file_name=f"maritime_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Workflow Visualization Modal
    if st.session_state.get('show_workflow_modal') and st.session_state.workflow_history:
        st.markdown("---")
        st.markdown("### 🔄 Agent Workflow Visualization")
        
        # Get latest workflow
        latest_workflow = st.session_state.workflow_history[-1]
        
        # Generate graph
        try:
            dot_graph = st.session_state.orchestrator.get_workflow_visualization()
            st.graphviz_chart(dot_graph)
            
            # Show workflow steps
            with st.expander("📋 Detailed Workflow Steps"):
                for i, step in enumerate(latest_workflow.get('workflow_steps', []), 1):
                    st.write(f"**Step {i}:** {step.get('step', 'unknown').title()}")
                    if step.get('intent'):
                        st.write(f"  - Intent: {step['intent']}")
                    if step.get('from_agent'):
                        st.write(f"  - Handoff: {step['from_agent']} → {step['to_agent']}")
                        st.write(f"  - Reason: {step['reason']}")
                    if step.get('rag_used'):
                        st.write(f"  - RAG: ✅ Used")
        except Exception as e:
            st.error(f"Error visualizing workflow: {e}")

    # Conversation Summary
    if st.session_state.chat_history:
        with st.expander("💬 Conversation Summary"):
            try:
                summary = st.session_state.orchestrator.get_conversation_summary()
                st.markdown(summary)
            except:
                st.write("No summary available")

# ==============================
# 📊 ANALYTICS PAGE
# ==============================
elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Advanced Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🔮 Predictive Analytics Powered by AI</h3>
        <p>Now includes delay predictions, anomaly detection, and trend forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        data_loader = st.session_state.data_loader
        
        # Predictive Analytics Section
        st.markdown("### 🔮 Predictive Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            port_for_prediction = st.selectbox(
                "Select Port for Prediction",
                sorted(data_loader.port_data['Port Name'].unique())
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔮 Predict Future Delays", type="primary", use_container_width=True):
                with st.spinner("Analyzing and predicting..."):
                    try:
                        analytics_agent = st.session_state.orchestrator.analytics_agent
                        result = analytics_agent.predict_future_delays(port_for_prediction, days_ahead=7)
                        
                        st.markdown(f"""
                        <div class="workflow-container">
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Port Performance Section
        st.markdown("### 🗺️ Port Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Top 10 Busiest Ports")
            top_ports = data_loader.port_data.nlargest(10, 'Vessels in Port')[
                ['Port Name', 'Country', 'Vessels in Port', 'Traffic Category']
            ]
            st.dataframe(top_ports, hide_index=True, use_container_width=True, height=400)
        
        with col2:
            st.markdown("#### ⏰ Average Delays by Port")
            port_delays = data_loader.daily_report.groupby('Port_Name')['Avg_Delay'].mean().nlargest(10)
            
            fig = px.bar(
                x=port_delays.values,
                y=port_delays.index,
                orientation='h',
                color=port_delays.values,
                color_continuous_scale='Reds',
                labels={'x': 'Average Delay (hours)', 'y': 'Port'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Cargo Performance
        st.markdown("### 📦 Cargo Performance Analysis")
        
        cargo_analysis = data_loader.shipments.groupby('Cargo_Type').agg({
            'Delay_Hours': ['mean', 'sum', 'count'],
            'Container_ID': 'count'
        }).round(2)
        cargo_analysis.columns = ['Avg Delay', 'Total Delay', 'Delayed Count', 'Total Shipments']
        cargo_analysis['Delay Rate %'] = (
            (cargo_analysis['Delayed Count'] / cargo_analysis['Total Shipments']) * 100
        ).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Average Delay by Cargo Type")
            fig = px.bar(
                cargo_analysis,
                x=cargo_analysis.index,
                y='Avg Delay',
                color='Avg Delay',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Cargo Performance Matrix")
            fig = px.scatter(
                cargo_analysis,
                x='Total Shipments',
                y='Avg Delay',
                size='Total Delay',
                text=cargo_analysis.index,
                color='Delay Rate %',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # AI-Powered Insights
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📍 Analyze Port Congestion Risks", use_container_width=True, type="primary"):
                with st.spinner("Analyzing with AI..."):
                    try:
                        analytics_agent = st.session_state.orchestrator.analytics_agent
                        result = analytics_agent.identify_congestion_risk()
                        st.markdown(f"""<div class="info-box">{result}</div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("📈 Cargo Performance Insights", use_container_width=True, type="primary"):
                with st.spinner("Analyzing with AI..."):
                    try:
                        analytics_agent = st.session_state.orchestrator.analytics_agent
                        result = analytics_agent.analyze_cargo_performance()
                        st.markdown(f"""<div class="info-box">{result}</div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# ==============================
# 📋 REPORTS PAGE
# ==============================
elif page == "📋 Reports":
    st.markdown('<h1 class="main-header">📋 Reports Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📄 Professional Report Generation</h3>
        <p>Generate comprehensive, AI-powered reports for executives and stakeholders.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Report Type Selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            report_type = st.selectbox(
                "📊 Select Report Type",
                [
                    "Executive Summary",
                    "Weekly Delay Report",
                    "Shipment Status Report",
                    "Port Performance Report"
                ]
            )
        
        # Port selection for Port Performance Report
        if report_type == "Port Performance Report":
            data_loader = st.session_state.data_loader
            port_options = sorted(data_loader.port_data['Port Name'].unique())
            selected_port = st.selectbox("🗺️ Select Port", port_options)
        else:
            selected_port = None
        
        st.markdown("---")
        
        # Generate Report Button
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("🤖 Generating report with AI..."):
                try:
                    report_agent = st.session_state.orchestrator.report_agent
                    
                    # Generate based on type
                    if report_type == "Executive Summary":
                        report = report_agent.generate_executive_summary()
                    elif report_type == "Weekly Delay Report":
                        report = report_agent.generate_weekly_delay_report()
                    elif report_type == "Shipment Status Report":
                        report = report_agent.generate_shipment_status_report()
                    else:  # Port Performance
                        report = report_agent.generate_port_performance_report(selected_port)
                    
                    # Display success and report
                    st.success("✅ Report generated successfully!")
                    
                    # Report container with styling
                    st.markdown("### 📄 Generated Report")
                    st.markdown(f"""
                    <div style="background: white; padding: 2rem; border-radius: 15px; 
                                border: 2px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        {report.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download TXT",
                            data=report,
                            file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            label="📥 Download MD",
                            data=report,
                            file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ==============================
# 📦 TRACKING PAGE
# ==============================
elif page == "📦 Tracking":
    st.markdown('<h1 class="main-header">📦 Container Tracking</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🔍 Real-Time Container Tracking</h3>
        <p>Track your containers, view shipment status, and get real-time updates on delays and ETAs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        data_loader = st.session_state.data_loader
        
        # Search Container Section
        st.markdown("### 🔎 Search Container")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            container_id = st.text_input(
                "Enter Container ID",
                placeholder="e.g., CNT10000",
                help="Enter the container ID to track (format: CNT#####)",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("")
            track_button = st.button("🔍 Track", type="primary", use_container_width=True)
        
        with col3:
            st.write("")
            ai_track = st.button("🤖 AI Track", type="secondary", use_container_width=True)
        
        # Handle tracking
        if (track_button or ai_track) and container_id:
            with st.spinner("🔍 Tracking container..."):
                try:
                    if ai_track:
                        # Use AI agent for detailed analysis
                        tracking_agent = st.session_state.orchestrator.tracking_agent
                        result = tracking_agent.track_container(container_id)
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>🤖 AI Analysis</h3>
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Get container details
                    container = data_loader.get_container_info(container_id)
                    
                    if container:
                        st.success(f"✅ Container {container_id} found!")
                        
                        # Status Card
                        status_color = {
                            'Arrived': '#4CAF50',
                            'In Transit': '#2196F3',
                            'Delayed': '#FF5722'
                        }
                        
                        color = status_color.get(container['Status'], '#9E9E9E')
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}20, {color}40); 
                                    padding: 2rem; border-radius: 15px; border-left: 6px solid {color};
                                    margin: 1rem 0;">
                            <h2 style="margin: 0; color: {color};">📦 {container_id}</h2>
                            <h3 style="margin: 0.5rem 0; color: #333;">{container['Status']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("🗺️ Port", container['Port_Name'])
                        
                        with col2:
                            st.metric("📅 ETA", str(container['ETA'])[:10])
                        
                        with col3:
                            delay_delta = f"+{container['Delay_Hours']} hrs" if container['Delay_Hours'] > 0 else "On time"
                            st.metric("⏰ Delay", f"{container['Delay_Hours']} hrs", delay_delta)
                        
                        with col4:
                            st.metric("📦 Cargo", container['Cargo_Type'])
                        
                        # Timeline visualization
                        st.markdown("---")
                        st.markdown("### 🛤️ Shipment Timeline")
                        
                        timeline_html = f"""
                        <div style="padding: 1rem;">
                            <div style="display: flex; align-items: center; margin: 1rem 0;">
                                <div style="width: 40px; height: 40px; border-radius: 50%; 
                                            background: #4CAF50; display: flex; align-items: center; 
                                            justify-content: center; color: white; font-weight: bold;">1</div>
                                <div style="flex: 1; height: 4px; background: #4CAF50; margin: 0 1rem;"></div>
                                <div style="padding: 0.5rem 1rem; background: #4CAF5020; border-radius: 10px;">
                                    <strong>Departure</strong><br/>
                                    <small>Origin Port</small>
                                </div>
                            </div>
                            
                            <div style="display: flex; align-items: center; margin: 1rem 0;">
                                <div style="width: 40px; height: 40px; border-radius: 50%; 
                                            background: {'#2196F3' if container['Status'] == 'In Transit' else '#9E9E9E'}; 
                                            display: flex; align-items: center; justify-content: center; 
                                            color: white; font-weight: bold;">2</div>
                                <div style="flex: 1; height: 4px; background: {'#2196F3' if container['Status'] in ['In Transit', 'Arrived'] else '#e0e0e0'}; 
                                            margin: 0 1rem;"></div>
                                <div style="padding: 0.5rem 1rem; background: {'#2196F320' if container['Status'] == 'In Transit' else '#e0e0e0'}; 
                                            border-radius: 10px;">
                                    <strong>In Transit</strong><br/>
                                    <small>{container['Port_Name']}</small>
                                </div>
                            </div>
                            
                            <div style="display: flex; align-items: center; margin: 1rem 0;">
                                <div style="width: 40px; height: 40px; border-radius: 50%; 
                                            background: {'#4CAF50' if container['Status'] == 'Arrived' else '#9E9E9E'}; 
                                            display: flex; align-items: center; justify-content: center; 
                                            color: white; font-weight: bold;">3</div>
                                <div style="flex: 1; height: 4px; background: {'#4CAF50' if container['Status'] == 'Arrived' else '#e0e0e0'}; 
                                            margin: 0 1rem;"></div>
                                <div style="padding: 0.5rem 1rem; background: {'#4CAF5020' if container['Status'] == 'Arrived' else '#e0e0e0'}; 
                                            border-radius: 10px;">
                                    <strong>Arrival</strong><br/>
                                    <small>ETA: {str(container['ETA'])[:10]}</small>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(timeline_html, unsafe_allow_html=True)
                        
                    else:
                        st.error(f"❌ Container {container_id} not found in the system.")
                        st.info("💡 Please check the container ID and try again.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # All Shipments Table
        st.markdown("### 📋 All Shipments")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.multiselect(
                "Status",
                options=data_loader.shipments['Status'].unique(),
                default=data_loader.shipments['Status'].unique()
            )
        
        with col2:
            cargo_filter = st.multiselect(
                "Cargo Type",
                options=data_loader.shipments['Cargo_Type'].unique(),
                default=data_loader.shipments['Cargo_Type'].unique()
            )
        
        with col3:
            port_filter = st.multiselect(
                "Port",
                options=sorted(data_loader.shipments['Port_Name'].unique()),
                default=[]
            )
        
        with col4:
            delay_filter = st.selectbox(
                "Delay Filter",
                ["All", "Delayed Only", "On Time Only"]
            )
        
        # Apply filters
        filtered_df = data_loader.shipments[
            (data_loader.shipments['Status'].isin(status_filter)) &
            (data_loader.shipments['Cargo_Type'].isin(cargo_filter))
        ]
        
        if port_filter:
            filtered_df = filtered_df[filtered_df['Port_Name'].isin(port_filter)]
        
        if delay_filter == "Delayed Only":
            filtered_df = filtered_df[filtered_df['Delay_Hours'] > 0]
        elif delay_filter == "On Time Only":
            filtered_df = filtered_df[filtered_df['Delay_Hours'] == 0]
        
        # Display count
        st.info(f"📊 Showing {len(filtered_df)} of {len(data_loader.shipments)} shipments")
        
        # Display dataframe
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"shipments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download as Excel (CSV)",
                data=csv,
                file_name=f"shipments_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Quick Stats
        st.markdown("---")
        st.markdown("### 📈 Filtered Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Filtered", len(filtered_df))
        
        with col2:
            delayed_count = len(filtered_df[filtered_df['Status'] == 'Delayed'])
            st.metric("Delayed", delayed_count)
        
        with col3:
            avg_delay_filtered = filtered_df['Delay_Hours'].mean()
            st.metric("Avg Delay", f"{avg_delay_filtered:.1f} hrs")
        
        with col4:
            unique_ports = len(filtered_df['Port_Name'].unique())
            st.metric("Unique Ports", unique_ports)
    
    except Exception as e:
        st.error(f"❌ Error loading tracking data: {str(e)}")

# ==============================
# Footer - Show on all pages except AI Assistant
# ==============================
if page != "💬 AI Assistant":
    st.markdown("""
    <div class="footer">
        <p><strong>Enhanced Maritime Operations Assistant</strong> | Built with LangGraph, Ollama & Streamlit by Mohammed Arsalan</p>
        <p>🚢 Multi-Agent Collaboration • RAG-Enhanced • Predictive Analytics</p>
    
    </div>
    """, unsafe_allow_html=True)