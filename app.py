import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stDownloadButton {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    h1 a {
        text-decoration: none;
        pointer-events: none;
    }
    /* Center specific columns */
    .stDataFrame td[data-testid="stDataFrameCell"][data-column="Company Id"],
    .stDataFrame td[data-testid="stDataFrameCell"][data-column="Probability (%)"] {
        text-align: center;
    }
    .stDataFrame th[data-testid="stDataFrameHeaderCell"][data-column="Company Id"],
    .stDataFrame th[data-testid="stDataFrameHeaderCell"][data-column="Probability (%)"] {
        text-align: center;
    }
    /* Fix padding and alignment for filters */
    [data-testid="stHorizontalBlock"] {
        padding-top: 2rem;
        align-items: flex-end;
    }
    div[data-testid="stCheckbox"] > label {
        display: flex;
        padding-top: 25px;
    }
    div[data-testid="stCheckbox"] > label > div[data-testid="stMarkdownContainer"] {
        line-height: 1.5;
    }
    #termination-risk-prediction-dashboard {
        padding-top: 0;
    }
    .stMainBlockContainer {
        padding-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Termination Risk Prediction Dashboard")

tab1, tab2 = st.tabs(["2025 Data", "Prior to 2025 Data"])

with tab1:
    post_cutoff_results = pd.read_csv('data/terminations.csv')

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Companies", format(len(post_cutoff_results), ','))
    col2.metric("High/Very High Risk Companies", len(post_cutoff_results[post_cutoff_results['Risk_Level'].isin(['High', 'Very High'])]))
    col3.metric("Percentage at Risk", f"{(len(post_cutoff_results[post_cutoff_results['Risk_Level'].isin(['High', 'Very High'])])) / len(post_cutoff_results) * 100:.2f}%")
    
    st.markdown("""
        <div style="background-color: #FFF3CD; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <strong>Note:</strong>  Companies with the Demo and InActive status were dropped from the training data.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Risk Level Distribution")
    
    st.markdown("""
        <style>
        [data-testid="column"] [data-testid="stDataFrame"] {
            padding-top: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))  
    risk_counts = post_cutoff_results['Risk_Level'].value_counts()
    risk_order = ['Very High', 'High', 'Medium', 'Low']
    risk_counts = risk_counts.reindex(risk_order)
    
    colors = ['#FF0000', '#FF6B00', '#FFC300', '#4CAF50']  # Red, Orange, Yellow, Green
    
    bars = ax.bar(range(len(risk_counts)), risk_counts, color=colors)
    
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Companies')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xticks(range(len(risk_counts)))
    ax.set_xticklabels(risk_order)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)


    st.markdown("## High-Risk Companies")
    if 'clear_global_search' not in st.session_state:
        st.session_state.clear_global_search = False
    
    if st.session_state.clear_global_search:
        st.session_state.global_search = ''
        st.session_state.clear_global_search = False

    # filters
    col1, col2, col3, col4 = st.columns([3, 1, 3, 1])
    with col1:
        risk_levels = st.multiselect(
            "Select Risk Levels",
            options=['Very High', 'High', 'Medium', 'Low'],
            default=['High', 'Very High'],
            key="global_risk_levels"
        )
    with col2:
        show_only_active = st.checkbox(
            "Active Only",
            value=True,
            help="Filter to show only currently Active companies that the model predicts might be terminated",
            key="show_active_checkbox"
        )
    with col3:
        search_query = st.text_input(
            "Search Company",
            placeholder="Type to search companies...",
            key="global_search",
            on_change=None
        )
    with col4:
        st.write("")  
        st.write("") 
        if st.button("Clear Search", key="clear_global_search_btn", on_click=lambda: setattr(st.session_state, 'clear_global_search', True)):
            pass
    
    # Filter data based on selected criteria
    filtered_data = post_cutoff_results[
        (post_cutoff_results['Risk_Level'].isin(risk_levels))  ]

    if show_only_active:
        filtered_data = filtered_data[filtered_data['Actual_Status'] == 'Active']
    
    if search_query:
        filtered_data = filtered_data[filtered_data['CompanyName'].str.contains(search_query, case=False, na=False)]
    
    display_columns = ['CompanyEntityId', 'CompanyName', 'Risk_Level', 'Actual_Status', 'Termination_Probability']
    column_names = {
        'CompanyEntityId': 'Company Id',
        'CompanyName': 'Company Name',
        'Risk_Level': 'Risk Level',
        'Actual_Status': 'Actual Status',
        'Termination_Probability': 'Probability (%)'
    }
    
    display_data = filtered_data[display_columns].copy()
    display_data = display_data.drop_duplicates(subset=['CompanyEntityId'], keep='first')
    display_data['Termination_Probability'] = (display_data['Termination_Probability'] * 100).round(2)
    
    st.dataframe(
        display_data.rename(columns=column_names).sort_values('Probability (%)', ascending=False),
        hide_index=True
    )


with tab2:
    results_df = pd.read_csv('data/demo_predictions.csv')

    top_50_high_risk = results_df.head(50)
    success_rate = top_50_high_risk['Prediction_Correct'].mean()
    correct_predictions = top_50_high_risk['Prediction_Correct'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Companies Analyzed", format(len(results_df), ','))
    col2.metric("High/Very High Risk Companies", format(len(results_df[results_df['Risk_Level'].isin(['High', 'Very High'])]), ','))
    col3.metric("Success Rate in Top 50", f"{success_rate:.2%}")
    col4.metric("Correct Predictions in Top 50", format(int(correct_predictions), ','))

    st.markdown("""
        <div style="background-color: #FFF3CD; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <strong>Note:</strong> Companies with the Demo and InActive status were dropped from the training data.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## Risk Level Distribution")
    
    fig, ax = plt.subplots(figsize=(12, 5))  
    risk_counts = results_df['Risk_Level'].value_counts()
    risk_order = ['Very High', 'High', 'Medium', 'Low']
    risk_counts = risk_counts.reindex(risk_order)
    
    colors = ['#FF0000', '#FF6B00', '#FFC300', '#4CAF50']  # Red, Orange, Yellow, Green
    
    bars = ax.bar(range(len(risk_counts)), risk_counts, color=colors)
    
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Companies')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xticks(range(len(risk_counts)))
    ax.set_xticklabels(risk_order)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)


    st.markdown("## High Risk Companies")
    
    if 'clear_tab2_search' not in st.session_state:
        st.session_state.clear_tab2_search = False
    if st.session_state.clear_tab2_search:
        st.session_state.tab2_search = ''
        st.session_state.clear_tab2_search = False

    # Add filters
    col1, col2, col3, col4 = st.columns([3, 1, 3, 1])
    with col1:
        risk_levels = st.multiselect(
            "Select Risk Levels",
            options=['Very High', 'High', 'Medium', 'Low'],
            default=['High', 'Very High'],
            key="tab2_risk_levels"
        )
    with col2:
        show_only_active = st.checkbox(
            "Active Only",
            value=True,
            help="Filter to show only currently Active companies that the model predicts might be terminated",
            key="tab2_show_active_checkbox"
        )
    with col3:
        search_query = st.text_input(
            "Search Company",
            placeholder="Type to search companies...",
            key="tab2_search",
            on_change=None
        )
    with col4:
        st.write("")  
        st.write("")  
        if st.button("Clear Search", key="clear_tab2_search_btn", on_click=lambda: setattr(st.session_state, 'clear_tab2_search', True)):
            pass
    
    # Filter data based on selected criteria
    filtered_data = results_df[
        (results_df['Risk_Level'].isin(risk_levels))  ]

    if show_only_active:
        filtered_data = filtered_data[filtered_data['Actual_Status'] == 'Active']

    if search_query:
        filtered_data = filtered_data[filtered_data['CompanyName'].str.contains(search_query, case=False, na=False)]

    display_columns = ['CompanyEntityId', 'CompanyName', 'Risk_Level', 'Actual_Status', 'Termination_Probability']
    column_names = {
        'CompanyEntityId': 'Company Id',
        'CompanyName': 'Company Name',
        'Risk_Level': 'Risk Level',
        'Actual_Status': 'Actual Status',
        'Termination_Probability': 'Probability (%)'
    }
    
    display_data = filtered_data[display_columns].copy()
    display_data = display_data.drop_duplicates(subset=['CompanyEntityId'], keep='first')
    display_data['Termination_Probability'] = (display_data['Termination_Probability'] * 100).round(2)
    
    st.dataframe(
        display_data.rename(columns=column_names).sort_values('Probability (%)', ascending=False),
        hide_index=True
    )