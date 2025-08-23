import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# Set page config
st.set_page_config(
    layout="wide",
    page_title="M.Tech Talent Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Enhanced color palette with better contrast
primary_color = "#2563eb"       # Blue-600
secondary_color = "#10b981"     # Emerald-500
accent_color = "#ef4444"        # Red-500
highlight_color = "#a855f7"     # Violet-500
background_color = "#f9fafb"    # Gray-50
text_color = "#111827"          # Gray-900 (darker for better contrast)
card_background = "#ffffff"     # Pure white for cards
border_color = "#e5e7eb"        # Light gray for borders

# Apply custom styling
def apply_custom_style():
    st.markdown(f"""
    <style>
        html, body, .main, .block-container {{
            background-color: {background_color} !important;
            color: {text_color};
        }}

        .title-text {{
            text-align: center;
            color: {text_color} !important;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
            font-weight: 700;
        }}

        .subtitle-text {{
            text-align: center;
            color: #4b5563 !important;
            margin-bottom: 2rem;
            font-size: 1.2rem;
        }}

        .metric-card {{
            background-color: {card_background};
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid {border_color};
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}

        .metric-card h3 {{
            font-size: 1rem;
            color: {text_color} !important;
            margin-bottom: 0.5rem;
        }}

        .metric-card p {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 0;
            color: {text_color} !important;
        }}

        .plot-container {{
            background-color: {card_background};
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid {border_color};
            height: 100%;
        }}

        .plot-container h3 {{
            color: {text_color} !important;
            margin-top: 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {border_color};
        }}

        div[data-baseweb="tab-list"] button[data-baseweb="tab"] {{
            color: {text_color} !important;
            font-weight: 500;
            padding: 8px 16px;
        }}

        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {primary_color};
            color: white !important;
            border-radius: 8px;
        }}
        
        /* Ensure all text in Streamlit components is visible */
        .stMarkdown, .stHeader, .stSubheader, .stText {{
            color: {text_color} !important;
        }}
        
        /* Plot text colors */
        .css-1v0mbdj {{
            color: {text_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        ai_df = pd.read_csv('data/AIMtechStudentsData.csv', skiprows=1, encoding='latin1')
        ds_df = pd.read_csv('data/DSMtechStudensData.csv', encoding='latin1')

        ai_df.columns = ai_df.columns.str.strip()
        ds_df.columns = ds_df.columns.str.strip()

        column_mapping = {
            'First Name': 'Student Name',
            'Last Name': 'Student Name',
            'Full Time Job (Yes/No)': 'Full Time Experience',
            'Internship (Yes/No)': 'Internship Status'
        }

        for old, new in column_mapping.items():
            if old in ai_df.columns:
                ai_df.rename(columns={old: new}, inplace=True)
            if old in ds_df.columns:
                ds_df.rename(columns={old: new}, inplace=True)

        ai_df['Program'] = 'AI'
        ds_df['Program'] = 'DS'

        combined_df = pd.concat([ai_df, ds_df], ignore_index=True)

        yes_no_cols = [col for col in combined_df.columns if any(x in col for x in ['Yes/No', 'Status', 'Experience'])]
        for col in yes_no_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].str.upper().str.strip().fillna('NO')

        numeric_cols = ['X Percentage', 'XII Percentage', 'UG Aggregate %', 
                       'PG Sem 1 Aggregate %', 'PG Sem II Aggregate %', 'CGPA']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        if 'Gender' in combined_df.columns:
            combined_df['Gender'] = combined_df['Gender'].str.upper().str.strip()
            combined_df['Gender'] = combined_df['Gender'].replace({
                'MALE': 'MALE',
                'FEMALE': 'FEMALE',
                'Male': 'MALE',
                'Female': 'FEMALE'
            })

        if 'UG Stream' in combined_df.columns:
            combined_df['UG Stream'] = combined_df['UG Stream'].str.strip().str.title()

        if 'CGPA' in combined_df.columns:
            combined_df = combined_df[combined_df['CGPA'] >= 6.0]

        return combined_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_positive_histogram(data, column, title, color=primary_color, min_value=None):
    if min_value is not None:
        data = data[data[column] >= min_value]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data[column], kde=True, bins=12, color=color, ax=ax)
    ax.set_title(title, fontsize=12, pad=10, color=text_color)
    ax.set_xlabel(column, color=text_color)
    ax.set_ylabel("Number of Students", color=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    
    # Add mean line
    mean_val = data[column].mean()
    ax.axvline(mean_val, color=accent_color, linestyle='--', linewidth=1.5)
    ax.text(mean_val*1.01, ax.get_ylim()[1]*0.9, f'Mean: {mean_val:.2f}', 
            color=text_color, fontsize=10)
    
    plt.tight_layout()
    return fig

def create_donut_chart(data, column, title, colors=None):
    if colors is None:
        colors = [primary_color, secondary_color, accent_color, highlight_color]
    
    value_counts = data[column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
           colors=colors[:len(value_counts)], startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.set_title(title, fontsize=12, pad=10, color=text_color)
    plt.tight_layout()
    return fig

def create_positive_barplot(data, x_col, y_col, title, palette="viridis", top_n=5):
    top_data = data[y_col].value_counts().nlargest(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(top_data.index, top_data.values, color=sns.color_palette(palette, len(top_data)))
    ax.set_title(title, fontsize=12, pad=10, color=text_color)
    ax.set_xlabel("Count", color=text_color)
    ax.set_ylabel(y_col, color=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar_label(bars, fmt='%d', padding=3, color=text_color)
    plt.tight_layout()
    return fig

def create_positive_scatter(data, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[x_col], data[y_col], color=accent_color, alpha=0.7)
    ax.set_title(title, fontsize=12, pad=10, color=text_color)
    ax.set_xlabel(x_col, color=text_color)
    ax.set_ylabel(y_col, color=text_color)
    
    # Add trendline
    z = np.polyfit(data[x_col], data[y_col], 1)
    p = np.poly1d(z)
    ax.plot(data[x_col], p(data[x_col]), color=primary_color, linewidth=1.5, linestyle='--')
    
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    plt.tight_layout()
    return fig

def get_dashboard_title(program_filter):
    if len(program_filter) == 2:
        return "🎓 MPSTME M.Tech 2025-26"
    elif 'AI' in program_filter:
        return "🎓 MPSTME M.Tech Artificial Intelligence 2025-26"
    elif 'DS' in program_filter:
        return "🎓 MPSTME M.Tech Data Science 2025-26"
    return "🎓 MPSTME M.Tech 2025-26"

def main():
    apply_custom_style()
    
    combined_df = load_data()

    if combined_df.empty:
        st.error("Failed to load data. Please check the data files and try again.")
        return

    st.sidebar.header("🔍 Quick Filters")
    program_filter = st.sidebar.multiselect("Program", ['AI', 'DS'], default=['AI', 'DS'])
    filtered_df = combined_df.copy()
    if program_filter:
        filtered_df = filtered_df[filtered_df['Program'].isin(program_filter)]

    dashboard_title = get_dashboard_title(program_filter)
    st.markdown(f"<h1 class='title-text'>{dashboard_title}</h1>", unsafe_allow_html=True)

    st.markdown("### 📌 Performance Highlights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>👨‍🎓 Total Candidates</h3><p>{len(filtered_df)}</p></div>', unsafe_allow_html=True)
    with col2:
        avg_cgpa = filtered_df['CGPA'].mean()
        st.markdown(f'<div class="metric-card"><h3>📈 Avg. CGPA</h3><p>{f"{avg_cgpa:.2f}" if not pd.isna(avg_cgpa) else "N/A"}</p></div>', unsafe_allow_html=True)
    with col3:
        internship_col = next((col for col in filtered_df.columns if 'Internship' in col), None)
        internship_perc = (filtered_df[internship_col].str.upper() == 'YES').mean() * 100 if internship_col else 0
        st.markdown(f'<div class="metric-card"><h3>🎯 Internship %</h3><p>{f"{internship_perc:.1f}%"}</p></div>', unsafe_allow_html=True)
    with col4:
        ft_col = next((col for col in filtered_df.columns if 'Full Time' in col), None)
        ft_perc = (filtered_df[ft_col].str.upper() == 'YES').mean() * 100 if ft_col else 0
        st.markdown(f'<div class="metric-card"><h3>💼 FT Experience %</h3><p>{f"{ft_perc:.1f}%"}</p></div>', unsafe_allow_html=True)

    # SINGLE TAB: Academic Profile + Experience
    st.markdown("### <span style='color:black'>📚 Academic Profile</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown('<div class="plot-container"><h3>CGPA Distribution (≥6.0)</h3>', unsafe_allow_html=True)
            fig = create_positive_histogram(filtered_df, 'CGPA', "", min_value=6.0)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="plot-container"><h3>Top 5 UG Streams</h3>', unsafe_allow_html=True)
            if 'UG Stream' in filtered_df:
                fig = create_positive_barplot(filtered_df, None, 'UG Stream', "", top_n=5)
                st.pyplot(fig)
            else:
                st.warning("UG Stream data not available")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Removed the PG Semester Performance plot and kept only the Performance by Program plot
        with st.container():
            st.markdown('<div class="plot-container"><h3>Performance by Program</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Program', y='CGPA', data=filtered_df, ax=ax, 
                       palette=[primary_color, secondary_color])
            ax.set_xlabel("Program", color=text_color)
            ax.set_ylabel("CGPA", color=text_color)
            ax.tick_params(colors=text_color)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # EXPERIENCE INSIGHTS under academic
    st.markdown("### <span style='color:black'>💼 Professional Experience Insights</span>", unsafe_allow_html=True)

    cols = st.columns(2)
    if internship_col and internship_col in filtered_df:
        intern_df = filtered_df[filtered_df[internship_col].str.upper() == 'YES']
        with cols[0]:
            with st.container():
                st.markdown('<div class="plot-container"><h3>Internship Participation</h3>', unsafe_allow_html=True)
                fig = create_donut_chart(filtered_df, internship_col, "", colors=[secondary_color, '#e5e7eb'])
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="plot-container"><h3>Top Internship Roles</h3>', unsafe_allow_html=True)
                if 'Internship Role' in intern_df.columns:
                    fig = create_positive_barplot(intern_df, None, 'Internship Role', "", palette="Greens_r")
                    st.pyplot(fig)
                else:
                    st.warning("Internship role data not available")
                st.markdown('</div>', unsafe_allow_html=True)

    if ft_col and ft_col in filtered_df:
        ft_df = filtered_df[filtered_df[ft_col].str.upper() == 'YES']
        with cols[1]:
            with st.container():
                st.markdown('<div class="plot-container"><h3>Full-time Experience</h3>', unsafe_allow_html=True)
                fig = create_donut_chart(filtered_df, ft_col, "", colors=[primary_color, '#e5e7eb'])
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="plot-container"><h3>Top Full-Time Roles</h3>', unsafe_allow_html=True)
                if 'Full Time Role' in ft_df.columns:
                    fig = create_positive_barplot(ft_df, None, 'Full Time Role', "", palette="Blues_r")
                    st.pyplot(fig)
                else:
                    st.warning("Full-time role data not available")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation insights
    st.markdown("### <span style='color:black'>🔄 Experience vs Academics</span>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container():
            st.markdown('<div class="plot-container"><h3>CGPA vs Internship Status</h3>', unsafe_allow_html=True)
            if internship_col and 'CGPA' in filtered_df:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=internship_col, y='CGPA', data=filtered_df, 
                           ax=ax, palette=[secondary_color, '#e5e7eb'])
                ax.set_xlabel("Internship Status", color=text_color)
                ax.set_ylabel("CGPA", color=text_color)
                ax.tick_params(colors=text_color)
                ax.spines['bottom'].set_color(text_color)
                ax.spines['left'].set_color(text_color)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Required data not available")
            st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        with st.container():
            st.markdown('<div class="plot-container"><h3>Experience by Program</h3>', unsafe_allow_html=True)
            if internship_col and 'Program' in filtered_df:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Program', hue=internship_col, data=filtered_df,ax=ax, palette=[secondary_color, '#e5e7eb'])
                ax.set_xlabel("Program", color=text_color)
                ax.set_ylabel("Count", color=text_color)
                ax.tick_params(colors=text_color)
                ax.spines['bottom'].set_color(text_color)
                ax.spines['left'].set_color(text_color)
                ax.legend(title="Internship", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Required data not available")
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
