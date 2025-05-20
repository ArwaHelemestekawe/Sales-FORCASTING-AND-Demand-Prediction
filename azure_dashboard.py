import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib


# === Custom Components and Styles ===
# This class defines reusable styling constants for the dashboard
# FONTS: Defines the font family to be used (Lato)
# COLORS: Defines color constants for text (BLACK/WHITE)
class components:
    FONTS = {
        "Lato": "Lato, sans-serif"  # Modern, clean font for the dashboard
    }
    COLORS = {
        "BLACK": "#000000",  # Text color for light mode
        "WHITE": "#FFFFFF"   # Text color for dark mode
    }

#----------------------------------------------------------------------------------------------------------- #

df = pd.read_csv(r"C:\Users\alghad\bank_marketing_dashboard\data.csv")
model = joblib.load(r"C:\Users\alghad\bank_marketing_dashboard\final_model.pkl")

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Bank Marketing Dashboard"

#----------------------------------------------------------------------------------------------------------- #

# === Tab Content Functions ===
# These functions generate the content for each tab, taking theme as a parameter
# to ensure proper dark/light mode styling

def get_tab_overview(theme):
    """
    Generates the Overview tab content with 4 graphs:
    - Age vs Deposit histogram
    - Job vs Deposit histogram
    - Education distribution pie chart
    - Monthly contacts bar chart
    All graphs use the provided theme for dark/light mode
    """
    return dbc.Container([
        html.H2("Overview", className="my-4"),
        dbc.Row([
            # First row: Two histograms side by side
            dbc.Col(dcc.Graph(figure=px.histogram(df, x="age", color="deposit", barmode="group", 
                title="Deposits by Age", template=theme)), md=6),
            dbc.Col(dcc.Graph(figure=px.histogram(df, x="job", color="deposit", 
                title="Deposits by Job", template=theme)), md=6)
        ]),
        dbc.Row([
            # Second row: Pie chart and bar chart
            html.Br(),
            dbc.Col(dcc.Graph(figure=px.pie(df, names="education", 
                title="Education Distribution", template=theme)), md=6),
            dbc.Col(dcc.Graph(figure=px.bar(df["month"].value_counts().sort_index(), 
                title="Contacts per Month", template=theme)), md=6)
        ])
    ])

def get_tab_predict():
    """
    Generates the Prediction tab content with:
    - Input fields for all prediction features
    - A predict button
    - Output display for prediction results
    """
    return dbc.Container([
        html.H2("Predict Deposit Likelihood", className="my-4"),
        # First row with input columns
        dbc.Row([
            # Left column: Basic customer information inputs
            dbc.Col([
                html.Br(),
                dbc.Label("Age"),
                dcc.RangeSlider(
                    id='input-age',
                    min=df["age"].min(),
                    max=df["age"].max(),
                    value=[35, 45],
                    marks={i: str(i) for i in range(18, 101, 10)},
                    step=1
                ),
                dbc.Label("Job"),
                dcc.Dropdown(id='input-job', options=[{'label': j, 'value': j} for j in df['job'].unique()], value='admin.'),
                html.Br(),
                dbc.Label("Marital Status"),
                dcc.Dropdown(id='input-marital', options=[{'label': m, 'value': m} for m in df['marital'].unique()], value='married'),
                html.Br(),
                dbc.Label("Education"),
                dcc.Dropdown(id='input-education', options=[{'label': e, 'value': e} for e in df['education'].unique()], value='secondary'),
                html.Br(),
            ], md=4),
            # Empty column for spacing
            dbc.Col([], md=4),
            # Right column: Campaign and contact information inputs
            dbc.Col([
                dbc.Label("Balance"),
                html.Br(),
                dcc.Input(id='input-balance', type='number', value=1),
                html.Br(),
                dbc.Label("Campaign Contacts"),
                html.Br(),
                dcc.Input(id='input-campaign', type='number', value=1),
                html.Br(),
                dbc.Label("Pdays"),
                html.Br(),
                dcc.Input(id='input-pdays', type='number', value=999),
                html.Br(),
                dbc.Label("Previous Contacts"),
                html.Br(),
                dcc.Input(id='input-previous', type='number', value=0),
                html.Br(),
                dbc.Label("Month"),
                html.Br(),
                dcc.Dropdown(id='input-month', options=[{'label': m, 'value': m} for m in df['month'].unique()], value='jan'),
                html.Br(),
            ], md=3, style={'padding': '30px'})
        ]),
        # Second row with centered predict button and output
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Button("Predict", id='predict-btn', color="primary", className="prediction"),
                    html.Div(id='prediction-output', style={'fontSize': 24, 'fontWeight': 'bold', 'marginTop': '10px'})
                ], style={'textAlign': 'center'})
            ], width=12)
        ])
    ])

def get_tab_features(theme):
    """
    Generates the Feature Insights tab content with:
    - Dropdown to select feature for analysis
    - Dynamic graph that updates based on selected feature
    Uses the provided theme for dark/light mode
    """
    return dbc.Container([
        html.H2("Feature Insights", className="my-4"),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(id='feature-dropdown', 
                    options=[{'label': col, 'value': col} for col in df.columns if col != 'deposit'], 
                    value='age'),
                dcc.Graph(id='feature-graph', figure=px.histogram(df, x='age', template=theme))
            ])
        ])
    ])

# === Main Layout Function ===
def get_layout(dark_mode):
    """
    Generates the main dashboard layout with:
    - Dark mode toggles
    - Header with title
    - Tabs for different sections
    All components adapt to dark/light mode
    """
    # Set theme and colors based on dark mode
    theme = 'plotly_dark' if dark_mode else 'plotly_white'
    text_color = components.COLORS["WHITE"] if dark_mode else components.COLORS["BLACK"]
    bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
    
    return dbc.Container([
        # Toggle switches in top-right corner
        dbc.Row([
            dbc.Col([
                dbc.Switch(
                    id="dark-mode-toggle",
                    label="Dark Mode",
                    value=dark_mode,
                    className="darkmode"
                )
            ], width="auto", className="position-absolute top-0 end-0 p-3")
        ]),

        # Dashboard header
        dbc.Row([
            dbc.Col(
                html.H1(
                    "Bank Marketing Dashboard",
                    style={
                        "fontFamily": components.FONTS["Lato"],
                        "fontStyle": 'Bold',
                        "textAlign": "center",
                        "color": text_color
                    },
                ),
                width="auto",
            ),
        ], justify="center", className="title"),

        # Main tabs
        dbc.Tabs(
            id="tabs",
            active_tab="tab-1",
            className="overview",
            children=[
                dbc.Tab(tab_id="tab-1", label="Overview", children=get_tab_overview(theme)),
                dbc.Tab(tab_id="tab-2", label="Prediction", children=get_tab_predict()),
                dbc.Tab(tab_id="tab-3", label="Feature Insights", children=get_tab_features(theme)),
            ]
        ),
    ], fluid=True, style={"backgroundColor": bg_color, "minHeight": "100vh"})

# === Callbacks ===
# These callbacks handle dynamic updates to the dashboard

@app.callback(
    [Output("main-container", "style"),
     Output("dashboard-title", "style"),
     Output("main-content", "children")],
    Input("dark-mode-toggle", "value")
)
def update_layout(dark_mode):
    """
    Updates the layout when dark mode is toggled
    """
    # Set theme and colors based on dark mode
    bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
    text_color = components.COLORS["WHITE"] if dark_mode else components.COLORS["BLACK"]
    
    return (
        {"backgroundColor": bg_color, "minHeight": "100vh", "color": text_color},
        {
            "fontFamily": components.FONTS["Lato"],
            "fontStyle": 'Bold',
            "textAlign": "center",
            "color": text_color
        },
        dbc.Tabs(
            id="tabs",
            active_tab="tab-1",
            className="overview",
            children=[
                dbc.Tab(tab_id="tab-1", label="Overview", children=get_tab_overview('plotly_dark' if dark_mode else 'plotly_white')),
                dbc.Tab(tab_id="tab-2", label="Prediction", children=get_tab_predict()),
                dbc.Tab(tab_id="tab-3", label="Feature Insights", children=get_tab_features('plotly_dark' if dark_mode else 'plotly_white')),
            ]
        )
    )

@app.callback(
    Output("feature-graph", "figure"),
    Input("feature-dropdown", "value"),
    Input("dark-mode-toggle", "value")
)
def update_feature_graph(feature, dark_mode):
    """
    Updates the feature graph when:
    1. A new feature is selected from the dropdown
    2. Dark mode is toggled
    Adapts the graph type based on the feature type (histogram for categorical, box plot for numerical)
    """
    theme = 'plotly_dark' if dark_mode else 'plotly_white'
    if df[feature].dtype == 'object':
        fig = px.histogram(df, x=feature, color='deposit', barmode='group', template=theme)
    else:
        fig = px.box(df, x='deposit', y=feature, color='deposit', template=theme)
    return fig

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-age", "value"),  # This will now receive [min_age, max_age]
    State("input-job", "value"),
    State("input-marital", "value"),
    State("input-education", "value"),
    State("input-balance", "value"),
    State("input-campaign", "value"),
    State("input-pdays", "value"),
    State("input-previous", "value"),
    State("input-month", "value"),
    prevent_initial_call=True
)
def predict(n, age_range, job, marital, education, balance, campaign, pdays, previous, month):
    """
    Handles the prediction process when the predict button is clicked:
    1. Collects all input values
    2. Creates a DataFrame with the input data
    3. Uses the loaded model to make a prediction
    4. Returns the prediction result
    
    Parameters:
    - age_range: List of [min_age, max_age] from the RangeSlider
    - other parameters: Individual input values from other components
    """
    # Use the average of the age range for prediction
    avg_age = sum(age_range) / 2
    
    input_df = pd.DataFrame([{
        'age': avg_age,  # Using average age from the range
        'job': job,
        'marital': marital,
        'education': education,
        'default': 'no',
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'month': month,
        'poutcome': 'unknown',
        'balance': balance,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous
    }])
    prediction = model.predict(input_df)[0]
    return f"Prediction: {'YES' if prediction == 'yes' else 'NO'} (Using average age: {avg_age:.1f} years)"

#----------------------------------------------------------------------------------------------------------- #

prevent_initial_call=True
allow_duplicate=True

# Set the initial layout
app.layout = html.Div(id='main-container', children=[
    # Toggle switches in top-right corner
    dbc.Row([
        dbc.Col([
            dbc.Switch(
                id="dark-mode-toggle",
                label="Dark Mode",
                value=False,
                className="mb-3"
            )
        ], width="auto", className="position-absolute top-0 end-0 p-3")
    ]),

    # Dashboard header
    dbc.Row([
        dbc.Col(
            html.H1(
                "Bank Marketing Dashboard",
                id="dashboard-title",
                style={
                    "fontFamily": components.FONTS["Lato"],
                    "fontStyle": 'Bold',
                    "textAlign": "center"
                },
            ),
            width="auto",
        ),
    ], justify="center", className="my-4"),

    # Main content container that will update with dark mode
    html.Div(id='main-content')
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
